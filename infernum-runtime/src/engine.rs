//! Backend-generic token-level inference engine with inflight batching.
//!
//! [`Engine`] is the backend-agnostic replacement for [`Engine`](super::Engine).
//! It is generic over `M: infernum::Model` and uses the [`Logits`](infernum::Logits)
//! and [`RuntimeStateInit`](infernum::RuntimeStateInit) traits for sampling and
//! backend state management. No CUDA-specific imports.

use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use infernum::backend::{Backend, TensorFactory};
use infernum::logits::Logits;
use infernum::runtime_state::RuntimeStateInit;
use infernum::{
    BlockAllocator, BlockConfig, GenerateOptions, Model, ModelConfig, Result, SamplingParams,
};

use crate::scheduler::{
    BatchConfig, DecodeTask, FinishReason, GenerationEvent, Scheduler, SequencePhase, TokenSender,
};

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// A generation request submitted to the engine's worker thread.
struct GenerationRequest {
    prompt_ids: Vec<u32>,
    options: GenerateOptions,
    token_tx: Box<dyn TokenSender>,
}

/// Backend-generic inference engine with inflight batching.
///
/// Wraps a model and manages paged KV caches, block allocation, and
/// scheduling. The engine spawns a long-lived worker thread at
/// construction. Callers submit generation requests via [`Engine::submit`]
/// and receive tokens through a [`TokenSender`].
pub struct Engine<M: Model> {
    request_tx: Option<mpsc::Sender<GenerationRequest>>,
    model_config: ModelConfig,
    worker: Option<JoinHandle<()>>,
    _phantom: std::marker::PhantomData<fn() -> M>,
}

impl<M: Model> Engine<M>
where
    M::B: TensorFactory,
{
    /// Create a new engine wrapping the given model with default batch config.
    ///
    /// # Errors
    /// Returns an error if KV cache allocation fails.
    pub fn new(model: M) -> Result<Self> {
        Self::with_config(model, BatchConfig::default())
    }

    /// Create a new engine with a custom batch configuration.
    ///
    /// # Errors
    /// Returns an error if KV cache allocation fails.
    pub fn with_config(model: M, batch_config: BatchConfig) -> Result<Self> {
        infernum::fusion::init();
        let model_config = model.config();

        let block_config = BlockConfig {
            block_size: batch_config.block_size,
            num_blocks: batch_config.num_blocks,
        };

        let kv_cache = model.allocate_kv_cache(&block_config)?;

        let runtime_batch_config = infernum::runtime_state::BatchConfig {
            max_batch_size: batch_config.max_batch_size,
            max_prefill_tokens: batch_config.max_prefill_tokens,
            block_size: batch_config.block_size,
            num_blocks: batch_config.num_blocks,
        };
        let runtime_state =
            <M::B as Backend>::RuntimeState::new(&runtime_batch_config, &block_config)?;

        let allocator = BlockAllocator::new(&block_config);

        let (request_tx, request_rx) = mpsc::channel::<GenerationRequest>();

        let worker = thread::spawn(move || {
            worker_loop(
                model,
                kv_cache,
                runtime_state,
                allocator,
                batch_config,
                request_rx,
            );
        });

        Ok(Self {
            request_tx: Some(request_tx),
            model_config,
            worker: Some(worker),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get the model configuration.
    #[must_use]
    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }

    /// Submit a generation request with a caller-provided token sender.
    pub fn submit(
        &self,
        input_ids: Vec<u32>,
        options: GenerateOptions,
        token_tx: impl TokenSender + 'static,
    ) {
        let request = GenerationRequest {
            prompt_ids: input_ids,
            options,
            token_tx: Box::new(token_tx),
        };
        if let Some(tx) = &self.request_tx {
            let _ = tx.send(request);
        }
    }

    /// Generate tokens, blocking until complete.
    ///
    /// # Errors
    /// Returns an error if a forward pass fails.
    pub fn generate(&self, input_ids: &[u32], options: &GenerateOptions) -> Result<Vec<u32>> {
        let (tx, rx) = mpsc::channel();
        self.submit(input_ids.to_vec(), options.clone(), tx);

        let mut tokens = input_ids.to_vec();
        for event in rx {
            match event {
                GenerationEvent::Token(id) => tokens.push(id),
                GenerationEvent::Error(e) => return Err(e),
                GenerationEvent::Finished(_) => break,
            }
        }
        Ok(tokens)
    }

    /// Generate tokens with streaming via a channel.
    pub fn generate_stream<F, R>(
        &self,
        input_ids: &[u32],
        options: &GenerateOptions,
        consumer: F,
    ) -> R
    where
        F: FnOnce(mpsc::Receiver<GenerationEvent>) -> R,
    {
        let (tx, rx) = mpsc::channel();
        self.submit(input_ids.to_vec(), options.clone(), tx);
        consumer(rx)
    }
}

impl<M: Model> Drop for Engine<M> {
    fn drop(&mut self) {
        self.request_tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Worker loop
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
fn worker_loop<M>(
    model: M,
    mut kv_cache: M::KvCache,
    mut runtime_state: <M::B as Backend>::RuntimeState,
    mut allocator: BlockAllocator,
    batch_config: BatchConfig,
    request_rx: mpsc::Receiver<GenerationRequest>,
) where
    M: Model,
    M::B: TensorFactory,
{
    let model_config = model.config();
    let mut scheduler = Scheduler::new(&batch_config);

    loop {
        // 1. Drain all pending requests
        while let Ok(req) = request_rx.try_recv() {
            let _ = scheduler.add_request(req.prompt_ids, req.options, req.token_tx);
        }

        // 2. If idle, block-wait for the next request
        if scheduler.is_idle() {
            match request_rx.recv() {
                Ok(req) => {
                    let _ = scheduler.add_request(req.prompt_ids, req.options, req.token_tx);
                }
                Err(_) => break,
            }
        }

        // 3. Scheduler decides what to process this iteration
        let output = scheduler.step(&mut allocator);

        // 4. Run prefills (one at a time, sequential)
        for task in &output.prefill {
            let idx = task.running_idx;
            let token_ids = task.token_ids.clone();
            process_prefill(
                idx,
                &token_ids,
                &model,
                &mut kv_cache,
                &mut runtime_state,
                &mut allocator,
                &mut scheduler,
                &model_config,
            );
        }

        // 5. Run batched decode
        if !output.decode.is_empty() {
            process_decode_batch(
                &output.decode,
                &model,
                &mut kv_cache,
                &mut runtime_state,
                &mut allocator,
                &mut scheduler,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Prefill
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn process_prefill<M: Model>(
    idx: usize,
    token_ids: &[u32],
    model: &M,
    kv_cache: &mut M::KvCache,
    runtime_state: &mut <M::B as Backend>::RuntimeState,
    allocator: &mut BlockAllocator,
    scheduler: &mut Scheduler,
    _model_config: &ModelConfig,
) {
    let (result, num_tokens) = {
        let seq = &scheduler.running()[idx];
        let start_pos = seq.block_table.seq_len();
        let r = model.forward_prefill(
            token_ids,
            kv_cache,
            runtime_state,
            &seq.block_table,
            start_pos,
        );
        (r, token_ids.len())
    };

    match result {
        Ok(logits) => {
            let seq = &mut scheduler.running_mut()[idx];
            seq.block_table.advance(num_tokens);
            seq.prefill_progress += num_tokens;

            if seq.prefill_progress < seq.prompt_ids.len() {
                return;
            }

            seq.phase = SequencePhase::Decode;

            let all_tokens = seq.prompt_ids.clone();
            let sampling_clone = seq.options.sampling.clone();
            let sampling = sampling_clone.as_ref();
            let mut rng_state = sampling.map(|s| s.seed);
            let recent = recent_token_window(&all_tokens, sampling);

            match select_token(&logits, 0, sampling, &mut rng_state, recent) {
                Ok(token_id) => {
                    handle_new_token(idx, token_id, allocator, scheduler);
                }
                Err(e) => {
                    let seq = &mut scheduler.running_mut()[idx];
                    seq.phase = SequencePhase::Finished(FinishReason::Stop);
                    let _ = seq.token_tx.send(GenerationEvent::Error(e));
                }
            }
        }
        Err(e) => {
            let seq = &mut scheduler.running_mut()[idx];
            seq.phase = SequencePhase::Finished(FinishReason::Stop);
            let _ = seq.token_tx.send(GenerationEvent::Error(e));
        }
    }
}

// ---------------------------------------------------------------------------
// Decode (backend-agnostic, no graph logic)
// ---------------------------------------------------------------------------

fn process_decode_batch<M>(
    tasks: &[DecodeTask],
    model: &M,
    kv_cache: &mut M::KvCache,
    runtime_state: &mut <M::B as Backend>::RuntimeState,
    allocator: &mut BlockAllocator,
    scheduler: &mut Scheduler,
) where
    M: Model,
    M::B: TensorFactory,
{
    let batch_size = tasks.len();
    let mut token_ids_host = Vec::with_capacity(batch_size);
    let mut seq_lens_host = Vec::with_capacity(batch_size);
    let mut positions_host = Vec::with_capacity(batch_size);
    let mut block_tables_raw: Vec<Vec<usize>> = Vec::with_capacity(batch_size);
    let running_indices: Vec<usize> = tasks.iter().map(|t| t.running_idx).collect();

    for &idx in &running_indices {
        let seq = &scheduler.running()[idx];
        let tok = seq
            .generated_ids
            .last()
            .copied()
            .unwrap_or_else(|| *seq.prompt_ids.last().unwrap_or(&0));
        token_ids_host.push(tok);
        let pos = seq.block_table.seq_len();
        #[allow(clippy::cast_possible_wrap)]
        {
            positions_host.push(pos as i32);
            seq_lens_host.push((pos + 1) as i32);
        }
        block_tables_raw.push(seq.block_table.blocks().to_vec());
    }

    // Flatten block tables into (batch_size * max_blocks_per_seq) with padding
    let max_blocks_per_seq = block_tables_raw.iter().map(Vec::len).max().unwrap_or(0);
    #[allow(clippy::cast_possible_wrap)]
    let block_tables_flat: Vec<i32> = {
        let mut flat = vec![0i32; batch_size * max_blocks_per_seq];
        for (i, bt) in block_tables_raw.iter().enumerate() {
            for (j, &block_id) in bt.iter().enumerate() {
                flat[i * max_blocks_per_seq + j] = block_id as i32;
            }
        }
        flat
    };

    #[allow(clippy::cast_sign_loss)]
    let max_seq_len = seq_lens_host.iter().copied().max().unwrap_or(0) as usize;
    let device = model.device();

    let token_ids_tensor =
        <M::B as TensorFactory>::from_u32_slice(device, &[batch_size], &token_ids_host)
            .expect("failed to upload token_ids");
    let block_tables_tensor = <M::B as TensorFactory>::from_i32_slice(
        device,
        &[batch_size * max_blocks_per_seq],
        &block_tables_flat,
    )
    .expect("failed to upload block_tables");
    let seq_lens_tensor =
        <M::B as TensorFactory>::from_i32_slice(device, &[batch_size], &seq_lens_host)
            .expect("failed to upload seq_lens");
    let positions_tensor =
        <M::B as TensorFactory>::from_i32_slice(device, &[batch_size], &positions_host)
            .expect("failed to upload positions");

    let result = model.forward_batch_decode(
        &token_ids_tensor,
        kv_cache,
        runtime_state,
        &block_tables_tensor,
        &seq_lens_tensor,
        &positions_tensor,
        batch_size,
        max_blocks_per_seq,
        max_seq_len,
    );

    match result {
        Ok(logits) => {
            for &idx in &running_indices {
                scheduler.running_mut()[idx].block_table.advance(1);
            }

            for (batch_pos, &idx) in running_indices.iter().enumerate() {
                let (all_tokens, sampling_clone, gen_len) = {
                    let seq = &scheduler.running()[idx];
                    let toks: Vec<u32> = seq
                        .prompt_ids
                        .iter()
                        .chain(seq.generated_ids.iter())
                        .copied()
                        .collect();
                    let s = seq.options.sampling.clone();
                    let gl = seq.generated_ids.len();
                    (toks, s, gl)
                };
                let sampling = sampling_clone.as_ref();
                let mut rng_state = sampling.map(|s| s.seed ^ (gen_len as u64));
                let recent = recent_token_window(&all_tokens, sampling);

                match select_token(&logits, batch_pos, sampling, &mut rng_state, recent) {
                    Ok(token_id) => {
                        handle_new_token(idx, token_id, allocator, scheduler);
                    }
                    Err(e) => {
                        let seq = &mut scheduler.running_mut()[idx];
                        seq.phase = SequencePhase::Finished(FinishReason::Stop);
                        let _ = seq.token_tx.send(GenerationEvent::Error(e));
                    }
                }
            }
        }
        Err(e) => {
            let msg = format!("{e}");
            if let Some(&first_idx) = running_indices.first() {
                let seq = &mut scheduler.running_mut()[first_idx];
                seq.phase = SequencePhase::Finished(FinishReason::Stop);
                let _ = seq.token_tx.send(GenerationEvent::Error(e));
            }
            for &idx in running_indices.iter().skip(1) {
                let seq = &mut scheduler.running_mut()[idx];
                seq.phase = SequencePhase::Finished(FinishReason::Stop);
                let _ = seq
                    .token_tx
                    .send(GenerationEvent::Error(infernum::Error::InvalidShape(
                        format!("batched decode failed: {msg}"),
                    )));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn handle_new_token(
    idx: usize,
    token_id: u32,
    allocator: &mut BlockAllocator,
    scheduler: &mut Scheduler,
) {
    let seq = &mut scheduler.running_mut()[idx];

    if seq.options.eos_token_id == Some(token_id) {
        seq.phase = SequencePhase::Finished(FinishReason::Stop);
        let _ = seq
            .token_tx
            .send(GenerationEvent::Finished(FinishReason::Stop));
        return;
    }

    seq.generated_ids.push(token_id);

    if !seq.token_tx.send(GenerationEvent::Token(token_id)) {
        seq.phase = SequencePhase::Finished(FinishReason::Cancelled);
        return;
    }

    if seq.generated_ids.len() >= seq.options.max_new_tokens {
        seq.phase = SequencePhase::Finished(FinishReason::Length);
        let _ = seq
            .token_tx
            .send(GenerationEvent::Finished(FinishReason::Length));
        return;
    }

    if seq.block_table.needs_new_block() {
        if let Some(block) = allocator.allocate() {
            seq.block_table.append_block(block);
        }
    }
}

/// Select the next token from logits via the `Logits` trait.
fn select_token<L: Logits>(
    logits: &L,
    batch_index: usize,
    sampling: Option<&SamplingParams>,
    rng_state: &mut Option<u64>,
    recent_tokens: &[u32],
) -> Result<u32> {
    if let (Some(params), Some(state)) = (sampling, rng_state) {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        logits.sample_top_p(
            batch_index,
            params.temperature,
            params.top_p,
            *state,
            params.repetition_penalty,
            recent_tokens,
        )
    } else {
        logits.argmax(batch_index)
    }
}

fn recent_token_window<'a>(tokens: &'a [u32], sampling: Option<&SamplingParams>) -> &'a [u32] {
    let window = sampling.map_or(0, |s| s.repetition_penalty_window);
    let start = tokens.len().saturating_sub(window);
    &tokens[start..]
}
