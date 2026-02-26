//! Token-level inference engine with inflight (continuous) batching
//!
//! The [`Engine`] manages a model and paged KV caches, providing
//! token-level generation (tokens in, tokens out). Multiple requests
//! share the GPU, with scheduling decisions made at every decode step.
//!
//! Decode steps use `forward_batch_decode` to process all active sequences
//! in a single forward pass (one weight read). Multi-GPU is supported:
//! one [`PagedKvCache`] is allocated per device, sharing the same logical
//! block indices via a single [`BlockAllocator`].
//!
//! The engine spawns a long-lived worker thread at construction. Callers
//! submit generation requests via [`Engine::submit`] and receive tokens
//! through a [`TokenSender`] implementation of their choice.

use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use infernum::{GenerateOptions, ModelConfig, Result, SamplingParams, Tensor};
use infernum_cuda::cuda::ops::{argmax_last_scalar, sample_top_p};
use infernum_cuda::cuda::{BatchedGraphInputs, CudaGraph, PagedKvCache};
use infernum_cuda::CudaTensor;
use infernum_cuda::Model;
use infernum_cuda::{BlockAllocator, BlockConfig};

use crate::scheduler::{BatchConfig, Scheduler, SequencePhase};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Model produced the end-of-sequence token.
    Stop,
    /// Reached the maximum number of tokens.
    Length,
    /// The receiver was dropped (client disconnect).
    Cancelled,
}

/// An event produced by the engine during generation.
///
/// Sent through a single channel so ordering is guaranteed:
/// zero or more `Token`s, then exactly one terminal event
/// (`Finished` or `Error`).
pub enum GenerationEvent {
    /// A newly generated token.
    Token(u32),
    /// An error occurred during generation.
    Error(infernum::Error),
    /// Generation completed with the given reason.
    Finished(FinishReason),
}

/// Trait for sending generation events from the engine to the caller.
///
/// Abstracted so callers can provide either a sync or async sender.
/// Return `false` to signal that the receiver has been dropped and
/// generation should stop.
pub trait TokenSender: Send {
    /// Send a generation event to the receiver.
    ///
    /// Returns `false` if the receiver has been dropped, signalling the
    /// engine to abort generation early.
    fn send(&self, event: GenerationEvent) -> bool;
}

impl TokenSender for mpsc::Sender<GenerationEvent> {
    fn send(&self, event: GenerationEvent) -> bool {
        mpsc::Sender::send(self, event).is_ok()
    }
}

impl TokenSender for Box<dyn TokenSender> {
    fn send(&self, event: GenerationEvent) -> bool {
        (**self).send(event)
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// A generation request submitted to the engine's worker thread.
struct GenerationRequest {
    prompt_ids: Vec<u32>,
    options: GenerateOptions,
    token_tx: Box<dyn TokenSender>,
}

/// Handle to the engine's worker thread.
///
/// The engine owns a long-lived thread that runs an iteration loop,
/// processing multiple requests concurrently via inflight batching
/// with paged KV caches.
pub struct Engine {
    request_tx: Option<mpsc::Sender<GenerationRequest>>,
    model_config: ModelConfig,
    worker: Option<JoinHandle<()>>,
}

impl Engine {
    /// Create a new engine wrapping the given model with default batch
    /// configuration.
    ///
    /// Uses [`BatchConfig::default()`] which provides sensible defaults
    /// for single-request usage (batch size 1, 512 blocks of size 16).
    ///
    /// # Errors
    /// Returns an error if paged KV cache allocation fails.
    pub fn new<M: Model + Send + 'static>(model: M) -> Result<Self> {
        Self::with_config(model, BatchConfig::default())
    }

    /// Create a new engine with a custom batch configuration.
    ///
    /// Spawns a worker thread that runs the batched iteration loop.
    ///
    /// # Errors
    /// Returns an error if paged KV cache allocation fails.
    pub fn with_config<M: Model + Send + 'static>(
        model: M,
        batch_config: BatchConfig,
    ) -> Result<Self> {
        infernum::fusion::init();
        let model_config = model.config();

        let block_config = BlockConfig {
            block_size: batch_config.block_size,
            num_blocks: batch_config.num_blocks,
        };
        let mut paged_kvs = Vec::new();
        for ctx in model.devices() {
            paged_kvs.push(PagedKvCache::new(
                ctx,
                model_config.num_layers,
                &block_config,
                model_config.num_kv_heads,
                model_config.head_dim,
                model_config.cache_dtype,
            )?);
        }
        let allocator = BlockAllocator::new(&block_config);

        let (request_tx, request_rx) = mpsc::channel::<GenerationRequest>();

        let worker = thread::spawn(move || {
            worker_loop(model, paged_kvs, allocator, batch_config, request_rx);
        });

        Ok(Self {
            request_tx: Some(request_tx),
            model_config,
            worker: Some(worker),
        })
    }

    /// Get the model configuration.
    #[must_use]
    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }

    /// Submit a generation request with a caller-provided token sender.
    ///
    /// Tokens are sent through `token_tx` as they are generated. The sender
    /// receives [`GenerationEvent::Token`] for each token, and generation
    /// stops when either the maximum token count is reached, an EOS token
    /// is produced, or the sender returns `false` (receiver dropped).
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
        // If the worker thread has panicked or the engine is shutting down,
        // the send will fail. The receiver will see the channel close.
        if let Some(tx) = &self.request_tx {
            let _ = tx.send(request);
        }
    }

    /// Generate tokens, blocking until complete.
    ///
    /// # Returns
    /// The full sequence (prompt + generated tokens).
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
    ///
    /// The provided `consumer` closure receives a [`mpsc::Receiver`] and is
    /// called on the current thread while tokens are being produced by the
    /// engine's worker thread.
    ///
    /// # Arguments
    /// * `input_ids` - Prompt token IDs
    /// * `options` - Generation options (sampling, max tokens, etc.)
    /// * `consumer` - Closure that processes tokens from the receiver
    ///
    /// # Returns
    /// The return value of the `consumer` closure.
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

impl Drop for Engine {
    fn drop(&mut self) {
        // Drop the sender first so the worker's recv() returns Err,
        // causing it to exit the loop.
        self.request_tx.take();
        // Join the worker thread to ensure all CUDA resources (model weights,
        // KV caches, graph state) are fully dropped before we return.
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Worker loop
// ---------------------------------------------------------------------------

/// Mutable state for CUDA graph capture/replay during batched decode.
struct GraphState {
    /// Pre-allocated GPU buffers for indirect kernels.
    graph_inputs: BatchedGraphInputs,
    /// CUDA graph manager (capture/replay).
    graph: CudaGraph,
    /// Decode step counter: 0 = first (warmup), 1 = capture, 2+ = replay.
    decode_step: usize,
    /// Logits tensor from the captured graph (same device addresses on replay).
    captured_logits: Option<CudaTensor>,
    /// Conservative upper bound for shared memory sizing in paged attention.
    graph_max_seq_len: usize,
}

/// The worker loop. Runs one iteration per scheduling step.
#[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
fn worker_loop<M: Model>(
    model: M,
    mut paged_kvs: Vec<PagedKvCache>,
    mut allocator: BlockAllocator,
    batch_config: BatchConfig,
    request_rx: mpsc::Receiver<GenerationRequest>,
) {
    let model_config = model.config();
    let mut scheduler = Scheduler::new(&batch_config);

    // Allocate graph state if CUDA graphs are enabled
    let mut graph_state = if batch_config.use_cuda_graphs {
        let device = model.devices()[0].device().clone();
        let dummy_block = allocator
            .allocate()
            .expect("need at least 1 block for CUDA graph padding");
        let max_blocks_per_seq = model_config
            .max_seq_len
            .div_ceil(batch_config.block_size)
            .min(batch_config.num_blocks);
        let mut graph_max_seq_len = max_blocks_per_seq * batch_config.block_size;

        // Cap graph_max_seq_len so the paged_decode_attention kernel's shared
        // memory fits within the device limit. The kernel allocates:
        //   (head_dim + max_active_len + threads) * sizeof(f32)
        // where threads ≤ 256.
        // 48 KB is the default shared memory limit per block on all modern
        // NVIDIA GPUs (higher requires cudaFuncSetAttribute opt-in).
        let max_shared: usize = 48 * 1024;
        let thread_count = 256;
        let max_floats = max_shared / std::mem::size_of::<f32>();
        let max_active = max_floats
            .saturating_sub(model_config.head_dim)
            .saturating_sub(thread_count);
        if graph_max_seq_len > max_active {
            graph_max_seq_len = max_active;
        }
        let graph_inputs = BatchedGraphInputs::new(
            &device,
            batch_config.max_batch_size,
            max_blocks_per_seq,
            dummy_block,
        )
        .expect("failed to allocate BatchedGraphInputs");
        let graph = CudaGraph::new(&device).expect("failed to allocate CudaGraph");
        Some(GraphState {
            graph_inputs,
            graph,
            decode_step: 0,
            captured_logits: None,
            graph_max_seq_len,
        })
    } else {
        None
    };

    loop {
        // 1. Drain all pending requests from the channel
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
                &mut paged_kvs,
                &mut allocator,
                &mut scheduler,
            );
        }

        // 5. Run decode steps
        if !output.decode.is_empty() {
            if let Some(gs) = &mut graph_state {
                process_decode_batch_graph(
                    &output.decode,
                    &model,
                    &mut paged_kvs,
                    &mut allocator,
                    &mut scheduler,
                    gs,
                );
            } else {
                process_decode_batch(
                    &output.decode,
                    &model,
                    &mut paged_kvs,
                    &mut allocator,
                    &mut scheduler,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Prefill
// ---------------------------------------------------------------------------

/// Process a prefill for one sequence.
fn process_prefill<M: Model>(
    idx: usize,
    token_ids: &[u32],
    model: &M,
    paged_kvs: &mut [PagedKvCache],
    allocator: &mut BlockAllocator,
    scheduler: &mut Scheduler,
) {
    // Run forward pass
    let (result, num_tokens) = {
        let seq = &scheduler.running()[idx];
        let start_pos = seq.block_table.seq_len();
        let r = model.forward_prefill_paged(token_ids, paged_kvs, &seq.block_table, start_pos);
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

            // Prefill complete → sample first token
            seq.phase = SequencePhase::Decode;

            let all_tokens = seq.prompt_ids.clone();
            let sampling_clone = seq.options.sampling.clone();
            let sampling = sampling_clone.as_ref();
            let mut rng_state = sampling.map(|s| s.seed);
            let recent = recent_token_window(&all_tokens, sampling);

            match select_token(&logits, sampling, &mut rng_state, recent) {
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
// Decode (eager)
// ---------------------------------------------------------------------------

/// Process all decode tasks in a single batched forward pass.
fn process_decode_batch<M: Model>(
    tasks: &[crate::scheduler::DecodeTask],
    model: &M,
    paged_kvs: &mut [PagedKvCache],
    allocator: &mut BlockAllocator,
    scheduler: &mut Scheduler,
) {
    // Gather tokens, block tables, and positions for all decode sequences
    let mut token_ids = Vec::with_capacity(tasks.len());
    let mut block_tables = Vec::with_capacity(tasks.len());
    let mut positions = Vec::with_capacity(tasks.len());
    let running_indices: Vec<usize> = tasks.iter().map(|t| t.running_idx).collect();

    for &idx in &running_indices {
        let seq = &scheduler.running()[idx];
        let tok = seq
            .generated_ids
            .last()
            .copied()
            .unwrap_or_else(|| *seq.prompt_ids.last().unwrap_or(&0));
        token_ids.push(tok);
        block_tables.push(seq.block_table.clone());
        positions.push(seq.block_table.seq_len());
    }

    let result = model.forward_batch_decode(&token_ids, paged_kvs, &block_tables, &positions);

    match result {
        Ok(logits) => {
            // Advance block tables for all decoded tokens
            for &idx in &running_indices {
                scheduler.running_mut()[idx].block_table.advance(1);
            }

            let vocab_size = logits.shape()[1];

            // Sample per-sequence from the batched logits
            for (batch_pos, &idx) in running_indices.iter().enumerate() {
                let seq_logits = logits.slice_view(batch_pos * vocab_size, &[1, vocab_size]);

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

                match select_token(&seq_logits, sampling, &mut rng_state, recent) {
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
            // Send the original error to the first sequence
            if let Some(&first_idx) = running_indices.first() {
                let seq = &mut scheduler.running_mut()[first_idx];
                seq.phase = SequencePhase::Finished(FinishReason::Stop);
                let _ = seq.token_tx.send(GenerationEvent::Error(e));
            }
            // Mark remaining sequences as failed
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
// Decode (CUDA graph)
// ---------------------------------------------------------------------------

/// Gather per-sequence data from the scheduler into `BatchedGraphInputs`.
///
/// Collects token IDs, positions, flattened block tables, and sequence
/// lengths from the running sequences and writes them to the pre-allocated
/// GPU buffers. Returns the scheduler indices of the real sequences.
fn gather_graph_inputs(
    tasks: &[crate::scheduler::DecodeTask],
    scheduler: &Scheduler,
    gs: &mut GraphState,
) -> Result<Vec<usize>> {
    let running_indices: Vec<usize> = tasks.iter().map(|t| t.running_idx).collect();
    let actual = running_indices.len();
    let max_blocks = gs.graph_inputs.max_blocks_per_seq();

    let mut token_ids = Vec::with_capacity(actual);
    let mut positions = Vec::with_capacity(actual);
    let mut block_tables_flat = Vec::with_capacity(actual * max_blocks);
    let mut seq_lens = Vec::with_capacity(actual);

    for &idx in &running_indices {
        let seq = &scheduler.running()[idx];
        let tok = seq
            .generated_ids
            .last()
            .copied()
            .unwrap_or_else(|| *seq.prompt_ids.last().unwrap_or(&0));
        token_ids.push(tok);

        let pos = seq.block_table.seq_len();
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        positions.push(pos as i32);

        // Flatten block table, padding unused slots with 0
        let blocks = seq.block_table.blocks();
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        block_tables_flat.extend(blocks.iter().take(max_blocks).map(|&b| b as i32));
        block_tables_flat.extend(std::iter::repeat_n(
            0i32,
            max_blocks.saturating_sub(blocks.len()),
        ));

        // seq_len for attention = pos + 1 (post-append length)
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        seq_lens.push((pos + 1) as i32);
    }

    gs.graph_inputs
        .update(&token_ids, &positions, &block_tables_flat, &seq_lens)?;

    Ok(running_indices)
}

/// Sample tokens from batched logits for the actual (non-padding) sequences.
fn sample_from_logits(
    logits: &CudaTensor,
    running_indices: &[usize],
    allocator: &mut BlockAllocator,
    scheduler: &mut Scheduler,
) {
    // Advance block tables for all decoded sequences
    for &idx in running_indices {
        scheduler.running_mut()[idx].block_table.advance(1);
    }

    let vocab_size = logits.shape()[1];

    for (batch_pos, &idx) in running_indices.iter().enumerate() {
        let seq_logits = logits.slice_view(batch_pos * vocab_size, &[1, vocab_size]);

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

        match select_token(&seq_logits, sampling, &mut rng_state, recent) {
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

/// Process decode tasks using CUDA graph capture/replay.
///
/// - Step 0 (warmup): run `forward_batch_decode_indirect` eagerly to load
///   all PTX modules.
/// - Step 1 (capture): `begin_capture` → forward → `end_capture` → `launch`.
/// - Steps 2+ (replay): `launch` the captured graph.
///
/// Between steps, `gather_graph_inputs` writes new token IDs, positions,
/// block tables, and `seq_lens` into the fixed-address GPU buffers.
#[allow(clippy::too_many_arguments)]
fn process_decode_batch_graph<M: Model>(
    tasks: &[crate::scheduler::DecodeTask],
    model: &M,
    paged_kvs: &mut [PagedKvCache],
    allocator: &mut BlockAllocator,
    scheduler: &mut Scheduler,
    gs: &mut GraphState,
) {
    let running_indices = match gather_graph_inputs(tasks, scheduler, gs) {
        Ok(ri) => ri,
        Err(e) => {
            mark_all_failed(tasks, scheduler, e);
            return;
        }
    };

    let max_seq_len = gs.graph_max_seq_len;
    let step = gs.decode_step;
    gs.decode_step += 1;

    let device = model.devices()[0].device().clone();

    let result = if step == 0 {
        // Warmup: run eagerly to load all PTX modules
        model.forward_batch_decode_indirect(&gs.graph_inputs, paged_kvs, max_seq_len)
    } else if step == 1 {
        // Capture the graph
        (|| -> Result<CudaTensor> {
            gs.graph.begin_capture()?;
            let logits =
                model.forward_batch_decode_indirect(&gs.graph_inputs, paged_kvs, max_seq_len);
            gs.graph.end_capture()?;
            let logits = logits?;

            gs.graph.launch()?;
            device.synchronize()?;

            gs.captured_logits = Some(logits.clone());
            Ok(logits)
        })()
    } else {
        // Replay the captured graph
        (|| -> Result<CudaTensor> {
            gs.graph.launch()?;
            device.synchronize()?;
            gs.captured_logits
                .clone()
                .ok_or_else(|| infernum::Error::CudaGraph("no captured logits for replay".into()))
        })()
    };

    match result {
        Ok(logits) => {
            sample_from_logits(&logits, &running_indices, allocator, scheduler);
        }
        Err(e) => {
            mark_all_failed(tasks, scheduler, e);
        }
    }
}

/// Mark all decode tasks as failed with an error message.
fn mark_all_failed(
    tasks: &[crate::scheduler::DecodeTask],
    scheduler: &mut Scheduler,
    error: infernum::Error,
) {
    let msg = format!("{error}");
    // Send the original error to the first task
    if let Some(first) = tasks.first() {
        let seq = &mut scheduler.running_mut()[first.running_idx];
        seq.phase = SequencePhase::Finished(FinishReason::Stop);
        let _ = seq.token_tx.send(GenerationEvent::Error(error));
    }
    // Remaining tasks get a formatted copy
    for task in tasks.iter().skip(1) {
        let seq = &mut scheduler.running_mut()[task.running_idx];
        seq.phase = SequencePhase::Finished(FinishReason::Stop);
        let _ = seq
            .token_tx
            .send(GenerationEvent::Error(infernum::Error::InvalidShape(
                format!("batched decode failed: {msg}"),
            )));
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Handle a newly sampled token: check EOS/max-tokens, send to channel,
/// allocate new block if needed.
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

/// Select the next token from logits, either via greedy argmax or sampling.
fn select_token(
    logits: &infernum_cuda::CudaTensor,
    sampling: Option<&SamplingParams>,
    rng_state: &mut Option<u64>,
    recent_tokens: &[u32],
) -> Result<u32> {
    if let (Some(params), Some(state)) = (sampling, rng_state) {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        sample_top_p(
            logits,
            params.temperature,
            params.top_p,
            *state,
            params.repetition_penalty,
            recent_tokens,
        )
    } else {
        argmax_last_scalar(logits)
    }
}

/// Extract the recent token window for repetition penalty.
fn recent_token_window<'a>(tokens: &'a [u32], sampling: Option<&SamplingParams>) -> &'a [u32] {
    let window = sampling.map_or(0, |s| s.repetition_penalty_window);
    let start = tokens.len().saturating_sub(window);
    &tokens[start..]
}
