//! Batched inference engine with inflight (continuous) batching
//!
//! The [`BatchedEngine`] replaces the sequential [`Engine`](super::Engine)
//! worker loop with an iteration-level scheduler. Multiple requests share
//! the GPU, with scheduling decisions made at every decode step.
//!
//! # Current Implementation
//!
//! Each sequence's decode step is processed individually via
//! `forward_prefill_paged` (single-token prefill). True batched decode
//! (`forward_batch_decode`) will be wired in a follow-up once the
//! scheduler and lifecycle are proven correct.

use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use infernum::cuda::block_allocator::{BlockAllocator, BlockConfig};
use infernum::cuda::ops::{argmax_last_scalar, sample_top_p};
use infernum::cuda::PagedKvCache;
use infernum::{GenerateOptions, Model, ModelConfig, Result, SamplingParams};

use crate::engine::{FinishReason, GenerationEvent, TokenSender};
use crate::scheduler::{BatchConfig, Scheduler, SequencePhase};

/// A generation request submitted to the batched engine.
struct BatchedRequest {
    prompt_ids: Vec<u32>,
    options: GenerateOptions,
    token_tx: Box<dyn TokenSender>,
}

/// Handle to the batched engine's worker thread.
///
/// The engine owns a long-lived thread that runs an iteration loop,
/// processing multiple requests concurrently via inflight batching.
pub struct BatchedEngine {
    request_tx: mpsc::Sender<BatchedRequest>,
    model_config: ModelConfig,
    _worker: JoinHandle<()>,
}

impl BatchedEngine {
    /// Create a new batched engine wrapping the given model.
    ///
    /// Spawns a worker thread that runs the batched iteration loop.
    ///
    /// # Errors
    /// Returns an error if paged KV cache allocation fails.
    pub fn new<M: Model + Send + 'static>(model: M, batch_config: BatchConfig) -> Result<Self> {
        infernum::fusion::init();
        let model_config = model.config();

        let block_config = BlockConfig {
            block_size: batch_config.block_size,
            num_blocks: batch_config.num_blocks,
        };
        let ctx = model.devices()[0].clone();
        let paged_kv = PagedKvCache::new(
            &ctx,
            model_config.num_layers,
            &block_config,
            model_config.num_kv_heads,
            model_config.head_dim,
        )?;
        let allocator = BlockAllocator::new(&block_config);

        let (request_tx, request_rx) = mpsc::channel::<BatchedRequest>();

        let worker = thread::spawn(move || {
            batched_worker_loop(model, paged_kv, allocator, batch_config, request_rx);
        });

        Ok(Self {
            request_tx,
            model_config,
            _worker: worker,
        })
    }

    /// Get the model configuration.
    #[must_use]
    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }

    /// Submit a generation request.
    ///
    /// Tokens are sent through `token_tx` as they are generated.
    pub fn submit(
        &self,
        input_ids: Vec<u32>,
        options: GenerateOptions,
        token_tx: impl TokenSender + 'static,
    ) {
        let request = BatchedRequest {
            prompt_ids: input_ids,
            options,
            token_tx: Box::new(token_tx),
        };
        let _ = self.request_tx.send(request);
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

/// The batched worker loop. Runs one iteration per scheduling step.
#[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
fn batched_worker_loop<M: Model>(
    model: M,
    mut paged_kv: PagedKvCache<M::CacheDtype>,
    mut allocator: BlockAllocator,
    batch_config: BatchConfig,
    request_rx: mpsc::Receiver<BatchedRequest>,
) {
    let mut scheduler = Scheduler::new(&batch_config);

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
                &mut paged_kv,
                &mut allocator,
                &mut scheduler,
            );
        }

        // 5. Run decode steps (one at a time for now; batched decode later)
        for task in &output.decode {
            process_decode(
                task.running_idx,
                &model,
                &mut paged_kv,
                &mut allocator,
                &mut scheduler,
            );
        }
    }
}

/// Process a prefill for one sequence.
fn process_prefill<M: Model>(
    idx: usize,
    token_ids: &[u32],
    model: &M,
    paged_kv: &mut PagedKvCache<M::CacheDtype>,
    allocator: &mut BlockAllocator,
    scheduler: &mut Scheduler,
) {
    // Run forward pass
    let (result, num_tokens) = {
        let seq = &scheduler.running()[idx];
        let start_pos = seq.block_table.seq_len();
        let r = model.forward_prefill_paged(
            token_ids,
            std::slice::from_mut(paged_kv),
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

/// Process a decode step for one sequence.
fn process_decode<M: Model>(
    idx: usize,
    model: &M,
    paged_kv: &mut PagedKvCache<M::CacheDtype>,
    allocator: &mut BlockAllocator,
    scheduler: &mut Scheduler,
) {
    // Extract the token to feed and run forward
    let result = {
        let seq = &scheduler.running()[idx];
        let tok = seq
            .generated_ids
            .last()
            .copied()
            .unwrap_or_else(|| *seq.prompt_ids.last().unwrap_or(&0));
        let start_pos = seq.block_table.seq_len();
        model.forward_prefill_paged(
            &[tok],
            std::slice::from_mut(paged_kv),
            &seq.block_table,
            start_pos,
        )
    };

    match result {
        Ok(logits) => {
            // Advance block table for the decoded token
            scheduler.running_mut()[idx].block_table.advance(1);

            // Extract data needed for sampling before mutable borrow
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
    logits: &infernum::CudaTensor<f32>,
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
