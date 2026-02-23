//! Token-level inference engine
//!
//! The [`Engine`] manages a model and its KV cache, providing token-level
//! generation (tokens in, tokens out). It is generic over any [`Model`]
//! implementation.
//!
//! The engine spawns a long-lived worker thread at construction. Callers
//! submit generation requests via [`Engine::submit`] and receive tokens
//! through a [`TokenSender`] implementation of their choice.

use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use infernum::cuda::ops::{argmax_last_scalar, sample_top_p};
use infernum::cuda::{CudaGraph, KvCache};
use infernum::{CudaTensor, GenerateOptions, Model, ModelConfig, Result, SamplingParams};

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

/// A generation request submitted to the engine's worker thread.
struct GenerationRequest {
    input_ids: Vec<u32>,
    options: GenerateOptions,
    token_tx: Box<dyn TokenSender>,
}

/// Default KV cache sequence length when no override is provided.
///
/// Models like DeepSeek-V3 declare `max_position_embeddings = 163840`,
/// which would require hundreds of GB of KV cache per GPU.  We default
/// to a much smaller value; callers can raise it with `Engine::with_max_seq_len`.
const DEFAULT_MAX_SEQ_LEN: usize = 4096;

/// Handle to the engine's worker thread.
///
/// The engine owns a long-lived thread that processes generation requests
/// sequentially. Callers submit requests via [`Engine::submit`] and receive
/// tokens through the provided [`TokenSender`].
pub struct Engine {
    request_tx: mpsc::Sender<GenerationRequest>,
    model_config: ModelConfig,
    _worker: JoinHandle<()>,
}

impl Engine {
    /// Create a new engine with the default KV cache size.
    ///
    /// The KV cache is sized to `min(model.max_seq_len, 4096)`.
    /// Use [`Engine::with_max_seq_len`] to override.
    ///
    /// # Errors
    /// Returns an error if KV cache allocation fails.
    pub fn new<M: Model + Send + 'static>(model: M) -> Result<Self> {
        Self::with_max_seq_len(model, None)
    }

    /// Create a new engine with an explicit KV cache size.
    ///
    /// Spawns a long-lived worker thread that owns the model and KV caches.
    /// The thread loops waiting for [`GenerationRequest`]s submitted via
    /// [`Engine::submit`].
    ///
    /// `max_seq_len` caps the KV cache allocation.  When `None`, defaults
    /// to `min(model.max_position_embeddings, 4096)`.
    ///
    /// # Errors
    /// Returns an error if KV cache allocation fails.
    pub fn with_max_seq_len<M: Model + Send + 'static>(
        model: M,
        max_seq_len: Option<usize>,
    ) -> Result<Self> {
        infernum::fusion::init();
        let mut model_config = model.config();
        let effective_max =
            max_seq_len.unwrap_or_else(|| model_config.max_seq_len.min(DEFAULT_MAX_SEQ_LEN));
        model_config.max_seq_len = effective_max;
        let kv_caches = model
            .devices()
            .iter()
            .map(|ctx| {
                KvCache::new(
                    ctx,
                    model_config.num_layers,
                    effective_max,
                    model_config.num_kv_heads,
                    model_config.head_dim,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let (request_tx, request_rx) = mpsc::channel::<GenerationRequest>();

        let worker = thread::spawn(move || {
            worker_loop(model, kv_caches, request_rx);
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

    /// Submit a generation request with a caller-provided token sender.
    ///
    /// Tokens are sent through `token_tx` as they are generated. The sender
    /// receives `Ok(token_id)` for each token, and generation stops when
    /// either the maximum token count is reached, an EOS token is produced,
    /// or the sender returns `false` (receiver dropped).
    pub fn submit(
        &self,
        input_ids: Vec<u32>,
        options: GenerateOptions,
        token_tx: impl TokenSender + 'static,
    ) {
        let request = GenerationRequest {
            input_ids,
            options,
            token_tx: Box::new(token_tx),
        };
        // If the worker thread has panicked, the send will fail.
        // That's fine — the receiver will see the channel close.
        let _ = self.request_tx.send(request);
    }

    /// Generate tokens, blocking until complete.
    ///
    /// # Arguments
    /// * `input_ids` - Prompt token IDs
    /// * `options` - Generation options (sampling, max tokens, KV cache, etc.)
    ///
    /// # Returns
    /// The generated token IDs (not including the prompt).
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
    /// * `options` - Generation options (sampling, max tokens, KV cache, etc.)
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

/// The worker thread's main loop. Processes requests sequentially.
#[allow(clippy::needless_pass_by_value)]
fn worker_loop<M: Model>(
    model: M,
    mut kv_caches: Vec<KvCache<M::CacheDtype>>,
    request_rx: mpsc::Receiver<GenerationRequest>,
) {
    while let Ok(request) = request_rx.recv() {
        if request.options.use_kv_cache {
            process_kv_cached(&model, &mut kv_caches, &request);
        } else {
            process_naive(&model, &request);
        }
    }
}

/// Process a single request using the KV cache (prefill + decode).
fn process_kv_cached<M: Model>(
    model: &M,
    kv_caches: &mut [KvCache<M::CacheDtype>],
    request: &GenerationRequest,
) {
    let mut run = || -> Result<FinishReason> {
        for kv in kv_caches.iter_mut() {
            kv.reset()?;
        }
        let mut tokens = request.input_ids.clone();
        let mut rng_state = request.options.sampling.as_ref().map(|s| s.seed);
        let sampling = request.options.sampling.as_ref();

        // Prefill: process entire prompt
        let logits = model.forward_with_kv_cache(&request.input_ids, kv_caches)?;
        let recent = recent_token_window(&tokens, sampling);
        let mut next_token = select_token(&logits, sampling, &mut rng_state, recent)?;

        if request.options.eos_token_id == Some(next_token) {
            return Ok(FinishReason::Stop);
        }
        tokens.push(next_token);
        if !request.token_tx.send(GenerationEvent::Token(next_token)) {
            return Ok(FinishReason::Cancelled);
        }

        // CUDA graphs only supported for single-device models
        if request.options.use_cuda_graphs
            && request.options.max_new_tokens > 1
            && kv_caches.len() == 1
        {
            stream_decode_loop_graph(
                model,
                kv_caches,
                &mut tokens,
                &mut next_token,
                &request.options,
                &mut rng_state,
                &*request.token_tx,
            )
        } else {
            stream_decode_loop_eager(
                model,
                kv_caches,
                &mut tokens,
                &mut next_token,
                &request.options,
                &mut rng_state,
                &*request.token_tx,
            )
        }
    };

    match run() {
        Ok(reason) => {
            let _ = request.token_tx.send(GenerationEvent::Finished(reason));
        }
        Err(e) => {
            let _ = request.token_tx.send(GenerationEvent::Error(e));
        }
    }
}

/// Process a single request without KV cache (recomputes full sequence each step).
fn process_naive<M: Model>(model: &M, request: &GenerationRequest) {
    let run = || -> Result<FinishReason> {
        let mut tokens = request.input_ids.clone();
        let mut rng_state = request.options.sampling.as_ref().map(|s| s.seed);
        let sampling = request.options.sampling.as_ref();

        for _ in 0..request.options.max_new_tokens {
            let logits = model.forward(&tokens)?;
            let recent = recent_token_window(&tokens, sampling);
            let next_token = select_token(&logits, sampling, &mut rng_state, recent)?;

            if request.options.eos_token_id == Some(next_token) {
                return Ok(FinishReason::Stop);
            }
            tokens.push(next_token);
            if !request.token_tx.send(GenerationEvent::Token(next_token)) {
                return Ok(FinishReason::Cancelled);
            }
        }

        Ok(FinishReason::Length)
    };

    match run() {
        Ok(reason) => {
            let _ = request.token_tx.send(GenerationEvent::Finished(reason));
        }
        Err(e) => {
            let _ = request.token_tx.send(GenerationEvent::Error(e));
        }
    }
}

/// Streaming decode loop without CUDA graph capture.
fn stream_decode_loop_eager<M: Model>(
    model: &M,
    kv_caches: &mut [KvCache<M::CacheDtype>],
    tokens: &mut Vec<u32>,
    next_token: &mut u32,
    options: &GenerateOptions,
    rng_state: &mut Option<u64>,
    tx: &dyn TokenSender,
) -> Result<FinishReason> {
    let sampling = options.sampling.as_ref();
    for _ in 1..options.max_new_tokens {
        let logits = model.forward_next_token(*next_token, kv_caches)?;
        let recent = recent_token_window(tokens, sampling);
        *next_token = select_token(&logits, sampling, rng_state, recent)?;

        if options.eos_token_id == Some(*next_token) {
            return Ok(FinishReason::Stop);
        }
        tokens.push(*next_token);
        if !tx.send(GenerationEvent::Token(*next_token)) {
            return Ok(FinishReason::Cancelled);
        }
    }
    Ok(FinishReason::Length)
}

/// Streaming decode loop with capture-once CUDA graph.
///
/// Step 1 runs eagerly (warmup — ensures all PTX modules are loaded).
/// Step 2: capture graph using indirect kernels, then launch.
/// Steps 3+: replay the captured graph (no re-capture).
///
/// `kv_cache.advance(1)` is called after each graph launch, outside the
/// captured region.
#[allow(clippy::too_many_arguments)]
fn stream_decode_loop_graph<M: Model>(
    model: &M,
    kv_caches: &mut [KvCache<M::CacheDtype>],
    tokens: &mut Vec<u32>,
    next_token: &mut u32,
    options: &GenerateOptions,
    rng_state: &mut Option<u64>,
    tx: &dyn TokenSender,
) -> Result<FinishReason> {
    let sampling = options.sampling.as_ref();

    // Cap the effective max_seq_len for graph-captured kernels
    let effective_max =
        (kv_caches[0].current_len() + options.max_new_tokens).min(kv_caches[0].max_seq_len());
    kv_caches[0].set_graph_max_seq_len(effective_max);

    let device = kv_caches[0].device();
    let mut graph = CudaGraph::new(&device)?;
    let mut captured_logits: Option<CudaTensor<f32>> = None;

    for step in 1..options.max_new_tokens {
        device.htod_copy_into(vec![*next_token], graph.token_input_mut())?;

        if step == 1 {
            // Warmup: eager (forward_next_token_device calls advance)
            let logits = model.forward_next_token_device(graph.token_input(), kv_caches)?;
            let recent = recent_token_window(tokens, sampling);
            *next_token = select_token(&logits, sampling, rng_state, recent)?;
        } else if step == 2 {
            // Capture once using indirect kernels
            graph.begin_capture()?;
            let logits = model.forward_next_token_indirect(graph.token_input(), kv_caches)?;
            graph.end_capture()?;

            graph.launch()?;
            device.synchronize()?;
            kv_caches[0].advance(1)?;

            let recent = recent_token_window(tokens, sampling);
            *next_token = select_token(&logits, sampling, rng_state, recent)?;
            captured_logits = Some(logits);
        } else {
            // Replay captured graph
            graph.launch()?;
            device.synchronize()?;
            kv_caches[0].advance(1)?;

            let logits = captured_logits
                .as_ref()
                .expect("graph must be captured before replay");
            let recent = recent_token_window(tokens, sampling);
            *next_token = select_token(logits, sampling, rng_state, recent)?;
        }

        if options.eos_token_id == Some(*next_token) {
            return Ok(FinishReason::Stop);
        }
        tokens.push(*next_token);
        if !tx.send(GenerationEvent::Token(*next_token)) {
            return Ok(FinishReason::Cancelled);
        }
    }
    Ok(FinishReason::Length)
}

/// Select the next token from logits, either via greedy argmax or sampling.
fn select_token(
    logits: &CudaTensor<f32>,
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
