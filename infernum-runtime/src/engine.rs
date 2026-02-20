//! Token-level inference engine
//!
//! The [`Engine`] manages a model and its KV cache, providing token-level
//! generation (tokens in, tokens out). It is generic over any [`Model`]
//! implementation.

use std::sync::mpsc;
use std::thread;

use infernum::cuda::ops::{argmax_last_scalar, sample_top_p};
use infernum::cuda::{CudaContext, CudaGraph, KvCache};
use infernum::{CudaTensor, GenerateOptions, Model, ModelConfig, Result, SamplingParams};

/// Token-level inference engine.
///
/// Wraps a model and manages its KV cache. Handles the prefill/decode loop
/// and sampling logic. Does not know about text — that is the Runtime's job.
pub struct Engine<M: Model> {
    model: M,
    model_config: ModelConfig,
    kv_cache: KvCache<M::CacheDtype>,
}

impl<M: Model> Engine<M> {
    /// Create a new engine wrapping the given model.
    ///
    /// # Errors
    /// Returns an error if KV cache allocation fails.
    pub fn new(ctx: &CudaContext, model: M) -> Result<Self> {
        infernum::fusion::init();
        let model_config = model.config();
        let kv_cache = KvCache::new(
            ctx,
            model_config.num_layers,
            model_config.max_seq_len,
            model_config.num_kv_heads,
            model_config.head_dim,
        )?;
        Ok(Self {
            model,
            model_config,
            kv_cache,
        })
    }

    /// Get a reference to the underlying model.
    #[must_use]
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get the model configuration.
    #[must_use]
    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }

    /// Generate tokens, blocking until complete.
    ///
    /// # Arguments
    /// * `input_ids` - Prompt token IDs
    /// * `options` - Generation options (sampling, max tokens, KV cache, etc.)
    ///
    /// # Returns
    /// The full token sequence (prompt + generated tokens)
    ///
    /// # Errors
    /// Returns an error if a forward pass fails.
    pub fn generate(&mut self, input_ids: &[u32], options: &GenerateOptions) -> Result<Vec<u32>> {
        if options.use_kv_cache {
            self.generate_kv_cached(input_ids, options)
        } else {
            self.generate_naive(input_ids, options)
        }
    }

    /// Generate tokens with streaming via a channel.
    ///
    /// Runs the generation loop in a scoped thread, sending each new token
    /// through a channel. The provided `consumer` closure receives the
    /// [`mpsc::Receiver`] and is called on the current thread while tokens
    /// are being produced.
    ///
    /// # Arguments
    /// * `input_ids` - Prompt token IDs
    /// * `options` - Generation options (sampling, max tokens, KV cache, etc.)
    /// * `consumer` - Closure that processes tokens from the receiver
    ///
    /// # Returns
    /// The return value of the `consumer` closure.
    #[allow(clippy::missing_panics_doc)]
    pub fn generate_stream<F, R>(
        &mut self,
        input_ids: &[u32],
        options: &GenerateOptions,
        consumer: F,
    ) -> R
    where
        M: Send,
        F: FnOnce(mpsc::Receiver<Result<u32>>) -> R,
    {
        let (tx, rx) = mpsc::channel();
        let input_ids = input_ids.to_vec();
        let options = options.clone();

        let mut result = None;

        thread::scope(|s| {
            s.spawn(|| {
                let gen_result = if options.use_kv_cache {
                    self.stream_kv_cached(&input_ids, &options, &tx)
                } else {
                    self.stream_naive(&input_ids, &options, &tx)
                };

                if let Err(e) = gen_result {
                    let _ = tx.send(Err(e));
                }
                // Drop tx so the receiver sees the channel close
                drop(tx);
            });

            result = Some(consumer(rx));
        });

        result.expect("thread::scope completed without producing a result")
    }

    /// Generate with KV cache (prefill + single-token decode steps).
    fn generate_kv_cached(
        &mut self,
        input_ids: &[u32],
        options: &GenerateOptions,
    ) -> Result<Vec<u32>> {
        self.kv_cache.reset()?;
        let mut tokens = input_ids.to_vec();
        let mut rng_state = options.sampling.as_ref().map(|s| s.seed);
        let sampling = options.sampling.as_ref();

        // Prefill: process entire prompt
        let logits = self
            .model
            .forward_with_kv_cache(input_ids, &mut self.kv_cache)?;
        let recent = recent_token_window(&tokens, sampling);
        let mut next_token = select_token(&logits, sampling, &mut rng_state, recent)?;

        if options.eos_token_id == Some(next_token) {
            return Ok(tokens);
        }
        tokens.push(next_token);

        if options.use_cuda_graphs && options.max_new_tokens > 1 {
            self.decode_loop_graph(&mut tokens, &mut next_token, options, &mut rng_state)?;
        } else {
            self.decode_loop_eager(&mut tokens, &mut next_token, options, &mut rng_state)?;
        }

        Ok(tokens)
    }

    /// Decode loop without CUDA graph capture (eager execution).
    fn decode_loop_eager(
        &mut self,
        tokens: &mut Vec<u32>,
        next_token: &mut u32,
        options: &GenerateOptions,
        rng_state: &mut Option<u64>,
    ) -> Result<()> {
        let sampling = options.sampling.as_ref();
        for _ in 1..options.max_new_tokens {
            let logits = self
                .model
                .forward_next_token(*next_token, &mut self.kv_cache)?;
            let recent = recent_token_window(tokens, sampling);
            *next_token = select_token(&logits, sampling, rng_state, recent)?;

            if options.eos_token_id == Some(*next_token) {
                break;
            }
            tokens.push(*next_token);
        }
        Ok(())
    }

    /// Decode loop with CUDA graph capture/replay.
    ///
    /// Step 1 runs eagerly (warmup — ensures all PTX modules are loaded).
    /// Capture-once CUDA graph decode loop.
    ///
    /// Step 1: eager warmup (loads PTX, validates shapes).
    /// Step 2: capture graph using indirect kernels, then launch.
    /// Steps 3+: replay the captured graph (no re-capture).
    ///
    /// `kv_cache.advance(1)` is called after each graph launch, outside the
    /// captured region.
    fn decode_loop_graph(
        &mut self,
        tokens: &mut Vec<u32>,
        next_token: &mut u32,
        options: &GenerateOptions,
        rng_state: &mut Option<u64>,
    ) -> Result<()> {
        use std::time::{Duration, Instant};

        let sampling = options.sampling.as_ref();

        // Cap the effective max_seq_len for graph-captured kernels to the actual
        // generation budget. This prevents shared memory from being sized for the
        // model's full max_position_embeddings (e.g. 131072) which would exceed
        // the per-block shared memory hardware limit.
        let effective_max =
            (self.kv_cache.current_len() + options.max_new_tokens).min(self.kv_cache.max_seq_len());
        self.kv_cache.set_graph_max_seq_len(effective_max);

        let device = self.kv_cache.device();
        let mut graph = CudaGraph::new(&device)?;

        let mut t_htod = Duration::ZERO;
        let mut t_advance = Duration::ZERO;
        let mut t_launch = Duration::ZERO;
        let mut t_sync = Duration::ZERO;
        let mut t_sample = Duration::ZERO;
        let mut graph_steps = 0_u64;

        // Holds the logits tensor from graph capture — its device address is
        // stable so we can sample from it after each replay.
        let mut captured_logits: Option<CudaTensor<f32>> = None;

        for step in 1..options.max_new_tokens {
            let t0 = Instant::now();
            device.htod_copy_into(vec![*next_token], graph.token_input_mut())?;
            t_htod += t0.elapsed();

            if step == 1 {
                // Warmup: eager execution to load all PTX kernels.
                // forward_next_token_device internally calls advance(1).
                let logits = self
                    .model
                    .forward_next_token_device(graph.token_input(), &mut self.kv_cache)?;
                let recent = recent_token_window(tokens, sampling);
                *next_token = select_token(&logits, sampling, rng_state, recent)?;
            } else if step == 2 {
                // Capture the graph once using indirect kernels.
                graph.begin_capture()?;
                let logits = self
                    .model
                    .forward_next_token_indirect(graph.token_input(), &mut self.kv_cache)?;
                graph.end_capture()?;

                // Launch the just-captured graph
                let t0 = Instant::now();
                graph.launch()?;
                t_launch += t0.elapsed();

                let t0 = Instant::now();
                device.synchronize()?;
                t_sync += t0.elapsed();

                // Advance position after graph execution (outside capture)
                let t0 = Instant::now();
                self.kv_cache.advance(1)?;
                t_advance += t0.elapsed();

                let t0 = Instant::now();
                let recent = recent_token_window(tokens, sampling);
                *next_token = select_token(&logits, sampling, rng_state, recent)?;
                t_sample += t0.elapsed();

                captured_logits = Some(logits);
                graph_steps += 1;
            } else {
                // Replay the captured graph (no re-capture)
                let t0 = Instant::now();
                graph.launch()?;
                t_launch += t0.elapsed();

                let t0 = Instant::now();
                device.synchronize()?;
                t_sync += t0.elapsed();

                let t0 = Instant::now();
                self.kv_cache.advance(1)?;
                t_advance += t0.elapsed();

                let t0 = Instant::now();
                let logits = captured_logits
                    .as_ref()
                    .expect("graph must be captured before replay");
                let recent = recent_token_window(tokens, sampling);
                *next_token = select_token(logits, sampling, rng_state, recent)?;
                t_sample += t0.elapsed();

                graph_steps += 1;
            }

            if options.eos_token_id == Some(*next_token) {
                break;
            }
            tokens.push(*next_token);
        }

        Self::print_graph_profiling(graph_steps, t_htod, t_launch, t_sync, t_advance, t_sample);

        Ok(())
    }

    #[allow(clippy::cast_precision_loss)]
    fn print_graph_profiling(
        steps: u64,
        t_htod: std::time::Duration,
        t_launch: std::time::Duration,
        t_sync: std::time::Duration,
        t_advance: std::time::Duration,
        t_sample: std::time::Duration,
    ) {
        use std::time::Duration;
        if steps == 0 {
            return;
        }
        let s = steps as f64;
        let row = |label: &str, d: Duration| {
            eprintln!(
                "  {label:<14} {:>8.2} ms total, {:>6.1} μs/step",
                d.as_secs_f64() * 1e3,
                d.as_secs_f64() * 1e6 / s
            );
        };
        eprintln!("--- CUDA graph decode profiling ({steps} steps, capture-once) ---");
        row("htod_copy:", t_htod);
        row("launch:", t_launch);
        row("sync:", t_sync);
        row("advance:", t_advance);
        row("sample:", t_sample);
        let total = t_htod + t_launch + t_sync + t_advance + t_sample;
        row("TOTAL:", total);
    }

    /// Generate without KV cache (recomputes full sequence each step).
    fn generate_naive(&self, input_ids: &[u32], options: &GenerateOptions) -> Result<Vec<u32>> {
        let mut tokens = input_ids.to_vec();
        let mut rng_state = options.sampling.as_ref().map(|s| s.seed);
        let sampling = options.sampling.as_ref();

        for _ in 0..options.max_new_tokens {
            let logits = self.model.forward(&tokens)?;
            let recent = recent_token_window(&tokens, sampling);
            let next_token = select_token(&logits, sampling, &mut rng_state, recent)?;

            if options.eos_token_id == Some(next_token) {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Stream with KV cache, sending each token through the channel.
    fn stream_kv_cached(
        &mut self,
        input_ids: &[u32],
        options: &GenerateOptions,
        tx: &mpsc::Sender<Result<u32>>,
    ) -> Result<()> {
        self.kv_cache.reset()?;
        let mut tokens = input_ids.to_vec();
        let mut rng_state = options.sampling.as_ref().map(|s| s.seed);
        let sampling = options.sampling.as_ref();

        // Prefill
        let logits = self
            .model
            .forward_with_kv_cache(input_ids, &mut self.kv_cache)?;
        let recent = recent_token_window(&tokens, sampling);
        let mut next_token = select_token(&logits, sampling, &mut rng_state, recent)?;

        if options.eos_token_id == Some(next_token) {
            return Ok(());
        }
        tokens.push(next_token);
        if tx.send(Ok(next_token)).is_err() {
            return Ok(());
        }

        // Decode
        if options.use_cuda_graphs && options.max_new_tokens > 1 {
            self.stream_decode_loop_graph(
                &mut tokens,
                &mut next_token,
                options,
                &mut rng_state,
                tx,
            )?;
        } else {
            self.stream_decode_loop_eager(
                &mut tokens,
                &mut next_token,
                options,
                &mut rng_state,
                tx,
            )?;
        }

        Ok(())
    }

    /// Streaming decode loop without CUDA graph capture.
    fn stream_decode_loop_eager(
        &mut self,
        tokens: &mut Vec<u32>,
        next_token: &mut u32,
        options: &GenerateOptions,
        rng_state: &mut Option<u64>,
        tx: &mpsc::Sender<Result<u32>>,
    ) -> Result<()> {
        let sampling = options.sampling.as_ref();
        for _ in 1..options.max_new_tokens {
            let logits = self
                .model
                .forward_next_token(*next_token, &mut self.kv_cache)?;
            let recent = recent_token_window(tokens, sampling);
            *next_token = select_token(&logits, sampling, rng_state, recent)?;

            if options.eos_token_id == Some(*next_token) {
                break;
            }
            tokens.push(*next_token);
            if tx.send(Ok(*next_token)).is_err() {
                break;
            }
        }
        Ok(())
    }

    /// Streaming decode loop with capture-once CUDA graph.
    fn stream_decode_loop_graph(
        &mut self,
        tokens: &mut Vec<u32>,
        next_token: &mut u32,
        options: &GenerateOptions,
        rng_state: &mut Option<u64>,
        tx: &mpsc::Sender<Result<u32>>,
    ) -> Result<()> {
        let sampling = options.sampling.as_ref();
        let device = self.kv_cache.device();
        let mut graph = CudaGraph::new(&device)?;
        let mut captured_logits: Option<CudaTensor<f32>> = None;

        for step in 1..options.max_new_tokens {
            device.htod_copy_into(vec![*next_token], graph.token_input_mut())?;

            if step == 1 {
                // Warmup: eager (forward_next_token_device calls advance)
                let logits = self
                    .model
                    .forward_next_token_device(graph.token_input(), &mut self.kv_cache)?;
                let recent = recent_token_window(tokens, sampling);
                *next_token = select_token(&logits, sampling, rng_state, recent)?;
            } else if step == 2 {
                // Capture once using indirect kernels
                graph.begin_capture()?;
                let logits = self
                    .model
                    .forward_next_token_indirect(graph.token_input(), &mut self.kv_cache)?;
                graph.end_capture()?;

                graph.launch()?;
                device.synchronize()?;
                self.kv_cache.advance(1)?;

                let recent = recent_token_window(tokens, sampling);
                *next_token = select_token(&logits, sampling, rng_state, recent)?;
                captured_logits = Some(logits);
            } else {
                // Replay captured graph
                graph.launch()?;
                device.synchronize()?;
                self.kv_cache.advance(1)?;

                let logits = captured_logits
                    .as_ref()
                    .expect("graph must be captured before replay");
                let recent = recent_token_window(tokens, sampling);
                *next_token = select_token(logits, sampling, rng_state, recent)?;
            }

            if options.eos_token_id == Some(*next_token) {
                break;
            }
            tokens.push(*next_token);
            if tx.send(Ok(*next_token)).is_err() {
                break;
            }
        }
        Ok(())
    }

    /// Stream without KV cache, sending each token through the channel.
    fn stream_naive(
        &self,
        input_ids: &[u32],
        options: &GenerateOptions,
        tx: &mpsc::Sender<Result<u32>>,
    ) -> Result<()> {
        let mut tokens = input_ids.to_vec();
        let mut rng_state = options.sampling.as_ref().map(|s| s.seed);
        let sampling = options.sampling.as_ref();

        for _ in 0..options.max_new_tokens {
            let logits = self.model.forward(&tokens)?;
            let recent = recent_token_window(&tokens, sampling);
            let next_token = select_token(&logits, sampling, &mut rng_state, recent)?;

            if options.eos_token_id == Some(next_token) {
                break;
            }
            tokens.push(next_token);
            if tx.send(Ok(next_token)).is_err() {
                break;
            }
        }

        Ok(())
    }
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
