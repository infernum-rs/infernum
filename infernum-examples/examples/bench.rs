//! Minimal benchmark for infernum decode throughput
//! Measures raw forward_next_token performance (no tokenizer decode needed)
//!
//! Four modes:
//! - **Eager (default):** Autoregressive decode — generates N tokens one-by-one
//!   using the runtime engine with KV cache.
//! - **Graph prefill (`--graph`):** Prefill throughput — processes N tokens in a
//!   single forward pass through the GPU graph executor. Llama family only.
//! - **Graph decode (`--graph-decode`):** Decode throughput — generates N tokens
//!   one-by-one through the GPU graph executor with KV cache management. Llama only.
//! - **CUDA graphs (`--cuda-graphs`):** Decode using CUDA graph capture/replay.
//!   Eliminates per-step kernel launch overhead by capturing the full decode
//!   step into a CUDA graph using indirect (GPU-resident position pointer) ops.
//!   Llama family SafeTensors dense only.
//!
//! Graph modes support SafeTensors (dense, FP8, GPTQ, AWQ) and GGUF (Q8_0, Q4_0, etc.).
//!
//! Usage:
//!   cargo run --release --example bench --features cuda -- /path/to/model 256
//!   cargo run --release --example bench --features cuda -- --graph /path/to/model 256
//!   cargo run --release --example bench --features cuda -- --graph-decode /path/to/model 128
//!   cargo run --release --example bench --features cuda -- --graph --dtype f32 /path/to/model 256
//!   cargo run --release --example bench --features cuda -- --graph /path/to/model.gguf 256

#![cfg(feature = "cuda")]

use std::path::Path;
use std::time::Instant;

use clap::Parser;
use serde::Deserialize;

use infernum::dtype::DType;
use infernum::graph::{plan, WeightId, WeightStore};
use infernum::tensor::Tensor;
use infernum::weights::WeightLoader;
use infernum::GenerateOptions;
use infernum_cuda::cuda::ops::{
    add_inplace, apply_rope_indirect, argmax_last_scalar, cast_f32_to_bf16,
    embedding_gather_from_device, fused_attention_decode_indirect, linear, rms_norm, swiglu,
    LinearWeight,
};
use infernum_cuda::cuda::{CudaContext, CudaGraph, CudaTensor, KvCache, SeqPosition};
use infernum_cuda::executor;
use infernum_cuda::{CudaBackend, CudaDecodeEngine};
use infernum_deepseek::DeepSeekModel;
use infernum_gemma::GemmaModel;
use infernum_llama::{
    build_decode_graph, build_indirect_decode_graph, build_prefill_graph, LlamaConfig, LlamaModel,
};
use infernum_qwen::QwenModel;
use infernum_runtime::Engine;

#[derive(Parser)]
#[command(name = "bench")]
struct Cli {
    /// Path to model directory or .gguf file
    model: String,

    /// Number of tokens to generate (eager) or process (graph)
    #[arg(default_value_t = 128)]
    n_gen: usize,

    /// Use graph executor for prefill throughput (SafeTensors + Llama only)
    #[arg(long)]
    graph: bool,

    /// Use graph executor for decode (autoregressive) throughput
    #[arg(long)]
    graph_decode: bool,

    /// Use CUDA graph capture/replay for decode (Llama SafeTensors dense only)
    #[arg(long)]
    cuda_graphs: bool,

    /// Use the architecture-correct CUDA graph engine (indirect ops, no bypass)
    #[arg(long)]
    cuda_graph_engine: bool,

    /// Weight dtype for graph mode (f32 or bf16)
    #[arg(long, default_value = "bf16")]
    dtype: String,
}

/// Peek at just the `model_type` field from config.json.
#[derive(Deserialize)]
struct ModelTypeProbe {
    #[serde(default = "default_model_type")]
    model_type: String,
}

fn default_model_type() -> String {
    "llama".to_string()
}

fn detect_model_type(model_path: &str) -> infernum::Result<String> {
    let config_path = Path::new(model_path).join("config.json");
    let data = std::fs::read_to_string(&config_path).map_err(|e| {
        infernum::Error::Io(std::io::Error::new(
            e.kind(),
            format!("Failed to read {}: {e}", config_path.display()),
        ))
    })?;
    let probe: ModelTypeProbe = serde_json::from_str(&data).map_err(|e| {
        infernum::Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to parse config.json: {e}"),
        ))
    })?;
    Ok(probe.model_type)
}

fn bench_model<M: infernum::Model + Send + 'static>(model: M, n_gen: usize) -> infernum::Result<()>
where
    M::B: infernum::DecodeBufferOps,
{
    let prompt = vec![1u32, 15043, 29892, 920, 526, 366, 2599, 13];

    let engine = Engine::new(model)?;

    let options = GenerateOptions {
        max_new_tokens: n_gen,
        ..GenerateOptions::default()
    };

    let start = Instant::now();
    let tokens = engine.generate(&prompt, &options)?;
    let elapsed = start.elapsed();

    let generated = tokens.len() - prompt.len();
    let tok_s = generated as f64 / elapsed.as_secs_f64();

    println!(
        "{generated} tokens in {:.2}s = {:.1} tok/s",
        elapsed.as_secs_f64(),
        tok_s,
    );

    Ok(())
}

fn bench_with_info<M: infernum::Model + Send + 'static>(
    model: M,
    num_layers: usize,
    hidden_size: usize,
    dtype: &str,
    n_gen: usize,
) -> infernum::Result<()>
where
    M::B: infernum::DecodeBufferOps,
{
    eprintln!(
        "Model loaded: {} layers, {} hidden, dtype={}",
        num_layers, hidden_size, dtype,
    );
    bench_model(model, n_gen)
}

/// Load graph weights from a SafeTensors directory into the `WeightStore`.
fn load_graph_weights(
    ctx: &CudaContext,
    graph: &infernum::graph::Graph<CudaBackend>,
    model_dir: &str,
    weights: &mut WeightStore<CudaTensor, LinearWeight>,
) -> infernum::Result<()> {
    let format_loader = infernum_cuda::SafeTensorsLoader::from_directory(Path::new(model_dir))?;
    let loader = infernum_cuda::CudaWeightLoader::new(ctx.clone(), format_loader);

    let config_path = Path::new(model_dir).join("config.json");
    let config_data = std::fs::read_to_string(&config_path).ok();
    let quant_config: Option<infernum::QuantizationConfig> = config_data
        .as_ref()
        .and_then(|d| serde_json::from_str::<serde_json::Value>(d).ok())
        .and_then(|v| v.get("quantization_config").cloned())
        .and_then(|v| serde_json::from_value(v).ok());

    for i in 0..graph.tensor_weight_count() {
        let meta = graph.tensor_weight_meta(WeightId::from_index(i as u32));
        let tensor = loader.load_tensor(&meta.name, meta.dtype)?;
        weights.push_tensor_weight(tensor);
    }

    // GPTQ/AWQ models store .qweight/.scales/.qzeros instead of .weight,
    // so loader.contains(name) returns false.  load_linear handles the
    // sub-tensor lookup internally, so skip the contains check for those.
    let is_packed_quant = quant_config.as_ref().map_or(false, |qc| {
        qc.quant_method == "gptq" || qc.quant_method == "awq"
    });

    for i in 0..graph.linear_weight_count() {
        let meta = graph.linear_weight_meta(WeightId::from_index(i as u32));
        if is_packed_quant || loader.contains(&meta.name) {
            // For tied embeddings (lm_head) in GPTQ/AWQ, the prefix
            // "lm_head.weight" won't have sub-tensors — fall through to
            // the embed_tokens fallback below on error.
            let result = loader.load_linear(&meta.name, meta.dtype, quant_config.as_ref());
            match result {
                Ok(linear) => {
                    weights.push_linear_weight(linear);
                    continue;
                }
                Err(_) if meta.name == "lm_head.weight" => {
                    // Fall through to tied-embedding path
                }
                Err(e) => return Err(e),
            }
        }
        if meta.name == "lm_head.weight" {
            // Tied embeddings: lm_head shares embed_tokens weight.
            // Pass None for quant_config — the embedding is always dense,
            // even in GPTQ/AWQ models where only projections are quantized.
            let linear = loader.load_linear("model.embed_tokens.weight", meta.dtype, None)?;
            weights.push_linear_weight(linear);
        } else {
            return Err(infernum::Error::WeightNotFound(meta.name.clone()));
        }
    }

    Ok(())
}

/// Map a SafeTensors-convention weight name to a GGUF tensor name.
fn safetensors_to_gguf_name(name: &str) -> String {
    match name {
        "model.embed_tokens.weight" => return "token_embd.weight".to_string(),
        "model.norm.weight" => return "output_norm.weight".to_string(),
        "lm_head.weight" => return "output.weight".to_string(),
        _ => {}
    }
    if let Some(rest) = name.strip_prefix("model.layers.") {
        let dot = rest.find('.').expect("malformed layer weight name");
        let layer_idx = &rest[..dot];
        let suffix = &rest[dot + 1..];
        let gguf_suffix = match suffix {
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "ffn_norm.weight",
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            other => panic!("Unknown layer suffix: {other}"),
        };
        return format!("blk.{layer_idx}.{gguf_suffix}");
    }
    panic!("Unknown weight name: {name}");
}

/// Returns true if this GGUF weight name is a Q or K projection that needs unpermuting.
fn needs_unpermute(name: &str) -> bool {
    name.ends_with(".attn_q.weight") || name.ends_with(".attn_k.weight")
}

/// Load graph weights from a GGUF file into the GPU `WeightStore`.
///
/// Loads quantized weights natively (Q8_0, Q4_0, Q6_K) and dense weights
/// in their original format with GPU transpose for linear layers.
fn load_graph_weights_gguf(
    ctx: &CudaContext,
    graph: &infernum::graph::Graph<CudaBackend>,
    config: &LlamaConfig,
    gguf_path: &str,
    weight_dtype: DType,
    weights: &mut WeightStore<CudaTensor, LinearWeight>,
) -> infernum::Result<()> {
    use infernum_cuda::cuda::ops::{transpose_2d, transpose_2d_bf16};
    use infernum_cuda::WeightLoader as CudaWeightLoader;

    let loader = infernum_cuda::GgufLoader::from_file(gguf_path)?;

    // Tensor weights (embeddings, layernorms) — loaded as f32 (load_f32
    // handles dequantization of Q8_0/Q4_0), then cast to the compute dtype
    // so that BF16 kernels (rmsnorm, embedding_gather) receive BF16 data.
    for i in 0..graph.tensor_weight_count() {
        let meta = graph.tensor_weight_meta(WeightId::from_index(i as u32));
        let gguf_name = safetensors_to_gguf_name(&meta.name);
        let tensor_f32 = loader.load_f32(ctx, &gguf_name)?;
        let tensor = if weight_dtype == DType::BF16 {
            cast_f32_to_bf16(&tensor_f32)?
        } else {
            tensor_f32
        };
        weights.push_tensor_weight(tensor);
    }

    // Linear weights — loaded in native format (quantized or dense)
    for i in 0..graph.linear_weight_count() {
        let meta = graph.linear_weight_meta(WeightId::from_index(i as u32));
        let gguf_name = safetensors_to_gguf_name(&meta.name);

        // Resolve actual GGUF name (handle tied embeddings)
        let actual_name = if loader.contains(&gguf_name) {
            gguf_name.clone()
        } else if meta.name == "lm_head.weight" {
            "token_embd.weight".to_string()
        } else {
            return Err(infernum::Error::WeightNotFound(gguf_name));
        };

        let dtype = loader.get_dtype(&actual_name)?;

        let linear = if dtype.is_quantized() {
            if needs_unpermute(&gguf_name) {
                let n_head = if gguf_name.contains("attn_q") {
                    config.num_attention_heads
                } else {
                    config
                        .num_key_value_heads
                        .unwrap_or(config.num_attention_heads)
                };
                LinearWeight::Quantized(loader.load_quantized_unpermute(
                    ctx,
                    &actual_name,
                    n_head,
                )?)
            } else {
                LinearWeight::Quantized(loader.load_quantized(ctx, &actual_name)?)
            }
        } else {
            // Dense: load, unpermute Q/K if needed, transpose for matmul
            let tensor = match weight_dtype {
                DType::BF16 => loader.load_bf16(ctx, &actual_name)?,
                _ => loader.load_f32(ctx, &actual_name)?,
            };
            // Note: dense GGUF Q/K weights from llama.cpp are stored in
            // interleaved RoPE order. We don't unpermute dense weights
            // on GPU currently — unpermuting packed byte data is complex.
            // For F32/BF16 GGUF, the standard llama.cpp convert script
            // does not apply permutation to dense weights, so this is safe.
            match weight_dtype {
                DType::BF16 => LinearWeight::Dense(transpose_2d_bf16(&tensor)?),
                _ => LinearWeight::Dense(transpose_2d(&tensor)?),
            }
        };

        weights.push_linear_weight(linear);
    }

    Ok(())
}

/// Benchmark prefill throughput using the GPU graph executor.
fn bench_graph(
    ctx: &CudaContext,
    model_path: &str,
    n_tokens: usize,
    weight_dtype: DType,
) -> infernum::Result<()> {
    let is_gguf = model_path.ends_with(".gguf");

    let config: LlamaConfig = if is_gguf {
        let gguf = infernum::weights::gguf::GgufLoader::from_file(model_path)?;
        LlamaConfig::from_gguf_metadata(gguf.metadata())?
    } else {
        let config_path = Path::new(model_path).join("config.json");
        let data = std::fs::read_to_string(&config_path).map_err(|e| {
            infernum::Error::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to read {}: {e}", config_path.display()),
            ))
        })?;
        serde_json::from_str(&data).map_err(|e| {
            infernum::Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to parse config.json: {e}"),
            ))
        })?
    };

    let head_dim = config.head_dim();
    let half_dim = head_dim / 2;

    eprintln!(
        "Model: {} layers, {} hidden (GPU graph prefill, seq_len={})",
        config.num_hidden_layers, config.hidden_size, n_tokens,
    );

    // Build graph
    let (graph, _model_weights) =
        build_prefill_graph::<CudaBackend>(&config, n_tokens, weight_dtype);
    let exec_plan = plan(&graph);

    let mut weights = WeightStore::<CudaTensor, LinearWeight>::new();
    eprintln!("Loading weights...");
    if is_gguf {
        load_graph_weights_gguf(ctx, &graph, &config, model_path, weight_dtype, &mut weights)?;
    } else {
        load_graph_weights(ctx, &graph, model_path, &mut weights)?;
    }

    // Build inputs: token_ids, cos_cache, sin_cache (on GPU)
    let token_ids: Vec<u32> = (0..n_tokens).map(|i| (i % 256) as u32).collect();
    let input_ids = CudaTensor::from_slice(ctx, &[n_tokens], &token_ids)?;

    let (cos_data, sin_data) =
        infernum::rope::precompute_rope_data(n_tokens, head_dim, config.rope_theta);
    let cos_cache = CudaTensor::from_slice(ctx, &[n_tokens, half_dim], &cos_data)?;
    let sin_cache = CudaTensor::from_slice(ctx, &[n_tokens, half_dim], &sin_data)?;

    let inputs = vec![input_ids, cos_cache, sin_cache];

    // Warm-up run
    {
        let outputs = executor::execute(
            &exec_plan,
            graph.nodes(),
            &weights,
            &inputs,
            graph.output_ids(),
        )?;
        ctx.synchronize()?;
        assert_eq!(outputs[0].shape()[0], n_tokens);
        eprintln!("Warm-up done, output shape: {:?}", outputs[0].shape());
    }

    // Benchmark: run 3 iterations and take the best
    let n_iters = 3;
    let mut best_elapsed = std::time::Duration::MAX;

    for iter in 0..n_iters {
        ctx.synchronize()?;
        let start = Instant::now();
        let _outputs = executor::execute(
            &exec_plan,
            graph.nodes(),
            &weights,
            &inputs,
            graph.output_ids(),
        )?;
        ctx.synchronize()?;
        let elapsed = start.elapsed();
        if elapsed < best_elapsed {
            best_elapsed = elapsed;
        }
        eprintln!(
            "  iter {}: {:.2}s ({:.1} tok/s)",
            iter + 1,
            elapsed.as_secs_f64(),
            n_tokens as f64 / elapsed.as_secs_f64(),
        );
    }

    let tok_s = n_tokens as f64 / best_elapsed.as_secs_f64();
    println!(
        "{n_tokens} tokens in {:.2}s = {:.1} tok/s",
        best_elapsed.as_secs_f64(),
        tok_s,
    );

    Ok(())
}

/// Benchmark decode throughput using the GPU graph executor with KV cache management.
///
/// Rebuilds the decode graph each step (node shapes depend on `kv_len`), loads
/// real weights once via a temporary prefill graph, then generates tokens
/// autoregressively using `argmax`.
#[allow(clippy::too_many_lines)]
fn bench_graph_decode(
    ctx: &CudaContext,
    model_path: &str,
    n_gen: usize,
    weight_dtype: DType,
) -> infernum::Result<()> {
    let is_gguf = model_path.ends_with(".gguf");

    let config: LlamaConfig = if is_gguf {
        let gguf = infernum::weights::gguf::GgufLoader::from_file(model_path)?;
        LlamaConfig::from_gguf_metadata(gguf.metadata())?
    } else {
        let config_path = Path::new(model_path).join("config.json");
        let data = std::fs::read_to_string(&config_path).map_err(|e| {
            infernum::Error::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to read {}: {e}", config_path.display()),
            ))
        })?;
        serde_json::from_str(&data).map_err(|e| {
            infernum::Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to parse config.json: {e}"),
            ))
        })?
    };

    let num_layers = config.num_hidden_layers;
    let head_dim = config.head_dim();
    let half_dim = head_dim / 2;
    let num_kv_heads = config.num_kv_heads();

    eprintln!(
        "Model: {} layers, {} hidden (GPU graph decode, n_gen={})",
        num_layers, config.hidden_size, n_gen,
    );

    // Load weights using a temporary prefill graph (weight order is consistent)
    let prompt_len = 8;
    let mut weights = WeightStore::<CudaTensor, LinearWeight>::new();
    eprintln!("Loading weights...");
    {
        let (tmp_graph, _) = build_prefill_graph::<CudaBackend>(&config, 1, weight_dtype);
        if is_gguf {
            load_graph_weights_gguf(
                ctx,
                &tmp_graph,
                &config,
                model_path,
                weight_dtype,
                &mut weights,
            )?;
        } else {
            load_graph_weights(ctx, &tmp_graph, model_path, &mut weights)?;
        }
    }

    // Pre-compute RoPE for all positions we'll need
    let max_pos = prompt_len + n_gen;
    let (all_cos, all_sin) =
        infernum::rope::precompute_rope_data(max_pos, head_dim, config.rope_theta);

    // KV caches: per-layer k and v tensors, grown each step via graph outputs.
    // Start empty — first decode step uses kv_len=0.
    let mut kv_caches: Vec<Option<CudaTensor>> = vec![None; 2 * num_layers];

    // Helper: run one decode step at the given position.
    let run_decode_step = |pos: usize,
                           token: u32,
                           kv_caches: &mut [Option<CudaTensor>],
                           weights: &WeightStore<CudaTensor, LinearWeight>|
     -> infernum::Result<u32> {
        let kv_len = pos; // number of cached KV entries before this step

        // Build decode graph for current kv_len
        let (graph, _) = build_decode_graph::<CudaBackend>(&config, kv_len, weight_dtype);
        let exec_plan = plan(&graph);

        // Build inputs: token_id, cos, sin, then per-layer k_cache, v_cache
        let cos_start = pos * half_dim;
        let mut inputs = vec![
            CudaTensor::from_slice(ctx, &[1], &[token])?,
            CudaTensor::from_slice(
                ctx,
                &[1, half_dim],
                &all_cos[cos_start..cos_start + half_dim],
            )?,
            CudaTensor::from_slice(
                ctx,
                &[1, half_dim],
                &all_sin[cos_start..cos_start + half_dim],
            )?,
        ];

        // Append KV cache inputs (empty tensors for kv_len=0)
        for layer in 0..num_layers {
            let k_idx = layer;
            let v_idx = num_layers + layer;
            if let Some(k) = &kv_caches[k_idx] {
                inputs.push(k.clone());
            } else {
                inputs.push(CudaTensor::zeros(
                    ctx,
                    &[0, num_kv_heads, head_dim],
                    weight_dtype,
                )?);
            }
            if let Some(v) = &kv_caches[v_idx] {
                inputs.push(v.clone());
            } else {
                inputs.push(CudaTensor::zeros(
                    ctx,
                    &[0, num_kv_heads, head_dim],
                    weight_dtype,
                )?);
            }
        }

        let outputs = executor::execute(
            &exec_plan,
            graph.nodes(),
            weights,
            &inputs,
            graph.output_ids(),
        )?;

        // Outputs: [logits, k_0, k_1, ..., k_{n-1}, v_0, v_1, ..., v_{n-1}]
        // Store updated KV caches for the next step
        for layer in 0..num_layers {
            kv_caches[layer] = Some(outputs[1 + layer].clone());
            kv_caches[num_layers + layer] = Some(outputs[1 + num_layers + layer].clone());
        }

        // Argmax the logits on GPU (transfers only 4 bytes)
        argmax_last_scalar(&outputs[0])
    };

    // --- Phase 1: Warm-up — decode prompt tokens one at a time (not timed) ---
    eprintln!("Warm-up: decoding {prompt_len} prompt tokens...");
    let mut last_token = 0u32;
    for pos in 0..prompt_len {
        let token = (pos % 256) as u32;
        last_token = run_decode_step(pos, token, &mut kv_caches, &weights)?;
    }
    ctx.synchronize()?;
    eprintln!("Warm-up done.");

    // --- Phase 2: Timed decode of n_gen tokens ---
    ctx.synchronize()?;
    let start = Instant::now();

    for step in 0..n_gen {
        let pos = prompt_len + step;
        last_token = run_decode_step(pos, last_token, &mut kv_caches, &weights)?;
    }

    ctx.synchronize()?;
    let elapsed = start.elapsed();
    let tok_s = n_gen as f64 / elapsed.as_secs_f64();

    println!(
        "{n_gen} tokens in {:.2}s = {:.1} tok/s",
        elapsed.as_secs_f64(),
        tok_s,
    );

    Ok(())
}

/// Benchmark decode throughput using CUDA graph capture/replay with indirect ops.
///
/// Each decode step is captured into a [`CudaGraph`]. The graph reads positions
/// and attention lengths from stable GPU-resident pointers (`SeqPosition`), so
/// only the values at those addresses need to change between replays — no graph
/// rebuild or re-instantiation is required after the first step.
///
/// Supports dense SafeTensors (F32, BF16) Llama family models only.
///
/// # How it works
///
/// 1. Load weights via a single-token prefill graph (same naming convention).
/// 2. Precompute a full cos/sin RoPE table on GPU for all positions.
/// 3. Allocate a [`KvCache`] sized for `prompt_len + n_gen` tokens.
/// 4. Warm up by appending prompt tokens eagerly (not captured).
/// 5. In the timed loop: write the next token ID into the graph's pre-allocated
///    device buffer, capture the forward pass (with indirect ops), launch the
///    graph, synchronize, advance the KV cache position counters.
///
/// # Errors
/// Returns an error if weight loading or the GPU kernel fails.
#[allow(clippy::too_many_lines)]
fn bench_cuda_graphs_decode(
    ctx: &CudaContext,
    model_path: &str,
    n_gen: usize,
    weight_dtype: DType,
) -> infernum::Result<()> {
    use infernum_cuda::cuda::ops::{apply_rope, fused_attention_decode, precompute_rope_cache};

    assert!(
        !model_path.ends_with(".gguf"),
        "--cuda-graphs only supports SafeTensors directories"
    );

    let config_path = Path::new(model_path).join("config.json");
    let config_data = std::fs::read_to_string(&config_path).map_err(|e| {
        infernum::Error::Io(std::io::Error::new(
            e.kind(),
            format!("Failed to read {}: {e}", config_path.display()),
        ))
    })?;
    let config: LlamaConfig = serde_json::from_str(&config_data).map_err(|e| {
        infernum::Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to parse config.json: {e}"),
        ))
    })?;

    let num_layers = config.num_hidden_layers;
    let hidden_size = config.hidden_size;
    let head_dim = config.head_dim();
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads();
    let eps = config.rms_norm_eps;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let half_dim = head_dim / 2;

    eprintln!(
        "Model: {} layers, {} hidden (CUDA graph decode, n_gen={})",
        num_layers, hidden_size, n_gen,
    );

    // Load weights via a temporary single-token prefill graph (identical weight layout).
    let mut weights = WeightStore::<CudaTensor, LinearWeight>::new();
    let model_weight_ids;
    eprintln!("Loading weights...");
    {
        let (tmp_graph, ids) = build_prefill_graph::<CudaBackend>(&config, 1, weight_dtype);
        model_weight_ids = ids;
        load_graph_weights(ctx, &tmp_graph, model_path, &mut weights)?;
    }

    // Full RoPE cache for all positions we'll ever need (prompt + generated).
    let prompt_len = 8_usize;
    let max_pos = prompt_len + n_gen;
    let (cos_cache, sin_cache) = precompute_rope_cache(ctx, max_pos, head_dim, config.rope_theta)?;

    // Pre-compute all rope data on host for the eager warmup loop.
    let (all_cos_host, all_sin_host) =
        infernum::rope::precompute_rope_data(max_pos, head_dim, config.rope_theta);

    // KV cache sized for the full generation window.
    let mut kv_cache = KvCache::new(
        ctx,
        num_layers,
        max_pos,
        num_kv_heads,
        head_dim,
        weight_dtype,
    )?;
    kv_cache.set_graph_max_seq_len(max_pos);
    let graph_max_seq_len = max_pos;

    // CUDA graph manager (pre-allocates a stable device buffer for the token ID).
    let mut cuda_graph = CudaGraph::new(ctx.device())?;

    // --- Phase 1: Eager warmup — populate KV cache with prompt tokens (not timed) ---
    eprintln!("Warm-up: decoding {prompt_len} prompt tokens...");

    let mut last_token = 0u32;
    for pos in 0..prompt_len {
        let token = (pos % 256) as u32;
        let mut hidden = infernum_cuda::cuda::ops::embedding_gather(
            ctx,
            weights.tensor_weight(model_weight_ids.embed_tokens),
            &[token],
        )?;

        for layer_idx in 0..num_layers {
            let ids = &model_weight_ids.layers[layer_idx];

            let normed = rms_norm(&hidden, weights.tensor_weight(ids.input_layernorm), eps)?;
            let q_flat = linear(&normed, weights.linear_weight(ids.q_proj))?;
            let k_flat = linear(&normed, weights.linear_weight(ids.k_proj))?;
            let v_flat = linear(&normed, weights.linear_weight(ids.v_proj))?;

            let q = q_flat.reshape(&[1, num_heads, head_dim]);
            let k = k_flat.reshape(&[1, num_kv_heads, head_dim]);
            let v = v_flat.reshape(&[1, num_kv_heads, head_dim]);

            // Host-scalar RoPE for warmup (not captured in graph).
            let cos_start = pos * half_dim;
            let cos_slice = CudaTensor::from_slice(
                ctx,
                &[1, half_dim],
                &all_cos_host[cos_start..cos_start + half_dim],
            )?;
            let sin_slice = CudaTensor::from_slice(
                ctx,
                &[1, half_dim],
                &all_sin_host[cos_start..cos_start + half_dim],
            )?;
            let q_rot = apply_rope(&q, &cos_slice, &sin_slice, 0)?;
            let k_rot = apply_rope(&k, &cos_slice, &sin_slice, 0)?;

            // Eager append — writes at current_len (host-side offset).
            kv_cache.append(layer_idx, &k_rot, &v)?;

            // Attend over all cached KV including the just-appended token.
            let total = kv_cache.current_len() + 1;
            let (k_full, v_full) = kv_cache.get_up_to(layer_idx, total);
            let attn = fused_attention_decode(
                &q_rot,
                &k_full,
                &v_full,
                Some(scale),
                None,
                config.effective_sliding_window(layer_idx),
            )?;
            let attn_flat = attn.reshape(&[1, hidden_size]);
            let attn_proj = linear(&attn_flat, weights.linear_weight(ids.o_proj))?;
            add_inplace(&mut hidden, &attn_proj)?;

            let ffn_normed = rms_norm(
                &hidden,
                weights.tensor_weight(ids.post_attention_layernorm),
                eps,
            )?;
            let gate = linear(&ffn_normed, weights.linear_weight(ids.gate_proj))?;
            let up = linear(&ffn_normed, weights.linear_weight(ids.up_proj))?;
            let activated = swiglu(&gate, &up)?;
            let ffn_out = linear(&activated, weights.linear_weight(ids.down_proj))?;
            add_inplace(&mut hidden, &ffn_out)?;
        }

        let final_normed = rms_norm(
            &hidden,
            weights.tensor_weight(model_weight_ids.final_norm),
            eps,
        )?;
        let logits = linear(
            &final_normed,
            weights.linear_weight(model_weight_ids.lm_head),
        )?;
        last_token = argmax_last_scalar(&logits)?;

        // Advance position counters after all layers have appended for this step.
        kv_cache.advance(1)?;
    }
    ctx.synchronize()?;
    eprintln!(
        "Warm-up done. kv_cache.current_len() = {}",
        kv_cache.current_len()
    );

    // --- Phase 2: Timed decode — CUDA graph capture + replay ---
    ctx.synchronize()?;
    let start = std::time::Instant::now();

    for _step in 0..n_gen {
        // Write next token into the stable device buffer before capture.
        ctx.device()
            .htod_copy_into(vec![last_token], cuda_graph.token_input_mut())?;

        // Begin stream capture — all subsequent GPU work is recorded, not executed.
        cuda_graph.begin_capture()?;

        // Embed from device-side token ID (stable GPU address → graph-capturable).
        let mut hidden = embedding_gather_from_device(
            ctx,
            weights.tensor_weight(model_weight_ids.embed_tokens),
            cuda_graph.token_input(),
            1,
        )?;

        for layer_idx in 0..num_layers {
            let ids = &model_weight_ids.layers[layer_idx];

            let normed = rms_norm(&hidden, weights.tensor_weight(ids.input_layernorm), eps)?;
            let q_flat = linear(&normed, weights.linear_weight(ids.q_proj))?;
            let k_flat = linear(&normed, weights.linear_weight(ids.k_proj))?;
            let v_flat = linear(&normed, weights.linear_weight(ids.v_proj))?;

            let q = q_flat.reshape(&[1, num_heads, head_dim]);
            let k = k_flat.reshape(&[1, num_kv_heads, head_dim]);
            let v = v_flat.reshape(&[1, num_kv_heads, head_dim]);

            // Indirect RoPE: position read from stable GPU pointer at execution time.
            let q_rot =
                apply_rope_indirect(&q, &cos_cache, &sin_cache, kv_cache.current_position())?;
            let k_rot =
                apply_rope_indirect(&k, &cos_cache, &sin_cache, kv_cache.current_position())?;

            // Indirect KV append: write offset read from stable GPU pointer.
            kv_cache.append_indirect(layer_idx, &k_rot, &v)?;

            // Full-length buffers: GPU addresses are stable across all replays.
            let (k_full, v_full) = kv_cache.full_buffers(layer_idx);

            // Indirect attention: total_len read from stable GPU pointer.
            let attn = fused_attention_decode_indirect(
                &q_rot,
                k_full,
                v_full,
                kv_cache.current_total_len(),
                graph_max_seq_len,
                Some(scale),
                None,
                config.effective_sliding_window(layer_idx),
            )?;
            let attn_flat = attn.reshape(&[1, hidden_size]);
            let attn_proj = linear(&attn_flat, weights.linear_weight(ids.o_proj))?;
            add_inplace(&mut hidden, &attn_proj)?;

            let ffn_normed = rms_norm(
                &hidden,
                weights.tensor_weight(ids.post_attention_layernorm),
                eps,
            )?;
            let gate = linear(&ffn_normed, weights.linear_weight(ids.gate_proj))?;
            let up = linear(&ffn_normed, weights.linear_weight(ids.up_proj))?;
            let activated = swiglu(&gate, &up)?;
            let ffn_out = linear(&activated, weights.linear_weight(ids.down_proj))?;
            add_inplace(&mut hidden, &ffn_out)?;
        }

        let final_normed = rms_norm(
            &hidden,
            weights.tensor_weight(model_weight_ids.final_norm),
            eps,
        )?;
        let logits = linear(
            &final_normed,
            weights.linear_weight(model_weight_ids.lm_head),
        )?;

        // End capture and instantiate (or update in-place via cuGraphExecUpdate_v2).
        cuda_graph.end_capture()?;
        cuda_graph.launch()?;

        // Synchronize before reading logits and advancing the KV cache.
        ctx.synchronize()?;

        // Advance position counters (htod_copy_into — must be outside the graph).
        kv_cache.advance(1)?;

        // Argmax from the GPU logits buffer (pool returns same address each recapture).
        last_token = argmax_last_scalar(&logits)?;
    }

    ctx.synchronize()?;
    let elapsed = start.elapsed();
    let tok_s = n_gen as f64 / elapsed.as_secs_f64();

    println!(
        "{n_gen} tokens in {:.2}s = {:.1} tok/s",
        elapsed.as_secs_f64(),
        tok_s,
    );

    Ok(())
}

/// Benchmark decode throughput using the architecture-correct `CudaDecodeEngine`.
///
/// Unlike `bench_cuda_graphs_decode` (the bypass), this path uses the computation
/// graph IR and `execute_indirect` — the same path that will become the production
/// inference path. Specifically:
///
/// - Builds a graph via `build_indirect_decode_graph` (no `kv_len` parameter).
/// - Populates a `WeightStore` with model weights, RoPE cos/sin caches, and
///   per-layer KV cache buffer handles.
/// - Runs `CudaDecodeEngine` which re-captures until the buffer pool stabilizes,
///   then replays with bare `launch()` calls.
///
/// Supports dense SafeTensors (F32, BF16) Llama-family models only.
///
/// # Errors
/// Returns an error if weight loading or any CUDA kernel launch fails.
#[allow(clippy::too_many_lines)]
fn bench_cuda_graph_engine(
    ctx: &CudaContext,
    model_path: &str,
    n_gen: usize,
    weight_dtype: DType,
) -> infernum::Result<()> {
    use infernum_cuda::cuda::ops::{apply_rope, fused_attention_decode, precompute_rope_cache};

    assert!(
        !model_path.ends_with(".gguf"),
        "--cuda-graph-engine only supports SafeTensors directories"
    );

    // -- Load config --
    let config_path = Path::new(model_path).join("config.json");
    let config_data = std::fs::read_to_string(&config_path).map_err(|e| {
        infernum::Error::Io(std::io::Error::new(
            e.kind(),
            format!("Failed to read {}: {e}", config_path.display()),
        ))
    })?;
    let config: LlamaConfig = serde_json::from_str(&config_data).map_err(|e| {
        infernum::Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to parse config.json: {e}"),
        ))
    })?;

    let num_layers = config.num_hidden_layers;
    let hidden_size = config.hidden_size;
    let head_dim = config.head_dim();
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads();
    let eps = config.rms_norm_eps;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let half_dim = head_dim / 2;
    let prompt_len: usize = 8;
    let max_seq_len = prompt_len + n_gen;

    eprintln!(
        "Model: {} layers, {} hidden (CudaDecodeEngine, n_gen={n_gen})",
        num_layers, hidden_size,
    );

    // -- Build indirect decode graph --
    let (graph, model_weight_ids, _extra_ids) =
        build_indirect_decode_graph::<CudaBackend>(&config, max_seq_len, weight_dtype);

    // -- Populate WeightStore --
    // Phase A: model weights (from SafeTensors files).
    // Use a temporary 1-token prefill graph to load weights — it has the same
    // model weight names and ordering as the indirect graph but no RoPE/KV extras,
    // so load_graph_weights won't try to look up "rope.cos_cache" on disk.
    let mut weights = WeightStore::<CudaTensor, LinearWeight>::new();
    eprintln!("Loading weights...");
    {
        let (tmp_graph, _) = build_prefill_graph::<CudaBackend>(&config, 1, weight_dtype);
        load_graph_weights(ctx, &tmp_graph, model_path, &mut weights)?;
    }

    // Phase B: RoPE cos/sin caches (precomputed on GPU) — pushed in the
    // same order as the indirect graph registered them (immediately after model weights).
    let (cos_cache_tensor, sin_cache_tensor) =
        precompute_rope_cache(ctx, max_seq_len, head_dim, config.rope_theta)?;
    weights.push_tensor_weight(cos_cache_tensor.clone());
    weights.push_tensor_weight(sin_cache_tensor.clone());

    // Phase C: KV cache buffers (pre-allocated on GPU).
    // Push one K and one V buffer per layer (interleaved per the graph's registration order).
    let mut kv_cache = KvCache::new(
        ctx,
        num_layers,
        max_seq_len,
        num_kv_heads,
        head_dim,
        weight_dtype,
    )?;
    kv_cache.set_graph_max_seq_len(max_seq_len);

    for layer_idx in 0..num_layers {
        let (k_buf, v_buf) = kv_cache.full_buffers(layer_idx);
        weights.push_tensor_weight(k_buf.clone());
        weights.push_tensor_weight(v_buf.clone());
    }

    // -- Sequence position counter (GPU-resident pointer) --
    let mut seq_pos = SeqPosition::new(ctx.device())?;
    seq_pos.set(prompt_len, ctx.device())?;

    // -- Phase 1: Eager warmup (prompt ingestion, not timed) --
    eprintln!("Warm-up: decoding {prompt_len} prompt tokens (eager)...");
    let (all_cos_host, all_sin_host) =
        infernum::rope::precompute_rope_data(max_seq_len, head_dim, config.rope_theta);

    let mut last_token = 0u32;
    for pos in 0..prompt_len {
        let token = (pos % 256) as u32;
        let mut hidden = infernum_cuda::cuda::ops::embedding_gather(
            ctx,
            weights.tensor_weight(model_weight_ids.embed_tokens),
            &[token],
        )?;

        for layer_idx in 0..num_layers {
            let ids = &model_weight_ids.layers[layer_idx];

            let normed = rms_norm(&hidden, weights.tensor_weight(ids.input_layernorm), eps)?;
            let q_flat = linear(&normed, weights.linear_weight(ids.q_proj))?;
            let k_flat = linear(&normed, weights.linear_weight(ids.k_proj))?;
            let v_flat = linear(&normed, weights.linear_weight(ids.v_proj))?;

            let q = q_flat.reshape(&[1, num_heads, head_dim]);
            let k = k_flat.reshape(&[1, num_kv_heads, head_dim]);
            let v = v_flat.reshape(&[1, num_kv_heads, head_dim]);

            let offset = pos * half_dim;
            let cos_slice = CudaTensor::from_slice(
                ctx,
                &[1, half_dim],
                &all_cos_host[offset..offset + half_dim],
            )?;
            let sin_slice = CudaTensor::from_slice(
                ctx,
                &[1, half_dim],
                &all_sin_host[offset..offset + half_dim],
            )?;
            let q_rot = apply_rope(&q, &cos_slice, &sin_slice, 0)?;
            let k_rot = apply_rope(&k, &cos_slice, &sin_slice, 0)?;

            kv_cache.append(layer_idx, &k_rot, &v)?;

            let total = kv_cache.current_len() + 1;
            let (k_full, v_full) = kv_cache.get_up_to(layer_idx, total);
            let attn = fused_attention_decode(
                &q_rot,
                &k_full,
                &v_full,
                Some(scale),
                None,
                config.effective_sliding_window(layer_idx),
            )?;
            let attn_flat = attn.reshape(&[1, hidden_size]);
            let attn_proj = linear(&attn_flat, weights.linear_weight(ids.o_proj))?;
            add_inplace(&mut hidden, &attn_proj)?;

            let ffn_normed = rms_norm(
                &hidden,
                weights.tensor_weight(ids.post_attention_layernorm),
                eps,
            )?;
            let gate = linear(&ffn_normed, weights.linear_weight(ids.gate_proj))?;
            let up = linear(&ffn_normed, weights.linear_weight(ids.up_proj))?;
            let activated = swiglu(&gate, &up)?;
            let ffn_out = linear(&activated, weights.linear_weight(ids.down_proj))?;
            add_inplace(&mut hidden, &ffn_out)?;
        }

        let final_normed = rms_norm(
            &hidden,
            weights.tensor_weight(model_weight_ids.final_norm),
            eps,
        )?;
        let logits = linear(
            &final_normed,
            weights.linear_weight(model_weight_ids.lm_head),
        )?;
        last_token = argmax_last_scalar(&logits)?;

        kv_cache.advance(1)?;
    }
    ctx.synchronize()?;
    eprintln!(
        "Warm-up done. kv_cache.current_len() = {}",
        kv_cache.current_len()
    );

    // -- Build CudaDecodeEngine --
    let mut engine = CudaDecodeEngine::new(ctx.clone(), graph, weights, kv_cache, seq_pos)?;

    // -- Phase 2: Timed decode via CudaDecodeEngine --
    ctx.synchronize()?;
    let start = Instant::now();

    for _step in 0..n_gen {
        let logits = engine.step(last_token)?;
        engine.advance()?;
        last_token = argmax_last_scalar(&logits)?;
    }

    ctx.synchronize()?;
    let elapsed = start.elapsed();
    let tok_s = n_gen as f64 / elapsed.as_secs_f64();

    eprintln!(
        "Graph stabilized after: {} steps",
        if engine.is_stabilized() {
            "~few"
        } else {
            "never"
        },
    );
    println!(
        "{n_gen} tokens in {:.2}s = {:.1} tok/s",
        elapsed.as_secs_f64(),
        tok_s,
    );

    Ok(())
}

fn main() -> infernum::Result<()> {
    let cli = Cli::parse();
    let ctx = CudaContext::new(0)?;

    if cli.graph || cli.graph_decode || cli.cuda_graphs || cli.cuda_graph_engine {
        let weight_dtype = match cli.dtype.as_str() {
            "f32" => DType::F32,
            "bf16" => DType::BF16,
            other => {
                eprintln!("ERROR: unsupported --dtype value: {other} (expected f32 or bf16)");
                std::process::exit(1);
            }
        };

        let is_gguf = cli.model.ends_with(".gguf");

        if cli.cuda_graphs && is_gguf {
            eprintln!("ERROR: --cuda-graphs does not support GGUF files");
            std::process::exit(1);
        }
        if cli.cuda_graph_engine && is_gguf {
            eprintln!("ERROR: --cuda-graph-engine does not support GGUF files");
            std::process::exit(1);
        }

        // GGUF tensor weights may be quantized; use bf16 compute dtype
        // for faster attention/elementwise ops. quantized_matmul handles
        // BF16 activations internally (cast to F16 for cuBLAS).
        let weight_dtype = if is_gguf { DType::BF16 } else { weight_dtype };

        let model_type = if is_gguf {
            "llama".to_string() // GGUF graph mode only supports Llama/Mistral
        } else {
            let mt = detect_model_type(&cli.model)?;
            match mt.as_str() {
                "llama" | "mistral" => {}
                other => {
                    eprintln!(
                        "ERROR: --graph/--graph-decode/--cuda-graphs/--cuda-graph-engine mode only supports Llama/Mistral, got: {other}"
                    );
                    std::process::exit(1);
                }
            }
            mt
        };

        let format_name = if is_gguf { "GGUF" } else { "SafeTensors" };
        let mode = if cli.cuda_graph_engine {
            "cuda-graph-engine"
        } else if cli.cuda_graphs {
            "cuda-graphs"
        } else if cli.graph_decode {
            "graph-decode"
        } else {
            "graph"
        };
        eprintln!(
            "Loading: {} (GPU, {format_name}, {model_type}, {mode}, dtype={weight_dtype})",
            cli.model,
        );

        let result = if cli.cuda_graph_engine {
            bench_cuda_graph_engine(&ctx, &cli.model, cli.n_gen, weight_dtype)
        } else if cli.cuda_graphs {
            bench_cuda_graphs_decode(&ctx, &cli.model, cli.n_gen, weight_dtype)
        } else if cli.graph_decode {
            bench_graph_decode(&ctx, &cli.model, cli.n_gen, weight_dtype)
        } else {
            bench_graph(&ctx, &cli.model, cli.n_gen, weight_dtype)
        };

        if let Some(pool) = ctx.buffer_pool() {
            eprintln!(
                "Pool stats: {} hits, {} misses, {:.1} MB cached",
                pool.hits(),
                pool.misses(),
                pool.free_bytes() as f64 / (1024.0 * 1024.0),
            );
        }

        return result;
    }

    let is_gguf = cli.model.ends_with(".gguf");
    let model_type = if is_gguf {
        "llama".to_string()
    } else {
        detect_model_type(&cli.model)?
    };
    let family = match model_type.as_str() {
        "llama" | "mistral" | "mixtral" => "llama",
        "qwen2" | "qwen3" | "qwen3_moe" => "qwen",
        "deepseek_v3" => "deepseek",
        "gemma2" | "gemma3_text" => "gemma",
        other => panic!("Unsupported model_type: {other}"),
    };

    let result = match family {
        "llama" => {
            let model = if is_gguf {
                LlamaModel::<CudaBackend>::from_gguf(&ctx, &cli.model)?
            } else {
                LlamaModel::<CudaBackend>::from_pretrained(&ctx, &cli.model)?
            };
            let dtype = format!("{}", model.dtype());
            let (nl, hs) = (model.config().num_hidden_layers, model.config().hidden_size);
            bench_with_info(model, nl, hs, &dtype, cli.n_gen)
        }
        "qwen" => {
            let model = QwenModel::<CudaBackend>::from_pretrained(&ctx, &cli.model)?;
            let dtype = format!("{}", model.dtype());
            let (nl, hs) = (model.config().num_hidden_layers, model.config().hidden_size);
            bench_with_info(model, nl, hs, &dtype, cli.n_gen)
        }
        "deepseek" => {
            let model = DeepSeekModel::<CudaBackend>::from_pretrained(&ctx, &cli.model)?;
            let dtype = format!("{}", model.dtype());
            let (nl, hs) = (model.config().num_hidden_layers, model.config().hidden_size);
            bench_with_info(model, nl, hs, &dtype, cli.n_gen)
        }
        "gemma" => {
            let model = GemmaModel::<CudaBackend>::from_pretrained(&ctx, &cli.model)?;
            let dtype = format!("{}", model.dtype());
            let (nl, hs) = (model.config().num_hidden_layers, model.config().hidden_size);
            bench_with_info(model, nl, hs, &dtype, cli.n_gen)
        }
        other => panic!("Unsupported family: {other}"),
    };

    if let Some(pool) = ctx.buffer_pool() {
        eprintln!(
            "Pool stats: {} hits, {} misses, {:.1} MB cached",
            pool.hits(),
            pool.misses(),
            pool.free_bytes() as f64 / (1024.0 * 1024.0),
        );
    }

    result
}
