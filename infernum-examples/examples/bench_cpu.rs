//! Benchmark for infernum CPU throughput.
//!
//! Three modes:
//! - **Eager (default):** Autoregressive decode — generates N tokens one-by-one
//!   using the runtime engine with KV cache.
//! - **Graph prefill (`--graph`):** Prefill throughput — processes N tokens in a
//!   single forward pass through the graph executor. Only supports Llama family.
//! - **Graph decode (`--graph-decode`):** Decode throughput — generates N tokens
//!   one-by-one through the graph executor with KV cache management. Only supports
//!   Llama family.
//!
//! Usage:
//!   cargo run --release --example bench_cpu --features cpu -- /path/to/model.gguf 128
//!   cargo run --release --example bench_cpu --features cpu -- /path/to/safetensors_dir 128
//!   cargo run --release --example bench_cpu --features cpu -- --graph /path/to/safetensors_dir 128
//!   cargo run --release --example bench_cpu --features cpu -- --graph-decode /path/to/model 128

use std::path::Path;
use std::time::Instant;

use clap::Parser;
use serde::Deserialize;

use infernum::backend::MatmulOps;
use infernum::dtype::DType;
use infernum::graph::GraphNode;
use infernum::graph::{plan, Arena, WeightId, WeightStore};
use infernum::Tensor;
use infernum::{GenerateOptions, NodeId};
use infernum_cpu::executor::{execute, KvCacheStore};
use infernum_cpu::{CpuBackend, CpuLinearWeight, CpuSafeTensorsLoader, CpuTensor};
use infernum_gemma::GemmaModel;
use infernum_llama::{
    build_decode_graph, build_prefill_graph, LlamaConfig, LlamaGraphEngine, LlamaModel,
};
use infernum_qwen::QwenModel;
use infernum_runtime::Engine;

#[derive(Parser)]
#[command(name = "bench_cpu")]
struct Cli {
    /// Path to model directory (SafeTensors) or .gguf file
    model: String,

    /// Number of tokens to generate (eager) or process (graph)
    #[arg(default_value_t = 128)]
    n_gen: usize,

    /// Number of threads (default: all cores)
    #[arg(short = 'j', long)]
    threads: Option<usize>,

    /// Use graph executor instead of eager engine (SafeTensors + Llama only)
    #[arg(long)]
    graph: bool,

    /// Use graph executor for decode (autoregressive) throughput
    #[arg(long)]
    graph_decode: bool,

    /// Use LlamaGraphEngine for decode throughput (SafeTensors or GGUF)
    #[arg(long)]
    graph_engine: bool,
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

/// Detect architecture from GGUF metadata.
fn detect_gguf_arch(path: &str) -> infernum::Result<String> {
    let loader = infernum::weights::gguf::GgufLoader::from_file(path)?;
    let arch = loader
        .metadata()
        .get("general.architecture")
        .and_then(infernum::GgufValue::as_str)
        .unwrap_or("llama")
        .to_string();
    Ok(arch)
}

fn bench_model<M: infernum::Model<B = CpuBackend> + Send + 'static>(
    model: M,
    n_gen: usize,
) -> infernum::Result<()> {
    // 8-token prompt (same as GPU bench for comparability)
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

/// Load graph weights from a SafeTensors directory into the `WeightStore`.
fn load_graph_weights_safetensors(
    graph: &infernum::graph::Graph<CpuBackend>,
    model_dir: &str,
    weights: &mut WeightStore<CpuTensor, CpuLinearWeight>,
) -> infernum::Result<()> {
    use infernum::weights::WeightLoader;

    let loader = CpuSafeTensorsLoader::new(Path::new(model_dir))?;

    for i in 0..graph.tensor_weight_count() {
        let meta = graph.tensor_weight_meta(WeightId::from_index(i as u32));
        let tensor = loader.load_tensor(&meta.name, DType::F32)?;
        weights.push_tensor_weight(tensor);
    }

    for i in 0..graph.linear_weight_count() {
        let meta = graph.linear_weight_meta(WeightId::from_index(i as u32));
        if loader.contains(&meta.name) {
            let linear = loader.load_linear(&meta.name, DType::F32, None)?;
            weights.push_linear_weight(linear);
        } else if meta.name == "lm_head.weight" {
            // Tied embeddings: lm_head shares embed_tokens weight.
            let linear = loader.load_linear("model.embed_tokens.weight", DType::F32, None)?;
            weights.push_linear_weight(linear);
        } else {
            return Err(infernum::Error::WeightNotFound(meta.name.clone()));
        }
    }

    Ok(())
}

/// Map a SafeTensors-convention weight name to a GGUF tensor name.
fn safetensors_to_gguf_name(name: &str) -> String {
    // Special top-level tensors
    match name {
        "model.embed_tokens.weight" => return "token_embd.weight".to_string(),
        "model.norm.weight" => return "output_norm.weight".to_string(),
        "lm_head.weight" => return "output.weight".to_string(),
        _ => {}
    }
    // Per-layer: model.layers.{N}.xxx → blk.{N}.yyy
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

/// Returns true if this weight name is a Q or K projection that needs unpermuting.
fn needs_unpermute(name: &str) -> bool {
    name.ends_with(".attn_q.weight") || name.ends_with(".attn_k.weight")
}

/// Load graph weights from a GGUF file into the `WeightStore`.
///
/// Loads weights in their native format (Q8_0, Q4_0, F32, etc.) using the
/// same quantized kernels as the eager path.
fn load_graph_weights_gguf(
    graph: &infernum::graph::Graph<CpuBackend>,
    config: &LlamaConfig,
    gguf_path: &str,
    weights: &mut WeightStore<CpuTensor, CpuLinearWeight>,
) -> infernum::Result<()> {
    use infernum::weights::format::{host_transpose_2d, host_unpermute_f32, FormatLoader};
    use infernum::weights::host::HostLinearWeight;

    let loader = infernum::weights::gguf::GgufLoader::from_file(gguf_path)?;

    // Tensor weights (embeddings, layernorms) — always loaded as f32
    for i in 0..graph.tensor_weight_count() {
        let meta = graph.tensor_weight_meta(WeightId::from_index(i as u32));
        let gguf_name = safetensors_to_gguf_name(&meta.name);
        let host = loader.load_f32(&gguf_name)?;
        weights.push_tensor_weight(CpuTensor::from_f32(&host.shape, host.as_f32_slice()));
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

        let dtype = FormatLoader::get_dtype(&loader, &actual_name)?;

        let host_linear = if dtype.is_quantized() {
            // Load as quantized — preserves Q8_0/Q4_0 blocks for fast kernels
            if needs_unpermute(&gguf_name) {
                let n_head = if gguf_name.contains("attn_q") {
                    config.num_attention_heads
                } else {
                    config
                        .num_key_value_heads
                        .unwrap_or(config.num_attention_heads)
                };
                HostLinearWeight::Quantized(FormatLoader::load_quantized_unpermute(
                    &loader,
                    &actual_name,
                    n_head,
                )?)
            } else {
                HostLinearWeight::Quantized(FormatLoader::load_quantized(&loader, &actual_name)?)
            }
        } else {
            // Dense: load as f32, unpermute Q/K if needed, transpose
            let host = loader.load_f32(&actual_name)?;
            let host = if needs_unpermute(&gguf_name) {
                let n_head = if gguf_name.contains("attn_q") {
                    config.num_attention_heads
                } else {
                    config
                        .num_key_value_heads
                        .unwrap_or(config.num_attention_heads)
                };
                host_unpermute_f32(&host, n_head)?
            } else {
                host
            };
            HostLinearWeight::Dense(host_transpose_2d(&host)?)
        };

        let linear = CpuBackend::upload_host_linear(&(), &host_linear)?;
        weights.push_linear_weight(linear);
    }

    Ok(())
}

/// Benchmark prefill throughput using the graph executor.
///
/// Builds a Llama prefill graph for `seq_len` tokens, loads real weights from
/// SafeTensors or GGUF, runs the graph once to warm up, then measures multiple
/// iterations.
fn bench_graph(model_path: &str, n_tokens: usize) -> infernum::Result<()> {
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
        "Model: {} layers, {} hidden (graph prefill, seq_len={})",
        config.num_hidden_layers, config.hidden_size, n_tokens,
    );

    // Build graph
    let (mut graph, _model_weights) =
        build_prefill_graph::<CpuBackend>(&config, n_tokens, DType::F32);
    infernum::graph::optimizer::optimize(&mut graph);
    let exec_plan = plan(&graph);

    let mut weights = WeightStore::<CpuTensor, CpuLinearWeight>::new();
    eprintln!("Loading weights...");

    if is_gguf {
        load_graph_weights_gguf(&graph, &config, model_path, &mut weights)?;
    } else {
        load_graph_weights_safetensors(&graph, model_path, &mut weights)?;
    }

    // Build inputs: token_ids, cos_cache, sin_cache
    let token_ids: Vec<u32> = (0..n_tokens).map(|i| (i % 256) as u32).collect();
    let input_ids = CpuTensor::from_u32(&[n_tokens], &token_ids);

    let (cos_data, sin_data) =
        infernum::rope::precompute_rope_data(n_tokens, head_dim, config.rope_theta);
    let cos_cache = CpuTensor::from_f32(&[n_tokens, half_dim], &cos_data);
    let sin_cache = CpuTensor::from_f32(&[n_tokens, half_dim], &sin_data);

    let inputs = vec![input_ids, cos_cache, sin_cache];

    // Warm-up run
    {
        let mut arena = Arena::new(exec_plan.arena_size);
        let outputs = execute(
            &exec_plan,
            graph.nodes(),
            &mut arena,
            &weights,
            &inputs,
            graph.output_ids(),
            None,
        )?;
        assert_eq!(outputs[0].shape()[0], n_tokens);
        eprintln!("Warm-up done, output shape: {:?}", outputs[0].shape());
    }

    // Benchmark: run 3 iterations and take the best
    let n_iters = 3;
    let mut best_elapsed = std::time::Duration::MAX;

    for iter in 0..n_iters {
        let mut arena = Arena::new(exec_plan.arena_size);
        let start = Instant::now();
        let _outputs = execute(
            &exec_plan,
            graph.nodes(),
            &mut arena,
            &weights,
            &inputs,
            graph.output_ids(),
            None,
        )?;
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

/// Find KV cache Input and ConcatSeq node IDs from a decode graph.
///
/// Returns `(cache_input_ids, concat_ids)` where each vec contains entries
/// in order `[k_layer0, v_layer0, k_layer1, v_layer1, ...]`.
fn find_kv_cache_node_ids(
    nodes: &[GraphNode<CpuBackend>],
    num_layers: usize,
) -> (Vec<NodeId>, Vec<NodeId>) {
    // KV cache inputs are Input nodes after the first 3 (token, cos, sin).
    let input_ids: Vec<NodeId> = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.op.name() == "input")
        .skip(3) // skip token_id, cos, sin
        .map(|(i, _)| NodeId::from_index(i as u32))
        .collect();
    assert_eq!(input_ids.len(), 2 * num_layers, "unexpected KV input count");

    // ConcatSeq nodes (in graph order) correspond to KV cache appends.
    let concat_ids: Vec<NodeId> = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.op.name() == "concat_seq")
        .map(|(i, _)| NodeId::from_index(i as u32))
        .collect();
    assert_eq!(
        concat_ids.len(),
        2 * num_layers,
        "unexpected ConcatSeq count"
    );

    (input_ids, concat_ids)
}

/// Benchmark decode throughput using the graph executor with persistent KV caches.
///
/// Uses `KvCacheStore` to avoid copying KV caches through the arena every step.
/// The graph is still rebuilt each step (node shapes depend on kv_len), but the
/// expensive O(seq_len) KV cache copies are eliminated.
fn bench_graph_decode(model_path: &str, n_gen: usize) -> infernum::Result<()> {
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
        "Model: {} layers, {} hidden (graph decode, n_gen={})",
        num_layers, config.hidden_size, n_gen,
    );

    // Load weights using a temporary graph (weight order is consistent across graphs)
    let prompt_len = 8;
    let mut weights = WeightStore::<CpuTensor, CpuLinearWeight>::new();
    eprintln!("Loading weights...");
    {
        let (tmp_graph, _) = build_prefill_graph::<CpuBackend>(&config, 1, DType::F32);
        if is_gguf {
            load_graph_weights_gguf(&tmp_graph, &config, model_path, &mut weights)?;
        } else {
            load_graph_weights_safetensors(&tmp_graph, model_path, &mut weights)?;
        }
    }

    // Pre-compute RoPE for all positions we'll need
    let max_pos = prompt_len + n_gen;
    let (all_cos, all_sin) =
        infernum::rope::precompute_rope_data(max_pos, head_dim, config.rope_theta);

    // Build decode graph once with kv_len=0. KvCacheStore intercepts all KV
    // cache Input and ConcatSeq nodes, so the graph never touches KV data
    // through the arena. The graph/plan/arena can be reused across all steps.
    let (mut decode_graph, _) = build_decode_graph::<CpuBackend>(&config, 0, DType::F32);
    infernum::graph::optimizer::optimize(&mut decode_graph);
    let decode_plan = plan(&decode_graph);
    let mut arena = Arena::new(decode_plan.arena_size);
    let logits_id = decode_graph.output_ids()[0];

    let (cache_input_ids, concat_ids) = find_kv_cache_node_ids(decode_graph.nodes(), num_layers);

    // Create persistent KV cache store, pre-allocated to avoid reallocations.
    let mut kv_cache = KvCacheStore::new(
        num_layers,
        num_kv_heads,
        head_dim,
        config.max_position_embeddings,
        cache_input_ids,
        concat_ids,
    );

    // Helper: run one decode step, reusing the cached graph/plan/arena.
    let run_decode_step = |pos: usize,
                           token: u32,
                           kv_cache: &mut KvCacheStore,
                           arena: &mut Arena|
     -> infernum::Result<Vec<CpuTensor>> {
        let cos_start = pos * half_dim;
        let inputs = vec![
            CpuTensor::from_u32(&[1], &[token]),
            CpuTensor::from_f32(&[1, half_dim], &all_cos[cos_start..cos_start + half_dim]),
            CpuTensor::from_f32(&[1, half_dim], &all_sin[cos_start..cos_start + half_dim]),
        ];

        execute(
            &decode_plan,
            decode_graph.nodes(),
            arena,
            &weights,
            &inputs,
            &[logits_id],
            Some(kv_cache),
        )
    };

    // Warm-up: decode the prompt tokens one at a time (not timed)
    eprintln!("Warm-up: decoding {prompt_len} prompt tokens...");
    let mut last_token = 0u32;
    for pos in 0..prompt_len {
        let token = (pos % 256) as u32;
        let _outputs = run_decode_step(pos, token, &mut kv_cache, &mut arena)?;
        last_token = token;
    }
    eprintln!("Warm-up done.");

    // --- Phase 2: Timed decode of n_gen tokens ---
    let start = Instant::now();

    for step in 0..n_gen {
        let pos = prompt_len + step;
        let outputs = run_decode_step(pos, last_token, &mut kv_cache, &mut arena)?;

        // Argmax the logits for next token
        let logits = outputs[0].as_f32_slice();
        last_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    let elapsed = start.elapsed();
    let tok_s = n_gen as f64 / elapsed.as_secs_f64();

    println!(
        "{n_gen} tokens in {:.2}s = {:.1} tok/s",
        elapsed.as_secs_f64(),
        tok_s,
    );

    Ok(())
}

/// Benchmark decode throughput using [`LlamaGraphEngine`].
///
/// Loads the model via `from_pretrained` (SafeTensors) or `from_gguf` (GGUF),
/// then measures autoregressive decode of `n_gen` tokens from a single BOS
/// token prompt and reports tok/s. The single-token prompt makes prompt
/// processing overhead negligible so the result is directly comparable to
/// `bench_graph_decode`.
fn bench_graph_engine(model_path: &str, n_gen: usize) -> infernum::Result<()> {
    let is_gguf = model_path.ends_with(".gguf");

    eprintln!(
        "Loading: {model_path} (CPU, {}, llama, graph-engine)",
        if is_gguf { "GGUF" } else { "SafeTensors" }
    );

    let engine = if is_gguf {
        LlamaGraphEngine::from_gguf(std::path::Path::new(model_path))?
    } else {
        LlamaGraphEngine::from_pretrained(std::path::Path::new(model_path))?
    };

    let config = engine.config();
    eprintln!(
        "Model: {} layers, {} hidden (graph-engine, n_gen={})",
        config.num_hidden_layers, config.hidden_size, n_gen,
    );

    // Use u32::MAX as a "never stop" EOS sentinel so the bench always runs
    // exactly n_gen decode steps, matching bench_graph_decode's behaviour.
    // Instruction-tuned models generate real EOS quickly on bare prompts,
    // which would make the denominator dishonest.
    let no_eos = u32::MAX;

    // Single BOS token prompt so prompt-processing overhead is negligible
    // (1 step) and the timed output is directly comparable to bench_graph_decode,
    // which times only the pure decode phase.
    let prompt: Vec<u32> = vec![1];

    // No warm-up: a warm-up run would pre-populate the PlanCache for every
    // decode step (each kv_len produces a distinct graph hash), making the
    // timed runs trivially fast (pure cache hits). All 3 timed runs include
    // the same cold-cache JIT cost, so best-of-3 still captures steady-state.
    let mut best_elapsed = std::time::Duration::MAX;
    for iter in 0..3 {
        let start = Instant::now();
        engine.generate(&prompt, n_gen, no_eos)?;
        let elapsed = start.elapsed();
        if elapsed < best_elapsed {
            best_elapsed = elapsed;
        }
        eprintln!(
            "  iter {}: {:.2}s ({:.1} tok/s)",
            iter + 1,
            elapsed.as_secs_f64(),
            n_gen as f64 / elapsed.as_secs_f64(),
        );
    }

    let tok_s = n_gen as f64 / best_elapsed.as_secs_f64();
    println!(
        "{n_gen} tokens in {:.2}s = {:.1} tok/s",
        best_elapsed.as_secs_f64(),
        tok_s,
    );

    Ok(())
}

fn main() -> infernum::Result<()> {
    let cli = Cli::parse();

    if let Some(threads) = cli.threads {
        // Set thread count for rayon / internal parallelism if supported
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
    }

    if cli.graph_engine {
        return bench_graph_engine(&cli.model, cli.n_gen);
    }

    if cli.graph || cli.graph_decode {
        let is_gguf = cli.model.ends_with(".gguf");

        let model_type = if is_gguf {
            detect_gguf_arch(&cli.model)?
        } else {
            detect_model_type(&cli.model)?
        };

        match model_type.as_str() {
            "llama" | "mistral" => {}
            other => {
                eprintln!(
                    "ERROR: --graph/--graph-decode mode only supports Llama/Mistral, got: {other}"
                );
                std::process::exit(1);
            }
        };

        let mode = if cli.graph_decode {
            "graph-decode"
        } else {
            "graph"
        };
        eprintln!(
            "Loading: {} (CPU, {}, {}, {})",
            cli.model,
            if is_gguf { "GGUF" } else { "SafeTensors" },
            model_type,
            mode,
        );

        return if cli.graph_decode {
            bench_graph_decode(&cli.model, cli.n_gen)
        } else {
            bench_graph(&cli.model, cli.n_gen)
        };
    }

    let is_gguf = cli.model.ends_with(".gguf");

    let (family, model_type) = if is_gguf {
        let arch = detect_gguf_arch(&cli.model)?;
        let family = match arch.as_str() {
            "llama" => "llama",
            "qwen2" => "qwen",
            "gemma2" | "gemma3" => "gemma",
            other => panic!("Unsupported GGUF architecture: {other}"),
        };
        (family, arch)
    } else {
        let mt = detect_model_type(&cli.model)?;
        let family = match mt.as_str() {
            "llama" | "mistral" | "mixtral" => "llama",
            "qwen2" | "qwen3" | "qwen3_moe" => "qwen",
            "gemma2" | "gemma3_text" => "gemma",
            other => panic!("Unsupported model_type: {other}"),
        };
        (family, mt)
    };

    eprintln!(
        "Loading: {} (CPU, {}, {})",
        cli.model,
        if is_gguf { "GGUF" } else { "SafeTensors" },
        model_type,
    );

    match family {
        "llama" => {
            let model = if is_gguf {
                LlamaModel::<CpuBackend>::from_gguf(&(), Path::new(&cli.model))?
            } else {
                LlamaModel::<CpuBackend>::from_pretrained(&(), &cli.model)?
            };
            let cfg = model.config();
            eprintln!(
                "Model: {} layers, {} hidden, dtype={}",
                cfg.num_hidden_layers,
                cfg.hidden_size,
                model.dtype(),
            );
            bench_model(model, cli.n_gen)
        }
        "qwen" => {
            let model = if is_gguf {
                QwenModel::<CpuBackend>::from_gguf(&(), Path::new(&cli.model))?
            } else {
                QwenModel::<CpuBackend>::from_pretrained(&(), &cli.model)?
            };
            let cfg = model.config();
            eprintln!(
                "Model: {} layers, {} hidden, dtype={}",
                cfg.num_hidden_layers,
                cfg.hidden_size,
                model.dtype(),
            );
            bench_model(model, cli.n_gen)
        }
        "gemma" => {
            let model = if is_gguf {
                GemmaModel::<CpuBackend>::from_gguf(&(), Path::new(&cli.model))?
            } else {
                GemmaModel::<CpuBackend>::from_pretrained(&(), &cli.model)?
            };
            let cfg = model.config();
            eprintln!(
                "Model: {} layers, {} hidden, dtype={}",
                cfg.num_hidden_layers,
                cfg.hidden_size,
                model.dtype(),
            );
            bench_model(model, cli.n_gen)
        }
        other => panic!("Unsupported family: {other}"),
    }
}
