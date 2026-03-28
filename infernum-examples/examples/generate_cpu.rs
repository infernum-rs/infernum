//! CPU text generation example (multi-architecture, SafeTensors + GGUF)
//!
//! Usage:
//!   cargo run --example generate_cpu --features cpu -- -m /path/to/model "Hello"
//!   cargo run --example generate_cpu --features cpu -- -m model.Q8_0.gguf "Hello"
//!   cargo run --example generate_cpu --features cpu -- -m /path/to/model --greedy "Hello"
//!   cargo run --example generate_cpu --features cpu -- -m /path/to/model -t 0.8 -p 0.95 "Hello"

use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use clap::Parser;

use infernum::tokenizer::{GgufTokenizer, LlamaTokenizer};
use infernum::Tokenizer as _;
use infernum::{GenerateOptions, Result, SamplingParams};
use infernum_cpu::CpuBackend;
use infernum_gemma::GemmaModel;
use infernum_llama::LlamaModel;
use infernum_qwen::QwenModel;
use infernum_runtime::{Engine, Runtime};

/// CPU text generation with LLMs (SafeTensors + GGUF)
///
/// Supports Llama, Mistral, Mixtral, Qwen, and Gemma model families.
/// SafeTensors models load as f32 (Llama family only).
/// GGUF models load quantized weights (Q8_0, Q4_0) with SIMD-accelerated
/// dequantize-on-the-fly inference. Architecture is auto-detected from the
/// GGUF metadata.
/// Uses KV cache and nucleus sampling by default.
#[derive(Parser)]
#[command(name = "generate_cpu")]
struct Cli {
    /// Path to model directory (SafeTensors) or .gguf file
    #[arg(short, long, default_value = "models/smollm2-360m")]
    model: String,

    /// Text prompt
    #[arg(default_value = "Hello")]
    prompt: String,

    /// Maximum tokens to generate
    #[arg(short = 'n', long, default_value_t = 100)]
    max_tokens: usize,

    /// Use greedy (argmax) decoding instead of sampling
    #[arg(long)]
    greedy: bool,

    /// Disable KV cache (recompute full sequence each step)
    #[arg(long)]
    no_kv_cache: bool,

    /// Sampling temperature
    #[arg(short, long, default_value_t = 0.7)]
    temperature: f32,

    /// Nucleus sampling threshold
    #[arg(short = 'p', long, default_value_t = 0.9)]
    top_p: f32,

    /// RNG seed for sampling
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Repetition penalty factor (1.0 = disabled, >1.0 penalises repeated tokens)
    #[arg(short = 'r', long, default_value_t = 1.1)]
    repetition_penalty: f32,

    /// Number of recent tokens to consider for repetition penalty
    #[arg(long, default_value_t = 64)]
    repetition_penalty_window: usize,

    /// Maximum KV cache sequence length (default: min(model max, 4096))
    #[arg(long)]
    max_seq_len: Option<usize>,

    /// Use graph executor instead of eager execution (Llama family only).
    /// Runs full-sequence prefill at each step (no KV cache, O(n²) compute).
    #[arg(long)]
    graph: bool,
}

/// Abstraction over tokenizer backends so we can use either one.
enum Tokenizer {
    HuggingFace(LlamaTokenizer),
    Gguf(GgufTokenizer),
}

impl infernum::Tokenizer for Tokenizer {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        match self {
            Self::HuggingFace(t) => t.encode(text, add_bos),
            Self::Gguf(t) => t.encode(text, add_bos),
        }
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        match self {
            Self::HuggingFace(t) => t.decode(ids),
            Self::Gguf(t) => t.decode(ids),
        }
    }

    fn decode_token(&self, id: u32) -> Result<String> {
        match self {
            Self::HuggingFace(t) => t.decode_token(id),
            Self::Gguf(t) => t.decode_token(id),
        }
    }

    fn eos_token_id(&self) -> u32 {
        match self {
            Self::HuggingFace(t) => t.eos_token_id(),
            Self::Gguf(t) => t.eos_token_id(),
        }
    }
}

impl Tokenizer {
    fn vocab_size(&self) -> usize {
        match self {
            Self::HuggingFace(t) => t.vocab_size(),
            Self::Gguf(t) => t.vocab_size(),
        }
    }
}

/// Run generation with a loaded model and tokenizer.
fn run_model<M>(model: M, tokenizer: Tokenizer, cli: &Cli) -> Result<()>
where
    M: infernum::Model<B = CpuBackend> + Send + 'static,
{
    let model_config = model.config();
    println!(
        "Model loaded ({} layers, vocab {})",
        model_config.num_layers,
        tokenizer.vocab_size(),
    );

    let options = GenerateOptions {
        max_new_tokens: cli.max_tokens,
        eos_token_id: None,
        sampling: if cli.greedy {
            None
        } else {
            Some(SamplingParams {
                temperature: cli.temperature,
                top_p: cli.top_p,
                seed: cli.seed,
                repetition_penalty: cli.repetition_penalty,
                repetition_penalty_window: cli.repetition_penalty_window,
            })
        },
        use_kv_cache: !cli.no_kv_cache,
    };

    if let Some(ref params) = options.sampling {
        println!(
            "Sampling: temperature={}, top_p={}, seed={}, repetition_penalty={} (window={})",
            params.temperature,
            params.top_p,
            params.seed,
            params.repetition_penalty,
            params.repetition_penalty_window,
        );
    } else {
        println!("Decoding: greedy (argmax)");
    }
    println!(
        "KV cache: {}",
        if options.use_kv_cache {
            "enabled"
        } else {
            "disabled"
        }
    );

    print!("{}", cli.prompt);
    io::stdout().flush()?;

    let start = Instant::now();

    let (output_tokens, prompt_len) = if !options.use_kv_cache {
        let engine = Engine::new(model)?;
        let input_ids = tokenizer.encode(&cli.prompt, true)?;
        let prompt_len = input_ids.len();

        let mut opts = options.clone();
        opts.eos_token_id = Some(tokenizer.eos_token_id());

        let tokens = engine.generate(&input_ids, &opts)?;

        for &tok in &tokens[prompt_len..] {
            let text = tokenizer.decode_token(tok)?;
            print!("{text}");
            io::stdout().flush()?;
        }

        (tokens, prompt_len)
    } else {
        let runtime = Runtime::with_max_seq_len(model, tokenizer, cli.max_seq_len)?;
        let prompt_len = runtime.tokenizer().encode(&cli.prompt, true)?.len();
        let tokens = runtime.generate_stream(&cli.prompt, &options)?;
        (tokens, prompt_len)
    };

    let elapsed = start.elapsed();
    let generated = output_tokens.len() - prompt_len;

    println!();
    println!(
        "Generated {} tokens in {:.2}s ({:.1} tokens/sec)",
        generated,
        elapsed.as_secs_f64(),
        generated as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}

/// Detect architecture from GGUF metadata.
fn detect_gguf_arch(path: &str) -> Result<String> {
    let loader = infernum::weights::gguf::GgufLoader::from_file(path)?;
    let arch = loader
        .metadata()
        .get("general.architecture")
        .and_then(infernum::GgufValue::as_str)
        .unwrap_or("llama")
        .to_string();
    Ok(arch)
}

// ---------------------------------------------------------------------------
// Graph-based generation
// ---------------------------------------------------------------------------

use serde::Deserialize;

use infernum::backend::MatmulOps;
use infernum::dtype::DType;
use infernum::graph::{plan, Arena, WeightId, WeightStore};
use infernum_cpu::executor::execute;
use infernum_cpu::{CpuLinearWeight, CpuTensor};
use infernum_llama::{build_prefill_graph, LlamaConfig};

/// Peek at just the `model_type` field from config.json.
#[derive(Deserialize)]
struct ModelTypeProbe {
    #[serde(default = "default_model_type")]
    model_type: String,
}

fn default_model_type() -> String {
    "llama".to_string()
}

fn detect_model_type(model_path: &str) -> Result<String> {
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

/// Map SafeTensors weight names to GGUF names.
fn safetensors_to_gguf_name(name: &str) -> String {
    //    Global weights
    match name {
        "model.embed_tokens.weight" => return "token_embd.weight".to_string(),
        "model.norm.weight" => return "output_norm.weight".to_string(),
        "lm_head.weight" => return "output.weight".to_string(),
        _ => {}
    }

    // Per-layer: model.layers.N.<suffix> → blk.N.<gguf_suffix>
    if let Some(rest) = name.strip_prefix("model.layers.") {
        if let Some(dot) = rest.find('.') {
            let layer_num = &rest[..dot];
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
                other => other,
            };
            return format!("blk.{layer_num}.{gguf_suffix}");
        }
    }

    name.to_string()
}

/// Whether a GGUF weight name needs Q/K unpermuting.
fn needs_unpermute(name: &str) -> bool {
    name.ends_with(".attn_q.weight") || name.ends_with(".attn_k.weight")
}

/// Load graph weights from SafeTensors into the `WeightStore`.
fn load_graph_weights_safetensors(
    graph: &infernum::graph::Graph<CpuBackend>,
    model_dir: &str,
    weights: &mut WeightStore<CpuTensor, CpuLinearWeight>,
) -> Result<()> {
    use infernum::weights::WeightLoader;

    let loader = infernum_cpu::CpuSafeTensorsLoader::new(Path::new(model_dir))?;

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
            let linear = loader.load_linear("model.embed_tokens.weight", DType::F32, None)?;
            weights.push_linear_weight(linear);
        } else {
            return Err(infernum::Error::WeightNotFound(meta.name.clone()));
        }
    }

    Ok(())
}

/// Load graph weights from GGUF in native format (quantized or dense).
fn load_graph_weights_gguf(
    graph: &infernum::graph::Graph<CpuBackend>,
    config: &LlamaConfig,
    gguf_path: &str,
    weights: &mut WeightStore<CpuTensor, CpuLinearWeight>,
) -> Result<()> {
    use infernum::weights::format::{host_transpose_2d, host_unpermute_f32, FormatLoader};
    use infernum::weights::host::HostLinearWeight;

    let loader = infernum::weights::gguf::GgufLoader::from_file(gguf_path)?;

    for i in 0..graph.tensor_weight_count() {
        let meta = graph.tensor_weight_meta(WeightId::from_index(i as u32));
        let gguf_name = safetensors_to_gguf_name(&meta.name);
        let host = loader.load_f32(&gguf_name)?;
        weights.push_tensor_weight(CpuTensor::from_f32(&host.shape, host.as_f32_slice()));
    }

    for i in 0..graph.linear_weight_count() {
        let meta = graph.linear_weight_meta(WeightId::from_index(i as u32));
        let gguf_name = safetensors_to_gguf_name(&meta.name);

        let actual_name = if loader.contains(&gguf_name) {
            gguf_name.clone()
        } else if meta.name == "lm_head.weight" {
            "token_embd.weight".to_string()
        } else {
            return Err(infernum::Error::WeightNotFound(gguf_name));
        };

        let dtype = FormatLoader::get_dtype(&loader, &actual_name)?;

        let host_linear = if dtype.is_quantized() {
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

/// Graph-based text generation (greedy, no KV cache).
///
/// Each step runs a full prefill graph over the entire sequence so far,
/// argmaxes the last position, and appends the new token. This is O(n²)
/// but functionally correct and demonstrates the graph executor end-to-end.
#[allow(clippy::cast_possible_truncation)]
fn run_graph_generation(model_path: &str, tokenizer: &Tokenizer, cli: &Cli) -> Result<()> {
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

    // Load weights once (use a dummy seq_len=1 graph to discover weight metadata)
    let (dummy_graph, _) = build_prefill_graph::<CpuBackend>(&config, 1, DType::F32);
    let mut weights = WeightStore::<CpuTensor, CpuLinearWeight>::new();
    eprint!("Loading weights...");
    if is_gguf {
        load_graph_weights_gguf(&dummy_graph, &config, model_path, &mut weights)?;
    } else {
        load_graph_weights_safetensors(&dummy_graph, model_path, &mut weights)?;
    }
    eprintln!(" done.");
    eprintln!(
        "Model: {} layers, {} hidden, vocab {}",
        config.num_hidden_layers, config.hidden_size, config.vocab_size,
    );
    eprintln!("Generation: greedy (graph, no KV cache)");

    // Tokenize prompt
    let mut token_ids = tokenizer.encode(&cli.prompt, true)?;
    let prompt_len = token_ids.len();
    let eos = tokenizer.eos_token_id();

    print!("{}", cli.prompt);
    io::stdout().flush()?;

    let start = Instant::now();

    for _ in 0..cli.max_tokens {
        let seq_len = token_ids.len();

        // Build graph for current sequence length
        let (mut graph, _) = build_prefill_graph::<CpuBackend>(&config, seq_len, DType::F32);
        infernum::graph::optimizer::optimize(&mut graph);
        let exec_plan = plan(&graph);

        // Build inputs
        let input_ids = CpuTensor::from_u32(&[seq_len], &token_ids);
        let (cos_data, sin_data) =
            infernum::rope::precompute_rope_data(seq_len, head_dim, config.rope_theta);
        let cos_cache = CpuTensor::from_f32(&[seq_len, half_dim], &cos_data);
        let sin_cache = CpuTensor::from_f32(&[seq_len, half_dim], &sin_data);
        let inputs = vec![input_ids, cos_cache, sin_cache];

        // Execute graph
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

        // Argmax the last position's logits
        let logits = &outputs[0];
        let logits_data = logits.as_f32_slice();
        let vocab_size = config.vocab_size;
        let last_row = &logits_data[(seq_len - 1) * vocab_size..seq_len * vocab_size];
        let next_token = last_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();

        if next_token == eos {
            break;
        }

        token_ids.push(next_token);

        let text = tokenizer.decode_token(next_token)?;
        print!("{text}");
        io::stdout().flush()?;
    }

    let elapsed = start.elapsed();
    let generated = token_ids.len() - prompt_len;

    println!();
    println!(
        "Generated {} tokens in {:.2}s ({:.1} tokens/sec)",
        generated,
        elapsed.as_secs_f64(),
        generated as f64 / elapsed.as_secs_f64(),
    );

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let is_gguf = cli.model.ends_with(".gguf");

    if cli.graph {
        // Graph mode — only supports Llama family
        let arch = if is_gguf {
            let a = detect_gguf_arch(&cli.model)?;
            println!(
                "Loading model from: {} (CPU, GGUF, arch={a}, graph)",
                cli.model
            );
            a
        } else {
            let model_type = detect_model_type(&cli.model).unwrap_or_default();
            println!(
                "Loading model from: {} (CPU, SafeTensors, {model_type}, graph)",
                cli.model
            );
            model_type
        };

        match arch.as_str() {
            "llama" | "mistral" => {}
            other => {
                eprintln!("Graph mode only supports Llama/Mistral, got: {other}");
                std::process::exit(1);
            }
        }

        let tokenizer: Tokenizer = if is_gguf {
            let loader = infernum::weights::gguf::GgufLoader::from_file(&cli.model)?;
            Tokenizer::Gguf(GgufTokenizer::from_gguf_metadata(loader.metadata())?)
        } else {
            Tokenizer::HuggingFace(LlamaTokenizer::from_pretrained(&cli.model)?)
        };

        return run_graph_generation(&cli.model, &tokenizer, &cli);
    }

    if is_gguf {
        let arch = detect_gguf_arch(&cli.model)?;
        println!("Loading model from: {} (CPU, GGUF, arch={arch})", cli.model);

        let loader = infernum::weights::gguf::GgufLoader::from_file(&cli.model)?;
        let tokenizer = Tokenizer::Gguf(GgufTokenizer::from_gguf_metadata(loader.metadata())?);

        match arch.as_str() {
            "llama" => {
                let model = LlamaModel::<CpuBackend>::from_gguf(&(), Path::new(&cli.model))?;
                run_model(model, tokenizer, &cli)
            }
            "qwen2" => {
                let model = QwenModel::<CpuBackend>::from_gguf(&(), Path::new(&cli.model))?;
                run_model(model, tokenizer, &cli)
            }
            "gemma2" | "gemma3" => {
                let model = GemmaModel::<CpuBackend>::from_gguf(&(), Path::new(&cli.model))?;
                run_model(model, tokenizer, &cli)
            }
            _ => {
                eprintln!("Unsupported GGUF architecture: {arch}");
                eprintln!("Supported: llama, qwen2, gemma2, gemma3");
                std::process::exit(1);
            }
        }
    } else {
        println!("Loading model from: {} (CPU, SafeTensors)", cli.model);
        let model = LlamaModel::<CpuBackend>::from_pretrained(&(), &cli.model)?;
        let tokenizer = Tokenizer::HuggingFace(LlamaTokenizer::from_pretrained(&cli.model)?);
        run_model(model, tokenizer, &cli)
    }
}
