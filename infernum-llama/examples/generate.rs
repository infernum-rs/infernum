//! Text generation example using Llama
//!
//! Usage:
//!   cargo run --example generate --features cuda -- --model /path/to/llama "Hello"
//!   cargo run --example generate --features cuda -- --model /path/to/llama --temperature 0.8 --top-p 0.9 "Hello"

use std::env;
use std::io::{self, Write};

use infernum::cuda::CudaContext;
use infernum::tokenizer::LlamaTokenizer;
use infernum::Result;
use infernum_llama::{LlamaModel, SamplingParams};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let cli = parse_args(&args);

    println!("Loading model from: {}", cli.model_path);

    // Initialize CUDA
    let ctx = CudaContext::new(0)?;
    println!("CUDA context initialized");

    // Load tokenizer
    let tokenizer = LlamaTokenizer::from_pretrained(&cli.model_path)?;
    println!("Tokenizer loaded (vocab size: {})", tokenizer.vocab_size());

    // Load model
    let model = LlamaModel::from_pretrained(&ctx, &cli.model_path)?;
    println!(
        "Model loaded ({} layers, {} hidden)",
        model.config().num_hidden_layers,
        model.config().hidden_size
    );

    // Encode prompt
    let tokens = tokenizer.encode(&cli.prompt, true)?;
    let prompt_len = tokens.len();
    println!("Prompt: \"{}\" ({} tokens)", cli.prompt, prompt_len);

    let eos = Some(tokenizer.eos_token_id());

    // Generate
    let output_tokens = if let Some(params) = &cli.sampling {
        println!(
            "Sampling: temperature={}, top_p={}, seed={}",
            params.temperature, params.top_p, params.seed
        );
        model.generate_sampled(&tokens, cli.max_tokens, eos, params)?
    } else {
        println!("Decoding: greedy (argmax)");
        model.generate(&tokens, cli.max_tokens, eos)?
    };

    // Decode and print the generated part
    print!("{}", cli.prompt);
    for &tok in &output_tokens[prompt_len..] {
        let text = tokenizer.decode_token(tok)?;
        print!("{}", text);
        io::stdout().flush()?;
    }

    println!();
    println!("Generated {} tokens total", output_tokens.len());

    Ok(())
}

struct CliArgs {
    model_path: String,
    prompt: String,
    max_tokens: usize,
    sampling: Option<SamplingParams>,
}

fn parse_args(args: &[String]) -> CliArgs {
    let mut model_path = String::new();
    let mut prompt = String::new();
    let mut max_tokens = 100;
    let mut temperature: Option<f32> = None;
    let mut top_p: Option<f32> = None;
    let mut seed: Option<u64> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                if i < args.len() {
                    model_path = args[i].clone();
                }
            }
            "--max-tokens" | "-n" => {
                i += 1;
                if i < args.len() {
                    max_tokens = args[i].parse().unwrap_or(100);
                }
            }
            "--temperature" | "-t" => {
                i += 1;
                if i < args.len() {
                    temperature = Some(args[i].parse().unwrap_or(0.7));
                }
            }
            "--top-p" | "-p" => {
                i += 1;
                if i < args.len() {
                    top_p = Some(args[i].parse().unwrap_or(0.9));
                }
            }
            "--seed" | "-s" => {
                i += 1;
                if i < args.len() {
                    seed = Some(args[i].parse().unwrap_or(42));
                }
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            arg if !arg.starts_with('-') && prompt.is_empty() => {
                prompt = arg.to_string();
            }
            _ => {}
        }
        i += 1;
    }

    if model_path.is_empty() {
        model_path =
            env::var("LLAMA_MODEL_PATH").unwrap_or_else(|_| "models/llama-3.2-1b".to_string());
    }

    if prompt.is_empty() {
        prompt = "Hello".to_string();
    }

    // Enable sampling if any sampling flag was provided
    let sampling = if temperature.is_some() || top_p.is_some() || seed.is_some() {
        let defaults = SamplingParams::default();
        Some(SamplingParams {
            temperature: temperature.unwrap_or(defaults.temperature),
            top_p: top_p.unwrap_or(defaults.top_p),
            seed: seed.unwrap_or(defaults.seed),
        })
    } else {
        None
    };

    CliArgs {
        model_path,
        prompt,
        max_tokens,
        sampling,
    }
}

fn print_usage() {
    eprintln!("Usage: generate [OPTIONS] [PROMPT]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  -m, --model <PATH>       Path to model directory (default: $LLAMA_MODEL_PATH or models/llama-3.2-1b)");
    eprintln!("  -n, --max-tokens <N>     Maximum tokens to generate (default: 100)");
    eprintln!("  -t, --temperature <F>    Sampling temperature (default: 0.7, enables sampling)");
    eprintln!(
        "  -p, --top-p <F>          Nucleus sampling threshold (default: 0.9, enables sampling)"
    );
    eprintln!("  -s, --seed <N>           RNG seed for sampling (default: 42, enables sampling)");
    eprintln!("  -h, --help               Show this help message");
    eprintln!();
    eprintln!("Without --temperature/--top-p/--seed, uses greedy (argmax) decoding.");
    eprintln!();
    eprintln!("Examples:");
    eprintln!(
        "  cargo run --example generate --features cuda -- -m /path/to/model \"Hello, world!\""
    );
    eprintln!("  cargo run --example generate --features cuda -- -m /path/to/model -t 0.8 -p 0.95 \"Once upon a time\"");
}
