//! Text generation example using Llama
//!
//! Usage:
//!   cargo run --example generate -- --model /path/to/llama "Hello"

use std::env;
use std::io::{self, Write};

use infernum::cuda::CudaContext;
use infernum::tensor::Tensor;
use infernum::tokenizer::LlamaTokenizer;
use infernum::Result;
use infernum_llama::LlamaModel;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let (model_path, prompt, max_tokens) = parse_args(&args);

    println!("Loading model from: {}", model_path);

    // Initialize CUDA
    let ctx = CudaContext::new(0)?;
    println!("CUDA context initialized");

    // Load tokenizer
    let tokenizer = LlamaTokenizer::from_pretrained(&model_path)?;
    println!("Tokenizer loaded (vocab size: {})", tokenizer.vocab_size());

    // Load model
    let model = LlamaModel::from_pretrained(&ctx, &model_path)?;
    println!(
        "Model loaded ({} layers, {} hidden)",
        model.config().num_hidden_layers,
        model.config().hidden_size
    );

    // Encode prompt
    let mut tokens = tokenizer.encode(&prompt, true)?;
    println!("Prompt: \"{}\" ({} tokens)", prompt, tokens.len());

    // Print prompt
    print!("{}", prompt);
    io::stdout().flush()?;

    // Generate tokens
    for _ in 0..max_tokens {
        // Forward pass
        let logits = model.forward(&tokens)?;

        // Get logits for last position: (seq_len, vocab_size) -> (vocab_size,)
        let logits_data = logits.to_vec()?;
        let seq_len = logits.shape()[0];
        let vocab_size = logits.shape()[1];
        let last_logits = &logits_data[(seq_len - 1) * vocab_size..];

        // Greedy decoding: argmax
        let next_token = argmax(last_logits);

        // Check for EOS
        if tokenizer.is_eos(next_token) {
            break;
        }

        // Decode and print
        let text = tokenizer.decode_token(next_token)?;
        print!("{}", text);
        io::stdout().flush()?;

        // Append to sequence
        tokens.push(next_token);
    }

    println!();
    println!("Generated {} tokens total", tokens.len());

    Ok(())
}

fn parse_args(args: &[String]) -> (String, String, usize) {
    let mut model_path = String::new();
    let mut prompt = String::new();
    let mut max_tokens = 100;

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
        // Default to checking environment variable
        model_path =
            env::var("LLAMA_MODEL_PATH").unwrap_or_else(|_| "models/llama-3.2-1b".to_string());
    }

    if prompt.is_empty() {
        prompt = "Hello".to_string();
    }

    (model_path, prompt, max_tokens)
}

fn print_usage() {
    eprintln!("Usage: generate [OPTIONS] [PROMPT]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  -m, --model <PATH>       Path to model directory (default: $LLAMA_MODEL_PATH or models/llama-3.2-1b)");
    eprintln!("  -n, --max-tokens <N>     Maximum tokens to generate (default: 100)");
    eprintln!("  -h, --help               Show this help message");
    eprintln!();
    eprintln!("Example:");
    eprintln!("  cargo run --example generate -- -m /path/to/llama \"Hello, world!\"");
}

fn argmax(slice: &[f32]) -> u32 {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;

    for (i, &val) in slice.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    max_idx as u32
}
