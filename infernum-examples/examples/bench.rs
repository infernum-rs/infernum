//! Minimal benchmark for infernum decode throughput
//! Measures raw forward_next_token performance (no tokenizer decode needed)

use std::time::Instant;

use infernum::cuda::CudaContext;
use infernum::{GenerateOptions, Model};
use infernum_llama::LlamaModel;
use infernum_runtime::Engine;

fn main() -> infernum::Result<()> {
    let model_path = std::env::args().nth(1).expect("Usage: bench <model_path>");
    let n_gen: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "128".to_string())
        .parse()
        .unwrap();

    let ctx = CudaContext::new(0)?;

    let is_gguf = model_path.ends_with(".gguf");
    let model = if is_gguf {
        LlamaModel::from_gguf(&ctx, &model_path)?
    } else {
        LlamaModel::from_pretrained(&ctx, &model_path)?
    };

    eprintln!(
        "Model loaded ({} layers, {} hidden)",
        model.config().num_hidden_layers,
        model.config().hidden_size,
    );

    let mut engine = Engine::new(&ctx, model)?;

    // Use simple numeric prompt tokens that are always valid
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let options = GenerateOptions {
        max_new_tokens: n_gen,
        eos_token_id: None, // don't stop early
        sampling: None,     // greedy
        use_kv_cache: true,
    };

    // Warmup
    let _ = engine.generate(
        &prompt,
        &GenerateOptions {
            max_new_tokens: 2,
            eos_token_id: None,
            sampling: None,
            use_kv_cache: true,
        },
    )?;

    // Benchmark
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
