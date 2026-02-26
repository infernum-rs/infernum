fn main() -> infernum::Result<()> {
    use infernum::tokenizer::GgufTokenizer;
    use infernum_cuda::GgufLoader;
    let loader = GgufLoader::from_file("models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf")?;
    let tokenizer = GgufTokenizer::from_gguf_metadata(loader.metadata())?;

    // Test cases with expected tokenization from HuggingFace
    let test_cases = [
        ("Hello", true, vec![1, 15043]),
        ("the", true, vec![1, 278]),
        ("The meaning of life", true, vec![1, 450, 6593, 310, 2834]),
        ("Hello, World!", true, vec![1, 15043, 29892, 2787, 29991]), // May differ
        ("", true, vec![1]),
        ("a", false, vec![263]),
    ];

    let mut passed = 0;
    let mut failed = 0;

    for (text, add_bos, expected) in test_cases {
        let tokens = tokenizer.encode(text, add_bos)?;
        let ok = tokens == expected;

        if ok {
            passed += 1;
            println!("✓ '{}' → {:?}", text, tokens);
        } else {
            failed += 1;
            println!("✗ '{}'", text);
            println!("  Expected: {:?}", expected);
            println!("  Got:      {:?}", tokens);
            for &id in &tokens {
                println!("    {} -> '{}'", id, tokenizer.decode_token(id)?);
            }
        }
    }

    println!(
        "
Passed: {}, Failed: {}",
        passed, failed
    );

    // Test decode
    println!(
        "
Decode test:"
    );
    let tokens = vec![1, 450, 6593, 310, 2834];
    let decoded = tokenizer.decode(&tokens)?;
    println!("decode({:?}) = '{}'", tokens, decoded);

    Ok(())
}
