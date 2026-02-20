//! Demonstrates Infernum's block fusion system.
//!
//! This example shows how `define_block!` and `define_fusion!` work together:
//!
//! 1. A "block" wraps a function into a decomposed implementation + dispatcher
//! 2. A "fusion" registers an optimized replacement
//! 3. `fusion::init()` activates all registered fusions
//!
//! Run with:
//!   cargo run --example fusion_example
//!
//! Compare debug vs release behavior:
//!   cargo run --example fusion_example                    # debug: always decomposed
//!   cargo run --example fusion_example --release          # release: uses fused
//!   cargo run --example fusion_example --features no-fuse --release  # release but forced decomposed

use infernum::fusion;

// --- Step 1: Define a block ---
//
// `define_block!` generates:
//   - `add_and_double_decomposed(a, b)` — the original body
//   - `ADD_AND_DOUBLE_FUSED` — OnceLock static for the fused replacement
//   - `add_and_double(a, b)` — dispatcher that checks for fusion
infernum_macros::define_block! {
    /// Add two numbers and double the result.
    fn add_and_double(a: i32, b: i32) -> i32 {
        let sum = a + b;
        sum * 2
    }
}

// --- Step 2: Define a fusion ---
//
// `define_fusion!` registers an optimized implementation that replaces
// the decomposed version. The `block:` path points to the OnceLock static.
infernum_macros::define_fusion! {
    block: ADD_AND_DOUBLE_FUSED,
    fn add_and_double_fused(a: i32, b: i32) -> i32 {
        // In a real fusion, this would be a single CUDA kernel instead of
        // two separate ops. Here we just mark it so we can tell which path ran.
        (a + b) * 2 + 1000  // +1000 so we can distinguish fused from decomposed
    }
}

fn main() {
    let a = 3;
    let b = 7;

    // Before init: fused replacement is not yet registered
    println!("=== Before fusion::init() ===");
    println!("add_and_double({a}, {b}) = {}", add_and_double(a, b));
    println!(
        "  (static populated: {})",
        ADD_AND_DOUBLE_FUSED.get().is_some()
    );

    // Initialize the fusion registry — populates all OnceLock statics
    fusion::init();

    println!(
        "
=== After fusion::init() ==="
    );
    println!(
        "  (static populated: {})",
        ADD_AND_DOUBLE_FUSED.get().is_some()
    );
    println!("add_and_double({a}, {b}) = {}", add_and_double(a, b));

    // The decomposed version is always available directly
    println!(
        "add_and_double_decomposed({a}, {b}) = {}",
        add_and_double_decomposed(a, b)
    );

    // Explain what happened
    let result = add_and_double(a, b);
    if result >= 1000 {
        println!(
            "
→ Dispatcher used the FUSED implementation"
        );
    } else {
        println!(
            "
→ Dispatcher used the DECOMPOSED implementation"
        );
    }

    println!(
        "
Tip: In debug builds, decomposed is always used (zero overhead)."
    );
    println!("     In release builds, fused is used after init().");
    println!("     Use --features force-fuse to force fused in debug.");
    println!("     Use --features no-fuse to force decomposed in release.");
}
