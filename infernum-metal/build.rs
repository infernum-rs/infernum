//! Build script: compiles Metal kernels (.metal) to .metallib at build time.
//!
//! Finds all `.metal` files in `kernels/`, invokes `xcrun metal` + `xcrun metallib`
//! for each, and writes the resulting `.metallib` file to `$OUT_DIR/kernels/`.
//! The Rust source then loads it via
//! `include_bytes!(concat!(env!("OUT_DIR"), "/kernels/infernum.metallib"))`.

use std::path::PathBuf;
use std::{env, fs};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir = manifest_dir.join("kernels");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let metal_out_dir = out_dir.join("kernels");

    fs::create_dir_all(&metal_out_dir).expect("Failed to create Metal output directory");

    // Re-run when kernel files change
    println!("cargo:rerun-if-changed={}", kernel_dir.display());

    let metal_files: Vec<PathBuf> = fs::read_dir(&kernel_dir)
        .map(|entries| {
            entries
                .filter_map(|entry| {
                    let path = entry.ok()?.path();
                    if path.extension().is_some_and(|ext| ext == "metal") {
                        Some(path)
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    if metal_files.is_empty() {
        // No kernels yet — write an empty marker so include_bytes! works
        // when we add the metallib loading later.
        return;
    }

    // Compile each .metal → .air
    let mut air_files = Vec::new();
    for metal_path in &metal_files {
        let stem = metal_path.file_stem().unwrap().to_str().unwrap();
        let air_path = metal_out_dir.join(format!("{stem}.air"));

        println!("cargo:rerun-if-changed={}", metal_path.display());

        let status = std::process::Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-c",
                metal_path.to_str().unwrap(),
                "-o",
                air_path.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to execute xcrun metal. Is Xcode installed?");

        assert!(
            status.success(),
            "xcrun metal failed to compile {}",
            metal_path.display()
        );

        air_files.push(air_path);
    }

    // Link all .air files into a single .metallib
    let metallib_path = metal_out_dir.join("infernum.metallib");
    let mut cmd = std::process::Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    for air in &air_files {
        cmd.arg(air.to_str().unwrap());
    }
    cmd.args(["-o", metallib_path.to_str().unwrap()]);

    let status = cmd
        .status()
        .expect("Failed to execute xcrun metallib. Is Xcode installed?");

    assert!(status.success(), "xcrun metallib failed to link .air files");
}
