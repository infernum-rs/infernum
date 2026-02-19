//! Build script: compiles CUDA kernels (.cu) to PTX at build time.
//!
//! When the `cuda` feature is enabled, this script finds all `.cu` files in
//! `kernels/`, invokes `nvcc --ptx` for each, and writes the resulting `.ptx`
//! files to `$OUT_DIR/kernels/`. The Rust source then loads them via
//! `include_str!(concat!(env!("OUT_DIR"), "/kernels/<name>.ptx"))`.

fn main() {
    #[cfg(feature = "cuda")]
    cuda::compile_kernels();
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::{env, fs};

    pub fn compile_kernels() {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let kernel_dir = manifest_dir.join("kernels");
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let ptx_dir = out_dir.join("kernels");

        fs::create_dir_all(&ptx_dir).expect("Failed to create PTX output directory");

        let cu_files: Vec<PathBuf> = fs::read_dir(&kernel_dir)
            .expect("Failed to read kernels/ directory")
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                if path.extension().is_some_and(|ext| ext == "cu") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        assert!(
            !cu_files.is_empty(),
            "No .cu files found in {}",
            kernel_dir.display()
        );

        for cu_path in &cu_files {
            compile_cu(&ptx_dir, cu_path);
        }
    }

    fn compile_cu(ptx_dir: &Path, cu_path: &Path) {
        let stem = cu_path.file_stem().unwrap().to_str().unwrap();
        let ptx_path = ptx_dir.join(format!("{stem}.ptx"));

        println!("cargo:rerun-if-changed={}", cu_path.display());

        let status = Command::new("nvcc")
            .args([
                "--ptx",
                "-o",
                ptx_path.to_str().unwrap(),
                cu_path.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to execute nvcc. Is the CUDA toolkit installed?");

        assert!(
            status.success(),
            "nvcc failed to compile {}",
            cu_path.display()
        );
    }
}
