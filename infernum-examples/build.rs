//! Build script for infernum-examples: compiles example CUDA and Triton kernels.
//!
//! - CUDA `.cu` files are compiled to PTX via `nvcc`
//! - Triton `.py` files are compiled to PTX via `python3`
//!
//! Both require the `cuda` feature to be enabled.

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

        // Compile CUDA kernels (.cu → .ptx via nvcc)
        for entry in fs::read_dir(&kernel_dir).expect("Failed to read kernels/ directory") {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|ext| ext == "cu") {
                compile_cu(&ptx_dir, &path);
            }
        }

        // Compile Triton kernels (.py → .ptx via python3)
        for entry in fs::read_dir(&kernel_dir).expect("Failed to read kernels/ directory") {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|ext| ext == "py") {
                compile_triton(&ptx_dir, &path);
            }
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

    fn python_path() -> PathBuf {
        // Prefer the virtualenv Python if VIRTUAL_ENV is set
        if let Ok(venv) = env::var("VIRTUAL_ENV") {
            let venv_python = PathBuf::from(venv).join("bin/python3");
            if venv_python.exists() {
                return venv_python;
            }
        }
        PathBuf::from("python3")
    }

    fn compile_triton(ptx_dir: &Path, py_path: &Path) {
        println!("cargo:rerun-if-changed={}", py_path.display());

        let python = python_path();
        let status = Command::new(&python)
            .args([py_path.to_str().unwrap(), ptx_dir.to_str().unwrap()])
            .status()
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to execute {}: {e}. Is Python installed with triton?",
                    python.display()
                )
            });

        assert!(
            status.success(),
            "Triton compilation failed for {}. Ensure `triton` is installed: pip install triton",
            py_path.display()
        );
    }
}
