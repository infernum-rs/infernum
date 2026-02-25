#!/usr/bin/env bash
# bench_comparison.sh — Compare infernum vs llama.cpp decode throughput
#
# Benchmarks Llama-3.2-1B across multiple quantization formats.
# Outputs a Markdown table to stdout.
#
# Prerequisites:
#   - CUDA GPU with nvidia-smi
#   - llama.cpp built at /home/amir/llama.cpp/ (with llama-bench, llama-quantize)
#   - Rust toolchain (cargo)
#   - hf CLI with token (for gated meta-llama model): pip install huggingface_hub[cli]
#   - Python 3 with torch, transformers, gguf packages
#
# Usage:
#   ./bench_comparison.sh                          # Run all benchmarks
#   ./bench_comparison.sh --tests fp8,gptq-int4    # Run only FP8 and GPTQ INT4
#   ./bench_comparison.sh --dry-run                # Show what would run without executing
#   ./bench_comparison.sh --dry-run --tests q8     # Combine flags
#
# Available test names (case-insensitive, comma-separated):
#   f32, bf16, f16, fp8, q8, q4, gptq-int4, all (default)

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

LLAMA_CPP="/home/amir/llama.cpp"
LLAMA_BENCH="${LLAMA_CPP}/build/bin/llama-bench"
LLAMA_QUANTIZE="${LLAMA_CPP}/build/bin/llama-quantize"
CONVERT_SCRIPT="${LLAMA_CPP}/convert_hf_to_gguf.py"

MODEL_CACHE="$HOME/.cache/infernum/models"
GGUF_DIR="/tmp"

N_GEN=256
PROMPT_LEN=8   # llama-bench -p for prompt eval (0 = skip prompt eval)
LLAMA_BENCH_REPS=3

# HuggingFace repos
HF_BASE_REPO="meta-llama/Llama-3.2-1B"
HF_FP8_REPO="RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic"
HF_GPTQ_REPO="shuyuej/Llama-3.2-1B-GPTQ"

# Local paths (derived)
BASE_MODEL_DIR="${MODEL_CACHE}/meta-llama/Llama-3.2-1B"
FP8_MODEL_DIR="${MODEL_CACHE}/RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic"
GPTQ_MODEL_DIR="${MODEL_CACHE}/shuyuej/Llama-3.2-1B-GPTQ"

GGUF_F32="${GGUF_DIR}/llama-3.2-1b-f32.gguf"
GGUF_F16="${GGUF_DIR}/llama-3.2-1b-f16.gguf"
GGUF_BF16="${GGUF_DIR}/llama-3.2-1b-bf16.gguf"
GGUF_Q8="${GGUF_DIR}/llama-3.2-1b-q8_0.gguf"
GGUF_Q4="${GGUF_DIR}/llama-3.2-1b-q4_0.gguf"

DRY_RUN=false
TESTS_FILTER="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --tests)   TESTS_FILTER="$2"; shift 2 ;;
        --tests=*) TESTS_FILTER="${1#--tests=}"; shift ;;
        -h|--help)
            echo "Usage: $0 [--tests <list>] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --tests <list>  Comma-separated formats to benchmark (default: all)"
            echo "                  Names: f32, bf16, f16, fp8, q8, q4, gptq-int4, all"
            echo "  --dry-run       Show plan without running benchmarks"
            exit 0 ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
    esac
done

# Normalize test filter into a lookup set
declare -A ENABLED_TESTS
if [[ "${TESTS_FILTER,,}" == "all" ]]; then
    for t in f32 bf16 f16 fp8 q8 q4 gptq-int4; do
        ENABLED_TESTS[$t]=1
    done
else
    IFS=',' read -ra _tests <<< "${TESTS_FILTER,,}"
    for t in "${_tests[@]}"; do
        t=$(echo "$t" | xargs)  # trim whitespace
        case "$t" in
            f32|bf16|f16|fp8|q8|q4|gptq-int4) ENABLED_TESTS[$t]=1 ;;
            *) echo "ERROR: Unknown test name: '$t'. Valid: f32, bf16, f16, fp8, q8, q4, gptq-int4, all" >&2; exit 1 ;;
        esac
    done
fi

# Check if a test is enabled. Maps benchmark display names to filter keys.
test_enabled() {
    local key
    case "$1" in
        F32)        key="f32" ;;
        BF16)       key="bf16" ;;
        F16)        key="f16" ;;
        FP8)        key="fp8" ;;
        "GGUF Q8_0") key="q8" ;;
        "GGUF Q4_0") key="q4" ;;
        "GPTQ INT4") key="gptq-int4" ;;
        *) return 1 ;;
    esac
    [[ -n "${ENABLED_TESTS[$key]:-}" ]]
}

export LD_LIBRARY_PATH="${LLAMA_CPP}/build/bin:${LD_LIBRARY_PATH:-}"

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { echo ">>> $*" >&2; }

die() { echo "ERROR: $*" >&2; exit 1; }

# Download a HuggingFace model to the infernum cache using the hf CLI.
# Usage: download_hf_model <repo_id> <dest_dir>
download_hf_model() {
    local repo_id="$1"
    local dest_dir="$2"
    if [[ -f "${dest_dir}/config.json" && -f "${dest_dir}/tokenizer.json" ]]; then
        log "Model already cached: ${dest_dir}"
        return 0
    fi
    log "Downloading ${repo_id} → ${dest_dir}"
    mkdir -p "${dest_dir}"

    local files=("config.json" "tokenizer.json" "tokenizer_config.json")
    # Check for model.safetensors or sharded model files
    if hf download "${repo_id}" "model.safetensors" --local-dir "${dest_dir}" --quiet 2>/dev/null; then
        : # single file model
    else
        # Try sharded — download the index then all shards
        hf download "${repo_id}" --local-dir "${dest_dir}" --quiet \
            --include "model.safetensors.index.json" "model-*.safetensors" 2>/dev/null || true
    fi
    for f in "${files[@]}"; do
        if [[ ! -f "${dest_dir}/${f}" ]]; then
            hf download "${repo_id}" "${f}" --local-dir "${dest_dir}" --quiet 2>/dev/null || true
        fi
    done

    [[ -f "${dest_dir}/config.json" ]] || die "Failed to download ${repo_id}"
}

# Convert a SafeTensors model to GGUF if the output doesn't already exist.
# Usage: convert_to_gguf <model_dir> <output_gguf> <outtype>
convert_to_gguf() {
    local model_dir="$1"
    local output="$2"
    local outtype="$3"
    if [[ -f "${output}" ]]; then
        log "GGUF already exists: ${output}"
        return 0
    fi
    log "Converting → ${output} (type=${outtype})"
    python3 "${CONVERT_SCRIPT}" "${model_dir}" --outfile "${output}" --outtype "${outtype}" 2>/dev/null
}

# Quantize a GGUF file if the output doesn't already exist.
# Usage: quantize_gguf <input_gguf> <output_gguf> <quant_type>
quantize_gguf() {
    local input="$1"
    local output="$2"
    local qtype="$3"
    if [[ -f "${output}" ]]; then
        log "Quantized GGUF already exists: ${output}"
        return 0
    fi
    log "Quantizing → ${output} (type=${qtype})"
    "${LLAMA_QUANTIZE}" "${input}" "${output}" "${qtype}" >/dev/null 2>&1
}

# Run llama-bench and extract decode tok/s.
# Usage: run_llama_bench <gguf_path>
run_llama_bench() {
    local gguf="$1"
    if $DRY_RUN; then
        echo "—"
        return
    fi
    local json
    json=$("${LLAMA_BENCH}" -m "${gguf}" -p 0 -n "${N_GEN}" -r "${LLAMA_BENCH_REPS}" -ngl 99 -o jsonl 2>/dev/null)
    echo "${json}" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(round(d["avg_ts"], 1))'
}

# Run infernum bench and extract tok/s.
# Tries --graphs first; falls back to no graphs if capture fails.
# Usage: run_infernum_bench <model_path> [--dtype <dtype>]
run_infernum_bench() {
    local model_path="$1"
    shift
    local extra_args=("$@")
    if $DRY_RUN; then
        echo "—"
        return
    fi

    local output toks
    # Try with CUDA graphs first (timeout guards against capture hangs)
    output=$(timeout 300 cargo run --release --example bench --features cuda -q -- \
        "${model_path}" "${N_GEN}" --graphs "${extra_args[@]}" 2>/dev/null || true)
    toks=$(echo "${output}" | grep -oP '[\d.]+(?= tok/s)' | tail -1)
    if [[ -n "${toks}" ]]; then
        echo "${toks}"
        return
    fi

    # Fallback: no graphs
    log "  (CUDA graphs failed, retrying without)"
    output=$(timeout 300 cargo run --release --example bench --features cuda -q -- \
        "${model_path}" "${N_GEN}" "${extra_args[@]}" 2>/dev/null || true)
    toks=$(echo "${output}" | grep -oP '[\d.]+(?= tok/s)' | tail -1)
    echo "${toks:-ERR}"
}

# Run infernum bench without CUDA graphs (for quantized formats where
# graph capture is not yet supported).
# Usage: run_infernum_bench_no_graphs <model_path> [--dtype <dtype>]
run_infernum_bench_no_graphs() {
    local model_path="$1"
    shift
    local extra_args=("$@")
    if $DRY_RUN; then
        echo "—"
        return
    fi

    local output toks
    output=$(timeout 300 cargo run --release --example bench --features cuda -q -- \
        "${model_path}" "${N_GEN}" "${extra_args[@]}" 2>/dev/null || true)
    toks=$(echo "${output}" | grep -oP '[\d.]+(?= tok/s)' | tail -1)
    echo "${toks:-ERR}"
}

# ── Preflight checks ─────────────────────────────────────────────────────────

check_prerequisites() {
    local missing=()

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        missing+=("  ✗ nvidia-smi    — CUDA driver not found")
    fi

    if ! command -v cargo >/dev/null 2>&1; then
        missing+=("  ✗ cargo         — install via https://rustup.rs")
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        missing+=("  ✗ python3       — required for GGUF conversion")
    fi

    if ! command -v hf >/dev/null 2>&1; then
        missing+=("  ✗ hf            — install: pip install 'huggingface_hub[cli]'")
    fi

    if [[ ! -x "${LLAMA_BENCH}" ]]; then
        missing+=("  ✗ llama-bench   — not found at ${LLAMA_BENCH}")
    fi

    if [[ ! -x "${LLAMA_QUANTIZE}" ]]; then
        missing+=("  ✗ llama-quantize — not found at ${LLAMA_QUANTIZE}")
    fi

    if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
        missing+=("  ✗ convert_hf_to_gguf.py — not found at ${CONVERT_SCRIPT}")
    fi

    if command -v python3 >/dev/null 2>&1; then
        for pkg in torch transformers gguf; do
            if ! python3 -c "import ${pkg}" 2>/dev/null; then
                missing+=("  ✗ python: ${pkg}  — install: pip install ${pkg}")
            fi
        done
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "ERROR: Missing prerequisites:" >&2
        echo "" >&2
        for m in "${missing[@]}"; do
            echo "${m}" >&2
        done
        echo "" >&2
        exit 1
    fi

    log "All prerequisites satisfied ✓"
}

check_prerequisites

# ── Model preparation ────────────────────────────────────────────────────────

log "Preparing models..."

needs_base=false
needs_gguf=false
for t in f32 bf16 f16; do
    [[ -n "${ENABLED_TESTS[$t]:-}" ]] && needs_base=true
done
for t in f32 f16 bf16 q8 q4; do
    [[ -n "${ENABLED_TESTS[$t]:-}" ]] && needs_gguf=true
done

if $needs_base || $needs_gguf; then
    download_hf_model "${HF_BASE_REPO}" "${BASE_MODEL_DIR}"
fi

if [[ -n "${ENABLED_TESTS[fp8]:-}" ]]; then
    download_hf_model "${HF_FP8_REPO}" "${FP8_MODEL_DIR}"
fi

if [[ -n "${ENABLED_TESTS[gptq-int4]:-}" ]]; then
    download_hf_model "${HF_GPTQ_REPO}" "${GPTQ_MODEL_DIR}"
fi

# Convert to GGUF formats for llama.cpp (only as needed)
if $needs_gguf; then
    # F16 GGUF is also the source for Q8/Q4 quantization
    needs_f16_gguf=false
    for t in f16 q8 q4; do
        [[ -n "${ENABLED_TESTS[$t]:-}" ]] && needs_f16_gguf=true
    done

    [[ -n "${ENABLED_TESTS[f32]:-}" ]] && convert_to_gguf "${BASE_MODEL_DIR}" "${GGUF_F32}" "f32"
    $needs_f16_gguf && convert_to_gguf "${BASE_MODEL_DIR}" "${GGUF_F16}" "f16"
    [[ -n "${ENABLED_TESTS[bf16]:-}" ]] && convert_to_gguf "${BASE_MODEL_DIR}" "${GGUF_BF16}" "bf16"
    [[ -n "${ENABLED_TESTS[q8]:-}" ]] && quantize_gguf "${GGUF_F16}" "${GGUF_Q8}" "q8_0"
    [[ -n "${ENABLED_TESTS[q4]:-}" ]] && quantize_gguf "${GGUF_F16}" "${GGUF_Q4}" "q4_0"
fi

# ── Build infernum ────────────────────────────────────────────────────────────

log "Building infernum (release)..."
if ! $DRY_RUN; then
    cargo build --release --example bench --features cuda -q 2>/dev/null
fi

# ── Run benchmarks ────────────────────────────────────────────────────────────

declare -A results_llama
declare -A results_infernum

# All possible benchmarks in display order
all_benchmarks=("F32" "BF16" "F16" "FP8" "GGUF Q8_0" "GGUF Q4_0" "GPTQ INT4")

# Filter to only enabled benchmarks
benchmarks=()
for b in "${all_benchmarks[@]}"; do
    test_enabled "$b" && benchmarks+=("$b")
done

total=${#benchmarks[@]}
if [[ $total -eq 0 ]]; then
    die "No benchmarks selected"
fi

log ""
log "Running ${total} benchmark(s) (${N_GEN} tokens each)..."
log "─────────────────────────────────────────────"

step=0
for bench in "${benchmarks[@]}"; do
    step=$((step + 1))
    case "$bench" in
        F32)
            log "[${step}/${total}] F32 — llama.cpp"
            results_llama["F32"]=$(run_llama_bench "${GGUF_F32}")
            log "[${step}/${total}] F32 — infernum"
            results_infernum["F32"]=$(run_infernum_bench "${BASE_MODEL_DIR}" --dtype f32)
            ;;
        BF16)
            log "[${step}/${total}] BF16 — llama.cpp"
            results_llama["BF16"]=$(run_llama_bench "${GGUF_BF16}")
            log "[${step}/${total}] BF16 — infernum"
            results_infernum["BF16"]=$(run_infernum_bench "${BASE_MODEL_DIR}" --dtype bf16)
            ;;
        F16)
            log "[${step}/${total}] F16 — llama.cpp"
            results_llama["F16"]=$(run_llama_bench "${GGUF_F16}")
            log "[${step}/${total}] F16 — infernum (n/a — uses bf16 path)"
            results_infernum["F16"]="—"
            ;;
        FP8)
            log "[${step}/${total}] FP8 — llama.cpp (n/a)"
            results_llama["FP8"]="—"
            log "[${step}/${total}] FP8 — infernum (no graphs — not yet supported)"
            results_infernum["FP8"]=$(run_infernum_bench_no_graphs "${FP8_MODEL_DIR}")
            ;;
        "GGUF Q8_0")
            log "[${step}/${total}] GGUF Q8_0 — llama.cpp"
            results_llama["GGUF Q8_0"]=$(run_llama_bench "${GGUF_Q8}")
            log "[${step}/${total}] GGUF Q8_0 — infernum (no graphs — not yet supported)"
            results_infernum["GGUF Q8_0"]=$(run_infernum_bench_no_graphs "${GGUF_Q8}")
            ;;
        "GGUF Q4_0")
            log "[${step}/${total}] GGUF Q4_0 — llama.cpp"
            results_llama["GGUF Q4_0"]=$(run_llama_bench "${GGUF_Q4}")
            log "[${step}/${total}] GGUF Q4_0 — infernum (no graphs — not yet supported)"
            results_infernum["GGUF Q4_0"]=$(run_infernum_bench_no_graphs "${GGUF_Q4}")
            ;;
        "GPTQ INT4")
            log "[${step}/${total}] GPTQ INT4 — llama.cpp (n/a)"
            results_llama["GPTQ INT4"]="—"
            log "[${step}/${total}] GPTQ INT4 — infernum (no graphs — not yet supported)"
            results_infernum["GPTQ INT4"]=$(run_infernum_bench_no_graphs "${GPTQ_MODEL_DIR}")
            ;;
    esac
done

log ""
log "Done. Results:"
log ""

# ── Output table ──────────────────────────────────────────────────────────────

# Compute the "gap" column
compute_gap() {
    local llama="$1"
    local infernum="$2"
    if [[ "${llama}" == "—" || "${infernum}" == "—" ]]; then
        echo "—"
        return
    fi
    python3 -c "
l, i = float('${llama}'), float('${infernum}')
pct = (i - l) / l * 100
sign = '+' if pct >= 0 else ''
print(f'{sign}{pct:.0f}%')
"
}

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
infernum_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
llama_commit="unknown"
if llama_json=$("${LLAMA_BENCH}" -m "${GGUF_Q8}" -p 0 -n 1 -r 1 -ngl 99 -o jsonl 2>/dev/null); then
    llama_commit=$(echo "${llama_json}" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["build_commit"])' 2>/dev/null || echo "unknown")
fi

echo ""
echo "## Infernum vs llama.cpp — Llama 3.2 1B decode throughput"
echo ""
echo "- **GPU:** ${gpu_name}"
echo "- **Tokens generated:** ${N_GEN} (${PROMPT_LEN}-token prompt, greedy, KV cache)"
echo "- **infernum:** commit \`${infernum_commit}\` (buffer pool + CUDA graphs)"
echo "- **llama.cpp:** commit \`${llama_commit}\` (\`-ngl 99\`, ${LLAMA_BENCH_REPS} reps)"
echo "- **Date:** $(date +%Y-%m-%d)"
echo ""
echo "| Format | llama.cpp (tok/s) | infernum (tok/s) | Gap |"
echo "| ------ | ----------------: | ---------------: | --: |"

for bench in "${benchmarks[@]}"; do
    l="${results_llama[$bench]}"
    i="${results_infernum[$bench]}"
    gap=$(compute_gap "${l}" "${i}")

    # Right-align numbers, left-align dashes
    printf "| %-14s | %17s | %16s | %3s |
" "${bench}" "${l}" "${i}" "${gap}"
done

echo ""
