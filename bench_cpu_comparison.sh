#!/usr/bin/env bash
# bench_cpu_comparison.sh — Compare infernum vs llama.cpp CPU throughput
#
# Reports two separate tables:
#   1. **Decode** — autoregressive token generation (seq_len=1 per step)
#      llama.cpp (-p 0 -n N) vs infernum eager (Engine::generate) vs graph decode
#   2. **Prefill** — prompt processing in one forward pass (seq_len=N)
#      llama.cpp (-p N -n 0) vs eager (—) vs infernum graph executor
#
# Prerequisites:
#   - llama.cpp built locally (with llama-bench, llama-quantize, convert_hf_to_gguf.py)
#     Override path: --llama-cpp <path> or set LLAMA_CPP env var
#   - Rust toolchain (cargo)
#   - hf CLI: pip install 'huggingface_hub[cli]'
#   - Python 3 with torch, transformers, gguf packages (for GGUF conversion)
#
# Usage:
#   ./bench_cpu_comparison.sh                       # Run all benchmarks (SmolLM2-360M)
#   ./bench_cpu_comparison.sh --tests q8,q4         # Run only Q8_0 and Q4_0
#   ./bench_cpu_comparison.sh --dry-run             # Show what would run
#   ./bench_cpu_comparison.sh --threads 4           # Override thread count
#   ./bench_cpu_comparison.sh --llama-cpp /path/to  # Override llama.cpp location
#   ./bench_cpu_comparison.sh --n-tokens 256        # Override token count
#
# Available test names (case-insensitive, comma-separated):
#   f32, q8, q4, all (default)

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

LLAMA_CPP="${LLAMA_CPP:-/home/amir/llama.cpp}"
LLAMA_BENCH="${LLAMA_CPP}/build/bin/llama-bench"
LLAMA_QUANTIZE="${LLAMA_CPP}/build/bin/llama-quantize"
CONVERT_SCRIPT="${LLAMA_CPP}/convert_hf_to_gguf.py"

MODEL_CACHE="$HOME/.cache/infernum/models"
GGUF_DIR="/tmp"

N_TOKENS=128
LLAMA_BENCH_REPS=3
THREADS=""  # empty = use system default (all cores)

# Model — SmolLM2-360M (ungated, small, fast to download)
HF_BASE_REPO="HuggingFaceTB/SmolLM2-360M"
MODEL_NAME="SmolLM2-360M"
BASE_MODEL_DIR="${MODEL_CACHE}/SmolLM2-360M"

GGUF_F32="${GGUF_DIR}/smollm2-360m-f32.gguf"
GGUF_F16="${GGUF_DIR}/smollm2-360m-f16.gguf"
GGUF_Q8="${GGUF_DIR}/smollm2-360m-q8_0.gguf"
GGUF_Q4="${GGUF_DIR}/smollm2-360m-q4_0.gguf"

DRY_RUN=false
TESTS_FILTER="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true; shift ;;
        --tests)      TESTS_FILTER="$2"; shift 2 ;;
        --tests=*)    TESTS_FILTER="${1#--tests=}"; shift ;;
        --threads)    THREADS="$2"; shift 2 ;;
        --threads=*)  THREADS="${1#--threads=}"; shift ;;
        --n-tokens)   N_TOKENS="$2"; shift 2 ;;
        --n-tokens=*) N_TOKENS="${1#--n-tokens=}"; shift ;;
        --llama-cpp)  LLAMA_CPP="$2"; LLAMA_BENCH="${LLAMA_CPP}/build/bin/llama-bench"; LLAMA_QUANTIZE="${LLAMA_CPP}/build/bin/llama-quantize"; CONVERT_SCRIPT="${LLAMA_CPP}/convert_hf_to_gguf.py"; shift 2 ;;
        --llama-cpp=*) LLAMA_CPP="${1#--llama-cpp=}"; LLAMA_BENCH="${LLAMA_CPP}/build/bin/llama-bench"; LLAMA_QUANTIZE="${LLAMA_CPP}/build/bin/llama-quantize"; CONVERT_SCRIPT="${LLAMA_CPP}/convert_hf_to_gguf.py"; shift ;;
        -h|--help)
            echo "Usage: $0 [--tests <list>] [--threads <n>] [--n-tokens <n>] [--llama-cpp <path>] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --tests <list>       Comma-separated formats to benchmark (default: all)"
            echo "                       Names: f32, q8, q4, all"
            echo "  --threads <n>        Number of CPU threads (default: all cores)"
            echo "  --n-tokens <n>       Number of tokens for decode/prefill (default: 128)"
            echo "  --llama-cpp <path>   Path to llama.cpp directory (default: \$LLAMA_CPP or /home/amir/llama.cpp)"
            echo "  --dry-run            Show plan without running benchmarks"
            exit 0 ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
    esac
done

# Normalize test filter
declare -A ENABLED_TESTS
if [[ "${TESTS_FILTER,,}" == "all" ]]; then
    for t in f32 q8 q4; do
        ENABLED_TESTS[$t]=1
    done
else
    IFS=',' read -ra _tests <<< "${TESTS_FILTER,,}"
    for t in "${_tests[@]}"; do
        t=$(echo "$t" | xargs)
        case "$t" in
            f32|q8|q4) ENABLED_TESTS[$t]=1 ;;
            *) echo "ERROR: Unknown test name: '$t'. Valid: f32, q8, q4, all" >&2; exit 1 ;;
        esac
    done
fi

test_enabled() {
    local key
    case "$1" in
        "GGUF F32")  key="f32" ;;
        "GGUF Q8_0") key="q8" ;;
        "GGUF Q4_0") key="q4" ;;
        *) return 1 ;;
    esac
    [[ -n "${ENABLED_TESTS[$key]:-}" ]]
}

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { echo ">>> $*" >&2; }

die() { echo "ERROR: $*" >&2; exit 1; }

download_hf_model() {
    local repo_id="$1"
    local dest_dir="$2"
    if [[ -f "${dest_dir}/config.json" && -f "${dest_dir}/tokenizer.json" ]]; then
        log "Model already cached: ${dest_dir}"
        return 0
    fi
    log "Downloading ${repo_id} → ${dest_dir}"
    if $DRY_RUN; then return 0; fi
    mkdir -p "${dest_dir}"

    local files=("config.json" "tokenizer.json" "tokenizer_config.json")
    if hf download "${repo_id}" "model.safetensors" --local-dir "${dest_dir}" --quiet 2>/dev/null; then
        :
    else
        hf download "${repo_id}" --local-dir "${dest_dir}" --quiet 
            --include "model.safetensors.index.json" "model-*.safetensors" 2>/dev/null || true
    fi
    for f in "${files[@]}"; do
        if [[ ! -f "${dest_dir}/${f}" ]]; then
            hf download "${repo_id}" "${f}" --local-dir "${dest_dir}" --quiet 2>/dev/null || true
        fi
    done

    [[ -f "${dest_dir}/config.json" ]] || die "Failed to download ${repo_id}"
}

convert_to_gguf() {
    local model_dir="$1"
    local output="$2"
    local outtype="$3"
    if [[ -f "${output}" ]]; then
        log "GGUF already exists: ${output}"
        return 0
    fi
    log "Converting → ${output} (type=${outtype})"
    if $DRY_RUN; then return 0; fi
    python3 "${CONVERT_SCRIPT}" "${model_dir}" --outfile "${output}" --outtype "${outtype}" 2>/dev/null
}

quantize_gguf() {
    local input="$1"
    local output="$2"
    local qtype="$3"
    if [[ -f "${output}" ]]; then
        log "Quantized GGUF already exists: ${output}"
        return 0
    fi
    log "Quantizing → ${output} (type=${qtype})"
    if $DRY_RUN; then return 0; fi
    "${LLAMA_QUANTIZE}" "${input}" "${output}" "${qtype}" >/dev/null 2>&1
}

# Run llama-bench for CPU decode: -p 0 -n N (ngl=0).
run_llama_bench_decode() {
    local gguf="$1"
    if $DRY_RUN; then
        echo "—"
        return
    fi
    local thread_args=()
    if [[ -n "${THREADS}" ]]; then
        thread_args=(-t "${THREADS}")
    fi
    local json
    json=$("${LLAMA_BENCH}" -m "${gguf}" -p 0 -n "${N_TOKENS}" -r "${LLAMA_BENCH_REPS}" 
        -ngl 0 "${thread_args[@]}" -o jsonl 2>/dev/null)
    echo "${json}" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(round(d["avg_ts"], 1))'
}

# Run llama-bench for CPU prefill: -p N -n 0 (ngl=0).
run_llama_bench_prefill() {
    local gguf="$1"
    if $DRY_RUN; then
        echo "—"
        return
    fi
    local thread_args=()
    if [[ -n "${THREADS}" ]]; then
        thread_args=(-t "${THREADS}")
    fi
    local json
    json=$("${LLAMA_BENCH}" -m "${gguf}" -p "${N_TOKENS}" -n 0 -r "${LLAMA_BENCH_REPS}" 
        -ngl 0 "${thread_args[@]}" -o jsonl 2>/dev/null)
    echo "${json}" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(round(d["avg_ts"], 1))'
}

# Run infernum bench_cpu (eager decode).
run_infernum_decode() {
    local model_path="$1"
    if $DRY_RUN; then
        echo "—"
        return
    fi
    local thread_args=()
    if [[ -n "${THREADS}" ]]; then
        thread_args=(-j "${THREADS}")
    fi
    local output toks
    output=$(timeout 600 cargo run --release --example bench_cpu --features cpu -q -- 
        "${model_path}" "${N_TOKENS}" "${thread_args[@]}" 2>/dev/null || true)
    toks=$(echo "${output}" | grep -oP '[\d.]+(?= tok/s)' | tail -1)
    echo "${toks:-ERR}"
}

# Run infernum bench_cpu --graph (graph prefill).
run_infernum_prefill() {
    local model_path="$1"
    if $DRY_RUN; then
        echo "—"
        return
    fi
    local thread_args=()
    if [[ -n "${THREADS}" ]]; then
        thread_args=(-j "${THREADS}")
    fi
    local output toks
    output=$(timeout 600 cargo run --release --example bench_cpu --features cpu -q -- 
        --graph "${model_path}" "${N_TOKENS}" "${thread_args[@]}" 2>/dev/null || true)
    toks=$(echo "${output}" | grep -oP '[\d.]+(?= tok/s)' | tail -1)
    echo "${toks:-ERR}"
}

# Run infernum bench_cpu --graph-decode (graph decode).
run_infernum_graph_decode() {
    local model_path="$1"
    if $DRY_RUN; then
        echo "—"
        return
    fi
    local thread_args=()
    if [[ -n "${THREADS}" ]]; then
        thread_args=(-j "${THREADS}")
    fi
    local output toks
    output=$(timeout 600 cargo run --release --example bench_cpu --features cpu -q -- \
        --graph-decode "${model_path}" "${N_TOKENS}" "${thread_args[@]}" 2>/dev/null || true)
    toks=$(echo "${output}" | grep -oP '[\d.]+(?= tok/s)' | tail -1)
    echo "${toks:-ERR}"
}

# ── Preflight checks ─────────────────────────────────────────────────────────

check_prerequisites() {
    local missing=()

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

log "Preparing models (${MODEL_NAME})..."

download_hf_model "${HF_BASE_REPO}" "${BASE_MODEL_DIR}"

# F16 GGUF is the source for Q8/Q4 quantization
needs_f16=false
for t in q8 q4; do
    [[ -n "${ENABLED_TESTS[$t]:-}" ]] && needs_f16=true
done

[[ -n "${ENABLED_TESTS[f32]:-}" ]] && convert_to_gguf "${BASE_MODEL_DIR}" "${GGUF_F32}" "f32"
$needs_f16 && convert_to_gguf "${BASE_MODEL_DIR}" "${GGUF_F16}" "f16"
[[ -n "${ENABLED_TESTS[q8]:-}" ]] && quantize_gguf "${GGUF_F16}" "${GGUF_Q8}" "q8_0"
[[ -n "${ENABLED_TESTS[q4]:-}" ]] && quantize_gguf "${GGUF_F16}" "${GGUF_Q4}" "q4_0"

# ── Build infernum ────────────────────────────────────────────────────────────

log "Building infernum (release, CPU)..."
if ! $DRY_RUN; then
    cargo build --release --example bench_cpu --features cpu -q 2>/dev/null
fi

# ── Run benchmarks ────────────────────────────────────────────────────────────

# Decode results
declare -A decode_llama
declare -A decode_eager
declare -A decode_graph

# Prefill results
declare -A prefill_llama
declare -A prefill_eager
declare -A prefill_graph

all_benchmarks=("GGUF F32" "GGUF Q8_0" "GGUF Q4_0")

benchmarks=()
for b in "${all_benchmarks[@]}"; do
    test_enabled "$b" && benchmarks+=("$b")
done

total=${#benchmarks[@]}
if [[ $total -eq 0 ]]; then
    die "No benchmarks selected"
fi

effective_threads="${THREADS:-$(nproc)}"
log ""
log "Running ${total} benchmark(s) (${MODEL_NAME}, ${N_TOKENS} tokens, ${effective_threads} threads)..."
log "─────────────────────────────────────────────"

step=0
for bench in "${benchmarks[@]}"; do
    step=$((step + 1))
    case "$bench" in
        "GGUF F32")
            log "[${step}/${total}] GGUF F32 — llama.cpp decode"
            decode_llama["GGUF F32"]=$(run_llama_bench_decode "${GGUF_F32}")
            log "[${step}/${total}] GGUF F32 — llama.cpp prefill"
            prefill_llama["GGUF F32"]=$(run_llama_bench_prefill "${GGUF_F32}")
            log "[${step}/${total}] GGUF F32 — infernum decode (eager)"
            decode_eager["GGUF F32"]=$(run_infernum_decode "${GGUF_F32}")
            log "[${step}/${total}] GGUF F32 — infernum decode (graph)"
            decode_graph["GGUF F32"]=$(run_infernum_graph_decode "${BASE_MODEL_DIR}")
            log "[${step}/${total}] GGUF F32 — infernum prefill (graph)"
            prefill_graph["GGUF F32"]=$(run_infernum_prefill "${BASE_MODEL_DIR}")
            prefill_eager["GGUF F32"]="—"
            ;;
        "GGUF Q8_0")
            log "[${step}/${total}] GGUF Q8_0 — llama.cpp decode"
            decode_llama["GGUF Q8_0"]=$(run_llama_bench_decode "${GGUF_Q8}")
            log "[${step}/${total}] GGUF Q8_0 — llama.cpp prefill"
            prefill_llama["GGUF Q8_0"]=$(run_llama_bench_prefill "${GGUF_Q8}")
            log "[${step}/${total}] GGUF Q8_0 — infernum decode (eager)"
            decode_eager["GGUF Q8_0"]=$(run_infernum_decode "${GGUF_Q8}")
            log "[${step}/${total}] GGUF Q8_0 — infernum decode (graph)"
            decode_graph["GGUF Q8_0"]=$(run_infernum_graph_decode "${GGUF_Q8}")
            log "[${step}/${total}] GGUF Q8_0 — infernum prefill (graph)"
            prefill_graph["GGUF Q8_0"]=$(run_infernum_prefill "${GGUF_Q8}")
            prefill_eager["GGUF Q8_0"]="—"
            ;;
        "GGUF Q4_0")
            log "[${step}/${total}] GGUF Q4_0 — llama.cpp decode"
            decode_llama["GGUF Q4_0"]=$(run_llama_bench_decode "${GGUF_Q4}")
            log "[${step}/${total}] GGUF Q4_0 — llama.cpp prefill"
            prefill_llama["GGUF Q4_0"]=$(run_llama_bench_prefill "${GGUF_Q4}")
            log "[${step}/${total}] GGUF Q4_0 — infernum decode (eager)"
            decode_eager["GGUF Q4_0"]=$(run_infernum_decode "${GGUF_Q4}")
            log "[${step}/${total}] GGUF Q4_0 — infernum decode (graph)"
            decode_graph["GGUF Q4_0"]=$(run_infernum_graph_decode "${GGUF_Q4}")
            log "[${step}/${total}] GGUF Q4_0 — infernum prefill (graph)"
            prefill_graph["GGUF Q4_0"]=$(run_infernum_prefill "${GGUF_Q4}")
            prefill_eager["GGUF Q4_0"]="—"
            ;;
    esac
done

log ""
log "Done. Results:"
log ""

# ── Output tables ─────────────────────────────────────────────────────────────

compute_ratio() {
    local llama="$1"
    local infernum="$2"
    if [[ "${llama}" == "—" || "${infernum}" == "—" || "${llama}" == "ERR" || "${infernum}" == "ERR" ]]; then
        echo "—"
        return
    fi
    python3 -c "
l, i = float('${llama}'), float('${infernum}')
if l == 0 or i == 0:
    print('—')
else:
    ratio = i / l
    if ratio >= 1:
        print(f'{ratio:.2f}x')
    else:
        print(f'{ratio:.2f}x')
"
}

cpu_name=$(lscpu 2>/dev/null | grep "Model name" | sed 's/.*:\s*//' || echo "unknown")
if [[ -z "${cpu_name}" || "${cpu_name}" == "unknown" ]]; then
    cpu_name=$(uname -m)
fi
infernum_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
llama_commit="unknown"
if [[ -d "${LLAMA_CPP}/.git" ]]; then
    llama_commit=$(git -C "${LLAMA_CPP}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
fi

echo ""
echo "## Infernum vs llama.cpp — ${MODEL_NAME} CPU throughput"
echo ""
echo "- **CPU:** ${cpu_name} (${effective_threads} threads)"
echo "- **Tokens:** ${N_TOKENS}"
echo "- **infernum:** commit \`${infernum_commit}\`"
echo "- **llama.cpp:** commit \`${llama_commit}\`" '(`-ngl 0`,' "${LLAMA_BENCH_REPS} reps)"
echo "- **Date:** $(date +%Y-%m-%d)"

# Decode table
echo ""
echo "### Decode throughput (tok/s)"
echo ""
echo "Autoregressive generation: each step processes 1 token, appending to KV cache."
echo ""
echo "| Format | llama.cpp | eager | graph | ratio |"
echo "| ------ | --------: | ----: | ----: | ----: |"

best_of() {
    local a="$1" b="$2"
    if [[ "${a}" == "—" || "${a}" == "ERR" ]]; then echo "${b}"; return; fi
    if [[ "${b}" == "—" || "${b}" == "ERR" ]]; then echo "${a}"; return; fi
    python3 -c "print(max(float('${a}'), float('${b}')))"
}

for bench in "${benchmarks[@]}"; do
    l="${decode_llama[$bench]}"
    e="${decode_eager[$bench]}"
    g="${decode_graph[$bench]}"
    best=$(best_of "${e}" "${g}")
    r=$(compute_ratio "${l}" "${best}")
    printf "| %-14s | %9s | %5s | %5s | %5s |\n" "${bench}" "${l}" "${e}" "${g}" "${r}"
done

# Prefill table
echo ""
echo "### Prefill throughput (tok/s)"
echo ""
echo "Prompt processing: all tokens processed in a single forward pass."
echo ""
echo "| Format | llama.cpp | eager | graph | ratio |"
echo "| ------ | --------: | ----: | ----: | ----: |"

for bench in "${benchmarks[@]}"; do
    l="${prefill_llama[$bench]}"
    e="${prefill_eager[$bench]}"
    g="${prefill_graph[$bench]}"
    r=$(compute_ratio "${l}" "${g}")
    printf "| %-14s | %9s | %5s | %5s | %5s |
" "${bench}" "${l}" "${e}" "${g}" "${r}"
done

echo ""
