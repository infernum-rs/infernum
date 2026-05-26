#!/usr/bin/env bash
# bench_cpu.sh — CPU throughput across all supported model families
#
# Measures decode and prefill throughput for Llama, Qwen, and Gemma families.
# llama.cpp comparison is optional — if LLAMA_CPP is not set and the tool is
# not found, the llama.cpp column shows "—".
#
# Prerequisites (required):
#   - Rust toolchain (cargo)
#   - hf CLI: pip install 'huggingface_hub[cli]'
#
# Prerequisites (optional, for llama.cpp comparison):
#   - llama.cpp built locally (llama-bench, llama-quantize, convert_hf_to_gguf.py)
#     Set: LLAMA_CPP=/path/to/llama.cpp  or  --llama-cpp /path/to/llama.cpp
#
# Usage:
#   ./bench_cpu.sh                        # Run all families
#   ./bench_cpu.sh --dry-run              # Show plan without running
#   ./bench_cpu.sh --threads 8            # Fix thread count
#   ./bench_cpu.sh --n-tokens 128         # Override token count
#   ./bench_cpu.sh --llama-cpp ~/llama.cpp  # Enable llama.cpp comparison
#   ./bench_cpu.sh --families llama,qwen  # Only specific families

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

LLAMA_CPP="${LLAMA_CPP:-}"
MODEL_CACHE="$HOME/.cache/infernum/models"
GGUF_DIR="/tmp"

N_TOKENS=256
N_PREFILL=512
LLAMA_BENCH_REPS=3
THREADS=""
DRY_RUN=false
FAMILIES_FILTER="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=true; shift ;;
        --threads)      THREADS="$2"; shift 2 ;;
        --threads=*)    THREADS="${1#--threads=}"; shift ;;
        --n-tokens)     N_TOKENS="$2"; shift 2 ;;
        --n-tokens=*)   N_TOKENS="${1#--n-tokens=}"; shift ;;
        --llama-cpp)    LLAMA_CPP="$2"; shift 2 ;;
        --llama-cpp=*)  LLAMA_CPP="${1#--llama-cpp=}"; shift ;;
        --families)     FAMILIES_FILTER="$2"; shift 2 ;;
        --families=*)   FAMILIES_FILTER="${1#--families=}"; shift ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--threads N] [--n-tokens N] [--llama-cpp PATH] [--families LIST]"
            echo ""
            echo "  --families LIST   Comma-separated: llama, qwen, gemma (default: all)"
            echo "  --llama-cpp PATH  Enable llama.cpp comparison (or set \$LLAMA_CPP)"
            exit 0 ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
    esac
done

# Parse family filter
declare -A ENABLED_FAMILIES
if [[ "${FAMILIES_FILTER,,}" == "all" ]]; then
    ENABLED_FAMILIES[llama]=1; ENABLED_FAMILIES[qwen]=1; ENABLED_FAMILIES[gemma]=1
else
    IFS=',' read -ra _fams <<< "${FAMILIES_FILTER,,}"
    for f in "${_fams[@]}"; do
        f=$(echo "$f" | xargs)
        case "$f" in
            llama|qwen|gemma) ENABLED_FAMILIES[$f]=1 ;;
            *) echo "ERROR: Unknown family '$f'. Valid: llama, qwen, gemma" >&2; exit 1 ;;
        esac
    done
fi

# Resolve llama.cpp tools
LLAMA_BENCH=""
LLAMA_QUANTIZE=""
CONVERT_SCRIPT=""
HAS_LLAMA_CPP=false
if [[ -n "${LLAMA_CPP}" ]]; then
    LLAMA_BENCH="${LLAMA_CPP}/build/bin/llama-bench"
    LLAMA_QUANTIZE="${LLAMA_CPP}/build/bin/llama-quantize"
    CONVERT_SCRIPT="${LLAMA_CPP}/convert_hf_to_gguf.py"
    if [[ -x "${LLAMA_BENCH}" && -x "${LLAMA_QUANTIZE}" && -f "${CONVERT_SCRIPT}" ]]; then
        HAS_LLAMA_CPP=true
    else
        echo ">>> WARNING: --llama-cpp set but tools not found at ${LLAMA_CPP} — comparison skipped" >&2
    fi
fi

# Model paths
SMOLLM_DIR="${MODEL_CACHE}/HuggingFaceTB/SmolLM2-360M"
SMOLLM_Q4="${MODEL_CACHE}/bartowski/SmolLM2-360M-Instruct-GGUF/SmolLM2-360M-Instruct-Q4_0.gguf"
SMOLLM_Q8="${MODEL_CACHE}/bartowski/SmolLM2-360M-Instruct-GGUF/SmolLM2-360M-Instruct-Q8_0.gguf"
SMOLLM_F32="${GGUF_DIR}/smollm2-360m-f32.gguf"

QWEN_DIR="${MODEL_CACHE}/Qwen/Qwen2.5-0.5B"

GEMMA_GGUF_Q8="${MODEL_CACHE}/bartowski/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q8_0.gguf"

effective_threads="${THREADS:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo '?')}"

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { echo ">>> $*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

download_hf_model() {
    local repo_id="$1" dest_dir="$2"
    if [[ -f "${dest_dir}/config.json" ]]; then
        log "Already cached: ${dest_dir}"
        return 0
    fi
    log "Downloading ${repo_id} → ${dest_dir}"
    $DRY_RUN && return 0
    mkdir -p "${dest_dir}"
    if ! hf download "${repo_id}" "model.safetensors" --local-dir "${dest_dir}" --quiet 2>/dev/null; then
        hf download "${repo_id}" --local-dir "${dest_dir}" --quiet \
            --include "model.safetensors.index.json" "model-*.safetensors" 2>/dev/null || true
    fi
    for f in config.json tokenizer.json tokenizer_config.json; do
        [[ -f "${dest_dir}/${f}" ]] || \
            hf download "${repo_id}" "${f}" --local-dir "${dest_dir}" --quiet 2>/dev/null || true
    done
    [[ -f "${dest_dir}/config.json" ]] || die "Failed to download ${repo_id}"
}

download_hf_file() {
    local repo_id="$1" filename="$2" dest="$3"
    [[ -f "${dest}" ]] && { log "Already cached: ${dest}"; return 0; }
    log "Downloading ${repo_id}/${filename} → ${dest}"
    $DRY_RUN && return 0
    hf download "${repo_id}" "${filename}" --local-dir "$(dirname "${dest}")" --quiet 2>/dev/null
    # hf CLI puts file in a subdirectory matching repo structure; move it
    local downloaded
    downloaded=$(find "$(dirname "${dest}")" -name "${filename}" | head -1)
    if [[ -n "${downloaded}" && "${downloaded}" != "${dest}" ]]; then
        mv "${downloaded}" "${dest}"
    fi
    [[ -f "${dest}" ]] || die "Failed to download ${repo_id}/${filename}"
}

convert_to_gguf() {
    local model_dir="$1" output="$2" outtype="$3"
    [[ -f "${output}" ]] && { log "Already exists: ${output}"; return 0; }
    $HAS_LLAMA_CPP || { log "Skipping GGUF conversion (no llama.cpp)"; return 0; }
    log "Converting → ${output} (${outtype})"
    $DRY_RUN && return 0
    python3 "${CONVERT_SCRIPT}" "${model_dir}" --outfile "${output}" --outtype "${outtype}" 2>/dev/null \
        || { log "WARNING: GGUF conversion failed (missing python deps?) — ${output} skipped"; return 0; }
}

quantize_gguf() {
    local input="$1" output="$2" qtype="$3"
    [[ -f "${output}" ]] && { log "Already exists: ${output}"; return 0; }
    $HAS_LLAMA_CPP || { log "Skipping quantization (no llama.cpp)"; return 0; }
    [[ -f "${input}" ]] || { log "Source GGUF missing, skipping ${output}"; return 0; }
    log "Quantizing → ${output} (${qtype})"
    $DRY_RUN && return 0
    "${LLAMA_QUANTIZE}" "${input}" "${output}" "${qtype}" >/dev/null 2>&1
}

extract_toks() {
    grep -oP '[\d.]+(?= tok/s)' | tail -1
}

run_infernum_decode() {
    local model_path="$1"
    $DRY_RUN && { echo "—"; return; }
    local thread_args=()
    [[ -n "${THREADS}" ]] && thread_args=(-j "${THREADS}")
    local out
    out=$(timeout 600 cargo run --release --example bench_cpu --features cpu -q -- \
        "${model_path}" "${N_TOKENS}" "${thread_args[@]}" 2>/dev/null || true)
    local toks; toks=$(echo "${out}" | extract_toks)
    echo "${toks:-ERR}"
}

run_infernum_prefill() {
    local model_path="$1"
    $DRY_RUN && { echo "—"; return; }
    local thread_args=()
    [[ -n "${THREADS}" ]] && thread_args=(-j "${THREADS}")
    local out
    out=$(timeout 600 cargo run --release --example bench_cpu --features cpu -q -- \
        --graph "${model_path}" "${N_PREFILL}" "${thread_args[@]}" 2>/dev/null || true)
    local toks; toks=$(echo "${out}" | extract_toks)
    echo "${toks:-ERR}"
}

run_llama_bench_decode() {
    local gguf="$1"
    $HAS_LLAMA_CPP || { echo "—"; return; }
    $DRY_RUN && { echo "—"; return; }
    local thread_args=()
    [[ -n "${THREADS}" ]] && thread_args=(-t "${THREADS}")
    local json
    json=$("${LLAMA_BENCH}" -m "${gguf}" -p 0 -n "${N_TOKENS}" -r "${LLAMA_BENCH_REPS}" \
        -ngl 0 "${thread_args[@]}" -o jsonl 2>/dev/null)
    echo "${json}" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(round(d["avg_ts"],1))' 2>/dev/null || echo "—"
}

run_llama_bench_prefill() {
    local gguf="$1"
    $HAS_LLAMA_CPP || { echo "—"; return; }
    $DRY_RUN && { echo "—"; return; }
    local thread_args=()
    [[ -n "${THREADS}" ]] && thread_args=(-t "${THREADS}")
    local json
    json=$("${LLAMA_BENCH}" -m "${gguf}" -p "${N_PREFILL}" -n 0 -r "${LLAMA_BENCH_REPS}" \
        -ngl 0 "${thread_args[@]}" -o jsonl 2>/dev/null)
    echo "${json}" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(round(d["avg_ts"],1))' 2>/dev/null || echo "—"
}

ratio() {
    local a="$1" b="$2"
    [[ "${a}" == "—" || "${b}" == "—" || "${a}" == "ERR" || "${b}" == "ERR" ]] && { echo "—"; return; }
    python3 -c "a,b=float('${a}'),float('${b}'); print('—') if a==0 or b==0 else print(f'{b/a:.2f}x')" 2>/dev/null || echo "—"
}

# ── Preflight ─────────────────────────────────────────────────────────────────

command -v cargo >/dev/null 2>&1 || die "cargo not found — install via https://rustup.rs"
command -v hf    >/dev/null 2>&1 || die "hf not found — pip install 'huggingface_hub[cli]'"

# ── Model preparation ─────────────────────────────────────────────────────────

log "Preparing models..."
$DRY_RUN && log "(dry-run: skipping downloads)"

[[ -n "${ENABLED_FAMILIES[llama]:-}" ]] && {
    download_hf_model "HuggingFaceTB/SmolLM2-360M" "${SMOLLM_DIR}"
    download_hf_file "bartowski/SmolLM2-360M-Instruct-GGUF" "SmolLM2-360M-Instruct-Q4_0.gguf" "${SMOLLM_Q4}"
    download_hf_file "bartowski/SmolLM2-360M-Instruct-GGUF" "SmolLM2-360M-Instruct-Q8_0.gguf" "${SMOLLM_Q8}"
    if $HAS_LLAMA_CPP; then
        _f16_smollm="${GGUF_DIR}/smollm2-360m-f16.gguf"
        convert_to_gguf "${SMOLLM_DIR}" "${SMOLLM_F32}" "f32"
    fi
}

[[ -n "${ENABLED_FAMILIES[qwen]:-}" ]] && {
    download_hf_model "Qwen/Qwen2.5-0.5B" "${QWEN_DIR}"
}

[[ -n "${ENABLED_FAMILIES[gemma]:-}" ]] && {
    download_hf_file "bartowski/gemma-2-2b-it-GGUF" "gemma-2-2b-it-Q8_0.gguf" "${GEMMA_GGUF_Q8}"
}

# ── Build ──────────────────────────────────────────────────────────────────────

log "Building infernum bench_cpu (release)..."
$DRY_RUN || cargo build --release --example bench_cpu --features cpu -q 2>/dev/null

# ── Run benchmarks ────────────────────────────────────────────────────────────

log ""
log "Running benchmarks (${N_TOKENS} decode tokens, ${N_PREFILL} prefill tokens, ${effective_threads} threads)..."
log "─────────────────────────────────────────────"

declare -A D_INFNUM D_LLAMA D_RATIO
declare -A P_INFNUM P_LLAMA P_RATIO

# Llama family
if [[ -n "${ENABLED_FAMILIES[llama]:-}" ]]; then
    log "[Llama] SmolLM2-360M decode (SafeTensors)"
    D_INFNUM["Llama/SmolLM2-360M SafeTensors"]=$(run_infernum_decode "${SMOLLM_DIR}")
    D_LLAMA["Llama/SmolLM2-360M SafeTensors"]="—"

    log "[Llama] SmolLM2-360M prefill (SafeTensors)"
    P_INFNUM["Llama/SmolLM2-360M SafeTensors"]=$(run_infernum_prefill "${SMOLLM_DIR}")
    P_LLAMA["Llama/SmolLM2-360M SafeTensors"]="—"

    if [[ -f "${SMOLLM_Q8}" ]]; then
        log "[Llama] SmolLM2-360M GGUF Q8_0 decode"
        D_INFNUM["Llama/SmolLM2-360M Q8_0"]=$(run_infernum_decode "${SMOLLM_Q8}")
        D_LLAMA["Llama/SmolLM2-360M Q8_0"]=$(run_llama_bench_decode "${SMOLLM_Q8}")
        P_INFNUM["Llama/SmolLM2-360M Q8_0"]=$(run_infernum_prefill "${SMOLLM_Q8}")
        P_LLAMA["Llama/SmolLM2-360M Q8_0"]=$(run_llama_bench_prefill "${SMOLLM_Q8}")
    fi

    if [[ -f "${SMOLLM_Q4}" ]]; then
        log "[Llama] SmolLM2-360M GGUF Q4_0 decode"
        D_INFNUM["Llama/SmolLM2-360M Q4_0"]=$(run_infernum_decode "${SMOLLM_Q4}")
        D_LLAMA["Llama/SmolLM2-360M Q4_0"]=$(run_llama_bench_decode "${SMOLLM_Q4}")
        P_INFNUM["Llama/SmolLM2-360M Q4_0"]=$(run_infernum_prefill "${SMOLLM_Q4}")
        P_LLAMA["Llama/SmolLM2-360M Q4_0"]=$(run_llama_bench_prefill "${SMOLLM_Q4}")
    fi

    if [[ -f "${SMOLLM_F32}" ]]; then
        log "[Llama] SmolLM2-360M GGUF F32 decode"
        D_INFNUM["Llama/SmolLM2-360M F32"]=$(run_infernum_decode "${SMOLLM_F32}")
        D_LLAMA["Llama/SmolLM2-360M F32"]=$(run_llama_bench_decode "${SMOLLM_F32}")
        P_INFNUM["Llama/SmolLM2-360M F32"]=$(run_infernum_prefill "${SMOLLM_DIR}")
        P_LLAMA["Llama/SmolLM2-360M F32"]=$(run_llama_bench_prefill "${SMOLLM_F32}")
    fi
fi

# Qwen family
if [[ -n "${ENABLED_FAMILIES[qwen]:-}" ]]; then
    log "[Qwen] Qwen2.5-0.5B decode"
    D_INFNUM["Qwen/Qwen2.5-0.5B F32"]=$(run_infernum_decode "${QWEN_DIR}")
    D_LLAMA["Qwen/Qwen2.5-0.5B F32"]="—"
    log "[Qwen] Qwen2.5-0.5B prefill"
    P_INFNUM["Qwen/Qwen2.5-0.5B F32"]=$(run_infernum_prefill "${QWEN_DIR}")
    P_LLAMA["Qwen/Qwen2.5-0.5B F32"]="—"
fi

# Gemma family
if [[ -n "${ENABLED_FAMILIES[gemma]:-}" ]]; then
    if [[ -f "${GEMMA_GGUF_Q8}" ]]; then
        log "[Gemma] gemma-2-2b-it Q8_0 decode"
        D_INFNUM["Gemma/gemma-2-2b-it Q8_0"]=$(run_infernum_decode "${GEMMA_GGUF_Q8}")
        D_LLAMA["Gemma/gemma-2-2b-it Q8_0"]=$(run_llama_bench_decode "${GEMMA_GGUF_Q8}")
        log "[Gemma] gemma-2-2b-it Q8_0 prefill"
        P_INFNUM["Gemma/gemma-2-2b-it Q8_0"]=$(run_infernum_prefill "${GEMMA_GGUF_Q8}")
        P_LLAMA["Gemma/gemma-2-2b-it Q8_0"]=$(run_llama_bench_prefill "${GEMMA_GGUF_Q8}")
    fi
fi

log ""
log "Done."

# ── Output ────────────────────────────────────────────────────────────────────

cpu_name=$(lscpu 2>/dev/null | grep "Model name" | sed 's/.*:\s*//' | head -1 || uname -m)
infernum_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
llama_commit="unknown"
$HAS_LLAMA_CPP && llama_commit=$(git -C "${LLAMA_CPP}" rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo ""
echo "## Infernum CPU — all model families"
echo ""
echo "- **CPU:** ${cpu_name} (${effective_threads} threads)"
echo "- **Decode tokens:** ${N_TOKENS}  |  **Prefill tokens:** ${N_PREFILL}"
echo "- **infernum commit:** \`${infernum_commit}\`"
$HAS_LLAMA_CPP && echo "- **llama.cpp commit:** \`${llama_commit}\` (\`-ngl 0\`, ${LLAMA_BENCH_REPS} reps)"
echo "- **Date:** $(date +%Y-%m-%d)"

row_order=(
    "Llama/SmolLM2-360M SafeTensors"
    "Llama/SmolLM2-360M Q8_0"
    "Llama/SmolLM2-360M Q4_0"
    "Llama/SmolLM2-360M F32"
    "Qwen/Qwen2.5-0.5B F32"
    "Gemma/gemma-2-2b-it Q8_0"
)

echo ""
echo "### Decode throughput (tok/s)"
echo ""
echo "| Model | infernum | llama.cpp | ratio |"
echo "| ----- | -------: | --------: | ----: |"
for key in "${row_order[@]}"; do
    [[ -v D_INFNUM["${key}"] ]] || continue
    inf="${D_INFNUM[${key}]}"
    lla="${D_LLAMA[${key}]}"
    r=$(ratio "${lla}" "${inf}")
    printf "| %-38s | %8s | %9s | %5s |\n" "${key}" "${inf}" "${lla}" "${r}"
done

echo ""
echo "### Prefill throughput (tok/s)"
echo ""
echo "| Model | infernum | llama.cpp | ratio |"
echo "| ----- | -------: | --------: | ----: |"
for key in "${row_order[@]}"; do
    [[ -v P_INFNUM["${key}"] ]] || continue
    inf="${P_INFNUM[${key}]}"
    lla="${P_LLAMA[${key}]}"
    r=$(ratio "${lla}" "${inf}")
    printf "| %-38s | %8s | %9s | %5s |\n" "${key}" "${inf}" "${lla}" "${r}"
done

echo ""
