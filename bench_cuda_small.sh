#!/usr/bin/env bash
# bench_cuda_small.sh — CUDA throughput across all model families (single GPU, ~24 GB VRAM)
#
# Designed for a single GPU with ~24 GB VRAM (L4 or equivalent).
# Covers Llama, Qwen, and Gemma families. DeepSeek is included as a smoke test
# with tiny random weights (no quality check, just verifies the pipeline runs).
#
# llama.cpp comparison is optional — set LLAMA_CPP or use --llama-cpp.
#
# Prerequisites (required):
#   - NVIDIA GPU with ~24 GB VRAM
#   - CUDA toolkit + nvidia-smi
#   - Rust toolchain (cargo)
#   - hf CLI: pip install 'huggingface_hub[cli]'
#
# Prerequisites (optional):
#   - llama.cpp built with CUDA (cmake -DGGML_CUDA=ON)
#   - python3 with torch, transformers, gguf (for GGUF conversion)
#
# Usage:
#   ./bench_cuda_small.sh                          # Run all families
#   ./bench_cuda_small.sh --dry-run                # Show plan without running
#   ./bench_cuda_small.sh --n-tokens 128           # Override token count
#   ./bench_cuda_small.sh --llama-cpp ~/llama.cpp  # Enable comparison
#   ./bench_cuda_small.sh --families llama,qwen    # Only specific families
#   ./bench_cuda_small.sh --skip-large             # Skip 8B models (faster)

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

LLAMA_CPP="${LLAMA_CPP:-}"
MODEL_CACHE="$HOME/.cache/infernum/models"
GGUF_DIR="/tmp"

N_TOKENS=256
N_PREFILL=512
LLAMA_BENCH_REPS=3
DRY_RUN=false
FAMILIES_FILTER="all"
SKIP_LARGE=false  # skip 8B+ models

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=true; shift ;;
        --n-tokens)     N_TOKENS="$2"; shift 2 ;;
        --n-tokens=*)   N_TOKENS="${1#--n-tokens=}"; shift ;;
        --llama-cpp)    LLAMA_CPP="$2"; shift 2 ;;
        --llama-cpp=*)  LLAMA_CPP="${1#--llama-cpp=}"; shift ;;
        --families)     FAMILIES_FILTER="$2"; shift 2 ;;
        --families=*)   FAMILIES_FILTER="${1#--families=}"; shift ;;
        --skip-large)   SKIP_LARGE=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--n-tokens N] [--llama-cpp PATH] [--families LIST] [--skip-large]"
            echo ""
            echo "  --families LIST   Comma-separated: llama, qwen, gemma, deepseek (default: all)"
            echo "  --skip-large      Skip 8B+ models (faster iteration)"
            echo "  --llama-cpp PATH  Enable llama.cpp comparison (or set \$LLAMA_CPP)"
            exit 0 ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
    esac
done

declare -A ENABLED_FAMILIES
if [[ "${FAMILIES_FILTER,,}" == "all" ]]; then
    ENABLED_FAMILIES[llama]=1; ENABLED_FAMILIES[qwen]=1
    ENABLED_FAMILIES[gemma]=1; ENABLED_FAMILIES[deepseek]=1
else
    IFS=',' read -ra _fams <<< "${FAMILIES_FILTER,,}"
    for f in "${_fams[@]}"; do
        f=$(echo "$f" | xargs)
        case "$f" in
            llama|qwen|gemma|deepseek) ENABLED_FAMILIES[$f]=1 ;;
            *) echo "ERROR: Unknown family '$f'. Valid: llama, qwen, gemma, deepseek" >&2; exit 1 ;;
        esac
    done
fi

LLAMA_BENCH="" LLAMA_QUANTIZE="" CONVERT_SCRIPT=""
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
LLAMA1B_DIR="${MODEL_CACHE}/meta-llama/Llama-3.2-1B"
LLAMA1B_FP8_DIR="${MODEL_CACHE}/RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic"
LLAMA1B_GPTQ_DIR="${MODEL_CACHE}/shuyuej/Llama-3.2-1B-GPTQ"
LLAMA1B_GGUF_Q8="${GGUF_DIR}/llama-3.2-1b-q8_0.gguf"
LLAMA1B_GGUF_Q4="${GGUF_DIR}/llama-3.2-1b-q4_0.gguf"

LLAMA8B_DIR="${MODEL_CACHE}/meta-llama/Llama-3.1-8B"

QWEN_SMALL_DIR="${MODEL_CACHE}/Qwen/Qwen3-0.6B"
QWEN_8B_DIR="${MODEL_CACHE}/Qwen/Qwen3-8B"

GEMMA2B_DIR="${MODEL_CACHE}/unsloth/gemma-2-2b"

DEEPSEEK_TINY_DIR="${MODEL_CACHE}/yujiepan/deepseek-v3-tiny-random"

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { echo ">>> $*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

download_hf_model() {
    local repo_id="$1" dest_dir="$2"
    if [[ -f "${dest_dir}/config.json" ]]; then
        log "Already cached: ${dest_dir}"; return 0
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

# Download only specific files from a repo (for quantized shards etc.)
download_hf_model_files() {
    local repo_id="$1" dest_dir="$2"; shift 2
    local files=("$@")
    if [[ -f "${dest_dir}/config.json" ]]; then
        log "Already cached: ${dest_dir}"; return 0
    fi
    log "Downloading ${repo_id} (selected files) → ${dest_dir}"
    $DRY_RUN && return 0
    mkdir -p "${dest_dir}"
    for f in "${files[@]}"; do
        hf download "${repo_id}" "${f}" --local-dir "${dest_dir}" --quiet 2>/dev/null || true
    done
    [[ -f "${dest_dir}/config.json" ]] || die "Failed to download ${repo_id}"
}

convert_to_gguf() {
    local model_dir="$1" output="$2" outtype="$3"
    [[ -f "${output}" ]] && { log "Already exists: ${output}"; return 0; }
    $HAS_LLAMA_CPP || { log "Skipping GGUF conversion (no llama.cpp)"; return 0; }
    log "Converting → ${output} (${outtype})"
    $DRY_RUN && return 0
    python3 "${CONVERT_SCRIPT}" "${model_dir}" --outfile "${output}" --outtype "${outtype}" 2>/dev/null
}

quantize_gguf() {
    local input="$1" output="$2" qtype="$3"
    [[ -f "${output}" ]] && { log "Already exists: ${output}"; return 0; }
    $HAS_LLAMA_CPP || return 0
    [[ -f "${input}" ]] || return 0
    log "Quantizing → ${output} (${qtype})"
    $DRY_RUN && return 0
    "${LLAMA_QUANTIZE}" "${input}" "${output}" "${qtype}" >/dev/null 2>&1
}

extract_toks() { grep -oP '[\d.]+(?= tok/s)' | tail -1; }

run_infernum_decode() {
    local model_path="$1"
    $DRY_RUN && { echo "—"; return; }
    local out
    out=$(timeout 600 cargo run --release --example bench --features cuda -q -- \
        --cuda-graph-engine "${model_path}" "${N_TOKENS}" 2>/dev/null || true)
    local toks; toks=$(echo "${out}" | extract_toks)
    echo "${toks:-ERR}"
}

# Eager decode path (for families not yet supported by CudaGraphEngine, e.g. DeepSeek)
run_infernum_decode_eager() {
    local model_path="$1"
    $DRY_RUN && { echo "—"; return; }
    local out
    out=$(timeout 600 cargo run --release --example bench --features cuda -q -- \
        "${model_path}" "${N_TOKENS}" 2>/dev/null || true)
    local toks; toks=$(echo "${out}" | extract_toks)
    echo "${toks:-ERR}"
}

run_infernum_prefill() {
    local model_path="$1"
    $DRY_RUN && { echo "—"; return; }
    local out
    out=$(timeout 600 cargo run --release --example bench --features cuda -q -- \
        --graph "${model_path}" "${N_PREFILL}" 2>/dev/null || true)
    local toks; toks=$(echo "${out}" | extract_toks)
    echo "${toks:-ERR}"
}

run_llama_decode() {
    local gguf="$1"
    $HAS_LLAMA_CPP || { echo "—"; return; }
    $DRY_RUN && { echo "—"; return; }
    local json
    json=$("${LLAMA_BENCH}" -m "${gguf}" -p 0 -n "${N_TOKENS}" -r "${LLAMA_BENCH_REPS}" \
        -ngl 99 -o jsonl 2>/dev/null)
    echo "${json}" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(round(d["avg_ts"],1))' 2>/dev/null || echo "—"
}

run_llama_prefill() {
    local gguf="$1"
    $HAS_LLAMA_CPP || { echo "—"; return; }
    $DRY_RUN && { echo "—"; return; }
    local json
    json=$("${LLAMA_BENCH}" -m "${gguf}" -p "${N_PREFILL}" -n 0 -r "${LLAMA_BENCH_REPS}" \
        -ngl 99 -o jsonl 2>/dev/null)
    echo "${json}" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(round(d["avg_ts"],1))' 2>/dev/null || echo "—"
}

ratio() {
    local a="$1" b="$2"
    [[ "${a}" == "—" || "${b}" == "—" || "${a}" == "ERR" || "${b}" == "ERR" ]] && { echo "—"; return; }
    python3 -c "a,b=float('${a}'),float('${b}'); print('—') if a==0 or b==0 else print(f'{b/a:.2f}x')" 2>/dev/null || echo "—"
}

# ── Preflight ─────────────────────────────────────────────────────────────────

command -v cargo >/dev/null 2>&1 || die "cargo not found"
command -v hf    >/dev/null 2>&1 || die "hf not found — pip install 'huggingface_hub[cli]'"
$DRY_RUN || command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found — CUDA driver required"

# ── Model preparation ─────────────────────────────────────────────────────────

log "Preparing models..."
$DRY_RUN && log "(dry-run: skipping downloads)"

if [[ -n "${ENABLED_FAMILIES[llama]:-}" ]]; then
    download_hf_model "meta-llama/Llama-3.2-1B" "${LLAMA1B_DIR}" || \
        log "WARNING: Llama-3.2-1B requires HuggingFace auth (hf login). Skipping."
    download_hf_model "RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic" "${LLAMA1B_FP8_DIR}"
    download_hf_model_files "shuyuej/Llama-3.2-1B-GPTQ" "${LLAMA1B_GPTQ_DIR}" \
        config.json tokenizer.json tokenizer_config.json \
        "model.safetensors" "quantize_config.json"

    if $HAS_LLAMA_CPP && [[ -d "${LLAMA1B_DIR}" ]]; then
        local _f16="${GGUF_DIR}/llama-3.2-1b-f16.gguf"
        convert_to_gguf "${LLAMA1B_DIR}" "${_f16}" "f16"
        quantize_gguf "${_f16}" "${LLAMA1B_GGUF_Q8}" "q8_0"
        quantize_gguf "${_f16}" "${LLAMA1B_GGUF_Q4}" "q4_0"
    fi

    if ! $SKIP_LARGE; then
        download_hf_model "meta-llama/Llama-3.1-8B" "${LLAMA8B_DIR}" || \
            log "WARNING: Llama-3.1-8B requires HuggingFace auth. Skipping."
    fi
fi

if [[ -n "${ENABLED_FAMILIES[qwen]:-}" ]]; then
    download_hf_model "Qwen/Qwen3-0.6B" "${QWEN_SMALL_DIR}"
    $SKIP_LARGE || download_hf_model "Qwen/Qwen3-8B" "${QWEN_8B_DIR}"
fi

if [[ -n "${ENABLED_FAMILIES[gemma]:-}" ]]; then
    download_hf_model "unsloth/gemma-2-2b" "${GEMMA2B_DIR}"
fi

if [[ -n "${ENABLED_FAMILIES[deepseek]:-}" ]]; then
    download_hf_model "yujiepan/deepseek-v3-tiny-random" "${DEEPSEEK_TINY_DIR}"
fi

# ── Build ──────────────────────────────────────────────────────────────────────

log "Building infernum bench (CUDA, release)..."
$DRY_RUN || cargo build --release --example bench --features cuda -q 2>/dev/null

# ── Run benchmarks ────────────────────────────────────────────────────────────

log ""
log "Running benchmarks (${N_TOKENS} decode / ${N_PREFILL} prefill tokens)..."
log "─────────────────────────────────────────────"

declare -A D_INFNUM D_LLAMA P_INFNUM P_LLAMA

# Llama family: 1B in multiple formats
if [[ -n "${ENABLED_FAMILIES[llama]:-}" ]]; then
    if [[ -d "${LLAMA1B_DIR}" ]]; then
        log "[Llama] Llama-3.2-1B BF16 decode"
        D_INFNUM["Llama/Llama-3.2-1B BF16"]=$(run_infernum_decode "${LLAMA1B_DIR}")
        D_LLAMA["Llama/Llama-3.2-1B BF16"]="—"
        log "[Llama] Llama-3.2-1B BF16 prefill"
        P_INFNUM["Llama/Llama-3.2-1B BF16"]=$(run_infernum_prefill "${LLAMA1B_DIR}")
        P_LLAMA["Llama/Llama-3.2-1B BF16"]="—"
    fi

    if [[ -d "${LLAMA1B_FP8_DIR}" ]]; then
        log "[Llama] Llama-3.2-1B FP8 decode"
        D_INFNUM["Llama/Llama-3.2-1B FP8"]=$(run_infernum_decode "${LLAMA1B_FP8_DIR}")
        D_LLAMA["Llama/Llama-3.2-1B FP8"]="—"
        P_INFNUM["Llama/Llama-3.2-1B FP8"]=$(run_infernum_prefill "${LLAMA1B_FP8_DIR}")
        P_LLAMA["Llama/Llama-3.2-1B FP8"]="—"
    fi

    if [[ -d "${LLAMA1B_GPTQ_DIR}" ]]; then
        log "[Llama] Llama-3.2-1B GPTQ INT4 decode"
        D_INFNUM["Llama/Llama-3.2-1B GPTQ"]=$(run_infernum_decode "${LLAMA1B_GPTQ_DIR}")
        D_LLAMA["Llama/Llama-3.2-1B GPTQ"]="—"
        P_INFNUM["Llama/Llama-3.2-1B GPTQ"]=$(run_infernum_prefill "${LLAMA1B_GPTQ_DIR}")
        P_LLAMA["Llama/Llama-3.2-1B GPTQ"]="—"
    fi

    if [[ -f "${LLAMA1B_GGUF_Q8}" ]]; then
        log "[Llama] Llama-3.2-1B GGUF Q8_0 decode"
        D_INFNUM["Llama/Llama-3.2-1B GGUF Q8"]=$(run_infernum_decode "${LLAMA1B_GGUF_Q8}")
        D_LLAMA["Llama/Llama-3.2-1B GGUF Q8"]=$(run_llama_decode "${LLAMA1B_GGUF_Q8}")
        P_INFNUM["Llama/Llama-3.2-1B GGUF Q8"]=$(run_infernum_prefill "${LLAMA1B_GGUF_Q8}")
        P_LLAMA["Llama/Llama-3.2-1B GGUF Q8"]=$(run_llama_prefill "${LLAMA1B_GGUF_Q8}")
    fi

    if [[ -f "${LLAMA1B_GGUF_Q4}" ]]; then
        log "[Llama] Llama-3.2-1B GGUF Q4_0 decode"
        D_INFNUM["Llama/Llama-3.2-1B GGUF Q4"]=$(run_infernum_decode "${LLAMA1B_GGUF_Q4}")
        D_LLAMA["Llama/Llama-3.2-1B GGUF Q4"]=$(run_llama_decode "${LLAMA1B_GGUF_Q4}")
        P_INFNUM["Llama/Llama-3.2-1B GGUF Q4"]=$(run_infernum_prefill "${LLAMA1B_GGUF_Q4}")
        P_LLAMA["Llama/Llama-3.2-1B GGUF Q4"]=$(run_llama_prefill "${LLAMA1B_GGUF_Q4}")
    fi

    if ! $SKIP_LARGE && [[ -d "${LLAMA8B_DIR}" ]]; then
        log "[Llama] Llama-3.1-8B BF16 decode"
        D_INFNUM["Llama/Llama-3.1-8B BF16"]=$(run_infernum_decode "${LLAMA8B_DIR}")
        D_LLAMA["Llama/Llama-3.1-8B BF16"]="—"
        log "[Llama] Llama-3.1-8B BF16 prefill"
        P_INFNUM["Llama/Llama-3.1-8B BF16"]=$(run_infernum_prefill "${LLAMA8B_DIR}")
        P_LLAMA["Llama/Llama-3.1-8B BF16"]="—"
    fi
fi

# Qwen family
if [[ -n "${ENABLED_FAMILIES[qwen]:-}" ]]; then
    if [[ -d "${QWEN_SMALL_DIR}" ]]; then
        log "[Qwen] Qwen3-0.6B BF16 decode"
        D_INFNUM["Qwen/Qwen3-0.6B BF16"]=$(run_infernum_decode "${QWEN_SMALL_DIR}")
        D_LLAMA["Qwen/Qwen3-0.6B BF16"]="—"
        log "[Qwen] Qwen3-0.6B BF16 prefill"
        P_INFNUM["Qwen/Qwen3-0.6B BF16"]=$(run_infernum_prefill "${QWEN_SMALL_DIR}")
        P_LLAMA["Qwen/Qwen3-0.6B BF16"]="—"
    fi

    if ! $SKIP_LARGE && [[ -d "${QWEN_8B_DIR}" ]]; then
        log "[Qwen] Qwen3-8B BF16 decode"
        D_INFNUM["Qwen/Qwen3-8B BF16"]=$(run_infernum_decode "${QWEN_8B_DIR}")
        D_LLAMA["Qwen/Qwen3-8B BF16"]="—"
        log "[Qwen] Qwen3-8B BF16 prefill"
        P_INFNUM["Qwen/Qwen3-8B BF16"]=$(run_infernum_prefill "${QWEN_8B_DIR}")
        P_LLAMA["Qwen/Qwen3-8B BF16"]="—"
    fi
fi

# Gemma family
if [[ -n "${ENABLED_FAMILIES[gemma]:-}" ]]; then
    if [[ -d "${GEMMA2B_DIR}" ]]; then
        log "[Gemma] Gemma-2-2B BF16 decode"
        D_INFNUM["Gemma/Gemma-2-2B BF16"]=$(run_infernum_decode "${GEMMA2B_DIR}")
        D_LLAMA["Gemma/Gemma-2-2B BF16"]="—"
        log "[Gemma] Gemma-2-2B BF16 prefill"
        P_INFNUM["Gemma/Gemma-2-2B BF16"]=$(run_infernum_prefill "${GEMMA2B_DIR}")
        P_LLAMA["Gemma/Gemma-2-2B BF16"]="—"
    fi
fi

# DeepSeek family — smoke test only (random weights, no quality check)
if [[ -n "${ENABLED_FAMILIES[deepseek]:-}" ]]; then
    if [[ -d "${DEEPSEEK_TINY_DIR}" ]]; then
        log "[DeepSeek] deepseek-v3-tiny (random weights, smoke test)"
        D_INFNUM["DeepSeek/tiny-random (smoke)"]=$(run_infernum_decode_eager "${DEEPSEEK_TINY_DIR}")
        D_LLAMA["DeepSeek/tiny-random (smoke)"]="—"
        P_INFNUM["DeepSeek/tiny-random (smoke)"]="—"  # graph prefill not yet supported
        P_LLAMA["DeepSeek/tiny-random (smoke)"]="—"
    fi
fi

log ""
log "Done."

# ── Output ────────────────────────────────────────────────────────────────────

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "?")
infernum_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
llama_commit="unknown"
$HAS_LLAMA_CPP && llama_commit=$(git -C "${LLAMA_CPP}" rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo ""
echo "## Infernum CUDA (single GPU) — all model families"
echo ""
echo "- **GPU:** ${gpu_name} (${gpu_mem})"
echo "- **Decode tokens:** ${N_TOKENS}  |  **Prefill tokens:** ${N_PREFILL}"
echo "- **infernum commit:** \`${infernum_commit}\`"
$HAS_LLAMA_CPP && echo "- **llama.cpp commit:** \`${llama_commit}\` (\`-ngl 99\`, ${LLAMA_BENCH_REPS} reps)"
echo "- **Date:** $(date +%Y-%m-%d)"

row_order=(
    "Llama/Llama-3.2-1B BF16"
    "Llama/Llama-3.2-1B FP8"
    "Llama/Llama-3.2-1B GPTQ"
    "Llama/Llama-3.2-1B GGUF Q8"
    "Llama/Llama-3.2-1B GGUF Q4"
    "Llama/Llama-3.1-8B BF16"
    "Qwen/Qwen3-0.6B BF16"
    "Qwen/Qwen3-8B BF16"
    "Gemma/Gemma-2-2B BF16"
    "DeepSeek/tiny-random (smoke)"
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
