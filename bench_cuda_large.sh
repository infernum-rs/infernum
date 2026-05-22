#!/usr/bin/env bash
# bench_cuda_large.sh — CUDA throughput for large models (H100 node)
#
# Targets a multi-GPU H100 node. Benchmarks models that don't fit or aren't
# interesting on a single L4 (< 24 GB VRAM).
#
# Tiers:
#   Standard (always run): 8B models per family — these fit on a single H100
#     and establish a per-family baseline at a size that's more meaningful than 1B.
#
#   Large (opt-in with --download-large): 70B+ models.
#     Downloads are 50–150 GB each. Pass --download-large to auto-download.
#     Without the flag, large models are only benchmarked if already cached.
#
# NOTE: Multi-GPU tensor-parallel benchmarking (70B+ in BF16) is not yet
# supported by the bench binary. A bench_parallel example is planned. Until
# then, 70B+ models run on single-GPU via quantization (GGUF Q4 fits in ~40 GB).
#
# Prerequisites (required):
#   - NVIDIA GPUs (H100 recommended), nvidia-smi
#   - CUDA toolkit + Rust toolchain (cargo)
#   - hf CLI: pip install 'huggingface_hub[cli]'
#
# Usage:
#   ./bench_cuda_large.sh                          # Standard models only
#   ./bench_cuda_large.sh --download-large         # Also download 70B+ models
#   ./bench_cuda_large.sh --dry-run                # Show plan without running
#   ./bench_cuda_large.sh --n-tokens 128           # Override token count
#   ./bench_cuda_large.sh --families llama,qwen    # Only specific families

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_CACHE="$HOME/.cache/infernum/models"
GGUF_DIR="/tmp"

N_TOKENS=256
N_PREFILL=512
DRY_RUN=false
DOWNLOAD_LARGE=false
FAMILIES_FILTER="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=true; shift ;;
        --download-large) DOWNLOAD_LARGE=true; shift ;;
        --n-tokens)       N_TOKENS="$2"; shift 2 ;;
        --n-tokens=*)     N_TOKENS="${1#--n-tokens=}"; shift ;;
        --families)       FAMILIES_FILTER="$2"; shift 2 ;;
        --families=*)     FAMILIES_FILTER="${1#--families=}"; shift ;;
        -h|--help)
            echo "Usage: $0 [--download-large] [--dry-run] [--n-tokens N] [--families LIST]"
            echo ""
            echo "  --download-large  Auto-download 70B+ models (50–150 GB each)"
            echo "  --families LIST   Comma-separated: llama, qwen, gemma, deepseek (default: all)"
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

# Model paths — standard (8B)
LLAMA8B_DIR="${MODEL_CACHE}/meta-llama/Llama-3.1-8B"
QWEN8B_DIR="${MODEL_CACHE}/Qwen/Qwen3-8B"
GEMMA27B_DIR="${MODEL_CACHE}/google/gemma-3-27b"
DEEPSEEK_TINY_DIR="${MODEL_CACHE}/yujiepan/deepseek-v3-tiny-random"

# Model paths — large (70B+, single-GPU via GGUF Q4, ~40 GB each)
LLAMA70B_DIR="${MODEL_CACHE}/meta-llama/Llama-3.1-70B"
LLAMA70B_GGUF_Q4="${GGUF_DIR}/llama-3.1-70b-q4_0.gguf"
QWEN72B_DIR="${MODEL_CACHE}/Qwen/Qwen3-72B"
QWEN72B_GGUF_Q4="${GGUF_DIR}/qwen3-72b-q4_0.gguf"

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

# Try to download a large model; skip gracefully if it fails (auth, disk, etc.)
try_download_hf_model() {
    local repo_id="$1" dest_dir="$2"
    [[ -f "${dest_dir}/config.json" ]] && { log "Already cached: ${dest_dir}"; return 0; }
    $DOWNLOAD_LARGE || { log "SKIP (use --download-large to download): ${repo_id}"; return 0; }
    log "Downloading large model ${repo_id} (~50–150 GB) → ${dest_dir}"
    $DRY_RUN && return 0
    mkdir -p "${dest_dir}"
    if ! hf download "${repo_id}" --local-dir "${dest_dir}" --quiet 2>/dev/null; then
        log "WARNING: Failed to download ${repo_id} (check auth with: hf login)"
        return 0
    fi
}

extract_toks() { grep -oP '[\d.]+(?= tok/s)' | tail -1; }

run_infernum_decode() {
    local model_path="$1"
    $DRY_RUN && { echo "—"; return; }
    local out
    out=$(timeout 1200 cargo run --release --example bench --features cuda -q -- \
        --cuda-graph-engine "${model_path}" "${N_TOKENS}" 2>/dev/null || true)
    local toks; toks=$(echo "${out}" | extract_toks)
    echo "${toks:-ERR}"
}

run_infernum_decode_eager() {
    local model_path="$1"
    $DRY_RUN && { echo "—"; return; }
    local out
    out=$(timeout 1200 cargo run --release --example bench --features cuda -q -- \
        "${model_path}" "${N_TOKENS}" 2>/dev/null || true)
    local toks; toks=$(echo "${out}" | extract_toks)
    echo "${toks:-ERR}"
}

run_infernum_prefill() {
    local model_path="$1"
    $DRY_RUN && { echo "—"; return; }
    local out
    out=$(timeout 1200 cargo run --release --example bench --features cuda -q -- \
        --graph "${model_path}" "${N_PREFILL}" 2>/dev/null || true)
    local toks; toks=$(echo "${out}" | extract_toks)
    echo "${toks:-ERR}"
}

# ── Preflight ─────────────────────────────────────────────────────────────────

command -v cargo >/dev/null 2>&1 || die "cargo not found"
command -v hf    >/dev/null 2>&1 || die "hf not found — pip install 'huggingface_hub[cli]'"
$DRY_RUN || command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"

if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
else
    gpu_count=0
fi
[[ "${gpu_count}" -gt 0 ]] || $DRY_RUN || die "No GPUs detected"
log "Detected ${gpu_count} GPU(s)"

# ── Model preparation ─────────────────────────────────────────────────────────

log "Preparing models..."
$DRY_RUN && log "(dry-run: skipping downloads)"

[[ -n "${ENABLED_FAMILIES[llama]:-}" ]] && {
    download_hf_model "meta-llama/Llama-3.1-8B" "${LLAMA8B_DIR}" || \
        log "WARNING: Llama-3.1-8B requires HuggingFace auth (hf login)."
    try_download_hf_model "meta-llama/Llama-3.1-70B" "${LLAMA70B_DIR}"
}

[[ -n "${ENABLED_FAMILIES[qwen]:-}" ]] && {
    download_hf_model "Qwen/Qwen3-8B" "${QWEN8B_DIR}"
    try_download_hf_model "Qwen/Qwen3-72B" "${QWEN72B_DIR}"
}

[[ -n "${ENABLED_FAMILIES[gemma]:-}" ]] && {
    download_hf_model "google/gemma-3-27b" "${GEMMA27B_DIR}" || \
        log "WARNING: Gemma-3-27B requires HuggingFace auth (hf login)."
}

[[ -n "${ENABLED_FAMILIES[deepseek]:-}" ]] && {
    download_hf_model "yujiepan/deepseek-v3-tiny-random" "${DEEPSEEK_TINY_DIR}"
}

# ── Build ──────────────────────────────────────────────────────────────────────

log "Building infernum bench (CUDA, release)..."
$DRY_RUN || cargo build --release --example bench --features cuda -q 2>/dev/null

# ── Run benchmarks ────────────────────────────────────────────────────────────

log ""
log "Running benchmarks (${N_TOKENS} decode / ${N_PREFILL} prefill tokens)..."
log "─────────────────────────────────────────────"

declare -A D_INFNUM P_INFNUM NOTES

# Standard 8B models (single H100)
if [[ -n "${ENABLED_FAMILIES[llama]:-}" ]] && [[ -d "${LLAMA8B_DIR}" ]]; then
    log "[Llama] Llama-3.1-8B BF16 decode"
    D_INFNUM["Llama/Llama-3.1-8B BF16"]=$(run_infernum_decode "${LLAMA8B_DIR}")
    log "[Llama] Llama-3.1-8B BF16 prefill"
    P_INFNUM["Llama/Llama-3.1-8B BF16"]=$(run_infernum_prefill "${LLAMA8B_DIR}")
fi

if [[ -n "${ENABLED_FAMILIES[qwen]:-}" ]] && [[ -d "${QWEN8B_DIR}" ]]; then
    log "[Qwen] Qwen3-8B BF16 decode"
    D_INFNUM["Qwen/Qwen3-8B BF16"]=$(run_infernum_decode "${QWEN8B_DIR}")
    log "[Qwen] Qwen3-8B BF16 prefill"
    P_INFNUM["Qwen/Qwen3-8B BF16"]=$(run_infernum_prefill "${QWEN8B_DIR}")
fi

if [[ -n "${ENABLED_FAMILIES[gemma]:-}" ]] && [[ -d "${GEMMA27B_DIR}" ]]; then
    log "[Gemma] Gemma-3-27B BF16 decode"
    D_INFNUM["Gemma/Gemma-3-27B BF16"]=$(run_infernum_decode "${GEMMA27B_DIR}")
    log "[Gemma] Gemma-3-27B BF16 prefill"
    P_INFNUM["Gemma/Gemma-3-27B BF16"]=$(run_infernum_prefill "${GEMMA27B_DIR}")
fi

if [[ -n "${ENABLED_FAMILIES[deepseek]:-}" ]] && [[ -d "${DEEPSEEK_TINY_DIR}" ]]; then
    log "[DeepSeek] deepseek-v3-tiny (random weights, smoke test)"
    D_INFNUM["DeepSeek/tiny-random (smoke)"]=$(run_infernum_decode_eager "${DEEPSEEK_TINY_DIR}")
    P_INFNUM["DeepSeek/tiny-random (smoke)"]="—"
    NOTES["DeepSeek/tiny-random (smoke)"]="random weights; no quality check"
fi

# Large models (70B+, single GPU via GGUF Q4 ~40 GB)
if [[ -n "${ENABLED_FAMILIES[llama]:-}" ]] && [[ -d "${LLAMA70B_DIR}" ]]; then
    # GGUF Q4 conversion for single-GPU inference
    if [[ ! -f "${LLAMA70B_GGUF_Q4}" ]]; then
        log "[Llama] Llama-3.1-70B: BF16 available but GGUF Q4 needed for single-GPU"
        log "  Convert with: llama-quantize (requires llama.cpp)"
        log "  Skipping 70B single-GPU benchmark — set LLAMA_CPP to enable conversion"
    else
        log "[Llama] Llama-3.1-70B GGUF Q4 decode (single GPU)"
        D_INFNUM["Llama/Llama-3.1-70B Q4 (1 GPU)"]=$(run_infernum_decode "${LLAMA70B_GGUF_Q4}")
        P_INFNUM["Llama/Llama-3.1-70B Q4 (1 GPU)"]=$(run_infernum_prefill "${LLAMA70B_GGUF_Q4}")
        NOTES["Llama/Llama-3.1-70B Q4 (1 GPU)"]="single GPU; multi-GPU BF16 requires bench_parallel (pending)"
    fi
elif [[ -n "${ENABLED_FAMILIES[llama]:-}" ]]; then
    log "[Llama] Llama-3.1-70B not cached (use --download-large)"
fi

if [[ -n "${ENABLED_FAMILIES[qwen]:-}" ]] && [[ -d "${QWEN72B_DIR}" ]]; then
    if [[ ! -f "${QWEN72B_GGUF_Q4}" ]]; then
        log "[Qwen] Qwen3-72B: GGUF Q4 needed for single-GPU — convert with llama-quantize"
    else
        log "[Qwen] Qwen3-72B GGUF Q4 decode (single GPU)"
        D_INFNUM["Qwen/Qwen3-72B Q4 (1 GPU)"]=$(run_infernum_decode "${QWEN72B_GGUF_Q4}")
        P_INFNUM["Qwen/Qwen3-72B Q4 (1 GPU)"]=$(run_infernum_prefill "${QWEN72B_GGUF_Q4}")
        NOTES["Qwen/Qwen3-72B Q4 (1 GPU)"]="single GPU; multi-GPU BF16 requires bench_parallel (pending)"
    fi
elif [[ -n "${ENABLED_FAMILIES[qwen]:-}" ]]; then
    log "[Qwen] Qwen3-72B not cached (use --download-large)"
fi

log ""
log "Done."

# ── Output ────────────────────────────────────────────────────────────────────

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "?")
infernum_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo ""
echo "## Infernum CUDA (H100 / large models) — all model families"
echo ""
echo "- **GPU:** ${gpu_name} × ${gpu_count} (${gpu_mem} each)"
echo "- **Decode tokens:** ${N_TOKENS}  |  **Prefill tokens:** ${N_PREFILL}"
echo "- **infernum commit:** \`${infernum_commit}\`"
echo "- **Date:** $(date +%Y-%m-%d)"
echo "- **Multi-GPU note:** 70B+ BF16 with tensor parallelism requires a"
echo "  \`bench_parallel\` example (not yet implemented). The 70B rows use"
echo "  single-GPU GGUF Q4 (~40 GB) as a proxy until that lands."

row_order=(
    "Llama/Llama-3.1-8B BF16"
    "Qwen/Qwen3-8B BF16"
    "Gemma/Gemma-3-27B BF16"
    "DeepSeek/tiny-random (smoke)"
    "Llama/Llama-3.1-70B Q4 (1 GPU)"
    "Qwen/Qwen3-72B Q4 (1 GPU)"
)

echo ""
echo "### Decode throughput (tok/s)"
echo ""
echo "| Model | infernum | notes |"
echo "| ----- | -------: | ----- |"
for key in "${row_order[@]}"; do
    [[ -v D_INFNUM["${key}"] ]] || continue
    inf="${D_INFNUM[${key}]}"
    note="${NOTES[${key}]:-}"
    printf "| %-38s | %8s | %s |\n" "${key}" "${inf}" "${note}"
done

echo ""
echo "### Prefill throughput (tok/s)"
echo ""
echo "| Model | infernum | notes |"
echo "| ----- | -------: | ----- |"
for key in "${row_order[@]}"; do
    [[ -v P_INFNUM["${key}"] ]] || continue
    inf="${P_INFNUM[${key}]}"
    note="${NOTES[${key}]:-}"
    printf "| %-38s | %8s | %s |\n" "${key}" "${inf}" "${note}"
done

echo ""
