#!/usr/bin/env bash
# bench_cuda_large.sh — multi-GPU CUDA throughput for large models
#
# Benchmarks models at TP=2 and TP=8 on a multi-GPU node (A100 ×8 / H100 ×8).
# Single-GPU baselines live in bench_cuda_small.sh / bench_comparison.sh.
#
# Model coverage:
#   Llama-3.1-70B  Q4_0    GGUF   ~40 GB   always downloaded
#   Qwen3-72B      Q4_K_M  GGUF   ~42 GB   always downloaded
#   Qwen3-235B-A22B Q4_K_M GGUF  ~142 GB   downloaded with --download-large
#
# Output: one markdown table with TP=2 and TP=8 columns for each model.
#
# Prerequisites:
#   - 2+ NVIDIA GPUs, nvidia-smi, NCCL
#   - Rust toolchain (cargo), CUDA toolkit
#   - hf CLI: pip install --break-system-packages huggingface_hub
#
# Usage:
#   ./bench_cuda_large.sh                        # 70B + 72B (always cached)
#   ./bench_cuda_large.sh --download-large       # also download 235B
#   ./bench_cuda_large.sh --n-tokens 200         # override decode length
#   ./bench_cuda_large.sh --dry-run              # show plan without running
#   ./bench_cuda_large.sh --models 70b,235b      # specific models only

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_CACHE="$HOME/.cache/infernum/models"
N_TOKENS=200
DRY_RUN=false
DOWNLOAD_LARGE=false
MODELS_FILTER="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=true; shift ;;
        --download-large) DOWNLOAD_LARGE=true; shift ;;
        --n-tokens)       N_TOKENS="$2"; shift 2 ;;
        --n-tokens=*)     N_TOKENS="${1#--n-tokens=}"; shift ;;
        --models)         MODELS_FILTER="$2"; shift 2 ;;
        --models=*)       MODELS_FILTER="${1#--models=}"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "  --download-large   Auto-download Qwen3-235B-A22B (~142 GB)"
            echo "  --n-tokens N       Decode token count (default: 200)"
            echo "  --models LIST      Comma-separated: 70b, 72b, 235b (default: all cached)"
            echo "  --dry-run          Show plan without running benchmarks"
            exit 0 ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
    esac
done

# ── Model paths ───────────────────────────────────────────────────────────────

LLAMA70B_GGUF="${MODEL_CACHE}/llama-3.1-70b-gguf/Meta-Llama-3.1-70B-Instruct.Q4_0.gguf"
QWEN72B_GGUF="${MODEL_CACHE}/Qwen/Qwen3-72B-GGUF/Qwen3-72B-Instruct.Q4_K_M.gguf"
QWEN235B_DIR="${MODEL_CACHE}/Qwen/Qwen3-235B-A22B-GGUF/Q4_K_M"
# first shard; infernum discovers the rest via the GGUF split-file header
QWEN235B_GGUF="${QWEN235B_DIR}/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf"

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { echo ">>> $*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

extract_toks() {
    grep -oP '[\d.]+(?= tok/s)' | tail -1
}

run_bench() {
    local model_path="$1" gpus="$2"
    $DRY_RUN && { echo "—"; return; }
    local out
    out=$(timeout 900 cargo run --release --example bench --features nccl -q -- \
        --cuda-graph-engine --gpus "${gpus}" "${model_path}" "${N_TOKENS}" 2>&1 || true)
    local toks; toks=$(echo "${out}" | extract_toks)
    echo "${toks:-ERR}"
}

model_enabled() {
    local key="$1"
    [[ "${MODELS_FILTER}" == "all" ]] || \
        echo ",${MODELS_FILTER}," | grep -qi ",${key},"
}

# ── Preflight ─────────────────────────────────────────────────────────────────

command -v cargo >/dev/null 2>&1 || die "cargo not found"
$DRY_RUN || command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"

gpu_count=0
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
fi
[[ "${gpu_count}" -ge 2 ]] || $DRY_RUN || die "Need at least 2 GPUs (found ${gpu_count})"
log "Detected ${gpu_count} GPU(s)"

# ── Model download ────────────────────────────────────────────────────────────

if model_enabled "235b"; then
    if [[ ! -f "${QWEN235B_GGUF}" ]]; then
        if $DOWNLOAD_LARGE; then
            log "Downloading Qwen3-235B-A22B Q4_K_M (~142 GB)..."
            $DRY_RUN || {
                mkdir -p "${QWEN235B_DIR}"
                command -v hf >/dev/null 2>&1 || die "hf not found — pip install --break-system-packages huggingface_hub"
                hf download Qwen/Qwen3-235B-A22B-GGUF \
                    --local-dir "$(dirname "${QWEN235B_DIR}")" \
                    --include "Q4_K_M/*" --quiet
            }
        else
            log "Qwen3-235B-A22B not cached — use --download-large to download"
        fi
    fi
fi

# ── Build ─────────────────────────────────────────────────────────────────────

log "Building infernum bench (NCCL, release)..."
$DRY_RUN || cargo build --release --example bench --features nccl -q

# ── Benchmarks ────────────────────────────────────────────────────────────────

log ""
log "Running multi-GPU benchmarks (${N_TOKENS} decode tokens, greedy argmax)..."
log "Each run: TP=2 then TP=8 — GPUs idle between runs."
log "──────────────────────────────────────────────────"

declare -A R2 R8  # tok/s at TP=2 and TP=8

if model_enabled "70b" && [[ -f "${LLAMA70B_GGUF}" ]]; then
    log "[Llama-3.1-70B Q4_0] TP=2"
    R2["llama70b"]=$(run_bench "${LLAMA70B_GGUF}" 2)
    log "[Llama-3.1-70B Q4_0] TP=8"
    R8["llama70b"]=$(run_bench "${LLAMA70B_GGUF}" 8)
elif model_enabled "70b"; then
    log "Llama-3.1-70B not found at ${LLAMA70B_GGUF} — skipping"
fi

if model_enabled "72b" && [[ -f "${QWEN72B_GGUF}" ]]; then
    log "[Qwen3-72B Q4_K_M] TP=2"
    R2["qwen72b"]=$(run_bench "${QWEN72B_GGUF}" 2)
    log "[Qwen3-72B Q4_K_M] TP=8"
    R8["qwen72b"]=$(run_bench "${QWEN72B_GGUF}" 8)
elif model_enabled "72b"; then
    log "Qwen3-72B not found at ${QWEN72B_GGUF} — skipping"
fi

if model_enabled "235b" && [[ -f "${QWEN235B_GGUF}" ]]; then
    log "[Qwen3-235B-A22B Q4_K_M] TP=2"
    R2["qwen235b"]=$(run_bench "${QWEN235B_GGUF}" 2)
    log "[Qwen3-235B-A22B Q4_K_M] TP=8"
    R8["qwen235b"]=$(run_bench "${QWEN235B_GGUF}" 8)
elif model_enabled "235b"; then
    log "Qwen3-235B-A22B not found (use --download-large) — skipping"
fi

log ""
log "Done."

# ── Output ────────────────────────────────────────────────────────────────────

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "?")
infernum_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo ""
echo "### infernum multi-GPU decode (greedy argmax, ${N_TOKENS} tokens)"
echo ""
echo "- **Node:** ${gpu_count}× ${gpu_name} (${gpu_mem} each)"
echo "- **infernum commit:** \`${infernum_commit}\`"
echo "- **Date:** $(date +%Y-%m-%d)"
echo "- **TP>1 note:** NCCL AllReduce blocks CUDA graph capture; TP>1 runs eager."
echo ""
echo "| Model | Format | 2 GPUs (tok/s) | 8 GPUs (tok/s) |"
echo "| ----- | ------ | -------------: | -------------: |"

[[ -v R2[llama70b]  ]] && printf "| Llama-3.1-70B        | Q4_0    | %14s | %14s |\n" \
    "${R2[llama70b]}"  "${R8[llama70b]:-—}"
[[ -v R2[qwen72b]   ]] && printf "| Qwen3-72B            | Q4_K_M  | %14s | %14s |\n" \
    "${R2[qwen72b]}"   "${R8[qwen72b]:-—}"
[[ -v R2[qwen235b]  ]] && printf "| Qwen3-235B-A22B      | Q4_K_M  | %14s | %14s |\n" \
    "${R2[qwen235b]}"  "${R8[qwen235b]:-—}"

echo ""
