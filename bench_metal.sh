#!/opt/homebrew/bin/bash
# bench_metal.sh — Metal (Apple Silicon) throughput across all supported model families
#
# Measures decode throughput for Llama, Qwen, and Gemma families.
# Prefill is not yet measured (Metal graph engine not yet implemented).
# DeepSeek is not supported on Metal.
#
# llama.cpp comparison is optional — if not found, that column shows "—".
#
# Prerequisites (required):
#   - Apple Silicon Mac
#   - Rust toolchain (cargo)
#   - hf CLI: pip install 'huggingface_hub[cli]'
#
# Prerequisites (optional, for llama.cpp comparison):
#   - llama.cpp built with Metal (cmake -DGGML_METAL=ON)
#     Set: LLAMA_CPP=~/llama.cpp  or  --llama-cpp ~/llama.cpp
#
# Usage:
#   ./bench_metal.sh                          # Run all families
#   ./bench_metal.sh --dry-run                # Show plan without running
#   ./bench_metal.sh --n-gen 128              # Override token count
#   ./bench_metal.sh --llama-cpp ~/llama.cpp  # Enable comparison
#   ./bench_metal.sh --families llama,qwen    # Only specific families

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

LLAMA_CPP="${LLAMA_CPP:-}"
MODEL_CACHE="$HOME/.cache/infernum/models"
GGUF_DIR="/tmp"

N_GEN=256
LLAMA_BENCH_REPS=3
DRY_RUN=false
FAMILIES_FILTER="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=true; shift ;;
        --n-gen)        N_GEN="$2"; shift 2 ;;
        --n-gen=*)      N_GEN="${1#--n-gen=}"; shift ;;
        --llama-cpp)    LLAMA_CPP="$2"; shift 2 ;;
        --llama-cpp=*)  LLAMA_CPP="${1#--llama-cpp=}"; shift ;;
        --families)     FAMILIES_FILTER="$2"; shift 2 ;;
        --families=*)   FAMILIES_FILTER="${1#--families=}"; shift ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--n-gen N] [--llama-cpp PATH] [--families LIST]"
            echo ""
            echo "  --families LIST   Comma-separated: llama, qwen, gemma (default: all)"
            echo "  --llama-cpp PATH  Enable llama.cpp comparison (or set \$LLAMA_CPP)"
            exit 0 ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
    esac
done

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

LLAMA_BENCH=""
HAS_LLAMA_CPP=false
if [[ -n "${LLAMA_CPP}" ]]; then
    LLAMA_BENCH="${LLAMA_CPP}/build/bin/llama-bench"
    if [[ -x "${LLAMA_BENCH}" ]]; then
        HAS_LLAMA_CPP=true
    else
        echo ">>> WARNING: --llama-cpp set but llama-bench not found at ${LLAMA_BENCH} — comparison skipped" >&2
    fi
fi

SMOLLM_Q4="${GGUF_DIR}/smollm2-360m-q4_0.gguf"
SMOLLM_Q8="${GGUF_DIR}/smollm2-360m-q8_0.gguf"
SMOLLM_F32="${GGUF_DIR}/smollm2-360m-f32.gguf"
SMOLLM_DIR="${MODEL_CACHE}/HuggingFaceTB/SmolLM2-360M"

QWEN_DIR="${MODEL_CACHE}/Qwen/Qwen3-0.6B"

GEMMA_GGUF_Q8="${GGUF_DIR}/gemma-2-2b-it-q8_0.gguf"
GEMMA_GGUF_Q4="${GGUF_DIR}/gemma-2-2b-it-q4_k_m.gguf"

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
    local tmpdir; tmpdir=$(mktemp -d)
    hf download "${repo_id}" "${filename}" --local-dir "${tmpdir}" --quiet 2>/dev/null
    local downloaded
    downloaded=$(find "${tmpdir}" -name "${filename}" | head -1)
    [[ -n "${downloaded}" ]] || die "Failed to download ${repo_id}/${filename}"
    mv "${downloaded}" "${dest}"
    rm -rf "${tmpdir}"
}

extract_toks() {
    sed -n 's/.*= \([0-9.]*\) tok\/s.*/\1/p' | tail -1
}

run_infernum() {
    local model_path="$1"
    $DRY_RUN && { echo "—"; return; }
    local out
    out=$(cargo run --release --example bench_metal --features metal -q -- \
        "${model_path}" "${N_GEN}" 2>/dev/null || true)
    local toks; toks=$(echo "${out}" | extract_toks)
    echo "${toks:-ERR}"
}

run_llama_bench() {
    local gguf="$1"
    $HAS_LLAMA_CPP || { echo "—"; return; }
    $DRY_RUN && { echo "—"; return; }
    local json
    json=$("${LLAMA_BENCH}" -m "${gguf}" -p 0 -n "${N_GEN}" -r "${LLAMA_BENCH_REPS}" \
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

# ── Model preparation ─────────────────────────────────────────────────────────

log "Preparing models..."
$DRY_RUN && log "(dry-run: skipping downloads)"

[[ -n "${ENABLED_FAMILIES[llama]:-}" ]] && {
    # Use pre-built GGUF from bartowski — no llama.cpp conversion needed
    download_hf_file "bartowski/SmolLM2-360M-Instruct-GGUF" "SmolLM2-360M-Instruct-Q4_0.gguf" "${SMOLLM_Q4}"
    download_hf_file "bartowski/SmolLM2-360M-Instruct-GGUF" "SmolLM2-360M-Instruct-Q8_0.gguf" "${SMOLLM_Q8}"
    # F32 GGUF only if llama.cpp is available (needed to convert)
    if $HAS_LLAMA_CPP; then
        download_hf_model "HuggingFaceTB/SmolLM2-360M" "${SMOLLM_DIR}"
        _f16="${GGUF_DIR}/smollm2-360m-f16.gguf"
        [[ -f "${SMOLLM_F32}" ]] || {
            log "Converting SmolLM2-360M → F32 GGUF"
            $DRY_RUN || python3 "${LLAMA_CPP}/convert_hf_to_gguf.py" "${SMOLLM_DIR}" \
                --outfile "${SMOLLM_F32}" --outtype f32 2>/dev/null
        }
    fi
}

[[ -n "${ENABLED_FAMILIES[qwen]:-}" ]] && {
    download_hf_model "Qwen/Qwen3-0.6B" "${QWEN_DIR}"
}

[[ -n "${ENABLED_FAMILIES[gemma]:-}" ]] && {
    download_hf_file "bartowski/gemma-2-2b-it-GGUF" "gemma-2-2b-it-Q8_0.gguf" "${GEMMA_GGUF_Q8}"
    download_hf_file "bartowski/gemma-2-2b-it-GGUF" "gemma-2-2b-it-Q4_K_M.gguf" "${GEMMA_GGUF_Q4}"
}

# ── Build ──────────────────────────────────────────────────────────────────────

log "Building infernum bench_metal (release)..."
$DRY_RUN || cargo build --release --example bench_metal --features metal -q 2>/dev/null

# ── Run benchmarks ────────────────────────────────────────────────────────────

log ""
log "Running benchmarks (${N_GEN} tokens, Metal GPU)..."
log "─────────────────────────────────────────────"

declare -A D_INFNUM D_LLAMA

# D_INFNUM and D_LLAMA share the same key space. Where only one side can run
# a given format, the other side is pre-set to "—".
#
# Infernum Metal only supports SafeTensors (GGUF loading not yet implemented).
# llama.cpp only supports GGUF. So Llama and Gemma get separate rows per format.

[[ -n "${ENABLED_FAMILIES[llama]:-}" ]] && {
    # infernum: SafeTensors F32 (only format supported on Metal today)
    if [[ -d "${SMOLLM_DIR}" ]]; then
        log "[Llama] SmolLM2-360M SafeTensors F32 — infernum"
        D_INFNUM["Llama/SmolLM2-360M|SafeTensors F32"]=$(run_infernum "${SMOLLM_DIR}")
        D_LLAMA["Llama/SmolLM2-360M|SafeTensors F32"]="—"
    fi
    # llama.cpp: GGUF Q8_0 and Q4_0 (no infernum comparison — different format)
    if [[ -f "${SMOLLM_Q8}" ]]; then
        log "[Llama] SmolLM2-360M GGUF Q8_0 — llama.cpp"
        D_INFNUM["Llama/SmolLM2-360M|GGUF Q8_0"]="—"
        D_LLAMA["Llama/SmolLM2-360M|GGUF Q8_0"]=$(run_llama_bench "${SMOLLM_Q8}")
    fi
    if [[ -f "${SMOLLM_Q4}" ]]; then
        log "[Llama] SmolLM2-360M GGUF Q4_0 — llama.cpp"
        D_INFNUM["Llama/SmolLM2-360M|GGUF Q4_0"]="—"
        D_LLAMA["Llama/SmolLM2-360M|GGUF Q4_0"]=$(run_llama_bench "${SMOLLM_Q4}")
    fi
}

[[ -n "${ENABLED_FAMILIES[qwen]:-}" ]] && {
    if [[ -d "${QWEN_DIR}" ]]; then
        log "[Qwen] Qwen3-0.6B SafeTensors BF16 — infernum"
        D_INFNUM["Qwen/Qwen3-0.6B|SafeTensors BF16"]=$(run_infernum "${QWEN_DIR}")
        D_LLAMA["Qwen/Qwen3-0.6B|SafeTensors BF16"]="—"
    fi
}

[[ -n "${ENABLED_FAMILIES[gemma]:-}" ]] && {
    # infernum: no ungated SafeTensors available for Gemma 2-2B — skip
    # llama.cpp: GGUF Q8_0 for reference
    if [[ -f "${GEMMA_GGUF_Q8}" ]]; then
        log "[Gemma] gemma-2-2b-it GGUF Q8_0 — llama.cpp only"
        D_INFNUM["Gemma/gemma-2-2b-it|GGUF Q8_0"]="—"
        D_LLAMA["Gemma/gemma-2-2b-it|GGUF Q8_0"]=$(run_llama_bench "${GEMMA_GGUF_Q8}")
    fi
    if [[ -f "${GEMMA_GGUF_Q4}" ]]; then
        log "[Gemma] gemma-2-2b-it GGUF Q4_K_M — llama.cpp only"
        D_INFNUM["Gemma/gemma-2-2b-it|GGUF Q4_K_M"]="—"
        D_LLAMA["Gemma/gemma-2-2b-it|GGUF Q4_K_M"]=$(run_llama_bench "${GEMMA_GGUF_Q4}")
    fi
}

log ""
log "Done."

# ── Output ────────────────────────────────────────────────────────────────────

gpu_name=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model" | sed 's/.*: //' | head -1 || uname -m)
mem_gb=$(system_profiler SPHardwareDataType 2>/dev/null | grep "Memory:" | awk '{print $2,$3}' | head -1 || echo "?")
infernum_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
llama_commit="unknown"
$HAS_LLAMA_CPP && llama_commit=$(git -C "${LLAMA_CPP}" rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo ""
echo "## Infernum Metal — all model families"
echo ""
echo "- **Chip:** ${gpu_name} (${mem_gb} unified memory)"
echo "- **Tokens generated:** ${N_GEN} (8-token prompt, greedy)"
echo "- **infernum commit:** \`${infernum_commit}\`"
$HAS_LLAMA_CPP && echo "- **llama.cpp commit:** \`${llama_commit}\` (\`-ngl 99\`, ${LLAMA_BENCH_REPS} reps)"
echo "- **Date:** $(date +%Y-%m-%d)"
echo "- **Note:** Prefill not yet benchmarked (Metal graph engine pending)"

row_order=(
    "Llama/SmolLM2-360M|SafeTensors F32"
    "Llama/SmolLM2-360M|GGUF Q8_0"
    "Llama/SmolLM2-360M|GGUF Q4_0"
    "Qwen/Qwen3-0.6B|SafeTensors BF16"
    "Gemma/gemma-2-2b-it|GGUF Q8_0"
    "Gemma/gemma-2-2b-it|GGUF Q4_K_M"
)

echo ""
echo "### Decode throughput (tok/s)"
echo ""
echo "- infernum Metal supports SafeTensors only (GGUF loading not yet implemented)"
echo "- llama.cpp supports GGUF only, so Llama and Gemma rows use different formats"
echo ""
echo "| Model | Format | infernum | llama.cpp | ratio |"
echo "| ----- | ------ | -------: | --------: | ----: |"
for key in "${row_order[@]}"; do
    [[ -v D_INFNUM["${key}"] ]] || continue
    model="${key%%|*}"
    fmt="${key##*|}"
    inf="${D_INFNUM[${key}]}"
    lla="${D_LLAMA[${key}]}"
    r=$(ratio "${lla}" "${inf}")
    printf "| %-28s | %-16s | %8s | %9s | %5s |\n" "${model}" "${fmt}" "${inf}" "${lla}" "${r}"
done

echo ""
