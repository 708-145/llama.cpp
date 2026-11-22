#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the matched dense-versus-DNA llama-perplexity sweep.

Usage:
  scripts/dna-ppl-sweep.sh MODEL.gguf WIKITEXT.txt [OUTPUT_DIR]

Environment:
  LLAMA_PERPLEXITY              Binary path (default: REPO_ROOT/build/bin/llama-perplexity)
  DNA_PPL_COOLDOWN_SECONDS      Pause between runs (default: 120)

The sweep runs dense, iq4_xs, iq3_xxs, iq2_xs, iq1_m, and iq1_s in sequence.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if (( $# < 2 || $# > 3 )); then
    usage >&2
    exit 2
fi

MODEL=$1
CORPUS=$2
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LLAMA_PERPLEXITY="${LLAMA_PERPLEXITY:-${REPO_ROOT}/build/bin/llama-perplexity}"
COOLDOWN_SECONDS="${DNA_PPL_COOLDOWN_SECONDS:-120}"
OUTPUT_DIR="${3:-${TMPDIR:-/tmp}/dna-ppl-sweep-$(date +%Y%m%d-%H%M%S)}"

if [[ ! -f "$MODEL" ]]; then
    printf 'Model file not found: %s\n' "$MODEL" >&2
    exit 1
fi
if [[ ! -f "$CORPUS" ]]; then
    printf 'Corpus file not found: %s\n' "$CORPUS" >&2
    exit 1
fi
if [[ "$LLAMA_PERPLEXITY" == */* ]]; then
    if [[ ! -x "$LLAMA_PERPLEXITY" ]]; then
        printf 'llama-perplexity is not executable: %s\n' "$LLAMA_PERPLEXITY" >&2
        exit 1
    fi
else
    LLAMA_PERPLEXITY="$(command -v "$LLAMA_PERPLEXITY")" || {
        printf 'llama-perplexity not found on PATH: %s\n' "$LLAMA_PERPLEXITY" >&2
        exit 1
    }
fi
if [[ ! "$COOLDOWN_SECONDS" =~ ^[0-9]+$ ]]; then
    printf 'DNA_PPL_COOLDOWN_SECONDS must be a non-negative integer: %s\n' "$COOLDOWN_SECONDS" >&2
    exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    printf 'Output directory already exists; choose a new path: %s\n' "$OUTPUT_DIR" >&2
    exit 1
fi
mkdir -p "$(dirname -- "$OUTPUT_DIR")"
mkdir -- "$OUTPUT_DIR"

run_eval() {
    local name=$1
    local predictor=$2
    local logfile="${OUTPUT_DIR}/${name}.log"
    local -a args=(
        -m "$MODEL"
        -f "$CORPUS"
        -c 512
        --chunks 8
        -t 4
        -b 512
        -ub 512
        --fit off
    )
    local -a pipeline_status

    if [[ -n "$predictor" ]]; then
        args+=(--dnapred "$predictor")
    fi

    printf '\n=== %s%s ===\n' "$name" "${predictor:+ ($predictor)}"
    printf 'Log: %s\n' "$logfile"
    {
        printf 'model: %s\n' "$MODEL"
        printf 'corpus: %s\n' "$CORPUS"
        printf 'context=512 chunks=8 threads=4 batch=512 ubatch=512 fit=off\n'
        if [[ -n "$predictor" ]]; then
            printf 'predictor=%s T=0.5 calibration=512 min_active=0.02 zmode=0 max_batch=512\n' \
                "$predictor"
        else
            printf 'dense baseline\n'
        fi
    } > "$logfile"

    set +e
    if [[ -n "$predictor" ]]; then
        (
            unset LLAMA_ARG_DNA LLAMA_ARG_DNAPRED LLAMA_DNA_PRED
            export LLAMA_GRAPH_REUSE_DISABLE=1
            export LLAMA_DNA_MAX_BATCH=512
            export LLAMA_DNA_T=0.5
            export LLAMA_DNA_CALIB=512
            export LLAMA_DNA_MIN_ACTIVE=0.02
            export LLAMA_DNA_ZMODE=0
            export LLAMA_DNA_VERBOSE=1
            "$LLAMA_PERPLEXITY" "${args[@]}"
        ) 2>&1 | tee -a "$logfile"
    else
        (
            unset LLAMA_ARG_DNA LLAMA_ARG_DNAPRED LLAMA_DNA_PRED
            unset LLAMA_DNA_MAX_BATCH LLAMA_DNA_T LLAMA_DNA_CALIB
            unset LLAMA_DNA_MIN_ACTIVE LLAMA_DNA_ZMODE LLAMA_DNA_VERBOSE
            export LLAMA_GRAPH_REUSE_DISABLE=1
            "$LLAMA_PERPLEXITY" "${args[@]}"
        ) 2>&1 | tee -a "$logfile"
    fi
    pipeline_status=("${PIPESTATUS[@]}")
    set -e

    if (( pipeline_status[0] != 0 )); then
        printf 'llama-perplexity failed for %s (exit %d); see %s\n' \
            "$name" "${pipeline_status[0]}" "$logfile" >&2
        return "${pipeline_status[0]}"
    fi
    if (( pipeline_status[1] != 0 )); then
        printf 'Could not write log for %s (tee exit %d)\n' \
            "$name" "${pipeline_status[1]}" >&2
        return "${pipeline_status[1]}"
    fi
    if ! grep -q 'Final estimate: PPL =' "$logfile"; then
        printf 'No final PPL estimate found for %s; see %s\n' "$name" "$logfile" >&2
        return 1
    fi
    grep 'Final estimate: PPL =' "$logfile" | tail -n 1
}

run_eval dense ""
predictors=(iq4_xs iq3_xxs iq2_xs iq1_m iq1_s)
for predictor in "${predictors[@]}"; do
    if (( COOLDOWN_SECONDS > 0 )); then
        printf 'Cooling down for %s seconds...\n' "$COOLDOWN_SECONDS"
        sleep "$COOLDOWN_SECONDS"
    fi
    run_eval "$predictor" "$predictor"
done

printf '\nSweep complete. Logs: %s\n' "$OUTPUT_DIR"
