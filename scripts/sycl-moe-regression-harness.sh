#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${BENCH:-$ROOT/build/bin/llama-bench}"
GPTOSS_MODEL="${GPTOSS_MODEL:-/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf}"
MISTRAL_MODEL="${MISTRAL_MODEL:-/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf}"
B50_SELECTOR="${B50_SELECTOR:-level_zero:1}"
B580_SELECTOR="${B580_SELECTOR:-level_zero:0}"
OUTDIR="${OUTDIR:-/tmp/sycl-moe-regression.$(date +%Y%m%d-%H%M%S)}"
GPTOSS_TOKENS="${GPTOSS_TOKENS:-128}"
GPTOSS_REPS="${GPTOSS_REPS:-3}"
PROFILE_TOKENS="${PROFILE_TOKENS:-16}"
MISTRAL_PROMPT="${MISTRAL_PROMPT:-512}"
MISTRAL_TOKENS="${MISTRAL_TOKENS:-128}"
MISTRAL_REPS="${MISTRAL_REPS:-3}"
RUN_GRAPH_OFF="${RUN_GRAPH_OFF:-0}"
RUN_MISTRAL="${RUN_MISTRAL:-1}"

usage() {
    cat <<'EOF'
Usage: scripts/sycl-moe-regression-harness.sh

Runs the local SYCL optimization regression set sequentially so benchmark
numbers are not contaminated by concurrent GPU work.

Environment:
  BENCH             llama-bench path, default build/bin/llama-bench
  GPTOSS_MODEL      GPT-OSS MXFP4 GGUF path
  MISTRAL_MODEL     Mistral Q4_0 GGUF path
  B50_SELECTOR      B50 oneAPI selector, default level_zero:1
  B580_SELECTOR     B580 oneAPI selector, default level_zero:0
  OUTDIR            log directory
  GPTOSS_TOKENS     GPT-OSS decode tokens, default 128
  GPTOSS_REPS       GPT-OSS benchmark repetitions, default 3
  PROFILE_TOKENS    GPT-OSS profile decode tokens, default 16
  MISTRAL_PROMPT    Mistral prompt tokens, default 512
  MISTRAL_TOKENS    Mistral decode tokens, default 128
  MISTRAL_REPS      Mistral benchmark repetitions, default 3
  RUN_GRAPH_OFF=1   Add a GPT-OSS graph-disabled diagnostic
  RUN_MISTRAL=0     Skip the B580 Mistral guard
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -x "$BENCH" ]]; then
    echo "llama-bench not executable: $BENCH" >&2
    exit 2
fi

mkdir -p "$OUTDIR"

summary="$OUTDIR/summary.tsv"
printf "case\tselector\tfa\ttg_batch\tpp\ttg\tmissing_compute\tlog\n" >"$summary"

extract_rate() {
    local log="$1"
    local kind="$2"
    awk -v kind="$kind" '
        $0 ~ "\\|[[:space:]]*" kind "[0-9]+[[:space:]]*\\|" {
            for (i = 1; i <= NF; ++i) {
                if ($i == "±" && i > 1) {
                    rate = $(i - 1)
                }
            }
        }
        END { if (rate != "") print rate; else print "NA" }
    ' "$log"
}

extract_missing_compute() {
    local log="$1"
    local missing
    missing="$(sed -n 's/.*moe_compute_done() NOT called: \([0-9][0-9]*\).*/\1/p' "$log" | tail -n 1)"
    if [[ -n "$missing" ]]; then
        printf "%s" "$missing"
    elif grep -q 'All sub-ops had moe_compute_done() called' "$log"; then
        printf "0"
    else
        printf "NA"
    fi
}

run_case() {
    local name="$1"
    local selector="$2"
    local model="$3"
    local prompt="$4"
    local tokens="$5"
    local reps="$6"
    local fa="$7"
    local tg_batch="$8"
    shift 8
    local -a env_args=("$@")
    local log="$OUTDIR/${name}.log"

    echo "== $name =="
    env ONEAPI_DEVICE_SELECTOR="$selector" "${env_args[@]}" \
        "$BENCH" -m "$model" -p "$prompt" -n "$tokens" -r "$reps" -fa "$fa" --tg-batch "$tg_batch" \
        >"$log" 2>&1

    local pp tg missing
    pp="$(extract_rate "$log" pp)"
    tg="$(extract_rate "$log" tg)"
    missing="$(extract_missing_compute "$log")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$selector" "$fa" "$tg_batch" "$pp" "$tg" "$missing" \
        "$log" | tee -a "$summary"
}

run_case gptoss-b50-fa0-tgb1 "$B50_SELECTOR" "$GPTOSS_MODEL" 0 "$GPTOSS_TOKENS" "$GPTOSS_REPS" 0 1
run_case gptoss-b50-fa1-tgb1 "$B50_SELECTOR" "$GPTOSS_MODEL" 0 "$GPTOSS_TOKENS" "$GPTOSS_REPS" 1 1
run_case gptoss-b50-fa0-tgb16 "$B50_SELECTOR" "$GPTOSS_MODEL" 0 "$GPTOSS_TOKENS" "$GPTOSS_REPS" 0 16
run_case gptoss-b50-fa1-tgb16 "$B50_SELECTOR" "$GPTOSS_MODEL" 0 "$GPTOSS_TOKENS" "$GPTOSS_REPS" 1 16

run_case gptoss-b50-profile-fa0 "$B50_SELECTOR" "$GPTOSS_MODEL" 0 "$PROFILE_TOKENS" 1 0 1 \
    GGML_SYCL_MOE_PROFILE=1 GGML_SYCL_MXFP4_TG_PROFILE=1
run_case gptoss-b50-profile-fa1 "$B50_SELECTOR" "$GPTOSS_MODEL" 0 "$PROFILE_TOKENS" 1 1 1 \
    GGML_SYCL_MOE_PROFILE=1 GGML_SYCL_MXFP4_TG_PROFILE=1

if [[ "$RUN_GRAPH_OFF" == "1" ]]; then
    run_case gptoss-b50-graph-off-fa1 "$B50_SELECTOR" "$GPTOSS_MODEL" 0 "$GPTOSS_TOKENS" "$GPTOSS_REPS" 1 1 \
        GGML_SYCL_DISABLE_GRAPH=1
fi

if [[ "$RUN_MISTRAL" == "1" ]]; then
    run_case mistral-b580-fa0 "$B580_SELECTOR" "$MISTRAL_MODEL" "$MISTRAL_PROMPT" "$MISTRAL_TOKENS" "$MISTRAL_REPS" 0 1
    run_case mistral-b580-fa1 "$B580_SELECTOR" "$MISTRAL_MODEL" "$MISTRAL_PROMPT" "$MISTRAL_TOKENS" "$MISTRAL_REPS" 1 1
fi

echo "summary=$summary"
