#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${BENCH:-$ROOT/build/bin/llama-bench}"
COMPLETION="${COMPLETION:-$ROOT/build/bin/llama-completion}"
GPTOSS_MODEL="${GPTOSS_MODEL:-/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf}"
MISTRAL_MODEL="${MISTRAL_MODEL:-/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf}"
B50_SELECTOR="${B50_SELECTOR:-level_zero:1}"
B580_SELECTOR="${B580_SELECTOR:-level_zero:0}"
OUTDIR="${OUTDIR:-/tmp/sycl-moe-regression.$(date +%Y%m%d-%H%M%S)}"
NGL="${NGL:-99}"
GPTOSS_PROMPT="${GPTOSS_PROMPT:-512}"
GPTOSS_TOKENS="${GPTOSS_TOKENS:-128}"
GPTOSS_REPS="${GPTOSS_REPS:-1}"
PROFILE_TOKENS="${PROFILE_TOKENS:-16}"
MISTRAL_PROMPT="${MISTRAL_PROMPT:-512}"
MISTRAL_TOKENS="${MISTRAL_TOKENS:-128}"
MISTRAL_REPS="${MISTRAL_REPS:-1}"
RUN_GRAPH_OFF="${RUN_GRAPH_OFF:-0}"
RUN_B580="${RUN_B580:-1}"
RUN_B50="${RUN_B50:-1}"
RUN_MISTRAL="${RUN_MISTRAL:-1}"
RUN_TGB16="${RUN_TGB16:-0}"
RUN_PROFILES="${RUN_PROFILES:-0}"
RUN_CORRECTNESS="${RUN_CORRECTNESS:-1}"
SMOKE_CTX="${SMOKE_CTX:-512}"
ONEAPI_SETVARS="${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"
LOCK_FILE="${LOCK_FILE:-/tmp/sycl-moe-regression-harness.lock}"

usage() {
    cat <<'EOF'
Usage: scripts/sycl-moe-regression-harness.sh

Runs the local SYCL optimization regression set sequentially so benchmark
numbers are not contaminated by concurrent GPU work.

Environment:
  BENCH             llama-bench path, default build/bin/llama-bench
  COMPLETION        llama-completion path, default build/bin/llama-completion
  GPTOSS_MODEL      GPT-OSS MXFP4 GGUF path
  MISTRAL_MODEL     Mistral Q4_0 GGUF path
  B50_SELECTOR      B50 oneAPI selector, default level_zero:1
  B580_SELECTOR     B580 oneAPI selector, default level_zero:0
  OUTDIR            log directory
  NGL               GPU layer count passed to llama-bench, default 99
  GPTOSS_PROMPT     GPT-OSS prompt tokens, default 512
  GPTOSS_TOKENS     GPT-OSS decode tokens, default 128
  GPTOSS_REPS       GPT-OSS benchmark repetitions, default 1
  PROFILE_TOKENS    GPT-OSS profile decode tokens, default 16
  MISTRAL_PROMPT    Mistral prompt tokens, default 512
  MISTRAL_TOKENS    Mistral decode tokens, default 128
  MISTRAL_REPS      Mistral benchmark repetitions, default 1
  RUN_GRAPH_OFF=1   Add a GPT-OSS graph-disabled diagnostic
  RUN_B580=0        Skip GPT-OSS B580 target path
  RUN_B50=0         Skip GPT-OSS B50 all-VRAM guard
  RUN_MISTRAL=0     Skip the B580 Mistral guard
  RUN_TGB16=1       Add optional GPT-OSS --tg-batch 16 diagnostic cases
  RUN_PROFILES=1    Add short GPT-OSS MoE/MXFP4 profile cases
  RUN_CORRECTNESS=0 Skip completion smoke checks
  SMOKE_CTX         Context size for completion smoke checks, default 512
  ONEAPI_SETVARS    oneAPI setvars path
  LOCK_FILE         flock path used to prevent overlapping harness runs
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

if [[ "$RUN_CORRECTNESS" == "1" && ! -x "$COMPLETION" ]]; then
    echo "llama-completion not executable: $COMPLETION" >&2
    exit 2
fi

if [[ -f "$ONEAPI_SETVARS" ]]; then
    # shellcheck disable=SC1090
    set +u
    source "$ONEAPI_SETVARS" --force >/dev/null
    set -u
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another SYCL MoE regression harness is already running: $LOCK_FILE" >&2
    exit 3
fi

mkdir -p "$OUTDIR"

summary="$OUTDIR/summary.tsv"
printf "case\tselector\tfa\tprompt\ttokens\ttg_batch\tpp\ttg\tmissing_compute\tlog\n" >"$summary"
smoke_summary="$OUTDIR/smoke.tsv"
printf "case\tselector\tstatus\tlog\n" >"$smoke_summary"

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
        "$BENCH" -m "$model" -p "$prompt" -n "$tokens" -ngl "$NGL" -r "$reps" -fa "$fa" --tg-batch "$tg_batch" \
        >"$log" 2>&1

    local pp tg missing
    pp="$(extract_rate "$log" pp)"
    tg="$(extract_rate "$log" tg)"
    missing="$(extract_missing_compute "$log")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$selector" "$fa" "$prompt" "$tokens" "$tg_batch" \
        "$pp" "$tg" "$missing" "$log" | tee -a "$summary"
}

run_smoke() {
    local name="$1"
    local selector="$2"
    local model="$3"
    shift 3
    local log="$OUTDIR/${name}.log"

    echo "== $name =="
    if env ONEAPI_DEVICE_SELECTOR="$selector" "$COMPLETION" -m "$model" "$@" >"$log" 2>&1; then
        printf "%s\t%s\tpass\t%s\n" "$name" "$selector" "$log" | tee -a "$smoke_summary"
    else
        printf "%s\t%s\tfail\t%s\n" "$name" "$selector" "$log" | tee -a "$smoke_summary"
        return 1
    fi
}

if [[ "$RUN_B580" == "1" ]]; then
    run_case gptoss-b580-fa1 "$B580_SELECTOR" "$GPTOSS_MODEL" "$GPTOSS_PROMPT" "$GPTOSS_TOKENS" "$GPTOSS_REPS" 1 1
    run_case gptoss-b580-fa0 "$B580_SELECTOR" "$GPTOSS_MODEL" "$GPTOSS_PROMPT" "$GPTOSS_TOKENS" "$GPTOSS_REPS" 0 1
    if [[ "$RUN_TGB16" == "1" ]]; then
        run_case gptoss-b580-fa1-tgb16 "$B580_SELECTOR" "$GPTOSS_MODEL" "$GPTOSS_PROMPT" "$GPTOSS_TOKENS" \
            "$GPTOSS_REPS" 1 16
        run_case gptoss-b580-fa0-tgb16 "$B580_SELECTOR" "$GPTOSS_MODEL" "$GPTOSS_PROMPT" "$GPTOSS_TOKENS" \
            "$GPTOSS_REPS" 0 16
    fi
fi

if [[ "$RUN_B50" == "1" ]]; then
    run_case gptoss-b50-fa1 "$B50_SELECTOR" "$GPTOSS_MODEL" "$GPTOSS_PROMPT" "$GPTOSS_TOKENS" "$GPTOSS_REPS" 1 1
    run_case gptoss-b50-fa0 "$B50_SELECTOR" "$GPTOSS_MODEL" "$GPTOSS_PROMPT" "$GPTOSS_TOKENS" "$GPTOSS_REPS" 0 1
    if [[ "$RUN_TGB16" == "1" ]]; then
        run_case gptoss-b50-fa1-tgb16 "$B50_SELECTOR" "$GPTOSS_MODEL" "$GPTOSS_PROMPT" "$GPTOSS_TOKENS" \
            "$GPTOSS_REPS" 1 16
        run_case gptoss-b50-fa0-tgb16 "$B50_SELECTOR" "$GPTOSS_MODEL" "$GPTOSS_PROMPT" "$GPTOSS_TOKENS" \
            "$GPTOSS_REPS" 0 16
    fi
fi

if [[ "$RUN_PROFILES" == "1" ]]; then
    if [[ "$RUN_B580" == "1" ]]; then
        run_case gptoss-b580-profile-fa1 "$B580_SELECTOR" "$GPTOSS_MODEL" 64 "$PROFILE_TOKENS" 1 1 1 \
            GGML_SYCL_MOE_PROFILE=1 GGML_SYCL_MXFP4_TG_PROFILE=1
    fi
    if [[ "$RUN_B50" == "1" ]]; then
        run_case gptoss-b50-profile-fa1 "$B50_SELECTOR" "$GPTOSS_MODEL" 64 "$PROFILE_TOKENS" 1 1 1 \
            GGML_SYCL_MOE_PROFILE=1 GGML_SYCL_MXFP4_TG_PROFILE=1
    fi
fi

if [[ "$RUN_GRAPH_OFF" == "1" ]]; then
    if [[ "$RUN_B580" == "1" ]]; then
        run_case gptoss-b580-graph-off-fa1 "$B580_SELECTOR" "$GPTOSS_MODEL" "$GPTOSS_PROMPT" "$GPTOSS_TOKENS" \
            "$GPTOSS_REPS" 1 1 GGML_SYCL_DISABLE_GRAPH=1
    fi
    if [[ "$RUN_B50" == "1" ]]; then
        run_case gptoss-b50-graph-off-fa1 "$B50_SELECTOR" "$GPTOSS_MODEL" "$GPTOSS_PROMPT" "$GPTOSS_TOKENS" \
            "$GPTOSS_REPS" 1 1 GGML_SYCL_DISABLE_GRAPH=1
    fi
fi

if [[ "$RUN_MISTRAL" == "1" ]]; then
    run_case mistral-b580-fa0 "$B580_SELECTOR" "$MISTRAL_MODEL" "$MISTRAL_PROMPT" "$MISTRAL_TOKENS" "$MISTRAL_REPS" 0 1
    run_case mistral-b580-fa1 "$B580_SELECTOR" "$MISTRAL_MODEL" "$MISTRAL_PROMPT" "$MISTRAL_TOKENS" "$MISTRAL_REPS" 1 1
fi

if [[ "$RUN_CORRECTNESS" == "1" ]]; then
    if [[ "$RUN_MISTRAL" == "1" ]]; then
        run_smoke mistral-b580-completion "$B580_SELECTOR" "$MISTRAL_MODEL" \
            -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 -c "$SMOKE_CTX" -ngl "$NGL"
    fi
    if [[ "$RUN_B50" == "1" ]]; then
        run_smoke gptoss-b50-completion-fa1 "$B50_SELECTOR" "$GPTOSS_MODEL" \
            -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 -c "$SMOKE_CTX" -ngl "$NGL" -fa 1
    fi
fi

echo "summary=$summary"
echo "smoke_summary=$smoke_summary"
