#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${MODEL:-/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf}"
BENCH="${BENCH:-./build/bin/llama-bench}"
SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:1}"
OUTDIR="${OUTDIR:-/tmp/sycl-moe-layout-compare.$(date +%Y%m%d-%H%M%S)}"

PROFILE_TOKENS="${PROFILE_TOKENS:-4}"
BENCH_TOKENS="${BENCH_TOKENS:-128}"
PROMPT_TOKENS="${PROMPT_TOKENS:-0}"
BENCH_REPS="${BENCH_REPS:-3}"

DO_BENCH=0
BENCH_INVALID=0

# shellcheck disable=SC1091
source "$ROOT/scripts/sycl-gpu-preflight.sh"

usage() {
    cat <<'EOF'
Usage: scripts/sycl-moe-layout-compare.sh [--bench] [--bench-invalid]

Sequentially validates actual GPT-OSS MXFP4 MoE TG layout routing before
benchmarking. This avoids comparing a requested AOS/SOA knob when the executed
MoE kernel still routes through a different layout.

Environment:
  MODEL                 GGUF path, default /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf
  BENCH                 llama-bench path, default ./build/bin/llama-bench
  ONEAPI_DEVICE_SELECTOR default level_zero:1
  OUTDIR                log output directory
  PROFILE_TOKENS        short profile decode tokens, default 4
  BENCH_TOKENS          full decode tokens, default 128
  PROMPT_TOKENS         prompt tokens, default 0
  BENCH_REPS            llama-bench repetitions, default 3

Options:
  --bench               run full TG bench only for cases whose actual layout
                        matches the expected layout
  --bench-invalid       also run full TG bench for invalid cases
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bench)
            DO_BENCH=1
            ;;
        --bench-invalid)
            DO_BENCH=1
            BENCH_INVALID=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ ! -x "$BENCH" ]]; then
    echo "llama-bench not executable: $BENCH" >&2
    exit 2
fi

mkdir -p "$OUTDIR"

run_case() {
    local name="$1"
    local expected="$2"
    shift 2
    local -a env_args=("$@")
    local profile_log="$OUTDIR/${name}.profile.log"
    local bench_log="$OUTDIR/${name}.bench.log"

    echo "== $name =="
    echo "requested=$expected selector=$SELECTOR log=$profile_log"

    sycl_gpu_preflight_check "$SELECTOR"
    env ONEAPI_DEVICE_SELECTOR="$SELECTOR" \
        GGML_SYCL_MXFP4_TG_PROFILE=1 \
        "${env_args[@]}" \
        "$BENCH" -m "$MODEL" -p "$PROMPT_TOKENS" -n "$PROFILE_TOKENS" -r 1 \
        >"$profile_log" 2>&1

    local line
    line="$(grep '\[MXFP4-MOE-TG-PROFILE\]' "$profile_log" | tail -n 1 || true)"
    if [[ -z "$line" ]]; then
        echo "actual=unknown status=invalid reason=no-profile-line"
        echo
        return 1
    fi

    local soa coalesced aos actual
    soa="$(sed -n 's/.* soa=\([0-9][0-9]*\).*/\1/p' <<<"$line")"
    coalesced="$(sed -n 's/.* coalesced=\([0-9][0-9]*\).*/\1/p' <<<"$line")"
    aos="$(sed -n 's/.* aos=\([0-9][0-9]*\).*/\1/p' <<<"$line")"
    soa="${soa:-0}"
    coalesced="${coalesced:-0}"
    aos="${aos:-0}"

    actual="unknown"
    if (( soa > 0 && coalesced == 0 && aos == 0 )); then
        actual="soa"
    elif (( coalesced > 0 && soa == 0 && aos == 0 )); then
        actual="coalesced"
    elif (( aos > 0 && soa == 0 && coalesced == 0 )); then
        actual="aos"
    elif (( soa > 0 || coalesced > 0 || aos > 0 )); then
        actual="mixed"
    fi

    local valid=0
    if [[ "$actual" == "$expected" ]]; then
        valid=1
        echo "actual=$actual status=valid counters=soa:$soa,coalesced:$coalesced,aos:$aos"
    else
        echo "actual=$actual status=invalid expected=$expected counters=soa:$soa,coalesced:$coalesced,aos:$aos"
    fi

    if (( DO_BENCH == 1 )) && { (( valid == 1 )) || (( BENCH_INVALID == 1 )); }; then
        echo "bench_log=$bench_log"
        sycl_gpu_preflight_check "$SELECTOR"
        env ONEAPI_DEVICE_SELECTOR="$SELECTOR" \
            "${env_args[@]}" \
            "$BENCH" -m "$MODEL" -p "$PROMPT_TOKENS" -n "$BENCH_TOKENS" -r "$BENCH_REPS" \
            >"$bench_log" 2>&1
        grep -E '\|.*tg[0-9]+.*\|' "$bench_log" | tail -n 1 || true
    fi
    echo
    if (( valid == 1 )); then
        return 0
    fi
    return 1
}

status=0
run_case default-soa soa || status=1
run_case override-aos aos GGML_SYCL_LAYOUT_OVERRIDE=aos || status=1

echo "logs=$OUTDIR"
exit "$status"
