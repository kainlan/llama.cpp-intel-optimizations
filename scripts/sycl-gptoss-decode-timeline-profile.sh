#!/usr/bin/env bash
set -euo pipefail

EXECUTE=0
ACK=0
OUT_ROOT="/tmp/sycl_decode_timeline_$(date +%Y%m%d_%H%M%S)"
DEVICE_SELECTOR="level_zero:1"
MODEL="/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf"
TOKEN_START="1"
BENCH="./build/bin/llama-bench"

usage() {
    printf 'usage: %s [--dry-run|--execute] [--i-understand-this-runs-gpu-models] [options]\n' "$0"
    printf 'default mode is dry-run; real execution requires --execute and --i-understand-this-runs-gpu-models\n'
    printf 'options: --out-root DIR --device-selector SELECTOR --model GGUF --token-start N\n'
}

require_value() {
    local opt="$1"
    local value="${2-}"
    if [[ -z "${value}" || "${value}" == --* ]]; then
        printf 'error: %s requires a value\n' "${opt}" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            EXECUTE=0
            ;;
        --execute)
            EXECUTE=1
            ;;
        --i-understand-this-runs-gpu-models)
            ACK=1
            ;;
        --out-root)
            require_value "$1" "${2-}"
            OUT_ROOT="$2"
            shift
            ;;
        --device-selector)
            require_value "$1" "${2-}"
            DEVICE_SELECTOR="$2"
            shift
            ;;
        --model)
            require_value "$1" "${2-}"
            MODEL="$2"
            shift
            ;;
        --token-start)
            require_value "$1" "${2-}"
            TOKEN_START="$2"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf 'unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ "${EXECUTE}" -eq 1 && "${ACK}" -ne 1 ]]; then
    printf 'error: --execute requires --i-understand-this-runs-gpu-models\n' >&2
    exit 2
fi

env_args=(
    "ONEAPI_DEVICE_SELECTOR=${DEVICE_SELECTOR}"
    "GGML_SYCL_TIMELINE=timeline+events"
    "GGML_SYCL_TIMELINE_OUTPUT=${OUT_ROOT}/sycl-timeline.json"
    "GGML_SYCL_TIMELINE_TOKEN_START=${TOKEN_START}"
    "GGML_SYCL_KERNEL_PROFILE=1"
    "GGML_SYCL_KERNEL_PROFILE_OUTPUT=${OUT_ROOT}/sycl-kernels"
    "GGML_SYCL_KERNEL_PROFILE_FORMAT=both"
    "GGML_SYCL_KERNEL_PROFILE_RAW=1"
    "GGML_SYCL_KERNEL_PROFILE_TOP_N=80"
    "GGML_SYCL_KERNEL_PROFILE_FLUSH=window"
    "GGML_SYCL_MOE_PHASE_MATERIALIZE=1"
    "GGML_SYCL_MOE_PHASE_BULK_XMX=1"
    "GGML_SYCL_MOE_DOWN_SUM_DIRECT=1"
)

bench_args=(
    "${BENCH}"
    -m "${MODEL}"
    -ngl 99
    -fa 1
    -p 512
    -n 128
    -r 1
)

timeline_gaps_parse="${OUT_ROOT}/timeline.gaps.parse"
cost_ranking_parse="${OUT_ROOT}/cost-ranking.parse"
wall_ledger_parse="${OUT_ROOT}/wall-ledger.parse"

print_cmd() {
    printf 'env'
    local item
    for item in "${env_args[@]}"; do
        printf ' %q' "${item}"
    done
    for item in "${bench_args[@]}"; do
        printf ' %q' "${item}"
    done
}

if [[ "${EXECUTE}" -ne 1 ]]; then
    printf 'DRY RUN: would execute lead-only GPT-OSS decode timeline profile.\n'
    printf '# output root: %s\n' "${OUT_ROOT}"
    print_cmd
    printf ' >%q 2>%q\n' "${OUT_ROOT}/bench.stdout" "${OUT_ROOT}/bench.stderr"
    printf 'python3 %q %q >%q\n' "scripts/parse-sycl-timeline.py" "${OUT_ROOT}/sycl-timeline.json" "${OUT_ROOT}/timeline.parse"
    printf 'python3 %q --top-gaps 20 --top-host-gap-overlaps 40 %q >%q\n' "scripts/parse-sycl-timeline.py" "${OUT_ROOT}/sycl-timeline.json" "${timeline_gaps_parse}"
    printf 'python3 %q %q >%q\n' "scripts/parse-sycl-kernel-profile.py" "${OUT_ROOT}/sycl-kernels.csv" "${OUT_ROOT}/kernels.parse"
    printf 'python3 %q --top-kernels 30 %q >%q\n' "scripts/parse-sycl-kernel-profile.py" "${OUT_ROOT}/sycl-kernels.csv" "${cost_ranking_parse}"
    printf 'python3 %q %q %q >%q\n' "scripts/parse-sycl-profile-ledger.py" "${OUT_ROOT}/sycl-timeline.json" "${OUT_ROOT}/sycl-kernels.csv" "${wall_ledger_parse}"
    exit 0
fi

mkdir -p "${OUT_ROOT}"
print_cmd >"${OUT_ROOT}/command.txt"
printf ' >%q 2>%q\n' "${OUT_ROOT}/bench.stdout" "${OUT_ROOT}/bench.stderr" >>"${OUT_ROOT}/command.txt"
env "${env_args[@]}" "${bench_args[@]}" >"${OUT_ROOT}/bench.stdout" 2>"${OUT_ROOT}/bench.stderr"
python3 scripts/parse-sycl-timeline.py "${OUT_ROOT}/sycl-timeline.json" >"${OUT_ROOT}/timeline.parse"
python3 scripts/parse-sycl-timeline.py \
    --top-gaps 20 \
    --top-host-gap-overlaps 40 \
    "${OUT_ROOT}/sycl-timeline.json" >"${timeline_gaps_parse}"
python3 scripts/parse-sycl-kernel-profile.py "${OUT_ROOT}/sycl-kernels.csv" >"${OUT_ROOT}/kernels.parse"
python3 scripts/parse-sycl-kernel-profile.py --top-kernels 30 "${OUT_ROOT}/sycl-kernels.csv" >"${OUT_ROOT}/cost-ranking.parse"
python3 scripts/parse-sycl-profile-ledger.py "${OUT_ROOT}/sycl-timeline.json" "${OUT_ROOT}/sycl-kernels.csv" >"${OUT_ROOT}/wall-ledger.parse"
printf 'Artifacts: %s\n' "${OUT_ROOT}"
