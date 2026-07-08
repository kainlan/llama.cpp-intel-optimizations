#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf}"
DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:1}"
OUT_DIR="${OUT_DIR:-/tmp/sycl_gptoss_moe_gateup_work_reduction_$(date +%Y%m%d_%H%M%S)}"
RUN=0
ACK=0

require_value() {
    local opt="$1"
    local value="${2-}"
    if [[ -z "${value}" || "${value}" == --* ]]; then
        echo "error: ${opt} requires a value" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            RUN=0
            ;;
        --run)
            RUN=1
            ;;
        --i-understand-this-runs-gpu-models)
            ACK=1
            ;;
        --model)
            require_value "$1" "${2-}"
            MODEL="$2"
            shift
            ;;
        --device-selector)
            require_value "$1" "${2-}"
            DEVICE_SELECTOR="$2"
            shift
            ;;
        --out-dir)
            require_value "$1" "${2-}"
            OUT_DIR="$2"
            shift
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
    shift
done

if [[ "${RUN}" -eq 1 && "${ACK}" -ne 1 ]]; then
    echo "error: real execution requires --i-understand-this-runs-gpu-models" >&2
    exit 2
fi

COMMON_ENV=(
    "GGML_SYCL_E2E_TG_PROFILE=1"
    "GGML_SYCL_MXFP4_TG_PROFILE=1"
    "GGML_SYCL_MOE_PROFILE=1"
    "GGML_SYCL_GRAPH_DIAG=1"
    "GGML_SYCL_MOE_PHASE_MATERIALIZE=1"
    "GGML_SYCL_MOE_PHASE_BULK_XMX=1"
    "GGML_SYCL_MOE_DOWN_SUM_DIRECT=1"
)

BENCH_ARGS=(
    "${ROOT_DIR}/build/bin/llama-bench"
    -m "${MODEL}"
    -ngl 99
    -fa 1
    -p 512
    -n 128
)

COUNT_ARGS=(
    "${ROOT_DIR}/build/bin/llama-cli"
    -m "${MODEL}"
    -ngl 99
    -cnv
    -st
    --simple-io
    --no-display-prompt
    --chat-template-kwargs '{"reasoning_effort":"medium"}'
    --reasoning-format none
    --reasoning-budget 0
    -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5'
    -n 48
    --seed 42
    --temp 0
)

PARSER_COMMON=(
    "${ROOT_DIR}/scripts/parse-sycl-moe-profile.py"
    --require-no-fatal-markers
    --require-mxfp4-profile-evidence
    --require-e2e-profile-evidence
    --require-e2e-stage moe
    --require-e2e-stage attention
)

PARSER_CANDIDATE=(
    --require-mxfp4-tg-path singlecol-gateup
    --require-mxfp4-gateup-max-ms 4.2
    --require-bench-min pp512=1200
    --require-bench-min tg128=45
)

print_cmd() {
    printf '%q' "$1"
    shift
    for item in "$@"; do
        printf ' %q' "${item}"
    done
}

print_env_command() {
    local -n env_ref=$1
    shift
    printf 'env'
    for item in "${env_ref[@]}"; do
        printf ' %q' "${item}"
    done
    printf ' '
    print_cmd "$@"
}

run_correctness_case() {
    local name="correctness_candidate"
    local case_dir="${OUT_DIR}/${name}"
    local stdout="${case_dir}/count.stdout"
    local stderr="${case_dir}/count.stderr"
    local parse_out="${case_dir}/parse.stdout"
    local envs=("ONEAPI_DEVICE_SELECTOR=${DEVICE_SELECTOR}" "GGML_SYCL_MOE_GATEUP_SINGLECOL=1")

    printf '\n# case: %s\n' "${name}"
    printf 'mkdir -p %q\n' "${case_dir}"
    print_env_command envs "${COUNT_ARGS[@]}"
    printf ' >%q 2>%q\n' "${stdout}" "${stderr}"
    printf 'python3 '
    print_cmd "${ROOT_DIR}/scripts/parse-sycl-moe-profile.py" --require-no-fatal-markers --require-generated-count-exact \
        "${stdout}" "${stderr}"
    printf ' >%q\n' "${parse_out}"

    if [[ "${RUN}" -eq 1 ]]; then
        mkdir -p "${case_dir}"
        env "${envs[@]}" "${COUNT_ARGS[@]}" >"${stdout}" 2>"${stderr}"
        python3 "${ROOT_DIR}/scripts/parse-sycl-moe-profile.py" --require-no-fatal-markers --require-generated-count-exact \
            "${stdout}" "${stderr}" >"${parse_out}"
    fi
}

run_bench_case() {
    local name="$1"
    local candidate="$2"
    shift 2
    local case_dir="${OUT_DIR}/${name}"
    local stdout="${case_dir}/bench.stdout"
    local stderr="${case_dir}/bench.stderr"
    local parse_out="${case_dir}/parse.stdout"
    local envs=("ONEAPI_DEVICE_SELECTOR=${DEVICE_SELECTOR}" "${COMMON_ENV[@]}" "$@")

    printf '\n# case: %s\n' "${name}"
    printf 'mkdir -p %q\n' "${case_dir}"
    print_env_command envs "${BENCH_ARGS[@]}"
    printf ' >%q 2>%q\n' "${stdout}" "${stderr}"
    printf 'python3 '
    if [[ "${candidate}" == "1" ]]; then
        print_cmd "${PARSER_COMMON[@]}" "${PARSER_CANDIDATE[@]}" "${stdout}" "${stderr}"
    else
        print_cmd "${PARSER_COMMON[@]}" "${stdout}" "${stderr}"
    fi
    printf ' >%q\n' "${parse_out}"

    if [[ "${RUN}" -eq 1 ]]; then
        mkdir -p "${case_dir}"
        env "${envs[@]}" "${BENCH_ARGS[@]}" >"${stdout}" 2>"${stderr}"
        if [[ "${candidate}" == "1" ]]; then
            python3 "${PARSER_COMMON[@]}" "${PARSER_CANDIDATE[@]}" "${stdout}" "${stderr}" >"${parse_out}"
        else
            python3 "${PARSER_COMMON[@]}" "${stdout}" "${stderr}" >"${parse_out}"
        fi
    fi
}

if [[ "${RUN}" -eq 1 ]]; then
    mkdir -p "${OUT_DIR}"
    set +u
    source /opt/intel/oneapi/setvars.sh --force >"${OUT_DIR}/setvars.log" 2>&1
    set -u
fi

echo "# output: ${OUT_DIR}"
echo "# selector: ${DEVICE_SELECTOR}"
echo "# model: ${MODEL}"

run_correctness_case
run_bench_case baseline 0
run_bench_case candidate_singlecol 1 "GGML_SYCL_MOE_GATEUP_SINGLECOL=1"
