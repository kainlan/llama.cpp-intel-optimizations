#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf}"
DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:1}"
OUT_DIR="${OUT_DIR:-/tmp/sycl_gptoss_e2e_profile_$(date +%Y%m%d_%H%M%S)}"
RUN=0
ACK=0
INCLUDE_MULTIGPU=0

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
        --include-multigpu)
            INCLUDE_MULTIGPU=1
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

COMMON_ARGS=(
    "${ROOT_DIR}/build/bin/llama-bench"
    "-m" "${MODEL}"
    "-ngl" "99"
    "-fa" "1"
    "-p" "512"
    "-n" "128"
)

PARSER_ARGS=(
    "${ROOT_DIR}/scripts/parse-sycl-moe-profile.py"
    "--require-no-fatal-markers"
    "--require-mxfp4-profile-evidence"
    "--require-e2e-profile-evidence"
    "--require-e2e-stage" "moe"
    "--require-e2e-stage" "attention"
    "--require-diag-path" "packed-q8-m2"
)

run_case() {
    local name="$1"
    shift
    local case_dir="${OUT_DIR}/${name}"
    local stdout="${case_dir}/bench.stdout"
    local stderr="${case_dir}/bench.stderr"
    local envs=("ONEAPI_DEVICE_SELECTOR=${DEVICE_SELECTOR}" "${COMMON_ENV[@]}" "$@")
    printf '\n# case: %s\n' "${name}"
    printf 'mkdir -p %q\n' "${case_dir}"
    printf 'env'
    for item in "${envs[@]}"; do
        printf ' %q' "${item}"
    done
    for item in "${COMMON_ARGS[@]}"; do
        printf ' %q' "${item}"
    done
    printf ' >%q 2>%q\n' "${stdout}" "${stderr}"
    printf 'python3'
    for item in "${PARSER_ARGS[@]}"; do
        printf ' %q' "${item}"
    done
    printf ' %q %q >%q\n' "${stdout}" "${stderr}" "${case_dir}/parse.stdout"

    if [[ "${RUN}" -eq 1 ]]; then
        mkdir -p "${case_dir}"
        env "${envs[@]}" "${COMMON_ARGS[@]}" >"${stdout}" 2>"${stderr}"
        python3 "${PARSER_ARGS[@]}" "${stdout}" "${stderr}" >"${case_dir}/parse.stdout"
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

run_case baseline
run_case graph_disabled "GGML_SYCL_DISABLE_GRAPH=1"
run_case fa_kv_detail "GGML_SYCL_FA_DISPATCH_DEBUG=1" "GGML_SYCL_PACKED_K_DEBUG_LIMIT=8"
run_case vram_pressure "GGML_SYCL_VRAM_BUDGET_PCT=85"
run_case cpu_sharing "GGML_SYCL_PIPELINE_CPU=1" "GGML_SYCL_CPU_EXPERT_THREADS=8"

if [[ "${INCLUDE_MULTIGPU}" -eq 1 ]]; then
    run_case multigpu_host_bounce "GGML_SYCL_MOE_ROUTE_LOG=1"
fi
