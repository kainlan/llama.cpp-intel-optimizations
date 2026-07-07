#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${SYCL_GPTOSS_MODEL:-/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf}"
OUT_ROOT="${SYCL_DOWN_VARIANT_PROFILE_OUT:-/tmp/sycl_down_variant_profile_$(date +%Y%m%d_%H%M%S)}"
BENCH="${SYCL_LLAMA_BENCH:-${ROOT_DIR}/build/bin/llama-bench}"
EXECUTE=0

usage() {
    printf 'usage: %s [--dry-run|--execute]\n' "$0"
    printf 'default mode is dry-run; --execute runs B50 GPT-OSS llama-bench commands\n'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            EXECUTE=0
            ;;
        --execute)
            EXECUTE=1
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

variants=(
    "baseline|"
    "row2|GGML_SYCL_MOE_DOWN_SUM_Q8_SOA_TG_VARIANT=row2"
    "row4|GGML_SYCL_MOE_DOWN_SUM_Q8_SOA_TG_VARIANT=row4"
    "atomic|GGML_SYCL_MOE_DOWN_SUM_DIRECT_ATOMIC=1"
    "down-dpas-tile2|GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE=tile2"
    "down-dpas-tile4|GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE=tile4"
    "cached-vector-qs|GGML_SYCL_MOE_DOWN_SUM_DIRECT=0 GGML_SYCL_MOE_DOWN_CACHED_Q8_SOA_TG_VARIANT=vector-qs"
    "cached-cache-y|GGML_SYCL_MOE_DOWN_SUM_DIRECT=0 GGML_SYCL_MOE_DOWN_CACHED_Q8_SOA_TG_VARIANT=cache-y"
    "cached-vector-qs-cache-y|GGML_SYCL_MOE_DOWN_SUM_DIRECT=0 GGML_SYCL_MOE_DOWN_CACHED_Q8_SOA_TG_VARIANT=vector-qs-cache-y"
)

common_env=(
    "ONEAPI_DEVICE_SELECTOR=level_zero:1"
    "GGML_SYCL_KERNEL_PROFILE=1"
    "GGML_SYCL_KERNEL_PROFILE_FORMAT=both"
    "GGML_SYCL_KERNEL_PROFILE_TOP_N=80"
    "GGML_SYCL_KERNEL_PROFILE_FLUSH=window"
    "GGML_SYCL_MXFP4_TG_PROFILE=1"
    "GGML_SYCL_MOE_PROFILE=1"
    "GGML_SYCL_MOE_PHASE_MATERIALIZE=1"
    "GGML_SYCL_MOE_PHASE_BULK_XMX=1"
    "GGML_SYCL_MOE_DOWN_SUM_DIRECT=1"
)

variant_env_clear_names=(
    "GGML_SYCL_MOE_DOWN_SUM_DIRECT"
    "GGML_SYCL_MOE_DOWN_SUM_Q8_SOA_TG_VARIANT"
    "GGML_SYCL_MOE_DOWN_CACHED_Q8_SOA_TG_VARIANT"
    "GGML_SYCL_MOE_DOWN_SUM_DIRECT_ATOMIC"
    "GGML_SYCL_MOE_DOWN_Q8_DPAS_TILE"
    "GGML_SYCL_MOE_DOWN_SUM_XMX_DIRECT_FINAL"
    "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL"
    "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_I8"
    "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_DPAS"
    "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_RANK_PARALLEL_ATOMIC"
    "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_SCRATCH_REDUCE"
    "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_SAME_EXPERT_GROUPED"
)

variant_env_clear_args=()
for name in "${variant_env_clear_names[@]}"; do
    variant_env_clear_args+=("-u" "${name}")
done

bench_args=(
    "${BENCH}"
    -m "${MODEL}"
    -ngl 99
    -fa 1
    -p 512
    -n 128
    -r 1
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
    for item in "${variant_env_clear_args[@]}"; do
        printf ' %q' "${item}"
    done
    for item in "${env_ref[@]}"; do
        printf ' %q' "${item}"
    done
    printf ' '
    print_cmd "$@"
}

run_one() {
    local name="$1"
    local extra_env="$2"
    local case_dir="${OUT_ROOT}/${name}"
    local profile_base="${case_dir}/sycl-kernels"
    local stdout="${case_dir}/bench.stdout"
    local stderr="${case_dir}/bench.stderr"
    local parse_out="${case_dir}/parse.stdout"
    local cmd_env=("${common_env[@]}" "GGML_SYCL_KERNEL_PROFILE_OUTPUT=${profile_base}")

    if [[ -n "${extra_env}" ]]; then
        local extra_items=()
        read -r -a extra_items <<< "${extra_env}"
        cmd_env+=("${extra_items[@]}")
    fi

    printf '\n# case: %s\n' "${name}"
    printf 'mkdir -p %q\n' "${case_dir}"
    print_env_command cmd_env "${bench_args[@]}"
    printf ' >%q 2>%q\n' "${stdout}" "${stderr}"
    print_cmd python3 "${ROOT_DIR}/scripts/parse-sycl-kernel-profile.py" "${profile_base}.csv"
    printf ' >%q\n' "${parse_out}"

    if [[ "${EXECUTE}" -eq 1 ]]; then
        mkdir -p "${case_dir}"
        printf '%s\n' "variant=${name}" "profile=${profile_base}.csv" "model=${MODEL}" >"${case_dir}/metadata.txt"
        env "${variant_env_clear_args[@]}" "${cmd_env[@]}" "${bench_args[@]}" >"${stdout}" 2>"${stderr}"
        python3 "${ROOT_DIR}/scripts/parse-sycl-kernel-profile.py" "${profile_base}.csv" >"${parse_out}"
    fi
}

if [[ "${EXECUTE}" -eq 0 ]]; then
    printf 'DRY RUN: pass --execute to run B50 GPT-OSS down-variant profile matrix\n'
else
    mkdir -p "${OUT_ROOT}"
    set +u
    source /opt/intel/oneapi/setvars.sh --force >"${OUT_ROOT}/setvars.log" 2>&1
    set -u
fi

printf '# output root: %s\n' "${OUT_ROOT}"
printf '# model: %s\n' "${MODEL}"

for variant in "${variants[@]}"; do
    name="${variant%%|*}"
    extra="${variant#*|}"
    run_one "${name}" "${extra}"
done
