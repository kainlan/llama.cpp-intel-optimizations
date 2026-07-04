#!/usr/bin/env bash
set -euo pipefail

EXECUTE=0
ACK=0
OUT_ROOT="/tmp/sycl_vtune_source_line_$(date +%Y%m%d_%H%M%S)"
BUILD_DIR="build-vtune-line"
DEVICE_SELECTOR="level_zero:1"
TARGET_KERNEL="mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias"
TASK_GLOB="*mxfp4_pair_glu_xmx_tiled*"
REQUIRE_MATRIX_PASS=""

usage() {
    printf 'usage: %s [--dry-run|--execute] [--i-understand-this-runs-gpu-microbenchmarks] [--out-root DIR] [--build-dir DIR] [--device-selector SELECTOR] [--target-kernel NAME] [--require-matrix-pass PATH]\n' "$0"
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
        --dry-run) EXECUTE=0 ;;
        --execute) EXECUTE=1 ;;
        --i-understand-this-runs-gpu-microbenchmarks) ACK=1 ;;
        --out-root) require_value "$1" "${2-}"; OUT_ROOT="$2"; shift ;;
        --build-dir) require_value "$1" "${2-}"; BUILD_DIR="$2"; shift ;;
        --device-selector) require_value "$1" "${2-}"; DEVICE_SELECTOR="$2"; shift ;;
        --target-kernel) require_value "$1" "${2-}"; TARGET_KERNEL="$2"; TASK_GLOB="*$2*"; shift ;;
        --require-matrix-pass) require_value "$1" "${2-}"; REQUIRE_MATRIX_PASS="$2"; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if [[ "${EXECUTE}" -eq 1 && "${ACK}" -ne 1 ]]; then
    printf 'error: --execute requires --i-understand-this-runs-gpu-microbenchmarks\n' >&2
    exit 2
fi

configure_cmd=(cmake -S . -B "${BUILD_DIR}" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_F16=ON -DGGML_SYCL_PROFILING_DEBUG=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx)
build_cmd=(cmake --build "${BUILD_DIR}" --config Release --target sycl-kernel-bench -j "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}")
bench_cmd=("./${BUILD_DIR}/bin/sycl-kernel-bench" --kernel="${TARGET_KERNEL}" --quant=MXFP4 --dim_m=2880 --dim_n=4 --dim_k=2880 --iterations=100 --warmup=10 --validate --output=json)
vtune_dir="${OUT_ROOT}/vtune-source-line"

print_plan() {
    printf 'DRY RUN: would execute lead-only SYCL VTune source-line microbench feasibility.\n'
    printf '# output root: %s\n' "${OUT_ROOT}"
    printf '%q ' "${configure_cmd[@]}"; printf '> %q 2>&1\n' "${OUT_ROOT}/profiling-debug-build.log"
    printf '%q ' "${build_cmd[@]}"; printf '>> %q 2>&1\n' "${OUT_ROOT}/profiling-debug-build.log"
    printf 'env ONEAPI_DEVICE_SELECTOR=%q vtune -collect gpu-hotspots -knob target-gpu=0:7:0.0 -knob gpu-profiling-mode=source-analysis -knob source-analysis=mem-latency -knob dump-compute-task-binaries=true -knob computing-tasks-of-interest=%q -result-dir %q -- ' "${DEVICE_SELECTOR}" "${TASK_GLOB}#1#1#20" "${vtune_dir}"
    printf '%q ' "${bench_cmd[@]}"; printf '> %q 2> %q\n' "${OUT_ROOT}/bench.stdout" "${OUT_ROOT}/bench.stderr"
    printf 'readelf -S %q > %q\n' "${vtune_dir}/data.0/<first-zebin>" "${OUT_ROOT}/zebin-debug-sections.txt"
    printf 'vtune -report hotspots -r %q -group-by gpu-source-line -format csv > %q\n' "${vtune_dir}" "${OUT_ROOT}/vtune-gpu-source-line.csv"
    if [[ -n "${REQUIRE_MATRIX_PASS}" ]]; then
        printf 'matrix gate: require %q to contain source_line.status pass\n' "${REQUIRE_MATRIX_PASS}"
        printf 'grep -qx %q %q\n' "source_line.status pass" "${REQUIRE_MATRIX_PASS}"
    fi
    printf 'python3 scripts/check-sycl-vtune-source-lines.py --readelf-sections %q --vtune-csv %q --require-kernel %q > %q\n' "${OUT_ROOT}/zebin-debug-sections.txt" "${OUT_ROOT}/vtune-gpu-source-line.csv" "mxfp4_pair_glu_xmx_tiled" "${OUT_ROOT}/source-line-feasibility.parse"
}

if [[ "${EXECUTE}" -ne 1 ]]; then
    print_plan
    exit 0
fi

if [[ -n "${REQUIRE_MATRIX_PASS}" ]] && ! grep -qx "source_line.status pass" "${REQUIRE_MATRIX_PASS}"; then
    printf 'error: MXFP4 source-line matrix gate failed: %s must contain source_line.status pass\n' "${REQUIRE_MATRIX_PASS}" >&2
    exit 2
fi

mkdir -p "${OUT_ROOT}"
set +u
source /opt/intel/oneapi/setvars.sh --force >"${OUT_ROOT}/setvars.log" 2>&1
set -u
"${configure_cmd[@]}" >"${OUT_ROOT}/profiling-debug-build.log" 2>&1
"${build_cmd[@]}" >>"${OUT_ROOT}/profiling-debug-build.log" 2>&1
env ONEAPI_DEVICE_SELECTOR="${DEVICE_SELECTOR}" vtune -collect gpu-hotspots \
    -knob target-gpu=0:7:0.0 \
    -knob gpu-profiling-mode=source-analysis \
    -knob source-analysis=mem-latency \
    -knob dump-compute-task-binaries=true \
    -knob computing-tasks-of-interest="${TASK_GLOB}#1#1#20" \
    -result-dir "${vtune_dir}" \
    -- "${bench_cmd[@]}" >"${OUT_ROOT}/bench.stdout" 2>"${OUT_ROOT}/bench.stderr"
first_zebin="$(find "${vtune_dir}" -path '*/data.0/*.zebin' -type f | head -n 1)"
readelf -S "${first_zebin}" >"${OUT_ROOT}/zebin-debug-sections.txt"
vtune -report hotspots -r "${vtune_dir}" -group-by gpu-source-line -format csv >"${OUT_ROOT}/vtune-gpu-source-line.csv"
python3 scripts/check-sycl-vtune-source-lines.py \
    --readelf-sections "${OUT_ROOT}/zebin-debug-sections.txt" \
    --vtune-csv "${OUT_ROOT}/vtune-gpu-source-line.csv" \
    --require-kernel "mxfp4_pair_glu_xmx_tiled" >"${OUT_ROOT}/source-line-feasibility.parse"
printf 'Artifacts: %s\n' "${OUT_ROOT}"
