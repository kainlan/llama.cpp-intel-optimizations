#!/usr/bin/env bash
set -euo pipefail

EXECUTE=0
ACK=0
OUT_ROOT="source-line"
DEFAULT_MATRIX_ROOT="source-line/build-matrix"
DEVICE_SELECTOR="level_zero:1"
TARGET_KERNEL="sycl_source_line_probe"
TASK_GLOB="*sycl_source_line_probe*"

CASE_NAMES=(release_split debug_line_tables debug_full debug_no_inline)
CASE_BUILD_TYPES=(Release Release RelWithDebInfo RelWithDebInfo)
CASE_CXX_FLAGS=(
    "-fsycl-device-code-split=per_kernel"
    "-gline-tables-only -fdebug-info-for-profiling -fsycl-instrument-device-code -fsycl-device-code-split=per_kernel"
    "-g -fdebug-info-for-profiling -fsycl-instrument-device-code -fsycl-device-code-split=per_kernel"
    "-g -fdebug-info-for-profiling -fsycl-instrument-device-code -fno-inline -fno-inline-functions -fsycl-device-code-split=per_kernel"
)
CASE_LINK_FLAGS=(
    "-fsycl-device-code-split=per_kernel"
    "-fsycl-instrument-device-code -fsycl-device-code-split=per_kernel"
    "-fsycl-instrument-device-code -fsycl-device-code-split=per_kernel"
    "-fsycl-instrument-device-code -fsycl-device-code-split=per_kernel"
)

usage() {
    printf 'usage: %s [--dry-run|--execute] [--i-understand-this-runs-gpu-source-probe] [--out-root DIR] [--device-selector SELECTOR]\n' "$0"
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
        --i-understand-this-runs-gpu-source-probe) ACK=1 ;;
        --out-root) require_value "$1" "${2-}"; OUT_ROOT="$2"; shift ;;
        --device-selector) require_value "$1" "${2-}"; DEVICE_SELECTOR="$2"; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if [[ "${EXECUTE}" -eq 1 && "${ACK}" -ne 1 ]]; then
    printf 'error: --execute requires --i-understand-this-runs-gpu-source-probe\n' >&2
    exit 2
fi

quote_cmd() {
    printf '%q ' "$@"
}

case_dir() {
    printf '%s/build-matrix/%s' "${OUT_ROOT}" "$1"
}

make_configure_cmd() {
    local name="$1"
    local build_type="$2"
    local cxx_flags="$3"
    local link_flags="$4"
    local dir
    dir="$(case_dir "${name}")"
    CONFIGURE_CMD=(
        cmake -S . -B "${dir}/build" -G Ninja
        "-DCMAKE_BUILD_TYPE=${build_type}"
        -DGGML_SYCL=ON
        -DGGML_SYCL_TARGET=INTEL
        -DGGML_SYCL_F16=ON
        -DGGML_SYCL_PROFILING_DEBUG=OFF
        -DCMAKE_C_COMPILER=icx
        -DCMAKE_CXX_COMPILER=icpx
        "-DCMAKE_CXX_FLAGS=${cxx_flags}"
        "-DCMAKE_EXE_LINKER_FLAGS=${link_flags}"
    )
}

make_build_cmd() {
    local name="$1"
    local dir
    dir="$(case_dir "${name}")"
    BUILD_CMD=(cmake --build "${dir}/build" --config "$2" --target sycl-source-line-probe -j "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}")
}

make_probe_cmd() {
    local name="$1"
    local dir
    dir="$(case_dir "${name}")"
    PROBE_CMD=("./${dir}/build/bin/sycl-source-line-probe" --iterations 100 --size 1048576 --json "${dir}/probe.json")
}

print_plan() {
    printf 'DRY RUN: source-line debug-info matrix; no commands are executed.\n'
    printf '# output root: %s\n' "${OUT_ROOT}"
    printf '# default matrix root: %s\n' "${DEFAULT_MATRIX_ROOT}"

    for index in "${!CASE_NAMES[@]}"; do
        local name="${CASE_NAMES[$index]}"
        local build_type="${CASE_BUILD_TYPES[$index]}"
        local cxx_flags="${CASE_CXX_FLAGS[$index]}"
        local link_flags="${CASE_LINK_FLAGS[$index]}"
        local dir
        local vtune_dir
        dir="$(case_dir "${name}")"
        vtune_dir="${dir}/vtune-source-line"

        make_configure_cmd "${name}" "${build_type}" "${cxx_flags}" "${link_flags}"
        make_build_cmd "${name}" "${build_type}"
        make_probe_cmd "${name}"

        printf '\n# matrix row: %s\n' "${name}"
        printf '# build_type=%s\n' "${build_type}"
        printf '# cxx_flags=%s\n' "${cxx_flags}"
        quote_cmd "${CONFIGURE_CMD[@]}"; printf '> %q 2>&1\n' "${dir}/configure.log"
        quote_cmd "${BUILD_CMD[@]}"; printf '> %q 2>&1\n' "${dir}/build.log"
        printf 'env ONEAPI_DEVICE_SELECTOR=%q vtune -collect gpu-hotspots -knob gpu-profiling-mode=source-analysis -knob source-analysis=mem-latency -knob dump-compute-task-binaries=true -knob computing-tasks-of-interest=%q -result-dir %q -- ' "${DEVICE_SELECTOR}" "${TASK_GLOB}#1#1#20" "${vtune_dir}"
        quote_cmd "${PROBE_CMD[@]}"; printf '> %q 2> %q\n' "${dir}/probe.stdout" "${dir}/probe.stderr"
        printf 'readelf -S %q > %q\n' "${vtune_dir}/data.0/<first-zebin>" "${dir}/zebin-debug-sections.txt"
        printf 'vtune -report hotspots -r %q -group-by gpu-source-line -format csv > %q\n' "${vtune_dir}" "${dir}/vtune-gpu-source-line.csv"
        printf 'python3 scripts/check-sycl-vtune-source-lines.py --readelf-sections %q --vtune-csv %q --require-kernel %q > %q\n' "${dir}/zebin-debug-sections.txt" "${dir}/vtune-gpu-source-line.csv" "${TARGET_KERNEL}" "${dir}/source-line-feasibility.parse"
    done
}

if [[ "${EXECUTE}" -ne 1 ]]; then
    print_plan
    exit 0
fi

mkdir -p "${OUT_ROOT}/build-matrix"
set +u
source /opt/intel/oneapi/setvars.sh --force >"${OUT_ROOT}/setvars.log" 2>&1
set -u

for index in "${!CASE_NAMES[@]}"; do
    name="${CASE_NAMES[$index]}"
    build_type="${CASE_BUILD_TYPES[$index]}"
    cxx_flags="${CASE_CXX_FLAGS[$index]}"
    link_flags="${CASE_LINK_FLAGS[$index]}"
    dir="$(case_dir "${name}")"
    vtune_dir="${dir}/vtune-source-line"
    mkdir -p "${dir}"

    make_configure_cmd "${name}" "${build_type}" "${cxx_flags}" "${link_flags}"
    make_build_cmd "${name}" "${build_type}"
    make_probe_cmd "${name}"

    "${CONFIGURE_CMD[@]}" >"${dir}/configure.log" 2>&1
    "${BUILD_CMD[@]}" >"${dir}/build.log" 2>&1
    env ONEAPI_DEVICE_SELECTOR="${DEVICE_SELECTOR}" vtune -collect gpu-hotspots \
        -knob gpu-profiling-mode=source-analysis \
        -knob source-analysis=mem-latency \
        -knob dump-compute-task-binaries=true \
        -knob computing-tasks-of-interest="${TASK_GLOB}#1#1#20" \
        -result-dir "${vtune_dir}" \
        -- "${PROBE_CMD[@]}" >"${dir}/probe.stdout" 2>"${dir}/probe.stderr"

    first_zebin="$(find "${vtune_dir}" -name '*.zebin' -type f -print | head -n 1)"
    if [[ -z "${first_zebin}" ]]; then
        printf 'error: no .zebin found for matrix row %s in %s\n' "${name}" "${vtune_dir}" >&2
        exit 1
    fi

    readelf -S "${first_zebin}" >"${dir}/zebin-debug-sections.txt"
    vtune -report hotspots -r "${vtune_dir}" -group-by gpu-source-line -format csv >"${dir}/vtune-gpu-source-line.csv"
    python3 scripts/check-sycl-vtune-source-lines.py \
        --readelf-sections "${dir}/zebin-debug-sections.txt" \
        --vtune-csv "${dir}/vtune-gpu-source-line.csv" \
        --require-kernel "${TARGET_KERNEL}" >"${dir}/source-line-feasibility.parse"
done

printf 'Artifacts: %s/build-matrix\n' "${OUT_ROOT}"
