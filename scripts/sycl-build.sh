#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/sycl-build.sh [options] [target...] [-- <extra build args>]

Configure and build the SYCL backend with Ninja in build/.

With no target this builds everything, including every SYCL test executable;
each of those embeds the backend objects and pays its own device link. Name
the targets you will run, or use --dev, to skip them.

Options:
  -r, --reconfigure       Force CMake reconfigure before building
  -c, --clean             Remove build/ and configure from scratch
  -B, --build-dir <dir>   Override build directory (default: build)
      --dev               Add the everyday tools: llama-bench, llama-cli,
                          llama-completion
  -h, --help              Show this help

Compiler and device-link temporaries go to a private directory under
${TMPDIR:-/tmp} that is removed when the script exits.

Examples:
  ./scripts/sycl-build.sh
  ./scripts/sycl-build.sh llama-completion
  ./scripts/sycl-build.sh --dev test-mem-ops
  ./scripts/sycl-build.sh -r llama-bench
  ./scripts/sycl-build.sh -c
  ./scripts/sycl-build.sh llama-completion -- -v
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

force_reconfigure=0
clean_build=0
targets=()
extra_build_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--reconfigure)
            force_reconfigure=1
            shift
            ;;
        -c|--clean)
            clean_build=1
            force_reconfigure=1
            shift
            ;;
        -B|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --dev)
            targets+=(llama-bench llama-cli llama-completion)
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            extra_build_args=("$@")
            break
            ;;
        -*)
            extra_build_args+=("$1")
            shift
            ;;
        *)
            targets+=("$1")
            shift
            ;;
    esac
done

ONEAPI_SETVARS="${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"
if [[ ! -f "${ONEAPI_SETVARS}" ]]; then
    echo "error: ${ONEAPI_SETVARS} not found" >&2
    exit 1
fi

set +u
# shellcheck disable=SC1090
source "${ONEAPI_SETVARS}" --force >/dev/null
set -u

if ! command -v cmake >/dev/null 2>&1; then
    echo "error: cmake not found in PATH" >&2
    exit 1
fi

if ! command -v ninja >/dev/null 2>&1; then
    echo "error: ninja not found in PATH" >&2
    exit 1
fi

if ! command -v icx >/dev/null 2>&1 || ! command -v icpx >/dev/null 2>&1; then
    echo "error: icx/icpx not found after sourcing oneAPI" >&2
    exit 1
fi

if [[ -z "${CCL_ROOT:-}" || ! -d "${CCL_ROOT}" ]]; then
    echo "error: CCL_ROOT is not a directory after sourcing oneAPI: ${CCL_ROOT:-<unset>}" >&2
    exit 1
fi

# Resolve a possible `latest` symlink once so both CMake and diagnostics identify
# the exact oneCCL installation selected by setvars.sh.
CCL_ROOT="$(cd -P "${CCL_ROOT}" && pwd)"
ONECCL_DIR="${CCL_ROOT}/lib/cmake/oneCCL"

if [[ ! -f "${CCL_ROOT}/include/oneapi/ccl.hpp" ]]; then
    echo "error: oneCCL header not found: ${CCL_ROOT}/include/oneapi/ccl.hpp" >&2
    exit 1
fi

if [[ ! -f "${ONECCL_DIR}/oneCCLConfig.cmake" ]]; then
    echo "error: oneCCL CMake config not found: ${ONECCL_DIR}/oneCCLConfig.cmake" >&2
    exit 1
fi

C_COMPILER="$(command -v icx)"
CXX_COMPILER="$(command -v icpx)"
echo "[sycl-build] C compiler: ${C_COMPILER}"
echo "[sycl-build] C++ compiler: ${CXX_COMPILER}"
echo "[sycl-build] oneCCL: ${ONECCL_DIR}"

if (( clean_build )); then
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

# icpx strands its temporaries (hundreds of MB per device link) in TMPDIR when
# a build is interrupted; give this build its own directory and remove it on
# exit. Only SIGKILL escapes this -- scripts/sycl-tmp-leak-report.sh lists
# what is left behind.
build_tmp="$(mktemp -d "${TMPDIR:-/tmp}/sycl-build.XXXXXX")"
trap 'rm -rf "${build_tmp}"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
export TMPDIR="${build_tmp}"

cmake_input_changed() {
    local stamp="${BUILD_DIR}/build.ninja"

    if [[ ! -f "${stamp}" ]]; then
        return 0
    fi

    find "${ROOT_DIR}" \
        -path "${BUILD_DIR}" -prune -o \
        \( -name 'CMakeLists.txt' -o -path "${ROOT_DIR}/cmake/*.cmake" \) \
        -newer "${stamp}" -print -quit | grep -q .
}

needs_configure=0

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" || ! -f "${BUILD_DIR}/build.ninja" ]]; then
    needs_configure=1
fi

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]] && ! grep -q '^CMAKE_GENERATOR:INTERNAL=Ninja$' "${BUILD_DIR}/CMakeCache.txt"; then
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    needs_configure=1
fi

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]] && ! grep -q '^CMAKE_C_COMPILER:FILEPATH=.*/icx$' "${BUILD_DIR}/CMakeCache.txt"; then
    needs_configure=1
fi

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]] && ! grep -q '^CMAKE_CXX_COMPILER:FILEPATH=.*/icpx$' "${BUILD_DIR}/CMakeCache.txt"; then
    needs_configure=1
fi

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]] && {
    ! grep -q '^CMAKE_C_FLAGS_RELEASE:STRING=.*DNDEBUG' "${BUILD_DIR}/CMakeCache.txt" ||
        ! grep -q '^CMAKE_CXX_FLAGS_RELEASE:STRING=.*DNDEBUG' "${BUILD_DIR}/CMakeCache.txt";
}; then
    needs_configure=1
fi

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    cached_oneccl_dir="$(grep '^oneCCL_DIR:' "${BUILD_DIR}/CMakeCache.txt" | tail -n 1 | cut -d= -f2- || true)"
    if [[ "${cached_oneccl_dir}" != "${ONECCL_DIR}" ]]; then
        echo "[sycl-build] refreshing cached oneCCL_DIR: ${cached_oneccl_dir:-<unset>} -> ${ONECCL_DIR}"
        needs_configure=1
    fi
fi

if (( force_reconfigure )) || cmake_input_changed; then
    needs_configure=1
fi

configure_args=(
    -S "${ROOT_DIR}"
    -B "${BUILD_DIR}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    '-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG'
    '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'
    -DGGML_SYCL=ON
    -DGGML_SYCL_TARGET=INTEL
    -DGGML_SYCL_ONECCL=ON
    -UoneCCL_DIR
    "-DoneCCL_DIR=${ONECCL_DIR}"
    -DGGML_SYCL_F16=ON
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
    '-DCMAKE_INSTALL_RPATH=$ORIGIN'
    -DCMAKE_C_COMPILER=icx
    -DCMAKE_CXX_COMPILER=icpx
)

compiler_launcher=""
if command -v ccache >/dev/null 2>&1; then
    compiler_launcher="ccache"
    # base_dir makes ccache rewrite absolute paths under this tree relative to
    # the build directory, so a checkout at another path -- a worktree -- hits
    # the entries this one stored instead of recompiling from cold. It also
    # makes __FILE__ relative ("../ggml/src/..."); tests that locate the tree
    # get the absolute LLAMA_CPP_SOURCE_ROOT, which ccache does not rewrite.
    # `ccache KEY=VALUE compiler` needs ccache 4.8.
    ccache_version="$(ccache --version 2>/dev/null | sed -n '1s/^ccache version \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')"
    if [[ -n "${ccache_version}" ]] && read -r ccache_major ccache_minor <<< "${ccache_version}" &&
        (( ccache_major > 4 || (ccache_major == 4 && ccache_minor >= 8) )); then
        compiler_launcher="ccache;base_dir=${ROOT_DIR}"
    fi
    configure_args+=(
        "-DCMAKE_C_COMPILER_LAUNCHER=${compiler_launcher}"
        "-DCMAKE_CXX_COMPILER_LAUNCHER=${compiler_launcher}"
    )
fi

if [[ -n "${compiler_launcher}" && -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    cached_launcher="$(sed -n 's/^CMAKE_CXX_COMPILER_LAUNCHER:[A-Z]*=//p' "${BUILD_DIR}/CMakeCache.txt" | tail -n 1)"
    if [[ "${cached_launcher}" != "${compiler_launcher}" ]]; then
        echo "[sycl-build] refreshing compiler launcher: ${cached_launcher:-<unset>} -> ${compiler_launcher}"
        needs_configure=1
    fi
fi

if (( needs_configure )); then
    echo "[sycl-build] configuring ${BUILD_DIR}"
    cmake "${configure_args[@]}"
fi

jobs="${CMAKE_BUILD_PARALLEL_LEVEL:-}"
if [[ -z "${jobs}" ]]; then
    jobs="$(nproc)"
fi

build_cmd=(cmake --build "${BUILD_DIR}" --config Release -j "${jobs}")

if [[ ${#targets[@]} -gt 0 ]]; then
    build_cmd+=(--target "${targets[@]}")
fi

if [[ ${#extra_build_args[@]} -gt 0 ]]; then
    build_cmd+=(-- "${extra_build_args[@]}")
fi

echo "[sycl-build] building${targets[*]:+ targets ${targets[*]}} with Ninja in ${BUILD_DIR}"
"${build_cmd[@]}"
