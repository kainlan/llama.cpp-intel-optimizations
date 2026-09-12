#!/usr/bin/env bash
#
# Configure-only smoke test for .devops/intel.Dockerfile's build-stage cmake
# flags (llama.cpp-9goq). The Dockerfile's server-image build stage
# configures with GGML_BACKEND_DL=ON + GGML_CPU_ALL_VARIANTS=ON +
# LLAMA_BUILD_TESTS=OFF, and the fork's GitHub Actions are disabled (memory
# fork-github-actions-disabled-manually), so nothing else exercises that
# combination. This script is the substitute: it re-derives the exact -D
# flags from the Dockerfile itself (never a hardcoded copy that can drift
# out of sync with it) and runs a throwaway `cmake -S -B` with them -- no
# build step, so this is safe to run without a GPU and without restricting
# to a `--target`.
#
# Usage: ./scripts/sycl-dockerfile-configure-check.sh [REPO_ROOT]
#
# REPO_ROOT defaults to this checkout. Pass another tree (e.g. /Apps/llama.cpp)
# to run the identical check read-only against a different checkout -- the
# script only ever calls `cmake -S REPO_ROOT -B <its own throwaway tmpdir>`,
# so it never writes into REPO_ROOT.
#
# Exit codes:
#   0   configure succeeded with the Dockerfile's exact flag set
#   1   configure failed, or the Dockerfile's cmake line could not be parsed
#   77  oneAPI not sourced / icx or icpx not on PATH (ctest SKIP_RETURN_CODE)

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/sycl-dockerfile-configure-check.sh [REPO_ROOT]

Configure-only: extracts the -D flags from .devops/intel.Dockerfile's own
`cmake -B build` line, runs `cmake -S REPO_ROOT -B <tmpdir>` with them, prints
the flags and any CMake errors, then deletes the tmpdir. No build step.

REPO_ROOT defaults to this repo.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${1:-${DEFAULT_ROOT}}" && pwd)"
DOCKERFILE="${REPO_ROOT}/.devops/intel.Dockerfile"

if [[ ! -f "${DOCKERFILE}" ]]; then
    echo "error: ${DOCKERFILE} not found" >&2
    exit 1
fi

# --- Source oneAPI if icx/icpx are not already on PATH -----------------------
if ! command -v icx >/dev/null 2>&1 || ! command -v icpx >/dev/null 2>&1; then
    ONEAPI_SETVARS="${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"
    if [[ ! -f "${ONEAPI_SETVARS}" ]]; then
        echo "SKIP: oneAPI not sourced and ${ONEAPI_SETVARS} not found" >&2
        exit 77
    fi
    # set -u is incompatible with sourcing setvars.sh, which reads unset
    # variables (memory: set-u-kills-a-script-that-sources-oneapi-setvars).
    set +u
    # shellcheck disable=SC1090
    source "${ONEAPI_SETVARS}" --force >/dev/null
    set -u
fi

if ! command -v icx >/dev/null 2>&1 || ! command -v icpx >/dev/null 2>&1; then
    echo "SKIP: icx/icpx still not on PATH after sourcing oneAPI" >&2
    exit 77
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "SKIP: cmake not found in PATH" >&2
    exit 77
fi

# -G Ninja is this script's own choice (see the PASS message below), not the
# Dockerfile's -- the Dockerfile's cmake line names no generator. Check for it
# here too so a missing generator reads as SKIP, not a confusing CMake error.
if ! command -v ninja >/dev/null 2>&1; then
    echo "SKIP: ninja not found in PATH" >&2
    exit 77
fi

# --- Extract the Dockerfile's OWN -D flags, never a hardcoded copy ----------
# The build stage's cmake invocation lives in one RUN block:
#   RUN if [ "${GGML_SYCL_F16}" = "ON" ]; then \
#           ... && export OPT_SYCL_F16="-DGGML_SYCL_F16=ON" ... ; fi && \
#       ... && \
#       cmake -B build -DGGML_NATIVE=OFF -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx \
#             -DCMAKE_CXX_COMPILER=icpx -DGGML_BACKEND_DL=ON \
#             -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=OFF ${OPT_SYCL_F16} && \
#       cmake --build build --config Release -j$(nproc)
# Capture the whole block (RUN line through the matching `cmake --build`
# line) so a future reflow across more lines is still captured, then:
#   1. Grab every literal -DNAME=VALUE token on the `cmake -B build` line.
#   2. Resolve any ${VAR} token on that line against an
#      `export VAR="..."` assignment earlier in the same block, so a flag
#      gated behind the shell `if` (GGML_SYCL_F16) is not silently dropped.
run_block="$(awk '/^RUN .*GGML_SYCL_F16/ { flag = 1 } flag { print } /cmake --build build/ { if (flag) exit }' "${DOCKERFILE}")"

if [[ -z "${run_block}" ]]; then
    echo "error: could not locate the cmake -B build RUN block in ${DOCKERFILE}" >&2
    exit 1
fi

cmake_line="$(printf '%s\n' "${run_block}" | grep -m1 'cmake -B build' || true)"
if [[ -z "${cmake_line}" ]]; then
    echo "error: could not find a 'cmake -B build' line inside the extracted RUN block" >&2
    exit 1
fi

flags=()
while IFS= read -r flag; do
    [[ -n "${flag}" ]] && flags+=("${flag}")
done < <(grep -oE -- '-D[A-Za-z0-9_]+=[^ \\"]+' <<<"${cmake_line}")

# Resolve ${VAR} references on the cmake line against export assignments
# earlier in the same RUN block.
while IFS= read -r var_ref; do
    var_name="${var_ref#\$\{}"
    var_name="${var_name%\}}"
    export_line="$(printf '%s\n' "${run_block}" | grep -m1 "export ${var_name}=" || true)"
    if [[ -n "${export_line}" ]]; then
        var_value="$(sed -E "s/.*export ${var_name}=\"([^\"]*)\".*/\\1/" <<<"${export_line}")"
        while IFS= read -r extra_flag; do
            [[ -n "${extra_flag}" ]] && flags+=("${extra_flag}")
        done < <(grep -oE -- '-D[A-Za-z0-9_]+=[^ "]+' <<<"${var_value}")
    fi
done < <(grep -oE -- '\$\{[A-Za-z0-9_]+\}' <<<"${cmake_line}")

if [[ ${#flags[@]} -eq 0 ]]; then
    echo "error: parsed zero -D flags out of the Dockerfile's cmake line: ${cmake_line}" >&2
    exit 1
fi

echo "Parsed Dockerfile configure flags (from ${DOCKERFILE}):"
printf '  %s\n' "${flags[@]}"

# --- Configure-only, throwaway build dir -------------------------------------
tmp_build="$(mktemp -d "${TMPDIR:-/tmp}/sycl-dockerfile-configure-check.XXXXXX")"
cleanup() { rm -rf "${tmp_build}"; }
trap cleanup EXIT

set +e
cmake_out="$(cmake -S "${REPO_ROOT}" -B "${tmp_build}" -G Ninja "${flags[@]}" 2>&1)"
rc=$?
set -e

echo "${cmake_out}"

if [[ ${rc} -ne 0 ]]; then
    echo "FAIL: cmake configure exited ${rc} against ${REPO_ROOT}" >&2
    exit 1
fi

echo "PASS: configure succeeded against ${REPO_ROOT} with the Dockerfile's flag set (plus -G Ninja)"
