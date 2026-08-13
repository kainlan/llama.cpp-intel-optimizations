#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_SCRIPT="${ROOT_DIR}/scripts/sycl-build.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

MOCK_BIN="${TMP}/bin"
CCL_ROOT_VERSION="${TMP}/oneapi/ccl/2022.1"
CCL_ROOT_FIXTURE="${TMP}/oneapi/ccl/latest"
BUILD_DIR="${TMP}/build"
CMAKE_LOG="${TMP}/cmake.log"
mkdir -p "${MOCK_BIN}" "${CCL_ROOT_VERSION}" "${BUILD_DIR}"
ln -s "${CCL_ROOT_VERSION}" "${CCL_ROOT_FIXTURE}"

cat > "${TMP}/setvars.sh" <<'EOF'
export PATH="${MOCK_BIN}:${PATH}"
export CCL_ROOT="${TEST_CCL_ROOT}"
EOF

cat > "${MOCK_BIN}/cmake" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' --- "$@" >> "${CMAKE_LOG}"
EOF

for tool in ninja icx icpx; do
    cat > "${MOCK_BIN}/${tool}" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod +x "${MOCK_BIN}/${tool}"
done
chmod +x "${MOCK_BIN}/cmake"

run_script() {
    env \
        ONEAPI_SETVARS="${TMP}/setvars.sh" \
        MOCK_BIN="${MOCK_BIN}" \
        TEST_CCL_ROOT="${CCL_ROOT_FIXTURE}" \
        CMAKE_LOG="${CMAKE_LOG}" \
        CMAKE_BUILD_PARALLEL_LEVEL=1 \
        "${BUILD_SCRIPT}" -B "${BUILD_DIR}" 2>&1
}

expect_failure_containing() {
    local expected="$1" output rc=0
    output="$(run_script)" || rc=$?
    if [[ ${rc} -ne 1 ]]; then
        echo "FAIL: expected discovery failure status 1, got ${rc}" >&2
        exit 1
    fi
    if ! grep -Fq -- "${expected}" <<< "${output}"; then
        echo "FAIL: missing diagnostic: ${expected}" >&2
        echo "${output}" >&2
        exit 1
    fi
}

# Discovery must stop before configure when either required oneCCL artifact is
# absent. These controls keep the checks behavioral rather than source-grep-only.
expect_failure_containing "include/oneapi/ccl.hpp"
mkdir -p "${CCL_ROOT_FIXTURE}/include/oneapi"
: > "${CCL_ROOT_FIXTURE}/include/oneapi/ccl.hpp"
expect_failure_containing "lib/cmake/oneCCL/oneCCLConfig.cmake"

ONECCL_DIR="${CCL_ROOT_VERSION}/lib/cmake/oneCCL"
mkdir -p "${ONECCL_DIR}"
: > "${ONECCL_DIR}/oneCCLConfig.cmake"

# A valid-looking pre-existing Ninja build with a stale package path must be
# reconfigured in place. The marker proves this does not silently clean it.
cat > "${BUILD_DIR}/CMakeCache.txt" <<EOF
CMAKE_GENERATOR:INTERNAL=Ninja
CMAKE_C_COMPILER:FILEPATH=${MOCK_BIN}/icx
CMAKE_CXX_COMPILER:FILEPATH=${MOCK_BIN}/icpx
CMAKE_C_FLAGS_RELEASE:STRING=-O3 -DNDEBUG
CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG
oneCCL_DIR:PATH=/opt/intel/oneapi/ccl/old/lib/cmake/oneCCL
EOF
: > "${BUILD_DIR}/build.ninja"
: > "${BUILD_DIR}/keep-me"
touch -t 203701010000 "${BUILD_DIR}/build.ninja"

output="$(run_script)"

for expected_arg in \
    "-UoneCCL_DIR" \
    "-DoneCCL_DIR=${ONECCL_DIR}"; do
    if ! grep -Fxq -- "${expected_arg}" "${CMAKE_LOG}"; then
        echo "FAIL: CMake did not receive ${expected_arg}" >&2
        cat "${CMAKE_LOG}" >&2
        exit 1
    fi
done

if [[ ! -f "${BUILD_DIR}/keep-me" ]]; then
    echo "FAIL: stale oneCCL cache handling destructively cleaned the build" >&2
    exit 1
fi

for expected_output in \
    "[sycl-build] C compiler: ${MOCK_BIN}/icx" \
    "[sycl-build] C++ compiler: ${MOCK_BIN}/icpx" \
    "[sycl-build] oneCCL: ${ONECCL_DIR}" \
    "refreshing cached oneCCL_DIR:"; do
    if ! grep -Fq -- "${expected_output}" <<< "${output}"; then
        echo "FAIL: missing output: ${expected_output}" >&2
        echo "${output}" >&2
        exit 1
    fi
done

echo "test-sycl-build-oneccl-discovery: PASS" >&2
