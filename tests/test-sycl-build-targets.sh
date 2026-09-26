#!/usr/bin/env bash
# scripts/sycl-build.sh passes every named target to one build, expands --dev,
# and runs the build in a private TMPDIR that is gone once the script exits,
# whether the build passed or failed (llama.cpp-vuy0).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_SCRIPT="${ROOT_DIR}/scripts/sycl-build.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

MOCK_BIN="${TMP}/bin"
CCL_ROOT_FIXTURE="${TMP}/oneapi/ccl/2022.1"
BUILD_DIR="${TMP}/build"
CMAKE_LOG="${TMP}/cmake.log"
OUTER_TMP="${TMP}/outer-tmp"
mkdir -p "${MOCK_BIN}" "${CCL_ROOT_FIXTURE}/include/oneapi" \
    "${CCL_ROOT_FIXTURE}/lib/cmake/oneCCL" "${BUILD_DIR}" "${OUTER_TMP}"
: > "${CCL_ROOT_FIXTURE}/include/oneapi/ccl.hpp"
: > "${CCL_ROOT_FIXTURE}/lib/cmake/oneCCL/oneCCLConfig.cmake"

cat > "${TMP}/setvars.sh" <<'EOF'
export PATH="${MOCK_BIN}:${PATH}"
export CCL_ROOT="${TEST_CCL_ROOT}"
EOF

# The mock records each build invocation and the TMPDIR it ran under, leaves
# a file there the way a compiler would, and fails on request.
cat > "${MOCK_BIN}/cmake" <<'EOF'
#!/usr/bin/env bash
if [[ "$1" == "--build" ]]; then
    printf 'TMPDIR=%s\n' "${TMPDIR}" >> "${CMAKE_LOG}"
    printf '%s\n' "$@" >> "${CMAKE_LOG}"
    [[ -d "${TMPDIR}" ]] || exit 90
    : > "${TMPDIR}/icpx-stranded.out"
    exit "${MOCK_BUILD_RC:-0}"
fi
EOF
for tool in ninja icx icpx; do
    printf '#!/usr/bin/env bash\nexit 0\n' > "${MOCK_BIN}/${tool}"
    chmod +x "${MOCK_BIN}/${tool}"
done
chmod +x "${MOCK_BIN}/cmake"

run_script() {
    : > "${CMAKE_LOG}"
    env \
        ONEAPI_SETVARS="${TMP}/setvars.sh" \
        MOCK_BIN="${MOCK_BIN}" \
        TEST_CCL_ROOT="${CCL_ROOT_FIXTURE}" \
        CMAKE_LOG="${CMAKE_LOG}" \
        CMAKE_BUILD_PARALLEL_LEVEL=1 \
        TMPDIR="${OUTER_TMP}" \
        "${BUILD_SCRIPT}" -B "${BUILD_DIR}" "$@" > "${TMP}/out.log" 2>&1
}

# Arguments cmake --build received after --target, up to "--" or the end.
build_targets() {
    awk '/^--target$/ { on = 1; next } /^--$/ { on = 0 } on' "${CMAKE_LOG}" | paste -sd' '
}

build_tmpdir() {
    sed -n 's/^TMPDIR=//p' "${CMAKE_LOG}"
}

expect_private_tmp_removed() {
    local used
    used="$(build_tmpdir)"
    [[ -n "${used}" ]] || fail "the build did not run"
    [[ "${used}" == "${OUTER_TMP}"/sycl-build.* ]] ||
        fail "build TMPDIR ${used} is not private under ${OUTER_TMP}"
    [[ ! -e "${used}" ]] || fail "private TMPDIR ${used} survived the script"
    [[ -z "$(ls -A "${OUTER_TMP}")" ]] || fail "left behind: $(ls -A "${OUTER_TMP}")"
}

# Several targets reach a single build, in order.
run_script llama-completion test-mem-ops llama-bench
[[ "$(build_targets)" == "llama-completion test-mem-ops llama-bench" ]] ||
    fail "targets: '$(build_targets)'"
expect_private_tmp_removed

# --dev adds the everyday tools alongside anything named.
run_script --dev test-mem-ops
[[ "$(build_targets)" == "llama-bench llama-cli llama-completion test-mem-ops" ]] ||
    fail "--dev targets: '$(build_targets)'"

# No target builds everything; arguments after -- still reach the build tool.
run_script -- -v
[[ -z "$(build_targets)" ]] || fail "unexpected targets: '$(build_targets)'"
tail -n 2 "${CMAKE_LOG}" | paste -sd' ' | grep -qx -- '-- -v' ||
    fail "extra build args lost: $(cat "${CMAKE_LOG}")"

# A failing build keeps its status and still removes its temporaries.
rc=0
MOCK_BUILD_RC=7 run_script llama-cli || rc=$?
[[ ${rc} -eq 7 ]] || fail "failing build returned ${rc}, want 7"
expect_private_tmp_removed

echo "test-sycl-build-targets: PASS"
