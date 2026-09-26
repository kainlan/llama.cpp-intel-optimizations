#!/usr/bin/env bash
# scripts/sycl-build.sh passes every named target to one build, expands --dev,
# runs the build in a private TMPDIR that is gone once the script exits,
# whether the build passed or failed, exits 130/143/129 on INT/TERM/HUP, and
# configures ccache with base_dir set to the tree -- so another checkout path
# reuses its entries -- only when GGML_SYCL_CCACHE_BASE_DIR=1 (llama.cpp-vuy0).
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
CONFIGURE_LOG="${TMP}/configure.log"
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
    sleep "${MOCK_BUILD_SLEEP:-0}"
    : > "${TMPDIR}/icpx-stranded.out"
    exit "${MOCK_BUILD_RC:-0}"
fi
printf '%s\n' --- "$@" >> "${CONFIGURE_LOG}"
EOF
cat > "${MOCK_BIN}/ccache" <<'EOF'
#!/usr/bin/env bash
[[ "$1" == "--version" ]] && echo "ccache version ${MOCK_CCACHE_VERSION:-4.12.3}"
exit 0
EOF
chmod +x "${MOCK_BIN}/ccache"
for tool in ninja icx icpx; do
    printf '#!/usr/bin/env bash\nexit 0\n' > "${MOCK_BIN}/${tool}"
    chmod +x "${MOCK_BIN}/${tool}"
done
chmod +x "${MOCK_BIN}/cmake"

run_script() {
    : > "${CMAKE_LOG}"
    : > "${CONFIGURE_LOG}"
    env \
        ONEAPI_SETVARS="${TMP}/setvars.sh" \
        MOCK_BIN="${MOCK_BIN}" \
        TEST_CCL_ROOT="${CCL_ROOT_FIXTURE}" \
        CMAKE_LOG="${CMAKE_LOG}" \
        CONFIGURE_LOG="${CONFIGURE_LOG}" \
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

# A signal ends the script with the conventional status and still removes
# its temporaries. The build is in flight when the signal lands.
signal_case() {
    local sig="$1" want_rc="$2" pid rc=0
    : > "${CMAKE_LOG}"
    : > "${CONFIGURE_LOG}"
    # set -m: default dispositions, as from a terminal; a plain background job
    # would start with SIGINT ignored.
    set -m
    env ONEAPI_SETVARS="${TMP}/setvars.sh" MOCK_BIN="${MOCK_BIN}" TEST_CCL_ROOT="${CCL_ROOT_FIXTURE}" \
        CMAKE_LOG="${CMAKE_LOG}" CONFIGURE_LOG="${CONFIGURE_LOG}" CMAKE_BUILD_PARALLEL_LEVEL=1 \
        TMPDIR="${OUTER_TMP}" MOCK_BUILD_SLEEP=1 \
        "${BUILD_SCRIPT}" -B "${BUILD_DIR}" llama-cli > "${TMP}/out.log" 2>&1 &
    pid=$!
    set +m
    for _ in $(seq 100); do
        [[ -n "$(build_tmpdir)" ]] && break
        sleep 0.05
    done
    [[ -n "$(build_tmpdir)" ]] || fail "SIG${sig}: the build never started"
    kill "-${sig}" "${pid}"
    wait "${pid}" || rc=$?
    [[ ${rc} -eq ${want_rc} ]] || fail "SIG${sig}: script exited ${rc}, want ${want_rc}"
    expect_private_tmp_removed
}
signal_case INT 130
signal_case TERM 143
signal_case HUP 129

write_cache() {
    cat > "${BUILD_DIR}/CMakeCache.txt" <<EOF
CMAKE_GENERATOR:INTERNAL=Ninja
CMAKE_C_COMPILER:FILEPATH=${MOCK_BIN}/icx
CMAKE_CXX_COMPILER:FILEPATH=${MOCK_BIN}/icpx
CMAKE_C_FLAGS_RELEASE:STRING=-O3 -DNDEBUG
CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG
oneCCL_DIR:PATH=${CCL_ROOT_FIXTURE}/lib/cmake/oneCCL
CMAKE_CXX_COMPILER_LAUNCHER:UNINITIALIZED=$1
GGML_SYCL_PROFILING_DEBUG:BOOL=${2:-OFF}
EOF
    : > "${BUILD_DIR}/build.ninja"
    touch -t 203701010000 "${BUILD_DIR}/build.ninja"
}
expect_launcher() {
    local lang
    for lang in C CXX; do
        grep -Fxq -- "-DCMAKE_${lang}_COMPILER_LAUNCHER=$1" "${CONFIGURE_LOG}" ||
            fail "$2: ${lang} launcher is not '$1': $(grep LAUNCHER "${CONFIGURE_LOG}")"
    done
}
base_dir_launcher="ccache;base_dir=${ROOT_DIR}"

# By default ccache runs plain: base_dir is opt-in until it is verified on a
# real SYCL build.
rm -f "${BUILD_DIR}/CMakeCache.txt" "${BUILD_DIR}/build.ninja"
run_script
expect_launcher ccache "default"

# GGML_SYCL_CCACHE_BASE_DIR=1 sets base_dir at the tree root, for both compilers.
rm -f "${BUILD_DIR}/CMakeCache.txt" "${BUILD_DIR}/build.ninja"
GGML_SYCL_CCACHE_BASE_DIR=1 run_script
expect_launcher "${base_dir_launcher}" "opt-in"

# An existing build is moved onto whichever launcher is asked for, and left
# alone when it already has it.
write_cache "${base_dir_launcher}"
GGML_SYCL_CCACHE_BASE_DIR=1 run_script
[[ ! -s "${CONFIGURE_LOG}" ]] || fail "reconfigured a build already on '${base_dir_launcher}'"
write_cache ccache
GGML_SYCL_CCACHE_BASE_DIR=1 run_script
expect_launcher "${base_dir_launcher}" "plain -> opt-in"
grep -Fq 'refreshing compiler launcher: ccache ->' "${TMP}/out.log" ||
    fail "no launcher refresh message: $(cat "${TMP}/out.log")"
write_cache "${base_dir_launcher}"
run_script
expect_launcher ccache "base_dir -> default"

# ccache before 4.8 has no KEY=VALUE syntax and would take base_dir=... for the
# compiler; it keeps the plain launcher.
rm -f "${BUILD_DIR}/CMakeCache.txt" "${BUILD_DIR}/build.ninja"
GGML_SYCL_CCACHE_BASE_DIR=1 MOCK_CCACHE_VERSION=4.7.4 run_script
expect_launcher ccache "ccache 4.7"

# A GGML_SYCL_PROFILING_DEBUG build (-g) keeps plain ccache: its DWARF paths
# would become relative, and the zebin line-table tooling matches absolute ones.
write_cache ccache ON
GGML_SYCL_CCACHE_BASE_DIR=1 run_script
[[ ! -s "${CONFIGURE_LOG}" ]] || fail "moved a profiling-debug build onto base_dir"
grep -Fq 'GGML_SYCL_PROFILING_DEBUG' "${TMP}/out.log" ||
    fail "no note about keeping plain ccache for profiling debug: $(cat "${TMP}/out.log")"

echo "test-sycl-build-targets: PASS"
