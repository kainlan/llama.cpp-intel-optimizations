#!/usr/bin/env bash
# scripts/sycl-build.sh passes every named target to one build, expands --dev,
# runs the build in a private TMPDIR that is gone once the script exits,
# whether the build passed or failed, stops the build and exits 130/143/129 on
# INT/TERM/HUP, refuses options it does not know, and
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
    if [[ -n "${MOCK_PID_FILE:-}" ]]; then
        # Like cmake --build: no handler, so the signal kills it at once, while
        # its ninja child in the same process group takes a while to reap its
        # jobs and clean up.
        "${MOCK_BIN}/mock-ninja" "${MOCK_NINJA_READY}" "${MOCK_NINJA_DONE}" &
        ninja_pid=$!
        # Bounded, and inside signal_case's 5 s wait for it: the mock runs in
        # the script's build group, out of reach of the test's exit and
        # ctest's TIMEOUT, so it must not spin forever.
        for _ in $(seq 200); do
            [[ -e "${MOCK_NINJA_READY}" ]] && break
            sleep 0.01
        done
        if [[ ! -e "${MOCK_NINJA_READY}" ]]; then
            kill -KILL "${ninja_pid}" 2>/dev/null
            exit 91
        fi
        echo "$$" > "${MOCK_PID_FILE}"
        wait
        exit 0
    fi
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
# Python, not bash: an async child of a non-interactive shell starts with
# SIGINT ignored, which bash cannot trap but Python can override -- real
# ninja is exec'd by cmake with default dispositions.
cat > "${MOCK_BIN}/mock-ninja" <<'EOF'
#!/usr/bin/env python3
import os, signal, sys, time
ready, done = sys.argv[1], sys.argv[2]
def stop(signum, frame):
    time.sleep(1)
    with open(done, "w") as f:
        f.write(str(os.getpid()))
    sys.exit(1)
for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(s, stop)
open(ready, "w").close()
while True:
    time.sleep(0.1)
EOF
chmod +x "${MOCK_BIN}/mock-ninja"

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

# A signal to the script's pid alone -- not its process group -- stops the
# build in flight, ends the script with the conventional status and still
# removes its temporaries, but only once the whole build has stopped: the mock
# cmake dies at once, and its ninja child needs another second to clean up.
# With "stopped", the build group is SIGSTOPped first (as a background job
# writing under stty tostop would be); the signal must still end it.
signal_case() {
    local sig="$1" want_rc="$2" stopped="${3:-}" pid build_pid child rc=0 start_ms elapsed_ms
    : > "${CMAKE_LOG}"
    : > "${CONFIGURE_LOG}"
    rm -f "${TMP}/build.pid" "${TMP}/ninja.ready" "${TMP}/ninja.done"
    # set -m: default dispositions, as from a terminal; a plain background job
    # would start with SIGINT ignored.
    set -m
    env ONEAPI_SETVARS="${TMP}/setvars.sh" MOCK_BIN="${MOCK_BIN}" TEST_CCL_ROOT="${CCL_ROOT_FIXTURE}" \
        CMAKE_LOG="${CMAKE_LOG}" CONFIGURE_LOG="${CONFIGURE_LOG}" CMAKE_BUILD_PARALLEL_LEVEL=1 \
        TMPDIR="${OUTER_TMP}" MOCK_PID_FILE="${TMP}/build.pid" \
        MOCK_NINJA_READY="${TMP}/ninja.ready" MOCK_NINJA_DONE="${TMP}/ninja.done" \
        "${BUILD_SCRIPT}" -B "${BUILD_DIR}" llama-cli > "${TMP}/out.log" 2>&1 &
    pid=$!
    set +m
    for _ in $(seq 100); do
        [[ -s "${TMP}/build.pid" ]] && break
        sleep 0.05
    done
    if [[ ! -s "${TMP}/build.pid" ]]; then
        # The script and its build run in process groups of their own, which
        # neither this test's exit nor ctest's TIMEOUT reaches.
        for child in $(pgrep -P "${pid}"); do
            kill -KILL -- "-${child}" 2>/dev/null || true
        done
        kill -KILL -- "-${pid}" 2>/dev/null || true
        fail "SIG${sig}: the build never started: $(cat "${TMP}/out.log")"
    fi
    build_pid="$(cat "${TMP}/build.pid")"
    if [[ -n "${stopped}" ]]; then
        kill -STOP -- "-${build_pid}"
    fi
    start_ms=$(( $(date +%s%N) / 1000000 ))
    kill "-${sig}" "${pid}"
    for _ in $(seq 100); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 0.05
    done
    if kill -0 "${pid}" 2>/dev/null; then
        kill -KILL -- "-${build_pid}" "-${pid}" 2>/dev/null || true
        fail "SIG${sig}${stopped:+ (build stopped)}: the script did not return within 5 s"
    fi
    wait "${pid}" || rc=$?
    elapsed_ms=$(( $(date +%s%N) / 1000000 - start_ms ))
    [[ -e "${TMP}/ninja.done" ]] ||
        fail "SIG${sig}: the script returned while ninja was still cleaning up"
    [[ ${rc} -eq ${want_rc} ]] || fail "SIG${sig}: script exited ${rc}, want ${want_rc}"
    # The mock build runs until signalled and stops within ~1 s; much longer
    # means the signal waited for the build instead of stopping it.
    (( elapsed_ms < 3000 )) || fail "SIG${sig}: the script took ${elapsed_ms} ms to stop the build"
    if kill -0 -- "-${build_pid}" 2>/dev/null; then
        kill -KILL -- "-${build_pid}" 2>/dev/null || true
        fail "SIG${sig}: the build's process group ${build_pid} outlived the script"
    fi
    expect_private_tmp_removed
}
signal_case INT 130
signal_case TERM 143
signal_case HUP 129
signal_case INT 130 stopped

# An unknown option is refused before anything runs, pointing at "--": passed
# through bare, a value-taking one such as "-k 0" would split into an option
# and a target.
rc=0
run_script llama-cli -k 0 || rc=$?
[[ ${rc} -eq 2 ]] || fail "unknown option: script exited ${rc}, want 2"
[[ ! -s "${CMAKE_LOG}" && ! -s "${CONFIGURE_LOG}" ]] || fail "unknown option: cmake ran: $(cat "${CMAKE_LOG}")"
grep -q -- "unknown option '-k'.*after --" "${TMP}/out.log" ||
    fail "unknown option: no usage error pointing at --: $(cat "${TMP}/out.log")"

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
