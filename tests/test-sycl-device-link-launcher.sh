#!/usr/bin/env bash
# Behavioural test for ggml/src/ggml-sycl/sycl-device-link.sh, the linker
# launcher every SYCL device-linking target runs under (llama.cpp-vuy0).
# Host-only: the "compiler", "ocloc" and toolchain libraries are mocks, so no
# GPU, no oneAPI and no real device link is involved.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="${ROOT_DIR}/ggml/src/ggml-sycl/sycl-device-link.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

MOCK_BIN="${TMP}/bin"
PARENT_TMP="${TMP}/parent-tmp"
CACHE_ROOT="${TMP}/cache"
LOG="${TMP}/log"
mkdir -p "${MOCK_BIN}" "${PARENT_TMP}" "${CACHE_ROOT}" "${LOG}"

# The mock compiler records what the launcher handed it, then behaves as asked:
# `--version` prints ${MOCK_COMPILER_VERSION}; otherwise it creates a file in
# its TMPDIR (as icpx does), optionally runs ocloc like llvm-foreach would,
# optionally waits on a sleeping grandchild (as icpx waits on llvm-foreach and
# ocloc) so a signal can land, and exits ${MOCK_RC}.
cat > "${MOCK_BIN}/mock-icpx" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
    echo "${MOCK_COMPILER_VERSION:-mock icpx 1.0}"
    exit 0
fi
{
    echo "TMPDIR=${TMPDIR}"
    echo "NEO_CACHE_PERSISTENT=${NEO_CACHE_PERSISTENT:-<unset>}"
    echo "NEO_CACHE_DIR=${NEO_CACHE_DIR:-<unset>}"
    echo "NEO_CACHE_MAX_SIZE=${NEO_CACHE_MAX_SIZE:-<unset>}"
    echo "ARGS=$*"
} > "${MOCK_ENV_LOG}"
mkdir -p "${TMPDIR}/icpx-mock" && echo temp > "${TMPDIR}/icpx-mock/linked.bc"
if [[ -n "${MOCK_RUN_OCLOC:-}" ]]; then
    printf 'spirv-bytes' > "${TMPDIR}/img_0.spv"
    ocloc -output "${TMPDIR}/img_0.out" -file "${TMPDIR}/img_0.spv" \
        -output_no_suffix -spirv_input -device bmg_g21 -allow_caching || exit 9
fi
if [[ -n "${MOCK_RUN_OCLOC_NO_FILE:-}" ]]; then
    # An ocloc call without -file (e.g. a query), as the driver can make.
    ocloc -output "${TMPDIR}/query.out" -device bmg_g21 || exit 9
fi
if [[ -n "${MOCK_SLEEP:-}" ]]; then
    sleep "${MOCK_SLEEP}" &
    echo "$! $$" > "${MOCK_PID_FILE}.tmp"
    mv "${MOCK_PID_FILE}.tmp" "${MOCK_PID_FILE}"
    wait
fi
exit "${MOCK_RC:-0}"
EOF

# The mock ocloc behaves like NEO's persistent cache: with
# NEO_CACHE_PERSISTENT=1 and -allow_caching it serves a stored binary for the
# same input (logging "hit") or compiles and stores one ("miss").
cat > "${MOCK_BIN}/ocloc" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
    echo "${MOCK_OCLOC_VERSION:-26.31.1}"
    exit 0
fi
out="" in="" caching=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -output) out="$2"; shift 2 ;;
        -file) in="$2"; shift 2 ;;
        -allow_caching) caching=1; shift ;;
        *) shift ;;
    esac
done
entry=""
if [[ "${NEO_CACHE_PERSISTENT:-}" == 1 && ${caching} -eq 1 && -n "${in}" ]]; then
    entry="${NEO_CACHE_DIR}/$(md5sum < "${in}" | cut -d' ' -f1).bin"
fi
if [[ -n "${entry}" && -f "${entry}" ]]; then
    cp "${entry}" "${out}"
    echo hit >> "${MOCK_OCLOC_LOG:-/dev/null}"
else
    printf 'aot-binary' > "${out}"
    [[ -z "${entry}" ]] || cp "${out}" "${entry}"
    echo miss >> "${MOCK_OCLOC_LOG:-/dev/null}"
fi
EOF
chmod +x "${MOCK_BIN}/mock-icpx" "${MOCK_BIN}/ocloc"

KEY_FILE_A="${TMP}/libigc.so.2"
KEY_FILE_B="${TMP}/libocloc.so"
echo igc > "${KEY_FILE_A}"
echo ocloc-lib > "${KEY_FILE_B}"

LAUNCH_ENV=(
    PATH="${MOCK_BIN}:${PATH}"
    TMPDIR="${PARENT_TMP}"
    MOCK_ENV_LOG="${LOG}/env"
    GGML_SYCL_OCLOC_CACHE_ROOT="${CACHE_ROOT}"
    GGML_SYCL_OCLOC_KEY_FILES="${KEY_FILE_A}:${KEY_FILE_B}"
    GGML_SYCL_OCLOC_KEY_PKGS=""
)

launch() {
    env "${LAUNCH_ENV[@]}" "$@"
}

key() {
    launch "$@" "${LAUNCHER}" --print-cache-key "${MOCK_BIN}/mock-icpx"
}

assert_parent_tmp_empty() {
    if [[ -n "$(ls -A "${PARENT_TMP}")" ]]; then
        ls -la "${PARENT_TMP}" >&2
        fail "$1: private temp directory was left behind"
    fi
}

# 1. The command runs, its status is propagated, and it sees a PRIVATE TMPDIR
#    under the caller's TMPDIR that is gone once the launcher returns.
launch "${LAUNCHER}" -- "${MOCK_BIN}/mock-icpx" -o out.so a.o || fail "success case returned non-zero"
grep -q "^TMPDIR=${PARENT_TMP}/sycl-device-link\." "${LOG}/env" || fail "TMPDIR not private: $(cat "${LOG}/env")"
grep -q '^ARGS=-o out.so a.o$' "${LOG}/env" || fail "arguments not passed through verbatim"
assert_parent_tmp_empty "success"

rc=0
launch env MOCK_RC=3 "${LAUNCHER}" -- "${MOCK_BIN}/mock-icpx" || rc=$?
[[ ${rc} -eq 3 ]] || fail "exit status not propagated (want 3, got ${rc})"
assert_parent_tmp_empty "failure"

# 2. Without --ocloc-cache the NEO cache is not switched on.
launch "${LAUNCHER}" -- "${MOCK_BIN}/mock-icpx"
grep -q '^NEO_CACHE_PERSISTENT=<unset>$' "${LOG}/env" || fail "NEO cache enabled without --ocloc-cache"

# 3. With --ocloc-cache the child gets the persistent cache in a
#    toolchain-keyed directory under the cache root, with a size cap.
launch "${LAUNCHER}" --ocloc-cache -- "${MOCK_BIN}/mock-icpx"
grep -q '^NEO_CACHE_PERSISTENT=1$' "${LOG}/env" || fail "NEO_CACHE_PERSISTENT not set"
grep -Eq '^NEO_CACHE_MAX_SIZE=[1-9][0-9]+$' "${LOG}/env" || fail "NEO_CACHE_MAX_SIZE not set"
cache_dir="$(sed -n 's/^NEO_CACHE_DIR=//p' "${LOG}/env")"
[[ "${cache_dir}" == "${CACHE_ROOT}/"* ]] || fail "cache dir ${cache_dir} not under ${CACHE_ROOT}"
[[ -d "${cache_dir}" ]] || fail "cache dir was not created"
[[ "${cache_dir}" == "${CACHE_ROOT}/$(key)" ]] || fail "cache dir does not match --print-cache-key"
grep -q "mock icpx 1.0" "${cache_dir}/KEY" || fail "KEY file does not record the compiler version"
assert_parent_tmp_empty "ocloc-cache"

# 4. The key is deterministic, and a change to ANY toolchain component moves
#    it -- so a driver or compiler upgrade can never hit an entry built by the
#    old one. Positive control first: the same inputs give the same key.
base="$(key)"
[[ -n "${base}" ]] || fail "empty cache key"
[[ "$(key)" == "${base}" ]] || fail "cache key is not deterministic"
[[ "$(key env MOCK_COMPILER_VERSION='mock icpx 1.1')" != "${base}" ]] || fail "compiler version change kept the key"
[[ "$(key env MOCK_OCLOC_VERSION='26.32.0')" != "${base}" ]] || fail "ocloc version change kept the key"
echo igc-upgraded-with-more-bytes > "${KEY_FILE_A}"
[[ "$(key)" != "${base}" ]] || fail "IGC library change kept the key"
upgraded="$(key)"
echo ocloc-lib > "${KEY_FILE_B}"
touch -d '2001-01-01' "${KEY_FILE_B}"
[[ "$(key)" != "${upgraded}" ]] || fail "same-size library rebuild (mtime only) kept the key"
[[ "$(key env GGML_SYCL_OCLOC_KEY_PKGS='bash')" != "$(key)" ]] || fail "package list does not reach the key"

# 5. Inventory mode records every ocloc invocation (input and output md5,
#    device) and still runs the real ocloc; the recorder is itself temporary.
inventory="${TMP}/inventory.tsv"
launch env MOCK_RUN_OCLOC=1 GGML_SYCL_DEVICE_LINK_INVENTORY="${inventory}" \
    "${LAUNCHER}" --ocloc-cache -- "${MOCK_BIN}/mock-icpx" || fail "inventory run failed"
[[ -s "${inventory}" ]] || fail "inventory is empty"
spv_md5="$(printf 'spirv-bytes' | md5sum | cut -d' ' -f1)"
out_md5="$(printf 'aot-binary' | md5sum | cut -d' ' -f1)"
grep -Pq "^bmg_g21\t${spv_md5}\t${out_md5}\t" "${inventory}" || fail "inventory row wrong: $(cat "${inventory}")"
assert_parent_tmp_empty "inventory"

# 6. A signal to the launcher (ninja interrupted, a harness kill) stops the
#    whole link -- icpx and the llvm-foreach/ocloc processes under it, up to
#    8 x 1.4 GB of them -- exits with the signal's conventional status, and
#    still removes the private temp directory, the leak this launcher exists
#    to close.
signal_case() {
    local sig="$1" want_rc="$2" pid_file="${TMP}/child-$1.pid" launcher_pid rc=0 grandchild_pid child_pid pid
    # Started directly rather than through launch(): a backgrounded shell
    # function runs in a subshell, and $! would name that subshell, not the
    # launcher. set -m starts it with default signal dispositions, as ninja
    # does; a plain background job would inherit SIGINT ignored, and bash
    # cannot trap a signal that was ignored on entry.
    set -m
    env "${LAUNCH_ENV[@]}" MOCK_SLEEP=30 MOCK_PID_FILE="${pid_file}" "${LAUNCHER}" -- "${MOCK_BIN}/mock-icpx" &
    launcher_pid=$!
    set +m
    for _ in $(seq 100); do
        [[ -s "${pid_file}" ]] && break
        sleep 0.1
    done
    [[ -s "${pid_file}" ]] || fail "${sig}: mock link never started"
    read -r grandchild_pid child_pid < "${pid_file}"
    local start_ms elapsed_ms
    start_ms=$(( $(date +%s%N) / 1000000 ))
    kill "-${sig}" "${launcher_pid}"
    wait "${launcher_pid}" || rc=$?
    elapsed_ms=$(( $(date +%s%N) / 1000000 - start_ms ))
    [[ ${rc} -eq ${want_rc} ]] || fail "${sig}: launcher exited ${rc}, want ${want_rc}"
    # The mock link dies on the first TERM; waiting out its 30 s sleep, or the
    # launcher's 5 s KILL fallback, means the signal did not reach the group.
    (( elapsed_ms < 3000 )) || fail "${sig}: launcher took ${elapsed_ms} ms to stop the link"
    for pid in "${child_pid}" "${grandchild_pid}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill -KILL "${child_pid}" "${grandchild_pid}" 2>/dev/null || true
            fail "${sig}: process ${pid} of the link survived the launcher (child ${child_pid}, grandchild ${grandchild_pid})"
        fi
    done
    assert_parent_tmp_empty "SIG${sig}"
}
signal_case TERM 143
signal_case INT 130
signal_case HUP 129

# 7. Each toolchain key gets its own directory, so an upgrade strands the old
#    one. A link keeps its own key and the most recently used other one (a
#    rollback target) and removes the rest -- only launcher-made directories,
#    recognised by name and KEY file.
PRUNE_ROOT="${TMP}/prune-cache"
mkdir -p "${PRUNE_ROOT}/ocloc-26.20.1_00000000000000aa" "${PRUNE_ROOT}/ocloc-26.22.1_00000000000000bb" \
    "${PRUNE_ROOT}/ocloc-26.18.1_00000000000000cc" "${PRUNE_ROOT}/ocloc-26.10.1_00000000000000dd" \
    "${PRUNE_ROOT}/notes"
for d in ocloc-26.20.1_00000000000000aa ocloc-26.22.1_00000000000000bb ocloc-26.18.1_00000000000000cc; do
    echo key > "${PRUNE_ROOT}/${d}/KEY"
done
touch -d '5 days ago' "${PRUNE_ROOT}/ocloc-26.20.1_00000000000000aa"
touch -d '1 day ago' "${PRUNE_ROOT}/ocloc-26.22.1_00000000000000bb"
touch -d '9 days ago' "${PRUNE_ROOT}/ocloc-26.18.1_00000000000000cc"
touch -d '9 days ago' "${PRUNE_ROOT}/ocloc-26.10.1_00000000000000dd"
launch env GGML_SYCL_OCLOC_CACHE_ROOT="${PRUNE_ROOT}" "${LAUNCHER}" --ocloc-cache -- "${MOCK_BIN}/mock-icpx" ||
    fail "prune run failed"
kept="$(cd "${PRUNE_ROOT}" && ls | sort | tr '\n' ' ')"
want="$(printf '%s\n' "$(key)" ocloc-26.22.1_00000000000000bb ocloc-26.10.1_00000000000000dd notes | sort | tr '\n' ' ')"
[[ "${kept}" == "${want}" ]] || fail "prune kept '${kept}', want '${want}'"

# "Most recently used" is use, not creation: a link refreshes its own key's
# directory, so after the next toolchain change that one is the one kept.
touch -d '10 days ago' "${PRUNE_ROOT}/$(key)"
launch env GGML_SYCL_OCLOC_CACHE_ROOT="${PRUNE_ROOT}" "${LAUNCHER}" --ocloc-cache -- "${MOCK_BIN}/mock-icpx" ||
    fail "prune reuse run failed"
launch env GGML_SYCL_OCLOC_CACHE_ROOT="${PRUNE_ROOT}" MOCK_COMPILER_VERSION='mock icpx 2.0' \
    "${LAUNCHER}" --ocloc-cache -- "${MOCK_BIN}/mock-icpx" || fail "prune upgrade run failed"
kept="$(cd "${PRUNE_ROOT}" && ls | sort | tr '\n' ' ')"
want="$(printf '%s\n' "$(key env MOCK_COMPILER_VERSION='mock icpx 2.0')" "$(key)" \
    ocloc-26.10.1_00000000000000dd notes | sort | tr '\n' ' ')"
[[ "${kept}" == "${want}" ]] || fail "after upgrade prune kept '${kept}', want '${want}'"

# 8. Below the free-space floor the link still gets a working NEO cache, but
#    a throwaway one inside its private TMPDIR, so nothing grows on the full
#    filesystem and no stray ./ocloc_cache appears; it says so on stderr.
LOW_ROOT="${TMP}/low-cache"
launch env GGML_SYCL_OCLOC_CACHE_ROOT="${LOW_ROOT}" GGML_SYCL_OCLOC_CACHE_MIN_FREE=1152921504606846976 \
    "${LAUNCHER}" --ocloc-cache -- "${MOCK_BIN}/mock-icpx" 2> "${LOG}/low-stderr" || fail "low-space run failed"
grep -q '^NEO_CACHE_PERSISTENT=1$' "${LOG}/env" || fail "low space: NEO cache not set up"
grep -q "^NEO_CACHE_DIR=${PARENT_TMP}/sycl-device-link\.[^/]*/" "${LOG}/env" ||
    fail "low space: cache not in the private TMPDIR: $(cat "${LOG}/env")"
[[ -z "$(ls -A "${LOW_ROOT}" 2>/dev/null)" ]] || fail "low space: cache root grew: $(ls -A "${LOW_ROOT}")"
grep -q 'free' "${LOG}/low-stderr" || fail "low space: no warning: $(cat "${LOG}/low-stderr")"
assert_parent_tmp_empty "low space"

# ...and at a floor of 0 the same root is used normally.
launch env GGML_SYCL_OCLOC_CACHE_ROOT="${LOW_ROOT}" GGML_SYCL_OCLOC_CACHE_MIN_FREE=0 \
    "${LAUNCHER}" --ocloc-cache -- "${MOCK_BIN}/mock-icpx" || fail "floor-0 run failed"
grep -q "^NEO_CACHE_DIR=${LOW_ROOT}/$(key)$" "${LOG}/env" || fail "floor 0: persistent cache not used"

# 9. IGC and NEO debug keys change or dump the ISA, and neither NEO's cache key
#    nor the launcher's covers the environment. With one set, a warm
#    persistent cache must not be read (a dump run would dump nothing) and
#    must not be written (a non-default image would be served to later
#    builds): the link gets a throwaway cache and one warning naming the key.
DBG_ROOT="${TMP}/dbg-cache"
ocloc_log="${LOG}/ocloc"
launch env GGML_SYCL_OCLOC_CACHE_ROOT="${DBG_ROOT}" MOCK_RUN_OCLOC=1 MOCK_OCLOC_LOG="${ocloc_log}" \
    "${LAUNCHER}" --ocloc-cache -- "${MOCK_BIN}/mock-icpx" || fail "warm-up run failed"
launch env GGML_SYCL_OCLOC_CACHE_ROOT="${DBG_ROOT}" MOCK_RUN_OCLOC=1 MOCK_OCLOC_LOG="${ocloc_log}" \
    "${LAUNCHER}" --ocloc-cache -- "${MOCK_BIN}/mock-icpx" || fail "warm run failed"
[[ "$(paste -sd' ' "${ocloc_log}")" == "miss hit" ]] ||
    fail "positive control: warm cache did not hit: $(paste -sd' ' "${ocloc_log}")"
# Two stale toolchain keys, so a prune (which keeps only the newest other key)
# would visibly change the root if hygiene ran on the debug path.
for d in ocloc-26.20.1_00000000000000aa ocloc-26.18.1_00000000000000bb; do
    mkdir -p "${DBG_ROOT}/${d}"
    echo key > "${DBG_ROOT}/${d}/KEY"
done
touch -d '1 day ago' "${DBG_ROOT}/ocloc-26.20.1_00000000000000aa"
touch -d '2 days ago' "${DBG_ROOT}/ocloc-26.18.1_00000000000000bb"
snapshot() { (cd "${DBG_ROOT}" && find . -type f -exec md5sum {} + | sort; find . | sort) | md5sum; }
for debug_env in IGC_ShaderDumpEnable=1 NEOReadDebugKeys=1; do
    before="$(snapshot)"
    : > "${ocloc_log}"
    launch env GGML_SYCL_OCLOC_CACHE_ROOT="${DBG_ROOT}" MOCK_RUN_OCLOC=1 MOCK_OCLOC_LOG="${ocloc_log}" \
        "${debug_env}" "${LAUNCHER}" --ocloc-cache -- "${MOCK_BIN}/mock-icpx" 2> "${LOG}/dbg-stderr" ||
        fail "${debug_env}: link failed"
    [[ "$(paste -sd' ' "${ocloc_log}")" == "miss" ]] ||
        fail "${debug_env}: persistent cache was read: $(paste -sd' ' "${ocloc_log}")"
    [[ "$(snapshot)" == "${before}" ]] || fail "${debug_env}: persistent cache changed"
    grep -q "^NEO_CACHE_DIR=${PARENT_TMP}/sycl-device-link\.[^/]*/" "${LOG}/env" ||
        fail "${debug_env}: cache not in the private TMPDIR: $(cat "${LOG}/env")"
    [[ "$(grep -c "${debug_env%%=*}" "${LOG}/dbg-stderr")" -eq 1 ]] ||
        fail "${debug_env}: want one warning naming it: $(cat "${LOG}/dbg-stderr")"
    assert_parent_tmp_empty "${debug_env}"
done
# ...nor create a root that was not there.
launch env GGML_SYCL_OCLOC_CACHE_ROOT="${TMP}/dbg-absent" IGC_ShaderDumpEnable=1 \
    "${LAUNCHER}" --ocloc-cache -- "${MOCK_BIN}/mock-icpx" 2>/dev/null || fail "debug run on an absent root failed"
[[ ! -e "${TMP}/dbg-absent" ]] || fail "a debug run created the persistent cache root"

# 10. An ocloc call without -file (a query) is not an image: inventory mode
#     records nothing for it, rather than a row with an empty input md5 and an
#     error on stderr, and the call still succeeds.
inv_nofile="${TMP}/inventory-nofile.tsv"
launch env MOCK_RUN_OCLOC_NO_FILE=1 GGML_SYCL_DEVICE_LINK_INVENTORY="${inv_nofile}" \
    "${LAUNCHER}" -- "${MOCK_BIN}/mock-icpx" 2> "${LOG}/inv-stderr" ||
    fail "inventory mode failed an ocloc call without -file"
[[ ! -s "${inv_nofile}" ]] || fail "inventory recorded a row for a call without -file: $(cat "${inv_nofile}")"
[[ ! -s "${LOG}/inv-stderr" ]] || fail "inventory mode wrote to stderr: $(cat "${LOG}/inv-stderr")"

# 11. A stale key directory that cannot be removed (EACCES here; a link on a
#     third toolchain writing into it in practice) does not fail the link.
STUCK_ROOT="${TMP}/stuck-cache"
for d in ocloc-1.0_00000000000000e1 ocloc-1.0_00000000000000e2; do
    mkdir -p "${STUCK_ROOT}/${d}/locked"
    echo key > "${STUCK_ROOT}/${d}/KEY"
    echo entry > "${STUCK_ROOT}/${d}/locked/entry"
    chmod 555 "${STUCK_ROOT}/${d}/locked"
done
touch -d '1 day ago' "${STUCK_ROOT}/ocloc-1.0_00000000000000e1"
touch -d '2 days ago' "${STUCK_ROOT}/ocloc-1.0_00000000000000e2"
rc=0
launch env GGML_SYCL_OCLOC_CACHE_ROOT="${STUCK_ROOT}" "${LAUNCHER}" --ocloc-cache -- "${MOCK_BIN}/mock-icpx" || rc=$?
chmod -R u+w "${STUCK_ROOT}"
[[ ${rc} -eq 0 ]] || fail "an unremovable stale cache directory failed the link (rc ${rc})"

# 12. KEY is informational: a cache directory that cannot take it (a concurrent
#     prune in practice, a read-only directory here) does not fail the link.
RO_ROOT="${TMP}/ro-cache"
mkdir -p "${RO_ROOT}/$(key)"
chmod 555 "${RO_ROOT}/$(key)"
rc=0
launch env GGML_SYCL_OCLOC_CACHE_ROOT="${RO_ROOT}" "${LAUNCHER}" --ocloc-cache -- "${MOCK_BIN}/mock-icpx" \
    2> "${LOG}/ro-stderr" || rc=$?
chmod -R u+w "${RO_ROOT}"
[[ ${rc} -eq 0 ]] || fail "a cache directory that cannot take KEY failed the link (rc ${rc}): $(cat "${LOG}/ro-stderr")"
[[ ! -s "${LOG}/ro-stderr" ]] || fail "a KEY write failure reached stderr: $(cat "${LOG}/ro-stderr")"

echo "test-sycl-device-link-launcher: PASS" >&2
