#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKER="$ROOT_DIR/scripts/check-sycl-device-index-guard.sh"
TARGET="$ROOT_DIR/ggml/src/ggml-sycl/common.hpp"

if ! command -v awk >/dev/null 2>&1; then
    echo "test-sycl-device-index-policy: no awk; skipping" >&2
    exit 77
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Assert an EXACT status, never merely non-zero. Exit 2 means the check could not
# run (vanished file, renamed struct, patterns that stopped matching); accepting
# "non-zero" would let every one of those masquerade as a detected violation and
# the suite would pass while testing nothing.
expect_status() {
    local want="$1" what="$2" file="$3" rc=0
    "$CHECKER" "$file" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne "$want" ]; then
        echo "expected $what to exit $want, got $rc" >&2
        exit 1
    fi
}

# The checker is exercised against synthetic fixtures BEFORE the real header. A
# source assertion whose regex has silently stopped matching is indistinguishable
# from one that passes -- that is the failure mode this whole file exists to
# prevent, and it already bit once during development: an early version judged
# each member as it closed, so members declared before the arrays they index
# matched nothing and passed. It reported a green "0 unguarded" while actually
# checking 6 of 15 members.
emit_struct() {
    # $1 = destination, $2 = struct body
    { printf 'struct ggml_tensor_extra_gpu {\n'; printf '%s\n' "$2"; printf '};\n'; } > "$1"
}

# --- accepted shapes -------------------------------------------------------

emit_struct "$TMP/good-helper.cpp" '
    void *                data_device[GGML_SYCL_MAX_DEVICES];
    void * data_device_ptr(int dev) const {
        if (!ggml_sycl_valid_device_index(dev)) {
            return nullptr;
        }
        return data_device[dev];
    }'
expect_status 0 "helper-guarded accessor" "$TMP/good-helper.cpp"

# The two members pending llama.cpp-uc7s still use the inline literal. If this
# stopped being accepted the check would fail on unmodified upstream code.
emit_struct "$TMP/good-inline-literal.cpp" '
    bool                  moe_device_table_valid[GGML_SYCL_MAX_DEVICES];
    bool forget_moe_storage_handle_on_device(int expert_id, int owner_device) {
        if (owner_device >= 0 && owner_device < GGML_SYCL_MAX_DEVICES) {
            moe_device_table_valid[owner_device] = false;
        }
        return true;
    }'
expect_status 0 "inline-literal-guarded member" "$TMP/good-inline-literal.cpp"

# A loop bounded by the constant needs no per-entry guard (clear_data_authority).
emit_struct "$TMP/good-loop.cpp" '
    void *                data_device[GGML_SYCL_MAX_DEVICES];
    void clear_data_authority() {
        for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) {
            data_device[d] = nullptr;
        }
    }'
expect_status 0 "loop-bounded member" "$TMP/good-loop.cpp"

# --- rejected shapes -------------------------------------------------------

emit_struct "$TMP/bad-read.cpp" '
    void *                data_device[GGML_SYCL_MAX_DEVICES];
    void * data_device_ptr(int dev) const {
        return data_device[dev];
    }'
expect_status 1 "unguarded read accessor" "$TMP/bad-read.cpp"

emit_struct "$TMP/bad-write.cpp" '
    void *                data_device[GGML_SYCL_MAX_DEVICES];
    void set_data_device(int dev, void * ptr) {
        data_device[dev] = ptr;
    }'
expect_status 1 "unguarded write accessor" "$TMP/bad-write.cpp"

# The index identifier is free-form. This is the case the original sweep missed:
# a regex keyed on [dev]/[device] cannot match [owner_device], so it returned a
# confident wrong answer. Keying on the ARRAY name is what makes this detectable.
emit_struct "$TMP/bad-odd-index-name.cpp" '
    bool                  moe_device_table_valid[GGML_SYCL_MAX_DEVICES];
    bool take_moe_storage_handle_on_device(int owner_device) {
        return moe_device_table_valid[owner_device];
    }'
expect_status 1 "unguarded member indexing by owner_device" "$TMP/bad-odd-index-name.cpp"

# The regression this check exists for: a NEW per-device array and accessor added
# later. Nothing here is named in the script, so this only fails if array
# discovery genuinely works.
emit_struct "$TMP/bad-future-array.cpp" '
    void *                data_device[GGML_SYCL_MAX_DEVICES];
    void * data_device_ptr(int dev) const {
        if (!ggml_sycl_valid_device_index(dev)) {
            return nullptr;
        }
        return data_device[dev];
    }
    ggml_sycl::mem_handle some_future_handle[GGML_SYCL_MAX_DEVICES];
    void * some_future_ptr(int slot) const {
        return some_future_handle[slot].resolve(slot).ptr;
    }'
expect_status 1 "newly added array with unguarded accessor" "$TMP/bad-future-array.cpp"

# A member declared BEFORE the array it indexes. Ordering must not hide it --
# this exact case passed green in an early version of the checker.
emit_struct "$TMP/bad-declared-before-array.cpp" '
    void * late_ptr(int dev) const {
        return declared_later[dev];
    }
    void *                declared_later[GGML_SYCL_MAX_DEVICES];'
expect_status 1 "member indexing an array declared after it" "$TMP/bad-declared-before-array.cpp"

# --- "cannot run" cases, each status 2 and distinct from a violation --------

expect_status 2 "missing file" "$TMP/does-not-exist.cpp"

printf 'struct something_else {\n    int x;\n};\n' > "$TMP/no-struct.cpp"
expect_status 2 "renamed/absent struct" "$TMP/no-struct.cpp"

printf 'struct ggml_tensor_extra_gpu {\n    int refcount;\n    int f() { return refcount; }\n};\n' > "$TMP/no-arrays.cpp"
expect_status 2 "struct with no per-device arrays" "$TMP/no-arrays.cpp"

emit_struct "$TMP/no-members.cpp" '
    void *                data_device[GGML_SYCL_MAX_DEVICES];'
expect_status 2 "struct with no member functions" "$TMP/no-members.cpp"

emit_struct "$TMP/no-indexing.cpp" '
    void *                data_device[GGML_SYCL_MAX_DEVICES];
    int unrelated() const {
        return 0;
    }'
expect_status 2 "no member subscripts a per-device array" "$TMP/no-indexing.cpp"

# --- report classification, driven directly (llama.cpp-n68j) ----------------
# The checker printed findings AS IT DISCOVERED THEM, interleaved with any FATAL
# line, while its dispatch matched FATAL only at the START of the report. A
# finding landing ahead of the FATAL therefore downgraded "the check could not
# run" (exit 2) to "your code is wrong" (exit 1) -- sending a reader after real
# code on the strength of a checker that had already established it could not
# answer. Found by the SIBLING script's helper-renamed control, which exists to
# prove a checker fails loudly when its own premise is destroyed, and which found
# a bug in the checker rather than in the code under test.
#
# WHY THIS IS DRIVEN THROUGH --classify-report RATHER THAN A FIXTURE: the defect
# is unreachable through the front door. Every FATAL condition in the awk implies
# zero findings (no struct, no arrays, no members, or nothing indexing -- each
# leaves the finding loop with nothing to emit), so no input file can produce the
# interleaving. That is exactly why it was filed as LATENT and why it needs a
# seam: a defect that is one edit away from reachable, with no way to test it,
# comes back silently. Judged against the dispatch at eca9214c9 these four cases
# read 2 / 1 / 0 / **1** -- the last is the bug.
classify() {
    local want="$1" what="$2" report="$3" rc=0
    printf '%s\n' "$report" | "$CHECKER" --classify-report >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne "$want" ]; then
        echo "expected $what to exit $want, got $rc" >&2
        exit 1
    fi
}

FINDING='common.hpp:12: f() subscripts data_device[] with no bounds check -- guard with ggml_sycl_valid_device_index(dev)'
FATAL='FATAL: struct ggml_tensor_extra_gpu not found -- renamed or moved; this check no longer describes the code'

classify 2 "a leading FATAL"                    "$FATAL"
classify 1 "findings with no FATAL"             "$FINDING
OK 31 arrays, 21 members, 15 indexing, 1 unguarded"
classify 0 "a clean inventory"                  "OK 31 arrays, 21 members, 15 indexing, 0 unguarded"

# THE CASE. A FATAL that does not lead must still win: cannot-check outranks a
# finding, whatever order they were emitted in.
classify 2 "a FATAL preceded by a finding"      "$FINDING
$FATAL"

# A report with none of the three markers is a broken premise, not a pass. This
# used to exit 1 silently -- `printf ... | grep '^OK '` failing under `set -e`.
classify 2 "a report with no marker at all"     ""

# A file literally named --classify-report must still be readable as a file.
# The flag shares the positional slot with the target path, so `--` ends option
# parsing. Ported from the symmetry sibling under qual-x54y-r2: 564c12974 added
# identical `--` handling to BOTH scripts and pinned it in only one, so the fix
# for an unfalsifiable change was itself partly unfalsifiable in the mirror-image
# file. Sibling files pull attention to the one being reasoned about; the
# identical edit to the other rides along unexamined BECAUSE it is identical.
emit_struct "$TMP/--classify-report" '
    void *                data_device[GGML_SYCL_MAX_DEVICES];
    void * data_device_ptr(int dev) const {
        if (!ggml_sycl_valid_device_index(dev)) {
            return nullptr;
        }
        return data_device[dev];
    }'
rc=0
( cd "$TMP" && "$CHECKER" -- ./--classify-report ) >/dev/null 2>&1 || rc=$?
if [ "$rc" -ne 0 ]; then
    echo "expected '-- ./--classify-report' to be read as a FILE (exit 0), got $rc" >&2
    echo "The end-of-options marker is gone or broken: the path was consumed as" >&2
    echo "the --classify-report flag, so the checker read a report from stdin" >&2
    echo "instead of checking the file it was handed." >&2
    exit 1
fi

# --- determinism -------------------------------------------------------------
# This checker has no piped early-exiting grep and never had the SIGPIPE flake of
# llama.cpp-x54y, so this loop is a guard against reintroducing one rather than a
# reproduction of anything. It is cheap here (~10 ms/run) because the checker
# parses one header, unlike the xmx sibling.
det_first=""
det_bad=0
for _ in $(seq 30); do
    det_rc=0
    "$CHECKER" "$TARGET" >/dev/null 2>&1 || det_rc=$?
    if [ -z "$det_first" ]; then
        det_first="$det_rc"
    elif [ "$det_rc" != "$det_first" ]; then
        det_bad=1
    fi
done
if [ "$det_bad" -ne 0 ] || [ "$det_first" != "0" ]; then
    echo "the checker is not deterministic over an unchanged tree" >&2
    echo "(first status $det_first, and at least one run disagreed)." >&2
    echo "Do NOT re-run until it goes green -- that is the habit this case exists" >&2
    echo "to prevent. Suspect a pipeline under 'set -o pipefail': grep -q/-l/-m" >&2
    echo "exits early, its writer takes SIGPIPE, and a successful match is" >&2
    echo "reported as a failure. Use grep pattern <<< \"\$VAR\"." >&2
    exit 1
fi

# Only now, against the real header.
"$CHECKER" "$TARGET"
