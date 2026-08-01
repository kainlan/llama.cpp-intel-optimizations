#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKER="$ROOT_DIR/scripts/check-sycl-device-guard-symmetry.sh"
TARGET="$ROOT_DIR/ggml/src/ggml-sycl/common.hpp"

if ! command -v awk >/dev/null 2>&1; then
    echo "test-sycl-device-guard-symmetry-policy: no awk; skipping" >&2
    exit 77
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Assert an EXACT status, never merely non-zero. Exit 2 means the check could not
# run, and for an assertion of ABSENCE that distinction carries more weight than
# it does for its sibling: an absence check passes cleanly against an empty or
# restructured file, so "non-zero" and "zero" are each reachable for the wrong
# reason. Both must be pinned.
expect_status() {
    local want="$1" what="$2" file="$3" rc=0
    "$CHECKER" "$file" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne "$want" ]; then
        echo "expected $what to exit $want, got $rc" >&2
        exit 1
    fi
}

# Every fixture must satisfy the checker's two positive controls, or it exits 2
# and the case proves nothing about the shape it was written to test. That is not
# hypothetical bookkeeping: it is the same trap the checker guards against, one
# level up. This preamble supplies both -- a helper call and a device parameter.
emit() {
    # $1 = destination, $2 = body under test
    {
        printf 'inline bool preamble_guarded(int device) {\n'
        printf '    return ggml_sycl_valid_device_index(device);\n'
        printf '}\n'
        printf '%s\n' "$2"
    } > "$1"
}

# --- accepted shapes -------------------------------------------------------

emit "$TMP/good-helper.cpp" '
inline bool resident(const ggml_tensor * tensor, int device) {
    if (!tensor || !ggml_sycl_valid_device_index(device)) {
        return false;
    }
    return true;
}'
expect_status 0 "helper-guarded function" "$TMP/good-helper.cpp"

# The fix for this very defect documents the REJECTED form in prose directly
# above the guard that replaced it. A checker that matched its own rationale
# would fail the moment it succeeded, so comments are stripped rather than
# skipped. This fixture is that comment, verbatim in shape.
emit "$TMP/good-comment-mentions-form.cpp" '
// The device guard is the full range check, not the `device < 0` half that
// stood here: the asymmetry was not inert.
/* device < 0 in a block comment too */
inline bool resident(int device) {
    return ggml_sycl_valid_device_index(device);
}'
expect_status 0 "rejected form named only inside comments" "$TMP/good-comment-mentions-form.cpp"

# A device id held in a member, where -1 is a documented "any device" sentinel
# rather than a bounds guard (layout->device_id in the real header). Skipped on
# purpose; the header records the cost of that choice.
emit "$TMP/good-member-sentinel.cpp" '
inline bool matches(const ggml_tensor_layout * layout, int device) {
    return layout->device_id < 0 || layout->device_id == device;
}'
expect_status 0 "member-held -1 sentinel" "$TMP/good-member-sentinel.cpp"

# A loop bounded by the constant is not a half guard (clear_data_authority).
emit "$TMP/good-loop.cpp" '
inline void reset(ggml_tensor_extra_gpu * extra) {
    for (int dev = 0; dev < GGML_SYCL_MAX_DEVICES; ++dev) {
        extra->data_device[dev] = nullptr;
    }
}'
expect_status 0 "full-range loop bound" "$TMP/good-loop.cpp"

# --- rejected shapes -------------------------------------------------------

# The defect itself (llama.cpp-vvcq): lower bound only, upper bound delegated to
# a callee that folds an out-of-range id to device 0 instead of rejecting it.
emit "$TMP/bad-half-guard.cpp" '
inline bool resident(const ggml_tensor * tensor, int device) {
    if (!tensor || device < 0) {
        return false;
    }
    return true;
}'
expect_status 1 "lower-bound-only guard" "$TMP/bad-half-guard.cpp"

# The index identifier is free-form. uc7s re-ran a sweep keyed on the guard'"'"'s
# SPELLING three times and got the same wrong answer each time; keying on the
# DOMAIN vocabulary is what makes this one detectable.
emit "$TMP/bad-odd-index-name.cpp" '
inline bool on_owner(int owner_device) {
    if (owner_device < 0) {
        return false;
    }
    return true;
}'
expect_status 1 "half guard on owner_device" "$TMP/bad-odd-index-name.cpp"

# A COMPLETE guard, written inline instead of collapsed to the helper. Flagged on
# purpose: within common.hpp the comment above ggml_sycl_valid_device_index calls
# a re-spelled bound "a regression to collapse, not a style choice", and this is
# the only mechanism that enforces it -- check-sycl-device-index-guard.sh accepts
# the inline literal by design and explicitly cannot.
emit "$TMP/bad-inline-literal-not-collapsed.cpp" '
inline bool resident(int dev) {
    if (dev < 0 || dev >= GGML_SYCL_MAX_DEVICES) {
        return false;
    }
    return true;
}'
expect_status 1 "complete but uncollapsed inline bound" "$TMP/bad-inline-literal-not-collapsed.cpp"

# --- "cannot run" cases, each status 2 and distinct from a violation --------

expect_status 2 "missing file" "$TMP/does-not-exist.cpp"

# An absence assertion against a file with nothing in it is the vacuous pass this
# whole exit-2 path exists to prevent.
: > "$TMP/empty.cpp"
expect_status 2 "empty file" "$TMP/empty.cpp"

# Helper renamed or moved: the convention this check is written against no longer
# exists, so its silence means nothing. This case found a real bug in the checker
# during development -- findings were printed as discovered, so a violation line
# landed ahead of the FATAL and downgraded exit 2 to exit 1, pointing a reader at
# the source instead of at the broken premise. Findings are buffered now.
printf 'inline bool f(int device) {\n    if (device < 0) { return false; }\n    return true;\n}\n' > "$TMP/no-helper.cpp"
expect_status 2 "helper never called, with a violation present" "$TMP/no-helper.cpp"

printf 'inline bool f() {\n    return ggml_sycl_valid_device_index(0);\n}\n' > "$TMP/no-device-param.cpp"
expect_status 2 "no function takes a device parameter" "$TMP/no-device-param.cpp"

# --- report classification, driven directly (llama.cpp-n68j) ----------------
# ⚠️ THE no-helper CASE ABOVE IS NOT COVERAGE FOR THIS, though it looks like it.
# Its report is a LONE FATAL line: nhelper hits 0, the FATAL is emitted, and awk
# exits before any buffered finding is printed. "FATAL at start" and "FATAL
# anywhere" agree on that input, so it passed before the fix and would still pass
# with the fix reverted. The 35-line FATAL-anywhere change to the checker had
# nothing pinning it at all until these cases -- not wrong, UNFALSIFIABLE, which
# is the property that let every defect this wave fixed survive as long as it
# did. Found by qual-x54y-r1.
#
# Driven through --classify-report because the interleaving cannot be produced
# from a file: both FATAL conditions (nhelper==0, nparams==0) are evaluated
# before any finding is emitted, so no fixture reaches the shape.
classify() {
    local want="$1" what="$2" report="$3" rc=0
    printf '%s\n' "$report" | "$CHECKER" --classify-report >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne "$want" ]; then
        echo "expected $what to exit $want, got $rc" >&2
        exit 1
    fi
}

# The backticks are literal -- this string reproduces the checker's real finding
# format byte for byte, so the fixture cannot drift from what it stands in for.
# shellcheck disable=SC2016
FINDING='common.hpp:1229: `device < 0` guards the lower bound only -- use ggml_sycl_valid_device_index(device)'
FATAL='FATAL: ggml_sycl_valid_device_index() is never called in this file -- renamed, moved, or the wrong file was passed; an absence check here would pass vacuously'

classify 2 "a leading FATAL"        "$FATAL"
classify 1 "findings with no FATAL" "$FINDING
OK 26 helper calls, 98 device params, 1 half-guards"
classify 0 "a clean inventory"      "OK 26 helper calls, 98 device params, 0 half-guards"

# THE CASE THAT PINS THE FIX. Revert the checker's FATAL-anywhere dispatch and
# this one goes to 1: cannot-check silently downgraded to "your code is wrong".
classify 2 "a FATAL preceded by a finding" "$FINDING
$FATAL"

# A report with none of the three markers. This used to exit 1 SILENTLY --
# `printf ... | grep '^OK '` failing under `set -e` -- which is the same wrong
# direction arrived at without any ordering involved.
classify 2 "a report with no marker at all" ""

# A file literally named --classify-report must still be readable as a file.
# The flag shares the positional slot with the target path, so `--` ends option
# parsing. Cheap to pin now; expensive to discover once the seam is in N scripts.
printf 'inline bool f(int device) {\n    return ggml_sycl_valid_device_index(device);\n}\n' > "$TMP/--classify-report"
rc=0
( cd "$TMP" && "$CHECKER" -- ./--classify-report ) >/dev/null 2>&1 || rc=$?
if [ "$rc" -ne 0 ]; then
    echo "expected '-- ./--classify-report' to be read as a FILE (exit 0), got $rc" >&2
    exit 1
fi

# --- determinism -------------------------------------------------------------
# Matches the sibling's guard. This checker has no piped early-exiting grep and
# never had the SIGPIPE flake of llama.cpp-x54y, so this protects against
# reintroducing one rather than reproducing anything.
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
    echo "Do NOT re-run until it goes green. Suspect a pipeline under" >&2
    echo "'set -o pipefail': grep -q/-l/-m exits early, its writer takes" >&2
    echo "SIGPIPE, and a successful match is reported as a failure." >&2
    exit 1
fi

# Only now, against the real header.
"$CHECKER" "$TARGET"
