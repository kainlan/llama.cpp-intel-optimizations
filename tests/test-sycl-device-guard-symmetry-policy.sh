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

# Only now, against the real header.
"$CHECKER" "$TARGET"
