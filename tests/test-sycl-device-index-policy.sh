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

# Only now, against the real header.
"$CHECKER" "$TARGET"
