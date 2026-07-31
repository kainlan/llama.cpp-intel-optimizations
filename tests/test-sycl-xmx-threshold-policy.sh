#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKER="$ROOT_DIR/scripts/check-sycl-xmx-threshold-default.sh"
TARGET="$ROOT_DIR/ggml/src/ggml-sycl/ggml-sycl.cpp"

if ! command -v grep >/dev/null 2>&1; then
    echo "test-sycl-xmx-threshold-policy: no grep; skipping" >&2
    exit 77
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

TABLE_ROW='            { "GGML_SYCL_XMX_THRESHOLD",    &g_ggml_sycl_xmx_threshold,    64 },'

# The checker is exercised against synthetic fixtures BEFORE it is pointed at the
# real file. A source assertion whose pattern has silently stopped matching looks
# identical to a passing one -- these cases are what make a green run mean
# something. Each asserts status exactly 1 (violation), never merely non-zero:
# a missing fixture exits 2, and accepting that would pass vacuously.
expect_status() {
    local want="$1" what="$2" file="$3" rc=0
    "$CHECKER" "$file" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne "$want" ]; then
        echo "expected $what to exit $want, got $rc" >&2
        exit 1
    fi
}

# Accepts the shape the source is supposed to have.
printf '%s\nint g_ggml_sycl_xmx_threshold = 0;\n' "$TABLE_ROW" > "$TMP/good.cpp"
expect_status 0 "good fixture" "$TMP/good.cpp"

# Rejects a nonzero initializer -- the original 1024-vs-64 bug, and the
# plausible-looking "64" restoration that would be accidentally right today and
# wrong the moment llama.cpp-eju9 changes the table.
printf '%s\nint g_ggml_sycl_xmx_threshold = 64;\n' "$TABLE_ROW" > "$TMP/bad-init-64.cpp"
expect_status 1 "nonzero (64) initializer fixture" "$TMP/bad-init-64.cpp"

printf '%s\nint g_ggml_sycl_xmx_threshold = 1024;\n' "$TABLE_ROW" > "$TMP/bad-init-1024.cpp"
expect_status 1 "nonzero (1024) initializer fixture" "$TMP/bad-init-1024.cpp"

# Rejects loss of the table row: the default would then be stated nowhere, and
# nothing would overwrite the declaration.
printf 'int g_ggml_sycl_xmx_threshold = 0;\n' > "$TMP/bad-no-table-row.cpp"
expect_status 1 "missing table row fixture" "$TMP/bad-no-table-row.cpp"

# Rejects a renamed/removed declaration rather than passing vacuously.
printf '%s\nint g_ggml_sycl_xmx_limit = 0;\n' "$TABLE_ROW" > "$TMP/bad-renamed-decl.cpp"
expect_status 1 "renamed declaration fixture" "$TMP/bad-renamed-decl.cpp"

# A vanished target is status 2, distinct from a violation.
expect_status 2 "missing file" "$TMP/does-not-exist.cpp"

# Only now, against the real source.
"$CHECKER" "$TARGET"
