#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKER="$ROOT_DIR/scripts/check-sycl-handle-usage.sh"

# Same portability guard as test-sycl-alloc-policy.sh: the patterns rely on
# GNU/rg regex behavior. Exit 77 = ctest SKIP_RETURN_CODE.
if ! command -v rg >/dev/null 2>&1 && ! printf 'a b' | grep -Eq 'a\sb' 2>/dev/null; then
    echo "test-sycl-handle-policy: no ripgrep and no GNU grep; skipping" >&2
    exit 77
fi

"$CHECKER" "$ROOT_DIR/tests/sycl-handle-policy-fixtures/good"

if "$CHECKER" "$ROOT_DIR/tests/sycl-handle-policy-fixtures/bad-unchecked" >/dev/null 2>&1; then
    echo "expected bad-unchecked fixture to fail policy check" >&2
    exit 1
fi

"$CHECKER" "$ROOT_DIR/ggml/src/ggml-sycl"
