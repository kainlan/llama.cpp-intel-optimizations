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

# Each false-negative case gets its OWN directory and its own assertion. Sharing
# one bad-* directory would let a regression hide: the directory keeps failing
# on the other violation, so the gate stays green while the case it was meant to
# lock in has silently stopped being detected.
if "$CHECKER" "$ROOT_DIR/tests/sycl-handle-policy-fixtures/bad-comment-before" >/dev/null 2>&1; then
    echo "expected comment-before-assignment fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-handle-policy-fixtures/bad-comment-after" >/dev/null 2>&1; then
    echo "expected comment-after-assignment fixture to fail policy check" >&2
    exit 1
fi

"$CHECKER" "$ROOT_DIR/ggml/src/ggml-sycl"
