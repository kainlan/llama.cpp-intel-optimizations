#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKER="$ROOT_DIR/scripts/check-sycl-handle-usage.sh"
FIXTURES="$ROOT_DIR/tests/sycl-handle-policy-fixtures"

# Same portability guard as test-sycl-alloc-policy.sh: the patterns rely on
# GNU/rg regex behavior. Exit 77 = ctest SKIP_RETURN_CODE.
#
# Herestring, not a pipeline into `grep -Eq` -- see the sibling for why (a false
# negative here does not fail the gate, it SKIPS it, and a skipped gate reads as
# green). Unreachable at this probe's size; converted because the form is the
# defect, not the rate (llama.cpp-x54y).
if ! command -v rg >/dev/null 2>&1 && ! grep -Eq 'a\sb' <<< 'a b' 2>/dev/null; then
    echo "test-sycl-handle-policy: no ripgrep and no GNU grep; skipping" >&2
    exit 77
fi

# Assert a bad fixture is REJECTED. Requires exit status exactly 1 (violations
# found), not merely non-zero: a missing fixture directory exits 2, and a
# "non-zero" test would accept that and pass vacuously -- reporting a green gate
# for a fixture that was never checked at all.
expect_rejected() {
    local dir="$1" what="$2" rc=0
    [ -d "$dir" ] || { echo "missing fixture directory: $dir" >&2; exit 1; }
    "$CHECKER" "$dir" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne 1 ]; then
        echo "expected $what to fail policy check with status 1, got $rc" >&2
        exit 1
    fi
}

"$CHECKER" "$FIXTURES/good"

expect_rejected "$FIXTURES/bad-unchecked" "bad-unchecked fixture"

# Each false-negative case gets its OWN directory and its own assertion. Sharing
# one bad-* directory would let a regression hide: the directory keeps failing
# on the other violation, so the gate stays green while the case it was meant to
# lock in has silently stopped being detected.
expect_rejected "$FIXTURES/bad-comment-before" "comment-before-assignment fixture"
expect_rejected "$FIXTURES/bad-comment-after" "comment-after-assignment fixture"

"$CHECKER" "$ROOT_DIR/ggml/src/ggml-sycl"
