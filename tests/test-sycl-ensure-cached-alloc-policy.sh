#!/usr/bin/env bash
set -euo pipefail

# Mechanical gate for unified_cache::ensure_cached_alloc()'s test-only seam
# (llama.cpp-kwwv, following llama.cpp-og9dt's route-3 disposition 7af1375e8).
#
# og9dt kept ensure_cached_alloc() as an entry-creating, budget-gated,
# fail-closed seam the cache fixtures need, and recorded its zero-production-
# caller state as a grep in unified-cache.hpp's declaration comment. That was a
# convention: nothing ran it. This is the same grep, registered.
#
# The pattern and its scope are copied verbatim from that comment and are NOT
# interchangeable with a bare-name search. `grep -rn ensure_cached_alloc ggml/
# src/ ...` returns 14 lines here -- diagnostic strings, the declaration
# comment's own prose, and a planning doc -- so it reads as a violation when
# nothing is wrong. Requiring the open paren is what makes 2 the healthy count.
#
# --exclude-dir=tests drops both tests/ and ggml/src/ggml-sycl/tests/, which are
# the legitimate callers (34 of them per og9dt).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PATTERN='ensure_cached_alloc\('
SCAN_DIRS=(ggml/src src common tools examples)

scan() {
    # `|| true`: grep exits 1 on no-match, which set -e would turn into an
    # abort. Zero matches is a RESULT this gate must be able to see and judge
    # (it means the pattern stopped working), not a reason to stop.
    grep -rnE "$PATTERN" "$@" --include='*.cpp' --include='*.hpp' --exclude-dir=tests 2>/dev/null || true
}

# Portability guard, matching the test-sycl-*-policy.sh family: exit 77 =
# ctest SKIP_RETURN_CODE. --exclude-dir is not universal, and without it the
# scan would sweep in the legitimate tests/ callers and report a false
# violation. Probed rather than assumed from `grep --version`.
#
# ⚠️ The probe pattern is a FIXED literal and must never be "$PATTERN". Written
# that way first, and the consequence was found by breaking $PATTERN on purpose:
# the probe failed, the script exited 77, and ctest scored the run GREEN. A gate
# whose pattern has stopped working must go RED, and only a probe that is
# independent of the pattern lets it. This tests grep's capabilities; the
# positive control below tests the pattern.
probe_dir="$(mktemp -d)"
mkdir -p "$probe_dir/tests"
printf 'PROBE_TOKEN(\n' > "$probe_dir/tests/probe.cpp"
printf 'PROBE_TOKEN(\n' > "$probe_dir/keep.cpp"
probe_hits="$(grep -rnE 'PROBE_TOKEN\(' "$probe_dir" --include='*.cpp' --exclude-dir=tests 2>/dev/null | wc -l || true)"
rm -rf "$probe_dir"
if [ "$probe_hits" -ne 1 ]; then
    echo "test-sycl-ensure-cached-alloc-policy: grep lacks -r/-E/--include/--exclude-dir" >&2
    echo "  (probe expected 1 hit, got $probe_hits); skipping" >&2
    exit 77
fi

# ---------------------------------------------------------------------------
# POSITIVE CONTROL, and it runs BEFORE the real scan on purpose.
#
# The healthy answer to this gate is a count of 2, which is also what a
# subtly-broken pattern would report if it stopped matching call syntax while
# still matching the two declarations. og9dt verified the pattern "both ways:
# 2 today, 3 with an injected caller" by hand; this reproduces the injected
# half every run, against a fixture rather than the tree, so the gate cannot
# pass without first demonstrating it can still see a caller.
#
# A control placed after the check it guards can only explain a vacuous pass,
# not prevent one.
# ---------------------------------------------------------------------------
control_dir="$(mktemp -d)"
trap 'rm -rf "$control_dir"' EXIT
cat > "$control_dir/injected-production-caller.cpp" <<'FIXTURE'
// Shaped like the violation this gate exists to catch: a production-side call.
void injected(unified_cache * cache) {
    bool needs_fill = false;
    (void) cache->ensure_cached_alloc(key, src, 64, 64, type, 0, 0, layout, false, &needs_fill);
}
FIXTURE

control_hits="$(scan "$control_dir" | wc -l)"
if [ "$control_hits" -ne 1 ]; then
    echo "POSITIVE CONTROL FAILED: the pattern did not match an injected caller" >&2
    echo "  expected 1 hit in the fixture, got $control_hits" >&2
    echo "  the pattern no longer detects call syntax, so a count of 2 below" >&2
    echo "  would prove nothing. Fix the pattern, do not relax this gate." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# The real scan.
# ---------------------------------------------------------------------------
cd "$ROOT_DIR"
hits="$(scan "${SCAN_DIRS[@]}")"
count="$(printf '%s' "$hits" | grep -c '' || true)"

if [ "$count" -eq 2 ]; then
    echo "test-sycl-ensure-cached-alloc-policy: PASS (2 lines: declaration + definition, no production caller)"
    exit 0
fi

if [ "$count" -lt 2 ]; then
    echo "FAIL: expected 2 lines (declaration + definition), got $count" >&2
    echo "  Fewer than 2 does NOT mean the seam is clean -- it means the scan" >&2
    echo "  stopped finding the declaration or definition, so it could not have" >&2
    echo "  found a production caller either. Check that the paths still exist" >&2
    echo "  and that ensure_cached_alloc() was not renamed." >&2
else
    echo "FAIL: ensure_cached_alloc() has $((count - 2)) production caller(s)" >&2
    echo "  It is a TEST-ONLY seam. Read unified-cache.hpp's declaration comment" >&2
    echo "  before adding one: unified_alloc(), ensure_cached() and" >&2
    echo "  allocate_slot() are each unable to replace it for a REASON, so" >&2
    echo "  reaching for it in production is usually a sign the caller wants a" >&2
    echo "  property (budget-gated, fail-closed, entry-creating) that a real" >&2
    echo "  production path should be getting some other way." >&2
fi

echo "  scan: grep -rnE '$PATTERN' ${SCAN_DIRS[*]} --include='*.cpp' --include='*.hpp' --exclude-dir=tests" >&2
echo "  lines found:" >&2
printf '%s\n' "$hits" | sed 's/^/    /' >&2
exit 1
