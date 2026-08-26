#!/usr/bin/env bash
# RED/GREEN driver for check-merge-source-coverage.sh. Run from repo root.
#
# Deliberately unregistered with ctest -- merge-campaign guard, run by the
# campaign tasks; see docs/plans/2026-08-25-phase-c-upstream-merge.md.
set -euo pipefail
G=scripts/check-merge-source-coverage.sh
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# Positive control (RED): a build.ninja missing one real sycl source must fail,
# and must fail BECAUSE of that exact source (rc==1, naming it), not merely
# fail for some other reason.
VICTIM=$(find ggml/src/ggml-sycl -maxdepth 1 -name '*.cpp' | sort | head -1)
grep -vF "$VICTIM" build/build.ninja > "$TMP/broken.ninja"
rc=0; out=$(bash "$G" --build-ninja "$TMP/broken.ninja") || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: guard rc=$rc for seeded gap, want 1"; exit 1; }
grep -qF "UNREACHABLE: $VICTIM" <<<"$out" || { echo "FAIL: guard did not name $VICTIM as unreachable"; exit 1; }
echo "RED ok: guard fires on seeded gap"
# Vacuous-pass refusal, case (a): a repo-root with neither scan directory
# present must hit the MISSING SCAN ROOT refusal (exit 2) -- a specific
# refusal, not merely "some nonzero code".
MISSING_ROOT="$TMP/missing-root"
mkdir -p "$MISSING_ROOT"
rc=0; out=$(bash "$G" --build-ninja build/build.ninja --repo-root "$MISSING_ROOT" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: missing-scan-root case returned $rc, want 2"; exit 1; }
grep -qF "MISSING SCAN ROOT" <<<"$out" || { echo "FAIL: missing-scan-root case did not report MISSING SCAN ROOT"; exit 1; }
echo "missing-root ok"
# Vacuous-pass refusal, case (b): both scan roots exist but are empty -- a
# DIFFERENT code path from (a), also exit 2, must hit EMPTY SCAN specifically.
EMPTY_ROOT="$TMP/empty-root"
mkdir -p "$EMPTY_ROOT/ggml/src/ggml-sycl" "$EMPTY_ROOT/src"
rc=0; out=$(bash "$G" --build-ninja build/build.ninja --repo-root "$EMPTY_ROOT" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: empty-scan case returned $rc, want 2"; exit 1; }
grep -qF "EMPTY SCAN" <<<"$out" || { echo "FAIL: empty-scan case did not report EMPTY SCAN"; exit 1; }
echo "empty-scan ok"
# GREEN: real tree passes.
bash "$G" --build-ninja build/build.ninja
echo "GREEN ok"
# --strict, positive: the real allowlist must itself stay honest (every entry
# absent from build.ninja and present on disk).
bash "$G" --build-ninja build/build.ninja --strict
echo "strict-real ok: real allowlist satisfies --strict"
# --strict, negative control: a fabricated allowlist entry naming a file that
# IS reachable in build.ninja (the real victim, unmodified) must be rejected.
# The fixture allowlist has only this one entry, so the main scan's other 61
# genuinely-unbuilt files also fail -- rc==1 alone would pass for the WRONG
# reason. Grep the exact violation line (on stderr) and the victim path so
# this is discriminating.
STRICT_ALLOW="$TMP/strict-allow.txt"
{ echo "# fixture: stale entry, file is actually built"; echo "$VICTIM"; } > "$STRICT_ALLOW"
rc=0; out=$(bash "$G" --build-ninja build/build.ninja --allowlist "$STRICT_ALLOW" --strict 2>&1) || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: --strict returned $rc for a fabricated allowlist entry, want 1"; exit 1; }
grep -qF "STRICT VIOLATION: allowlisted entry is reachable in build.ninja" <<<"$out" \
    || { echo "FAIL: --strict did not report the reachable-in-ninja violation"; exit 1; }
grep -qF "$VICTIM" <<<"$out" || { echo "FAIL: --strict violation did not name $VICTIM"; exit 1; }
echo "strict ok: --strict fires on a fabricated allowlist entry"
# --strict, negative control: an allowlist entry naming a file that does not
# exist on disk at all must also be rejected.
STRICT_ALLOW_MISSING="$TMP/strict-allow-missing.txt"
echo "ggml/src/ggml-sycl/does-not-exist.cpp" > "$STRICT_ALLOW_MISSING"
rc=0; out=$(bash "$G" --build-ninja build/build.ninja --allowlist "$STRICT_ALLOW_MISSING" --strict 2>&1) || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: --strict returned $rc for a missing-on-disk entry, want 1"; exit 1; }
grep -qF "STRICT VIOLATION: allowlisted entry missing on disk" <<<"$out" \
    || { echo "FAIL: --strict did not report the missing-on-disk violation"; exit 1; }
echo "strict-missing ok"
