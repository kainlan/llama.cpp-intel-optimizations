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
# Vacuous-pass refusal: empty scan must be exit 2, not 0.
rc=0; bash "$G" --build-ninja build/build.ninja --repo-root "$TMP" || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: empty scan returned $rc, want 2"; exit 1; }
echo "vacuous-refusal ok"
# GREEN: real tree passes.
bash "$G" --build-ninja build/build.ninja
echo "GREEN ok"
# --strict, positive: the real allowlist must itself stay honest (every entry
# absent from build.ninja and present on disk).
bash "$G" --build-ninja build/build.ninja --strict
echo "strict-real ok: real allowlist satisfies --strict"
# --strict, negative control: a fabricated allowlist entry naming a file that
# IS reachable in build.ninja (the real victim, unmodified) must be rejected.
STRICT_ALLOW="$TMP/strict-allow.txt"
{ echo "# fixture: stale entry, file is actually built"; echo "$VICTIM"; } > "$STRICT_ALLOW"
rc=0; bash "$G" --build-ninja build/build.ninja --allowlist "$STRICT_ALLOW" --strict || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: --strict returned $rc for a fabricated allowlist entry, want 1"; exit 1; }
echo "strict ok: --strict fires on a fabricated allowlist entry"
