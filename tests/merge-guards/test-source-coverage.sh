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
# R4 refusal: --strict with a nonexistent allowlist file must fail closed
# (rc==2) with its own distinct message, not silently skip strict checking
# and fall through to the unrelated rc==1/rc==0 result of the main scan.
rc=0; out=$(bash "$G" --build-ninja build/build.ninja --strict --allowlist "$TMP/no-such-allowlist.txt" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: --strict with missing allowlist returned $rc, want 2"; exit 1; }
grep -qF "STRICT requested but allowlist missing" <<<"$out" || { echo "FAIL: missing-allowlist case did not report the R4 message"; exit 1; }
echo "r4-missing-allowlist ok"
# Item 5: the allowlist must stay newline-terminated. Both the main scan and
# --strict read it with `while IFS= read -r`, which silently drops a final
# line lacking a trailing newline -- so a stale entry appended without one
# would never be validated by --strict at all: not a false pass, an INVISIBLE
# one. Confirmed by naming a nonexistent file as an unterminated fixture
# entry: "STRICT VIOLATION: allowlisted entry missing on disk" never fires
# and the entry is never even named in the output -- only present once the
# trailing newline is restored (rc alone does not discriminate this: an
# allowlist this small also fails the UNRELATED main scan on every
# genuinely-unbuilt file it doesn't cover, so both cases exit 1 for
# different reasons -- grep the exact message, not the rc, the same
# discipline the --strict tests above already use). Guard the byte
# directly, and prove the guard itself fires: a scratch copy with the
# trailing newline stripped must turn this arm red, and the real committed
# file must turn it green.
ALLOW_REAL="scripts/merge-source-coverage-allowlist.txt"
check_allowlist_newline_terminated() { [ -z "$(tail -c1 "$1")" ]; }
STRIPPED_ALLOW="$TMP/stripped-allow.txt"
printf '%s' "$(cat "$ALLOW_REAL")" > "$STRIPPED_ALLOW"
if check_allowlist_newline_terminated "$STRIPPED_ALLOW"; then
    echo "FAIL: newline-terminated check did not go red on a stripped-trailing-newline copy"; exit 1
fi
echo "allowlist-newline-check red ok: fires on a stripped-trailing-newline copy"
check_allowlist_newline_terminated "$ALLOW_REAL" \
    || { echo "FAIL: allowlist not newline-terminated -- --strict silently drops the final entry: $ALLOW_REAL"; exit 1; }
echo "allowlist-newline-check green ok: the real committed allowlist is newline-terminated"
