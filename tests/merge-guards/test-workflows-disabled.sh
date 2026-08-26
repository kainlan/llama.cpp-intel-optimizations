#!/usr/bin/env bash
# RED/GREEN driver for check-fork-workflows-disabled.sh. Run from repo root.
#
# Deliberately unregistered with ctest -- merge-campaign guard, run by the
# campaign tasks; see docs/plans/2026-08-25-phase-c-upstream-merge.md.
set -euo pipefail
G=scripts/check-fork-workflows-disabled.sh
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/one-active.json" <<'EOF'
[{"id":1,"name":"CI","state":"disabled_manually"},{"id":2,"name":"CI (wasm)","state":"active"}]
EOF
cat > "$TMP/all-off.json" <<'EOF'
[{"id":1,"name":"CI","state":"disabled_manually"},{"id":2,"name":"CI (wasm)","state":"disabled_manually"}]
EOF
echo '[]' > "$TMP/empty.json"

# RED: one active workflow must fail, and must fail BECAUSE of that exact
# workflow (rc==1, naming it), not merely fail for some other reason.
rc=0; out=$(bash "$G" --input "$TMP/one-active.json" 2>&1) || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: rc=$rc for one active workflow, want 1: $out"; exit 1; }
grep -qF "NOT DISABLED:" <<<"$out" || { echo "FAIL: RED did not report NOT DISABLED: $out"; exit 1; }
grep -qF "CI (wasm)" <<<"$out" || { echo "FAIL: RED did not name the active workflow CI (wasm): $out"; exit 1; }
echo "RED ok"

# Vacuous-pass refusal: an empty listing must fail closed (exit 2), and must
# fail BECAUSE it is empty, not merely fail for some other reason.
rc=0; out=$(bash "$G" --input "$TMP/empty.json" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: empty listing returned $rc, want 2: $out"; exit 1; }
grep -qF "EMPTY workflow listing" <<<"$out" || { echo "FAIL: empty-listing case did not report EMPTY workflow listing: $out"; exit 1; }
echo "vacuous-refusal ok"

# GREEN: all workflows disabled_manually passes with rc==0 and reports the
# exact count, so a silent "all N" with wrong N can't slip through.
rc=0; out=$(bash "$G" --input "$TMP/all-off.json" 2>&1) || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: rc=$rc for all-disabled fixture, want 0: $out"; exit 1; }
grep -qF "all 2 workflows disabled_manually" <<<"$out" || { echo "FAIL: GREEN did not report the exact count: $out"; exit 1; }
echo "GREEN ok"

# gh-outage: a gh failure (auth outage, network) must exit 2, DISTINCT from
# the rc==1 "a workflow is not disabled" case above -- T24 scores rc, so the
# two must never collide. Mirrors the --ldd-cmd mock-override pattern from
# test-sycl-build-live.sh, applied here as --gh-cmd.
cat > "$TMP/gh-fail" <<'EOF'
#!/usr/bin/env bash
echo "mock gh: auth outage" >&2
exit 1
EOF
chmod +x "$TMP/gh-fail"
rc=0; out=$(bash "$G" --gh-cmd "$TMP/gh-fail" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: gh failure returned $rc, want 2: $out"; exit 1; }
grep -qF "gh workflow list failed" <<<"$out" || { echo "FAIL: gh-outage case did not report gh workflow list failed: $out"; exit 1; }
echo "gh-outage ok"

# malformed-JSON: a payload jq cannot parse must exit 2 with a script-prefixed
# message, not jq's own bare, unprefixed error text and rc (jq exits 5 on a
# parse error, which would otherwise escape uncaught under `set -e`).
echo 'not-json-at-all' > "$TMP/malformed.json"
rc=0; out=$(bash "$G" --input "$TMP/malformed.json" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: malformed JSON returned $rc, want 2: $out"; exit 1; }
grep -qF "jq could not parse the workflow listing" <<<"$out" || { echo "FAIL: malformed-JSON case did not report the jq-parse message: $out"; exit 1; }
echo "malformed-json ok"

# empty-payload: a zero-byte payload must exit 2 with the empty-payload
# message and must NOT leak bash's "integer expected" arithmetic-comparison
# noise on the way there (jq succeeds with empty output on empty input, so
# this is a distinct failure mode from malformed JSON above and from the
# valid-empty-array "[]" case tested as vacuous-refusal).
: > "$TMP/empty-payload.json"
rc=0; out=$(bash "$G" --input "$TMP/empty-payload.json" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: empty payload returned $rc, want 2: $out"; exit 1; }
if grep -qF "integer expected" <<<"$out"; then echo "FAIL: empty-payload case leaked bash's integer-expected noise: $out"; exit 1; fi
grep -qF "empty workflow listing payload" <<<"$out" || { echo "FAIL: empty-payload case did not report the empty-payload message: $out"; exit 1; }
echo "empty-payload ok"

# gh-success-seam: an argv-asserting mock confirms the guard invokes exactly
# `gh workflow list ... --json id,name,state` -- not merely "some gh call
# that happens to return JSON". This is a positive control for the live path:
# it must both pass on a compliant mock AND be able to fail (see the
# state-dropping mutant check performed manually alongside this file; a
# mock that silently accepted a narrower --json would make the live guard
# cry wolf on a clean fork by never seeing which workflows are active).
cat > "$TMP/gh-assert" <<'EOF'
#!/usr/bin/env bash
[ "$1 $2" = "workflow list" ] || { echo "mock gh-assert: expected 'workflow list', got '$1 $2'" >&2; exit 9; }
shift 2
fields=""
while [ $# -gt 0 ]; do case "$1" in
    --json) fields="$2"; shift 2;;
    *) shift;;
esac; done
[ "$fields" = "id,name,state" ] || { echo "mock gh-assert: expected --json id,name,state, got '$fields'" >&2; exit 9; }
echo '[{"id":1,"name":"CI","state":"disabled_manually"}]'
EOF
chmod +x "$TMP/gh-assert"
rc=0; out=$(bash "$G" --gh-cmd "$TMP/gh-assert" 2>&1) || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: gh-success-seam returned $rc, want 0: $out"; exit 1; }
grep -qF "all 1 workflows disabled_manually" <<<"$out" || { echo "FAIL: gh-success-seam did not report the exact count: $out"; exit 1; }
echo "gh-success-seam ok"
