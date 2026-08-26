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
