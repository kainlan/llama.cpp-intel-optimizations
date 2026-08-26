#!/usr/bin/env bash
# RED/GREEN driver for check-ctest-safety-net.sh. Run from repo root.
#
# Deliberately unregistered with ctest -- merge-campaign guard, run by the
# campaign tasks; see docs/plans/2026-08-25-phase-c-upstream-merge.md.
set -euo pipefail
G=scripts/check-ctest-safety-net.sh
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# RED-1: label net empty (no "  Test #" lines for -L) must fail, and must fail
# BECAUSE of that exact cause (rc==1, naming it), not merely fail for some
# other reason. The sweep call (any invocation without a bare "-L" arg) is
# clean, so this mock isolates the label-net half.
cat > "$TMP/ctest-nolabel" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do [ "$a" = "-L" ] && exec echo "Total Tests: 0"; done
echo "  Test #1: test-something"
EOF
chmod +x "$TMP/ctest-nolabel"
rc=0; out=$(bash "$G" --ctest-cmd "$TMP/ctest-nolabel" 2>&1) || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: rc=$rc for empty label net, want 1"; exit 1; }
grep -qF "LABEL NET EMPTY" <<<"$out" || { echo "FAIL: RED-1 did not name the cause: $out"; exit 1; }
echo "RED-1 ok (empty label net caught)"

# RED-2: label net ok, but the filtered sweep still leaks test-backend-ops --
# a DIFFERENT code path from RED-1, must fail BECAUSE of that exact cause
# (rc==1, naming it).
cat > "$TMP/ctest-leak" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do [ "$a" = "-L" ] && { echo "  Test #5: test-unified-cache-x"; exit 0; }; done
echo "  Test #9: test-backend-ops"
EOF
chmod +x "$TMP/ctest-leak"
rc=0; out=$(bash "$G" --ctest-cmd "$TMP/ctest-leak" 2>&1) || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: rc=$rc for backend-ops leak, want 1"; exit 1; }
grep -qF "SWEEP LEAK" <<<"$out" || { echo "FAIL: RED-2 did not name the cause: $out"; exit 1; }
echo "RED-2 ok (sweep leak caught)"

# GREEN-mock: hermetic positive control for the --ctest-cmd seam -- a mock
# that returns a labelled test for -L and a clean filtered listing (no
# backend-ops) must pass with rc==0 and report the count, independent of the
# real build/ tree (proves the seam is actually wired, not merely that
# build/ happens to pass).
cat > "$TMP/ctest-clean" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do [ "$a" = "-L" ] && { echo "  Test #5: test-unified-cache-x"; exit 0; }; done
echo "  Test #9: test-something-else"
EOF
chmod +x "$TMP/ctest-clean"
rc=0; out=$(bash "$G" --ctest-cmd "$TMP/ctest-clean" 2>&1) || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: rc=$rc for mock clean listing, want 0: $out"; exit 1; }
grep -qF "safety net intact" <<<"$out" || { echo "FAIL: GREEN-mock did not report safety net intact: $out"; exit 1; }
echo "GREEN-mock ok"

# GREEN: real pre-merge build/ tree passes (read-only -N listings only -- no
# test execution, no GPU).
bash "$G" --build-dir build
echo "GREEN ok"
