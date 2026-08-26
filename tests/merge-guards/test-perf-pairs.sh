#!/usr/bin/env bash
# Run from repo root.
# Not registered with ctest/CMake -- a standalone mock-wrap driver for
# scripts/run-merge-perf-pairs.sh (Task 5, docs/plans/2026-08-25-phase-c-upstream-merge.md).
# Proves interleave order, log naming, and the arg-validation refusals using a
# mock bench-guard wrapper. No GPU, no real binaries.
set -euo pipefail

R=scripts/run-merge-perf-pairs.sh
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/mock-wrap" <<EOF
#!/usr/bin/env bash
# mock bench-guard: record --log value and the binary, do no GPU work
log=""; while [ \$# -gt 0 ]; do case "\$1" in
  --log) log="\$2"; shift 2;; --budget) shift 2;; --) shift; break;; *) shift;;
esac; done
echo "\$(basename "\$log") \$1" >> "$TMP/sequence.txt"
touch "\$log"
EOF
chmod +x "$TMP/mock-wrap"

# --- interleave order + log naming (RED proves a shuffled expectation fails; GREEN the real order) ---
bash "$R" --arm b50-mistral --a-bin /bin/pre-bench --b-bin /bin/cand-bench \
    --outdir "$TMP/logs" --pairs 3 --bench-wrap "$TMP/mock-wrap"
expected="b50-mistral-pre-1.log /bin/pre-bench
b50-mistral-1.log /bin/cand-bench
b50-mistral-pre-2.log /bin/pre-bench
b50-mistral-2.log /bin/cand-bench
b50-mistral-pre-3.log /bin/pre-bench
b50-mistral-3.log /bin/cand-bench"
diff <(echo "$expected") "$TMP/sequence.txt" || { echo "FAIL: order/naming wrong"; exit 1; }
echo "interleave+naming ok"

# --- unknown arm -> exit 2, with the exact refusal message ---
rc=0
out=$(bash "$R" --arm b99-nope --a-bin /bin/a --b-bin /bin/b --outdir "$TMP/x" --pairs 1 \
    --bench-wrap "$TMP/mock-wrap" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: unknown arm returned $rc, want 2"; exit 1; }
echo "$out" | grep -q "unknown arm" || { echo "FAIL: unknown-arm refusal missing 'unknown arm' message"; exit 1; }
echo "unknown-arm ok"

# --- missing required flags (--a-bin/--b-bin/--outdir) -> exit 2, with the named message ---
rc=0
out=$(bash "$R" --arm b50-mistral --bench-wrap "$TMP/mock-wrap" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: missing required flags returned $rc, want 2"; exit 1; }
echo "$out" | grep -q -- "--a-bin --b-bin --outdir" || { echo "FAIL: missing-flags refusal missing named message"; exit 1; }
echo "missing-required-flags ok"

echo "PASS"
