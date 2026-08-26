#!/usr/bin/env bash
# Run from repo root.
# Not registered with ctest/CMake -- a standalone mock-wrap driver for
# scripts/run-merge-perf-pairs.sh (Task 5, docs/plans/2026-08-25-phase-c-upstream-merge.md).
# Proves interleave order, log naming, the completion manifest, and the
# arg-validation refusals using mock bench-guard wrappers. No GPU, no real
# benchmark binaries -- --a-bin/--b-bin point at inert executable stubs.
set -euo pipefail

R=scripts/run-merge-perf-pairs.sh
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Inert executable stubs. The runner requires --a-bin/--b-bin to be
# executable; neither is ever actually invoked here, since the mocks below
# stand in for bench-guard and record the intended command instead of
# running it.
cat > "$TMP/pre-bench" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
cat > "$TMP/cand-bench" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$TMP/pre-bench" "$TMP/cand-bench"

# mock bench-guard: records --log's basename, ONEAPI_DEVICE_SELECTOR, and the
# full remaining command line (binary + flags) that would be benchmarked,
# into the file named by $MOCK_SEQ_FILE. Do no GPU work. Single-quoted
# heredoc so nothing is interpolated at creation time; the sequence-file path
# is threaded through an exported env var instead, resolved when the mock
# itself runs.
export MOCK_SEQ_FILE="$TMP/sequence.txt"
cat > "$TMP/mock-wrap" <<'EOF'
#!/usr/bin/env bash
log=""; while [ $# -gt 0 ]; do case "$1" in
  --log) log="$2"; shift 2;; --budget) shift 2;; --) shift; break;; *) shift;;
esac; done
echo "$(basename "$log") $ONEAPI_DEVICE_SELECTOR $*" >> "$MOCK_SEQ_FILE"
touch "$log"
EOF
chmod +x "$TMP/mock-wrap"

# --- interleave order, log naming, selector, and full flag tail: exact diff
# against the per-call sequence the runner is expected to produce for the
# b50-mistral arm (selector level_zero:1, model mistral-7b-v0.1.Q4_0.gguf).
# This check's discriminating power against selector/flag mutants was
# verified in a scratch git-archive copy outside this checkout, never here
# (llama.cpp-0v40 review record: obs A mutant proof).
bash "$R" --arm b50-mistral --a-bin "$TMP/pre-bench" --b-bin "$TMP/cand-bench" \
    --outdir "$TMP/logs" --pairs 3 --bench-wrap "$TMP/mock-wrap"
expected="b50-mistral-pre-1.log level_zero:1 $TMP/pre-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128 -fa 1 -r 5 -v
b50-mistral-1.log level_zero:1 $TMP/cand-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128 -fa 1 -r 5 -v
b50-mistral-pre-2.log level_zero:1 $TMP/pre-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128 -fa 1 -r 5 -v
b50-mistral-2.log level_zero:1 $TMP/cand-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128 -fa 1 -r 5 -v
b50-mistral-pre-3.log level_zero:1 $TMP/pre-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128 -fa 1 -r 5 -v
b50-mistral-3.log level_zero:1 $TMP/cand-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128 -fa 1 -r 5 -v"
diff <(echo "$expected") "$TMP/sequence.txt" || { echo "FAIL: order/naming/selector/flags wrong"; exit 1; }
echo "interleave+naming ok"

# --- the outdir's actual file list must match exactly: the 6 logs plus the
# completion manifest, nothing more, nothing missing ---
find "$TMP/logs" -maxdepth 1 -type f -printf '%f\n' | sort > "$TMP/actual-files.txt"
printf '%s\n' b50-mistral-1.log b50-mistral-2.log b50-mistral-3.log \
    b50-mistral-pre-1.log b50-mistral-pre-2.log b50-mistral-pre-3.log \
    b50-mistral.complete | sort > "$TMP/expected-files.txt"
diff "$TMP/expected-files.txt" "$TMP/actual-files.txt" || { echo "FAIL: outdir file list wrong"; exit 1; }
echo "outdir-file-list ok"

# --- refuse to run into a non-empty outdir: a second run into the same
# --outdir would otherwise silently merge into the existing logs and corrupt
# the matrix ---
rc=0
out=$(bash "$R" --arm b50-mistral --a-bin "$TMP/pre-bench" --b-bin "$TMP/cand-bench" \
    --outdir "$TMP/logs" --pairs 1 --bench-wrap "$TMP/mock-wrap" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: rerun into existing outdir returned $rc, want 2"; exit 1; }
grep -qF -- "already exists" <<<"$out" || { echo "FAIL: existing-logs refusal missing message"; exit 1; }
grep -qF -- "b50-mistral-1.log" <<<"$out" || { echo "FAIL: existing-logs refusal does not name the offending file"; exit 1; }
echo "refuse-existing-logs ok"

# --- unknown arm -> exit 2, with the exact refusal message ---
rc=0
out=$(bash "$R" --arm b99-nope --a-bin /bin/a --b-bin /bin/b --outdir "$TMP/x" --pairs 1 \
    --bench-wrap "$TMP/mock-wrap" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: unknown arm returned $rc, want 2"; exit 1; }
grep -qF -- "unknown arm" <<<"$out" || { echo "FAIL: unknown-arm refusal missing 'unknown arm' message"; exit 1; }
echo "unknown-arm ok"

# --- missing required flags (--a-bin/--b-bin/--outdir) -> exit 2, with the named message ---
rc=0
out=$(bash "$R" --arm b50-mistral --bench-wrap "$TMP/mock-wrap" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: missing required flags returned $rc, want 2"; exit 1; }
grep -qF -- "--a-bin --b-bin --outdir" <<<"$out" || { echo "FAIL: missing-flags refusal missing named message"; exit 1; }
echo "missing-required-flags ok"

# --- non-numeric / zero --pairs -> exit 2, with a named message ---
for badpairs in abc 0; do
    rc=0
    out=$(bash "$R" --arm b50-mistral --a-bin "$TMP/pre-bench" --b-bin "$TMP/cand-bench" \
        --outdir "$TMP/pairs-$badpairs" --pairs "$badpairs" --bench-wrap "$TMP/mock-wrap" 2>&1) || rc=$?
    [ "$rc" -eq 2 ] || { echo "FAIL: --pairs $badpairs returned $rc, want 2"; exit 1; }
    grep -qF -- "--pairs must be a positive integer" <<<"$out" || { echo "FAIL: --pairs $badpairs refusal missing message"; exit 1; }
done
echo "invalid-pairs ok"

# --- non-numeric --budget -> exit 2, with a named message ---
rc=0
out=$(bash "$R" --arm b50-mistral --a-bin "$TMP/pre-bench" --b-bin "$TMP/cand-bench" \
    --outdir "$TMP/budget-abc" --budget abc --bench-wrap "$TMP/mock-wrap" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: --budget abc returned $rc, want 2"; exit 1; }
grep -qF -- "--budget must be a positive integer" <<<"$out" || { echo "FAIL: --budget abc refusal missing message"; exit 1; }
echo "invalid-budget ok"

# --- --a-bin/--b-bin must be executable ---
touch "$TMP/not-executable"
rc=0
out=$(bash "$R" --arm b50-mistral --a-bin "$TMP/not-executable" --b-bin "$TMP/cand-bench" \
    --outdir "$TMP/nonexec-outdir" --pairs 1 --bench-wrap "$TMP/mock-wrap" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: non-executable --a-bin returned $rc, want 2"; exit 1; }
grep -qF -- "not executable" <<<"$out" || { echo "FAIL: non-executable --a-bin refusal missing message"; exit 1; }
echo "non-executable-bin ok"

# --- mid-arm bench-guard failure: fails on its 4th call (cand-2), before
# touching --log, so the runner must fail fast (exit 3, propagated straight
# through), leave exactly the three prior logs behind, and write no
# completion manifest. The mid-arm fail-fast behavior itself is unchanged;
# this only adds coverage for it. Single-quoted heredoc + exported env var,
# same reasoning as the mock above.
export MOCK_COUNT_FILE="$TMP/mock-count"
cat > "$TMP/mock-wrap-fail4" <<'EOF'
#!/usr/bin/env bash
log=""; while [ $# -gt 0 ]; do case "$1" in
  --log) log="$2"; shift 2;; --budget) shift 2;; --) shift; break;; *) shift;;
esac; done
n=0
[ -f "$MOCK_COUNT_FILE" ] && n=$(cat "$MOCK_COUNT_FILE")
n=$((n+1))
echo "$n" > "$MOCK_COUNT_FILE"
if [ "$n" -eq 4 ]; then
    echo "mock-wrap-fail4: simulated failure on call $n" >&2
    exit 3
fi
touch "$log"
EOF
chmod +x "$TMP/mock-wrap-fail4"

rc=0
bash "$R" --arm b50-mistral --a-bin "$TMP/pre-bench" --b-bin "$TMP/cand-bench" \
    --outdir "$TMP/fail-outdir" --pairs 3 --bench-wrap "$TMP/mock-wrap-fail4" || rc=$?
[ "$rc" -eq 3 ] || { echo "FAIL: mid-arm failure returned $rc, want 3"; exit 1; }
find "$TMP/fail-outdir" -maxdepth 1 -type f -printf '%f\n' | sort > "$TMP/fail-actual.txt"
printf '%s\n' b50-mistral-1.log b50-mistral-pre-1.log b50-mistral-pre-2.log | sort > "$TMP/fail-expected.txt"
diff "$TMP/fail-expected.txt" "$TMP/fail-actual.txt" || { echo "FAIL: mid-arm outdir contents wrong"; exit 1; }
[ -e "$TMP/fail-outdir/b50-mistral.complete" ] && { echo "FAIL: completion manifest present after mid-arm failure"; exit 1; }
echo "mid-arm-failure ok"

echo "PASS"
