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
for a in "$@"; do [ "$a" = "-L" ] && { exit 0; }; done
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

# RED-3: the sweep call itself fails (label net still healthy) -- a failed or
# truncated ctest listing must NOT read as "zero backend-ops matches, so
# clean". This is the fail-open mode a naive `grep -c ... || true` on the
# sweep half has: empty/error output greps to 0, which is indistinguishable
# from a genuinely clean sweep. Must refuse (exit 2), naming the cause, not
# silently pass.
cat > "$TMP/ctest-sweep-fails" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do [ "$a" = "-L" ] && { echo "  Test #5: test-unified-cache-x"; exit 0; }; done
exit 1
EOF
chmod +x "$TMP/ctest-sweep-fails"
rc=0; out=$(bash "$G" --ctest-cmd "$TMP/ctest-sweep-fails" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: rc=$rc for failed sweep listing, want 2"; exit 1; }
grep -qF "SWEEP LISTING FAILED" <<<"$out" || { echo "FAIL: RED-3 did not name the cause: $out"; exit 1; }
echo "RED-3 ok (failed sweep listing refused, not vacuously passed)"

# RED-4: a nonexistent build dir must be refused with its own named cause
# (MISSING BUILD DIR), not misreported as LABEL NET EMPTY just because ctest
# against a missing --test-dir happens to select nothing. Hermetic: reuses
# the RED-1 mock purely so `command -v "$CTEST"` has something to resolve --
# the mock is never actually invoked, since the build-dir check must fire
# first, so this arm does not depend on a real ctest being on PATH.
MISSING_BUILD="$TMP/does-not-exist"
rc=0; out=$(bash "$G" --build-dir "$MISSING_BUILD" --ctest-cmd "$TMP/ctest-nolabel" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: rc=$rc for missing build dir, want 2"; exit 1; }
grep -qF "MISSING BUILD DIR" <<<"$out" || { echo "FAIL: RED-4 did not name the cause: $out"; exit 1; }
echo "RED-4 ok (missing build dir refused vacuous pass)"

# RED-5: the LABEL half fails the same way RED-3 covers for the sweep half --
# a truncated-but-nonempty label listing (one real Test line, then an error
# and a nonzero exit) must NOT read as "1 labelled test, safety net intact".
# Symmetric fix to RED-3: ctest's own exit status must be checked BEFORE the
# label-net grep is trusted.
cat > "$TMP/ctest-label-fails" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do
    [ "$a" = "-L" ] && { echo "  Test #1: test-something"; echo "ctest: internal error" >&2; exit 1; }
done
echo "  Test #9: test-something-else"
EOF
chmod +x "$TMP/ctest-label-fails"
rc=0; out=$(bash "$G" --ctest-cmd "$TMP/ctest-label-fails" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: rc=$rc for failed label listing, want 2"; exit 1; }
grep -qF "LABEL LISTING FAILED" <<<"$out" || { echo "FAIL: RED-5 did not name the cause: $out"; exit 1; }
echo "RED-5 ok (failed label listing refused, not vacuously passed)"

# RED-6: the sweep half's non-vacuity check -- an -LE that exits 0 but
# selects NOTHING (empty stdout) must not certify an empty set as "no leak,
# safe". Must refuse (exit 2), naming the cause, distinctly from RED-3's
# ctest-failure case (this one is a clean exit 0 with vacuous output).
cat > "$TMP/ctest-sweep-empty" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do [ "$a" = "-L" ] && { echo "  Test #5: test-unified-cache-x"; exit 0; }; done
exit 0
EOF
chmod +x "$TMP/ctest-sweep-empty"
rc=0; out=$(bash "$G" --ctest-cmd "$TMP/ctest-sweep-empty" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: rc=$rc for empty sweep listing, want 2"; exit 1; }
grep -qF "SWEEP LISTING EMPTY" <<<"$out" || { echo "FAIL: RED-6 did not name the cause: $out"; exit 1; }
echo "RED-6 ok (empty sweep listing refused, not vacuously passed)"

# The model-loader half reads `ctest -N --show-only=json-v1` three times:
# unfiltered (the registry the loader set is derived from), through the
# documented sweep filters, and through the -LE alone (what the label exclusion
# keeps). mock_json writes a mock that answers the text listings like the mocks
# above and the JSON listings from files: a call carrying -LE gets the sweep
# file, so the mock's label exclusion keeps exactly its sweep, and a call
# without it gets the registry.
mock_json() {  # name registry.json sweep.json
    cat > "$TMP/$1" <<EOF
#!/usr/bin/env bash
json=0 sweep=0
for a in "\$@"; do
    [ "\$a" = "-L" ] && { echo "  Test #5: test-unified-cache-x"; exit 0; }
    [ "\$a" = "--show-only=json-v1" ] && json=1
    [ "\$a" = "-LE" ] && sweep=1
done
if [ \$json -eq 1 ]; then
    if [ \$sweep -eq 1 ]; then cat "$TMP/$3"; else cat "$TMP/$2"; fi
    exit 0
fi
echo "  Test #9: test-something-else"
EOF
    chmod +x "$TMP/$1"
}
# One JSON test entry: name, command, then FIXTURES_REQUIRED and FIXTURES_SETUP
# as JSON arrays.
entry() {
    printf '{"name":"%s","command":["%s"],"properties":[{"name":"FIXTURES_REQUIRED","value":%s},{"name":"FIXTURES_SETUP","value":%s}]}' \
        "$1" "$2" "$3" "$4"
}
# Same, with no fixtures and one LABELS list: name, command, labels array.
labelled() {
    printf '{"name":"%s","command":["%s"],"properties":[{"name":"LABELS","value":%s}]}' "$1" "$2" "$3"
}
tests_json() { local IFS=,; printf '{"kind":"ctestInfo","tests":[%s]}\n' "$*"; }

PLAIN=$(entry test-something-else /b/test-something-else '[]' '[]')
LOADER=$(entry test-model-user /b/test-model-user '["gen-weights"]' '[]')
GENERATOR=$(entry test-make-weights /b/test-llama-archs '[]' '["gen-weights"]')
DOWNLOAD=$(entry test-fetch-weights /usr/bin/cmake '[]' '["fetch-weights"]')
ENVMODEL=$(labelled test-env-model /b/test-env-model '["model"]')
tests_json "$PLAIN" "$LOADER" "$GENERATOR" "$DOWNLOAD" "$ENVMODEL" > "$TMP/registry.json"

# The documented exclusions must appear in CLAUDE.md; the mocks do not exercise
# that, so they get a doc that carries them for whatever the guard's defaults are.
sweep_le=$(eval "$(grep -E '^(MODEL_LABEL_RE|SWEEP_LE_DEFAULT)=' "$G")"; printf '%s' "$SWEEP_LE_DEFAULT")
sweep_e=$(sed -n "s/^SWEEP_E_DEFAULT='\(.*\)'$/\1/p" "$G")
[ -n "$sweep_le" ] || { echo "FAIL: could not read SWEEP_LE_DEFAULT from $G"; exit 1; }
[ -n "$sweep_e" ] || { echo "FAIL: could not read SWEEP_E_DEFAULT from $G"; exit 1; }
printf "form 2: -LE '%s' -E '%s'\nPR step 3: -LE '%s' -E '%s'\n" "$sweep_le" "$sweep_e" "$sweep_le" "$sweep_e" \
    > "$TMP/doc-ok.md"

# GREEN-mock: hermetic positive control for the --ctest-cmd seam. The sweep
# keeps the plain test and the cmake download setup, which is not a loader,
# and drops every loader, so it must pass with rc==0.
tests_json "$PLAIN" "$DOWNLOAD" > "$TMP/sweep-clean.json"
mock_json ctest-clean registry.json sweep-clean.json
rc=0; out=$(bash "$G" --ctest-cmd "$TMP/ctest-clean" --claude-md "$TMP/doc-ok.md" 2>&1) || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: rc=$rc for mock clean listing, want 0: $out"; exit 1; }
grep -qF "safety net intact" <<<"$out" || { echo "FAIL: GREEN-mock did not report safety net intact: $out"; exit 1; }
echo "GREEN-mock ok"

# RED-7: the sweep keeps a test that requires a fixture. The fixture's name
# says nothing about models ("gen-weights"), so this also proves the loader
# set comes from FIXTURES_REQUIRED and not from a name match.
tests_json "$PLAIN" "$LOADER" > "$TMP/sweep-loader.json"
mock_json ctest-loader registry.json sweep-loader.json
rc=0; out=$(bash "$G" --ctest-cmd "$TMP/ctest-loader" --claude-md "$TMP/doc-ok.md" 2>&1) || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: rc=$rc for a swept fixture user, want 1: $out"; exit 1; }
grep -qF "SWEEP RUNS MODEL LOADER: test-model-user" <<<"$out" || { echo "FAIL: RED-7 did not name the test: $out"; exit 1; }
echo "RED-7 ok (swept fixture user named)"

# RED-8: the sweep keeps a fixture SETUP test that runs a binary (a model
# generator), which is a loader even though it requires nothing.
tests_json "$PLAIN" "$GENERATOR" > "$TMP/sweep-generator.json"
mock_json ctest-generator registry.json sweep-generator.json
rc=0; out=$(bash "$G" --ctest-cmd "$TMP/ctest-generator" --claude-md "$TMP/doc-ok.md" 2>&1) || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: rc=$rc for a swept generator, want 1: $out"; exit 1; }
grep -qF "SWEEP RUNS MODEL LOADER: test-make-weights" <<<"$out" || { echo "FAIL: RED-8 did not name the test: $out"; exit 1; }
echo "RED-8 ok (swept fixture generator named)"

# RED-9: a registry with no fixture users derives an empty loader set, and the
# guard must refuse rather than certify a sweep against an empty set.
tests_json "$PLAIN" > "$TMP/registry-empty.json"
mock_json ctest-noloaders registry-empty.json sweep-clean.json
rc=0; out=$(bash "$G" --ctest-cmd "$TMP/ctest-noloaders" --claude-md "$TMP/doc-ok.md" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: rc=$rc for an empty loader set, want 2: $out"; exit 1; }
grep -qF "MODEL-LOADER SET EMPTY" <<<"$out" || { echo "FAIL: RED-9 did not name the cause: $out"; exit 1; }
echo "RED-9 ok (empty loader set refused)"

# RED-10: the JSON registry listing is not JSON (a failed or foreign ctest).
echo "not json" > "$TMP/garbage.json"
mock_json ctest-garbage garbage.json sweep-clean.json
rc=0; out=$(bash "$G" --ctest-cmd "$TMP/ctest-garbage" --claude-md "$TMP/doc-ok.md" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: rc=$rc for an unreadable registry, want 2: $out"; exit 1; }
grep -qF "REGISTRY LISTING UNREADABLE" <<<"$out" || { echo "FAIL: RED-10 did not name the cause: $out"; exit 1; }
echo "RED-10 ok (unreadable registry refused)"

# RED-11: CLAUDE.md no longer carries the exclusion the guard checks, so the
# documented sweep and the checked sweep have drifted apart.
echo "form 2: -E '^test-backend-ops\$'" > "$TMP/doc-stale.md"
rc=0; out=$(bash "$G" --ctest-cmd "$TMP/ctest-clean" --claude-md "$TMP/doc-stale.md" 2>&1) || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: rc=$rc for a stale CLAUDE.md, want 1: $out"; exit 1; }
grep -qF "DOC DRIFT" <<<"$out" || { echo "FAIL: RED-11 did not name the cause: $out"; exit 1; }
echo "RED-11 ok (CLAUDE.md drift caught)"

# RED-12: the sweep keeps a test labelled `model` (it loads a model named by
# the environment). It has no fixture, so only the label makes it a loader.
tests_json "$PLAIN" "$ENVMODEL" > "$TMP/sweep-envmodel.json"
mock_json ctest-envmodel registry.json sweep-envmodel.json
rc=0; out=$(bash "$G" --ctest-cmd "$TMP/ctest-envmodel" --claude-md "$TMP/doc-ok.md" 2>&1) || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: rc=$rc for a swept model-labelled test, want 1: $out"; exit 1; }
grep -qF "SWEEP RUNS MODEL LOADER: test-env-model (carries label model)" <<<"$out" \
    || { echo "FAIL: RED-12 did not name the test: $out"; exit 1; }
echo "RED-12 ok (swept model-labelled test named)"

# RED-13: the SWEEP JSON listing is not JSON. RED-10 covers the registry
# listing only.
mock_json ctest-sweep-garbage registry.json garbage.json
rc=0; out=$(bash "$G" --ctest-cmd "$TMP/ctest-sweep-garbage" --claude-md "$TMP/doc-ok.md" 2>&1) || rc=$?
[ "$rc" -eq 2 ] || { echo "FAIL: rc=$rc for an unreadable sweep listing, want 2: $out"; exit 1; }
grep -qF "SWEEP LISTING UNREADABLE" <<<"$out" || { echo "FAIL: RED-13 did not name the cause: $out"; exit 1; }
echo "RED-13 ok (unreadable sweep listing refused)"

# RED-14: the label exclusion drops a test that is neither a model loader nor
# in the label net, so the sweep silently loses it.
tests_json "$DOWNLOAD" > "$TMP/sweep-overdrop.json"
mock_json ctest-overdrop registry.json sweep-overdrop.json
rc=0; out=$(bash "$G" --ctest-cmd "$TMP/ctest-overdrop" --claude-md "$TMP/doc-ok.md" 2>&1) || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: rc=$rc for an over-dropping label exclusion, want 1: $out"; exit 1; }
grep -qF "SWEEP DROPS NON-LOADER: test-something-else" <<<"$out" || { echo "FAIL: RED-14 did not name the test: $out"; exit 1; }
echo "RED-14 ok (label exclusion dropping a non-loader named)"

# RED-real-unanchored: against the real build/, an unanchored `model` in the
# label exclusion (ctest -LE is a regex search per label) also drops the
# `cross-model` tests and test-layout-cache (`model-load`), none of which loads
# a model. The guard must name them.
rc=0; out=$(bash "$G" --build-dir build --sweep-label-exclude 'residency|mem-handle|cache|model' 2>&1) || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: rc=$rc with label model unanchored, want 1: $out"; exit 1; }
for t in test-layout-cache cross-model-weight-usage; do
    grep -qF "SWEEP DROPS NON-LOADER: $t " <<<"$out" \
        || { echo "FAIL: RED-real-unanchored did not name $t: $out"; exit 1; }
done
echo "RED-real-unanchored ok (unanchored model drops test-layout-cache and the cross-model tests)"

# RED-real-label: against the real build/, drop `model` from the label
# exclusion. The guard must name the environment-model tests and the G1
# lifecycle test, none of which has a fixture.
rc=0; out=$(bash "$G" --build-dir build --sweep-label-exclude 'residency|mem-handle|cache' 2>&1) || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: rc=$rc with label model dropped, want 1: $out"; exit 1; }
for t in test-model-load-cancel test-autorelease test-backend-sampler sycl-lifecycle-gpu-sequential; do
    grep -qF "SWEEP RUNS MODEL LOADER: $t (carries label model)" <<<"$out" \
        || { echo "FAIL: RED-real-label did not name $t: $out"; exit 1; }
done
echo "RED-real-label ok (label model dropped: env-model and G1 tests named against build/)"

# RED-real: against the real build/, drop test-save-load-state from the
# exclusion. The guard must fail and name exactly that test. Read-only -N
# listings only -- no test execution, no GPU.
dropped=${sweep_e/|test-save-load-state/}
[ "$dropped" != "$sweep_e" ] || { echo "FAIL: test-save-load-state is not in the default exclusion"; exit 1; }
rc=0; out=$(bash "$G" --build-dir build --sweep-exclude "$dropped" 2>&1) || rc=$?
[ "$rc" -eq 1 ] || { echo "FAIL: rc=$rc with test-save-load-state dropped, want 1: $out"; exit 1; }
grep -qF "SWEEP RUNS MODEL LOADER: test-save-load-state" <<<"$out" || { echo "FAIL: RED-real did not name the test: $out"; exit 1; }
[ "$(grep -c 'SWEEP RUNS MODEL LOADER' <<<"$out")" -eq 1 ] || { echo "FAIL: RED-real named more than the dropped test: $out"; exit 1; }
echo "RED-real ok (dropped exclusion named against build/)"

# GREEN: the real build/ tree and the real CLAUDE.md pass (read-only -N
# listings only -- no test execution, no GPU).
bash "$G" --build-dir build
echo "GREEN ok"
