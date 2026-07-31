#!/usr/bin/env bash
# Wrapper + control suite for scripts/check-sycl-xmx-threshold-default.sh
# (llama.cpp-d5h0, extended under llama.cpp-x54y). Sibling of
# test-sycl-fattn-onednn-scale-gate.sh and test-sycl-device-index-policy.sh,
# written to the same rule: the checker and the wrapper are NOT duplicates. The
# wrapper drives the checker against fixtures and asserts an EXACT exit status,
# because what the checker asserts is the PRESENCE of a table row and of a
# declaration -- and a presence check passes cleanly against a file it failed to
# parse, a file renamed out from under it, or an empty one.
#
# The cases this suite exists for are S1/S2: the checker greps a COMMENT-STRIPPED
# view, and nothing but a control proves the stripping happens. A stripper that
# silently strips nothing looks exactly like one that works -- until someone
# comments out the guarded line while leaving prose behind that still contains
# the matched string. Both S1 and S2 were live false-PASSes against the shipping
# checker before llama.cpp-x54y, reproduced, not hypothesised.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKER="$ROOT_DIR/scripts/check-sycl-xmx-threshold-default.sh"
TARGET="$ROOT_DIR/ggml/src/ggml-sycl/ggml-sycl.cpp"

# Matches the checker's own guard exactly. A wider condition would skip on
# machines where it runs fine, which is a vacuous pass in the other direction.
for tool in grep awk; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "test-sycl-xmx-threshold-policy: no $tool; skipping" >&2
        exit 77
    fi
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail=0

# The checker is exercised against synthetic fixtures BEFORE it is pointed at the
# real file. A source assertion whose pattern has silently stopped matching looks
# identical to a passing one -- these cases are what make a green run mean
# something. Each asserts status exactly 1 (violation), never merely non-zero:
# a missing fixture exits 2, and accepting that would pass vacuously.
#
# Fixtures are synthesized with heredocs rather than lifted from git history: a
# control that depends on `git show <sha>` stops working in a tarball export or a
# shallow clone, and would then skip rather than fail.
expect_status() {
    local want="$1" what="$2" file="$3" rc=0
    "$CHECKER" "$file" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne "$want" ]; then
        echo "FAIL: expected $what to exit $want, got $rc" >&2
        fail=1
    fi
}

TABLE_ROW='            { "GGML_SYCL_XMX_THRESHOLD",    &g_ggml_sycl_xmx_threshold,    64 },'

# --- accepted shape ---------------------------------------------------------

{ printf '%s\n' "$TABLE_ROW"; printf 'int g_ggml_sycl_xmx_threshold = 0;\n'; } > "$TMP/good.cpp"
expect_status 0 "good fixture" "$TMP/good.cpp"

# The declaration in the real file carries a fifteen-line comment naming
# GGML_SYCL_XMX_THRESHOLD, the table, 1024 and 64. Stripping must not make that
# prose look like a violation any more than it makes it look like the code.
cat > "$TMP/good-with-prose.cpp" <<EOF
$TABLE_ROW
// The DEFAULT is NOT here. It is the GGML_SYCL_XMX_THRESHOLD row of the
// sycl_env_settings table. It used to read 1024 and lose to the table's 64:
//     int g_ggml_sycl_xmx_threshold = 1024;
int g_ggml_sycl_xmx_threshold = 0;
EOF
expect_status 0 "healthy code documented by heavy prose" "$TMP/good-with-prose.cpp"

# --- rejected shapes --------------------------------------------------------

# Rejects a nonzero initializer -- the original 1024-vs-64 bug, and the
# plausible-looking "64" restoration that would be accidentally right today and
# wrong the moment llama.cpp-eju9 changes the table.
{ printf '%s\n' "$TABLE_ROW"; printf 'int g_ggml_sycl_xmx_threshold = 64;\n'; } > "$TMP/bad-init-64.cpp"
expect_status 1 "nonzero (64) initializer fixture" "$TMP/bad-init-64.cpp"

{ printf '%s\n' "$TABLE_ROW"; printf 'int g_ggml_sycl_xmx_threshold = 1024;\n'; } > "$TMP/bad-init-1024.cpp"
expect_status 1 "nonzero (1024) initializer fixture" "$TMP/bad-init-1024.cpp"

# Rejects loss of the table row: the default would then be stated nowhere, and
# nothing would overwrite the declaration.
printf 'int g_ggml_sycl_xmx_threshold = 0;\n' > "$TMP/bad-no-table-row.cpp"
expect_status 1 "missing table row fixture" "$TMP/bad-no-table-row.cpp"

# Rejects a renamed/removed declaration rather than passing vacuously.
{ printf '%s\n' "$TABLE_ROW"; printf 'int g_ggml_sycl_xmx_limit = 0;\n'; } > "$TMP/bad-renamed-decl.cpp"
expect_status 1 "renamed declaration fixture" "$TMP/bad-renamed-decl.cpp"

# --- S1: THE CONTROL. Table row commented out, string left in prose ----------
# Without comment-stripping this fixture PASSES and a real regression ships.
# Measured against the checker at eca9214c9: exit 0.
cat > "$TMP/shadow-table-row.cpp" <<EOF
// Removed while we rework the table; it used to read
//$TABLE_ROW
int g_ggml_sycl_xmx_threshold = 0;
EOF
expect_status 1 "a commented-out table row shadowed by prose" "$TMP/shadow-table-row.cpp"

# --- S2: same shape for the initializer, via a TRAILING comment --------------
# The nastier of the two. IN THIS FIXTURE the uncommented declaration says 1024 --
# the shape of a d5h0 reintroduction -- and the checker read "= 0;" out of the
# comment beside it. Measured against the checker at eca9214c9: exit 0.
#
# The fixture is synthesized. The real ggml-sycl.cpp:367 has always read `= 0`,
# so this describes what the gate WOULD have missed, not something it did miss.
cat > "$TMP/shadow-initializer.cpp" <<EOF
$TABLE_ROW
int g_ggml_sycl_xmx_threshold = 1024;  // was = 0; raised for broader XMX usage
EOF
expect_status 1 "an initializer read out of a trailing comment" "$TMP/shadow-initializer.cpp"

# --- S3: rename that KEEPS THE OLD NAME AS A PREFIX --------------------------
# The table row's control was a plain substring match, so it certified a binding
# to g_ggml_sycl_xmx_threshold_RENAMED -- a global that does not exist -- as a
# healthy row. Same defect the scale gate's C5 case pins. The declaration here is
# left intact so this isolates check 1; with both renamed the fixture would exit
# 1 on check 2 alone and prove nothing about check 1.
cat > "$TMP/renamed-global.cpp" <<'EOF'
            { "GGML_SYCL_XMX_THRESHOLD",    &g_ggml_sycl_xmx_threshold_RENAMED,    64 },
int g_ggml_sycl_xmx_threshold = 0;
EOF
expect_status 1 "table row binding a renamed-away global" "$TMP/renamed-global.cpp"

# --- "cannot run" cases, each status 2 and distinct from a violation ---------

expect_status 2 "missing file" "$TMP/does-not-exist.cpp"

# An empty stripped view means the file is empty, or is all comment, or the
# stripper broke. Each check would then report its target missing and send the
# reader after a rename that never happened, so this is cannot-check, not a
# violation.
: > "$TMP/empty.cpp"
expect_status 2 "an empty file" "$TMP/empty.cpp"

printf '// nothing but prose\n// and more of it\n' > "$TMP/all-comment.cpp"
expect_status 2 "a file that is entirely comment" "$TMP/all-comment.cpp"

# --- D1: DETERMINISM ---------------------------------------------------------
# ⚠️ READ THIS BEFORE TRUSTING IT: this control would NOT have caught the defect
# it shipped alongside.
#
# The checker matched a variable with `printf '%s\n' "$decl" | grep -Eq PAT`
# under `set -o pipefail`, which is the SIGPIPE race documented in
# llama.cpp-x54y: grep -q exits on first match, printf takes SIGPIPE and exits
# 141, pipefail promotes it, and a SUCCESSFUL match reads as a failure. That form
# was live in this script and this loop finds nothing, because $decl is a single
# ~40-byte line: the writer finishes into the 64 KiB pipe buffer long before grep
# can close it. Measured 200/200 identical, and the lead measured 12/12 before
# that.
#
# What the loop DOES catch is the same form once the stream grows. Isolating the
# construct and varying only input size, match always on line 1 so every reported
# miss is false: ~6 KiB -> 0/200 false failures, ~68 KiB -> 141/200, ~683 KiB ->
# 200/200. It is a step function against the pipe buffer, not a low background
# rate -- so a sibling checker that strips a whole 2.5 MB file and pipes it fails
# most of the time, and this loop would catch that on the first iteration.
#
# Keep it for that, and for the day this checker's matched declaration spans more
# than one line. Do not cite a green D1 as evidence that a piped `grep -q`
# elsewhere is safe: it is evidence about THIS input size only.
#
# COST: ~8 s of this suite's ~8.6 s, because the checker strips a 2.5 MB file on
# every iteration. That is 30x the sibling scale gate's C8 and it is the honest
# price -- pointing the loop at a small fixture instead would make it cheap and
# would stop exercising the input this checker actually runs against.
det_first=""
det_bad=0
for _ in $(seq 30); do
    det_rc=0
    "$CHECKER" "$TARGET" >/dev/null 2>&1 || det_rc=$?
    if [ -z "$det_first" ]; then
        det_first="$det_rc"
    elif [ "$det_rc" != "$det_first" ]; then
        det_bad=1
    fi
done
if [ "$det_bad" -ne 0 ] || [ "$det_first" != "0" ]; then
    echo "FAIL: the checker is not deterministic over an unchanged tree" >&2
    echo "      (first status $det_first, and at least one run disagreed)." >&2
    echo "      Do NOT re-run until it goes green -- that is the habit this case" >&2
    echo "      exists to prevent. Suspect a pipeline under 'set -o pipefail':" >&2
    echo "      grep -q exits early, its writer takes SIGPIPE, and a successful" >&2
    echo "      match is reported as a failure. Use grep pattern <<< \"\$VAR\"." >&2
    fail=1
fi

# --- Only now, against the real source --------------------------------------
if ! "$CHECKER" "$TARGET"; then
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "test-sycl-xmx-threshold-policy: FAILED" >&2
    exit 1
fi

echo "test-sycl-xmx-threshold-policy: PASS (invariant holds; comment-stripping, rename and cannot-check paths controlled)" >&2
exit 0
