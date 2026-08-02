#!/usr/bin/env bash
#
# End-to-end gate for -x/--exclude of test-llama-archs (llama.cpp-fepd).
#
# tests/test-archs-exclude.cpp gates the exclusion logic; this gates the one link that unit
# test cannot see -- that main() actually calls it and actually fails the run. Acceptance
# criterion 4 ("an unknown arch name is an error, not a silent no-op") is a property of the
# BINARY: before this change `-x __no_such_arch__` was an unrecognised argument, silently
# dropped, and the run went on to sweep everything while the operator believed an architecture
# had been skipped.
#
# ⚠️  EVERY invocation below must exit inside argument parsing. test-llama-archs loads models
# onto a GPU -- it is on this project's never-loop list, and a single run peaks near 200 GB of
# 255 GB -- so no probe here may reach test_backends(). That is why each one carries either
# --nan-trace or a rejected argument: both return before ggml_backend_dev_count() is ever
# called. Measured on the pre-change binary: Shmem 2506060 kB before, 2496840 kB after.
# Do not add a probe without checking it against that rule.
#
# Exit-code discipline, both traps having cost time on this project:
#   * `rc=$?` is captured on its own line, straight from the binary. Command substitution
#     resets $?, and a pipeline's $? is its LAST stage.
#   * grep reads from a herestring, never `printf ... | grep -q`, which drops SIGPIPE on the
#     writer and intermittently reports a real match as a failure.
# `set -o pipefail` is deliberately not used, for the same reason.

set -u

BIN="${1:-}"
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "SKIP: test-llama-archs binary not given or not executable: '${BIN}'" >&2
    exit 77
fi

n_fail=0
out=""
err=""
rc=0

check() {
    if [ "$1" = "yes" ]; then
        echo "ok   : $2" >&2
    else
        echo "FAIL : $2" >&2
        n_fail=$((n_fail + 1))
    fi
}

# Run the binary, capturing stdout, stderr and the status of the BINARY itself.
run() {
    local errfile
    errfile="$(mktemp)"
    out="$("$BIN" "$@" 2>"$errfile")"
    rc=$?
    err="$(cat "$errfile")"
    rm -f "$errfile"
}

has() { grep -qF -- "$2" <<< "$1"; }

# --- positive control -------------------------------------------------------------------------
# Before asserting that anything is absent from $err, show this harness can see something in it.
# `-a <bogus>` is rejected by a check that predates this change, so it fails on any binary; if
# this control does not fire, every "message present" check below is meaningless.
run -a __no_such_arch__
check "$([ "$rc" -eq 1 ] && echo yes || echo no)" "control: a bogus -a exits 1"
check "$(has "$err" "LLM architecture: __no_such_arch__" && echo yes || echo no)" \
      "control: this harness can read an error message off stderr"

# --- criterion 4: an unknown --exclude name is an error ----------------------------------------
# --nan-trace keeps the pre-change binary safe too: without -a it returns 1 immediately, so this
# same probe can be (and was) run against the unpatched binary to demonstrate RED. The exit code
# alone cannot tell the two apart -- both are 1 -- so the assertion is on the MESSAGE.
run --nan-trace -x __no_such_arch__
check "$([ "$rc" -eq 1 ] && echo yes || echo no)" "an unknown -x exits 1"
check "$(has "$err" "--exclude: not a usable LLM architecture: __no_such_arch__" && echo yes || echo no)" \
      "an unknown -x names the rejected argument"

run --nan-trace --exclude __no_such_arch__
check "$([ "$rc" -eq 1 ] && echo yes || echo no)" "the long spelling --exclude is recognised too"
check "$(has "$err" "not a usable LLM architecture" && echo yes || echo no)" \
      "the long spelling rejects an unknown name"

# "(unknown)" resolves through llm_arch_from_string() instead of failing to match, so it would
# be accepted-then-ignored by a check that only looked for a lookup failure.
run --nan-trace -x '(unknown)'
check "$([ "$rc" -eq 1 ] && echo yes || echo no)" "-x '(unknown)' is rejected rather than silently ignored"
check "$(has "$err" "not a usable LLM architecture: (unknown)" && echo yes || echo no)" \
      "-x '(unknown)' is rejected BY THE EXCLUDE CHECK, not by something else that also exits 1"

# --- criterion 2: -x is repeatable, and each occurrence is validated ----------------------------
# The rejected name is SECOND, so a parser that stopped after the first -x would accept this.
run --nan-trace -x llama -x __no_such_arch__
check "$([ "$rc" -eq 1 ] && echo yes || echo no)" "a second -x is parsed rather than ignored"
check "$(has "$err" "__no_such_arch__" && echo yes || echo no)" "the second -x is the one reported"

# --- a missing value is an error, not a swallowed flag -----------------------------------------
# usage() writes to stdout, and the pre-change binary prints no usage at all here, so this is
# checked on stdout rather than on the exit code the two share.
run --nan-trace -x
check "$([ "$rc" -eq 1 ] && echo yes || echo no)" "-x without a value exits 1"
check "$(has "$out" "Usage:" && echo yes || echo no)" "-x without a value prints usage"
check "$(has "$out" "-x/--exclude" && echo yes || echo no)" "usage documents -x/--exclude"

# --- --exclude is rejected where it cannot apply, rather than ignored ---------------------------
# --nan-trace traces the single -a architecture and iterates no list, so -x has nothing to
# filter. Inert-but-accepted is the same silent no-op criterion 4 rules out.
run --nan-trace -x llama
check "$([ "$rc" -eq 1 ] && echo yes || echo no)" "--exclude under --nan-trace exits 1"
check "$(has "$err" "--exclude has nothing to filter under --nan-trace" && echo yes || echo no)" \
      "--exclude under --nan-trace says why"

echo "$(basename "$0"): ${n_fail} check(s) failed" >&2
[ "$n_fail" -eq 0 ]
