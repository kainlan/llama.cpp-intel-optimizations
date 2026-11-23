#!/usr/bin/env bash
# Host-only test of the watchdog ceiling variable of scripts/sycl-qwen4exp-census-run.sh. It runs the real
# script, so every invocation sets LLAMA_COMPLETION=/nonexistent: a case that is wrongly NOT refused would
# otherwise go on to settle and launch llama-completion on a GPU from a test that is not a GPU test. With the
# variable set, such a fall-through dies at the script's binary check instead (it also exits 3), which is why
# each case asserts its own refusal message and not only the exit code. ONEAPI_ROOT is set so the script does
# not source oneAPI. No model, no GPU.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT_DIR/scripts/sycl-qwen4exp-census-run.sh"
[ -r "$RUN" ] || { echo "FAIL: $RUN is missing"; exit 1; }
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0 cases=0

BAD_MSG="WATCHDOG_HOST_BACKING_GIB must be a whole number of GiB from 1 to 999 with no leading zero"
OLD_MSG="WATCHDOG_SHMEM_GB is no longer read: use WATCHDOG_HOST_BACKING_GIB"

run_script() { # env assignments...; output on stdout, exit code in $rc
    rc=0
    out="$(env LLAMA_COMPLETION=/nonexistent ONEAPI_ROOT=/nonexistent "$@" bash "$RUN" control-on 0 "$T/out" 2>&1)" || rc=$?
}

expect_refusal() { # $1 = label, $2 = the exact message, rest = env assignments
    local label="$1" want="$2"; shift 2
    local out rc
    cases=$((cases + 1))
    rm -rf "$T/out"
    run_script "$@"
    [ "$rc" -eq 3 ] || { echo "FAIL: $label: expected exit 3, got $rc ($out)"; fail=1; }
    case "$out" in *"$want"*) ;; *) echo "FAIL: $label: expected the message '$want' (got: $out)"; fail=1;; esac
    [ ! -e "$T/out" ] || { echo "FAIL: $label: the script created its output directory before refusing"; fail=1; }
}

expect_accepted() { # $1 = value: validation passes and the script goes on to its binary or build check
    local out rc
    cases=$((cases + 1))
    rm -rf "$T/out"
    run_script WATCHDOG_HOST_BACKING_GIB="$1"
    case "$out" in *WATCHDOG_HOST_BACKING_GIB*) echo "FAIL: '$1' must be accepted (got: $out)"; fail=1;; esac
    [ "$rc" -eq 3 ] || { echo "FAIL: '$1' should stop at the binary or build check with exit 3, got $rc ($out)"; fail=1; }
}

for v in 150.5 abc "" 0 00 08 09 018 0100 -5 " 5" 1000 123456789012345; do
    expect_refusal "value '$v'" "$BAD_MSG" WATCHDOG_HOST_BACKING_GIB="$v"
done
expect_refusal "the old name names the new one" "$OLD_MSG" WATCHDOG_SHMEM_GB=100
expect_refusal "the old name set empty is still refused" "$OLD_MSG" WATCHDOG_SHMEM_GB=
for v in 1 9 10 100 999; do expect_accepted "$v"; done

[ "$fail" -eq 0 ] || exit 1
echo "OK: test-sycl-qwen4exp-census-run-env ($cases cases)"
