#!/usr/bin/env bash
# Host-only test of the watchdog ceiling variable of scripts/sycl-qwen4exp-census-run.sh. Every case exits
# at the argument check, before the script sources oneAPI or launches anything: no GPU, no model.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT_DIR/scripts/sycl-qwen4exp-census-run.sh"
[ -r "$RUN" ] || { echo "FAIL: $RUN is missing"; exit 1; }
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0 cases=0

expect_refusal() { # $1 = label, $2 = text the message must carry, rest = env assignments
    local label="$1" want="$2"; shift 2
    local out rc=0
    cases=$((cases + 1))
    out="$(env "$@" bash "$RUN" control-on 0 "$T/out" 2>&1)" || rc=$?
    [ "$rc" -eq 3 ] || { echo "FAIL: $label: expected exit 3, got $rc ($out)"; fail=1; }
    case "$out" in *"$want"*) ;; *) echo "FAIL: $label: message must name $want (got: $out)"; fail=1;; esac
    [ ! -e "$T/out" ] || { echo "FAIL: $label: the script created its output directory before refusing"; fail=1; }
}

expect_refusal "a fraction" WATCHDOG_HOST_BACKING_GIB WATCHDOG_HOST_BACKING_GIB=150.5
expect_refusal "a word" WATCHDOG_HOST_BACKING_GIB WATCHDOG_HOST_BACKING_GIB=abc
expect_refusal "empty" WATCHDOG_HOST_BACKING_GIB WATCHDOG_HOST_BACKING_GIB=
expect_refusal "the old name names the new one" WATCHDOG_HOST_BACKING_GIB WATCHDOG_SHMEM_GB=100
expect_refusal "the old name set empty is still refused" WATCHDOG_HOST_BACKING_GIB WATCHDOG_SHMEM_GB=

[ "$fail" -eq 0 ] || exit 1
echo "OK: test-sycl-qwen4exp-census-run-env ($cases cases)"
