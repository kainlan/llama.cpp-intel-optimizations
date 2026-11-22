#!/usr/bin/env bash
# Stub test of scripts/sycl-inkling-profile.sh: --dry-run only, so nothing is locked, loaded or run.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S="$ROOT/scripts/sycl-inkling-profile.sh"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
n_fail=0
check() { # LABEL EXPECTED_RC REGEX ARGS...
    local label="$1" want="$2" re="$3"; shift 3
    local out rc=0
    out="$(bash "$S" "$@" 2>&1)" || rc=$?
    if [ "$rc" -ne "$want" ] || ! printf '%s\n' "$out" | grep -qE -- "$re"; then
        echo "FAIL: ($label) rc=$rc want=$want re=/$re/ out: $out"; n_fail=$((n_fail + 1))
    else
        echo "PASS: ($label)"
    fi
}
D=(--dry-run --out "$T/o")
check "b70 selector"   0 'selector=level_zero:0 '            --setup b70  --phase bench "${D[@]}"
check "b50 selector"   0 'selector=level_zero:1 '            --setup b50  --phase bench "${D[@]}"
check "both selector"  0 'selector=level_zero:0,1 '          --setup both --phase bench "${D[@]}"
check "multi gpu off"  0 'GGML_SYCL_MOE_MULTI_GPU=0'         --setup both --phase kprof-decode "${D[@]}"
check "tuning dir"     0 'GGML_SYCL_TUNING_CACHE_DIR=.*/tuning-b70-kprof-decode-c8192' --setup b70 --phase kprof-decode "${D[@]}"
check "kprof env"      0 'GGML_SYCL_KERNEL_PROFILE=1 '       --setup b70  --phase kprof-prompt "${D[@]}"
check "e2e env"        0 'GGML_SYCL_E2E_TG_PROFILE=1 '       --setup b70  --phase e2e-decode "${D[@]}"
check "census env"     0 'GGML_SCHED_DEBUG=2 .* -v'          --setup b70  --phase census "${D[@]}"
check "ctx flag"       0 'llama-completion .* -c 32768 '     --setup b70  --phase kprof-prompt --ctx 32768 "${D[@]}"
check "bench no ctx"   0 'llama-bench -m [^ ]+ -p 512 -n 128 -r 3$' --setup b70 --phase bench "${D[@]}"
check "bad setup"      1 'unknown setup'                     --setup b580 --phase bench "${D[@]}"
check "bad phase"      1 'unknown phase'                     --setup b70  --phase warp "${D[@]}"
# Every phase: no -ngl (token-anchored), and a positive control that the line names the model.
for ph in bench kprof-prompt kprof-decode e2e-decode census; do
    out="$(bash "$S" --setup b70 --phase "$ph" "${D[@]}" 2>&1)"
    if printf '%s\n' "$out" | grep -qE -- '(^| )-ngl( |$)'; then
        echo "FAIL: (no ngl, $ph) $out"; n_fail=$((n_fail + 1))
    elif ! printf '%s\n' "$out" | grep -q -- 'Inkling-2-Mini'; then
        echo "FAIL: (no ngl control, $ph: model path missing) $out"; n_fail=$((n_fail + 1))
    else
        echo "PASS: (no ngl, $ph)"
    fi
done
# Both kernel-profile phases ask for JSON and CSV; no other phase carries any kernel-profile variable.
check "kprof format prompt" 0 'GGML_SYCL_KERNEL_PROFILE_FORMAT=both' --setup b70 --phase kprof-prompt "${D[@]}"
check "kprof format decode" 0 'GGML_SYCL_KERNEL_PROFILE_FORMAT=both' --setup b70 --phase kprof-decode "${D[@]}"
out="$(bash "$S" --setup b70 --phase bench "${D[@]}" 2>&1)"
case "$out" in *GGML_SYCL_KERNEL_PROFILE*) echo "FAIL: (bench has no kprof env) $out"; n_fail=$((n_fail + 1));;
               *) echo "PASS: (bench has no kprof env)";; esac
# The printed command is the one that runs, kill escalation included.
check "timeout kill" 0 'timeout -k 30 900 ' --setup b70 --phase bench "${D[@]}"
out="$(bash "$S" --setup b70 --phase kprof-prompt "${D[@]}" 2>&1)"
case "$out" in *E2E_TG_PROFILE*|*GGML_SCHED_DEBUG*) echo "FAIL: (one instrument) $out"; n_fail=$((n_fail + 1));;
               *) echo "PASS: (one instrument)";; esac
if [ ! -e "$T/o" ]; then echo "PASS: (dry run creates nothing)"; else echo "FAIL: (dry run created $T/o)"; n_fail=$((n_fail + 1)); fi
[ "$n_fail" -eq 0 ] || exit 1
