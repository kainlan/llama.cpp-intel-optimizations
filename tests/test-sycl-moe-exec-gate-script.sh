#!/usr/bin/env bash
# GPU-free tests for scripts/sycl-moe-exec-gate.sh (llama.cpp-x6zo, plan Task 15 of llama.cpp-36j0). The script's
# scoring is a pure function of files, so this feeds it fixture logs shaped like the strings the code prints
# ([MOE-EXEC] layer/summary/loan/refill, [MOE-EXEC-TRACE] executor, [PP-MOE]) and llama-bench csv, and asserts
# the verdict strings. It loads no model, touches no device, takes no GPU.lock and runs no llama binary.
#
# Cases (each asserts the exact verdict text the lead will read):
#   - the grep controls hold (every regex matches its own line and not its nearest decoy; the GPT-OSS echo line
#     scores 0 under grep -cx and 2 under a loose grep -c);
#   - Q_pp: a pass (Q_pp=PASS), a failing pp512 (Q_pp=FAIL(pp512=412<500)), the log a real default-verbosity auto run
#     prints (24 host layers, "ring-unavailable" reasons, one summary per llama-bench context, zero ring
#     layer-ubatches: ring_engaged=no), ring_engaged read from the summary and not from the unavailable WARN
#     (which auto does not print), layer count, missing summary and [PP-MOE] refused each fail by name,
#     loan_exercised counts a granted loan and not a refused one;
#   - Q_staged: only the layers the ring served are scored (the executor also dispatches resident layers, whose
#     staged=0 lines are correct): a pass with resident staged=0 lines, a ring-served layer at staged=0, a non-f16
#     ring-served arm, a refused line, a missing arm=f16 line, no ring-served dispatch, and NOT_EXERCISED when no
#     layer-ubatch used the ring;
#   - the summary: ring_engaged_any yes, no and na, exit 3 for a Q_pp that passed without the ring, loan_exercised_any,
#     a FAIL beating it;
#   - Q_tok / Q_tok_2 / Q_tok_cpu_1 recorded; Q_tok_cpu across two loads per arm, scored by common prefix because
#     the B70's greedy near-tie flips in both policies: some auto md5 equals some cpu md5 and every auto output
#     shares >= 64 bytes with its best cpu match (fixtures at 64 and 63 bytes), a defect (pairs_equal=0, early
#     min_prefix) against a flip (pairs_equal >= 1, late), a truncated output, a missing reference;
#   - Q_tg / Q_noc: floor, md5 equal to the reference, or a flip (>= 128 bytes shared with the committed reference
#     text, fixtures at 128 and 127 bytes) and the md5 of that committed text (25b7738a); the place line (unscored
#     by default, unchanged, changed); refill balanced / unbalanced / not_exercised; watchdog;
#   - GO_gate / GO_noc count and layer lines (the echo line alone must not score), GO_pp window, MI digits,
#     resets;
#   - --dry-run lists every arm, pins the selector in all fourteen device arms, passes no -ngl, no -cnv to llama-cli,
#     and runs nothing; --help names the arms;
#   - a real run refuses without a <sha> or with a wrong one, prints verdicts_pass=0 GATE_DONE, exits 2 and never
#     creates GPU.lock; the lock helper takes, refuses a second take, and releases;
#   - --make-prompt is deterministic and the committed prompt exists;
#   - oneAPI's setvars.sh overwrites shell variables (the advisor and vtune vars.sh assign BIN_DIR=bin64), which made
#     the first real run refuse with "no bin64/llama-bench": no bare BIN_DIR=, setvars sourced before the first use of
#     GATE_BIN_DIR, no top-level name in common with oneAPI's, and a real run's refusal names the build's own bin;
#   - the exit-143 path: a TERM kills the child, releases GPU.lock, prints the final line;
#   - RING_ARMS: every arm in the list prints ring_engaged, and an arm outside it cannot move ring_engaged_any;
#   - the arms fail closed: every arm scores rc != 0 as a FAIL (a GO or MI run that prints the answer and aborts in
#     teardown must FAIL); the resets arm keeps journalctl's rc and fails a missing journal; Q_tok_cpu needs all four
#     outputs non-empty and all four rcs 0; a RECORDED line with rc != 0 is exit 1; tps anchors on the eval line, so
#     a nan eval is no_tg; refill is checked per device line;
#   - a usage error prints the final line; the sha format is checked under --no-head-check; the head check reads
#     the build's own source tree (--src-dir, a snapshot copy of the script); an exported GGML_SYCL_* or LLAMA_ARG_*
#     variable and a non-empty --out-dir are refused (the journal arm alone runs the real flow GPU-free);
#   - a value-taking option given last is refused (rc 2, final line), never spun on;
#   - builders: builders_pre and builders_post on every verdict (the larger of the count after an arm and the
#     highest the sampler saw during it), a taint makes NOT_SCORED (exit 3), the start waits for a build before
#     the settle check and the lock, a builder wait is bounded, and every stop once the lock is held prints the
#     summary over the verdicts scored so far (the --refuse-selftest, --signal-selftest and --wait-builders-selftest
#     hooks);
#   - prerequisites: a missing awk, md5sum, cmp, comm or journalctl is exit 77, in the test and in the gate.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GATE="$ROOT_DIR/scripts/sycl-moe-exec-gate.sh"
[ -f "$GATE" ] || { echo "SKIP: no $GATE"; exit 77; }
# A missing tool is a skip (77), never a pass: a score through a missing awk or cmp would prove nothing.
for tool in awk md5sum cmp comm journalctl git pgrep timeout; do
    command -v "$tool" > /dev/null 2>&1 || { echo "SKIP: no $tool on PATH"; exit 77; }
done
# The gate refuses a real run when a GGML_SYCL_* or LLAMA_ARG_* variable is exported; this test must not inherit one.
for v in $(compgen -e | grep -E '^(GGML_SYCL_|LLAMA_ARG_)'); do unset "$v"; done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/moe-exec-gate-test.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
FAILS=0
CASES=0

pass() { CASES=$((CASES + 1)); echo "PASS: $1"; }
fail() { CASES=$((CASES + 1)); FAILS=$((FAILS + 1)); echo "FAIL: $1"; [ -z "${2:-}" ] || echo "      got: $2"; }

# expect <name> <haystack> <needle>: needle appears (fixed string).
expect() {
    if printf '%s\n' "$2" | grep -qF -- "$3"; then pass "$1"; else fail "$1 (wanted: $3)" "$2"; fi
}
# expect_not <name> <haystack> <needle>
expect_not() {
    if printf '%s\n' "$2" | grep -qF -- "$3"; then fail "$1 (must not contain: $3)" "$2"; else pass "$1"; fi
}
# expect_rc <name> <got> <want>
expect_rc() {
    if [ "$2" = "$3" ]; then pass "$1"; else fail "$1 (rc $2, wanted $3)"; fi
}

FIELDS='build_commit build_number cpu_info gpu_info backends model_filename model_type model_size model_n_params
n_batch n_ubatch n_threads cpu_mask cpu_strict poll type_k type_v n_gpu_layers n_cpu_moe split_mode main_gpu
no_kv_offload flash_attn devices tensor_split tensor_buft_overrides load_mode lazy_mode embeddings no_op_offload
no_host repack fit_target fit_min_ctx n_prompt n_gen n_depth test_time avg_ns stddev_ns avg_ts stddev_ts'

# mkcsv <file> <row>...: the llama-bench csv header, then one row per
# "n_prompt:n_gen:avg_ts:stddev_ts:n_ubatch[:n_depth]".
mkcsv() {
    local f="$1" r
    shift
    # shellcheck disable=SC2086
    echo $FIELDS | tr ' ' ',' > "$f"
    for r in "$@"; do
        IFS=: read -r np ng ts sd ub dp <<< "$r"
        # shellcheck disable=SC2086
        for k in $FIELDS; do
            case "$k" in
                n_prompt) v=$np ;; n_gen) v=$ng ;; avg_ts) v=$ts ;; stddev_ts) v=$sd ;; n_ubatch) v=$ub ;;
                n_depth) v=${dp:-0} ;; cpu_info) v='Intel(R) Core(TM), 24 threads' ;; *) v=x ;;
            esac
            printf '"%s",' "$v"
        done | sed 's/,$//' >> "$f"
        echo >> "$f"
    done
}

# mklayers <file> <n> [word]: n distinct "[MOE-EXEC] layer N:" lines.
mklayers() {
    local i
    for i in $(seq 0 $(($2 - 1))); do
        echo "0.05.100.000 W [MOE-EXEC] layer $i: ${3:-ring} (p=0.99 t_cpu=900 us t_ring=300 us cheaper)" >> "$1"
    done
}

TRACE_EXEC_PREFIX='0.05.100.000 I [MOE-EXEC-TRACE] executor device 0'
SUMMARY='0.09.000.000 W [MOE-EXEC] summary: 96 layer-ubatches ring, 0 cpu, 0 forced, 1500 MiB streamed'
TRACE_DECOY='0.05.200.000 I [MOE-EXEC-TRACE] dev 0 layer 3: ring (p=0.99 t_cpu=900 us t_ring=300 us cheaper)'
UNAVAIL='0.04.000.000 W [MOE-EXEC] device 0: stream ring unavailable: cap'
UNAVAIL="$UNAVAIL (host_matrices_per_layer_max=3 n_slots=0 bytes=0)"
SUMMARY_NORING='0.09.000.000 W [MOE-EXEC] summary: 0 layer-ubatches ring, 96 cpu, 0 forced, 0 MiB streamed'
SUMMARY_NORING2='0.19.000.000 W [MOE-EXEC] summary: 0 layer-ubatches ring, 372 cpu, 0 forced, 0 MiB streamed'
SUMMARY_RING2='0.19.000.000 W [MOE-EXEC] summary: 40 layer-ubatches ring, 7 cpu, 0 forced, 600 MiB streamed'
PPMOE_REFUSED='0.07.000.000 W [PP-MOE] refused device=0 cause=x detail=- tensor=blk.3.ffn_up_exps.weight'
PPMOE_REFUSED="$PPMOE_REFUSED type=q4_0 layout=aos staged=4"
LOAN_OK='0.06.000.000 W [MOE-EXEC] loan device 0: 24 triplets, 300 MiB lent to the ring (ok)'
LOAN_REFUSED='0.06.000.000 W [MOE-EXEC] loan device 0 refused: host storage too small'

score() { bash "$GATE" --score-arm "$@" 2>&1; }

# ---- controls
out="$(bash "$GATE" --check-controls 2>&1)"; rc=$?
expect_rc "controls rc" "$rc" 0
expect "controls print ok" "$out" "controls: ok"

# ---- Q_pp
mkcsv "$WORK/pp-pass.csv" 512:0:523.4:3.1:512 2048:0:851.2:5.0:2048
mkcsv "$WORK/pp-fail.csv" 512:0:412.3:2.0:512 2048:0:851.2:5.0:2048
mkcsv "$WORK/pp-slow.csv" 512:0:14.1:0.2:256 2048:0:31.5:0.3:256
{ mklayers "$WORK/pp-pass.err" 48; echo "$SUMMARY" >> "$WORK/pp-pass.err"; echo "$TRACE_DECOY" >> "$WORK/pp-pass.err"; }
cp "$WORK/pp-pass.err" "$WORK/pp-fail.err"
# The log a real auto run prints at default verbosity (3460-gpu2-q38-tip-1): layers 24..47 each decide cpu with
# the reason ring-unavailable, and llama-bench prints one summary per context (two for -p 512,2048), with zero
# ring layer-ubatches. The "stream ring unavailable" WARN is absent: auto does not print it.
for i in $(seq 24 47); do
    echo "1.36.000.000 W [MOE-EXEC] layer $i: cpu (p=1.00 t_cpu=201827 us t_ring=45000 us ring-unavailable)"
done > "$WORK/pp-refused.err"
echo "$SUMMARY_NORING" >> "$WORK/pp-refused.err"
echo "$SUMMARY_NORING2" >> "$WORK/pp-refused.err"

out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-pass.err")"; rc=$?
expect "Q_pp pass verdict" "$out" "Q_pp=PASS"
expect "Q_pp pass ring_engaged=yes" "$out" "ring_engaged=yes"
expect "Q_pp pass loan_exercised=no" "$out" "loan_exercised=no"
expect "Q_pp pass reports the auto ubatch" "$out" "ub=512"
expect_rc "Q_pp pass rc" "$rc" 0

out="$(score Q_pp --csv "$WORK/pp-fail.csv" --err "$WORK/pp-fail.err")"; rc=$?
expect "Q_pp fail verdict" "$out" "Q_pp=FAIL(pp512=412<500)"
expect_not "Q_pp fail names only pp512" "$out" "pp2048=851.2<"
expect_rc "Q_pp fail rc" "$rc" 1

out="$(score Q_pp --csv "$WORK/pp-slow.csv" --err "$WORK/pp-refused.err")"
expect "Q_pp ring-refused says ring_engaged=no" "$out" "ring_engaged=no"
expect "Q_pp ring-refused counts no ring layer-ubatches" "$out" "ring_layer_ubatches=0"
expect "Q_pp the real log has 24 layers and two summaries" "$out" "layers=24 summary=2"
expect "Q_pp ring-refused fails the floors" "$out" "Q_pp=FAIL(pp512=14<500,pp2048=32<800)"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-refused.err")"
expect "Q_pp ring_engaged=no is independent of the verdict" "$out" "ring_engaged=no"
expect "Q_pp passing numbers without the ring still pass honestly" "$out" "Q_pp=PASS"
# The unavailable WARN is not the signal: with it printed AND ring layer-ubatches counted, the ring engaged; with
# it absent and zero ring layer-ubatches (the real auto log above), it did not.
{ echo "$UNAVAIL"; cat "$WORK/pp-pass.err"; } > "$WORK/pp-mixed.err"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-mixed.err")"
expect "Q_pp the unavailable WARN alone does not decide ring_engaged" "$out" "ring_engaged=yes"
{ cat "$WORK/pp-pass.err"; echo "$SUMMARY_RING2"; } > "$WORK/pp-sum2.err"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-sum2.err")"
expect "Q_pp ring layer-ubatches add up over the summaries" "$out" "ring_layer_ubatches=136"
expect "Q_pp two summaries pass (one per llama-bench context)" "$out" "Q_pp=PASS"

cp "$WORK/pp-pass.err" "$WORK/pp-47.err"
sed -i '/layer 47:/d' "$WORK/pp-47.err"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-47.err" --q-layers 48)"
expect "Q_pp --q-layers 48 and 47 layers fails by name" "$out" "Q_pp=FAIL(layers=47!=48"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-47.err")"
expect "Q_pp by default any layer count of at least one passes" "$out" "Q_pp=PASS"
grep -v 'layer [0-9]*:' "$WORK/pp-pass.err" > "$WORK/pp-nolayer.err"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-nolayer.err")"
expect "Q_pp no layer line fails by name" "$out" "Q_pp=FAIL(layers=0<1"
grep -v 'summary:' "$WORK/pp-pass.err" > "$WORK/pp-nosum.err"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-nosum.err")"
expect "Q_pp no summary fails by name" "$out" "Q_pp=FAIL(summary=0<1"
expect "Q_pp no summary also means ring_engaged=no" "$out" "ring_engaged=no"
cp "$WORK/pp-pass.err" "$WORK/pp-ref.err"
echo "$PPMOE_REFUSED" >> "$WORK/pp-ref.err"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-ref.err")"
expect "Q_pp [PP-MOE] refused fails by name" "$out" "Q_pp=FAIL(pp_moe_refused=1)"
cp "$WORK/pp-pass.err" "$WORK/pp-loan.err"; echo "$LOAN_REFUSED" >> "$WORK/pp-loan.err"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-loan.err")"
expect "Q_pp a refused loan is not loan_exercised" "$out" "loan_exercised=no"
echo "$LOAN_OK" >> "$WORK/pp-loan.err"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-loan.err")"
expect "Q_pp a granted loan is loan_exercised" "$out" "loan_exercised=yes"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-pass.err" --rc 124)"
expect "Q_pp a timeout is a failure" "$out" "Q_pp=FAIL(rc=124"
mkcsv "$WORK/pp-none.csv" 128:0:900:1:128
out="$(score Q_pp --csv "$WORK/pp-none.csv" --err "$WORK/pp-pass.err")"
expect "Q_pp a csv without the rows fails, never passes" "$out" "Q_pp=FAIL(no_pp512_row"

out="$(score Q_pp_cpu --csv "$WORK/pp-slow.csv" --err "$WORK/pp-refused.err")"
expect "Q_pp_cpu is recorded" "$out" "Q_pp_cpu=RECORDED pp512=14.1 pp2048=31.5"

# ---- Q_staged: layers 0-2 are ring layers (host experts), layers 10 and 11 are resident and run the executor at
# staged=0, layer 5 decided cpu. Only blk.0-2 are scored.
mklayers "$WORK/st-pass.err" 3
{
    echo "0.05.050.000 W [MOE-EXEC] layer 5: cpu (p=1.00 t_cpu=100 us t_ring=900 us cheaper)"
    echo '0.05.000.000 I [PP-MOE] device=0 type=q4_0 layout=aos arm=f16 experts=8 groups=1 planned_weight_slot=100'
    for i in 0 1 2; do
        echo "$TRACE_EXEC_PREFIX tensor=blk.$i.ffn_up_exps.weight experts=8 staged=8 arm=f16"
    done
    for i in 5 10 11; do
        echo "$TRACE_EXEC_PREFIX tensor=blk.$i.ffn_up_exps.weight experts=8 staged=0 arm=f16"
    done
    echo "$TRACE_DECOY"
    echo "$SUMMARY"
} >> "$WORK/st-pass.err"
out="$(score Q_staged --err "$WORK/st-pass.err")"
expect "Q_staged passes with resident staged=0 lines present" "$out" \
    "Q_staged=PASS ring_layers=3 executor_dispatches=6 ring_layer_dispatches=3"
sed 's/blk.1.ffn_up_exps.weight experts=8 staged=8/blk.1.ffn_up_exps.weight experts=8 staged=0/' \
    "$WORK/st-pass.err" > "$WORK/st-unstaged.err"
out="$(score Q_staged --err "$WORK/st-unstaged.err")"
expect "Q_staged a ring-served layer at staged=0 fails by name" "$out" "Q_staged=FAIL(unstaged=1/3)"
sed 's/blk.2.ffn_up_exps.weight experts=8 staged=8 arm=f16/blk.2.ffn_up_exps.weight experts=8 staged=8 arm=woq/' \
    "$WORK/st-pass.err" > "$WORK/st-woq.err"
out="$(score Q_staged --err "$WORK/st-woq.err")"
expect "Q_staged a non-f16 ring-served arm fails by name" "$out" "Q_staged=FAIL(arm_not_f16=1/3)"
sed 's/blk.10.ffn_up_exps.weight experts=8 staged=0 arm=f16/blk.10.ffn_up_exps.weight experts=8 staged=0 arm=woq/' \
    "$WORK/st-pass.err" > "$WORK/st-resident-woq.err"
out="$(score Q_staged --err "$WORK/st-resident-woq.err")"
expect "Q_staged a resident layer's arm is not scored" "$out" "Q_staged=PASS"
cp "$WORK/st-pass.err" "$WORK/st-refused.err"
echo "$PPMOE_REFUSED" >> "$WORK/st-refused.err"
out="$(score Q_staged --err "$WORK/st-refused.err")"; rc=$?
expect "Q_staged a [PP-MOE] refused line fails by name" "$out" "Q_staged=FAIL(pp_moe_refused=1)"
expect_rc "Q_staged a refused line rc" "$rc" 1
grep -v 'PP-MOE\] device=' "$WORK/st-pass.err" > "$WORK/st-noarm.err"
out="$(score Q_staged --err "$WORK/st-noarm.err")"
expect "Q_staged a missing [PP-MOE] arm=f16 line fails by name" "$out" "Q_staged=FAIL(no_pp_moe_arm_f16_line)"
grep -v 'tensor=blk\.[012]\.' "$WORK/st-pass.err" > "$WORK/st-nodispatch.err"
out="$(score Q_staged --err "$WORK/st-nodispatch.err")"
expect "Q_staged no ring-served dispatch fails, never passes" "$out" "Q_staged=FAIL(no_ring_layer_dispatch)"
{ echo "$UNAVAIL"; grep -v 'summary:' "$WORK/st-pass.err"; echo "$SUMMARY_NORING"; } > "$WORK/st-unavail.err"
out="$(score Q_staged --err "$WORK/st-unavail.err")"; rc=$?
expect "Q_staged ring-unavailable is NOT_EXERCISED" "$out" "Q_staged=NOT_EXERCISED(ring_engaged=no)"
expect_not "Q_staged ring-unavailable is not a pass" "$out" "Q_staged=PASS"
expect_rc "Q_staged ring-unavailable rc" "$rc" 3
{ echo "$TRACE_DECOY"; echo "$SUMMARY"; } > "$WORK/st-none.err"
out="$(score Q_staged --err "$WORK/st-none.err")"
expect "Q_staged without executor lines fails, never passes" "$out" "no_ring_layer_dispatch"

# ---- Q_tok / Q_tok_cpu: two loads per arm, stdout only
printf 'the same generated text\n' > "$WORK/tok-a1.out"
printf 'the same generated text\n' > "$WORK/tok-a2.out"
printf 'the same generated text\n' > "$WORK/tok-c1.out"
printf 'the same generated text\n' > "$WORK/tok-c2.out"
printf 'the same generated tent\n' > "$WORK/tok-flip.out"
printf 'a wholly different text\n' > "$WORK/tok-bad.out"
for arm in Q_tok Q_tok_2 Q_tok_cpu_1; do
    out="$(score "$arm" --out "$WORK/tok-a1.out")"
    expect "$arm is recorded" "$out" "$arm=RECORDED md5="
done
tokcpu() { score Q_tok_cpu --out "$1" --out2 "$2" --ref-out "$3" --ref-out2 "$4"; }
out="$(tokcpu "$WORK/tok-c2.out" "$WORK/tok-c1.out" "$WORK/tok-a1.out" "$WORK/tok-a2.out")"
expect "Q_tok_cpu four equal outputs pass" "$out" "Q_tok_cpu=PASS"
expect "Q_tok_cpu reports no flip when there is none" "$out" "auto_flip=no cpu_flip=no"
# the B70 greedy flip: the second cpu load differs, and the auto outputs equal one of the two cpu outputs
out="$(tokcpu "$WORK/tok-flip.out" "$WORK/tok-c1.out" "$WORK/tok-a1.out" "$WORK/tok-a2.out")"
expect "Q_tok_cpu a cpu flip with auto equal to one cpu output passes" "$out" "Q_tok_cpu=PASS"
expect "Q_tok_cpu reports the cpu flip" "$out" "cpu_flip=yes"
out="$(tokcpu "$WORK/tok-c2.out" "$WORK/tok-c1.out" "$WORK/tok-a1.out" "$WORK/tok-flip.out")"
expect "Q_tok_cpu an auto output that diverges early from every cpu output fails" "$out" \
    "Q_tok_cpu=FAIL(auto2_prefix=21<64)"
out="$(tokcpu "$WORK/tok-flip.out" "$WORK/tok-c1.out" "$WORK/tok-a1.out" "$WORK/tok-flip.out")"
expect "Q_tok_cpu an auto flip that either cpu load also made passes" "$out" "Q_tok_cpu=PASS"
expect "Q_tok_cpu reports the auto flip" "$out" "auto_flip=yes"
out="$(tokcpu "$WORK/tok-c2.out" "$WORK/tok-c1.out" "$WORK/tok-bad.out" "$WORK/tok-a2.out")"; rc=$?
expect "Q_tok_cpu a real mismatch fails by name" "$out" "Q_tok_cpu=FAIL(auto1_prefix=0<64)"
expect "Q_tok_cpu a mismatch reports its prefix" "$out" "min_prefix=0"
expect_rc "Q_tok_cpu a mismatch rc" "$rc" 1
out="$(tokcpu "$WORK/tok-c2.out" "$WORK/tok-c1.out" "$WORK/tok-flip.out" "$WORK/tok-a2.out")"
expect "Q_tok_cpu a one-byte difference at byte 22 is a prefix of 21" "$out" "auto1_prefix=21<64"
out="$(tokcpu "$WORK/tok-c2.out" "$WORK/tok-c1.out" "$WORK/absent.out" "$WORK/tok-a2.out")"; rc=$?
expect "Q_tok_cpu without Q_tok is not scored" "$out" "Q_tok_cpu=NOT_SCORED(no_Q_tok_output)"
expect_rc "Q_tok_cpu not scored rc" "$rc" 3
out="$(tokcpu "$WORK/tok-c2.out" "$WORK/absent.out" "$WORK/tok-a1.out" "$WORK/tok-a2.out")"
expect "Q_tok_cpu without Q_tok_cpu_1 is not scored" "$out" "Q_tok_cpu=NOT_SCORED(no_Q_tok_cpu_1_output)"
: > "$WORK/tok-empty.out"
out="$(tokcpu "$WORK/tok-empty.out" "$WORK/tok-c1.out" "$WORK/tok-a1.out" "$WORK/tok-a2.out")"
expect "Q_tok_cpu an empty cpu output fails by name" "$out" "Q_tok_cpu=FAIL(empty_output"
out="$(score Q_tok_cpu --out "$WORK/tok-c2.out" --out2 "$WORK/tok-c1.out" --ref-out "$WORK/tok-a1.out" \
    --ref-out2 "$WORK/tok-a2.out" --rc 124)"
expect "Q_tok_cpu a timeout fails" "$out" "Q_tok_cpu=FAIL(rc=124"

# Q_tok_cpu: the greedy near-tie flips in both policies and cascades after its token. The rule is: some auto md5
# equals some cpu md5, and every auto output is identical to a cpu output or shares >= 64 bytes with its
# best-matching one.
TOKBASE='The user wants me to summarize the document about GPT-OSS Testing and Prompt Template Rationale'
TOKBASE="$TOKBASE in plain words."
printf '%s\n' "$TOKBASE" > "$WORK/lt-base.out"
sed 's/plain words/plain WORDS/' "$WORK/lt-base.out" > "$WORK/lt-late1.out"
sed 's/in plain words/in plain sentences/' "$WORK/lt-base.out" > "$WORK/lt-late2.out"
sed 's/summarize/SUMMARIZE/' "$WORK/lt-base.out" > "$WORK/lt-early.out"
printf 'An entirely different opening that shares nothing with the other text at all.\n' > "$WORK/lt-other.out"
ltok() { # <cpu2> <cpu1> <auto1> <auto2>, names under $WORK/lt-
    tokcpu "$WORK/lt-$1.out" "$WORK/lt-$2.out" "$WORK/lt-$3.out" "$WORK/lt-$4.out"
}
out="$(ltok base base base base)"
expect "Q_tok_cpu four identical long outputs: distinct 1/1, pairs_equal 4" "$out" \
    "Q_tok_cpu=PASS distinct_auto=1 distinct_cpu=1 pairs_equal=4"
# run 2's shape: auto 1 equals cpu 2, every other pair diverges late, both policies flip
out="$(ltok base late1 base late2)"; rc=$?
expect "Q_tok_cpu run-2 shape (one equal pair, late flips on both sides) passes" "$out" \
    "Q_tok_cpu=PASS distinct_auto=2 distinct_cpu=2 pairs_equal=1 min_prefix=105"
expect_rc "Q_tok_cpu run-2 shape rc" "$rc" 0
out="$(ltok late1 base base late2)"
expect "Q_tok_cpu an auto flip against a late-flipped cpu pair still passes" "$out" "Q_tok_cpu=PASS"
# a defect: the policies agree on nothing, whatever the prefix
out="$(ltok base base late1 late1)"; rc=$?
expect "Q_tok_cpu no equal pair fails even with a late prefix" "$out" "Q_tok_cpu=FAIL(pairs_equal=0)"
expect "Q_tok_cpu pairs_equal=0 prints it" "$out" "pairs_equal=0"
expect_rc "Q_tok_cpu no equal pair rc" "$rc" 1
out="$(ltok base base base early)"
expect "Q_tok_cpu an auto output that diverges early fails naming the prefix" "$out" \
    "Q_tok_cpu=FAIL(auto2_prefix=21<64)"
expect "Q_tok_cpu an early auto prints min_prefix" "$out" "min_prefix=21"
out="$(ltok base base other other)"
expect "Q_tok_cpu a wholly different pair fails on both counts" "$out" \
    "Q_tok_cpu=FAIL(auto1_prefix=0<64,auto2_prefix=0<64,pairs_equal=0)"
# the 64-byte boundary, pinned both sides: a 64-byte shared prefix passes, 63 fails
P64="$(printf 'a%.0s' $(seq 64))"
P63="$(printf 'a%.0s' $(seq 63))"
printf '%sX\n' "$P64" > "$WORK/lt-p64a.out"; printf '%sY\n' "$P64" > "$WORK/lt-p64c.out"
printf '%sX\n' "$P63" > "$WORK/lt-p63a.out"; printf '%sY\n' "$P63" > "$WORK/lt-p63c.out"
out="$(tokcpu "$WORK/lt-base.out" "$WORK/lt-p64c.out" "$WORK/lt-base.out" "$WORK/lt-p64a.out")"
expect "Q_tok_cpu a shared prefix of exactly 64 bytes passes" "$out" "Q_tok_cpu=PASS"
out="$(tokcpu "$WORK/lt-base.out" "$WORK/lt-p63c.out" "$WORK/lt-base.out" "$WORK/lt-p63a.out")"
expect "Q_tok_cpu a shared prefix of 63 bytes fails" "$out" "Q_tok_cpu=FAIL(auto2_prefix=63<64)"
# a prefix of the other output (one is a prefix of the other) counts the shorter length
head -c 80 "$WORK/lt-base.out" > "$WORK/lt-trunc.out"
out="$(tokcpu "$WORK/lt-base.out" "$WORK/lt-base.out" "$WORK/lt-base.out" "$WORK/lt-trunc.out")"
expect "Q_tok_cpu a truncated auto output counts the shared length (80 >= 64)" "$out" "Q_tok_cpu=PASS"
head -c 30 "$WORK/lt-base.out" > "$WORK/lt-trunc30.out"
out="$(tokcpu "$WORK/lt-base.out" "$WORK/lt-base.out" "$WORK/lt-base.out" "$WORK/lt-trunc30.out")"
expect "Q_tok_cpu a truncated auto output of 30 bytes fails" "$out" "Q_tok_cpu=FAIL(auto2_prefix=30<64)"

# ---- Q_tg / Q_noc
echo "decode output" > "$WORK/dec.out"
MD5="$(md5sum < "$WORK/dec.out" | cut -c1-8)"
PLACE_TG='MoE experts: 38004 on device, 35724 on host'
PERF_EVAL_HEAD='1.42.962.387 I llama_perf_context_print:        eval time =    4954.02 ms /    59 runs'
dec_err() { # <file> <tps> <place> [extra line]
    {
        echo "0.01.601.984 I [PLACEMENT] $3 (of 73728 total)"
        echo "$PERF_EVAL_HEAD   (83.97 ms per token,    $2 tokens per second)"
        [ -z "${4:-}" ] || echo "$4"
    } > "$1"
}
dec_err "$WORK/tg.err" 11.91 "$PLACE_TG" '0.09.000.000 W [MOE-EXEC] refill device 0: 24/24 triplets re-admitted'
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg.err" --qmd5 "$MD5" --place-ref-tg "$PLACE_TG")"
expect "Q_tg pass" "$out" "Q_tg=PASS tg=11.91"
expect "Q_tg refill balanced" "$out" "refill=balanced(24/24)"
expect "Q_tg place unchanged" "$out" "place=unchanged"
dec_err "$WORK/tg-slow.err" 10.20 "$PLACE_TG"
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg-slow.err" --qmd5 "$MD5" --place-ref-tg "$PLACE_TG")"
expect "Q_tg below the floor fails" "$out" "Q_tg=FAIL(tg=10.20<11.0)"
expect "Q_tg without a loan says refill not_exercised" "$out" "refill=not_exercised"
expect "Q_tg without a loan says loan_exercised=no" "$out" "loan_exercised=no"
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg.err" --qmd5 deadbeef --place-ref-tg "$PLACE_TG")"
expect "Q_tg wrong md5 with no shared text fails, flip=no" "$out" \
    "Q_tg=FAIL(md5=$MD5!=deadbeef,flip=no(prefix=0<128))"
expect "Q_tg an equal md5 is flip=no" "$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg.err" --qmd5 "$MD5" \
    --place-ref-tg "$PLACE_TG")" "md5=$MD5 flip=no prefix="
dec_err "$WORK/tg-unbal.err" 11.91 "$PLACE_TG" '0.09.000.000 W [MOE-EXEC] refill device 0: 23/24 triplets re-admitted'
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg-unbal.err" --qmd5 "$MD5" --place-ref-tg "$PLACE_TG")"
expect "Q_tg unbalanced refill fails" "$out" "Q_tg=FAIL(refill=unbalanced(23/24))"
dec_err "$WORK/tg-place.err" 11.91 'MoE experts: 37000 on device, 36728 on host'
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg-place.err" --qmd5 "$MD5" --place-ref-tg "$PLACE_TG")"
expect "Q_tg changed place line fails" "$out" "Q_tg=FAIL(place='MoE experts: 37000 on device, 36728 on host'!="
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg-place.err" --qmd5 "$MD5" --place-ref-tg -)"
expect "Q_tg place ref - is reported as unscored" "$out" "place=unscored"
dec_err "$WORK/tg-wd.err" 11.91 "$PLACE_TG" '0.09.000.000 W [SYCL-WATCHDOG] No GPU progress for 30 s'
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg-wd.err" --qmd5 "$MD5" --place-ref-tg "$PLACE_TG")"
expect "Q_tg watchdog fails by name" "$out" "Q_tg=FAIL(watchdog=1)"
dec_err "$WORK/noc.err" 9.80 'MoE experts: 38163 on device, 35565 on host'
out="$(score Q_noc --out "$WORK/dec.out" --err "$WORK/noc.err" --qmd5 "$MD5" \
    --place-ref-noc 'MoE experts: 38163 on device, 35565 on host')"
expect "Q_noc pass at 9.80 against the 9.5 floor" "$out" "Q_noc=PASS tg=9.80"
dec_err "$WORK/noc-slow.err" 9.30 'MoE experts: 38163 on device, 35565 on host'
out="$(score Q_noc --out "$WORK/dec.out" --err "$WORK/noc-slow.err" --qmd5 "$MD5" --place-ref-noc -)"
expect "Q_noc below the floor fails" "$out" "Q_noc=FAIL(tg=9.30<9.5)"

# Q_tg and Q_noc: a different md5 is a greedy near-tie flip when the output shares >= 128 bytes with the reference TEXT
head -c 300 /dev/zero | tr '\0' 'a' > "$WORK/dref.txt"
DREF_MD5="$(md5sum < "$WORK/dref.txt" | cut -c1-8)"
{ head -c 199 "$WORK/dref.txt"; printf 'Z'; head -c 100 "$WORK/dref.txt"; } > "$WORK/dflip.out"
{ head -c 49 "$WORK/dref.txt"; printf 'Z'; head -c 250 "$WORK/dref.txt"; } > "$WORK/dearly.out"
{ head -c 128 "$WORK/dref.txt"; printf 'Z'; head -c 171 "$WORK/dref.txt"; } > "$WORK/d128.out"
{ head -c 127 "$WORK/dref.txt"; printf 'Z'; head -c 172 "$WORK/dref.txt"; } > "$WORK/d127.out"
dscore() { # <arm> <out> [extra option...]
    local arm="$1" o="$2" err="$WORK/tg.err" pl="$PLACE_TG"
    shift 2
    [ "$arm" != Q_noc ] || { err="$WORK/noc.err"; pl=-; }
    local place_opt=--place-ref-tg
    [ "$arm" != Q_noc ] || place_opt=--place-ref-noc
    score "$arm" --out "$o" --err "$err" --qmd5 "$DREF_MD5" --decode-ref "$WORK/dref.txt" "$place_opt" "$pl" "$@"
}
out="$(dscore Q_tg "$WORK/dref.txt")"
expect "Q_tg the reference text itself is PASS with flip=no" "$out" \
    "Q_tg=PASS tg=11.91 md5=$DREF_MD5 flip=no prefix=300"
out="$(dscore Q_tg "$WORK/dflip.out")"; rc=$?
expect "Q_tg a late divergence (byte 200) is PASS with flip=yes" "$out" "Q_tg=PASS tg=11.91"
expect "Q_tg a flip prints flip=yes and its prefix" "$out" "flip=yes prefix=199"
expect_rc "Q_tg a flip rc" "$rc" 0
out="$(dscore Q_tg "$WORK/dearly.out")"; rc=$?
expect "Q_tg an early divergence (byte 50) is a FAIL with flip=no" "$out" "flip=no(prefix=49<128))"
expect "Q_tg an early divergence names the verdict" "$out" "Q_tg=FAIL(md5="
expect_rc "Q_tg an early divergence rc" "$rc" 1
out="$(dscore Q_tg "$WORK/d128.out")"
expect "Q_tg a shared prefix of exactly 128 bytes is PASS flip=yes" "$out" "Q_tg=PASS tg=11.91 md5="
out="$(dscore Q_tg "$WORK/d127.out")"
expect "Q_tg a shared prefix of 127 bytes is a FAIL" "$out" "flip=no(prefix=127<128))"
out="$(score Q_tg --out "$WORK/dflip.out" --err "$WORK/tg.err" --qmd5 "$DREF_MD5" --decode-ref "$WORK/absent-ref.txt" \
    --place-ref-tg "$PLACE_TG")"
expect "Q_tg a missing reference text FAILs naming it, never a flip" "$out" "no_reference_text"
out="$(dscore Q_tg "$WORK/dflip.out" --rc 134)"
expect "Q_tg a flip with rc 134 is still a FAIL" "$out" "Q_tg=FAIL(rc=134)"
out="$(dscore Q_noc "$WORK/dflip.out")"
expect "Q_noc a late divergence is PASS with flip=yes" "$out" "Q_noc=PASS tg=9.80"
expect "Q_noc a flip prints flip=yes" "$out" "flip=yes prefix=199"
out="$(dscore Q_noc "$WORK/dearly.out")"
expect "Q_noc an early divergence is a FAIL with flip=no" "$out" "Q_noc=FAIL(md5="
# the placement references default to "-": the line is recorded and not scored, because the right value moves with the
# cost-based KV room and only the lead can name it for a tree
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg.err" --qmd5 "$MD5")"
expect "Q_tg without --place-ref-tg the place line is recorded, not scored" "$out" "place=unscored"
expect "Q_tg unscored place still passes" "$out" "Q_tg=PASS"
out="$(score Q_noc --out "$WORK/dec.out" --err "$WORK/noc.err" --qmd5 "$MD5")"
expect "Q_noc without --place-ref-noc the place line is recorded, not scored" "$out" "place=unscored"
# the committed reference text is the q0xu output: its md5 is the gate's default reference md5
REF_COMMITTED="$ROOT_DIR/scripts/sycl-moe-exec-prompts/code-ref-25b7738a.txt"
if [ -f "$REF_COMMITTED" ] && [ "$(md5sum < "$REF_COMMITTED" | cut -c1-8)" = 25b7738a ]; then
    pass "the committed reference text has md5 25b7738a"
else
    fail "the committed reference text is missing or its md5 is not 25b7738a"
fi

# ---- GPT-OSS and Mistral
ECHO='> Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5'
printf '%s\n1, 2, 3, 4, 5\n' "$ECHO" > "$WORK/go.out"
printf '%s\n' "$ECHO" > "$WORK/go-echo.out"
: > "$WORK/empty.err"
out="$(score GO_gate --out "$WORK/go.out" --err "$WORK/empty.err")"
expect "GO_gate answer line scores 1 and zero layer lines" "$out" "GO_gate=PASS count=1 layer_lines=0"
printf '%s\n1, 2, 3, 4, 5\n1, 2, 3, 4, 5\n' "$ECHO" > "$WORK/go-two.out"
out="$(score GO_gate --out "$WORK/go-two.out" --err "$WORK/empty.err")"
expect "GO_gate a count of 2 fails: the gate is exactly 1" "$out" "GO_gate=FAIL(count=2!=1)"
out="$(score GO_gate --out "$WORK/go-echo.out" --err "$WORK/empty.err")"
expect "GO_gate the echo line alone must not pass" "$out" "GO_gate=FAIL(count=0!=1)"
out="$(score GO_gate --out "$WORK/go.out" --err "$WORK/pp-pass.err")"
expect "GO_gate layer lines fail it" "$out" "GO_gate=FAIL(layer_lines=48!=0)"
mklayers "$WORK/go12.err" 12
out="$(score GO_noc --out "$WORK/go.out" --err "$WORK/go12.err")"
expect "GO_noc 12 layer lines pass by default" "$out" "GO_noc=PASS count=1 layer_lines=12"
mklayers "$WORK/go11.err" 11
out="$(score GO_noc --out "$WORK/go.out" --err "$WORK/go11.err" --go-noc-layers 12)"
expect "GO_noc --go-noc-layers 12 and 11 layer lines fail" "$out" "GO_noc=FAIL(layer_lines=11!=12)"
out="$(score GO_noc --out "$WORK/go.out" --err "$WORK/empty.err")"
expect "GO_noc with no layer line passes: layer_lines is recorded, not gated" "$out" \
    "GO_noc=PASS count=1 layer_lines=0"
out="$(score GO_noc --out "$WORK/go.out" --err "$WORK/empty.err" --go-noc-layers any)"
expect "GO_noc --go-noc-layers any with no layer line fails" "$out" "GO_noc=FAIL(layer_lines=0<1)"

mkcsv "$WORK/gopp-ok.csv" 512:0:880.0:4.0:512 0:128:32.1:0.1:512
mkcsv "$WORK/gopp-low.csv" 512:0:770.0:4.0:512 0:128:32.1:0.1:512
mkcsv "$WORK/gopp-tg.csv" 512:0:880.0:4.0:512 0:128:30.0:0.1:512
out="$(score GO_pp --csv "$WORK/gopp-ok.csv")"
expect "GO_pp inside both windows passes" "$out" "GO_pp=PASS pp512=880.0"
out="$(score GO_pp --csv "$WORK/gopp-low.csv")"
expect "GO_pp pp512 770 fails low" "$out" "GO_pp=FAIL(pp512=770.0<827."
expect "GO_pp pp512 770 names llama.cpp-zuze (within 15 % below the reference)" "$out" "see=llama.cpp-zuze"
mkcsv "$WORK/gopp-745.csv" 512:0:745.0:4.0:512 0:128:32.1:0.1:512
mkcsv "$WORK/gopp-735.csv" 512:0:735.0:4.0:512 0:128:32.1:0.1:512
out="$(score GO_pp --csv "$WORK/gopp-745.csv")"
expect "GO_pp pp512 745 (within 15 % below 871) names zuze" "$out" "see=llama.cpp-zuze"
out="$(score GO_pp --csv "$WORK/gopp-735.csv")"
expect "GO_pp pp512 735 (more than 15 % below) fails without naming zuze" "$out" "GO_pp=FAIL(pp512=735.0<"
expect_not "GO_pp pp512 735 is not the zuze regression" "$out" "zuze"
out="$(score GO_pp --csv "$WORK/gopp-ok.csv")"
expect_not "GO_pp a pass does not name zuze" "$out" "zuze"
out="$(score GO_pp --csv "$WORK/gopp-tg.csv")"
expect "GO_pp tg128 30.0 fails low against 32.0 less 3 %" "$out" "GO_pp=FAIL(tg128=30.0<31.0"
mkcsv "$WORK/gopp-tg32.csv" 512:0:880.0:4.0:512 0:128:32.0:0.1:512
out="$(score GO_pp --csv "$WORK/gopp-tg32.csv")"
expect "GO_pp tg128 32.0 passes: the B50 reads about 32" "$out" "GO_pp=PASS pp512=880.0"
mkcsv "$WORK/gopp-tg31.csv" 512:0:880.0:4.0:512 0:128:31.1:0.1:512
out="$(score GO_pp --csv "$WORK/gopp-tg31.csv")"
expect "GO_pp tg128 31.1 passes (inside 3 %)" "$out" "GO_pp=PASS"
mkcsv "$WORK/gopp-tg30b.csv" 512:0:880.0:4.0:512 0:128:30.9:0.1:512
out="$(score GO_pp --csv "$WORK/gopp-tg30b.csv")"
expect "GO_pp tg128 30.9 fails (outside 3 %)" "$out" "GO_pp=FAIL(tg128=30.9<31.0"

printf '1, 2, 3, 4, 5, 6, 7, 8, 9, 10\n' > "$WORK/mi.out"
printf '1, 2, 3, 4, 5,\n' > "$WORK/mi-prompt.out"
out="$(score MI_b70 --out "$WORK/mi.out")"
expect "MI_b70 digits pass" "$out" "MI_b70=PASS digits=1"
out="$(score MI_b50 --out "$WORK/mi-prompt.out")"
expect "MI_b50 the prompt alone must not pass" "$out" "MI_b50=FAIL(digits=0!=1)"

echo 'xe 0000:03:00.0: [drm] GT0: suspended' > "$WORK/j-clean.txt"
printf 'xe 0000:03:00.0: [drm] GT0: suspended\nxe 0000:03:00.0: [drm] GT reset started\n' > "$WORK/j-reset.txt"
out="$(score resets --journal "$WORK/j-clean.txt")"
expect "resets clean journal" "$out" "resets=PASS count=0"
out="$(score resets --journal "$WORK/j-reset.txt")"
expect "resets a GT reset fails" "$out" "resets=FAIL(count=1)"

# ---- dry run and help
out="$(bash "$GATE" --dry-run 2>&1)"; rc=$?
expect_rc "dry-run rc" "$rc" 0
for arm in Q_pp Q_pp_cpu Q_staged Q_tok Q_tok_2 Q_tok_cpu_1 Q_tok_cpu Q_tg Q_noc \
           GO_gate GO_pp GO_noc MI_b70 MI_b50 resets; do
    expect "dry-run lists $arm" "$out" "$arm: "
done
n_sel="$(printf '%s\n' "$out" | grep -c 'ONEAPI_DEVICE_SELECTOR=level_zero:[01] ')"
expect_rc "dry-run pins the selector in all fourteen device arms" "$n_sel" 14
n_q900="$(printf '%s\n' "$out" | grep -E '^Q_[A-Za-z0-9_]+: ' | grep -c 'timeout -k 15 900 ')"
expect_rc "dry-run gives the seven short Qwen arms timeout 900" "$n_q900" 7
n_q1500="$(printf '%s\n' "$out" | grep -E '^Q_pp(_cpu)?: ' | grep -c 'timeout -k 15 1500 ')"
expect_rc "dry-run gives Q_pp and Q_pp_cpu timeout 1500 (run 1 took 11.5 min of 15 with the ring off)" "$n_q1500" 2
expect_not "dry-run passes no -ngl" "$out" " -ngl"
n_cnv="$(printf '%s\n' "$out" | grep -E '^GO_(gate|noc): ' | grep -cE '(^| )-cnv( |$)')"
expect_rc "dry-run passes no -cnv to llama-cli" "$n_cnv" 0
expect "dry-run states it runs nothing" "$out" "nothing is executed"
out="$(bash "$GATE" --help 2>&1)"; rc=$?
expect_rc "help rc" "$rc" 0
expect "help names the arms" "$out" "Q_staged"
expect "help names the end marker" "$out" "verdicts_pass=N GATE_DONE"
not_pass="$(grep -n 'say ".*GATE_DONE' "$GATE" | grep -vc 'verdicts_pass=')"
expect_rc "no print of GATE_DONE without verdicts_pass" "$not_pass" 0

# ---- refusals and the lock
out="$(bash "$GATE" --lock-dir "$WORK/GPU.lock" --arm resets 2>&1)"; rc=$?
expect_rc "no sha refuses with rc 2" "$rc" 2
expect "no sha says REFUSED" "$out" "REFUSED: no <sha> argument"
expect "a refusal ends with the marker line" "$out" "verdicts_pass=0 GATE_DONE"
if [ -e "$WORK/GPU.lock" ]; then fail "a refusal created GPU.lock"; else pass "a refusal never creates GPU.lock"; fi
out="$(bash "$GATE" --lock-dir "$WORK/GPU.lock" --arm resets 0000000000000000000000000000000000000000 2>&1)"; rc=$?
expect_rc "wrong sha refuses with rc 2" "$rc" 2
expect "wrong sha says REFUSED" "$out" "REFUSED: head "
if [ -e "$WORK/GPU.lock" ]; then
    fail "a wrong-sha refusal created GPU.lock"
else
    pass "a wrong-sha refusal never creates GPU.lock"
fi
for badsha in abc abcdef xyz1234 0000000G; do
    out="$(bash "$GATE" --lock-dir "$WORK/GPU.lock" --arm resets "$badsha" 2>&1)"; rc=$?
    expect_rc "a <sha> of '$badsha' (under 7 hex characters or not hex) refuses with rc 2" "$rc" 2
    expect "a <sha> of '$badsha' says why" "$out" "is not 7 to 40 lowercase hex characters"
done
out="$(bash "$GATE" --lock-dir "$WORK/GPU.lock" --lock-selftest 2>&1)"; rc=$?
expect_rc "lock selftest rc" "$rc" 0
expect "lock selftest ok" "$out" "lock-selftest: ok"
out="$(bash "$GATE" --arm Q_nope 2>&1)"; rc=$?
expect_rc "an unknown arm is a usage error" "$rc" 2

# ---- the summary: ring_engaged_any and the exit code, scored from real verdict lines
PASS_RING="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-pass.err")"
PASS_NORING="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-refused.err")"
FAIL_LINE="$(score Q_pp --csv "$WORK/pp-fail.csv" --err "$WORK/pp-fail.err")"
CTRL_PASS="GO_gate=PASS count=1 layer_lines=0"
summ() { # <name> <verdict line>...: writes them to a file and runs --score-summary
    local f="$WORK/summ-$1.txt"
    shift
    printf '%s\n' "$@" > "$f"
    bash "$GATE" --score-summary "$f" 2>&1
}
out="$(summ ring "$PASS_RING" "$CTRL_PASS")"; rc=$?
expect "summary ring engaged: ring_engaged_any=yes" "$out" "ring_engaged_any=yes"
expect_rc "summary ring engaged rc" "$rc" 0
expect "summary ends with the marker" "$out" "verdicts_pass=2 GATE_DONE"
out="$(summ noring "$PASS_NORING" "$CTRL_PASS")"; rc=$?
expect "summary Q_pp PASS without the ring: ring_engaged_any=no" "$out" "ring_engaged_any=no"
expect "summary Q_pp PASS without the ring still counts as a pass" "$out" "verdicts_pass=2 GATE_DONE"
expect_rc "summary nothing failed but no ring engaged is exit 3" "$rc" 3
out="$(summ mixed "$PASS_NORING" "$PASS_RING")"; rc=$?
expect "summary one ring-engaged arm makes ring_engaged_any=yes" "$out" "ring_engaged_any=yes"
expect_rc "summary mixed rc" "$rc" 0
out="$(summ na "$CTRL_PASS" "MI_b70=PASS digits=1")"; rc=$?
expect "summary without a ring-scoring arm: ring_engaged_any=na" "$out" "ring_engaged_any=na"
expect_rc "summary without a ring-scoring arm rc" "$rc" 0
out="$(summ fail "$FAIL_LINE" "$CTRL_PASS")"; rc=$?
expect_rc "summary a FAIL is exit 1 even without the ring" "$rc" 1
out="$(summ loan "$PASS_RING" "$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-loan.err")")"
expect "summary loan_exercised_any=yes from a granted loan" "$out" "loan_exercised_any=yes"
out="$(summ noloan "$PASS_RING")"
expect "summary loan_exercised_any=no without one" "$out" "loan_exercised_any=no"
out="$(summ notx "$PASS_RING" "Q_staged=NOT_EXERCISED(ring_engaged=no) rc=0")"; rc=$?
expect_rc "summary a NOT_EXERCISED arm is exit 3" "$rc" 3
out="$(bash "$GATE" --score-summary "$WORK/absent.txt" 2>&1)"; rc=$?
expect_rc "summary of a missing file is a usage error" "$rc" 2
out="$(bash "$GATE" --help 2>&1)"
expect "help documents exit 143 on a signal" "$out" "143"
expect "help documents ring_engaged_any" "$out" "ring_engaged_any=yes|no|na"

# ---- the oneAPI variable collision (found by the first real run, 2026-10-23)
# setvars.sh sources the advisor and vtune vars.sh files, which assign BIN_DIR=bin64 as a plain variable. A gate that
# set BIN_DIR and then sourced setvars looked for bin64/llama-bench. --dry-run never sources setvars, so only this
# static check and the real-flow refusal below can see it.
if cat "$GATE" | grep -vE '^[[:space:]]*#' | grep -qE '(^|[^A-Za-z_])BIN_DIR='; then
    fail "the gate assigns a bare BIN_DIR, which oneAPI's vars.sh overwrites"
else
    pass "the gate assigns no bare BIN_DIR"
fi
src_line="$(cat "$GATE" | grep -vE '^[[:space:]]*#' | grep -n 'source /opt/intel/oneapi/setvars.sh' | head -1 \
    | cut -d: -f1)"
use_line="$(cat "$GATE" | grep -vE '^[[:space:]]*#' | grep -n 'GATE_BIN_DIR' | head -1 | cut -d: -f1)"
if [ -n "$src_line" ] && [ -n "$use_line" ] && [ "$src_line" -lt "$use_line" ]; then
    pass "setvars is sourced (line $src_line) before the first use of GATE_BIN_DIR (line $use_line)"
else
    fail "setvars source line '${src_line:-none}' must precede the first GATE_BIN_DIR line '${use_line:-none}'"
fi
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    for f in /opt/intel/oneapi/setvars.sh /opt/intel/oneapi/*/latest/env/vars.sh \
             /opt/intel/oneapi/*/latest/oneapi-vars.sh \
             /opt/intel/oneapi/*/latest/etc/*/vars.sh /opt/intel/oneapi/*/latest/etc/*/*.sh; do
        [ -f "$f" ] && cat "$f"
    done 2>/dev/null | grep -oE '^[[:space:]]*(export +|local +|declare +)?[A-Za-z_][A-Za-z_0-9]*=' \
        | sed -E 's/^[[:space:]]*(export |local |declare )? *//; s/=$//' | LC_ALL=C sort -u > "$WORK/oneapi-names.txt"
    cat "$GATE" | grep -vE '^[[:space:]]*#' \
        | grep -oE '(^|[;{(]|then|do)[[:space:]]*[A-Za-z_][A-Za-z_0-9]*(\[[^]]*\])?\+?=' \
        | sed -E 's/^(;|\{|\(|then|do)?[[:space:]]*//; s/(\[[^]]*\])?\+?=$//' \
        | LC_ALL=C sort -u > "$WORK/gate-names.txt"
    shared="$(LC_ALL=C comm -12 "$WORK/oneapi-names.txt" "$WORK/gate-names.txt" | tr '\n' ' ')"
    if [ -s "$WORK/oneapi-names.txt" ] && [ -s "$WORK/gate-names.txt" ] && [ -z "$shared" ]; then
        pass "no variable the gate assigns is also assigned by oneAPI's setvars and vars.sh files"
    else
        fail "the gate assigns names oneAPI also assigns (or an extraction was empty): '${shared}'"
    fi
else
    pass "oneAPI is not installed here: the name audit is skipped (the static BIN_DIR checks above still ran)"
fi
# The real flow, up to the first refusal and no further: a build dir that lacks the binaries. The refusal must name
# the build's own bin directory; with the old BIN_DIR it named bin64 once setvars had run.
mkdir -p "$WORK/fakebuild"
echo 'GGML_SYCL:BOOL=ON' > "$WORK/fakebuild/CMakeCache.txt"
out="$(bash "$GATE" --no-head-check --no-lock --arm Q_pp --build-dir "$WORK/fakebuild" --out-dir "$WORK/fakeout" \
    abcdef0 2>&1)"; rc=$?
expect_rc "a real run without binaries refuses with rc 2" "$rc" 2
expect "the refusal names the build's own bin directory" "$out" "REFUSED: no $WORK/fakebuild/bin/llama-bench"
expect_not "the refusal never names bin64" "$out" "bin64"

# ---- the exit-143 path: no device, a sleeping child, a TERM
LOCKD="$WORK/sig.lock"
bash "$GATE" --signal-selftest --lock-dir "$LOCKD" > "$WORK/sig.out" 2>&1 &
gpid=$!
for _ in $(seq 100); do
    grep -q 'signal-selftest: child=' "$WORK/sig.out" 2>/dev/null && break
    sleep 0.1
done
child="$(sed -n 's/.*signal-selftest: child=\([0-9][0-9]*\) .*/\1/p' "$WORK/sig.out")"
if [ -d "$LOCKD" ] && [ -f "$LOCKD/holder" ] && [ -n "$child" ] && kill -0 "$child" 2>/dev/null; then
    pass "signal selftest: the lock is held with a holder file and the child runs before the TERM"
else
    fail "signal selftest did not reach its waiting state (lock '$LOCKD', child '${child:-none}')" \
        "$(cat "$WORK/sig.out" 2>/dev/null)"
fi
kill -TERM "$gpid" 2>/dev/null
wait "$gpid"; rc=$?
out="$(cat "$WORK/sig.out")"
expect_rc "a TERM ends the gate with rc 143" "$rc" 143
expect "a TERM prints ABORTED" "$out" "ABORTED by signal"
expect "a TERM still ends with the marker line" "$out" "verdicts_pass=0 GATE_DONE"
if [ -e "$LOCKD" ]; then fail "a TERM left GPU.lock behind"; else pass "a TERM releases GPU.lock"; fi
if [ -n "$child" ] && kill -0 "$child" 2>/dev/null; then
    fail "a TERM left the child $child running"
else
    pass "a TERM kills the child"
fi

# ---- RING_ARMS: the arms ring_engaged_any is read from
read -ra RARMS <<< "$(bash "$GATE" --ring-arms 2>&1)"
if [ "${#RARMS[@]}" -gt 0 ]; then
    pass "the gate names its ring arms: ${RARMS[*]}"
else
    fail "--ring-arms printed nothing"
fi
for arm in "${RARMS[@]}"; do
    case "$arm" in
        Q_pp) line="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-pass.err")" ;;
        Q_staged) line="$(score Q_staged --err "$WORK/st-pass.err")" ;;
        *) line="" ;;
    esac
    if [ -z "$line" ]; then
        fail "ring arm $arm has no fixture in this test: add one"
    else
        expect "ring arm $arm prints ring_engaged=" "$line" "ring_engaged="
    fi
done
out="$(summ outside "$PASS_RING" "Q_tg=PASS tg=11.9 ring_engaged=no")"
expect "an arm outside RING_ARMS cannot move ring_engaged_any to no" "$out" "ring_engaged_any=yes"
out="$(summ outside2 "$CTRL_PASS" "Q_tg=PASS tg=11.9 ring_engaged=no")"; rc=$?
expect "an arm outside RING_ARMS does not make ring_engaged_any=no" "$out" "ring_engaged_any=na"
expect_rc "an arm outside RING_ARMS leaves the exit code 0" "$rc" 0

# ---- the arms fail closed ---------------------------------------------------------------------

# the resets arm reads journalctl's own rc and a missing journal; an empty journal is a pass only with rc 0
out="$(score resets --journal "$WORK/absent-journal.txt")"; rc=$?
expect "resets a missing journal file is a FAIL, never a clean log" "$out" "resets=FAIL(no_journal"
expect_rc "resets a missing journal rc" "$rc" 1
: > "$WORK/j-empty.txt"
out="$(score resets --journal "$WORK/j-empty.txt")"
expect "resets an empty journal with journalctl rc 0 is a legitimate PASS" "$out" "resets=PASS count=0"
out="$(score resets --journal "$WORK/j-empty.txt" --rc 1)"; rc=$?
expect "resets an empty journal with journalctl rc 1 is a FAIL" "$out" "resets=FAIL(journalctl_rc=1)"
expect_rc "resets journalctl failure rc" "$rc" 1
out="$(score resets --journal "$WORK/j-clean.txt" --rc 7)"
expect "resets a clean journal with journalctl rc 7 is a FAIL naming the rc" "$out" "resets=FAIL(journalctl_rc=7)"
printf 'xe 0000:03:00.0: [drm] gt reset started\n' > "$WORK/j-lower.txt"
out="$(score resets --journal "$WORK/j-lower.txt")"
expect "resets matches case-insensitively" "$out" "resets=FAIL(count=1)"

# Q_tok_cpu needs all four outputs non-empty and all four loads rc 0; a recorded arm that failed is a FAIL
: > "$WORK/tok-none.out"
out="$(score Q_tok_cpu --out "$WORK/tok-c2.out" --out2 "$WORK/tok-none.out" --ref-out "$WORK/tok-none.out" \
    --ref-out2 "$WORK/tok-none.out" --rc-out2 134 --rc-ref 134 --rc-ref2 134)"; rc=$?
expect "Q_tok_cpu three crashed loads and one good cpu load FAIL" "$out" "Q_tok_cpu=FAIL("
expect_not "Q_tok_cpu three crashed loads never PASS" "$out" "Q_tok_cpu=PASS"
expect "Q_tok_cpu names the crashed Q_tok load" "$out" "rc_Q_tok=134"
expect "Q_tok_cpu names the empty Q_tok output" "$out" "empty_Q_tok_output"
expect_rc "Q_tok_cpu three crashed loads rc" "$rc" 1
out="$(score Q_tok_cpu --out "$WORK/tok-c2.out" --out2 "$WORK/tok-c1.out" --ref-out "$WORK/tok-a1.out" \
    --ref-out2 "$WORK/tok-a2.out" --rc-ref 134)"
expect "Q_tok_cpu equal md5s with one auto load at rc 134 still FAIL" "$out" "Q_tok_cpu=FAIL(rc_Q_tok=134)"
out="$(score Q_tok_cpu --out "$WORK/tok-c2.out" --out2 "$WORK/tok-c1.out" --ref-out "$WORK/tok-a1.out" \
    --ref-out2 "$WORK/tok-a2.out" --rc-out2 137)"
expect "Q_tok_cpu a killed first cpu load FAILs" "$out" "Q_tok_cpu=FAIL(rc_Q_tok_cpu_1=137)"
out="$(score Q_tok --out "$WORK/tok-a1.out" --rc 134)"
expect "Q_tok a crashed load is a FAIL, not a recorded md5" "$out" "Q_tok=FAIL(rc=134)"
out="$(score Q_tok_2 --out "$WORK/tok-none.out")"
expect "Q_tok_2 an empty stdout is a FAIL, not md5 d41d8cd9" "$out" "Q_tok_2=FAIL(empty_output)"
out="$(score Q_pp_cpu --csv "$WORK/pp-slow.csv" --err "$WORK/pp-refused.err" --rc 124)"
expect "Q_pp_cpu a timeout is a FAIL" "$out" "Q_pp_cpu=FAIL(rc=124)"
out="$(summ rec134 "Q_tok=RECORDED md5=d41d8cd9 rc=134")"; rc=$?
expect_rc "summary a RECORDED line with rc 134 is exit 1" "$rc" 1
out="$(summ rec0 "Q_tok=RECORDED md5=c4589dac rc=0")"; rc=$?
expect_rc "summary a RECORDED line with rc 0 is neutral" "$rc" 0
expect "summary a RECORDED line is not counted as a pass" "$out" "verdicts_pass=0 GATE_DONE"
out="$(summ recpass "Q_tok=RECORDED md5=c4589dac rc=0" "$CTRL_PASS")"
expect "summary RECORDED beside one PASS counts one pass" "$out" "verdicts_pass=1 GATE_DONE"

# a usage error ends with the final line so a detached waiter is released; help and dry-run keep the plain form
out="$(bash "$GATE" --no-such-option 2>&1)"; rc=$?
expect_rc "an unknown option exits 2" "$rc" 2
expect "an unknown option still prints the final line" "$out" "verdicts_pass=0 GATE_DONE"
expect "an unknown option says REFUSED: usage" "$out" "REFUSED: usage: unknown option: --no-such-option"
# a value-taking option given last must be refused, not spun on: `shift 2` fails on one remaining word, so without the
# guard the loop re-reads the option forever. Every value-taking option, spelled out below and not read back from the
# script (a list read from the script would shrink with a removed guard), is tried as the final word under a timeout.
out="$(timeout 10 bash "$GATE" --dry-run --out-dir 2>&1)"; rc=$?
expect_rc "a trailing --out-dir exits 2, not 124" "$rc" 2
expect "a trailing --out-dir says REFUSED: usage ... needs a value" "$out" "REFUSED: usage: --out-dir needs a value"
expect "a trailing --out-dir prints the final line" "$out" "verdicts_pass=0 GATE_DONE"
out="$(timeout 10 bash "$GATE" --since 2>&1)"; rc=$?
expect_rc "a trailing --since exits 2, not 124" "$rc" 2
n_opts=0
bad_opts=""
VALUE_OPTS="--prompt-out --csv --out --out2 --err --ref-out --ref-out2 --journal --rc --rc-out2 --rc-ref --rc-ref2
--builders-pre --builders-post --src-dir --seed-verdicts --builders-file --builders-wait-max --builders-poll
--score-summary --build-dir --out-dir --lock-dir --since --qwen --gptoss --mistral --tuning-dir --decode-ref
--place-ref-tg --place-ref-noc --qmd5 --q-layers --go-noc-layers --pp512-min --pp2048-min --go-pp-ref --go-tg-ref
--arm --score-arm --builders-post-selftest"
for opt in $VALUE_OPTS; do
    n_opts=$((n_opts + 1))
    out="$(timeout 10 bash "$GATE" "$opt" 2>&1)"; rc=$?
    case "$opt" in
        --arm | --score-arm | --builders-post-selftest) want="REFUSED: usage: $opt '' is not one of" ;;
        *) want="REFUSED: usage: $opt needs a value" ;;
    esac
    [ "$rc" = 2 ] && printf '%s\n' "$out" | grep -qF -- "$want" || bad_opts="$bad_opts $opt(rc=$rc)"
done
expect_rc "no guarded option is left without a refusal when it is the last word:$bad_opts" "${bad_opts:+bad}" ""
expect_rc "the loop above covered all 41 value-taking options" "$n_opts" 41
# every `shift 2` arm in the script carries the guard, except the three whose is_arm check refuses an empty value
# itself, so a guard removed from an option is caught even if the list above forgot the option
unguarded="$(cat "$GATE" | grep -n 'shift 2 ;;' | grep -v 'need_value' | cut -d: -f2- \
    | grep -c -v -E 'ARMS\+=|SCORE_ARM=|BUILDERS_POST_SELFTEST=')"
expect_rc "every shift 2 arm in the option loop has a need_value guard" "$unguarded" 0
n_shift2="$(cat "$GATE" | grep -c 'shift 2 ;;')"
expect_rc "the option loop has 41 shift 2 arms (a new one must join the list above)" \
    "$n_shift2" 41
out="$(bash "$GATE" --arm Q_nope 2>&1)"
expect "an unknown arm prints the final line" "$out" "verdicts_pass=0 GATE_DONE"
out="$(bash "$GATE" abcdef0 1234567 2>&1)"; rc=$?
expect_rc "two <sha> arguments exit 2" "$rc" 2
expect "two <sha> arguments print the final line" "$out" "verdicts_pass=0 GATE_DONE"
out="$(bash "$GATE" --help 2>&1)"
expect_not "help is not a usage refusal" "$out" "REFUSED: usage"

# the sha format is checked under --no-head-check; the head check reads the build's own source tree
mkdir -p "$WORK/stubbin"
mkstub() { printf '#!/bin/sh\n%s\n' "$1" > "$WORK/stubbin/journalctl"; chmod +x "$WORK/stubbin/journalctl"; }
# rf <build-dir> <gate option>... <sha>: a real run of the journal arm alone (no device, no model, no lock)
rf() { PATH="$WORK/stubbin:$PATH" bash "${GATE_UNDER:-$GATE}" --no-lock --arm resets --build-dir "$@" 2>&1; }
mkstub 'exit 0'
out="$(rf "$WORK/fakebuild" --no-head-check --out-dir "$WORK/rf-badsha" xyz1234)"; rc=$?
expect_rc "--no-head-check still validates the sha format" "$rc" 2
expect "--no-head-check with a bad sha says why" "$out" "is not 7 to 40 lowercase hex characters"
out="$(rf "$WORK/fakebuild" --no-head-check --out-dir "$WORK/rf-nosha")"; rc=$?
expect_rc "--no-head-check still requires a <sha>" "$rc" 2
expect "--no-head-check without a sha says so" "$out" "REFUSED: no <sha> argument"
git init -q "$WORK/srcrepo"
git -C "$WORK/srcrepo" -c user.name=gate-test -c user.email=gate-test@example.invalid \
    commit -q --allow-empty -m "the tree under test"
SRC_SHA="$(git -C "$WORK/srcrepo" rev-parse HEAD)"
mkdir -p "$WORK/srcrepo/build"
echo 'GGML_SYCL:BOOL=ON' > "$WORK/srcrepo/build/CMakeCache.txt"
out="$(rf "$WORK/srcrepo/build" --out-dir "$WORK/rf-src-ok" "$SRC_SHA")"; rc=$?
expect "the head check compares the sha with the build's own source tree" "$out" "resets=PASS"
expect_rc "the head check against the build's source tree passes" "$rc" 0
out="$(rf "$WORK/srcrepo/build" --out-dir "$WORK/rf-src-short" "${SRC_SHA:0:7}")"
expect "a 7-character sha prefix of the build's source head is accepted" "$out" "resets=PASS"
out="$(rf "$WORK/srcrepo/build" --out-dir "$WORK/rf-src-bad" 0000000000000000000000000000000000000000)"; rc=$?
expect_rc "a sha that is not the build's source head refuses" "$rc" 2
expect "the head refusal names the source tree it read" "$out" "of $WORK/srcrepo does not start with"
out="$(rf "$WORK/fakebuild" --src-dir "$WORK/srcrepo" --out-dir "$WORK/rf-srcdir" "$SRC_SHA")"
expect "--src-dir names the tree the head check reads" "$out" "resets=PASS"
out="$(rf "$WORK/srcrepo/build" --no-head-check --out-dir "$WORK/rf-nohead" 0000000)"; rc=$?
expect_rc "--no-head-check is the explicit escape: the wrong head passes" "$rc" 0
mkdir -p "$WORK/snap/scripts"
cp -a "$ROOT_DIR/scripts/sycl-moe-exec-gate.sh" "$ROOT_DIR/scripts/sycl-moe-exec-prompts" \
    "$ROOT_DIR/scripts/qwen4exp-mtp-prompts" "$WORK/snap/scripts/"
SNAP="$WORK/snap/scripts/sycl-moe-exec-gate.sh"
out="$(GATE_UNDER="$SNAP" rf "$WORK/srcrepo/build" --out-dir "$WORK/rf-snap" "$SRC_SHA")"; rc=$?
expect "the lead form: a snapshot copy of the script runs against another tree's build" "$out" "resets=PASS"
expect_rc "the snapshot run rc" "$rc" 0
out="$(bash "$SNAP" --dry-run 2>&1)"
expect "snapshot prompts resolve next to the script (code-2k)" "$out" \
    "-f $WORK/snap/scripts/sycl-moe-exec-prompts/code-2k.txt"
expect "snapshot prompts resolve next to the script (decode)" "$out" \
    "-f $WORK/snap/scripts/qwen4exp-mtp-prompts/code.txt"
out="$(bash "$SNAP" --dry-run --build-dir "$WORK/srcrepo/build" 2>&1)"
expect "the dry-run names the source tree the build's head is read from" "$out" "src-dir: $WORK/srcrepo "

# in the real flow journalctl's rc survives into the verdict, and the verdict line carries the builders fields
mkstub 'exit 7'
out="$(rf "$WORK/fakebuild" --no-head-check --out-dir "$WORK/rf-j7" abcdef0)"; rc=$?
expect "a failing journalctl scores resets=FAIL naming its rc" "$out" "resets=FAIL(journalctl_rc=7)"
expect_rc "a failing journalctl is exit 1" "$rc" 1
mkstub 'echo "xe 0000:03:00.0: [drm] GT reset started"'
out="$(rf "$WORK/fakebuild" --no-head-check --out-dir "$WORK/rf-reset" abcdef0)"; rc=$?
expect "a GT reset in the journal scores resets=FAIL" "$out" "resets=FAIL(count=1)"
expect_rc "a GT reset is exit 1" "$rc" 1
mkstub 'exit 0'
out="$(rf "$WORK/fakebuild" --no-head-check --out-dir "$WORK/rf-clean" abcdef0)"; rc=$?
expect "an empty journal with rc 0 passes" "$out" "resets=PASS count=0 builders_pre=na builders_post=na"
expect "the run ends with the final line" "$out" "verdicts_pass=1 GATE_DONE"
expect_rc "a clean journal run is exit 0" "$rc" 0

# a real run sources oneAPI, and an option VALUE that equals a mode flag must not skip it
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    out="$(rf "$WORK/fakebuild" --no-head-check --out-dir "$WORK/rf-oneapi" abcdef0)"
    expect "a real run sources oneAPI" "$out" "oneapi=sourced"
    out="$(rf "$WORK/fakebuild" --no-head-check --since --dry-run --out-dir "$WORK/rf-oneapi2" abcdef0)"
    expect "an option value equal to a mode flag does not skip setvars" "$out" "oneapi=sourced"
    expect "an option value equal to a mode flag does not turn the run into a dry run" "$out" "resets=PASS"
else
    pass "oneAPI is not installed here: the setvars-sourced checks are skipped"
fi
out="$(bash "$GATE" --score-arm resets --journal "$WORK/j-clean.txt" 2>&1)"
expect_not "a scoring mode does not source oneAPI" "$out" "oneapi="

# an exported GGML_SYCL_* or LLAMA_ARG_* variable would reach every arm: refuse and name it
out="$(GGML_SYCL_MOE_EXEC=cpu rf "$WORK/fakebuild" --no-head-check --out-dir "$WORK/rf-leak" abcdef0)"; rc=$?
expect_rc "an exported GGML_SYCL_ variable refuses with rc 2" "$rc" 2
expect "the refusal names the variable" "$out" "GGML_SYCL_MOE_EXEC"
out="$(LLAMA_ARG_N_GPU_LAYERS=99 rf "$WORK/fakebuild" --no-head-check --out-dir "$WORK/rf-leak2" abcdef0)"; rc=$?
expect_rc "an exported LLAMA_ARG_ variable refuses with rc 2" "$rc" 2
expect "the refusal names the LLAMA_ARG_ variable" "$out" "LLAMA_ARG_N_GPU_LAYERS"
expect "a leak refusal ends with the final line" "$out" "verdicts_pass=0 GATE_DONE"

# a reused --out-dir: stale outputs must not be scored as this run's, so it is refused unless --arm names what to redo
# or --clobber says to overwrite
mkdir -p "$WORK/rf-used"
echo stale > "$WORK/rf-used/Q_pp.out"
out="$(PATH="$WORK/stubbin:$PATH" bash "$GATE" --no-lock --build-dir "$WORK/fakebuild" --no-head-check \
    --out-dir "$WORK/rf-used" abcdef0 2>&1)"; rc=$?
expect_rc "a non-empty --out-dir without --arm or --clobber refuses with rc 2" "$rc" 2
expect "the out-dir refusal says why" "$out" "is not empty: use a fresh one, --arm to redo named arms, or --clobber"
expect "the stale file is untouched by the refusal" "$(cat "$WORK/rf-used/Q_pp.out")" "stale"
out="$(rf "$WORK/fakebuild" --no-head-check --out-dir "$WORK/rf-used" abcdef0)"; rc=$?
expect "--arm lets a non-empty --out-dir be reused (only that arm's files are written)" "$out" "resets=PASS"
expect "the other arms' files in the reused out-dir are untouched" "$(cat "$WORK/rf-used/Q_pp.out")" "stale"
echo stale > "$WORK/rf-used2.marker"
mkdir -p "$WORK/rf-used2"; echo stale > "$WORK/rf-used2/Q_pp.out"
out="$(PATH="$WORK/stubbin:$PATH" bash "$GATE" --no-lock --build-dir "$WORK/fakebuild" --no-head-check --clobber \
    --out-dir "$WORK/rf-used2" abcdef0 2>&1)"
expect_not "--clobber accepts a non-empty --out-dir (it gets past the out-dir check)" "$out" "is not empty"

# builders_pre and builders_post on every verdict line; a build during the arm taints a PASS
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-pass.err" --builders-pre 0 --builders-post 0)"; rc=$?
expect "a verdict line carries builders_pre and builders_post" "$out" "builders_pre=0 builders_post=0"
expect "builders 0 and 0 leaves the PASS" "$out" "Q_pp=PASS"
expect_not "builders 0 and 0 is not tainted" "$out" "tainted"
expect_rc "builders 0 and 0 rc" "$rc" 0
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-pass.err" --builders-pre 0 --builders-post 2)"; rc=$?
expect "a PASS with builders_post 2 becomes NOT_SCORED(tainted=builders,was=PASS)" "$out" \
    "Q_pp=NOT_SCORED(tainted=builders,was=PASS)"
expect "a tainted verdict carries the flag" "$out" "builders_pre=0 builders_post=2 tainted=builders"
expect_rc "a tainted PASS is exit class 3" "$rc" 3
out="$(score Q_pp --csv "$WORK/pp-fail.csv" --err "$WORK/pp-fail.err" --builders-pre 0 --builders-post 3)"; rc=$?
expect "a FAIL with builders_post 3 stays a FAIL" "$out" "Q_pp=FAIL(pp512=412<500)"
expect "a tainted FAIL carries the flag" "$out" "tainted=builders"
expect_rc "a tainted FAIL is exit 1" "$rc" 1
out="$(score MI_b70 --out "$WORK/mi.out" --builders-pre 1 --builders-post 0)"
expect "builders_pre above 0 with builders_post 0 does not taint" "$out" \
    "MI_b70=PASS digits=1 builders_pre=1 builders_post=0"
TAINTED="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-pass.err" --builders-pre 0 --builders-post 2)"
out="$(summ taint "$TAINTED" "$CTRL_PASS")"; rc=$?
expect_rc "summary a tainted arm makes exit 3" "$rc" 3
expect "summary a tainted PASS is not counted as a pass" "$out" "verdicts_pass=1 GATE_DONE"

# every stop once the lock is held prints the verdicts scored so far; a builder wait is bounded like settle
printf '%s\n' "$PASS_RING" "$CTRL_PASS" > "$WORK/seed.txt"
out="$(bash "$GATE" --refuse-selftest --seed-verdicts "$WORK/seed.txt" --lock-dir "$WORK/ref.lock" 2>&1)"; rc=$?
expect_rc "a refusal after arms ran exits 2" "$rc" 2
expect "a refusal after arms ran prints STOPPED with the reason" "$out" "STOPPED: synthetic stop (2 of 15 arms scored)"
expect "a refusal after arms ran prints the summary over the scored verdicts" "$out" "summary: ring_engaged_any=yes"
expect "a refusal after arms ran keeps the scored passes in the final line" "$out" "verdicts_pass=2 GATE_DONE"
expect_not "a refusal after arms ran is not a bare REFUSED" "$out" "REFUSED"
if [ -e "$WORK/ref.lock" ]; then
    fail "a stop left GPU.lock behind"
else
    pass "a stop after arms ran releases GPU.lock"
fi
out="$(bash "$GATE" --refuse-selftest --lock-dir "$WORK/ref.lock" 2>&1)"; rc=$?
expect "a refusal before any arm keeps the plain REFUSED form" "$out" "REFUSED: synthetic stop"
expect "a refusal before any arm ends with verdicts_pass=0" "$out" "verdicts_pass=0 GATE_DONE"
LOCKS="$WORK/sig2.lock"
bash "$GATE" --signal-selftest --seed-verdicts "$WORK/seed.txt" --lock-dir "$LOCKS" > "$WORK/sig2.out" 2>&1 &
gpid=$!
for _ in $(seq 100); do
    grep -q 'signal-selftest: child=' "$WORK/sig2.out" 2>/dev/null && break
    sleep 0.1
done
kill -TERM "$gpid" 2>/dev/null
wait "$gpid"; rc=$?
out="$(cat "$WORK/sig2.out")"
expect_rc "a TERM after arms ran exits 143" "$rc" 143
expect "a TERM after arms ran prints STOPPED: signal" "$out" "STOPPED: signal (2 of 15 arms scored)"
expect "a TERM after arms ran prints the summary over the scored verdicts" "$out" "summary: ring_engaged_any=yes"
expect "a TERM after arms ran keeps the scored passes" "$out" "verdicts_pass=2 GATE_DONE"
if [ -e "$LOCKS" ]; then
    fail "a TERM after arms ran left GPU.lock behind"
else
    pass "a TERM after arms ran releases GPU.lock"
fi
echo 2 > "$WORK/b-count"
out="$(bash "$GATE" --wait-builders-selftest --builders-file "$WORK/b-count" --builders-wait-max 3 \
    --builders-poll 1 2>&1)"; rc=$?
expect_rc "a builder count that never clears refuses after the bound with rc 2" "$rc" 2
expect "the expired wait names the count and the wait" "$out" "REFUSED: builders = 2 after waiting 3s"
expect "the expired wait ends with the final line" "$out" "verdicts_pass=0 GATE_DONE"
( sleep 1.5; echo 0 > "$WORK/b-count" ) &
bpid=$!
out="$(bash "$GATE" --wait-builders-selftest --builders-file "$WORK/b-count" --builders-wait-max 20 \
    --builders-poll 1 2>&1)"; rc=$?
wait "$bpid"
expect_rc "a builder count that clears inside the bound continues with rc 0" "$rc" 0
expect "the wait says it waited" "$out" "waiting up to 20s"
expect "the wait says it cleared" "$out" "builders cleared after"
expect_not "a cleared wait does not refuse" "$out" "REFUSED"
echo 0 > "$WORK/b-count"
out="$(bash "$GATE" --wait-builders-selftest --builders-file "$WORK/b-count" --builders-wait-max 3 \
    --builders-poll 1 2>&1)"; rc=$?
expect_rc "zero builders does not wait" "$rc" 0
expect_not "zero builders prints no wait line" "$out" "waiting up to"

# builders_post is the larger of the count after the arm and the highest count the sampler saw during it: a build that
# started and ended inside the arm taints it although none is running afterwards.
mkdir -p "$WORK/bp"
bpost() { # <during> <after>
    printf '%s\n' "$1" > "$WORK/bp/Q_pp.builders_max"
    printf '%s\n' "$2" > "$WORK/bp/b-after"
    bash "$GATE" --builders-post-selftest Q_pp --out-dir "$WORK/bp" --builders-file "$WORK/bp/b-after" 2>&1
}
expect "builders_post: a build during the arm only is counted" "$(bpost 2 0)" "post=2"
expect "builders_post: a build after the arm only is counted" "$(bpost 0 3)" "post=3"
expect "builders_post: the larger of the two wins" "$(bpost 1 4)" "post=4"
expect "builders_post: none is zero" "$(bpost 0 0)" "post=0"
rm -f "$WORK/bp/Q_pp.builders_max"
echo 1 > "$WORK/bp/b-after"
expect "builders_post: no sampler file reads as zero during" \
    "$(bash "$GATE" --builders-post-selftest Q_pp --out-dir "$WORK/bp" --builders-file "$WORK/bp/b-after" 2>&1)" \
    "post=1"
# the wait for a running build comes before the lock and the settle check: an arm list with a device arm must not
# take GPU.lock or sample the host while a build still runs
wb_line="$(cat "$GATE" | grep -n -F 'wait_builders start' | cut -d: -f1 | head -1)"
lk_line="$(cat "$GATE" | grep -n -F 'lock_take || refuse "could not take' | cut -d: -f1 | head -1)"
st_line="$(cat "$GATE" | grep -n -F 'host_settled || refuse' | cut -d: -f1 | head -1)"
if [ -n "$wb_line" ] && [ -n "$lk_line" ] && [ -n "$st_line" ] \
    && [ "$wb_line" -lt "$st_line" ] && [ "$st_line" -lt "$lk_line" ]; then
    pass "the start waits for builders before the settle check and before the lock"
else
    fail "wait_builders start ($wb_line) must precede host_settled ($st_line) and lock_take ($lk_line)"
fi

# every arm scores a non-zero rc as a FAIL; a GO run that prints the answer and aborts in teardown must FAIL
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg.err" --qmd5 "$MD5" --place-ref-tg "$PLACE_TG" --rc 134)"; rc=$?
expect "Q_tg rc 134 fails with the rc named" "$out" "Q_tg=FAIL(rc=134)"
expect_rc "Q_tg rc 134 exit" "$rc" 1
out="$(score Q_noc --out "$WORK/dec.out" --err "$WORK/noc.err" --qmd5 "$MD5" --place-ref-noc - --rc 124)"
expect "Q_noc a timeout fails with the rc named" "$out" "Q_noc=FAIL(rc=124)"
out="$(score GO_gate --out "$WORK/go.out" --err "$WORK/empty.err" --rc 134)"; rc=$?
expect "GO_gate prints the answer and aborts in teardown: FAIL" "$out" "GO_gate=FAIL(rc=134)"
expect_rc "GO_gate rc 134 exit" "$rc" 1
out="$(score GO_noc --out "$WORK/go.out" --err "$WORK/go12.err" --rc 134)"
expect "GO_noc rc 134 fails" "$out" "GO_noc=FAIL(rc=134)"
out="$(score GO_pp --csv "$WORK/gopp-ok.csv" --rc 137)"
expect "GO_pp rc 137 fails" "$out" "GO_pp=FAIL(rc=137)"
out="$(score MI_b70 --out "$WORK/mi.out" --rc 134)"
expect "MI_b70 prints the digits and aborts in teardown: FAIL" "$out" "MI_b70=FAIL(rc=134)"
out="$(score MI_b50 --out "$WORK/mi.out" --rc 124)"
expect "MI_b50 a timeout fails" "$out" "MI_b50=FAIL(rc=124)"
out="$(score Q_staged --err "$WORK/st-pass.err" --rc 134)"
expect "Q_staged rc 134 fails" "$out" "Q_staged=FAIL(rc=134"

# scorer details a wrong implementation would otherwise pass
# the last " eval time = " line wins, not the first (head vs tail)
{
    echo "0.01.601.984 I [PLACEMENT] $PLACE_TG (of 73728 total)"
    echo "1.40.000.000 I llama_perf_context_print:        eval time =    9999.00 ms /    30 runs" \
        "  (333.30 ms per token,     3.00 tokens per second)"
    echo "$PERF_EVAL_HEAD   (83.97 ms per token,    11.91 tokens per second)"
} > "$WORK/tg-two.err"
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg-two.err" --qmd5 "$MD5" --place-ref-tg "$PLACE_TG")"
expect "Q_tg reads the last eval line, not the first" "$out" "Q_tg=PASS tg=11.91"
# a nan eval line must not let the prompt-eval rate stand in for tg
{
    echo "0.01.601.984 I [PLACEMENT] $PLACE_TG (of 73728 total)"
    echo "1.40.000.000 I llama_perf_context_print: prompt eval time =     400.00 ms /   800 tokens" \
        "(    0.50 ms per token,  2000.00 tokens per second)"
    echo "1.42.962.387 I llama_perf_context_print:        eval time =       0.00 ms /     1 runs" \
        "  (  nan ms per token,   nan tokens per second)"
} > "$WORK/tg-nan.err"
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg-nan.err" --qmd5 "$MD5" --place-ref-tg "$PLACE_TG")"
expect "Q_tg a nan eval line is no_tg, never the prompt-eval rate" "$out" "Q_tg=FAIL(no_tg)"
expect_not "Q_tg a nan eval line does not report tg=2000" "$out" "tg=2000"
# Mistral digits: exactly one line
printf '1, 2, 3, 4, 5, 6, 7, 8, 9, 10\n1, 2, 3, 4, 5, 6, 7, 8, 9, 10\n' > "$WORK/mi-two.out"
out="$(score MI_b70 --out "$WORK/mi-two.out")"
expect "MI_b70 two digit lines fail: the gate is exactly 1" "$out" "MI_b70=FAIL(digits=2!=1)"
# GO_pp upper bounds
mkcsv "$WORK/gopp-high.csv" 512:0:1000.0:4.0:512 0:128:33.9:0.1:512
mkcsv "$WORK/gopp-tghigh.csv" 512:0:880.0:4.0:512 0:128:40.0:0.1:512
out="$(score GO_pp --csv "$WORK/gopp-high.csv")"
expect "GO_pp pp512 1000 fails high" "$out" "GO_pp=FAIL(pp512=1000.0>"
out="$(score GO_pp --csv "$WORK/gopp-tghigh.csv")"
expect "GO_pp tg128 40.0 fails high" "$out" "GO_pp=FAIL(tg128=40.0>"
# refill is checked per device line: 24/23 on one device and 23/24 on another sum to a balanced 47/47
dec_err "$WORK/tg-twodev.err" 11.91 "$PLACE_TG" '0.09.000.000 W [MOE-EXEC] refill device 0: 24/23 triplets re-admitted'
echo '0.09.100.000 W [MOE-EXEC] refill device 1: 23/24 triplets re-admitted' >> "$WORK/tg-twodev.err"
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg-twodev.err" --qmd5 "$MD5" --place-ref-tg "$PLACE_TG")"
expect "Q_tg a two-device refill that sums to balanced is still unbalanced" "$out" "Q_tg=FAIL(refill=unbalanced(47/47))"
out="$(score Q_noc --out "$WORK/dec.out" --err "$WORK/tg-twodev.err" --qmd5 "$MD5" --place-ref-noc "$PLACE_TG")"
expect "Q_noc applies the same refill rule" "$out" "Q_noc=FAIL(refill=unbalanced(47/47))"
# a llama-bench depth row never stands in for the n_depth 0 row, wherever it sits in the csv
mkcsv "$WORK/pp-depth-first.csv" 512:0:99.0:1.0:512:4096 512:0:523.4:3.1:512 2048:0:851.2:5.0:2048
mkcsv "$WORK/pp-depth-last.csv" 512:0:523.4:3.1:512 2048:0:851.2:5.0:2048 512:0:99.0:1.0:512:4096
for f in pp-depth-first pp-depth-last; do
    out="$(score Q_pp --csv "$WORK/$f.csv" --err "$WORK/pp-pass.err")"
    expect "Q_pp ignores a depth row ($f)" "$out" "Q_pp=PASS pp512=523.4"
done

# the C-locale cmp spelling, and a missing tool is a skip in the gate too
mkdir -p "$WORK/nocmp"
for t in dirname cat grep sed tr cut head tail wc sort date ls env awk md5sum comm git pgrep timeout journalctl \
         mkdir rm cp stat seq sleep uname basename readlink mktemp; do
    tp="$(command -v "$t" 2>/dev/null)" && ln -sf "$tp" "$WORK/nocmp/$t"
done
out="$(PATH="$WORK/nocmp" "$BASH" "$GATE" --score-arm resets --journal "$WORK/j-clean.txt" 2>&1)"; rc=$?
expect_rc "the gate without cmp on PATH exits 77, never a verdict" "$rc" 77
expect "the gate names the missing tool" "$out" "missing tool(s): cmp"
expect_not "the gate without cmp prints no verdict" "$out" "resets="

# ---- the prompt
bash "$GATE" --make-prompt --prompt-out "$WORK/p1.txt" > /dev/null 2>&1
bash "$GATE" --make-prompt --prompt-out "$WORK/p2.txt" > /dev/null 2>&1
if [ -s "$WORK/p1.txt" ] && cmp -s "$WORK/p1.txt" "$WORK/p2.txt"; then
    pass "--make-prompt is deterministic"
else
    fail "--make-prompt differs between runs"
fi
sz="$(wc -c < "$ROOT_DIR/scripts/sycl-moe-exec-prompts/code-2k.txt" 2>/dev/null || echo 0)"
if [ "$sz" -ge 4000 ] && [ "$sz" -le 9000 ]; then
    pass "the committed code-2k.txt is $sz bytes"
else
    fail "the committed code-2k.txt is $sz bytes, expected 4000-9000"
fi

echo "cases=$CASES fails=$FAILS"
[ "$FAILS" -eq 0 ]
