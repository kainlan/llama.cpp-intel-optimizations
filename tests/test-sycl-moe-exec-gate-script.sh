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
#   - Q_tok / Q_tok_2 / Q_tok_cpu_1 recorded, Q_tok_cpu across two loads per arm: equal, a flip on either side,
#     a real mismatch naming the first differing byte, a missing reference;
#   - Q_tg / Q_noc: floor, md5, place line (unchanged, changed, unscored), refill balanced / unbalanced /
#     not_exercised, watchdog;
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
#   - RING_ARMS: every arm in the list prints ring_engaged, and an arm outside it cannot move ring_engaged_any.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GATE="$ROOT_DIR/scripts/sycl-moe-exec-gate.sh"
[ -f "$GATE" ] || { echo "SKIP: no $GATE"; exit 77; }

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

# mkcsv <file> <row>...: the llama-bench csv header, then one row per "n_prompt:n_gen:avg_ts:stddev_ts:n_ubatch".
mkcsv() {
    local f="$1" r
    shift
    # shellcheck disable=SC2086
    echo $FIELDS | tr ' ' ',' > "$f"
    for r in "$@"; do
        IFS=: read -r np ng ts sd ub <<< "$r"
        # shellcheck disable=SC2086
        for k in $FIELDS; do
            case "$k" in
                n_prompt) v=$np ;; n_gen) v=$ng ;; avg_ts) v=$ts ;; stddev_ts) v=$sd ;; n_ubatch) v=$ub ;;
                n_depth) v=0 ;; cpu_info) v='Intel(R) Core(TM), 24 threads' ;; *) v=x ;;
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

SUMMARY='0.09.000.000 W [MOE-EXEC] summary: 96 layer-ubatches ring, 0 cpu, 0 forced, 1500 MiB streamed'
TRACE_DECOY='0.05.200.000 I [MOE-EXEC-TRACE] dev 0 layer 3: ring (p=0.99 t_cpu=900 us t_ring=300 us cheaper)'
UNAVAIL='0.04.000.000 W [MOE-EXEC] device 0: stream ring unavailable: cap (host_matrices_per_layer_max=3 n_slots=0 bytes=0)'
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
echo '0.09.000.000 W [MOE-EXEC] summary: 0 layer-ubatches ring, 96 cpu, 0 forced, 0 MiB streamed' >> "$WORK/pp-refused.err"
echo '0.19.000.000 W [MOE-EXEC] summary: 0 layer-ubatches ring, 372 cpu, 0 forced, 0 MiB streamed' >> "$WORK/pp-refused.err"

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
{ cat "$WORK/pp-pass.err"; echo '0.19.000.000 W [MOE-EXEC] summary: 40 layer-ubatches ring, 7 cpu, 0 forced, 600 MiB streamed'; } > "$WORK/pp-sum2.err"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-sum2.err")"
expect "Q_pp ring layer-ubatches add up over the summaries" "$out" "ring_layer_ubatches=136"
expect "Q_pp two summaries pass (one per llama-bench context)" "$out" "Q_pp=PASS"

cp "$WORK/pp-pass.err" "$WORK/pp-47.err"
sed -i '/layer 47:/d' "$WORK/pp-47.err"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-47.err" --q-layers 48)"
expect "Q_pp --q-layers 48 and 47 layers fails by name" "$out" "layers=47!=48"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-47.err")"
expect "Q_pp by default any layer count of at least one passes" "$out" "Q_pp=PASS"
grep -v 'layer [0-9]*:' "$WORK/pp-pass.err" > "$WORK/pp-nolayer.err"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-nolayer.err")"
expect "Q_pp no layer line fails by name" "$out" "layers=0<1"
grep -v 'summary:' "$WORK/pp-pass.err" > "$WORK/pp-nosum.err"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-nosum.err")"
expect "Q_pp no summary fails by name" "$out" "summary=0<1"
expect "Q_pp no summary also means ring_engaged=no" "$out" "ring_engaged=no"
cp "$WORK/pp-pass.err" "$WORK/pp-ref.err"
echo '0.07.000.000 W [PP-MOE] refused device=0 cause=x detail=- tensor=blk.3.ffn_up_exps.weight type=q4_0 layout=aos staged=4' >> "$WORK/pp-ref.err"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-ref.err")"
expect "Q_pp [PP-MOE] refused fails by name" "$out" "Q_pp=FAIL(pp_moe_refused=1)"
cp "$WORK/pp-pass.err" "$WORK/pp-loan.err"; echo "$LOAN_REFUSED" >> "$WORK/pp-loan.err"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-loan.err")"
expect "Q_pp a refused loan is not loan_exercised" "$out" "loan_exercised=no"
echo "$LOAN_OK" >> "$WORK/pp-loan.err"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-loan.err")"
expect "Q_pp a granted loan is loan_exercised" "$out" "loan_exercised=yes"
out="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-pass.err" --rc 124)"
expect "Q_pp a timeout is a failure" "$out" "rc=124"
mkcsv "$WORK/pp-none.csv" 128:0:900:1:128
out="$(score Q_pp --csv "$WORK/pp-none.csv" --err "$WORK/pp-pass.err")"
expect "Q_pp a csv without the rows fails, never passes" "$out" "no_pp512_row"

out="$(score Q_pp_cpu --csv "$WORK/pp-slow.csv" --err "$WORK/pp-refused.err")"
expect "Q_pp_cpu is recorded" "$out" "Q_pp_cpu=RECORDED pp512=14.1 pp2048=31.5"

# ---- Q_staged: layers 0-2 are ring layers (host experts), layers 10 and 11 are resident and run the executor at
# staged=0, layer 5 decided cpu. Only blk.0-2 are scored.
mklayers "$WORK/st-pass.err" 3
{
    echo "0.05.050.000 W [MOE-EXEC] layer 5: cpu (p=1.00 t_cpu=100 us t_ring=900 us cheaper)"
    echo '0.05.000.000 I [PP-MOE] device=0 type=q4_0 layout=aos arm=f16 experts=8 groups=1 planned_weight_slot=100'
    for i in 0 1 2; do
        echo "0.05.100.000 I [MOE-EXEC-TRACE] executor device 0 tensor=blk.$i.ffn_up_exps.weight experts=8 staged=8 arm=f16"
    done
    for i in 5 10 11; do
        echo "0.05.150.000 I [MOE-EXEC-TRACE] executor device 0 tensor=blk.$i.ffn_up_exps.weight experts=8 staged=0 arm=f16"
    done
    echo "$TRACE_DECOY"
    echo "$SUMMARY"
} >> "$WORK/st-pass.err"
out="$(score Q_staged --err "$WORK/st-pass.err")"
expect "Q_staged passes with resident staged=0 lines present" "$out" \
    "Q_staged=PASS ring_layers=3 executor_dispatches=6 ring_layer_dispatches=3"
sed 's/blk.1.ffn_up_exps.weight experts=8 staged=8/blk.1.ffn_up_exps.weight experts=8 staged=0/' "$WORK/st-pass.err" > "$WORK/st-unstaged.err"
out="$(score Q_staged --err "$WORK/st-unstaged.err")"
expect "Q_staged a ring-served layer at staged=0 fails by name" "$out" "Q_staged=FAIL(unstaged=1/3)"
sed 's/blk.2.ffn_up_exps.weight experts=8 staged=8 arm=f16/blk.2.ffn_up_exps.weight experts=8 staged=8 arm=woq/' "$WORK/st-pass.err" > "$WORK/st-woq.err"
out="$(score Q_staged --err "$WORK/st-woq.err")"
expect "Q_staged a non-f16 ring-served arm fails by name" "$out" "Q_staged=FAIL(arm_not_f16=1/3)"
sed 's/blk.10.ffn_up_exps.weight experts=8 staged=0 arm=f16/blk.10.ffn_up_exps.weight experts=8 staged=0 arm=woq/' "$WORK/st-pass.err" > "$WORK/st-resident-woq.err"
out="$(score Q_staged --err "$WORK/st-resident-woq.err")"
expect "Q_staged a resident layer's arm is not scored" "$out" "Q_staged=PASS"
cp "$WORK/st-pass.err" "$WORK/st-refused.err"
echo '0.07.000.000 W [PP-MOE] refused device=0 cause=x detail=- tensor=blk.3.ffn_up_exps.weight type=q4_0 layout=aos staged=4' >> "$WORK/st-refused.err"
out="$(score Q_staged --err "$WORK/st-refused.err")"; rc=$?
expect "Q_staged a [PP-MOE] refused line fails by name" "$out" "Q_staged=FAIL(pp_moe_refused=1)"
expect_rc "Q_staged a refused line rc" "$rc" 1
grep -v 'PP-MOE\] device=' "$WORK/st-pass.err" > "$WORK/st-noarm.err"
out="$(score Q_staged --err "$WORK/st-noarm.err")"
expect "Q_staged a missing [PP-MOE] arm=f16 line fails by name" "$out" "Q_staged=FAIL(no_pp_moe_arm_f16_line)"
grep -v 'tensor=blk\.[012]\.' "$WORK/st-pass.err" > "$WORK/st-nodispatch.err"
out="$(score Q_staged --err "$WORK/st-nodispatch.err")"
expect "Q_staged no ring-served dispatch fails, never passes" "$out" "Q_staged=FAIL(no_ring_layer_dispatch)"
{ echo "$UNAVAIL"; grep -v 'summary:' "$WORK/st-pass.err"; echo '0.09.000.000 W [MOE-EXEC] summary: 0 layer-ubatches ring, 96 cpu, 0 forced, 0 MiB streamed'; } > "$WORK/st-unavail.err"
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
expect "Q_tok_cpu an auto output that no cpu load produced fails" "$out" "Q_tok_cpu=FAIL(auto2_md5="
out="$(tokcpu "$WORK/tok-flip.out" "$WORK/tok-c1.out" "$WORK/tok-a1.out" "$WORK/tok-flip.out")"
expect "Q_tok_cpu an auto flip that either cpu load also made passes" "$out" "Q_tok_cpu=PASS"
expect "Q_tok_cpu reports the auto flip" "$out" "auto_flip=yes"
out="$(tokcpu "$WORK/tok-c2.out" "$WORK/tok-c1.out" "$WORK/tok-bad.out" "$WORK/tok-a2.out")"; rc=$?
expect "Q_tok_cpu a real mismatch fails by name" "$out" "Q_tok_cpu=FAIL(auto1_md5="
expect "Q_tok_cpu a mismatch names the first differing byte" "$out" "first_diff_byte=1"
expect_rc "Q_tok_cpu a mismatch rc" "$rc" 1
out="$(tokcpu "$WORK/tok-c2.out" "$WORK/tok-c1.out" "$WORK/tok-flip.out" "$WORK/tok-a2.out")"
expect "Q_tok_cpu a one-byte difference names byte 22" "$out" "first_diff_byte=22"
out="$(tokcpu "$WORK/tok-c2.out" "$WORK/tok-c1.out" "$WORK/absent.out" "$WORK/tok-a2.out")"; rc=$?
expect "Q_tok_cpu without Q_tok is not scored" "$out" "Q_tok_cpu=NOT_SCORED(no_Q_tok_output)"
expect_rc "Q_tok_cpu not scored rc" "$rc" 3
out="$(tokcpu "$WORK/tok-c2.out" "$WORK/absent.out" "$WORK/tok-a1.out" "$WORK/tok-a2.out")"
expect "Q_tok_cpu without Q_tok_cpu_1 is not scored" "$out" "Q_tok_cpu=NOT_SCORED(no_Q_tok_cpu_1_output)"
: > "$WORK/tok-empty.out"
out="$(tokcpu "$WORK/tok-empty.out" "$WORK/tok-c1.out" "$WORK/tok-a1.out" "$WORK/tok-a2.out")"
expect "Q_tok_cpu an empty cpu output fails by name" "$out" "empty_output"
out="$(score Q_tok_cpu --out "$WORK/tok-c2.out" --out2 "$WORK/tok-c1.out" --ref-out "$WORK/tok-a1.out" --ref-out2 "$WORK/tok-a2.out" --rc 124)"
expect "Q_tok_cpu a timeout fails" "$out" "rc=124"

# ---- Q_tg / Q_noc
echo "decode output" > "$WORK/dec.out"
MD5="$(md5sum < "$WORK/dec.out" | cut -c1-8)"
PLACE_TG='MoE experts: 38004 on device, 35724 on host'
dec_err() { # <file> <tps> <place> [extra line]
    {
        echo "0.01.601.984 I [PLACEMENT] $3 (of 73728 total)"
        echo "1.42.962.387 I llama_perf_context_print:        eval time =    4954.02 ms /    59 runs   (83.97 ms per token,    $2 tokens per second)"
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
expect "Q_tg below the floor fails" "$out" "tg=10.20<11.0"
expect "Q_tg without a loan says refill not_exercised" "$out" "refill=not_exercised"
expect "Q_tg without a loan says loan_exercised=no" "$out" "loan_exercised=no"
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg.err" --qmd5 deadbeef --place-ref-tg "$PLACE_TG")"
expect "Q_tg wrong md5 fails" "$out" "md5=$MD5!=deadbeef"
dec_err "$WORK/tg-unbal.err" 11.91 "$PLACE_TG" '0.09.000.000 W [MOE-EXEC] refill device 0: 23/24 triplets re-admitted'
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg-unbal.err" --qmd5 "$MD5" --place-ref-tg "$PLACE_TG")"
expect "Q_tg unbalanced refill fails" "$out" "refill=unbalanced(23/24)"
dec_err "$WORK/tg-place.err" 11.91 'MoE experts: 37000 on device, 36728 on host'
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg-place.err" --qmd5 "$MD5" --place-ref-tg "$PLACE_TG")"
expect "Q_tg changed place line fails" "$out" "place='MoE experts: 37000 on device, 36728 on host'!="
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg-place.err" --qmd5 "$MD5" --place-ref-tg -)"
expect "Q_tg place ref - is reported as unscored" "$out" "place=unscored"
dec_err "$WORK/tg-wd.err" 11.91 "$PLACE_TG" '0.09.000.000 W [SYCL-WATCHDOG] No GPU progress for 30 s'
out="$(score Q_tg --out "$WORK/dec.out" --err "$WORK/tg-wd.err" --qmd5 "$MD5" --place-ref-tg "$PLACE_TG")"
expect "Q_tg watchdog fails by name" "$out" "watchdog=1"
dec_err "$WORK/noc.err" 9.80 'MoE experts: 38163 on device, 35565 on host'
out="$(score Q_noc --out "$WORK/dec.out" --err "$WORK/noc.err" --qmd5 "$MD5" --place-ref-noc 'MoE experts: 38163 on device, 35565 on host')"
expect "Q_noc pass at 9.80 against the 9.5 floor" "$out" "Q_noc=PASS tg=9.80"
dec_err "$WORK/noc-slow.err" 9.30 'MoE experts: 38163 on device, 35565 on host'
out="$(score Q_noc --out "$WORK/dec.out" --err "$WORK/noc-slow.err" --qmd5 "$MD5" --place-ref-noc -)"
expect "Q_noc below the floor fails" "$out" "tg=9.30<9.5"

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
expect "GO_gate layer lines fail it" "$out" "layer_lines=48!=0"
mklayers "$WORK/go12.err" 12
out="$(score GO_noc --out "$WORK/go.out" --err "$WORK/go12.err")"
expect "GO_noc 12 layer lines pass by default" "$out" "GO_noc=PASS count=1 layer_lines=12"
mklayers "$WORK/go11.err" 11
out="$(score GO_noc --out "$WORK/go.out" --err "$WORK/go11.err" --go-noc-layers 12)"
expect "GO_noc --go-noc-layers 12 and 11 layer lines fail" "$out" "layer_lines=11!=12"
out="$(score GO_noc --out "$WORK/go.out" --err "$WORK/empty.err")"
expect "GO_noc with no layer line fails: the demoted layers were never consulted" "$out" "layer_lines=0<1"

mkcsv "$WORK/gopp-ok.csv" 512:0:880.0:4.0:512 0:128:33.9:0.1:512
mkcsv "$WORK/gopp-low.csv" 512:0:770.0:4.0:512 0:128:33.9:0.1:512
mkcsv "$WORK/gopp-tg.csv" 512:0:880.0:4.0:512 0:128:32.0:0.1:512
out="$(score GO_pp --csv "$WORK/gopp-ok.csv")"
expect "GO_pp inside both windows passes" "$out" "GO_pp=PASS pp512=880.0"
out="$(score GO_pp --csv "$WORK/gopp-low.csv")"
expect "GO_pp pp512 770 fails low" "$out" "GO_pp=FAIL(pp512=770.0<827."
out="$(score GO_pp --csv "$WORK/gopp-tg.csv")"
expect "GO_pp tg128 32.0 fails low" "$out" "tg128=32.0<33.0"

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
for arm in Q_pp Q_pp_cpu Q_staged Q_tok Q_tok_2 Q_tok_cpu_1 Q_tok_cpu Q_tg Q_noc GO_gate GO_pp GO_noc MI_b70 MI_b50 resets; do
    expect "dry-run lists $arm" "$out" "$arm: "
done
n_sel="$(printf '%s\n' "$out" | grep -c 'ONEAPI_DEVICE_SELECTOR=level_zero:[01] ')"
expect_rc "dry-run pins the selector in all fourteen device arms" "$n_sel" 14
n_q900="$(printf '%s\n' "$out" | grep -E '^Q_[A-Za-z0-9_]+: ' | grep -c 'timeout -k 15 900 ')"
expect_rc "dry-run gives every Qwen arm timeout 900" "$n_q900" 9
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
if [ -e "$WORK/GPU.lock" ]; then fail "a wrong-sha refusal created GPU.lock"; else pass "a wrong-sha refusal never creates GPU.lock"; fi
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
if cat "$GATE" | grep -vE '^[[:space:]]*#' | grep -qE '(^|[^A-Za-z_])BIN_DIR='; then fail "the gate assigns a bare BIN_DIR, which oneAPI's vars.sh overwrites"; else pass "the gate assigns no bare BIN_DIR"; fi
src_line="$(cat "$GATE" | grep -vE '^[[:space:]]*#' | grep -n 'source /opt/intel/oneapi/setvars.sh' | head -1 | cut -d: -f1)"
use_line="$(cat "$GATE" | grep -vE '^[[:space:]]*#' | grep -n 'GATE_BIN_DIR' | head -1 | cut -d: -f1)"
if [ -n "$src_line" ] && [ -n "$use_line" ] && [ "$src_line" -lt "$use_line" ]; then
    pass "setvars is sourced (line $src_line) before the first use of GATE_BIN_DIR (line $use_line)"
else
    fail "setvars source line '${src_line:-none}' must precede the first GATE_BIN_DIR line '${use_line:-none}'"
fi
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    for f in /opt/intel/oneapi/setvars.sh /opt/intel/oneapi/*/latest/env/vars.sh /opt/intel/oneapi/*/latest/oneapi-vars.sh \
             /opt/intel/oneapi/*/latest/etc/*/vars.sh /opt/intel/oneapi/*/latest/etc/*/*.sh; do
        [ -f "$f" ] && cat "$f"
    done 2>/dev/null | grep -oE '^[[:space:]]*(export +|local +|declare +)?[A-Za-z_][A-Za-z_0-9]*=' \
        | sed -E 's/^[[:space:]]*(export |local |declare )? *//; s/=$//' | LC_ALL=C sort -u > "$WORK/oneapi-names.txt"
    cat "$GATE" | grep -vE '^[[:space:]]*#' | grep -oE '(^|[;{(]|then|do)[[:space:]]*[A-Za-z_][A-Za-z_0-9]*(\[[^]]*\])?\+?=' \
        | sed -E 's/^(;|\{|\(|then|do)?[[:space:]]*//; s/(\[[^]]*\])?\+?=$//' | LC_ALL=C sort -u > "$WORK/gate-names.txt"
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
out="$(bash "$GATE" --no-head-check --no-lock --arm Q_pp --build-dir "$WORK/fakebuild" --out-dir "$WORK/fakeout" 2>&1)"; rc=$?
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
    fail "signal selftest did not reach its waiting state (lock '$LOCKD', child '${child:-none}')" "$(cat "$WORK/sig.out" 2>/dev/null)"
fi
kill -TERM "$gpid" 2>/dev/null
wait "$gpid"; rc=$?
out="$(cat "$WORK/sig.out")"
expect_rc "a TERM ends the gate with rc 143" "$rc" 143
expect "a TERM prints ABORTED" "$out" "ABORTED by signal"
expect "a TERM still ends with the marker line" "$out" "verdicts_pass=0 GATE_DONE"
if [ -e "$LOCKD" ]; then fail "a TERM left GPU.lock behind"; else pass "a TERM releases GPU.lock"; fi
if [ -n "$child" ] && kill -0 "$child" 2>/dev/null; then fail "a TERM left the child $child running"; else pass "a TERM kills the child"; fi

# ---- RING_ARMS: the arms ring_engaged_any is read from
read -ra RARMS <<< "$(bash "$GATE" --ring-arms 2>&1)"
if [ "${#RARMS[@]}" -gt 0 ]; then pass "the gate names its ring arms: ${RARMS[*]}"; else fail "--ring-arms printed nothing"; fi
for arm in "${RARMS[@]}"; do
    case "$arm" in
        Q_pp) line="$(score Q_pp --csv "$WORK/pp-pass.csv" --err "$WORK/pp-pass.err")" ;;
        Q_staged) line="$(score Q_staged --err "$WORK/st-pass.err")" ;;
        *) line="" ;;
    esac
    if [ -z "$line" ]; then fail "ring arm $arm has no fixture in this test: add one"; else expect "ring arm $arm prints ring_engaged=" "$line" "ring_engaged="; fi
done
out="$(summ outside "$PASS_RING" "Q_tg=PASS tg=11.9 ring_engaged=no")"
expect "an arm outside RING_ARMS cannot move ring_engaged_any to no" "$out" "ring_engaged_any=yes"
out="$(summ outside2 "$CTRL_PASS" "Q_tg=PASS tg=11.9 ring_engaged=no")"; rc=$?
expect "an arm outside RING_ARMS does not make ring_engaged_any=no" "$out" "ring_engaged_any=na"
expect_rc "an arm outside RING_ARMS leaves the exit code 0" "$rc" 0

# ---- the prompt
bash "$GATE" --make-prompt --prompt-out "$WORK/p1.txt" > /dev/null 2>&1
bash "$GATE" --make-prompt --prompt-out "$WORK/p2.txt" > /dev/null 2>&1
if [ -s "$WORK/p1.txt" ] && cmp -s "$WORK/p1.txt" "$WORK/p2.txt"; then pass "--make-prompt is deterministic"; else fail "--make-prompt differs between runs"; fi
sz="$(wc -c < "$ROOT_DIR/scripts/sycl-moe-exec-prompts/code-2k.txt" 2>/dev/null || echo 0)"
if [ "$sz" -ge 4000 ] && [ "$sz" -le 9000 ]; then pass "the committed code-2k.txt is $sz bytes"; else fail "the committed code-2k.txt is $sz bytes, expected 4000-9000"; fi

echo "cases=$CASES fails=$FAILS"
[ "$FAILS" -eq 0 ]
