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
#   - Q_staged: pass, an unstaged dispatch, a non-f16 arm, and NOT_EXERCISED when no layer-ubatch used the ring;
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
#   - --make-prompt is deterministic and the committed prompt exists.
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
expect "Q_pp [PP-MOE] refused fails by name" "$out" "pp_moe_refused=1"
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

# ---- Q_staged
{
    echo '0.05.000.000 I [PP-MOE] device=0 type=q4_0 layout=aos arm=f16 experts=8 groups=1 planned_weight_slot=100'
    for i in 0 1 2; do
        echo "0.05.100.000 I [MOE-EXEC-TRACE] executor device 0 tensor=blk.$i.ffn_up_exps.weight experts=8 staged=8 arm=f16"
    done
    echo "$TRACE_DECOY"
    echo "$SUMMARY"
} > "$WORK/st-pass.err"
out="$(score Q_staged --err "$WORK/st-pass.err")"
expect "Q_staged pass" "$out" "Q_staged=PASS executor_dispatches=3"
sed 's/blk.1.ffn_up_exps.weight experts=8 staged=8/blk.1.ffn_up_exps.weight experts=8 staged=0/' "$WORK/st-pass.err" > "$WORK/st-unstaged.err"
out="$(score Q_staged --err "$WORK/st-unstaged.err")"
expect "Q_staged an unstaged dispatch fails by name" "$out" "Q_staged=FAIL(unstaged=1/3)"
sed 's/blk.2.ffn_up_exps.weight experts=8 staged=8 arm=f16/blk.2.ffn_up_exps.weight experts=8 staged=8 arm=woq/' "$WORK/st-pass.err" > "$WORK/st-woq.err"
out="$(score Q_staged --err "$WORK/st-woq.err")"
expect "Q_staged a non-f16 arm fails by name" "$out" "arm_not_f16=1/3"
{ echo "$UNAVAIL"; grep -v 'summary:' "$WORK/st-pass.err"; echo '0.09.000.000 W [MOE-EXEC] summary: 0 layer-ubatches ring, 96 cpu, 0 forced, 0 MiB streamed'; } > "$WORK/st-unavail.err"
out="$(score Q_staged --err "$WORK/st-unavail.err")"; rc=$?
expect "Q_staged ring-unavailable is NOT_EXERCISED" "$out" "Q_staged=NOT_EXERCISED(ring_engaged=no)"
expect_not "Q_staged ring-unavailable is not a pass" "$out" "Q_staged=PASS"
expect_rc "Q_staged ring-unavailable rc" "$rc" 3
{ echo "$TRACE_DECOY"; echo "$SUMMARY"; } > "$WORK/st-none.err"
out="$(score Q_staged --err "$WORK/st-none.err")"
expect "Q_staged without executor lines fails, never passes" "$out" "no_executor_trace_lines"

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
        echo "1.42.962.387 I common_perf_print:        eval time =    4954.02 ms /    59 runs   (83.97 ms per token,    $2 tokens per second)"
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
out="$(bash "$GATE" --lock-dir "$WORK/GPU.lock" --lock-selftest 2>&1)"; rc=$?
expect_rc "lock selftest rc" "$rc" 0
expect "lock selftest ok" "$out" "lock-selftest: ok"
out="$(bash "$GATE" --arm Q_nope 2>&1)"; rc=$?
expect_rc "an unknown arm is a usage error" "$rc" 2

# ---- the prompt
bash "$GATE" --make-prompt --prompt-out "$WORK/p1.txt" > /dev/null 2>&1
bash "$GATE" --make-prompt --prompt-out "$WORK/p2.txt" > /dev/null 2>&1
if [ -s "$WORK/p1.txt" ] && cmp -s "$WORK/p1.txt" "$WORK/p2.txt"; then pass "--make-prompt is deterministic"; else fail "--make-prompt differs between runs"; fi
sz="$(wc -c < "$ROOT_DIR/scripts/sycl-moe-exec-prompts/code-2k.txt" 2>/dev/null || echo 0)"
if [ "$sz" -ge 4000 ] && [ "$sz" -le 9000 ]; then pass "the committed code-2k.txt is $sz bytes"; else fail "the committed code-2k.txt is $sz bytes, expected 4000-9000"; fi

echo "cases=$CASES fails=$FAILS"
[ "$FAILS" -eq 0 ]
