#!/usr/bin/env bash
# sycl-moe-exec-gate: the lead-run gate for the MoE executor cost model and prefill ring (epic llama.cpp-36j0,
# plan Task 15, llama.cpp-x6zo). It runs the Qwen3.8 B70 prefill and decode arms, the GPT-OSS B50 and Mistral
# controls, and scores every arm with an exact string. It ends with one line `verdicts_pass=N GATE_DONE`; a waiter
# arms on the word GATE_DONE and on nothing else. A refusal prints `REFUSED: <why>` and the same final line.
#
# Run it from the lead session only, one at a time, detached, with the merge sha that the build in --build-dir
# was made from:
#   setsid nohup bash scripts/sycl-moe-exec-gate.sh <sha> > <scratchpad>/gate.log 2>&1 < /dev/null &
# It loads models onto both discrete cards. Never run it from a subagent, never in a loop.
#
# Arms (selector pinned in every arm; no -ngl anywhere; timeouts: the three Q_pp arms 1500 s, the other Qwen arms 900 s,
# GO_gate and GO_noc 600 s, the Mistral arms 300 s; a run that hits its timeout exits 124 and one that has to be
# killed exits 137, and every arm scores a non-zero rc as a FAIL naming it):
#   Q_pp       llama-bench -m $Q -p 512,2048 -n 0 -r 3 -fa 1 -o csv -v          B70  pp512 >= 200, pp2048 >= 500
#              (the bus-derived floors, see PP512_MIN), ring_engaged=yes, zero window misses, the window fields on
#              every summary line,
#              at least one distinct "[MOE-EXEC] layer N:" line (exactly --q-layers N when given), at least one
#              "[MOE-EXEC] summary:" (llama-bench makes one context per -p value, so -p 512,2048 prints two),
#              zero "[PP-MOE] refused".
#              The plan wrote 48 layers "at the first ubatch". Both were wrong, and the real log is the evidence:
#              Qwen3.8 prints 24 distinct "[MOE-EXEC] layer N:" lines (only layers with host experts have one), and
#              the line prints once per change of choice, so "at the first ubatch" names no fixed moment. The default
#              is therefore "at least one layer line", and --q-layers N makes the count exact.
#              The verdict carries ring_engaged=yes|no and ring_layer_ubatches=K. ring_engaged is yes only when the
#              "[MOE-EXEC] summary: K layer-ubatches ring" lines add up to K > 0. It is NOT read from the
#              "stream ring unavailable" WARN: under auto at default verbosity that WARN is not printed at all
#              (it needs GGML_SYCL_MOE_EXEC=ring or the trace), so its absence proves nothing (llama.cpp-bcre).
#              loan_exercised=yes|no is a "[MOE-EXEC] loan device N: K triplets" line.
#   Q_pp_ub2048  llama-bench -m $Q -ub 2048 -p 2048 -n 0 -r 3 -fa 1 -o csv -v    B70
#              pp2048 >= 500 only (no pp512 row), scored like Q_pp otherwise; the pp2048 floor's natural setting
#              and llama.cpp-dfo0's measurement arm
#   Q_pp_cpu   Q_pp with GGML_SYCL_MOE_EXEC=cpu                       B70  recorded, not gated (FAIL on rc != 0)
#   Q_staged   llama-bench -p 512 -n 0 -r 1 -fa 1 -o csv -v, GGML_SYCL_MOE_EXEC_TRACE=1   B70
#              the executor prints "[MOE-EXEC-TRACE] executor device D tensor=T experts=E staged=S arm=A" for EVERY
#              dispatch, resident layers included (staged=0 there is correct). So only the layers the ring served
#              are scored: a layer N with a "[MOE-EXEC] layer N: ring" line in the run is a host layer, and every
#              trace line whose tensor is blk.N.* must have S > 0 and A = f16. At least one such dispatch must exist.
#              Also at least one "[PP-MOE] device=D type=T layout=L arm=f16" line and zero "[PP-MOE] refused".
#              A "misses=N" on the trace lines sums to trace_misses=N, a FAIL when above 0; trace_misses=NA
#              (no trace line carries the field) and 0 are clean, and the verdict always prints the value.
#              NOT_EXERCISED (neither pass nor fail) when ring_engaged=no: nothing was staged to count.
#   Q_tok      llama-completion -m $Q -f code-2k.txt -n 32 --seed 42 --temp 0 -no-cnv --no-display-prompt -lv 4 -t 4
#              B70  md5 of stdout recorded (load 1 of 2); an rc != 0 or an empty stdout is a FAIL, not a recorded md5
#   Q_tok_2    Q_tok again, a second load                                         B70  md5 recorded
#   Q_tok_cpu_1  Q_tok with GGML_SYCL_MOE_EXEC=cpu                                B70  md5 recorded (load 1 of 2)
#   Q_tok_cpu  Q_tok_cpu_1 again, a second load, and the verdict                  B70  PASS when at least one auto md5
#              equals at least one cpu md5 AND every auto output is identical to some cpu output or shares a prefix
#              of at least 64 bytes with its best-matching cpu output. Qwen3.8 greedy decoding on the B70 has a
#              near-tie that flips from load to load in BOTH policies and cascades after its token (run 2: four loads,
#              three distinct 32-token outputs, every divergence at byte 106-116 after an identical prefix), so two
#              loads per arm cannot enumerate it; identical text across the policies is the evidence the math agrees.
#              The verdict prints distinct_auto= distinct_cpu= pairs_equal= min_prefix=: a DEFECT shows as
#              pairs_equal=0 with an early min_prefix, a FLIP as pairs_equal >= 1 with a late one. stdout only (the
#              log goes to stderr). All four outputs must be non-empty and all four loads must have exited 0.
#   Q_tg       the q0xu Q_c4096 arm verbatim (code.txt, -n 60 -c 4096 -ub 256 -t 4)   B70  tg >= 11.0, md5 25b7738a
#              (a different md5 is a greedy near-tie flip, still a PASS with flip=yes, when the output shares at
#              least 128 bytes with the reference text sycl-moe-exec-prompts/code-ref-25b7738a.txt, and a FAIL with
#              flip=no below that; the verdict prints md5= flip= prefix=), watchdog 0, place line as --place-ref-tg
#              (default "-": recorded, not scored), every "[MOE-EXEC] refill device D: R/L" has R = L (checked per
#              line, so 24/23 on one device and 23/24 on another is unbalanced although the sums agree). tg is the
#              figure on the last " eval time = " line, never the prompt-eval rate: a nan eval line is no_tg.
#              A loan whose prompt ring never engaged (loan_exercised=yes, ring_engaged=no) must come back in one
#              transaction at the first decode graph (llama.cpp-i4ny): a "[MOE-EXEC] bulk refill device D: R of L lent
#              triplets re-admitted in one transaction at decode graph 1 after the prompt" line, else FAIL(refill_late).
#              The verdict prints bulk_refill=first|late|not_owed. The bulk line never matches the refill regex above.
#   Q_noc      the q0xu Q_noc arm (the same without -c)                          B70  tg >= 9.5, md5 25b7738a,
#              watchdog 0, place line as --place-ref-noc, and the same md5/flip, refill and bulk refill rules as Q_tg
#   Q_decline  Q_tok with GGML_SYCL_MOE_EXEC_DECLINE_LAYER=20                    B70  the knob turns layer 20's
#              decision to cpu with reason "declined", so the executor does not claim it and it runs the hybrid path
#              (llama.cpp-ny6i, the forced-decline proof of llama.cpp-8p5x c-0blp). Its references are the two auto
#              loads that run before it, Q_tok and Q_tok_2: Qwen3.8 greedy decoding on the B70 flips between two texts
#              from load to load (see Q_tok_cpu), so one md5 cannot be the reference. PASS needs at least one
#              "[MOE-EXEC] layer 20: cpu (... declined)" line and no "[MOE-EXEC] layer 20: ring" line, the
#              "[MOE-EXEC] summary:" cpu counts adding up to >= 1, stdout identical to Q_tok's or Q_tok_2's (md5; a
#              mismatch with both prints the shared prefix against each), zero "runtime KV update rejected", zero
#              "mem_handle::resolve(device_id=N): wrong-device access", rc 0 and a non-empty stdout. NOT_SCORED when
#              Q_tok's or Q_tok_2's output is missing or empty or its load failed (that arm's own verdict carries the
#              failure). NOT_EXERCISED when it would pass but Q_tok never ran layer 20 on the ring (its log has no
#              "[MOE-EXEC] layer 20: ring" line): then the decline changed nothing, and identical text proves nothing.
#              The verdict prints ref_layer20=ring|cpu|none and declined_layers=N, how many distinct layers carry a
#              declined line (0 or 1 for the one-layer knob). begin_layer applies the knob only to a layer with host
#              experts, so a layer the placement keeps whole on the device prints no decline line: that reads as
#              FAIL(no_decline_line) with ref_layer20=none. Not a ring arm: it does not move ring_engaged_any.
#   GO_gate    llama-cli GPT-OSS count gate at -c 4096 (all in VRAM)            B50  grep -cx '1, 2, 3, 4, 5' = 1,
#              zero "[MOE-EXEC] layer" lines
#   GO_pp      llama-bench -m $GO -p 512 -n 128 -r 3 -o csv                     B50  pp512 within 5 % of 871,
#              tg128 within 3 % of 32.0 (docs/backend/sycl-perf-baselines.md says B50 GPT-OSS TG128 ~32 and the B50 is
#              steady, cv 0.7 %). llama.cpp-zuze: master reads pp512 ~770 on the B50, so a pp512 below the window but
#              within 15 % below 871 is a known master regression, expected to FAIL until zuze lands; the verdict
#              names it (see=llama.cpp-zuze).
#   GO_noc     the same llama-cli gate without -c                               B50  count 1; layer_lines is recorded,
#              not gated (0 on run 2: GPT-OSS without -c keeps every expert on the device, and what overflows at the
#              default context is KV, which the planner demotes to host tiers per llama.cpp-uize, so no executor
#              decision prints); --go-noc-layers N makes it an exact check.
#   MI_b70     llama-completion Mistral digits gate                             B70  grep -c '1, 2, ..., 10' = 1
#   MI_b50     the same                                                         B50
#   resets     journalctl -k --since <gate start> | grep -icE
#              'GT reset|guc_id|CAT error|Engine reset|exec queue reset' = 0
#              (journalctl's own rc is kept: a failed journalctl or a missing journal file is a FAIL, never a clean log)
# Duration: a complete run of the first 16 arms takes about 55 minutes on this host with the ring off (Q_pp 9.5 min,
# Q_pp_cpu 9.3, Q_pp_ub2048 about the same, Q_staged 3.5, each Q_tok load about 2, Q_tg and Q_noc about 2.5 each, the
# GPT-OSS and Mistral arms about 1 each; the 15 arms of run 2 took 45). Q_decline is one more Q_tok load on the B70 and
# takes about 2 minutes including the load, the same as each Q_tok arm (2026-10-25 B70 run).
# Note on the plan table: llama-bench has no -c option (its context is n_prompt + n_gen), so GO_pp carries none.
# Verdict-bearing arms (13): Q_pp Q_pp_ub2048 Q_staged Q_tok_cpu Q_tg Q_noc Q_decline GO_gate GO_pp GO_noc MI_b70
# MI_b50 resets.
# Arms in all (17): the thirteen, Q_pp_cpu, Q_tok, Q_tok_2 and Q_tok_cpu_1 (recorded, not gated).
#
# Every grep has a positive and a negative control (check_controls) that runs before any arm and is exposed as
# --check-controls. The GPT-OSS echo-line trap (the prompt ends with the answer, so a loose grep scores 2) is the
# negative control of the count scorer.
#
# Preflights refuse (rc 2): no <sha> or a head that differs from it; GGML_SYCL not ON; a binary missing, older than
# the head commit, or not linked to the SYCL runtime; a model file missing; a failed grep control; builders
# (ninja + icpx) that do not clear within 15 minutes; Shmem >= 30 GB or MemAvailable <= 150 GB; GPU.lock already held;
# a GGML_SYCL_* or LLAMA_ARG_* variable in the environment (every arm would inherit it; each arm sets its own); a
# non-empty --out-dir unless --arm names the arms to redo or --clobber is given (a reused directory would score one
# run's files as another's). Before each arm the host must settle (Shmem < 30 GB and MemAvailable > 150 GB, up to
# 10 minutes) and
# the builders must reach 0 (the gate waits for them like settle, up to 15 minutes, 10 s polls); only an expired wait
# stops the gate. Shmem, MemAvailable, load and builders are sampled before each arm and 5 s after it, and builders
# every 5 s during it; every verdict line carries builders_pre= and builders_post= (the highest count seen during or
# after the arm). A PASS with builders_post > 0 becomes NOT_SCORED(tainted=builders,was=PASS), exit class 3: a build
# running next to the arm depresses throughput and flips CPU-path timing. A FAIL stays a FAIL and carries the flag.
# Once the lock is held, every stop (a refusal such as an expired wait, or a signal) prints STOPPED: <reason>, the
# summary over the verdicts scored so far, and the final line; exit 2 for a refusal, 143 for a signal.
# GPU.lock is `mkdir $LOCK_DIR` with a `holder` file inside, released with `rm -f holder && rmdir`.
#
# Exit codes: 0 every requested verdict PASS; 1 at least one FAIL; 2 refused or usage; 3 none failed but at least
# one arm was NOT_EXERCISED or NOT_SCORED, or a ring-scoring arm ran and no arm engaged the ring (the summary line
# says ring_engaged_any=no: a Q_pp that PASSed its floors without the ring is still not the ring working); 143 the
# gate was stopped by TERM, INT or HUP (it prints ABORTED and the final line, and releases the lock).
# The summary line carries ring_engaged_any=yes|no|na (na: no arm that run carries a ring_engaged field) and
# loan_exercised_any=yes|no, both read from the verdict lines. --score-summary FILE scores a file of verdict lines.
# <sha> is the head the build was made from, at least 7 hex characters; a shorter or non-hex value is refused, also
# under --no-head-check. A usage error (a typo in a detached run) prints REFUSED and the final line too, so the waiter
# is released; --help and --dry-run keep the plain form.
#
# The lead form: the script runs from a snapshot copy, against a build whose source tree is the merge tree:
#   cp -a scripts/sycl-moe-exec-gate.sh scripts/sycl-moe-exec-prompts scripts/qwen4exp-mtp-prompts <snap>/scripts/
#   bash <snap>/scripts/sycl-moe-exec-gate.sh --build-dir <merge-tree>/build --out-dir <fresh dir> <sha>
# The prompts resolve next to the script (the snapshot); the head check, the binary-newer-than-head check and the
# prompt source docs come from --src-dir, which defaults to the parent of --build-dir. --no-head-check stays as an
# explicit escape: it skips the head comparison and the binary-newer-than-head check, and it gives up exactly the
# guarantee that the binaries were built from <sha>. It still validates the sha format and still refuses a missing
# binary or a missing SYCL backend.
#
# Caveat, found by the first real run (2026-10-23): oneAPI's setvars.sh overwrites shell variables. The advisor and
# vtune vars.sh files assign BIN_DIR=bin64 as a plain variable, so a gate that set BIN_DIR="$BUILD_DIR/bin" and then
# sourced setvars looked for bin64/llama-bench and refused every run. The --dry-run never sources setvars and could
# not see it. So: a real run sources setvars.sh once, at the top, before any variable is derived; the build's bin
# directory is GATE_BIN_DIR; and no top-level variable of this file shares a name with one oneAPI assigns (the test
# audits that against /opt/intel/oneapi when it is installed).
#
# Options:
#   --help | --dry-run (print each arm's command and scoring, run nothing) | --arm NAME (repeatable)
#   --make-prompt [--prompt-out FILE]   regenerate scripts/sycl-moe-exec-prompts/code-2k.txt from docs/
#   --score-arm NAME [--csv F] [--out F] [--out2 F] [--err F] [--ref-out F] [--ref-out2 F] [--ref-err F] [--journal F]
#                                       [--rc N] [--rc-out2 N --rc-ref N --rc-ref2 N: the rcs of Q_tok_cpu_1, Q_tok,
#                                       Q_tok_2] [--builders-pre N --builders-post N]
#                                       score existing files, GPU-free; for Q_tok_cpu: --out cpu load 2, --out2 cpu
#                                       load 1, --ref-out auto load 1, --ref-out2 auto load 2; for Q_decline: --ref-out
#                                       and --ref-out2 Q_tok's and Q_tok_2's stdout, --ref-err Q_tok's log, --rc-ref
#                                       and --rc-ref2 their rcs
#   --score-summary FILE                print the summary line and exit with the code a gate over those verdict lines
#                                       (one per line) would, GPU-free
#   --ring-arms                         print the arms whose verdict carries ring_engaged (RING_ARMS), which is what
#                                       ring_engaged_any is read from
#   --signal-selftest                   take GPU.lock, start a sleeping child, wait; a TERM must kill the child, release
#                                       the lock, print the final line and exit 143 (GPU-free, for the test)
#   --refuse-selftest / --wait-builders-selftest   GPU-free hooks of the stop paths (--seed-verdicts F preloads
#                                       verdict lines; --builders-file F, --builders-wait-max N and --builders-poll N
#                                       replace the pgrep count and the wait bound), for the test
#   --check-controls                    run only the grep controls
#   --build-dir DIR  --src-dir DIR (source tree of the build; default: the parent of --build-dir)
#   --out-dir DIR  --clobber (accept a non-empty --out-dir)  --lock-dir DIR  --no-lock  --no-head-check
#   --since 'YYYY-MM-DD HH:MM:SS'
#   --qwen F --gptoss F --mistral F   model files
#   --tuning-dir DIR   export GGML_SYCL_TUNING_CACHE_DIR (default: the real cache, so acceptance runs write it)
#   --place-ref-tg STR --place-ref-noc STR   expected "MoE experts: N on device, M on host" (default "-": recorded,
#                                       not scored; the lead passes the baselines table's values for the tree)
#   --qmd5 HEX8 (25b7738a)  --decode-ref FILE (sycl-moe-exec-prompts/code-ref-25b7738a.txt)
#   --q-layers N (any)  --go-noc-layers N (default "-": recorded, not scored)  --pp512-min N (500)  --pp2048-min N (800)
#   --go-pp-ref N (871)  --go-tg-ref N (32.0)

set -o pipefail
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="$ROOT_DIR/build"
OUT_DIR=""
LOCK_DIR="${LLAMA_GPU_LOCK:-/Apps/llama.cpp/GPU.lock}"
TAKE_LOCK=1
HEAD_CHECK=1
EXPECT_SHA=""
SCORE_SUMMARY=""
RING_ARMS_MODE=0
SIGNAL_SELFTEST=0
DRY_RUN=0
MAKE_PROMPT=0
CHECK_CONTROLS=0
SCORE_ARM=""
PROMPT_OUT="$SCRIPT_DIR/sycl-moe-exec-prompts/code-2k.txt"
SRC_DIR=""
CLOBBER=0
ARMS_EXPLICIT=0
SEED_VERDICTS=""
REFUSE_SELFTEST=0
WAIT_BUILDERS_SELFTEST=0
BUILDERS_POST_SELFTEST=""
BUILDERS_FILE=""
BUILDERS_WAIT_MAX=900
BUILDERS_POLL=10
RUN_STARTED=0
declare -a VERDICTS=()
SAMPLER=""
S_BPRE=na S_BPOST=na S_RC_OUT2=0 S_RC_REF=0 S_RC_REF2=0 S_REF_ERR=""
TUNING_DIR=""
SINCE=""
LOCK_SELFTEST=0

QWEN="/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
GPTOSS="/models/gpt-oss-20b-mxfp4.gguf"
MISTRAL="/models/mistral-7b-v0.1.Q4_0.gguf"
DECODE_PROMPT="$SCRIPT_DIR/qwen4exp-mtp-prompts/code.txt"
TOK_PROMPT="$SCRIPT_DIR/sycl-moe-exec-prompts/code-2k.txt"

# The prefill floors come from the bus, not from the old targets (plan 2026-10-24-prefill-window-ring.md, "Why this plan
# exists" item 4). Qwen3.8 on the B70 places 44091 of 73728 experts on the host; a 512-token ubatch touches almost every
# expert of a host layer, about 29 fully-host layers x 1536 matrices x 0.92 MB = ~41 GB per ubatch. At the probed
# ~25 GB/s that is ~1.6 s per ubatch, a pp512 ceiling near 312 tok/s. PP512_MIN is 200, 64 % of that ceiling. The old
# 500 is above the ceiling for this placement and is retired. At -ub 2048 the same bytes carry four times the tokens,
# so pp2048 is no longer bus-bound: PP2048_MIN is 500, the compute-bound floor the 2026-10-19 plan already defended
# (the old 800 needs -ub 2048 and is llama.cpp-dfo0's measurement, not a gate). --pp512-min / --pp2048-min override.
PP512_MIN=200
PP2048_MIN=500
QTG_MIN="11.0"
QNOC_MIN="9.5"
QMD5="25b7738a"
Q_LAYERS=any
GO_NOC_LAYERS="-"
GO_PP_REF=871
GO_PP_TOL=5
GO_TG_REF="32.0"
GO_TG_TOL=3
# The placement references default to "-" (recorded, not scored): the line moves with the tree (the cost-based KV room,
# llama.cpp-w013, demotes more experts at the default context), so the lead passes the values from the baselines table.
PLACE_REF_TG="-"
PLACE_REF_NOC="-"
# The decode arms' reference text (the q0xu code.txt output, md5 25b7738a) and the prefix lengths that tell a greedy
# near-tie flip from a defect: a flip diverges after a shared prefix, a defect diverges early.
DECODE_REF="$SCRIPT_DIR/sycl-moe-exec-prompts/code-ref-25b7738a.txt"
DECODE_FLIP_PREFIX_MIN=128
TOK_PREFIX_MIN=64
# The forced-decline arm (llama.cpp-8p5x c-0blp): Q_tok's load, where Qwen3.8 on the B70 runs layers 16..46 on the ring
# for the prompt (llama.cpp-ny6i c-nnce), and GGML_SYCL_MOE_EXEC_DECLINE_LAYER takes layer 20 off the executor. A layer
# the placement keeps whole on the device is never consulted, so if layer 20 stops being a host layer the arm reads
# FAIL(no_decline_line) with ref_layer20=none, and this value moves, not the code.
DECLINE_LAYER=20

# The arms whose verdict carries ring_engaged=yes|no; ring_engaged_any in the summary is read from these and only these.
RING_ARMS=(Q_pp Q_pp_ub2048 Q_staged)
ALL_ARMS=(Q_pp Q_pp_ub2048 Q_pp_cpu Q_staged Q_tok Q_tok_2 Q_tok_cpu_1 Q_tok_cpu Q_tg Q_noc Q_decline GO_gate GO_pp
          GO_noc MI_b70 MI_b50 resets)
declare -a ARMS=()
S_CSV="" S_OUT="" S_OUT2="" S_ERR="" S_REF="" S_REF2="" S_JOURNAL="" S_RC=0

# The exact strings the gate scores, spelled once. Each is the printf format the code prints (moe-exec-policy.hpp,
# moe-stream-plan.cpp, ggml-sycl.cpp), not a name of the macro that prints it; the exception is named per regex below.
RE_LAYER='\[MOE-EXEC\] layer [0-9]+:'
RE_SUMMARY='\[MOE-EXEC\] summary:'
RE_SUMMARY_RING='\[MOE-EXEC\] summary: [0-9]+ layer-ubatches ring'
# A summary line that carries the window fields: the teardown line the window ring's counters extend.
RE_SUMMARY_WINDOWS='\[MOE-EXEC\] summary: .*, [0-9]+ windows, [0-9]+ exposed, [0-9]+ misses'
# The window fields alone, applied only to lines RE_SUMMARY_WINDOWS already matched, to read the three counts.
RE_WINDOW_FIELDS=', [0-9]+ windows, [0-9]+ exposed, [0-9]+ misses'
RE_LOAN_OK='\[MOE-EXEC\] loan device [0-9]+: [0-9]+ triplets'
RE_REFILL='\[MOE-EXEC\] refill device [0-9]+: [0-9]+/[0-9]+ triplets re-admitted'
# The bulk refill a prompt whose ring never engaged owes its loan (llama.cpp-i4ny), and its position: the first decode
# graph after the prompt. Its "R of L" never matches RE_REFILL's "R/L", so the teardown sums count no bulk line.
RE_BULK_REFILL='\[MOE-EXEC\] bulk refill device [0-9]+: [0-9]+ of [0-9]+ lent triplets re-admitted in one transaction'
RE_BULK_REFILL_FIRST="$RE_BULK_REFILL at decode graph 1 after the prompt"
RE_PP_REFUSED='\[PP-MOE\] refused device='
RE_PP_ARM_F16='\[PP-MOE\] device=[0-9]+ type=[^ ]+ layout=[^ ]+ arm=f16 '
RE_TRACE_EXEC='\[MOE-EXEC-TRACE\] executor device [0-9]+ tensor=[^ ]+ experts=[0-9]+ staged=[0-9]+ arm=[a-z0-9]+'
# The executor's per-layer trace line with its misses=N field (llama.cpp-8epx): the experts its windows did not have
# staged ahead. Every executor trace line carries the field, so this matches each one the trace prints.
RE_TRACE_MISSES='\[MOE-EXEC-TRACE\] executor device [0-9]+ .* misses=[0-9]+'
RE_WATCHDOG='SYCL-WATCHDOG\] No GPU progress'
RE_PLACE='MoE experts: [0-9]+ on device, [0-9]+ on host'
RE_TPS='[0-9.]+ tokens per second'
RE_RESETS='GT reset|guc_id|CAT error|Engine reset|exec queue reset'
RE_MI='1, 2, 3, 4, 5, 6, 7, 8, 9, 10'
# The forced decline: the change line's fixed format with reason "declined" (moe_exec_reason_name) for the declined
# layer, the same layer's ring and cpu change lines (the reference's decision), the summary's cpu count, the backend's
# KV refusal (ggml-sycl.cpp prints it after "[SYCL-PLAN] "), and mem_handle::resolve(int)'s wrong-device refusal
# (mem-handle.cpp, the only wrong-device warning in ggml-sycl; matched up to the text before its em dash).
RE_DECLINE="\\[MOE-EXEC\\] layer $DECLINE_LAYER: cpu \\(.* declined\\)"
# Any layer's decline line: the verdict prints how many distinct layers carry one (declined_layers, 0 or 1 for the
# one-layer knob).
RE_DECLINE_ANY='\[MOE-EXEC\] layer [0-9]+: cpu \(.* declined\)'
RE_DECLINE_RING="\\[MOE-EXEC\\] layer $DECLINE_LAYER: ring "
RE_DECLINE_CPU="\\[MOE-EXEC\\] layer $DECLINE_LAYER: cpu "
RE_SUMMARY_CPU='\[MOE-EXEC\] summary: [0-9]+ layer-ubatches ring, [0-9]+ cpu'
RE_KV_REJECTED='runtime KV update rejected'
RE_WRONG_DEVICE='mem_handle::resolve\(device_id=-?[0-9]+\): wrong-device access'

usage() {
    sed -n '2,/^set -o pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
}

# die_usage: a typo in a detached run must not leave its waiter hanging, so the final line is printed too.
die_usage() {
    echo "sycl-moe-exec-gate: $*" >&2
    say "REFUSED: usage: $*"
    say "verdicts_pass=0 GATE_DONE"
    exit 2
}

say() { printf '%s\n' "$*"; }

is_arm() {
    local a
    for a in "${ALL_ARMS[@]}"; do
        [ "$a" = "$1" ] && return 0
    done
    return 1
}

# need_value <argc> <option>: a value-taking option with no word after it is a usage error. Without the guard `shift 2`
# fails on one remaining word, shifts nothing, and the loop re-reads the same option forever.
need_value() {
    [ "$1" -ge 2 ] || die_usage "$2 needs a value"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h | --help) usage; exit 0 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --arm)
            is_arm "${2:-}" || die_usage "--arm '${2:-}' is not one of: ${ALL_ARMS[*]}"
            ARMS+=("$2"); ARMS_EXPLICIT=1; shift 2 ;;
        --make-prompt) MAKE_PROMPT=1; shift ;;
        --prompt-out) need_value "$#" "$1"; PROMPT_OUT="$2"; shift 2 ;;
        --score-arm)
            is_arm "${2:-}" || die_usage "--score-arm '${2:-}' is not one of: ${ALL_ARMS[*]}"
            SCORE_ARM="$2"; shift 2 ;;
        --csv) need_value "$#" "$1"; S_CSV="$2"; shift 2 ;;
        --out) need_value "$#" "$1"; S_OUT="$2"; shift 2 ;;
        --out2) need_value "$#" "$1"; S_OUT2="$2"; shift 2 ;;
        --err) need_value "$#" "$1"; S_ERR="$2"; shift 2 ;;
        --ref-out) need_value "$#" "$1"; S_REF="$2"; shift 2 ;;
        --ref-out2) need_value "$#" "$1"; S_REF2="$2"; shift 2 ;;
        --ref-err) need_value "$#" "$1"; S_REF_ERR="$2"; shift 2 ;;
        --journal) need_value "$#" "$1"; S_JOURNAL="$2"; shift 2 ;;
        --rc) need_value "$#" "$1"; S_RC="$2"; shift 2 ;;
        --rc-out2) need_value "$#" "$1"; S_RC_OUT2="$2"; shift 2 ;;
        --rc-ref) need_value "$#" "$1"; S_RC_REF="$2"; shift 2 ;;
        --rc-ref2) need_value "$#" "$1"; S_RC_REF2="$2"; shift 2 ;;
        --builders-pre) need_value "$#" "$1"; S_BPRE="$2"; shift 2 ;;
        --builders-post) need_value "$#" "$1"; S_BPOST="$2"; shift 2 ;;
        --src-dir) need_value "$#" "$1"; SRC_DIR="$2"; shift 2 ;;
        --clobber) CLOBBER=1; shift ;;
        --seed-verdicts) need_value "$#" "$1"; SEED_VERDICTS="$2"; shift 2 ;;
        --refuse-selftest) REFUSE_SELFTEST=1; shift ;;
        --wait-builders-selftest) WAIT_BUILDERS_SELFTEST=1; shift ;;
        --builders-post-selftest)
            is_arm "${2:-}" || die_usage "--builders-post-selftest '${2:-}' is not one of: ${ALL_ARMS[*]}"
            BUILDERS_POST_SELFTEST="$2"; shift 2 ;;
        --builders-file) need_value "$#" "$1"; BUILDERS_FILE="$2"; shift 2 ;;
        --builders-wait-max) need_value "$#" "$1"; BUILDERS_WAIT_MAX="$2"; shift 2 ;;
        --builders-poll) need_value "$#" "$1"; BUILDERS_POLL="$2"; shift 2 ;;
        --score-summary) need_value "$#" "$1"; SCORE_SUMMARY="$2"; shift 2 ;;
        --ring-arms) RING_ARMS_MODE=1; shift ;;
        --signal-selftest) SIGNAL_SELFTEST=1; shift ;;
        --check-controls) CHECK_CONTROLS=1; shift ;;
        --lock-selftest) LOCK_SELFTEST=1; shift ;;
        --build-dir) need_value "$#" "$1"; BUILD_DIR="$2"; shift 2 ;;
        --out-dir) need_value "$#" "$1"; OUT_DIR="$2"; shift 2 ;;
        --lock-dir) need_value "$#" "$1"; LOCK_DIR="$2"; shift 2 ;;
        --no-lock) TAKE_LOCK=0; shift ;;
        --no-head-check) HEAD_CHECK=0; shift ;;
        --since) need_value "$#" "$1"; SINCE="$2"; shift 2 ;;
        --qwen) need_value "$#" "$1"; QWEN="$2"; shift 2 ;;
        --gptoss) need_value "$#" "$1"; GPTOSS="$2"; shift 2 ;;
        --mistral) need_value "$#" "$1"; MISTRAL="$2"; shift 2 ;;
        --tuning-dir) need_value "$#" "$1"; TUNING_DIR="$2"; shift 2 ;;
        --decode-ref) need_value "$#" "$1"; DECODE_REF="$2"; shift 2 ;;
        --place-ref-tg) need_value "$#" "$1"; PLACE_REF_TG="$2"; shift 2 ;;
        --place-ref-noc) need_value "$#" "$1"; PLACE_REF_NOC="$2"; shift 2 ;;
        --qmd5) need_value "$#" "$1"; QMD5="$2"; shift 2 ;;
        --q-layers) need_value "$#" "$1"; Q_LAYERS="$2"; shift 2 ;;
        --go-noc-layers) need_value "$#" "$1"; GO_NOC_LAYERS="$2"; shift 2 ;;
        --pp512-min) need_value "$#" "$1"; PP512_MIN="$2"; shift 2 ;;
        --pp2048-min) need_value "$#" "$1"; PP2048_MIN="$2"; shift 2 ;;
        --go-pp-ref) need_value "$#" "$1"; GO_PP_REF="$2"; shift 2 ;;
        --go-tg-ref) need_value "$#" "$1"; GO_TG_REF="$2"; shift 2 ;;
        -*) die_usage "unknown option: $1" ;;
        *) [ -z "$EXPECT_SHA" ] || die_usage "more than one <sha>: $EXPECT_SHA and $1"; EXPECT_SHA="$1"; shift ;;
    esac
done
[ "${#ARMS[@]}" -gt 0 ] || ARMS=("${ALL_ARMS[@]}")

# A real run sources oneAPI here, after the options are parsed and before any variable is derived from them, because
# setvars.sh overwrites shell variables (see the caveat in the header). The modes that run nothing on a device do not
# source it, so --dry-run, --score-arm and the test need no oneAPI. A refusal still sources it first (about 2 s). The
# decision reads the parsed mode flags, never the raw words, so an option value that equals a mode name cannot skip it.
GATE_NEEDS_ONEAPI=1
modes="$DRY_RUN$MAKE_PROMPT$CHECK_CONTROLS$LOCK_SELFTEST$RING_ARMS_MODE"
modes="$modes$SIGNAL_SELFTEST$REFUSE_SELFTEST$WAIT_BUILDERS_SELFTEST"
if [ "$modes" != 00000000 ] || [ -n "$SCORE_ARM$SCORE_SUMMARY$BUILDERS_POST_SELFTEST" ]; then
    GATE_NEEDS_ONEAPI=0
fi

# require_tools <tool...>: a missing tool is a skip (77), never a pass: scoring through a missing awk or cmp would be a
# clean answer about nothing.
require_tools() {
    local t miss=""
    for t in "$@"; do
        command -v "$t" > /dev/null 2>&1 || miss="$miss $t"
    done
    if [ -n "$miss" ]; then
        say "SKIP: missing tool(s):$miss"
        say "verdicts_pass=0 GATE_DONE"
        exit 77
    fi
}
require_tools awk md5sum cmp comm
[ "$GATE_NEEDS_ONEAPI" = 0 ] || require_tools journalctl pgrep timeout git

GATE_ONEAPI=not_needed
if [ "$GATE_NEEDS_ONEAPI" = 1 ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
    GATE_ONEAPI=absent
    [ -z "${ONEAPI_ROOT:-}" ] || GATE_ONEAPI=sourced
fi

# Everything the gate reads from the tree under test comes from SRC_DIR, the source tree of the build: the parent of
# --build-dir unless --src-dir says otherwise. The prompts and this script come from where the script lives, so a
# snapshot copy of the script runs against another tree's build.
[ -n "$SRC_DIR" ] || SRC_DIR="$(dirname "$BUILD_DIR")"
PROMPT_DOC="$SRC_DIR/docs/backend/gpt-oss-testing.md"
GATE_BIN_DIR="$BUILD_DIR/bin"

# ---------------------------------------------------------------- scoring primitives (pure functions of files)

# count_re <regex> <file...>: how many lines match; prints 0 for no match or a missing file.
count_re() {
    local re="$1" n
    shift
    n="$(cat "$@" 2>/dev/null | grep -acE -- "$re")"
    echo "${n:-0}"
}

# distinct_layers <file...>: how many different layer ids carry a "[MOE-EXEC] layer N:" line. The line prints
# once per change of choice, so a re-decided layer prints again; distinct ids are the layers consulted.
distinct_layers() {
    cat "$@" 2>/dev/null | grep -aoE -- "$RE_LAYER" | sort -u | wc -l | tr -d ' '
}

# fcmp <a> <op> <b>: float compare through awk; op is one of < > <= >=.
fcmp() {
    awk -v a="$1" -v b="$3" "BEGIN { exit !(a $2 b) }"
}

# csv_row <csv> <n_prompt> <n_gen>: prints "avg_ts stddev_ts n_ubatch" of the last n_depth=0 row with those sizes,
# or "NA NA NA". Columns are found by header name; llama-bench quotes every field of a data row.
csv_row() {
    awk -F',' -v np="$2" -v ng="$3" '
        NR == 1 { for (i = 1; i <= NF; i++) { gsub(/"/, "", $i); h[$i] = i } next }
        {
            line = $0
            gsub(/^"|"$/, "", line)
            n = split(line, f, /","/)
            if (!(("n_prompt" in h) && ("n_gen" in h) && ("avg_ts" in h))) next
            if (f[h["n_prompt"]] == np && f[h["n_gen"]] == ng && (!("n_depth" in h) || f[h["n_depth"]] == 0)) {
                ub = ("n_ubatch" in h) ? f[h["n_ubatch"]] : "NA"
                sd = ("stddev_ts" in h) ? f[h["stddev_ts"]] : "NA"
                row = f[h["avg_ts"]] " " sd " " ub
            }
        }
        END { print (row == "" ? "NA NA NA" : row) }' "$1" 2>/dev/null
}

md5_8() { md5sum < "$1" | cut -c1-8; }

# tps <err-log>: the figure on the last " eval time = " line (not "prompt eval time"), or empty. The prompt-eval rate
# must never stand in for it: a nan eval line leaves no figure, and the arm then fails as no_tg.
tps() {
    cat "$1" 2>/dev/null | grep -a ' eval time = ' | grep -av 'prompt eval time' | tail -1 \
        | grep -aoE -- "$RE_TPS" | cut -d' ' -f1
}

# place_line <err-log>: the last "MoE experts: N on device, M on host" line, or empty.
place_line() {
    grep -aoE -- "$RE_PLACE" "$1" 2>/dev/null | tail -1
}

# verdict_of <items...>: "FAIL(a,b)" for a non-empty list, else "PASS".
verdict_of() {
    local IFS=','
    if [ $# -eq 0 ]; then
        echo "PASS"
    else
        echo "FAIL($*)"
    fi
}

# ring_layer_ubatches <file...>: the sum of K over every "[MOE-EXEC] summary: K layer-ubatches ring" line. A summary
# prints once per backend teardown, so a llama-bench with two -p values prints two.
ring_layer_ubatches() {
    cat "$@" 2>/dev/null | grep -aoE -- "$RE_SUMMARY_RING" | awk '{ n += $3 } END { print n + 0 }'
}

# ring_engaged <file...>: yes when the ring served at least one layer-ubatch. The "stream ring unavailable" WARN is
# not the signal: under auto at default verbosity the phase prints nothing of its own (it needs ring or the trace).
ring_engaged() { [ "$(ring_layer_ubatches "$@")" -gt 0 ] && echo yes || echo no; }

# summary_fields <file>: "windows exposed misses", each the sum of that window field over every "[MOE-EXEC] summary:"
# line, or "NA NA NA" when no summary line carries the window fields or any summary line lacks them (an older binary, or
# a mixed run). Summed, not read from the last line: a summary prints once per backend teardown (llama-bench makes one
# context per -p value), so a miss in the first context must not be hidden by a clean last one, as ring_layer_ubatches
# sums too. Only summary lines are read, so another line ending in the same fields counts for nothing.
summary_fields() {
    local lines win n_all n_win
    lines="$(cat "$1" 2>/dev/null | grep -aE -- "$RE_SUMMARY")"
    win="$(grep -aE -- "$RE_SUMMARY_WINDOWS" <<< "$lines")"
    n_all="$(grep -c . <<< "$lines")"
    n_win="$(grep -c . <<< "$win")"
    if [ "$n_win" -eq 0 ] || [ "$n_win" -ne "$n_all" ]; then
        echo "NA NA NA"
        return
    fi
    grep -aoE -- "$RE_WINDOW_FIELDS" <<< "$win" | awk '{ w += $2; e += $4; m += $6 } END { print w + 0, e + 0, m + 0 }'
}

# trace_misses <file>: the sum of misses=N over the executor trace lines, or NA when the log has none: the arm ran
# with the trace off (GGML_SYCL_MOE_EXEC_TRACE unset) or never dispatched the batched executor. NA is not a failure;
# only a sum above 0 is.
trace_misses() {
    cat "$1" 2>/dev/null | grep -aE -- "$RE_TRACE_MISSES" | grep -aoE ' misses=[0-9]+' | cut -d= -f2 \
        | awk '{ n += $1; c++ } END { if (c == 0) print "NA"; else print n + 0 }'
}

# trace_misses_item <value>: a failure item when the trace counted any miss; NA and 0 are clean.
trace_misses_item() { case "$1" in NA | 0) ;; *) echo "trace_misses=$1" ;; esac; }

# layers_item <label> <layers> <want>: a failure item when the distinct layer count is not what the arm needs. want
# is a number (exactly that many, 0 for none), "any" (at least one layer was consulted) or "-" (recorded, not scored).
layers_item() {
    case "$3" in
        -) ;;
        any) [ "$2" -ge 1 ] || echo "$1=$2<1" ;;
        *) [ "$2" -eq "$3" ] || echo "$1=$2!=$3" ;;
    esac
}
loan_exercised() { [ "$(count_re "$RE_LOAN_OK" "$@")" -gt 0 ] && echo yes || echo no; }

# refill_state <err-log>: balanced(R/L) | unbalanced(R/L) | not_exercised. R and L are summed over every
# "[MOE-EXEC] refill device D: R/L triplets re-admitted" line; one device short is enough to be unbalanced.
refill_state() {
    cat "$1" 2>/dev/null | grep -aoE -- "$RE_REFILL" | awk '
        { split($5, p, "/"); r += p[1]; l += p[2]; if (p[1] != p[2]) bad = 1; n++ }
        END {
            if (n == 0) print "not_exercised"
            else if (bad) printf "unbalanced(%d/%d)\n", r, l
            else printf "balanced(%d/%d)\n", r, l
        }'
}

# rc_item <rc>: "rc=N" when the arm's process did not exit 0.
rc_item() { [ "${1:-0}" = 0 ] || echo "rc=$1"; }

# ---------------------------------------------------------------- arm scorers: each prints one verdict line

# score_q_pp <csv> <err> <rc> [label] [pp512 row: score|skip]: the label is the arm's name (Q_pp, Q_pp_ub2048);
# skip is for a -p 2048 run, which has no pp512 row and answers to the pp2048 floor alone, with ub read from its row.
score_q_pp() {
    local csv="$1" err="$2" rc="$3" label="${4:-Q_pp}" row512="${5:-score}"
    local fails=() r512 r2048 pp512 sd512 ub pp2048 sd2048 ub2048
    read -r pp512 sd512 ub <<< "$(csv_row "$csv" 512 0)"
    read -r pp2048 sd2048 ub2048 <<< "$(csv_row "$csv" 2048 0)"
    [ "$row512" != skip ] || ub="$ub2048"
    [ -z "$(rc_item "$rc")" ] || fails+=("$(rc_item "$rc")")
    if [ "$row512" = skip ]; then
        :
    elif [ "$pp512" = NA ]; then
        fails+=("no_pp512_row")
    elif fcmp "$pp512" "<" "$PP512_MIN"; then
        fails+=("$(awk -v v="$pp512" -v m="$PP512_MIN" 'BEGIN { printf "pp512=%.0f<%s", v, m }')")
    fi
    if [ "$pp2048" = NA ]; then
        fails+=("no_pp2048_row")
    elif fcmp "$pp2048" "<" "$PP2048_MIN"; then
        fails+=("$(awk -v v="$pp2048" -v m="$PP2048_MIN" 'BEGIN { printf "pp2048=%.0f<%s", v, m }')")
    fi
    local layers summary refused item windows exposed misses tmiss ring_ub
    layers="$(distinct_layers "$err")"
    summary="$(count_re "$RE_SUMMARY" "$err")"
    refused="$(count_re "$RE_PP_REFUSED" "$err")"
    item="$(layers_item layers "$layers" "$Q_LAYERS")"
    [ -z "$item" ] || fails+=("$item")
    [ "$summary" -ge 1 ] || fails+=("summary=$summary<1")
    [ "$refused" -eq 0 ] || fails+=("pp_moe_refused=$refused")
    # The window ring: it must have engaged, the summary must carry the window fields (an older binary prints none) and
    # no window may have missed, in the summary (misses) or in the executor trace (trace_misses). The trace prints the
    # same misses the summary counts, so it is its own field and its own failure, never added in. These arms run with
    # the trace off, so trace_misses=NA is normal and not a failure.
    read -r windows exposed misses <<< "$(summary_fields "$err")"
    tmiss="$(trace_misses "$err")"
    ring_ub="$(ring_layer_ubatches "$err")"
    [ "$ring_ub" -gt 0 ] || fails+=("ring_engaged=no")
    if [ "$windows" = NA ]; then
        fails+=("summary_windows=missing")
    elif [ "$misses" -ne 0 ]; then
        fails+=("misses=$misses")
    fi
    item="$(trace_misses_item "$tmiss")"
    [ -z "$item" ] || fails+=("$item")
    r512="$(awk -v v="$pp512" 'BEGIN { if (v == "NA") print "NA"; else printf "%.1f", v }')"
    r2048="$(awk -v v="$pp2048" 'BEGIN { if (v == "NA") print "NA"; else printf "%.1f", v }')"
    echo "$label=$(verdict_of "${fails[@]}") pp512=$r512 sd512=$sd512 pp2048=$r2048" \
        "sd2048=$sd2048 ub=$ub layers=$layers" \
        "summary=$summary pp_moe_refused=$refused ring_engaged=$([ "$ring_ub" -gt 0 ] && echo yes || echo no)" \
        "ring_layer_ubatches=$ring_ub loan_exercised=$(loan_exercised "$err")" \
        "windows=$windows exposed=$exposed misses=$misses trace_misses=$tmiss"
}

score_q_pp_cpu() { # <csv> <err> <rc>
    local pp512 pp2048 _ st=RECORDED
    read -r pp512 _ <<< "$(csv_row "$1" 512 0)"
    read -r pp2048 _ <<< "$(csv_row "$1" 2048 0)"
    [ -z "$(rc_item "$3")" ] || st="FAIL($(rc_item "$3"))"
    echo "Q_pp_cpu=$st pp512=$pp512 pp2048=$pp2048 layers=$(distinct_layers "$2")" \
        "forced_cpu_lines=$(count_re '\[MOE-EXEC\] layer [0-9]+: cpu .*forced' "$2") rc=$3"
}

# ring_layer_ids <file...>: the layer ids that have a "[MOE-EXEC] layer N: ring" line, one per line, sorted. A layer
# that is ring in any ubatch counts as a host layer whose dispatches must be staged.
ring_layer_ids() {
    cat "$@" 2>/dev/null | grep -aoE '\[MOE-EXEC\] layer [0-9]+: ring ' | grep -oE ' [0-9]+:' | tr -d ' :' | sort -un
}

# trace_counts <ids> <file>: "total served unstaged nonf16" over the "[MOE-EXEC-TRACE] executor device" lines. total is
# every dispatch; served are those whose tensor is blk.N.* for an N in <ids> (whitespace separated); unstaged and
# nonf16 are counted over the served ones only, because resident layers run the executor too, with staged=0.
trace_counts() {
    local ids
    ids="$(echo "$1" | tr '\n' ' ')"
    cat "$2" 2>/dev/null | awk -v ring="$ids" '
        BEGIN { n = split(ring, a, " "); for (i = 1; i <= n; i++) R[a[i]] = 1 }
        match($0, /executor device [0-9]+ tensor=blk\.[0-9]+\.[^ ]* experts=[0-9]+ staged=[0-9]+ arm=[a-z0-9]+/) {
            split(substr($0, RSTART, RLENGTH), f, " ")
            t = f[4]; sub(/^tensor=blk\./, "", t); sub(/\..*/, "", t)
            st = f[6]; sub(/^staged=/, "", st)
            ar = f[7]; sub(/^arm=/, "", ar)
            total++
            if (t in R) { served++; if (st + 0 == 0) unstaged++; if (ar != "f16") nonf16++ }
        }
        END { printf "%d %d %d %d\n", total + 0, served + 0, unstaged + 0, nonf16 + 0 }'
}

score_q_staged() { # <err> <rc>
    local err="$1" rc="$2" fails=() ids nids total served unstaged nonf16 refused arms tmiss item
    if [ "$(ring_engaged "$err")" = no ]; then
        echo "Q_staged=NOT_EXERCISED(ring_engaged=no) rc=$rc"
        return
    fi
    [ -z "$(rc_item "$rc")" ] || fails+=("$(rc_item "$rc")")
    ids="$(ring_layer_ids "$err")"
    nids="$(echo "$ids" | grep -c .)"
    read -r total served unstaged nonf16 <<< "$(trace_counts "$ids" "$err")"
    arms="$(count_re "$RE_PP_ARM_F16" "$err")"
    refused="$(count_re "$RE_PP_REFUSED" "$err")"
    [ "$served" -gt 0 ] || fails+=("no_ring_layer_dispatch")
    [ "$unstaged" -eq 0 ] || fails+=("unstaged=$unstaged/$served")
    [ "$nonf16" -eq 0 ] || fails+=("arm_not_f16=$nonf16/$served")
    [ "$arms" -gt 0 ] || fails+=("no_pp_moe_arm_f16_line")
    [ "$refused" -eq 0 ] || fails+=("pp_moe_refused=$refused")
    tmiss="$(trace_misses "$err")"
    item="$(trace_misses_item "$tmiss")"
    [ -z "$item" ] || fails+=("$item")
    echo "Q_staged=$(verdict_of "${fails[@]}") ring_layers=$nids executor_dispatches=$total" \
        "ring_layer_dispatches=$served pp_moe_arm_f16_lines=$arms pp_moe_refused=$refused ring_engaged=yes" \
        "ring_layer_ubatches=$(ring_layer_ubatches "$err") trace_misses=$tmiss"
}

# score_q_tok <name> <out> <rc>: a recorded md5 of stdout.
# A crashed or empty load is a FAIL, not a recorded md5: the md5 of nothing is d41d8cd9.
score_q_tok() {
    local fails=()
    [ -z "$(rc_item "$3")" ] || fails+=("$(rc_item "$3")")
    [ -s "$2" ] || fails+=("empty_output")
    if [ "${#fails[@]}" -eq 0 ]; then
        echo "$1=RECORDED md5=$(md5_8 "$2") rc=$3"
    else
        echo "$1=$(verdict_of "${fails[@]}") md5=$(md5_8 "$2") rc=$3"
    fi
}

# summary_cpu <file>: the sum of C over every "[MOE-EXEC] summary: K layer-ubatches ring, C cpu" line, 0 when none.
summary_cpu() {
    cat "$1" 2>/dev/null | grep -aoE -- "$RE_SUMMARY_CPU" | awk '{ n += $6 } END { print n + 0 }'
}

# ref_layer_decision <err>: ring when the reference ran the declined layer on the ring in any micro-batch, else cpu when
# it printed a cpu line for it, else none (the layer had no host expert, or the phase never consulted it).
ref_layer_decision() {
    if [ "$(count_re "$RE_DECLINE_RING" "$1")" -gt 0 ]; then
        echo ring
    elif [ "$(count_re "$RE_DECLINE_CPU" "$1")" -gt 0 ]; then
        echo cpu
    else
        echo none
    fi
}

# declined_layers <file>: how many different layer ids carry a "[MOE-EXEC] layer N: cpu (... declined)" line.
declined_layers() {
    cat "$1" 2>/dev/null | grep -aoE -- "$RE_DECLINE_ANY" | grep -oE 'layer [0-9]+:' | sort -u | wc -l | tr -d ' '
}

# score_q_decline <out> <err> <rc> <Q_tok out> <Q_tok_2 out> <Q_tok err> <Q_tok rc> <Q_tok_2 rc>: the forced-decline
# proof (llama.cpp-ny6i, c-0blp). A missing, empty or failed auto load leaves nothing to compare: NOT_SCORED (that
# arm's own verdict carries the failure). The declined load must print the declined line and no ring line for the
# layer, count a cpu layer-ubatch in its summary, print the text of one of the two auto loads (the B70 greedy flip
# gives them two), and print no KV refusal and no wrong-device resolve. A real failure is a FAIL whatever Q_tok
# decided; a run that would pass is NOT_EXERCISED unless Q_tok ran the layer on the ring, since only then did the
# decline change the path.
score_q_decline() {
    local out="$1" err="$2" rc="$3" refs=("$4" "$5") referr="$6" refrcs=("$7" "$8") names=(Q_tok Q_tok_2)
    local fails=() i n ring cpu kv wdev md5 rmd5a rmd5b refl st prefix=""
    for i in 0 1; do
        if [ ! -e "${refs[$i]}" ]; then
            echo "Q_decline=NOT_SCORED(no_${names[$i]}_output) rc=$rc"
            return
        elif [ "${refrcs[$i]:-0}" != 0 ]; then
            echo "Q_decline=NOT_SCORED(rc_${names[$i]}=${refrcs[$i]}) rc=$rc"
            return
        elif [ ! -s "${refs[$i]}" ]; then
            echo "Q_decline=NOT_SCORED(empty_${names[$i]}_output) rc=$rc"
            return
        fi
    done
    [ -z "$(rc_item "$rc")" ] || fails+=("$(rc_item "$rc")")
    [ -s "$out" ] || fails+=("empty_output")
    n="$(count_re "$RE_DECLINE" "$err")"
    [ "$n" -ge 1 ] || fails+=("no_decline_line")
    ring="$(count_re "$RE_DECLINE_RING" "$err")"
    [ "$ring" -eq 0 ] || fails+=("declined_layer_ring=$ring")
    cpu="$(summary_cpu "$err")"
    [ "$cpu" -ge 1 ] || fails+=("summary_cpu=$cpu<1")
    md5="$(md5_8 "$out")"
    rmd5a="$(md5_8 "${refs[0]}")"
    rmd5b="$(md5_8 "${refs[1]}")"
    if [ "$md5" != "$rmd5a" ] && [ "$md5" != "$rmd5b" ]; then
        fails+=("md5=$md5!=$rmd5a,$rmd5b")
        prefix=" prefix_Q_tok=$(common_prefix "$out" "${refs[0]}") prefix_Q_tok_2=$(common_prefix "$out" "${refs[1]}")"
    fi
    kv="$(count_re "$RE_KV_REJECTED" "$err")"
    [ "$kv" -eq 0 ] || fails+=("kv_rejected=$kv")
    wdev="$(count_re "$RE_WRONG_DEVICE" "$err")"
    [ "$wdev" -eq 0 ] || fails+=("wrong_device=$wdev")
    refl="$(ref_layer_decision "$referr")"
    st="$(verdict_of "${fails[@]}")"
    [ "$st" != PASS ] || [ "$refl" = ring ] || st="NOT_EXERCISED(ref_layer$DECLINE_LAYER=$refl)"
    echo "Q_decline=$st decline_lines=$n declined_layers=$(declined_layers "$err") summary_cpu=$cpu md5=$md5" \
        "ref_md5=$rmd5a,$rmd5b ref_layer$DECLINE_LAYER=$refl$prefix" \
        "layers=$(distinct_layers "$err") kv_rejected=$kv wrong_device=$wdev rc=$rc"
}

# first_diff_byte <a> <b>: the 1-based byte where two files first differ, "eof" when one is a prefix of the other.
first_diff_byte() {
    local r
    r="$(LC_ALL=C cmp "$1" "$2" 2>&1 | head -1)"
    case "$r" in
        *"differ: byte "*) r="${r#*differ: byte }"; echo "${r%%,*}" ;;
        *"differ: char "*) r="${r#*differ: char }"; echo "${r%%,*}" ;; # the C locale says char, the others byte
        *EOF*) echo eof ;;
        *) echo none ;;
    esac
}

# common_prefix <a> <b>: how many leading bytes two files share (0 when either is missing).
common_prefix() {
    local d sa sb
    [ -f "$1" ] && [ -f "$2" ] || { echo 0; return; }
    sa="$(wc -c < "$1")"
    sb="$(wc -c < "$2")"
    d="$(first_diff_byte "$1" "$2")"
    case "$d" in
        eof) [ "$sa" -le "$sb" ] && echo "$sa" || echo "$sb" ;;
        none) echo "$sa" ;;
        *) echo $((d - 1)) ;;
    esac
}

# score_q_tok_cpu <cpu load 2 out> <cpu load 1 out> <auto load 1 out> <auto load 2 out> <rc of the first> <rc of the
# second> <rc of the third> <rc of the fourth>. Qwen3.8 greedy decoding on the B70 has a near-tie that flips from load
# to load, in both policies, and the flip cascades after its token: four loads gave three distinct outputs. Two loads
# per arm cannot enumerate it, so the rule is: PASS when at least one auto md5 equals at least one cpu md5 (identical
# text across the two policies is the evidence the math agrees) AND every auto output is either identical to some cpu
# output or shares a prefix of at least TOK_PREFIX_MIN bytes with its best-matching cpu output. A defect shows as
# pairs_equal=0 with an early min_prefix, a flip as pairs_equal >= 1 with a late one. stdout only: the log goes to
# stderr. All four outputs must be non-empty and all four loads must have exited 0: three crashed loads and one good
# one would otherwise agree on the md5 of nothing.
score_q_tok_cpu() {
    local cb="$1" ca="$2" aa="$3" ab="$4" fails=() ma mb mca mcb k f g m best p min_prefix="" ident
    local rcs=("$5" "$6" "$7" "$8") names=(Q_tok_cpu Q_tok_cpu_1 Q_tok Q_tok_2) files=("$cb" "$ca" "$aa" "$ab") i
    local flips_a=no flips_c=no pairs_equal=0 distinct_a=2 distinct_c=2
    for i in 1 2 3; do
        [ -e "${files[$i]}" ] || { echo "Q_tok_cpu=NOT_SCORED(no_${names[$i]}_output)"; return; }
    done
    for i in 0 1 2 3; do
        if [ "$i" = 0 ]; then # this arm's own load keeps the plain item names
            [ "${rcs[0]:-0}" = 0 ] || fails+=("rc=${rcs[0]}")
            [ -s "${files[0]}" ] || fails+=("empty_output")
        else
            [ "${rcs[$i]:-0}" = 0 ] || fails+=("rc_${names[$i]}=${rcs[$i]}")
            [ -s "${files[$i]}" ] || fails+=("empty_${names[$i]}_output")
        fi
    done
    ma="$(md5_8 "$aa")"; mb="$(md5_8 "$ab")"; mca="$(md5_8 "$ca")"; mcb="$(md5_8 "$cb")"
    [ "$ma" = "$mb" ] && { flips_a=no; distinct_a=1; } || flips_a=yes
    [ "$mca" = "$mcb" ] && { flips_c=no; distinct_c=1; } || flips_c=yes
    for m in "$ma" "$mb"; do
        [ "$m" != "$mca" ] || pairs_equal=$((pairs_equal + 1))
        [ "$m" != "$mcb" ] || pairs_equal=$((pairs_equal + 1))
    done
    k=0
    for f in "$aa" "$ab"; do
        k=$((k + 1))
        m="$(md5_8 "$f")"
        best=0
        for g in "$ca" "$cb"; do
            p="$(common_prefix "$f" "$g")"
            [ "$p" -le "$best" ] || best="$p"
        done
        [ -z "$min_prefix" ] || [ "$best" -ge "$min_prefix" ] || min_prefix="$best"
        [ -n "$min_prefix" ] || min_prefix="$best"
        ident=no
        { [ "$m" = "$mca" ] || [ "$m" = "$mcb" ]; } && ident=yes
        if [ "$ident" = no ] && [ "$best" -lt "$TOK_PREFIX_MIN" ]; then
            fails+=("auto${k}_prefix=$best<$TOK_PREFIX_MIN")
        fi
    done
    [ "$pairs_equal" -ge 1 ] || fails+=("pairs_equal=0")
    echo "Q_tok_cpu=$(verdict_of "${fails[@]}") distinct_auto=$distinct_a distinct_cpu=$distinct_c" \
        "pairs_equal=$pairs_equal min_prefix=$min_prefix auto_md5=$ma,$mb cpu_md5=$mca,$mcb" \
        "auto_flip=$flips_a cpu_flip=$flips_c"
}

# score_decode <name> <out> <err> <rc> <floor> <place-ref>: Q_tg and Q_noc, the q0xu decode arms. The output md5 equal
# to the reference is a PASS. A different md5 is a greedy near-tie flip (flip=yes, still a PASS) when the output shares
# at least DECODE_FLIP_PREFIX_MIN bytes with the reference TEXT, and a FAIL (flip=no) below that.
score_decode() {
    local name="$1" out="$2" err="$3" rc="$4" floor="$5" pref="$6" fails=() tg md5 wd place refill
    [ -z "$(rc_item "$rc")" ] || fails+=("$(rc_item "$rc")")
    tg="$(tps "$err")"
    md5="$(md5_8 "$out")"
    wd="$(count_re "$RE_WATCHDOG" "$err")"
    place="$(place_line "$err")"
    refill="$(refill_state "$err")"
    if [ -z "$tg" ]; then
        fails+=("no_tg")
    elif fcmp "$tg" "<" "$floor"; then
        fails+=("tg=$tg<$floor")
    fi
    local flip=no prefix
    if [ "$md5" = "$QMD5" ]; then
        prefix="$(wc -c < "$out" 2>/dev/null || echo 0)"
    elif [ ! -f "$DECODE_REF" ]; then
        prefix=0
        fails+=("md5=$md5!=$QMD5,no_reference_text")
    else
        prefix="$(common_prefix "$out" "$DECODE_REF")"
        if [ "$prefix" -ge "$DECODE_FLIP_PREFIX_MIN" ]; then
            flip=yes
        else
            fails+=("md5=$md5!=$QMD5,flip=no(prefix=$prefix<$DECODE_FLIP_PREFIX_MIN)")
        fi
    fi
    [ "$wd" -eq 0 ] || fails+=("watchdog=$wd")
    local place_state="unscored"
    if [ "$pref" != "-" ]; then
        if [ "$place" = "$pref" ]; then
            place_state="unchanged"
        else
            place_state="changed"
            fails+=("place='${place:-none}'!='$pref'")
        fi
    fi
    case "$refill" in unbalanced*) fails+=("refill=$refill") ;; esac
    # A loan the prompt's ring never used is returned at the first decode graph, in one transaction (llama.cpp-i4ny).
    local loan ring bulk=not_owed
    loan="$(loan_exercised "$err")"
    ring="$(ring_engaged "$err")"
    if [ "$(count_re "$RE_BULK_REFILL_FIRST" "$err")" -gt 0 ]; then
        bulk=first
    elif [ "$loan" = yes ] && [ "$ring" = no ]; then
        bulk=late
        fails+=("refill_late")
    fi
    echo "${name}=$(verdict_of "${fails[@]}") tg=${tg:-NA} md5=$md5 flip=$flip prefix=${prefix:-0}" \
        "place=$place_state refill=$refill" \
        "loan_exercised=$loan ring_engaged=$ring bulk_refill=$bulk"
}

# score_go <name> <out> <err> <rc> <expected-layer-lines>: the GPT-OSS count gate.
score_go() {
    local name="$1" out="$2" err="$3" rc="$4" want="$5" fails=() count layers item
    [ -z "$(rc_item "$rc")" ] || fails+=("$(rc_item "$rc")")
    count="$(cat "$out" 2>/dev/null | grep -acx '1, 2, 3, 4, 5')"
    layers="$(distinct_layers "$out" "$err")"
    [ "${count:-0}" -eq 1 ] || fails+=("count=${count:-0}!=1")
    item="$(layers_item layer_lines "$layers" "$want")"
    [ -z "$item" ] || fails+=("$item")
    echo "${name}=$(verdict_of "${fails[@]}") count=${count:-0} layer_lines=$layers"
}

# within_item <label> <value> <ref> <tol-percent>: a failure item when value is outside ref +- tol %.
within_item() {
    awk -v l="$1" -v v="$2" -v r="$3" -v t="$4" 'BEGIN {
        lo = r * (1 - t / 100); hi = r * (1 + t / 100)
        if (v == "NA") printf "no_%s_row", l
        else if (v < lo) printf "%s=%.1f<%.1f", l, v, lo
        else if (v > hi) printf "%s=%.1f>%.1f", l, v, hi
    }'
}

score_go_pp() { # <csv> <rc>
    local csv="$1" rc="$2" fails=() pp sdp tg sdt _ item
    read -r pp sdp _ <<< "$(csv_row "$csv" 512 0)"
    read -r tg sdt _ <<< "$(csv_row "$csv" 0 128)"
    [ -z "$(rc_item "$rc")" ] || fails+=("$(rc_item "$rc")")
    item="$(within_item pp512 "$pp" "$GO_PP_REF" "$GO_PP_TOL")"
    [ -z "$item" ] || fails+=("$item")
    # llama.cpp-zuze: master reads pp512 about 770 on the B50 against the 871 floor. A pp512 below the window but
    # within 15 % below the reference is that known regression, and the verdict says so.
    item="$(awk -v v="$pp" -v r="$GO_PP_REF" -v t="$GO_PP_TOL" 'BEGIN {
        if (v != "NA" && v < r * (1 - t / 100) && v >= r * 0.85) print "see=llama.cpp-zuze(master_pp512_regression)"
    }')"
    [ -z "$item" ] || fails+=("$item")
    item="$(within_item tg128 "$tg" "$GO_TG_REF" "$GO_TG_TOL")"
    [ -z "$item" ] || fails+=("$item")
    echo "GO_pp=$(verdict_of "${fails[@]}") pp512=$pp sd512=$sdp tg128=$tg sd128=$sdt" \
        "ref_pp512=$GO_PP_REF ref_tg128=$GO_TG_REF"
}

score_mi() { # <name> <out> <rc>
    local name="$1" out="$2" rc="$3" fails=() n
    [ -z "$(rc_item "$rc")" ] || fails+=("$(rc_item "$rc")")
    n="$(cat "$out" 2>/dev/null | grep -ac -- "$RE_MI")"
    [ "${n:-0}" -eq 1 ] || fails+=("digits=${n:-0}!=1")
    echo "${name}=$(verdict_of "${fails[@]}") digits=${n:-0}"
}

# score_resets <journal-text> <journalctl rc>: an empty journal is a pass only when journalctl itself succeeded; a
# missing file or a failed journalctl is a FAIL, never a clean log.
score_resets() {
    local n fails=()
    [ -f "$1" ] || fails+=("no_journal")
    [ -z "$(rc_item "$2")" ] || fails+=("journalctl_$(rc_item "$2")")
    n="$(cat "$1" 2>/dev/null | grep -aciE -- "$RE_RESETS")"
    [ "${n:-0}" -eq 0 ] || fails+=("count=$n")
    echo "resets=$(verdict_of "${fails[@]}") count=${n:-0}"
}

# score_arm_raw <arm> <csv> <out> <err> <ref-out> <journal> <rc> <out2> <ref-out2>; the rcs of the recorded Q_tok arms
# come from S_RC_OUT2 (Q_tok_cpu_1), S_RC_REF (Q_tok) and S_RC_REF2 (Q_tok_2). Q_decline reads the same <ref-out>,
# <ref-out2> and rcs, plus S_REF_ERR, Q_tok's log.
score_arm_raw() {
    local arm="$1" csv="$2" out="$3" err="$4" ref="$5" journal="$6" rc="$7" out2="${8:-}" ref2="${9:-}"
    case "$arm" in
        Q_pp) score_q_pp "$csv" "$err" "$rc" "$arm" ;;
        Q_pp_ub2048) score_q_pp "$csv" "$err" "$rc" "$arm" skip ;;
        Q_pp_cpu) score_q_pp_cpu "$csv" "$err" "$rc" ;;
        Q_staged) score_q_staged "$err" "$rc" ;;
        Q_tok | Q_tok_2 | Q_tok_cpu_1) score_q_tok "$arm" "$out" "$rc" ;;
        Q_decline) score_q_decline "$out" "$err" "$rc" "$ref" "$ref2" "$S_REF_ERR" "$S_RC_REF" "$S_RC_REF2" ;;
        Q_tok_cpu) score_q_tok_cpu "$out" "$out2" "$ref" "$ref2" "$rc" "$S_RC_OUT2" "$S_RC_REF" "$S_RC_REF2" ;;
        Q_tg) score_decode Q_tg "$out" "$err" "$rc" "$QTG_MIN" "$PLACE_REF_TG" ;;
        Q_noc) score_decode Q_noc "$out" "$err" "$rc" "$QNOC_MIN" "$PLACE_REF_NOC" ;;
        GO_gate) score_go GO_gate "$out" "$err" "$rc" 0 ;;
        GO_noc) score_go GO_noc "$out" "$err" "$rc" "$GO_NOC_LAYERS" ;;
        GO_pp) score_go_pp "$csv" "$rc" ;;
        MI_b70 | MI_b50) score_mi "$arm" "$out" "$rc" ;;
        resets) score_resets "$journal" "$rc" ;;
    esac
}

# decorate_verdict <line> <builders pre> <builders post>: every verdict line carries builders_pre= and builders_post=
# ("na" when the run recorded none). A build running during the arm depresses throughput and flips CPU-path timing, so
# a PASS with builders_post above 0 becomes NOT_SCORED(tainted=builders,was=PASS), which is exit class 3. A FAIL stays
# a FAIL (a taint cannot make a real failure go away) and a RECORDED line stays recorded; both carry the flag.
decorate_verdict() {
    local line="$1" pre="$2" post="$3" name st rest
    if [[ "$post" =~ ^[0-9]+$ ]] && [ "$post" -gt 0 ]; then
        name="${line%%=*}"
        st="$(status_of "$line")"
        rest="${line#* }"
        if [ "$st" = PASS ]; then
            line="$name=NOT_SCORED(tainted=builders,was=PASS) $rest"
        fi
        line="$line builders_pre=$pre builders_post=$post tainted=builders"
    else
        line="$line builders_pre=$pre builders_post=$post"
    fi
    echo "$line"
}

# score_arm_files: the same arguments as score_arm_raw, plus the builders sampled around the arm (S_BPRE, S_BPOST).
score_arm_files() {
    decorate_verdict "$(score_arm_raw "$@")" "$S_BPRE" "$S_BPOST"
}

# status_of <verdict line>: PASS | FAIL | RECORDED | NOT_EXERCISED | NOT_SCORED
status_of() {
    local w="${1%% *}"
    w="${w#*=}"
    echo "${w%%(*}"
}

# ---------------------------------------------------------------- grep controls

# One positive line that must match and one decoy that must not, per regex. The decoys are the nearest real strings
# the same code prints: a refused loan for the loan regex, the trace line for the layer regex, and so on.
check_controls() {
    local bad=0 re_name pos neg re
    ctl() { # <name> <regex> <positive line> <decoy line>
        re_name="$1"; re="$2"; pos="$3"; neg="$4"
        if ! printf '%s\n' "$pos" | grep -qE -- "$re"; then
            echo "control FAIL: $re_name does not match its positive line: $pos" >&2
            bad=1
        fi
        if printf '%s\n' "$neg" | grep -qE -- "$re"; then
            echo "control FAIL: $re_name matches its decoy line: $neg" >&2
            bad=1
        fi
    }
    ctl layer "$RE_LAYER" '[MOE-EXEC] layer 7: ring (p=0.99 t_cpu=900 us t_ring=300 us cheaper)' \
        '[MOE-EXEC-TRACE] dev 0 layer 7: ring (p=0.99 t_cpu=900 us t_ring=300 us cheaper)'
    ctl summary "$RE_SUMMARY" '[MOE-EXEC] summary: 96 layer-ubatches ring, 0 cpu, 0 forced, 1500 MiB streamed' \
        '[MOE-EXEC] layer 7: ring (p=0.99)'
    local sum_long='[MOE-EXEC] summary: 24 layer-ubatches ring, 0 cpu, 0 forced, 41000 MiB streamed,'
    sum_long+=' 216 windows, 24 exposed, 0 misses'
    local sum_short='[MOE-EXEC] summary: 24 layer-ubatches ring, 0 cpu, 0 forced, 41000 MiB streamed'
    ctl summary_ring "$RE_SUMMARY_RING" "$sum_long" \
        '[MOE-EXEC] layer 7: ring (p=0.99 t_cpu=900 us t_ring=300 us cheaper)'
    ctl summary_windows "$RE_SUMMARY_WINDOWS" "$sum_long" "$sum_short"
    ctl summary_windows_trace_decoy "$RE_SUMMARY_WINDOWS" "$sum_long" \
        '[MOE-EXEC-TRACE] executor device 0 tensor=blk.0.x experts=8 staged=8 arm=f16, 5 windows, 0 exposed, 7 misses'
    ctl window_fields "$RE_WINDOW_FIELDS" ', 216 windows, 24 exposed, 0 misses' ', 216 windows, 24 exposed'
    ctl trace_misses "$RE_TRACE_MISSES" \
        '[MOE-EXEC-TRACE] executor device 0 tensor=blk.0.ffn_up_exps.weight experts=480 staged=300 arm=f16 misses=0' \
        '[MOE-EXEC-TRACE] stage layer 3 ubatch 512: regime=STREAM_ALL misses=2'
    ctl loan_ok "$RE_LOAN_OK" '[MOE-EXEC] loan device 0: 24 triplets, 300 MiB lent to the ring (ok)' \
        '[MOE-EXEC] loan device 0 refused: host storage too small'
    ctl refill "$RE_REFILL" '[MOE-EXEC] refill device 0: 24/24 triplets re-admitted' \
        '[MOE-EXEC] refill device 0: layer 3 expert 5 stays on the host: busy'
    local bulk='[MOE-EXEC] bulk refill device 0: 9 of 9 lent triplets re-admitted in one transaction'
    bulk+=' at decode graph 1 after the prompt (its ring did not engage)'
    ctl refill_not_bulk "$RE_REFILL" '[MOE-EXEC] refill device 0: 9/9 triplets re-admitted' "$bulk"
    ctl bulk_refill_first "$RE_BULK_REFILL_FIRST" "$bulk" "${bulk/decode graph 1 after/decode graph 12 after}"
    ctl bulk_refill "$RE_BULK_REFILL" \
        '[MOE-EXEC] bulk refill device 0: 3 of 3 lent triplets re-admitted in one transaction between contexts' \
        '[MOE-EXEC] refill device 0: 3/3 triplets re-admitted'
    ctl pp_refused "$RE_PP_REFUSED" \
        '[PP-MOE] refused device=0 cause=x detail=- tensor=blk.0.ffn_up_exps.weight type=q4_0 layout=aos staged=4' \
        '[PP-MOE] device=0 type=q4_0 layout=aos arm=f16 experts=8 groups=1 planned_weight_slot=100'
    ctl pp_arm "$RE_PP_ARM_F16" \
        '[PP-MOE] device=0 type=q4_0 layout=aos arm=f16 experts=8 groups=1 planned_weight_slot=100' \
        '[PP-MOE] device=0 type=q4_0 layout=aos arm=woq experts=8 groups=1 planned_weight_slot=100'
    ctl trace_exec "$RE_TRACE_EXEC" \
        '[MOE-EXEC-TRACE] executor device 0 tensor=blk.3.ffn_up_exps.weight experts=8 staged=8 arm=f16' \
        '[MOE-EXEC-TRACE] stage layer 3 ubatch 512: regime=STREAM_ALL p=0.99'
    ctl watchdog "$RE_WATCHDOG" '[SYCL-WATCHDOG] No GPU progress for 30 s' '[SYCL-WATCHDOG] armed'
    ctl place "$RE_PLACE" '[PLACEMENT] MoE experts: 38004 on device, 35724 on host (of 73728 total)' \
        '[PLACEMENT] MoE experts: all on device'
    local tps_head='llama_perf_context_print:        eval time =    4954.02 ms /    59 runs'
    ctl tps "$RE_TPS" "$tps_head   (   83.97 ms per token,    11.91 tokens per second)" "$tps_head"
    ctl resets "$RE_RESETS" 'xe 0000:03:00.0: [drm] GT reset started' 'xe 0000:03:00.0: [drm] GT0: suspended'
    ctl mi "$RE_MI" '1, 2, 3, 4, 5, 6, 7, 8, 9, 10' '1, 2, 3, 4, 5,'
    local dec_tail='(p=0.98 t_cpu=52000 us t_ring=9000 us declined)'
    ctl decline "$RE_DECLINE" "[MOE-EXEC] layer $DECLINE_LAYER: cpu $dec_tail" \
        "[MOE-EXEC-TRACE] dev 0 layer $DECLINE_LAYER: cpu $dec_tail"
    ctl decline_other_layer "$RE_DECLINE" "[MOE-EXEC] layer $DECLINE_LAYER: cpu $dec_tail" \
        "[MOE-EXEC] layer 1$DECLINE_LAYER: cpu $dec_tail"
    ctl decline_reason "$RE_DECLINE" "[MOE-EXEC] layer $DECLINE_LAYER: cpu $dec_tail" \
        "[MOE-EXEC] layer $DECLINE_LAYER: cpu (p=0.98 t_cpu=52000 us t_ring=9000 us ring-unavailable)"
    ctl decline_any "$RE_DECLINE_ANY" "[MOE-EXEC] layer 1$DECLINE_LAYER: cpu $dec_tail" \
        "[MOE-EXEC-TRACE] dev 0 layer 1$DECLINE_LAYER: cpu $dec_tail"
    # The ring and cpu lines decide ref_layer (a ring line turns NOT_EXERCISED into PASS), so each also refuses
    # another layer's line and the trace line.
    local ring_tail='(p=0.98 t_cpu=52000 us t_ring=9000 us cheaper)'
    ctl decline_ring "$RE_DECLINE_RING" "[MOE-EXEC] layer $DECLINE_LAYER: ring $ring_tail" \
        "[MOE-EXEC] layer $DECLINE_LAYER: cpu $dec_tail"
    ctl decline_ring_other_layer "$RE_DECLINE_RING" "[MOE-EXEC] layer $DECLINE_LAYER: ring $ring_tail" \
        "[MOE-EXEC] layer 1$DECLINE_LAYER: ring $ring_tail"
    ctl decline_ring_trace "$RE_DECLINE_RING" "[MOE-EXEC] layer $DECLINE_LAYER: ring $ring_tail" \
        "[MOE-EXEC-TRACE] dev 0 layer $DECLINE_LAYER: ring $ring_tail"
    ctl decline_cpu "$RE_DECLINE_CPU" "[MOE-EXEC] layer $DECLINE_LAYER: cpu $dec_tail" \
        "[MOE-EXEC-TRACE] dev 0 layer $DECLINE_LAYER: cpu $dec_tail"
    ctl decline_cpu_other_layer "$RE_DECLINE_CPU" "[MOE-EXEC] layer $DECLINE_LAYER: cpu $dec_tail" \
        "[MOE-EXEC] layer 1$DECLINE_LAYER: cpu $dec_tail"
    ctl summary_cpu "$RE_SUMMARY_CPU" "$sum_long" '[MOE-EXEC] summary: 24 layer-ubatches ring'
    ctl kv_rejected "$RE_KV_REJECTED" '[SYCL-PLAN] runtime KV update rejected: per-device budget exceeded --' \
        'W [SYCL-PLAN] KV overflow re-placed to host tier: 9 layer(s) demoted'
    ctl wrong_device "$RE_WRONG_DEVICE" \
        'mem_handle::resolve(device_id=0): wrong-device access — handle owns device 1 (kind=2). Returning null.' \
        '[SYCL-STORAGE]   blk.3.ffn_up_exps.weight: data_handle[0] valid handle_device=1 resolve=0 on_device=1'
    # The count scorer is grep -cx, not grep -c: the echo line ends with the answer (CLAUDE.md, GPT-OSS gate).
    local echo_line='> Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5'
    [ "$(printf '%s\n' "$echo_line" | grep -cx '1, 2, 3, 4, 5')" = 0 ] ||
        { echo "control FAIL: echo line scores" >&2; bad=1; }
    [ "$(printf '%s\n1, 2, 3, 4, 5\n' "$echo_line" | grep -cx '1, 2, 3, 4, 5')" = 1 ] ||
        { echo "control FAIL: answer line" >&2; bad=1; }
    [ "$(printf '%s\n1, 2, 3, 4, 5\n' "$echo_line" | grep -c '1, 2, 3, 4, 5')" = 2 ] ||
        { echo "control FAIL: loose grep premise" >&2; bad=1; }
    return "$bad"
}

# ---------------------------------------------------------------- arm definitions

# arm_setup <arm> sets A_KIND (bench|completion|cli|journal), A_BIN, A_SEL, A_TMO, A_ENV[] and A_ARGV[].
arm_setup() {
    local arm="$1"
    A_KIND=completion A_BIN="" A_SEL="level_zero:0" A_TMO=900
    A_ENV=()
    A_ARGV=()
    local gpt_cli=(-st --simple-io --no-display-prompt --chat-template-kwargs '{"reasoning_effort":"medium"}'
        --reasoning-format none --reasoning-budget 0 -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5'
        -n 48 --seed 42 --temp 0)
    local q_decode=(-m "$QWEN" -f "$DECODE_PROMPT" -n 60 --seed 42 --temp 0 -no-cnv -ub 256
                    --no-display-prompt -lv 4 -t 4)
    local q_tok=(-m "$QWEN" -f "$TOK_PROMPT" -n 32 --seed 42 --temp 0 -no-cnv --no-display-prompt -lv 4 -t 4)
    case "$arm" in
        Q_pp) A_KIND=bench A_BIN=llama-bench A_TMO=1500; A_ARGV=(-m "$QWEN" -p '512,2048' -n 0 -r 3 -fa 1 -o csv -v) ;;
        Q_pp_ub2048) A_KIND=bench A_BIN=llama-bench A_TMO=1500
            A_ARGV=(-m "$QWEN" -ub 2048 -p 2048 -n 0 -r 3 -fa 1 -o csv -v) ;;
        Q_pp_cpu) A_KIND=bench A_BIN=llama-bench A_TMO=1500; A_ENV=(GGML_SYCL_MOE_EXEC=cpu)
            A_ARGV=(-m "$QWEN" -p '512,2048' -n 0 -r 3 -fa 1 -o csv -v) ;;
        Q_staged) A_KIND=bench A_BIN=llama-bench; A_ENV=(GGML_SYCL_MOE_EXEC_TRACE=1)
            A_ARGV=(-m "$QWEN" -p 512 -n 0 -r 1 -fa 1 -o csv -v) ;;
        Q_tok | Q_tok_2) A_BIN=llama-completion; A_ARGV=("${q_tok[@]}") ;;
        Q_tok_cpu_1 | Q_tok_cpu) A_BIN=llama-completion; A_ENV=(GGML_SYCL_MOE_EXEC=cpu); A_ARGV=("${q_tok[@]}") ;;
        Q_tg) A_BIN=llama-completion; A_ARGV=("${q_decode[@]}" -c 4096) ;;
        Q_noc) A_BIN=llama-completion; A_ARGV=("${q_decode[@]}") ;;
        Q_decline) A_BIN=llama-completion; A_ENV=("GGML_SYCL_MOE_EXEC_DECLINE_LAYER=$DECLINE_LAYER")
            A_ARGV=("${q_tok[@]}") ;;
        GO_gate)
            A_KIND=cli A_BIN=llama-cli A_SEL=level_zero:1 A_TMO=600
            A_ARGV=(-m "$GPTOSS" -c 4096 "${gpt_cli[@]}") ;;
        GO_noc) A_KIND=cli A_BIN=llama-cli A_SEL=level_zero:1 A_TMO=600; A_ARGV=(-m "$GPTOSS" "${gpt_cli[@]}") ;;
        GO_pp)
            A_KIND=bench A_BIN=llama-bench A_SEL=level_zero:1 A_TMO=900
            A_ARGV=(-m "$GPTOSS" -p 512 -n 128 -r 3 -o csv) ;;
        MI_b70) A_BIN=llama-completion A_SEL=level_zero:0 A_TMO=300
            A_ARGV=(-m "$MISTRAL" -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0) ;;
        MI_b50) A_BIN=llama-completion A_SEL=level_zero:1 A_TMO=300
            A_ARGV=(-m "$MISTRAL" -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0) ;;
        resets) A_KIND=journal ;;
    esac
}

arm_doc() {
    case "$1" in
        Q_pp)
            echo "pp512 >= $PP512_MIN, pp2048 >= $PP2048_MIN, $Q_LAYERS distinct '$RE_LAYER' lines," \
                "at least one '$RE_SUMMARY', zero '$RE_PP_REFUSED'; ring_engaged=yes (the '$RE_SUMMARY_RING' lines" \
                "add up to > 0), the window fields on every summary line and zero misses (summary and" \
                "'$RE_TRACE_MISSES'); reports loan_exercised ('$RE_LOAN_OK') and windows exposed misses" ;;
        Q_pp_ub2048)
            echo "pp2048 >= $PP2048_MIN at -ub 2048 (no pp512 row, no pp512 floor); the same ring, window and miss" \
                "scoring as Q_pp" ;;
        Q_pp_cpu) echo "recorded only: pp512, pp2048, forced cpu layer lines" ;;
        Q_staged)
            echo "every '$RE_TRACE_EXEC' line of a layer with a '[MOE-EXEC] layer N: ring' line has" \
                "staged > 0 and arm=f16 (resident layers are ignored); at least one '$RE_PP_ARM_F16';" \
                "zero '$RE_PP_REFUSED'; trace_misses (the sum of misses= on the trace lines, NA when none carries it)" \
                "is 0 or NA; NOT_EXERCISED when the ring is unavailable" ;;
        Q_tok) echo "recorded only: md5 of stdout, load 1 of 2" ;;
        Q_tok_2) echo "recorded only: md5 of stdout, load 2 of 2" ;;
        Q_tok_cpu_1) echo "recorded only: md5 of stdout with GGML_SYCL_MOE_EXEC=cpu, load 1 of 2" ;;
        Q_tok_cpu)
            echo "cpu load 2 of 2; PASS when some auto md5 equals some cpu md5 and every auto output is identical" \
                "to a cpu output or shares >= $TOK_PREFIX_MIN bytes with its best match (the B70 greedy flip);" \
                "prints distinct_auto distinct_cpu pairs_equal min_prefix; all four loads rc 0, stdout non-empty" ;;
        Q_tg)
            echo "tg ('$RE_TPS' on the last eval line) >= $QTG_MIN, md5 $QMD5" \
                "(or flip=yes: >= $DECODE_FLIP_PREFIX_MIN bytes shared with $DECODE_REF)," \
                "zero '$RE_WATCHDOG', place line == '$PLACE_REF_TG'" \
                "(- = unscored), every '$RE_REFILL' has R == L (per line), and with a loan and no ring a" \
                "'$RE_BULK_REFILL_FIRST' line (else refill_late)" ;;
        Q_noc)
            echo "tg >= $QNOC_MIN, md5 $QMD5 (or flip=yes, as Q_tg), zero '$RE_WATCHDOG'," \
                "place line == '$PLACE_REF_NOC' (- = unscored)," \
                "every '$RE_REFILL' has R == L (per line, the same rule as Q_tg), and the same refill_late rule" ;;
        Q_decline)
            echo ">= 1 '$RE_DECLINE' and zero '$RE_DECLINE_RING', the '$RE_SUMMARY_CPU' counts >= 1, md5 of stdout" \
                "== Q_tok's or Q_tok_2's, zero '$RE_KV_REJECTED', zero '$RE_WRONG_DEVICE'; NOT_EXERCISED when" \
                "Q_tok's log has no '$RE_DECLINE_RING' line; NOT_SCORED when Q_tok's or Q_tok_2's output is missing" \
                "or empty or its load failed; prints declined_layers (distinct '$RE_DECLINE_ANY' layers)" ;;
        GO_gate)
            echo "grep -cx '1, 2, 3, 4, 5' == 1 (control: the echo line scores 0)," \
                "zero distinct '$RE_LAYER' lines" ;;
        GO_pp)
            echo "pp512 within $GO_PP_TOL % of $GO_PP_REF (names llama.cpp-zuze when within 15 % below)," \
                "tg128 within $GO_TG_TOL % of $GO_TG_REF" ;;
        GO_noc)
            echo "grep -cx '1, 2, 3, 4, 5' == 1; distinct '$RE_LAYER' lines recorded, gated only by" \
                "--go-noc-layers N (now: $GO_NOC_LAYERS, - = recorded)" ;;
        MI_b70 | MI_b50) echo "grep -c '$RE_MI' == 1" ;;
        resets) echo "journalctl -k --since <gate start> | grep -icE '$RE_RESETS' == 0" ;;
    esac
}

print_arm_plan() {
    local arm="$1" cmd
    arm_setup "$arm"
    if [ "$A_KIND" = journal ]; then
        say "$arm: journalctl -k --since '<gate start>' --no-pager > <out-dir>/$arm.out"
    else
        printf -v cmd '%q ' timeout -k 15 "$A_TMO" env "ONEAPI_DEVICE_SELECTOR=$A_SEL" "${A_ENV[@]}" \
            "$GATE_BIN_DIR/$A_BIN" "${A_ARGV[@]}"
        say "$arm: ${cmd}> <out-dir>/$arm.out 2> <out-dir>/$arm.err"
    fi
    say "    score: $(arm_doc "$arm")"
}

# ---------------------------------------------------------------- GPU.lock

lock_take() {
    mkdir "$LOCK_DIR" 2>/dev/null || return 1
    LOCK_HELD=1
    printf 'sycl-moe-exec-gate pid=%s sha=%s start=%s\n' "$$" "${EXPECT_SHA:-none}" "$(date +%FT%T)" \
        > "$LOCK_DIR/holder"
}

lock_release() {
    [ "${LOCK_HELD:-0}" = 1 ] || return 0
    rm -f "$LOCK_DIR/holder"
    rmdir "$LOCK_DIR" 2>/dev/null
    LOCK_HELD=0
}

# ---------------------------------------------------------------- host state

kb() { awk -v k="$1" '$1 == k ":" { print $2 }' /proc/meminfo; }
mem_line() { echo "Shmem=$(kb Shmem) kB MemAvailable=$(kb MemAvailable) kB"; }
builders() {
    local n i c
    if [ -n "$BUILDERS_FILE" ]; then # test hook: a file holding the count
        cat "$BUILDERS_FILE" 2>/dev/null || echo 0
        return
    fi
    n="$(pgrep -xc ninja)"
    i="$(pgrep -c icpx)"
    c=$((${n:-0} + ${i:-0}))
    echo "$c"
}
host_settled() {
    local s a
    s="$(kb Shmem)"
    a="$(kb MemAvailable)"
    [ "${s:-999999999}" -lt 31457280 ] && [ "${a:-0}" -gt 157286400 ]
}

# refuse <why>: before the first arm it prints REFUSED and the final line. Once arms have run, a refusal must not drop
# the verdicts already scored: it prints STOPPED and the summary over them, then the final line from the summary.
refuse() {
    if [ "$RUN_STARTED" = 1 ]; then
        say "STOPPED: $* (${#VERDICTS[@]} of ${#ARMS[@]} arms scored)"
        summarize "${VERDICTS[@]}" || true
    else
        say "REFUSED: $*"
        say "verdicts_pass=0 GATE_DONE"
    fi
    lock_release
    exit 2
}

# wait_builders <arm>: a build that is running is a reason to wait, not to end the run: poll like settle, up to
# BUILDERS_WAIT_MAX seconds, and refuse only when the bound expires.
wait_builders() {
    local waited=0 b
    b="$(builders)"
    while [ "$b" -ne 0 ]; do
        [ "$waited" -lt "$BUILDERS_WAIT_MAX" ] || refuse "builders = $b after waiting ${waited}s before arm ${1:-}"
        [ "$waited" -gt 0 ] || say "builders = $b before arm ${1:-}: waiting up to ${BUILDERS_WAIT_MAX}s"
        sleep "$BUILDERS_POLL"
        waited=$((waited + BUILDERS_POLL))
        b="$(builders)"
    done
    [ "$waited" -eq 0 ] || say "builders cleared after ${waited}s"
    return 0
}

settle() { # waits up to 10 minutes; GPU.lock serialises access, not memory recovery
    local i
    for i in $(seq 60); do
        host_settled && return 0
        sleep 10
    done
    refuse "host never settled (Shmem < 30 GB and MemAvailable > 150 GB): $(mem_line)"
}

# sample_builders_during <pid> <file>: the highest builder count seen every 5 s while <pid> runs.
sample_builders_during() {
    local m=0 b
    while kill -0 "$1" 2> /dev/null; do
        b="$(builders)"
        [ "$b" -le "$m" ] || m="$b"
        echo "$m" > "$2"
        sleep 5
    done
}

# arm_builders_post <arm>: the builders_post of an arm that has just ended: the count now, raised to the highest count
# the sampler saw while the arm ran (a build that started and finished inside the arm still depressed it).
arm_builders_post() {
    local post during
    post="$(builders)"
    during="$(cat "$OUT_DIR/$1.builders_max" 2>/dev/null || echo 0)"
    [ "$post" -ge "$during" ] || post="$during"
    echo "$post"
}

CHILD=""
# shellcheck disable=SC2329  # installed through trap below
on_signal() {
    if [ -n "$SAMPLER" ]; then
        kill -TERM "$SAMPLER" 2>/dev/null
    fi
    if [ -n "$CHILD" ]; then
        kill -TERM "$CHILD" 2>/dev/null
        wait "$CHILD" 2>/dev/null
    fi
    say "ABORTED by signal"
    if [ "$RUN_STARTED" = 1 ]; then
        say "STOPPED: signal (${#VERDICTS[@]} of ${#ARMS[@]} arms scored)"
        summarize "${VERDICTS[@]}" || true
    else
        say "verdicts_pass=0 GATE_DONE"
    fi
    lock_release
    exit 143
}

# ---------------------------------------------------------------- the final summary, shared with --score-summary

is_ring_arm() {
    local a
    for a in "${RING_ARMS[@]}"; do
        [ "$a" = "$1" ] && return 0
    done
    return 1
}

# summarize <verdict lines...>: prints the summary and final lines and returns the exit code. ring_engaged_any is read
# from the verdict lines of the RING_ARMS: yes when one says ring_engaged=yes, no when some carry ring_engaged=no and
# none says yes, na when no ring arm ran (a run of only the GPT-OSS or Mistral arms has no ring to score). Q_tg and
# Q_noc print ring_engaged too, but only as information about their prompt: a new arm counts once it is in RING_ARMS.
summarize() {
    local v pass=0 fail=0 notx=0 any_loan=no any_ring=na
    for v in "$@"; do
        case "$(status_of "$v")" in
            PASS) pass=$((pass + 1)) ;;
            RECORDED)
                # a recorded line whose process failed is a failure; the scorers already say FAIL, this covers old files
                case "$v" in *" rc="[1-9]*) fail=$((fail + 1)) ;; esac ;;
            NOT_EXERCISED | NOT_SCORED) notx=$((notx + 1)) ;;
            *) fail=$((fail + 1)) ;;
        esac
        case "$v" in *"loan_exercised=yes"*) any_loan=yes ;; esac
        if is_ring_arm "${v%%=*}"; then
            case "$v" in
                *"ring_engaged=yes"*) any_ring=yes ;;
                *"ring_engaged=no"*) [ "$any_ring" = yes ] || any_ring=no ;;
            esac
        fi
    done
    say "summary: ring_engaged_any=$any_ring loan_exercised_any=$any_loan verdicts_fail=$fail" \
        "not_exercised_or_unscored=$notx out=${OUT_DIR:-none}"
    say "verdicts_pass=$pass GATE_DONE"
    if [ "$fail" -gt 0 ]; then
        return 1
    elif [ "$notx" -gt 0 ] || [ "$any_ring" = no ]; then
        return 3
    fi
    return 0
}

# ---------------------------------------------------------------- modes that run no arm

if [ "$MAKE_PROMPT" = 1 ]; then
    [ -f "$PROMPT_DOC" ] || die_usage "no $PROMPT_DOC"
    mkdir -p "$(dirname "$PROMPT_OUT")"
    {
        # The first 8 KiB of the doc, cut back to a line boundary when it is longer; deterministic bytes.
        if [ "$(wc -c < "$PROMPT_DOC")" -gt 8192 ]; then
            LC_ALL=C head -c 8192 "$PROMPT_DOC" | sed '$d'
        else
            cat "$PROMPT_DOC"
        fi
        printf '\nSummarize the document above for a reader who has not seen it, in plain sentences.\n'
    } > "$PROMPT_OUT"
    say "wrote $PROMPT_OUT: $(wc -c < "$PROMPT_OUT") bytes, md5 $(md5_8 "$PROMPT_OUT")"
    exit 0
fi

if [ "$CHECK_CONTROLS" = 1 ]; then
    check_controls || { say "controls: FAIL"; exit 1; }
    say "controls: ok"
    exit 0
fi

if [ "$LOCK_SELFTEST" = 1 ]; then
    lock_take || { say "lock-selftest: first take failed"; exit 1; }
    [ -f "$LOCK_DIR/holder" ] || { say "lock-selftest: no holder file"; exit 1; }
    if lock_take; then say "lock-selftest: second take succeeded"; exit 1; fi
    lock_release
    [ ! -e "$LOCK_DIR" ] || { say "lock-selftest: lock dir survived release"; exit 1; }
    say "lock-selftest: ok"
    exit 0
fi

if [ "$RING_ARMS_MODE" = 1 ]; then
    say "${RING_ARMS[*]}"
    exit 0
fi

if [ -n "$SCORE_SUMMARY" ]; then
    [ -f "$SCORE_SUMMARY" ] || die_usage "no verdict file $SCORE_SUMMARY"
    mapfile -t SUM_LINES < "$SCORE_SUMMARY"
    summarize "${SUM_LINES[@]}"
    exit $?
fi

if [ "$WAIT_BUILDERS_SELFTEST" = 1 ]; then
    wait_builders selftest
    say "wait-builders-selftest: cleared"
    exit 0
fi

if [ -n "$BUILDERS_POST_SELFTEST" ]; then
    say "builders-post-selftest: post=$(arm_builders_post "$BUILDERS_POST_SELFTEST")"
    exit 0
fi

if [ -n "$SCORE_ARM" ]; then
    line="$(score_arm_files "$SCORE_ARM" "$S_CSV" "$S_OUT" "$S_ERR" "$S_REF" "$S_JOURNAL" "$S_RC" "$S_OUT2" "$S_REF2")"
    say "$line"
    case "$(status_of "$line")" in
        PASS | RECORDED) exit 0 ;;
        NOT_EXERCISED | NOT_SCORED) exit 3 ;;
        *) exit 1 ;;
    esac
fi

if [ "$DRY_RUN" = 1 ]; then
    say "dry-run: nothing is executed, no lock is taken, no device is touched"
    say "build-dir: $BUILD_DIR   src-dir: $SRC_DIR   lock: $LOCK_DIR (take=$TAKE_LOCK)"
    say "preflights: <sha> == head of src-dir, GGML_SYCL ON, binaries newer than head and linked to SYCL, models," \
        "grep controls, no exported GGML_SYCL_*/LLAMA_ARG_*, builders clear (waited for, 15 min), Shmem < 30 GB," \
        "MemAvailable > 150 GB," \
        "an empty or --arm/--clobber out-dir"
    for arm in "${ARMS[@]}"; do
        print_arm_plan "$arm"
    done
    say "final line: verdicts_pass=N GATE_DONE"
    exit 0
fi

# ---------------------------------------------------------------- the real run

LOCK_HELD=0
trap 'lock_release' EXIT
trap on_signal TERM INT HUP

# seed_verdicts: the selftests preload the verdicts a run would have scored, so the stop paths can be tested
# without arms.
seed_verdicts() {
    [ -z "$SEED_VERDICTS" ] || { mapfile -t VERDICTS < "$SEED_VERDICTS"; RUN_STARTED=1; }
}

if [ "$REFUSE_SELFTEST" = 1 ]; then
    lock_take || refuse "refuse-selftest: could not take $LOCK_DIR"
    seed_verdicts
    refuse "synthetic stop"
fi

if [ "$SIGNAL_SELFTEST" = 1 ]; then
    lock_take || refuse "signal-selftest: could not take $LOCK_DIR"
    seed_verdicts
    sleep 30 &
    CHILD=$!
    say "signal-selftest: child=$CHILD lock=$LOCK_DIR"
    wait "$CHILD"
    say "signal-selftest: the child exited without a signal"
    exit 1
fi

# Exported tuning variables would reach every arm; the arms set their own through env, so none may be inherited.
leaked="$(env | grep -aoE '^(GGML_SYCL_|LLAMA_ARG_)[A-Za-z0-9_]*' | tr '\n' ' ')"
[ -z "$leaked" ] ||
    refuse "exported variable(s) would leak into every arm: ${leaked}(unset them; each arm sets its own)"

# The sha format is checked even with --no-head-check. The head comparison reads the build's own source tree.
[ -n "$EXPECT_SHA" ] || refuse "no <sha> argument: pass the merge sha the build was made from"
[[ "$EXPECT_SHA" =~ ^[0-9a-f]{7,40}$ ]] || refuse "<sha> '$EXPECT_SHA' is not 7 to 40 lowercase hex characters"
if [ "$HEAD_CHECK" = 1 ]; then
    head_sha="$(git -C "$SRC_DIR" rev-parse HEAD 2>/dev/null)"
    case "$head_sha" in
        "$EXPECT_SHA"*) ;;
        *) refuse "head ${head_sha:-unknown} of $SRC_DIR does not start with $EXPECT_SHA" ;;
    esac
fi

if [ -z "$OUT_DIR" ]; then
    short_sha="$(git -C "$SRC_DIR" rev-parse --short HEAD 2>/dev/null || echo run)"
    OUT_DIR="$BUILD_DIR/moe-exec-gate-$short_sha-$(date +%H%M%S)"
fi
# A reused out-dir would let one run's files be scored as another's. A non-empty one is refused unless --arm names the
# arms to redo (only their files are overwritten) or --clobber says so.
if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ] && [ "$ARMS_EXPLICIT" = 0 ] && [ "$CLOBBER" = 0 ]; then
    refuse "out-dir $OUT_DIR is not empty: use a fresh one, --arm to redo named arms, or --clobber"
fi

# A reconfigure can drop the SYCL backend and a CPU-only build prints the same tokens about 13x slower.
grep -q '^GGML_SYCL:BOOL=ON' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null ||
    refuse "GGML_SYCL is not ON in $BUILD_DIR/CMakeCache.txt"
head_time="$(git -C "$SRC_DIR" log -1 --format=%ct 2>/dev/null)"
need_bins=()
for arm in "${ARMS[@]}"; do
    arm_setup "$arm"
    [ -z "$A_BIN" ] || need_bins+=("$A_BIN")
done
for b in $(printf '%s\n' "${need_bins[@]}" | sort -u); do
    [ -x "$GATE_BIN_DIR/$b" ] || refuse "no $GATE_BIN_DIR/$b"
    if [ "$HEAD_CHECK" = 1 ] && [ -n "$head_time" ] && [ "$(stat -c %Y "$GATE_BIN_DIR/$b")" -le "$head_time" ]; then
        refuse "$GATE_BIN_DIR/$b is not newer than the head commit"
    fi
    [ "$(ldd "$GATE_BIN_DIR/$b" 2>/dev/null | grep -cE 'libggml-sycl|libsycl')" -ge 2 ] ||
        refuse "$b does not link the SYCL runtime"
done
for arm in "${ARMS[@]}"; do
    case "$arm" in
        Q_*) [ -f "$QWEN" ] || refuse "no model $QWEN" ;;
        GO_*) [ -f "$GPTOSS" ] || refuse "no model $GPTOSS" ;;
        MI_*) [ -f "$MISTRAL" ] || refuse "no model $MISTRAL" ;;
    esac
    case "$arm" in
        Q_tok | Q_tok_2 | Q_tok_cpu_1 | Q_tok_cpu | Q_decline)
            [ -f "$TOK_PROMPT" ] || refuse "no prompt $TOK_PROMPT" ;;
        Q_tg | Q_noc)
            [ -f "$DECODE_PROMPT" ] || refuse "no prompt $DECODE_PROMPT"
            [ -f "$DECODE_REF" ] || refuse "no reference text $DECODE_REF" ;;
    esac
done
check_controls || refuse "a grep control failed (see stderr)"
# An arm list that is only the journal arm touches no device and loads no model: nothing to protect from a build.
device_arms=0
for arm in "${ARMS[@]}"; do
    arm_setup "$arm"
    [ "$A_KIND" = journal ] || device_arms=1
done
[ "$device_arms" = 0 ] || wait_builders start
[ "$device_arms" = 0 ] || host_settled || refuse "host not settled: $(mem_line)"
if [ "$TAKE_LOCK" = 1 ]; then
    lock_take || refuse "could not take $LOCK_DIR (held: $(cat "$LOCK_DIR/holder" 2>/dev/null || echo unknown))"
fi

mkdir -p "$OUT_DIR"
[ -z "$TUNING_DIR" ] || { mkdir -p "$TUNING_DIR"; export GGML_SYCL_TUNING_CACHE_DIR="$TUNING_DIR"; }
[ -n "$SINCE" ] || SINCE="$(date '+%Y-%m-%d %H:%M:%S')"

say "gate: sha=${EXPECT_SHA:-unchecked} build=$BUILD_DIR out=$OUT_DIR since='$SINCE' oneapi=$GATE_ONEAPI" \
    "arms=${ARMS[*]}"
say "pre: $(mem_line) load=$(cut -d' ' -f1 /proc/loadavg) builders=$(builders)"

RUN_STARTED=1

run_arm() { # <arm>: runs it, writes <arm>.out/.err/.rc, prints the sample lines
    local arm="$1" rc pre post
    arm_setup "$arm"
    if [ "$A_KIND" = journal ]; then
        journalctl -k --since "$SINCE" --no-pager > "$OUT_DIR/$arm.out" 2> "$OUT_DIR/$arm.err"
        echo "$?" > "$OUT_DIR/$arm.rc"
        echo na > "$OUT_DIR/$arm.builders_pre"
        echo na > "$OUT_DIR/$arm.builders_post"
        return 0
    fi
    wait_builders "$arm"
    settle
    pre="$(builders)"
    echo "$pre" > "$OUT_DIR/$arm.builders_pre"
    echo 0 > "$OUT_DIR/$arm.builders_max"
    say "arm $arm: pre $(mem_line) load=$(cut -d' ' -f1 /proc/loadavg) builders=$pre $(date +%T)"
    timeout -k 15 "$A_TMO" env ONEAPI_DEVICE_SELECTOR="$A_SEL" "${A_ENV[@]}" "$GATE_BIN_DIR/$A_BIN" "${A_ARGV[@]}" \
        > "$OUT_DIR/$arm.out" 2> "$OUT_DIR/$arm.err" < /dev/null &
    CHILD=$!
    sample_builders_during "$CHILD" "$OUT_DIR/$arm.builders_max" &
    SAMPLER=$!
    wait "$CHILD"
    rc=$?
    CHILD=""
    kill -TERM "$SAMPLER" 2>/dev/null
    wait "$SAMPLER" 2>/dev/null
    SAMPLER=""
    echo "$rc" > "$OUT_DIR/$arm.rc"
    sleep 5
    post="$(arm_builders_post "$arm")"
    echo "$post" > "$OUT_DIR/$arm.builders_post"
    say "arm $arm: rc=$rc post(+5s) $(mem_line) load=$(cut -d' ' -f1 /proc/loadavg) builders_post=$post (max of the" \
        "5 s samples during the arm and the one after it) $(date +%T)"
}

score_arm_in_outdir() { # <arm>
    local arm="$1" rc out err csv ref line
    rc="$(cat "$OUT_DIR/$arm.rc" 2>/dev/null || echo 99)"
    out="$OUT_DIR/$arm.out"
    err="$OUT_DIR/$arm.err"
    csv="$OUT_DIR/$arm.out"
    ref="$OUT_DIR/Q_tok.out"
    S_RC_OUT2="$(cat "$OUT_DIR/Q_tok_cpu_1.rc" 2>/dev/null || echo 0)"
    S_RC_REF="$(cat "$OUT_DIR/Q_tok.rc" 2>/dev/null || echo 0)"
    S_RC_REF2="$(cat "$OUT_DIR/Q_tok_2.rc" 2>/dev/null || echo 0)"
    S_REF_ERR="$OUT_DIR/Q_tok.err" # Q_decline reads Q_tok's layer decision; no other arm reads it
    S_BPRE="$(cat "$OUT_DIR/$arm.builders_pre" 2>/dev/null || echo na)"
    S_BPOST="$(cat "$OUT_DIR/$arm.builders_post" 2>/dev/null || echo na)"
    line="$(score_arm_files "$arm" "$csv" "$out" "$err" "$ref" "$out" "$rc" \
        "$OUT_DIR/Q_tok_cpu_1.out" "$OUT_DIR/Q_tok_2.out")"
    say "$line"
    VERDICTS+=("$line")
}

for arm in "${ARMS[@]}"; do
    run_arm "$arm"
    score_arm_in_outdir "$arm"
done

lock_release
say "post: $(mem_line)"
summarize "${VERDICTS[@]}"
exit $?
