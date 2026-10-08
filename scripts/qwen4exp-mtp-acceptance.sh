#!/usr/bin/env bash
# Measure Qwen3.8-Flash-Next MTP draft acceptance on a CPU-only build (llama.cpp-0rhb).
#
# Acceptance and tokens per verify round depend on the target model and the
# prompts, not on the backend, so this runs a CPU-only llama-speculative-simple:
# no GPU, no oneAPI, no TTM shmem.  It IS a model load (the IQ3_XXS target is
# ~76 GB of page cache), so run it by hand, one process at a time, and never
# from a subagent.
#
# Arms: head {q8 = Q8_0 MTP GGUF, q4 = Q4_0 MTP GGUF} x n-max {2,3} x prompt
# {code, chat, reasoning}; greedy, seed 42.  Two sets: "base" (the binary's
# default p-min 0.00, drafts never filtered) and "p05" (--spec-draft-p-min 0.5,
# the Strata comparison: keep a draft only while the MTP probability is >= 0.5).
# Each arm writes $OUT/<head>_n<k>_<prompt>.log (base) or
# $OUT/<head>_n<k>_p05_<prompt>.log (p05); the parser prints the table at the end.
#
# Usage:
#   scripts/qwen4exp-mtp-acceptance.sh --dry-run     # print the commands only
#   scripts/qwen4exp-mtp-acceptance.sh               # run all 24 arms (both sets), serially
#   SETS=base scripts/qwen4exp-mtp-acceptance.sh     # only the 12 unfiltered arms (SETS=p05: the other 12)
#   ARMS="q8_n3_code q4_n2_p05_chat" scripts/qwen4exp-mtp-acceptance.sh   # a subset
#   scripts/qwen4exp-mtp-acceptance.sh --pairs       # no-MTP baseline vs q4 n-max 2, one interleaved pair per prompt
#                                                    # (code AB, chat BA, reasoning AB); the parser prints the speedup
#
# Environment (defaults in brackets):
#   BIN      speculative binary   [<repo>/build-cpu/bin/llama-speculative-simple]
#   BASE_BIN no-MTP baseline binary, --pairs only [<repo>/build-cpu/bin/llama-completion]
#   TARGET   target model, first split of the IQ3_XXS pair
#   HEAD_Q8  Q8_0 MTP head        [/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf]
#   HEAD_Q4  Q4_0 MTP head        [/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q4_0.gguf]
#   OUT      log directory        [./qwen4exp-mtp-acceptance-out]
#   SETS [base p05], N_PREDICT, CTX, UBATCH
#   THREADS  CPU threads, passed as -t/-tb and -td/-tbd [16].  The binary's own default
#            was 4 threads on this 24-CPU host (nproc 24: 8P + 16E, no SMT): 0.225 t/s, ~20 min per arm.
#   NOMMAP   1 = load the models into anonymous memory instead of mmapping them [1]: `-lm none`
#            (this tree has no --no-mmap).  -lzm on is kept either way: --help says it
#            "requires mmap", but the loader maps a lazy tensor's own file even when use_mmap
#            is false, so the 27 GiB per_layer_token_embd (PLE table, a few rows read per
#            token) stays out of RAM (measured by the lead: RSS 47 GB under -lm none, not 76).
#            Why default 1: on /models (bcachefs) the page cache does not keep an mmapped IQ3
#            file; one mmap run read 2.78 TB from disk in 28 min at 15 GB RSS, and decode was
#            ~50x slower than with -lm none.
#   WARM     1 = once, before the first arm, dd every shard of the target and the
#            heads to /dev/null (userspace reads) and print fincore, so the page
#            cache is warm and the log proves it [0]
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-$ROOT/build-cpu/bin/llama-speculative-simple}
TARGET=${TARGET:-/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
HEAD_Q8=${HEAD_Q8:-/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}
HEAD_Q4=${HEAD_Q4:-/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q4_0.gguf}
OUT=${OUT:-$PWD/qwen4exp-mtp-acceptance-out}
N_PREDICT=${N_PREDICT:-256}
CTX=${CTX:-4096}
UBATCH=${UBATCH:-512}
THREADS=${THREADS:-16}
WARM=${WARM:-0}
NOMMAP=${NOMMAP:-1}
PROMPTS_DIR=$ROOT/scripts/qwen4exp-mtp-prompts
PARSER=$ROOT/scripts/parse-qwen4exp-mtp-acceptance.py

DRY=0
PAIRS=0
for a in "$@"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        --pairs) PAIRS=1 ;;
        *) echo "unknown argument: $a (want --dry-run and/or --pairs)" >&2; exit 1 ;;
    esac
done
BASE_BIN=${BASE_BIN:-$ROOT/build-cpu/bin/llama-completion}

SETS=${SETS:-"base p05"}
ALL_ARMS=""
for set in $SETS; do
    case "$set" in base) tag="" ;; p05) tag="p05_" ;; *) echo "bad set $set (want base and/or p05)" >&2; exit 1 ;; esac
    for head in q8 q4; do
        for k in 2 3; do
            for prompt in code chat reasoning; do
                ALL_ARMS="$ALL_ARMS ${head}_n${k}_${tag}${prompt}"
            done
        done
    done
done
ARMS=${ARMS:-$ALL_ARMS}

# refuse a malformed arm name up front: head_path() runs inside a command substitution below, where
# its failure would be swallowed and leave `-md ''` in the command line
for arm in $ARMS; do
    [[ "$arm" =~ ^(q8|q4)_n[0-9]+_(p05_)?(code|chat|reasoning)$ ]] || {
        echo "bad arm '$arm' (want <q8|q4>_n<k>_[p05_]<code|chat|reasoning>)" >&2; exit 1; }
done

head_path() { case "$1" in q8) echo "$HEAD_Q8" ;; q4) echo "$HEAD_Q4" ;; *) echo "bad head $1" >&2; return 1 ;; esac; }

run_args() { # the flags every run shares, baseline and MTP alike; one argument per line
    local prompt=$1
    printf '%s\n' -f "$PROMPTS_DIR/$prompt.txt" -n "$N_PREDICT" --seed 42 --temp 0 \
        -c "$CTX" -ub "$UBATCH" -ngl 0 -lv 4
    # -lzm on stays under -lm none: the loader maps a lazy tensor's own file even when use_mmap
    # is false, so the 27 GiB PLE table stays lazy (measured: RSS 47 GB, not 76 GB)
    [ "$NOMMAP" = 1 ] && printf '%s\n' -lm none
    printf '%s\n' -lzm on
    printf '%s\n' -t "$THREADS" -tb "$THREADS"
    return 0
}

arm_cmd() { # prints the command for one arm, one argument per line
    local arm=$1 head k prompt pmin=""
    head=${arm%%_*}; k=${arm#*_n}; k=${k%%_*}; prompt=${arm##*_}
    case "$arm" in *_p05_*) pmin=0.5 ;; esac
    printf '%s\n' "$BIN" -m "$TARGET" -md "$(head_path "$head")" \
        --spec-type draft-mtp --spec-draft-n-max "$k"
    run_args "$prompt"
    printf '%s\n' -td "$THREADS" -tbd "$THREADS"
    [ -n "$pmin" ] && printf '%s\n' --spec-draft-p-min "$pmin"
    return 0
}

base_cmd() { # the no-MTP baseline for one prompt: the same flags without speculation
    printf '%s\n' "$BASE_BIN" -m "$TARGET"
    run_args "$1"
    printf '%s\n' -no-cnv
    return 0
}

# --pairs: AB, BA, AB so that a slow drift in host load does not always favour the same arm
PAIR_ORDER="code:AB chat:BA reasoning:AB"
PAIR_ARM_N=${PAIR_ARM_N:-2}
pair_steps() { # prints "<kind> <prompt>" per step, in run order; kind = base or mtp
    local item prompt order
    for item in $PAIR_ORDER; do
        prompt=${item%%:*}; order=${item##*:}
        if [ "$order" = AB ]; then echo "base $prompt"; echo "mtp $prompt"; else echo "mtp $prompt"; echo "base $prompt"; fi
    done
}
step_cmd() { # $1 kind, $2 prompt
    if [ "$1" = base ]; then base_cmd "$2"; else arm_cmd "q4_n${PAIR_ARM_N}_$2"; fi
}

meminfo() { grep -E '^(MemAvailable|Shmem):' /proc/meminfo | tr '\n' ' '; echo; }

if [ "$DRY" = 1 ] && [ "$PAIRS" = 1 ]; then
    mapfile -t steps < <(pair_steps)
    for step in "${steps[@]}"; do
        kind=${step%% *}; prompt=${step##* }
        mapfile -t cmd < <(step_cmd "$kind" "$prompt")
        echo "# ${kind}_${prompt}_1"
        printf '%q ' "${cmd[@]}"; echo "> $OUT/pairs/${kind}_${prompt}_1.log 2>&1"
    done
    exit 0
fi
if [ "$DRY" = 1 ]; then
    for arm in $ARMS; do
        mapfile -t cmd < <(arm_cmd "$arm") || exit 1
        echo "# $arm"
        printf '%q ' "${cmd[@]}"; echo "> $OUT/$arm.log 2>&1"
    done
    exit 0
fi

# ---- preflight: refuse rather than run a CPU load that cannot mean what it says
bins=("$BIN")
[ "$PAIRS" = 1 ] && bins+=("$BASE_BIN")
for f in "${bins[@]}" "$TARGET" "$HEAD_Q8" "$HEAD_Q4" "$PARSER"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
for f in "${bins[@]}"; do
    if ldd "$f" 2>/dev/null | grep -qiE 'sycl|libze'; then
        echo "refusing: $f links SYCL/Level Zero; this measurement is CPU-only" >&2
        exit 1
    fi
done
# comm names only (a command line may merely mention a binary); comm is 15 chars wide
busy=$(ps -eo pid=,comm= | grep -E ' (llama-|test-)' || true)
if [ -n "$busy" ]; then
    echo "refusing: another llama-*/test-* process is running; one model load at a time" >&2
    echo "$busy" >&2
    exit 1
fi
avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
shmem_kb=$(awk '/^Shmem:/ {print $2}' /proc/meminfo)
if [ "$avail_kb" -lt $((100 * 1024 * 1024)) ] || [ "$shmem_kb" -gt $((30 * 1024 * 1024)) ]; then
    echo "refusing: host memory not settled (MemAvailable ${avail_kb} kB, Shmem ${shmem_kb} kB; want >100 GB / <30 GB)" >&2
    exit 1
fi
echo "uptime: $(uptime)"
echo "start:  $(meminfo)"

mkdir -p "$OUT"
if [ "$WARM" = 1 ]; then
    # every shard of the split target (<stem>-NNNNN-of-MMMMM.gguf) plus both heads.
    # dd, not `cat f > /dev/null`: coreutils cat can splice/copy_file_range straight to
    # /dev/null without populating the page cache (measured: 70 GB in 43 s, 0 B resident
    # on bcachefs).  dd read()s through userspace, so the pages do land in the cache.
    shards=("${TARGET%-*-of-*.gguf}"-*-of-*.gguf "$HEAD_Q8" "$HEAD_Q4")
    echo "warming page cache: ${#shards[@]} files ($(date +%H:%M:%S))"
    for f in "${shards[@]}"; do
        dd if="$f" of=/dev/null bs=16M status=none || { echo "warm-up read failed: $f" >&2; exit 1; }
    done
    echo "warmed ($(date +%H:%M:%S)): $(meminfo)"
    # proof for the run log: resident bytes per file (a warm file shows its full size)
    fincore -b "${shards[@]}" || echo "fincore unavailable or failed; warm state not proven" >&2
fi
if [ "$PAIRS" = 1 ]; then
    mkdir -p "$OUT/pairs"
    mapfile -t steps < <(pair_steps)
    for step in "${steps[@]}"; do
        kind=${step%% *}; prompt=${step##* }
        mapfile -t cmd < <(step_cmd "$kind" "$prompt")
        log=$OUT/pairs/${kind}_${prompt}_1.log
        echo "== ${kind}_${prompt}_1  ($(date +%H:%M:%S))"
        timeout 3600 "${cmd[@]}" < /dev/null > "$log" 2>&1
        rc=$?
        echo "   rc=$rc  $(meminfo)"
        if [ "$rc" -ne 0 ]; then
            echo "step ${kind}_${prompt}_1 failed (rc=$rc), see $log; stopping" >&2
            exit "$rc"
        fi
        sleep 5
    done
    echo
    exec /usr/bin/env python3 "$PARSER" --pairs "$OUT/pairs"
fi

logs=()
for arm in $ARMS; do
    mapfile -t cmd < <(arm_cmd "$arm") || exit 1
    log=$OUT/$arm.log
    echo "== $arm  ($(date +%H:%M:%S))"
    timeout 3600 "${cmd[@]}" > "$log" 2>&1
    rc=$?
    echo "   rc=$rc  $(meminfo)"
    logs+=("$log")
    if [ "$rc" -ne 0 ]; then
        # stop at the first failure: a failing load must not be retried 11 more times
        echo "arm $arm failed (rc=$rc), see $log; stopping" >&2
        break
    fi
    sleep 5
done

echo
/usr/bin/env python3 "$PARSER" "${logs[@]}"
