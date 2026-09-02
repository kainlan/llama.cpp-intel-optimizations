#!/usr/bin/env bash
# tests/test-sycl-canonical-gates-script.sh (llama.cpp-rou3)
#
# Stub-driven unit test of scripts/sycl-canonical-gates.sh's own logic. This
# is CPU-only: it never touches a GPU and never loads a real model. All
# "binaries" are shell scripts controlled by this file, wired through the
# script's SYCL_GATES_* overrides (SYCL_GATES_BIN_DIR, _CMAKE_CACHE, _LDD_CMD,
# _MISTRAL_MODEL, _GPTOSS_MODEL, _SKIP_ONEAPI_SOURCE, _SETTLE_SECONDS).
#
# Six spec-lettered cases, plus discriminator cases added on spec review
# (tracker comment c-2jsa):
#   (a)  correct digits                              -> PASS rc=0
#   (b)  digits only in the prompt echo (GPT-OSS)     -> FAIL
#   (b2) digits mid-line, not at line start (Mistral) -> FAIL (m1 finding)
#   (c)  digits correct but an abort line on stderr   -> FAIL
#   (d)  --list-devices: the CURRENTLY LIVE dev_mgr abort shape
#        (rc!=0, empty stdout, "SYCL" text only on stderr)   -> rc 77 (SKIP)
#   (d2) --list-devices: the FUTURE llama.cpp-1lrh graceful shape
#        (rc=0, stdout "Available devices:\n  (none)\n")     -> rc 77 (SKIP)
#   (d3) --list-devices: non-zero rc WITH a valid device line on stdout
#        -> rc 77 (SKIP) -- isolates the rc!=0 clause on its own (round-2
#        minor finding 2 in c-u3ji)
#   (e)  missing binary                               -> rc 1
#   (f)  CMakeCache GGML_SYCL:BOOL=OFF                 -> rc 1
#   (oneapi-sourcing) real /opt/intel/oneapi/setvars.sh survives under
#        `set -euo pipefail` (B1 finding: it used to die silently, rc=127,
#        zero output, on OCL_ICD_FILENAMES under `set -u`)
#   (oneapi-clobber) a FAKE, hostile setvars.sh that exports empty/wrong
#        values for SELECTOR/GATE/BUILD_DIR/BIN_DIR must not move the
#        selector actually used or the gate actually run (round-2 blocker
#        in c-u3ji: BUILD_DIR/GATE/SELECTOR are parsed BEFORE the source,
#        so they need an explicit snapshot-and-restore, not just "assign
#        after" like every other derived variable)
#   (stderr-only-mistral) correct digits on STDERR only, nothing on stdout
#        -> FAIL (round-2 minor finding 1: the check must read .out, not
#        a combined stream)
#   (stderr-only-gptoss)  same, GPT-OSS side                -> FAIL
#
# Run directly: bash tests/test-sycl-canonical-gates-script.sh; echo rc=$?

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATES_SCRIPT="$ROOT_DIR/scripts/sycl-canonical-gates.sh"

if [ ! -x "$GATES_SCRIPT" ]; then
    echo "test-sycl-canonical-gates-script: $GATES_SCRIPT missing or not executable" >&2
    exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

FAILURES=0
fail() {
    echo "FAIL: $*" >&2
    FAILURES=$((FAILURES + 1))
}

# --- fixture builders ------------------------------------------------------

write_exec() {
    # $1 = path, $2 = body (already a full script)
    printf '%s\n' "$2" > "$1"
    chmod +x "$1"
}

# A stub llama-completion: handles --list-devices (governed by
# STUB_DEVICE_MODE) and the plain completion invocation (governed by
# STUB_MISTRAL_MODE).
#
# STUB_DEVICE_MODE shapes, all reproducing the REAL binary's behaviour:
#   present       -- a device is enumerated: stdout has a "  SYCL0: ..." line.
#   none-graceful -- llama.cpp-1lrh's fixed dev_mgr: rc=0, stdout
#                    "Available devices:\n  (none)\n", no SYCL text anywhere.
#   none-abort    -- the CURRENTLY COMMITTED dev_mgr (dpct/helper.hpp): zero
#                    devices GGML_ABORTs from inside ggml_backend_load_all(),
#                    which runs BEFORE common_print_available_devices() ever
#                    prints "Available devices:" -- so stdout is EMPTY, and
#                    the only "SYCL" text is on STDERR ("SYCL device manager
#                    initialization failed: no devices found on any
#                    platform." + the ONEAPI_DEVICE_SELECTOR hint), with a
#                    non-zero exit status.
LLAMA_COMPLETION_STUB='#!/usr/bin/env bash
if [ "${1:-}" = "--list-devices" ]; then
    case "${STUB_DEVICE_MODE:-present}" in
        present)
            printf "Available devices:\n  SYCL0: Fake Battlemage (16384 MiB, 16384 MiB free)\n"
            exit 0
            ;;
        none-graceful)
            printf "Available devices:\n  (none)\n"
            exit 0
            ;;
        none-abort)
            printf "SYCL device manager initialization failed: no devices found on any platform.\n" >&2
            printf "Check ONEAPI_DEVICE_SELECTOR environment variable and available SYCL runtimes.\n" >&2
            exit 134
            ;;
        rc-nonzero-with-device)
            # Contradictory-looking on purpose: a valid device line on
            # stdout, but the process itself still exits non-zero. Isolates
            # the "non-zero probe rc is authoritative" clause from the
            # "no matching stdout line" clause -- a mutant that drops the
            # rc check would read this as device-present.
            printf "Available devices:\n  SYCL0: Fake Battlemage (16384 MiB, 16384 MiB free)\n"
            exit 3
            ;;
        *)
            echo "unknown STUB_DEVICE_MODE: ${STUB_DEVICE_MODE:-}" >&2
            exit 1
            ;;
    esac
fi
case "${STUB_MISTRAL_MODE:-pass}" in
    pass)
        printf "1, 2, 3, 4, 5, 6, 7, 8, 9, 10\n"
        exit 0
        ;;
    abort)
        printf "1, 2, 3, 4, 5, 6, 7, 8, 9, 10\n"
        printf "/build/ggml/src/ggml-sycl/ggml-sycl.cpp:12345: GGML_ASSERT(x) failed\n" >&2
        exit 0
        ;;
    mid-line)
        printf "> some other text 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 trailing\n"
        exit 0
        ;;
    stderr-only)
        # Correct digits, but on STDERR, not stdout. The pass check must
        # read the .out file, not a combined/either stream.
        printf "1, 2, 3, 4, 5, 6, 7, 8, 9, 10\n" >&2
        exit 0
        ;;
    wrong)
        printf "not the right digits\n"
        exit 0
        ;;
    *)
        echo "unknown STUB_MISTRAL_MODE: ${STUB_MISTRAL_MODE:-}" >&2
        exit 1
        ;;
esac
'

# A stub llama-cli: only the plain chat invocation, governed by
# STUB_GPTOSS_MODE. Case (b) reproduces the documented false-fail: the digits
# appear only inside an echoed prompt line, never alone on their own line.
LLAMA_CLI_STUB='#!/usr/bin/env bash
case "${STUB_GPTOSS_MODE:-pass}" in
    pass)
        printf "1, 2, 3, 4, 5\n"
        exit 0
        ;;
    prompt-echo)
        printf "> Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5\n"
        exit 0
        ;;
    stderr-only)
        printf "1, 2, 3, 4, 5\n" >&2
        exit 0
        ;;
    *)
        echo "unknown STUB_GPTOSS_MODE: ${STUB_GPTOSS_MODE:-}" >&2
        exit 1
        ;;
esac
'

# A stub ldd: STUB_LDD_MODE=ok reports both libs linked; =insufficient
# reports neither, simulating a CPU-fallback link.
LDD_STUB='#!/usr/bin/env bash
if [ "${STUB_LDD_MODE:-ok}" = "insufficient" ]; then
    printf "linux-vdso.so.1\n"
    printf "libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6\n"
else
    printf "libggml-sycl.so => /fake/libggml-sycl.so\n"
    printf "libsycl.so.7 => /fake/libsycl.so.7\n"
fi
'

write_cmake_cache() {
    # $1 = path, $2 = ON|OFF
    printf 'GGML_SYCL:BOOL=%s\n' "$2" > "$1"
}

# Standard fixture: both stub binaries, a passing ldd, GGML_SYCL ON. Callers
# override individual pieces (env vars, or by deleting a file) per case.
setup_case_dir() {
    local dir="$1"
    mkdir -p "$dir/bin"
    write_exec "$dir/bin/llama-completion" "$LLAMA_COMPLETION_STUB"
    write_exec "$dir/bin/llama-cli" "$LLAMA_CLI_STUB"
    write_exec "$dir/ldd" "$LDD_STUB"
    write_cmake_cache "$dir/CMakeCache.txt" "ON"
}

run_gates() {
    # $1 = case dir, $2 = --gate value, remaining = extra args
    local dir="$1" gate="$2"
    shift 2
    SYCL_GATES_BIN_DIR="$dir/bin" \
    SYCL_GATES_CMAKE_CACHE="$dir/CMakeCache.txt" \
    SYCL_GATES_LDD_CMD="$dir/ldd" \
    SYCL_GATES_MISTRAL_MODEL="$dir/fake-mistral.gguf" \
    SYCL_GATES_GPTOSS_MODEL="$dir/fake-gptoss.gguf" \
    SYCL_GATES_SKIP_ONEAPI_SOURCE=1 \
    SYCL_GATES_SETTLE_SECONDS=0 \
        "$GATES_SCRIPT" --build-dir "$dir/build" --gate "$gate" --selector level_zero:1 "$@"
}

# Deliberately does NOT set SYCL_GATES_SKIP_ONEAPI_SOURCE, and forcibly
# unsets ONEAPI_ROOT (via `env -u`, not just an empty assignment -- a bare
# `VAR=` still leaves the name present and non-empty-checkable, `env -u`
# removes it) so the child ALWAYS takes the real sourcing branch regardless
# of what this test runner's own ambient environment already has active.
# Requires /opt/intel/oneapi/setvars.sh to exist (true on this host); skips
# gracefully if it does not, since the B1 fix is then untestable here.
run_gates_real_oneapi() {
    local dir="$1" gate="$2"
    shift 2
    env -u ONEAPI_ROOT -u SYCL_GATES_SKIP_ONEAPI_SOURCE \
        SYCL_GATES_BIN_DIR="$dir/bin" \
        SYCL_GATES_CMAKE_CACHE="$dir/CMakeCache.txt" \
        SYCL_GATES_LDD_CMD="$dir/ldd" \
        SYCL_GATES_MISTRAL_MODEL="$dir/fake-mistral.gguf" \
        SYCL_GATES_GPTOSS_MODEL="$dir/fake-gptoss.gguf" \
        SYCL_GATES_SETTLE_SECONDS=0 \
        "$GATES_SCRIPT" --build-dir "$dir/build" --gate "$gate" --selector level_zero:1 "$@"
}

touch_fake_models() {
    local dir="$1"
    : > "$dir/fake-mistral.gguf"
    : > "$dir/fake-gptoss.gguf"
}

# A hostile, FAKE setvars.sh: exports empty/wrong values for every name this
# script parses from argv (SELECTOR, GATE, BUILD_DIR) plus the one already-
# proven collision (BIN_DIR), simulating the worst case an oneAPI component
# could do. Round-2 blocker (c-u3ji): BUILD_DIR/GATE/SELECTOR are consumed
# from argv BEFORE the source call, so "assign after sourcing" alone (which
# is sufficient for BIN_DIR and friends) does not protect them -- they need
# an explicit snapshot-and-restore, and this is what proves it.
FAKE_HOSTILE_SETVARS='#!/usr/bin/env bash
export SELECTOR=
export GATE=
export BUILD_DIR=/nonexistent
export BIN_DIR=bin64
'

# Forcibly unsets ONEAPI_ROOT and SYCL_GATES_SKIP_ONEAPI_SOURCE (so the real
# sourcing branch is always taken) and points SYCL_GATES_SETVARS at the
# hostile fake above instead of the real oneAPI install.
run_gates_fake_setvars() {
    local dir="$1" gate="$2" selector="$3"
    shift 3
    write_exec "$dir/hostile-setvars.sh" "$FAKE_HOSTILE_SETVARS"
    env -u ONEAPI_ROOT -u SYCL_GATES_SKIP_ONEAPI_SOURCE         SYCL_GATES_BIN_DIR="$dir/bin"         SYCL_GATES_CMAKE_CACHE="$dir/CMakeCache.txt"         SYCL_GATES_LDD_CMD="$dir/ldd"         SYCL_GATES_MISTRAL_MODEL="$dir/fake-mistral.gguf"         SYCL_GATES_GPTOSS_MODEL="$dir/fake-gptoss.gguf"         SYCL_GATES_SETTLE_SECONDS=0         SYCL_GATES_SETVARS="$dir/hostile-setvars.sh"         "$GATES_SCRIPT" --build-dir "$dir/build" --gate "$gate" --selector "$selector" "$@"
}

# --- (a) correct digits -> PASS rc=0 ---------------------------------------
case_a_dir="$TMP/a"
setup_case_dir "$case_a_dir"
touch_fake_models "$case_a_dir"
out="$(STUB_MISTRAL_MODE=pass run_gates "$case_a_dir" mistral)"
rc=$?
if [ "$rc" -ne 0 ]; then
    fail "(a) expected rc=0, got rc=$rc; output:\n$out"
elif ! printf '%s\n' "$out" | grep -qE '^GATE mistral PASS rc=0 '; then
    fail "(a) expected a 'GATE mistral PASS rc=0' line; got:\n$out"
else
    echo "PASS: (a) correct digits -> PASS rc=0"
fi

out="$(STUB_GPTOSS_MODE=pass run_gates "$case_a_dir" gptoss)"
rc=$?
if [ "$rc" -ne 0 ]; then
    fail "(a) gptoss expected rc=0, got rc=$rc; output:\n$out"
elif ! printf '%s\n' "$out" | grep -qE '^GATE gptoss PASS rc=0 '; then
    fail "(a) gptoss expected a 'GATE gptoss PASS rc=0' line; got:\n$out"
else
    echo "PASS: (a) gptoss correct digits -> PASS rc=0"
fi

# --- (b) digits only in the prompt echo (GPT-OSS) -> FAIL -------------------
case_b_dir="$TMP/b"
setup_case_dir "$case_b_dir"
touch_fake_models "$case_b_dir"
out="$(STUB_GPTOSS_MODE=prompt-echo run_gates "$case_b_dir" gptoss)"
rc=$?
if [ "$rc" -ne 1 ]; then
    fail "(b) expected rc=1, got rc=$rc; output:\n$out"
elif ! printf '%s\n' "$out" | grep -qE '^GATE gptoss FAIL rc=0 '; then
    fail "(b) expected a 'GATE gptoss FAIL rc=0' line (prompt-echo must not satisfy the anchored digit regex); got:\n$out"
else
    echo "PASS: (b) digits only in the prompt echo -> FAIL"
fi

# --- (b2) digits mid-line, not at line start (Mistral) -> FAIL -------------
# Spec-review finding m1: the Mistral check used to be `grep -qF` (match
# anywhere in the file), asymmetric with the GPT-OSS check and untested. This
# is the Mistral analogue of (b).
case_b2_dir="$TMP/b2"
setup_case_dir "$case_b2_dir"
touch_fake_models "$case_b2_dir"
out="$(STUB_MISTRAL_MODE=mid-line run_gates "$case_b2_dir" mistral)"
rc=$?
if [ "$rc" -ne 1 ]; then
    fail "(b2) expected rc=1, got rc=$rc; output:\n$out"
elif ! printf '%s\n' "$out" | grep -qE '^GATE mistral FAIL rc=0 '; then
    fail "(b2) expected a 'GATE mistral FAIL rc=0' line (mid-line digits must not satisfy the anchored regex); got:\n$out"
else
    echo "PASS: (b2) Mistral digits mid-line, not at line start -> FAIL"
fi

# --- (c) digits correct but an abort line on stderr -> FAIL -----------------
case_c_dir="$TMP/c"
setup_case_dir "$case_c_dir"
touch_fake_models "$case_c_dir"
out="$(STUB_MISTRAL_MODE=abort run_gates "$case_c_dir" mistral)"
rc=$?
if [ "$rc" -ne 1 ]; then
    fail "(c) expected rc=1, got rc=$rc; output:\n$out"
elif ! printf '%s\n' "$out" | grep -qE '^GATE mistral FAIL rc=0 '; then
    fail "(c) expected a 'GATE mistral FAIL rc=0' line (correct stdout, but an abort line on stderr must still fail it); got:\n$out"
else
    echo "PASS: (c) digits correct but an abort line on stderr -> FAIL"
fi

# --- (d) --list-devices: the LIVE dev_mgr abort shape -> rc 77 (SKIP) ------
# B2 finding: the old probe read 2>&1 and grepped bare 'SYCL', so this exact
# shape (abort text containing "SYCL" on stderr, non-zero rc, empty stdout)
# used to read as "device present" and let both model binaries launch.
case_d_dir="$TMP/d"
setup_case_dir "$case_d_dir"
touch_fake_models "$case_d_dir"
out="$(STUB_DEVICE_MODE=none-abort run_gates "$case_d_dir" all 2>"$TMP/d.err")"
rc=$?
if [ "$rc" -ne 77 ]; then
    fail "(d) expected rc=77, got rc=$rc; stdout:\n$out\nstderr:\n$(cat "$TMP/d.err")"
elif ! grep -q '^SKIP:' "$TMP/d.err"; then
    fail "(d) expected a 'SKIP: ...' line on stderr; got:\n$(cat "$TMP/d.err")"
elif ! printf '%s\n' "$out" | grep -qE '^GATE mistral SKIP rc=77 '; then
    fail "(d) expected a 'GATE mistral SKIP rc=77' line; got:\n$out"
else
    echo "PASS: (d) --list-devices abort shape (rc!=0, 'SYCL' only on stderr) -> rc=77 SKIP"
fi

# --- (d2) --list-devices: the FUTURE llama.cpp-1lrh graceful shape ---------
# rc=0, stdout "Available devices:\n  (none)\n", no SYCL text anywhere. The
# probe must recognise this as no-device too, not just the abort shape.
case_d2_dir="$TMP/d2"
setup_case_dir "$case_d2_dir"
touch_fake_models "$case_d2_dir"
out="$(STUB_DEVICE_MODE=none-graceful run_gates "$case_d2_dir" all 2>"$TMP/d2.err")"
rc=$?
if [ "$rc" -ne 77 ]; then
    fail "(d2) expected rc=77, got rc=$rc; stdout:\n$out\nstderr:\n$(cat "$TMP/d2.err")"
elif ! grep -q '^SKIP:' "$TMP/d2.err"; then
    fail "(d2) expected a 'SKIP: ...' line on stderr; got:\n$(cat "$TMP/d2.err")"
elif ! printf '%s\n' "$out" | grep -qE '^GATE mistral SKIP rc=77 '; then
    fail "(d2) expected a 'GATE mistral SKIP rc=77' line; got:\n$out"
else
    echo "PASS: (d2) --list-devices graceful '(none)' shape (rc=0, no SYCL line) -> rc=77 SKIP"
fi

# --- (d3) --list-devices: non-zero rc WITH a valid device line -> rc 77 ---
# Round-2 minor finding 2 (c-u3ji): isolates the "non-zero probe rc is
# authoritative" clause on its own -- (d) alone is caught by the empty-
# stdout half too, so a mutant that drops the `LIST_DEVICES_RC -ne 0` check
# but keeps the stdout-line grep would still pass (d). This case has a
# VALID "  SYCL0: ..." line on stdout, so only the rc check can catch it.
case_d3_dir="$TMP/d3"
setup_case_dir "$case_d3_dir"
touch_fake_models "$case_d3_dir"
out="$(STUB_DEVICE_MODE=rc-nonzero-with-device run_gates "$case_d3_dir" all 2>"$TMP/d3.err")"
rc=$?
if [ "$rc" -ne 77 ]; then
    fail "(d3) expected rc=77, got rc=$rc; stdout:
$out
stderr:
$(cat "$TMP/d3.err")"
elif ! grep -q '^SKIP:' "$TMP/d3.err"; then
    fail "(d3) expected a 'SKIP: ...' line on stderr; got:
$(cat "$TMP/d3.err")"
elif ! printf '%s
' "$out" | grep -qE '^GATE mistral SKIP rc=77 '; then
    fail "(d3) expected a 'GATE mistral SKIP rc=77' line; got:
$out"
else
    echo "PASS: (d3) --list-devices non-zero rc despite a valid device line -> rc=77 SKIP"
fi

# --- (e) missing binary -> rc 1 ---------------------------------------------
case_e_dir="$TMP/e"
setup_case_dir "$case_e_dir"
touch_fake_models "$case_e_dir"
rm -f "$case_e_dir/bin/llama-completion"
out="$(run_gates "$case_e_dir" mistral 2>"$TMP/e.err")"
rc=$?
if [ "$rc" -ne 1 ]; then
    fail "(e) expected rc=1, got rc=$rc; stdout:\n$out\nstderr:\n$(cat "$TMP/e.err")"
elif ! grep -q 'missing binary' "$TMP/e.err"; then
    fail "(e) expected 'missing binary' on stderr; got:\n$(cat "$TMP/e.err")"
else
    echo "PASS: (e) missing binary -> rc=1"
fi

# --- (f) CMakeCache GGML_SYCL:BOOL=OFF -> rc 1 ------------------------------
case_f_dir="$TMP/f"
setup_case_dir "$case_f_dir"
touch_fake_models "$case_f_dir"
write_cmake_cache "$case_f_dir/CMakeCache.txt" "OFF"
out="$(run_gates "$case_f_dir" mistral 2>"$TMP/f.err")"
rc=$?
if [ "$rc" -ne 1 ]; then
    fail "(f) expected rc=1, got rc=$rc; stdout:\n$out\nstderr:\n$(cat "$TMP/f.err")"
elif ! grep -q 'GGML_SYCL is not ON' "$TMP/f.err"; then
    fail "(f) expected a 'GGML_SYCL is not ON' explanation on stderr; got:\n$(cat "$TMP/f.err")"
else
    echo "PASS: (f) GGML_SYCL:BOOL=OFF -> rc=1"
fi

# --- negative control on the ldd/CPU-fallback precondition ------------------
# Not one of the six lettered cases, but the same shape as (f): a build that
# claims GGML_SYCL:BOOL=ON yet does not actually link the SYCL libs must
# still fail closed, exactly like a CMakeCache lie would.
case_g_dir="$TMP/g"
setup_case_dir "$case_g_dir"
touch_fake_models "$case_g_dir"
out="$(STUB_LDD_MODE=insufficient run_gates "$case_g_dir" mistral 2>"$TMP/g.err")"
rc=$?
if [ "$rc" -ne 1 ]; then
    fail "(ldd) expected rc=1, got rc=$rc; stdout:\n$out\nstderr:\n$(cat "$TMP/g.err")"
elif ! grep -q 'CPU-fallback build' "$TMP/g.err"; then
    fail "(ldd) expected a CPU-fallback explanation on stderr; got:\n$(cat "$TMP/g.err")"
else
    echo "PASS: (ldd) insufficient SYCL link count -> rc=1"
fi

# --- model-missing SKIP path (distinct from no-device) ----------------------
case_h_dir="$TMP/h"
setup_case_dir "$case_h_dir"
# deliberately do NOT touch_fake_models: model files stay absent
out="$(run_gates "$case_h_dir" all 2>"$TMP/h.err")"
rc=$?
if [ "$rc" -ne 77 ]; then
    fail "(model-missing) expected rc=77, got rc=$rc; stdout:\n$out\nstderr:\n$(cat "$TMP/h.err")"
elif ! grep -q 'model not found' "$TMP/h.err"; then
    fail "(model-missing) expected 'model not found' on stderr; got:\n$(cat "$TMP/h.err")"
else
    echo "PASS: (model-missing) both models absent -> rc=77 SKIP"
fi

# --- (oneapi-sourcing) real setvars.sh must survive under set -euo pipefail
# B1 finding: `source /opt/intel/oneapi/setvars.sh --force` used to die
# silently (rc=127, zero output -- not even a SKIP line) because
# compiler/latest/env/vars.sh reads ${OCL_ICD_FILENAMES} with no ":-"
# default right after unsetting it, which is a fatal unbound-variable error
# under `set -u` in a SOURCED file that `|| true` cannot catch. This case
# forces the real sourcing branch (env -u ONEAPI_ROOT, no
# SYCL_GATES_SKIP_ONEAPI_SOURCE) and asserts the script still reaches its
# own GATE line afterwards.
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    case_oneapi_dir="$TMP/oneapi"
    setup_case_dir "$case_oneapi_dir"
    touch_fake_models "$case_oneapi_dir"
    out="$(STUB_MISTRAL_MODE=pass run_gates_real_oneapi "$case_oneapi_dir" mistral 2>"$TMP/oneapi.err")"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        fail "(oneapi-sourcing) expected rc=0, got rc=$rc; stdout:\n$out\nstderr:\n$(cat "$TMP/oneapi.err")"
    elif ! printf '%s\n' "$out" | grep -qE '^GATE mistral PASS rc=0 '; then
        fail "(oneapi-sourcing) expected a 'GATE mistral PASS rc=0' line after sourcing the real oneAPI setvars.sh; got stdout:\n$out\nstderr:\n$(cat "$TMP/oneapi.err")"
    else
        echo "PASS: (oneapi-sourcing) real setvars.sh sourced under set -euo pipefail without killing the script"
    fi
else
    echo "SKIP: (oneapi-sourcing) /opt/intel/oneapi/setvars.sh not present on this host -- B1's fix is untestable here"
fi

# --- (oneapi-clobber) a hostile setvars.sh must not move SELECTOR/GATE/BUILD_DIR
# Round-2 blocker (c-u3ji): BUILD_DIR/GATE/SELECTOR are parsed from argv
# BEFORE the oneAPI source runs, so "assign after sourcing" (sufficient for
# BIN_DIR and friends) does not protect them on its own -- they need an
# explicit snapshot-and-restore. The fake setvars here exports SELECTOR=
# (empty), GATE= (empty), BUILD_DIR=/nonexistent, BIN_DIR=bin64. Passing
# --selector level_zero:0 (neither the default level_zero:1 NOR the fake's
# empty clobber) makes an unrestored SELECTOR unambiguous in the GATE line.
case_clobber_dir="$TMP/clobber"
setup_case_dir "$case_clobber_dir"
touch_fake_models "$case_clobber_dir"
out="$(STUB_MISTRAL_MODE=pass run_gates_fake_setvars "$case_clobber_dir" mistral level_zero:0 2>"$TMP/clobber.err")"
rc=$?
if [ "$rc" -ne 0 ]; then
    fail "(oneapi-clobber) expected rc=0, got rc=$rc; stdout:
$out
stderr:
$(cat "$TMP/clobber.err")"
elif ! printf '%s
' "$out" | grep -qE '^GATE mistral PASS rc=0 selector=level_zero:0 '; then
    fail "(oneapi-clobber) expected 'GATE mistral PASS rc=0 selector=level_zero:0' (the gate we asked for, the selector we passed) -- a hostile setvars.sh must not move either; got stdout:
$out
stderr:
$(cat "$TMP/clobber.err")"
else
    echo "PASS: (oneapi-clobber) hostile setvars.sh cannot move the selector, the gate, or the binary directory"
fi

# --- (stderr-only-mistral) correct digits on STDERR only -> FAIL -----------
# Round-2 minor finding 1: the pass check must read the .out file, not a
# combined or either stream.
case_serr_m_dir="$TMP/serr-m"
setup_case_dir "$case_serr_m_dir"
touch_fake_models "$case_serr_m_dir"
out="$(STUB_MISTRAL_MODE=stderr-only run_gates "$case_serr_m_dir" mistral)"
rc=$?
if [ "$rc" -ne 1 ]; then
    fail "(stderr-only-mistral) expected rc=1, got rc=$rc; output:
$out"
elif ! printf '%s
' "$out" | grep -qE '^GATE mistral FAIL rc=0 '; then
    fail "(stderr-only-mistral) expected 'GATE mistral FAIL rc=0' (correct digits on stderr must not satisfy the stdout-only check); got:
$out"
else
    echo "PASS: (stderr-only-mistral) correct digits on stderr only -> FAIL"
fi

# --- (stderr-only-gptoss) correct digits on STDERR only -> FAIL ------------
case_serr_g_dir="$TMP/serr-g"
setup_case_dir "$case_serr_g_dir"
touch_fake_models "$case_serr_g_dir"
out="$(STUB_GPTOSS_MODE=stderr-only run_gates "$case_serr_g_dir" gptoss)"
rc=$?
if [ "$rc" -ne 1 ]; then
    fail "(stderr-only-gptoss) expected rc=1, got rc=$rc; output:
$out"
elif ! printf '%s
' "$out" | grep -qE '^GATE gptoss FAIL rc=0 '; then
    fail "(stderr-only-gptoss) expected 'GATE gptoss FAIL rc=0' (correct digits on stderr must not satisfy the stdout-only check); got:
$out"
else
    echo "PASS: (stderr-only-gptoss) correct digits on stderr only -> FAIL"
fi

echo "---"
if [ "$FAILURES" -eq 0 ]; then
    echo "test-sycl-canonical-gates-script: all cases passed"
    exit 0
else
    echo "test-sycl-canonical-gates-script: $FAILURES case(s) failed"
    exit 1
fi
