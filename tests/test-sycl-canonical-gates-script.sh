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
#   (e)  missing binary                               -> rc 1
#   (f)  CMakeCache GGML_SYCL:BOOL=OFF                 -> rc 1
#   (oneapi-sourcing) real /opt/intel/oneapi/setvars.sh survives under
#        `set -euo pipefail` (B1 finding: it used to die silently, rc=127,
#        zero output, on OCL_ICD_FILENAMES under `set -u`)
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

echo "---"
if [ "$FAILURES" -eq 0 ]; then
    echo "test-sycl-canonical-gates-script: all cases passed"
    exit 0
else
    echo "test-sycl-canonical-gates-script: $FAILURES case(s) failed"
    exit 1
fi
