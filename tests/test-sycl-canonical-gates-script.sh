#!/usr/bin/env bash
# tests/test-sycl-canonical-gates-script.sh (llama.cpp-rou3)
#
# Stub-driven unit test of scripts/sycl-canonical-gates.sh's own logic. This
# is CPU-only: it never touches a GPU and never loads a real model. All
# "binaries" are shell scripts controlled by this file, wired through the
# script's SYCL_GATES_* overrides (SYCL_GATES_BIN_DIR, _CMAKE_CACHE, _LDD_CMD,
# _MISTRAL_MODEL, _GPTOSS_MODEL, _SKIP_ONEAPI_SOURCE, _SETTLE_SECONDS,
# _SETVARS).
#
# Cases, matching the task spec plus discriminators added on review:
#   (a)  correct digits                              -> PASS rc=0
#   (b)  digits only in the prompt echo (GPT-OSS)     -> FAIL
#   (b2) digits mid-line, not at line start (Mistral) -> FAIL
#   (c)  digits correct but an abort line on stderr   -> FAIL
#   (d)  --list-devices: the CURRENTLY LIVE dev_mgr abort shape
#        (rc!=0, empty stdout, "SYCL" text only on stderr)   -> rc 77 (SKIP)
#   (d2) --list-devices: the FUTURE llama.cpp-1lrh graceful shape
#        (rc=0, stdout "Available devices:\n  (none)\n")     -> rc 77 (SKIP)
#   (d3) --list-devices: non-zero rc WITH a valid device line -> rc 77 (SKIP)
#   (linker-failure) --list-devices exits 127 (dynamic-linker failure, a
#        broken oneAPI install) -> rc 1, fail-closed, NOT confused with
#        no-device
#   (e)  missing binary                               -> rc 1
#   (f)  CMakeCache GGML_SYCL:BOOL=OFF                 -> rc 1
#   (ldd) insufficient SYCL link count (CPU-fallback build) -> rc 1
#   (model-missing) both model files absent           -> rc 77 (SKIP)
#   (oneapi-sourcing) real /opt/intel/oneapi/setvars.sh survives under
#        `set -euo pipefail`
#   (oneapi-clobber) a hostile setvars.sh cannot move BUILD_DIR/GATE/
#        SELECTOR/ROOT_DIR
#   (stderr-only-mistral) correct digits on STDERR only -> FAIL
#   (stderr-only-gptoss)  same, GPT-OSS side            -> FAIL
#
# Run directly: bash tests/test-sycl-canonical-gates-script.sh; echo rc=$?
#
# To add a new case: build (or reuse) a case dir with setup_case_dir, set
# whichever STUB_*_MODE env vars select the stub behaviour you need, call
# run_gates (or one of its siblings below) to capture $out/$rc, and finish
# with one call to expect_gate (see its own comment for the parameter
# shape). Only reach past expect_gate by hand when a case needs MORE than
# "rc + a GATE-line regex + an optional stderr substring" -- see
# (oneapi-clobber) below for the pattern (call expect_gate first for the
# shared shape, then add the case-specific checks after).

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

# fail MESSAGE
# MESSAGE may contain literal "\n" escapes (not real newline bytes) -- %b
# interprets those, so every call site can stay on one physical line
# without losing multi-line diagnostics. Real embedded newlines (e.g. from
# $(cat ...)) also print correctly through %b; it only ever transforms
# backslash-escape sequences, never bytes that are already newlines.
fail() {
    printf 'FAIL: %b\n' "$*" >&2
    FAILURES=$((FAILURES + 1))
}

# expect_gate LABEL EXPECTED_RC OUT RC [GATE_REGEX [ERR_FILE [STDERR_SUBSTR]]]
#
# The shared assertion shape used by (nearly) every case: the overall exit
# code, optionally a "GATE ..." line appearing in OUT (pass "" to skip --
# used by cases that fail_closed before any GATE line is ever printed:
# (e)/(f)/(ldd)/(linker-failure)), and optionally a substring that must
# appear in the ERR_FILE (pass both ERR_FILE and STDERR_SUBSTR, or neither).
# Prints "PASS: (LABEL) ..." on success, calls fail() with full diagnostics
# on the first mismatch. Returns 0/1 so a caller can chain additional,
# case-specific checks after a shared-shape pass (see (oneapi-clobber)).
expect_gate() {
    local label="$1" expected_rc="$2" out="$3" rc="$4"
    local gate_regex="${5:-}" err_file="${6:-}" stderr_substr="${7:-}"

    if [ "$rc" -ne "$expected_rc" ]; then
        fail "($label) expected rc=$expected_rc, got rc=$rc; stdout:\n$out"
        return 1
    fi
    if [ -n "$gate_regex" ] && ! printf '%s\n' "$out" | grep -qE "$gate_regex"; then
        fail "($label) expected a line matching /$gate_regex/; got:\n$out"
        return 1
    fi
    if [ -n "$stderr_substr" ]; then
        if [ -z "$err_file" ] || ! grep -qF "$stderr_substr" "$err_file"; then
            fail "($label) expected '$stderr_substr' on stderr; got:\n$(cat "$err_file" 2>/dev/null)"
            return 1
        fi
    fi
    echo "PASS: ($label) rc=$expected_rc${gate_regex:+, /$gate_regex/}${stderr_substr:+, stderr has [$stderr_substr]}"
    return 0
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
#   rc-nonzero-with-device -- a valid device line on stdout, but the process
#                    still exits non-zero. Isolates the "non-zero probe rc
#                    is authoritative" clause from the "no matching stdout
#                    line" clause.
#   linker-failure -- exits 127 with a dynamic-linker-style message on
#                    stderr, simulating a broken oneAPI install. Must be
#                    routed to fail_closed (rc=1), never treated as
#                    no-device.
# shellcheck disable=SC2016
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
            printf "Available devices:\n  SYCL0: Fake Battlemage (16384 MiB, 16384 MiB free)\n"
            exit 3
            ;;
        linker-failure)
            printf "llama-completion: error while loading shared libraries: libsycl.so.7: cannot open shared object file: No such file or directory\n" >&2
            exit 127
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
# shellcheck disable=SC2016
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
# shellcheck disable=SC2016
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
# gracefully if it does not, since the real-sourcing fix is then untestable
# here.
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

# Creates a tiny, throwaway git repo -- deliberately NOT this checkout --
# with exactly one commit, so the ROOT_DIR-clobber case below has a real
# repository to redirect ROOT_DIR to, with its own real, DIFFERENT
# `git rev-parse --short HEAD` to tell apart from this tree's actual HEAD.
make_scratch_git_repo() {
    local dir="$1"
    mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" config user.email "test@example.invalid"
    git -C "$dir" config user.name "test"
    : > "$dir/marker"
    git -C "$dir" add marker
    git -C "$dir" commit -q -m "scratch commit, not this checkout"
}

# A hostile, FAKE setvars.sh: exports empty/wrong values for every name this
# script parses from argv (SELECTOR, GATE, BUILD_DIR) plus BIN_DIR (the
# already-proven collision) plus ROOT_DIR (ROOT_DIR is assigned before the
# source and consumed after it at the SHA line, so it needs the same
# snapshot-and-restore treatment as the other three), simulating the worst
# case an oneAPI component could do. $1 = destination path, $2 = the
# scratch git repo dir (from make_scratch_git_repo) that a clobbered
# ROOT_DIR would redirect the SHA computation into.
write_hostile_setvars() {
    local dest="$1" fake_root_dir="$2"
    write_exec "$dest" "#!/usr/bin/env bash
export SELECTOR=
export GATE=
export BUILD_DIR=/nonexistent
export BIN_DIR=bin64
export ROOT_DIR=$fake_root_dir
"
}

# Forcibly unsets ONEAPI_ROOT and SYCL_GATES_SKIP_ONEAPI_SOURCE (so the real
# sourcing branch is always taken) and points SYCL_GATES_SETVARS at the
# hostile fake above instead of the real oneAPI install.
run_gates_fake_setvars() {
    local dir="$1" gate="$2" selector="$3" fake_root_dir="$4"
    shift 4
    write_hostile_setvars "$dir/hostile-setvars.sh" "$fake_root_dir"
    env -u ONEAPI_ROOT -u SYCL_GATES_SKIP_ONEAPI_SOURCE \
        SYCL_GATES_BIN_DIR="$dir/bin" \
        SYCL_GATES_CMAKE_CACHE="$dir/CMakeCache.txt" \
        SYCL_GATES_LDD_CMD="$dir/ldd" \
        SYCL_GATES_MISTRAL_MODEL="$dir/fake-mistral.gguf" \
        SYCL_GATES_GPTOSS_MODEL="$dir/fake-gptoss.gguf" \
        SYCL_GATES_SETTLE_SECONDS=0 \
        SYCL_GATES_SETVARS="$dir/hostile-setvars.sh" \
        "$GATES_SCRIPT" --build-dir "$dir/build" --gate "$gate" --selector "$selector" "$@"
}

# --- (a) correct digits -> PASS rc=0 ---------------------------------------
case_a_dir="$TMP/a"
setup_case_dir "$case_a_dir"
touch_fake_models "$case_a_dir"
out="$(STUB_MISTRAL_MODE=pass run_gates "$case_a_dir" mistral)"
rc=$?
expect_gate "a" 0 "$out" "$rc" '^GATE mistral PASS rc=0 '

out="$(STUB_GPTOSS_MODE=pass run_gates "$case_a_dir" gptoss)"
rc=$?
expect_gate "a-gptoss" 0 "$out" "$rc" '^GATE gptoss PASS rc=0 '

# --- (b) digits only in the prompt echo (GPT-OSS) -> FAIL -------------------
case_b_dir="$TMP/b"
setup_case_dir "$case_b_dir"
touch_fake_models "$case_b_dir"
out="$(STUB_GPTOSS_MODE=prompt-echo run_gates "$case_b_dir" gptoss)"
rc=$?
expect_gate "b" 1 "$out" "$rc" '^GATE gptoss FAIL rc=0 '

# --- (b2) digits mid-line, not at line start (Mistral) -> FAIL -------------
# The Mistral analogue of (b): the check must be anchored, matching the
# GPT-OSS check's own anchoring.
case_b2_dir="$TMP/b2"
setup_case_dir "$case_b2_dir"
touch_fake_models "$case_b2_dir"
out="$(STUB_MISTRAL_MODE=mid-line run_gates "$case_b2_dir" mistral)"
rc=$?
expect_gate "b2" 1 "$out" "$rc" '^GATE mistral FAIL rc=0 '

# --- (c) digits correct but an abort line on stderr -> FAIL -----------------
case_c_dir="$TMP/c"
setup_case_dir "$case_c_dir"
touch_fake_models "$case_c_dir"
out="$(STUB_MISTRAL_MODE=abort run_gates "$case_c_dir" mistral)"
rc=$?
expect_gate "c" 1 "$out" "$rc" '^GATE mistral FAIL rc=0 '

# --- (d) --list-devices: the LIVE dev_mgr abort shape -> rc 77 (SKIP) ------
# The no-device probe must read stdout only: this exact shape (abort text
# containing "SYCL" on stderr, non-zero rc, empty stdout) must not read as
# "device present".
case_d_dir="$TMP/d"
setup_case_dir "$case_d_dir"
touch_fake_models "$case_d_dir"
out="$(STUB_DEVICE_MODE=none-abort run_gates "$case_d_dir" all 2>"$TMP/d.err")"
rc=$?
expect_gate "d" 77 "$out" "$rc" '^GATE mistral SKIP rc=77 ' "$TMP/d.err" 'SKIP:'

# --- (d2) --list-devices: the FUTURE llama.cpp-1lrh graceful shape ---------
# rc=0, stdout "Available devices:\n  (none)\n", no SYCL text anywhere. The
# probe must recognise this as no-device too, not just the abort shape.
case_d2_dir="$TMP/d2"
setup_case_dir "$case_d2_dir"
touch_fake_models "$case_d2_dir"
out="$(STUB_DEVICE_MODE=none-graceful run_gates "$case_d2_dir" all 2>"$TMP/d2.err")"
rc=$?
expect_gate "d2" 77 "$out" "$rc" '^GATE mistral SKIP rc=77 ' "$TMP/d2.err" 'SKIP:'

# --- (d3) --list-devices: non-zero rc WITH a valid device line -> rc 77 ---
# Isolates the "non-zero probe rc is authoritative" clause on its own -- (d)
# alone is caught by its empty-stdout half too, so a mutant that drops the
# rc check but keeps the stdout-line grep would still pass (d). This case
# has a VALID "  SYCL0: ..." line on stdout, so only the rc check can catch
# it.
case_d3_dir="$TMP/d3"
setup_case_dir "$case_d3_dir"
touch_fake_models "$case_d3_dir"
out="$(STUB_DEVICE_MODE=rc-nonzero-with-device run_gates "$case_d3_dir" all 2>"$TMP/d3.err")"
rc=$?
expect_gate "d3" 77 "$out" "$rc" '^GATE mistral SKIP rc=77 ' "$TMP/d3.err" 'SKIP:'

# --- (linker-failure) --list-devices exits 127 -> rc 1, fail closed --------
# rc 126/127 mean the probe binary itself failed to EXECUTE (a broken
# oneAPI install / dynamic-linker failure), which is a real infrastructure
# failure, not "no SYCL device" -- conflating the two would silently report
# a broken host as "nothing to test here" instead of failing loudly. This
# must NOT land in the same rc=77 bucket as (d)/(d2)/(d3).
case_linker_dir="$TMP/linker"
setup_case_dir "$case_linker_dir"
touch_fake_models "$case_linker_dir"
out="$(STUB_DEVICE_MODE=linker-failure run_gates "$case_linker_dir" all 2>"$TMP/linker.err")"
rc=$?
expect_gate "linker-failure" 1 "$out" "$rc" "" "$TMP/linker.err" '126/127'

# --- (e) missing binary -> rc 1 ---------------------------------------------
case_e_dir="$TMP/e"
setup_case_dir "$case_e_dir"
touch_fake_models "$case_e_dir"
rm -f "$case_e_dir/bin/llama-completion"
out="$(run_gates "$case_e_dir" mistral 2>"$TMP/e.err")"
rc=$?
expect_gate "e" 1 "$out" "$rc" "" "$TMP/e.err" 'missing binary'

# --- (f) CMakeCache GGML_SYCL:BOOL=OFF -> rc 1 ------------------------------
case_f_dir="$TMP/f"
setup_case_dir "$case_f_dir"
touch_fake_models "$case_f_dir"
write_cmake_cache "$case_f_dir/CMakeCache.txt" "OFF"
out="$(run_gates "$case_f_dir" mistral 2>"$TMP/f.err")"
rc=$?
expect_gate "f" 1 "$out" "$rc" "" "$TMP/f.err" 'GGML_SYCL is not ON'

# --- (ldd) insufficient SYCL link count (CPU-fallback build) -> rc 1 -------
# Not one of the six original spec-lettered cases, but the same shape as
# (f): a build that claims GGML_SYCL:BOOL=ON yet does not actually link the
# SYCL libs must still fail closed, exactly like a CMakeCache lie would.
case_ldd_dir="$TMP/ldd"
setup_case_dir "$case_ldd_dir"
touch_fake_models "$case_ldd_dir"
out="$(STUB_LDD_MODE=insufficient run_gates "$case_ldd_dir" mistral 2>"$TMP/ldd.err")"
rc=$?
expect_gate "ldd" 1 "$out" "$rc" "" "$TMP/ldd.err" 'CPU-fallback build'

# --- (model-missing) both models absent -> rc=77 SKIP -----------------------
case_model_missing_dir="$TMP/model-missing"
setup_case_dir "$case_model_missing_dir"
# deliberately do NOT touch_fake_models: model files stay absent
out="$(run_gates "$case_model_missing_dir" all 2>"$TMP/model-missing.err")"
rc=$?
expect_gate "model-missing" 77 "$out" "$rc" "" "$TMP/model-missing.err" 'model not found'

# --- (oneapi-sourcing) real setvars.sh must survive under set -euo pipefail
# `source /opt/intel/oneapi/setvars.sh --force` used to die silently
# (rc=127, zero output -- not even a SKIP line) because
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
    expect_gate "oneapi-sourcing" 0 "$out" "$rc" '^GATE mistral PASS rc=0 '
else
    echo "SKIP: (oneapi-sourcing) /opt/intel/oneapi/setvars.sh not present on this host -- untestable here"
fi

# --- (oneapi-clobber) a hostile setvars.sh must not move BUILD_DIR/GATE/SELECTOR/ROOT_DIR
# BUILD_DIR/GATE/SELECTOR are parsed from argv BEFORE the oneAPI source
# runs, so "assign after sourcing" (sufficient for BIN_DIR and friends)
# does not protect them on its own -- they need an explicit
# snapshot-and-restore. ROOT_DIR has the exact same shape (assigned before
# the source, consumed after it at the SHA line). The fake setvars here
# exports SELECTOR= (empty), GATE= (empty), BUILD_DIR=/nonexistent,
# BIN_DIR=bin64, and ROOT_DIR=<a scratch git repo, NOT this checkout>.
# Passing --selector level_zero:0 (neither the default level_zero:1 NOR the
# fake's empty clobber) makes an unrestored SELECTOR unambiguous in the
# GATE line; the scratch repo's own distinct HEAD sha does the same for
# ROOT_DIR -- if ROOT_DIR were left clobbered, the SHA computation would
# run `git -C <scratch repo> rev-parse --short HEAD` and report THAT
# commit, not this tree's.
case_clobber_dir="$TMP/clobber"
setup_case_dir "$case_clobber_dir"
touch_fake_models "$case_clobber_dir"
case_clobber_fake_repo="$TMP/clobber-fake-repo"
make_scratch_git_repo "$case_clobber_fake_repo"
real_sha="$(cd "$ROOT_DIR" && git rev-parse --short HEAD)"
fake_sha="$(cd "$case_clobber_fake_repo" && git rev-parse --short HEAD)"
# The fixture is worthless as a discriminator unless both shas exist and
# differ -- assert that BEFORE relying on them below, the same way a
# positive control must be checked live rather than assumed.
if [ -z "$real_sha" ] || [ -z "$fake_sha" ]; then
    fail "(oneapi-clobber) fixture broken: real_sha='$real_sha' fake_sha='$fake_sha' -- both must be non-empty"
elif [ "$real_sha" = "$fake_sha" ]; then
    fail "(oneapi-clobber) fixture broken: real_sha and fake_sha are identical ('$real_sha') -- the scratch repo does not have a distinct HEAD, so this case cannot discriminate a clobbered ROOT_DIR from a correctly-restored one"
else
    out="$(STUB_MISTRAL_MODE=pass run_gates_fake_setvars "$case_clobber_dir" mistral level_zero:0 "$case_clobber_fake_repo" 2>"$TMP/clobber.err")"
    rc=$?
    if expect_gate "oneapi-clobber" 0 "$out" "$rc" '^GATE mistral PASS rc=0 selector=level_zero:0 '; then
        if printf '%s\n' "$out" | grep -qF "sha=$fake_sha"; then
            fail "(oneapi-clobber) GATE line reports sha=$fake_sha -- the SCRATCH repo's HEAD, not this tree's ($real_sha). ROOT_DIR was clobbered and not restored; got:\n$out"
        elif ! printf '%s\n' "$out" | grep -qF "sha=$real_sha"; then
            fail "(oneapi-clobber) expected the GATE line to report this tree's real sha=$real_sha; got:\n$out"
        else
            echo "PASS: (oneapi-clobber) hostile setvars.sh moved none of BUILD_DIR (mkdir under set -e succeeded), GATE (mistral actually ran), SELECTOR (selector=level_zero:0), or ROOT_DIR (sha=$real_sha, not the scratch repo's $fake_sha)"
        fi
    fi
fi

# --- (stderr-only-mistral) correct digits on STDERR only -> FAIL -----------
# The pass check must read the .out file, not a combined or either stream.
case_serr_m_dir="$TMP/serr-m"
setup_case_dir "$case_serr_m_dir"
touch_fake_models "$case_serr_m_dir"
out="$(STUB_MISTRAL_MODE=stderr-only run_gates "$case_serr_m_dir" mistral)"
rc=$?
expect_gate "stderr-only-mistral" 1 "$out" "$rc" '^GATE mistral FAIL rc=0 '

# --- (stderr-only-gptoss) correct digits on STDERR only -> FAIL ------------
case_serr_g_dir="$TMP/serr-g"
setup_case_dir "$case_serr_g_dir"
touch_fake_models "$case_serr_g_dir"
out="$(STUB_GPTOSS_MODE=stderr-only run_gates "$case_serr_g_dir" gptoss)"
rc=$?
expect_gate "stderr-only-gptoss" 1 "$out" "$rc" '^GATE gptoss FAIL rc=0 '

echo "---"
if [ "$FAILURES" -eq 0 ]; then
    echo "test-sycl-canonical-gates-script: all cases passed"
    exit 0
else
    echo "test-sycl-canonical-gates-script: $FAILURES case(s) failed"
    exit 1
fi
