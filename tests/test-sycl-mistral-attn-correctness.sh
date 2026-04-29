#!/usr/bin/env bash
#
# Correctness gate for the SYCL Mistral-7B attention fast-path work.
#
# This is intentionally model-backed and is not wired into default CTest:
# it requires the local Mistral GGUF and an Intel SYCL runtime. Use it before
# treating a PP512/TG128 benchmark setting as default-safe.
#
# Usage:
#   ./tests/test-sycl-mistral-attn-correctness.sh
#   ./tests/test-sycl-mistral-attn-correctness.sh --require-fa-off
#
# Environment:
#   LLAMA_BIN_DIR          Path to llama binaries (default: ./build-sycl/bin, then ./build/bin)
#   MODEL_DIR              Model directory (default: /Storage/GenAI/models)
#   MISTRAL_MODEL          Full model path override
#   ONEAPI_DEVICE_SELECTOR SYCL device selector (default: level_zero:0)
#

set -euo pipefail

REQUIRE_FA_OFF=false
for arg in "$@"; do
    case "$arg" in
        --require-fa-off)
            REQUIRE_FA_OFF=true
            ;;
        -h|--help)
            sed -n '1,28p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

if [ -f /opt/intel/oneapi/setvars.sh ]; then
    set +e +u
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/tmp/oneapi-setvars.log 2>&1
    set -euo pipefail
fi

if [ -n "${LLAMA_BIN_DIR:-}" ]; then
    :
elif [ -x ./build-sycl/bin/llama-completion ]; then
    LLAMA_BIN_DIR=./build-sycl/bin
elif [ -x ./build/bin/llama-completion ]; then
    LLAMA_BIN_DIR=./build/bin
else
    LLAMA_BIN_DIR=./build-sycl/bin
fi

MODEL_DIR="${MODEL_DIR:-/Storage/GenAI/models}"
MISTRAL_MODEL="${MISTRAL_MODEL:-${MODEL_DIR}/mistral-7b-v0.1.Q4_0.gguf}"
DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
EXPECTED="6, 7, 8, 9, 10"
PROMPT="1, 2, 3, 4, 5,"

if [ ! -x "${LLAMA_BIN_DIR}/llama-completion" ]; then
    echo "ERROR: llama-completion not found at ${LLAMA_BIN_DIR}/llama-completion" >&2
    exit 1
fi

if [ ! -f "${MISTRAL_MODEL}" ]; then
    echo "ERROR: Mistral model not found at ${MISTRAL_MODEL}" >&2
    exit 1
fi

run_completion() {
    local flash_attn="$1"
    local out_file="$2"

    ONEAPI_DEVICE_SELECTOR="${DEVICE_SELECTOR}" \
    "${LLAMA_BIN_DIR}/llama-completion" \
        -m "${MISTRAL_MODEL}" \
        -ngl 99 \
        --flash-attn "${flash_attn}" \
        -c 2048 \
        -p "${PROMPT}" \
        --no-display-prompt \
        -n 15 \
        --seed 42 \
        --temp 0 \
        >"${out_file}" 2>"${out_file}.stderr"
}

contains_expected() {
    local out_file="$1"
    # SYCL diagnostics can be interleaved between streamed token bytes, so
    # accept the deterministic continuation with arbitrary log text between
    # token fragments.
    perl -0ne 'exit(/6.*?,.*?7.*?,.*?8.*?,.*?9.*?,.*?1.*?0/s ? 0 : 1)' "${out_file}"
}

print_tail() {
    local label="$1"
    local out_file="$2"
    echo "---- ${label} stdout ----"
    cat "${out_file}"
    echo
    echo "---- ${label} stderr tail ----"
    tail -n 40 "${out_file}.stderr"
    echo "-----------------------"
}

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

echo "SYCL Mistral attention correctness"
echo "  binary: ${LLAMA_BIN_DIR}/llama-completion"
echo "  model:  ${MISTRAL_MODEL}"
echo "  device: ${DEVICE_SELECTOR}"
echo "  expected continuation: ${EXPECTED}"
echo

fa_on_out="${tmp_dir}/fa_on.log"
echo "Running --flash-attn on..."
run_completion on "${fa_on_out}"
if contains_expected "${fa_on_out}"; then
    echo "PASS: --flash-attn on produced the expected continuation"
else
    echo "FAIL: --flash-attn on did not produce the expected continuation" >&2
    print_tail "flash-attn on" "${fa_on_out}" >&2
    exit 1
fi

fa_off_out="${tmp_dir}/fa_off.log"
echo "Running --flash-attn off..."
if run_completion off "${fa_off_out}" && contains_expected "${fa_off_out}"; then
    echo "PASS: --flash-attn off produced the expected continuation"
    exit 0
fi

echo "FAIL: --flash-attn off is not default-safe for this prompt/context"
print_tail "flash-attn off" "${fa_off_out}"

if [ "${REQUIRE_FA_OFF}" = true ]; then
    exit 1
fi

echo "NOTE: exiting 0 because --require-fa-off was not set; use that flag for the C1 blocker gate."
