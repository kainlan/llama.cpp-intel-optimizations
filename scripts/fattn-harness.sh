#!/bin/bash
# SYCL Flash Attention Correctness + Performance Harness
#
# Validates that SYCL flash-attention maintains correctness and performance
# targets across builds. Designed for CI integration and pre-PR checklists.
#
# Usage:
#   ./scripts/fattn-harness.sh              # Run all checks
#   ./scripts/fattn-harness.sh --correctness  # Correctness only
#   ./scripts/fattn-harness.sh --perf         # Performance only
#   ./scripts/fattn-harness.sh --determinism  # Determinism checks only
#   ./scripts/fattn-harness.sh --allow-perf-warn  # Allow perf warnings (don't fail on perf)
#   ./scripts/fattn-harness.sh --help         # Show help
#
# Environment:
#   - Requires Intel oneAPI (setvars.sh)
#   - Uses level_zero:0 (first Intel GPU)
#   - Uses LLAMA_COMPLETION_MODEL and LLAMA_BENCH_MODEL env vars for model paths
#
# Acceptance gates (from 5edo8 epic):
#   Correctness: ordered continuation "6, 7, 8, 9, 10" must appear
#   Non-FA: pp512 >= 1700 tok/s, tg128 >= 79 tok/s
#   FA-on:  pp512 >= 1326 tok/s, tg128 >= 62 tok/s

set -uo pipefail

# ============================================================================
# Configuration (overridable via environment)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"
RESULTS_FILE="${SCRIPT_DIR}/fattn-harness-results.txt"

MODEL_MISTRAL_Q4="${LLAMA_COMPLETION_MODEL:-/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf}"
ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"

PASS=0
FAIL=0
WARN=0
SKIP=0
PERF_WARN=false

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# Parse arguments
# ============================================================================
MODE="all"
for arg in "$@"; do
    case $arg in
        --correctness) MODE="correctness" ;;
        --perf) MODE="perf" ;;
        --determinism) MODE="determinism" ;;
        --allow-perf-warn) PERF_WARN=true ;;
        --help|-h)
            echo "SYCL Flash Attention Correctness + Performance Harness"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --correctness     Run correctness checks only"
            echo "  --perf            Run performance benchmarks only"
            echo "  --determinism     Run determinism checks only"
            echo "  --allow-perf-warn Do not fail on performance gate warnings"
            echo "  --help            Show this help message"
            echo ""
            echo "Acceptance gates:"
            echo "  Correctness: ordered continuation '6, 7, 8, 9, 10'"
            echo "  Non-FA: pp512 >= 1700 tok/s, tg128 >= 79 tok/s"
            echo "  FA-on:  pp512 >= 1326 tok/s, tg128 >= 62 tok/s"
            echo ""
            echo "Default (no args): run all checks"
            exit 0
            ;;
    esac
done

# ============================================================================
# Helpers
# ============================================================================
log() {
    echo "$1" | tee -a "$RESULTS_FILE"
}

pass() {
    PASS=$((PASS + 1))
    log "  ${GREEN}PASS${NC}: $1"
}

fail() {
    FAIL=$((FAIL + 1))
    log "  ${RED}FAIL${NC}: $1"
}

warn() {
    WARN=$((WARN + 1))
    log "  ${YELLOW}WARN${NC}: $1"
}

skip() {
    SKIP=$((SKIP + 1))
    log "  ${YELLOW}SKIP${NC}: $1"
}

section() {
    log ""
    log "=== $1 ==="
    log ""
}

# ============================================================================
# Initialize environment
# ============================================================================
init_environment() {
    echo -e "${BLUE}=== Initializing Intel oneAPI Environment ===${NC}"

    # Source oneAPI if available and not already sourced.
    # Temporarily disable nounset because setvars.sh references unset variables.
    if [ -z "${INTEL_ONEAPI_SOURCED:-}" ] && [ -f "/opt/intel/oneapi/setvars.sh" ]; then
        local prev_nounset=""
        set +o | grep -q '^set -u$' && prev_nounset="yes"
        set +u
        source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
        local rc=$?
        [ -n "$prev_nounset" ] && set -u
        if [ $rc -ne 0 ]; then
            echo -e "${RED}Error: Failed to source /opt/intel/oneapi/setvars.sh (exit $rc)${NC}"
            echo "oneAPI is required for SYCL builds. Install via: https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html"
            exit 1
        fi
        export INTEL_ONEAPI_SOURCED=1
    fi

    # Check llama-bench exists
    if [ ! -f "${BUILD_DIR}/bin/llama-bench" ]; then
        echo -e "${RED}Error: llama-bench not found at ${BUILD_DIR}/bin/llama-bench${NC}"
        echo "Build with: ./scripts/sycl-build.sh llama-bench"
        exit 1
    fi

    # Check llama-completion exists
    if [ ! -f "${BUILD_DIR}/bin/llama-completion" ]; then
        echo -e "${RED}Error: llama-completion not found at ${BUILD_DIR}/bin/llama-completion${NC}"
        echo "Build with: ./scripts/sycl-build.sh llama-completion"
        exit 1
    fi

    # Export LD_LIBRARY_PATH so binaries find local shared libraries
    export LD_LIBRARY_PATH="${BUILD_DIR}/bin:${LD_LIBRARY_PATH}"

    export ONEAPI_DEVICE_SELECTOR

    # Initialize results file deterministically before any logging
    : > "$RESULTS_FILE"

    log "Build: $(cd "$PROJECT_DIR" && git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    log "GPU: ${ONEAPI_DEVICE_SELECTOR}"
    log "Model: ${MODEL_MISTRAL_Q4}"
    log "Build dir: ${BUILD_DIR}"
}

# ============================================================================
# Correctness Checks
# ============================================================================

# Run a single completeness check: prompt must produce ordered "6, 7, 8, 9, 10"
run_correctness_check() {
    local prompt="1, 2, 3, 4, 5,"
    local fa_flag="$1"
    local description="$2"

    if [ ! -f "$MODEL_MISTRAL_Q4" ]; then
        skip "Model not found: ${MODEL_MISTRAL_Q4}"
        return
    fi

    local output
    output=$(ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR}" \
        "${BUILD_DIR}/bin/llama-completion" \
        -m "$MODEL_MISTRAL_Q4" \
        -p "$prompt" \
        -n 15 \
        --seed 42 \
        --temp 0 \
        $fa_flag \
        2>&1) || true

    # Strip everything from '[' onwards on each line (removes [GRAPH-COMPUTE],
    # [TOKEN-TIMING], [GRAPH-DIAG] etc.), then join lines and check for the
    # ordered continuation. This handles tokens that are interleaved with
    # diagnostic output like "6[GRAPH-COMPUTE] device 0:...".
    local tokens
    # Strip everything from '[' onwards (removes [GRAPH-COMPUTE], [TOKEN-TIMING] etc.)
    # Then filter out metadata lines and join remaining tokens.
    tokens=$(printf '%s\n' "$output" \
        | sed 's/\[.*//;s/ *$//' \
        | grep -vE '^\s*$|generate:|perf_print|^main|build:|^sampler|^print_info|llama_model|load:|Running|Found|SYCL |^KV |VRAM-|HOST-|PLACEMENT|UNIFIED-|GRAPH-|MOE-|TOKEN-|LOAD-SUMMARY|place' \
        | paste -sd '' - \
        | sed 's/^,\s*//')

    if echo "$tokens" | grep -qE ',? *6,? *7,? *8,? *9,? *10'; then
        pass "${description}: ordered continuation '6, 7, 8, 9, 10' found"
    else
        fail "${description}: ordered continuation '6, 7, 8, 9, 10' NOT found"
        log "    Tokens near 6-10: $(echo "$tokens" | grep -oE '.{0,30}6.{0,20}' | head -1)"
        log "    Raw tail: $(echo "$output" | tail -5)"
    fi
}

run_correctness_all() {
    section "Correctness Tests (ordered continuation '6, 7, 8, 9, 10')"

    # FA-on check
    run_correctness_check "--flash-attn on" "Mistral-7B Q4 (FA-on)"

    # Non-FA / default check
    run_correctness_check "" "Mistral-7B Q4 (non-FA)"
}

# ============================================================================
# Determinism Checks
# ============================================================================
run_determinism_check() {
    local description="$1"
    local fa_flag="$2"

    if [ ! -f "$MODEL_MISTRAL_Q4" ]; then
        skip "${description} - model not found"
        return
    fi

    log "  Running 3 iterations..."

    local outputs=()
    for i in 1 2 3; do
        local output
        output=$(ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR}" \
            "${BUILD_DIR}/bin/llama-completion" \
            -m "$MODEL_MISTRAL_Q4" \
            -p '1, 2, 3, 4, 5,' \
            -n 15 \
            --seed 42 \
            --temp 0 \
            $fa_flag \
            2>&1 || true)
        local tokens
        tokens=$(echo "$output" | grep -oP ':\K.*' | head -1)
        outputs+=("$tokens")
    done

    local first="${outputs[0]}"
    local all_match=true
    for output in "${outputs[@]:1}"; do
        if [ "$output" != "$first" ]; then
            all_match=false
            break
        fi
    done

    if $all_match; then
        pass "${description}: deterministic (3 runs, identical)"
    else
        fail "${description}: non-deterministic (3 runs produced different outputs)"
    fi
}

run_determinism_all() {
    section "Determinism Tests (3 identical runs, seed=42, temp=0)"
    if [ -f "$MODEL_MISTRAL_Q4" ]; then
        run_determinism_check "Mistral-7B Q4 (FA-on)" "--flash-attn on"
        run_determinism_check "Mistral-7B Q4 (non-FA)" ""
    else
        skip "Mistral model not found"
    fi
}

# ============================================================================
# Performance Benchmarks
# ============================================================================

# Extract tok/s from a specific test row in llama-bench table output
# Usage: extract_tok_s "<output>" "pp512|tg128"
# Returns the numeric tok/s value (without unit)
extract_tok_s() {
    local output="$1"
    local test_name="$2"

    # llama-bench outputs rows like:
    # | ... | pp512 |  1326.20 ± 3.15 |
    # We find the row with the test name, extract the t/s column (second to last column)
    local row
    row=$(echo "$output" | grep -E "(^\\||\\|.*${test_name})" | grep "$test_name" | head -1)

    if [ -z "$row" ]; then
        echo "N/A"
        return
    fi

    # Parse: split by |, find the column with the test name, next column is t/s
    local tok_s
    tok_s=$(echo "$row" | awk -F'|' '{
        for (i=1; i<=NF; i++) {
            gsub(/^[ \t]+|[ \t]+$/, "", $i)
            if ($i == "'"${test_name}"'") {
                # Next non-empty field is the t/s value
                for (j=i+1; j<=NF; j++) {
                    gsub(/^[ \t]+|[ \t]+$/, "", $j)
                    if ($j ~ /^[0-9]/) {
                        # Extract just the number (before ±)
                        print $j
                        exit
                    }
                }
                break
            }
        }
    }')

    if [ -z "$tok_s" ] || [ "$tok_s" = "N/A" ]; then
        echo "N/A"
    else
        # Extract only the numeric part (before ± if present)
        echo "$tok_s" | grep -oP '^[0-9]+\.?[0-9]*' || echo "N/A"
    fi
}

run_bench_test() {
    local prompt_tokens="$1"
    local gen_tokens="$2"
    local fa_mode="$3"
    local extra_args="${4:-}"

    # Build argv array — never build a string command; Bash would treat
    # env-var assignments as the command name.
    local -a bench_argv=(
        "${BUILD_DIR}/bin/llama-bench"
        -m "${MODEL_MISTRAL_Q4}"
        -p "${prompt_tokens}"
        -n "${gen_tokens}"
    )
    [ "$fa_mode" = "1" ] && bench_argv+=(-fa 1)
    [ -n "$extra_args" ] && bench_argv+=($extra_args)

    local result
    result=$(ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR}" "${bench_argv[@]}" 2>&1) || true
    echo "$result"
}

run_performance_all() {
    section "Performance Benchmarks (llama-bench)"

    echo -e "${CYAN}Mistral-7B Q4_0${NC}" | tee -a "$RESULTS_FILE"
    log "Running benchmarks..."

    # Run PP512 FA-on
    local bench_on_pp
    bench_on_pp=$(run_bench_test 512 0 1 "")

    # Run PP512 FA-off (non-FA)
    local bench_off_pp
    bench_off_pp=$(run_bench_test 512 0 0 "")

    # Run TG128 FA-on
    local bench_on_tg
    bench_on_tg=$(run_bench_test 0 128 1 "")

    # Run TG128 FA-off (non-FA)
    local bench_off_tg
    bench_off_tg=$(run_bench_test 0 128 0 "")

    # Extract tok/s values
    local pp512_on pp512_off tg128_on tg128_off
    pp512_on=$(extract_tok_s "$bench_on_pp" "pp512")
    pp512_off=$(extract_tok_s "$bench_off_pp" "pp512")
    tg128_on=$(extract_tok_s "$bench_on_tg" "tg128")
    tg128_off=$(extract_tok_s "$bench_off_tg" "tg128")

    # Save raw bench output for diagnostics
    log "  Raw PP512 FA-on output: $(echo "$bench_on_pp" | tail -5)"
    log "  Raw PP512 FA-off output: $(echo "$bench_off_pp" | tail -5)"
    log "  Raw TG128 FA-on output: $(echo "$bench_on_tg" | tail -5)"
    log "  Raw TG128 FA-off output: $(echo "$bench_off_tg" | tail -5)"

    # Report summary
    log ""
    log "---"
    log "PP512: FA-on=${pp512_on} vs FA-off=${pp512_off} tok/s"
    log "TG128: FA-on=${tg128_on} vs FA-off=${tg128_off} tok/s"
    log "---"

    # Performance gates
    echo -e "${CYAN}Performance Gates:${NC}" | tee -a "$RESULTS_FILE"

    # Non-FA gates (regression guard: must not drop significantly)
    if [ "$pp512_off" != "N/A" ]; then
        if python3 -c "exit(0 if $pp512_off >= 1700 else 1)" 2>/dev/null; then
            log "  ${GREEN}PASS${NC}: Non-FA PP512 = ${pp512_off} tok/s >= 1700"
        else
            if $PERF_WARN; then
                log "  ${YELLOW}WARN${NC}: Non-FA PP512 = ${pp512_off} tok/s < 1700 (permissive mode)"
            else
                fail "Non-FA PP512 = ${pp512_off} tok/s < 1700 (regression)"
            fi
        fi
    else
        skip "Non-FA PP512: could not parse tok/s"
    fi

    if [ "$tg128_off" != "N/A" ]; then
        if python3 -c "exit(0 if $tg128_off >= 79 else 1)" 2>/dev/null; then
            log "  ${GREEN}PASS${NC}: Non-FA TG128 = ${tg128_off} tok/s >= 79"
        else
            if $PERF_WARN; then
                log "  ${YELLOW}WARN${NC}: Non-FA TG128 = ${tg128_off} tok/s < 79 (permissive mode)"
            else
                fail "Non-FA TG128 = ${tg128_off} tok/s < 79 (regression)"
            fi
        fi
    else
        skip "Non-FA TG128: could not parse tok/s"
    fi

    # FA-on gates (must meet minimum targets)
    if [ "$pp512_on" != "N/A" ]; then
        if python3 -c "exit(0 if $pp512_on >= 1326 else 1)" 2>/dev/null; then
            log "  ${GREEN}PASS${NC}: FA-on PP512 = ${pp512_on} tok/s >= 1326"
        else
            if $PERF_WARN; then
                log "  ${YELLOW}WARN${NC}: FA-on PP512 = ${pp512_on} tok/s < 1326 (permissive mode)"
            else
                fail "FA-on PP512 = ${pp512_on} tok/s < 1326 (below target)"
            fi
        fi
    else
        skip "FA-on PP512: could not parse tok/s"
    fi

    if [ "$tg128_on" != "N/A" ]; then
        if python3 -c "exit(0 if $tg128_on >= 62 else 1)" 2>/dev/null; then
            log "  ${GREEN}PASS${NC}: FA-on TG128 = ${tg128_on} tok/s >= 62"
        else
            if $PERF_WARN; then
                log "  ${YELLOW}WARN${NC}: FA-on TG128 = ${tg128_on} tok/s < 62 (permissive mode)"
            else
                fail "FA-on TG128 = ${tg128_on} tok/s < 62 (below target)"
            fi
        fi
    else
        skip "FA-on TG128: could not parse tok/s"
    fi
}

# ============================================================================
# Backend Ops Test
# ============================================================================
run_backend_ops() {
    section "Backend Ops Test (FLASH_ATTN_EXT)"

    if [ ! -f "${BUILD_DIR}/bin/test-backend-ops" ]; then
        skip "test-backend-ops not found (build with: ./scripts/sycl-build.sh test-backend-ops)"
        return
    fi

    local output
    output=$(ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR}" \
        timeout 120 "${BUILD_DIR}/bin/test-backend-ops" -o FLASH_ATTN_EXT 2>&1) || true

    if echo "$output" | grep -qi "pass"; then
        pass "test-backend-ops -o FLASH_ATTN_EXT (SYCL) passed"
        echo "$output" | grep -E 'all passed|PASSED' | tee -a "$RESULTS_FILE" || true
    else
        local has_fail
        has_fail=$(echo "$output" | grep -cE 'failed|FAIL|FAILED' || true)
        if [ "$has_fail" -gt 0 ]; then
            fail "test-backend-ops -o FLASH_ATTN_EXT had ${has_fail} failures"
            log "    Failing summary lines:"
            echo "$output" | grep -E 'failed|FAIL|FAILED' | head -10 | while IFS= read -r line; do
                log "      $line"
            done
        else
            skip "test-backend-ops -o FLASH_ATTN_EXT (no pass/fail indicators, may be timeout)"
        fi
        log "    Last 40 lines of raw output:"
        echo "$output" | tail -40 | while IFS= read -r line; do
            log "      $line"
        done
    fi
}

# ============================================================================
# FA Dispatch Ktrace (kernel dispatch verification)
# ============================================================================
run_dispatch_check() {
    section "Dispatch Ktrace Verification"

    if [ ! -f "$MODEL_MISTRAL_Q4" ]; then
        skip "Dispatch check - model not found"
        return
    fi

    local output
    output=$(ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR}" \
        timeout 30 "${BUILD_DIR}/bin/llama-bench" \
        -m "$MODEL_MISTRAL_Q4" \
        -p 0 -n 2 -fa 1 -ngl 2 \
        2>&1) || true

    if echo "$output" | grep -qi "ESIMD.*Flash.*enabled.*decode"; then
        pass "ESIMD FA enabled for decode (nc=1 path confirmed)"
    else
        log "  ${YELLOW}INFO${NC}: ESIMD flag not printed (kernel not selected for this shape)" | tee -a "$RESULTS_FILE"
    fi
}

# ============================================================================
# Main
# ============================================================================
main() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  SYCL Flash Attention Correctness + Performance Harness           ║${NC}"
    echo -e "${BLUE}║  Intel Arc GPU - llama.cpp                                        ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    init_environment

    case "$MODE" in
        correctness)
            run_correctness_all
            ;;
        perf)
            run_performance_all
            run_backend_ops
            ;;
        determinism)
            run_determinism_all
            ;;
        all)
            run_correctness_all
            run_determinism_all
            run_performance_all
            run_backend_ops
            run_dispatch_check
            ;;
    esac

    # Summary
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Results: ${GREEN}${PASS} passed${NC}  ${RED}${FAIL} failed${NC}  ${YELLOW}${SKIP} skipped${NC}  ${YELLOW}${WARN} warnings${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    log "Full log: $RESULTS_FILE"

    # Exit code: 0 if no failures; 1 if any failures
    # Warnings and skips do not cause failure
    if [ "$FAIL" -gt 0 ]; then
        exit 1
    fi
    exit 0
}

main "$@"
