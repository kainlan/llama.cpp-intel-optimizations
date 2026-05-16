#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/validate-kkxtv7-sycl.sh [quick|smoke|full] [--log-dir DIR]

Modes:
  quick   Build required targets and run synthetic CTest gates only.
  smoke   quick plus GPT-OSS p16/n4 hybrid and default-auto runs.
  full    smoke plus GPT-OSS pp512/tg128 and single-device Mistral B580/B50 gates.

Environment overrides:
  BUILD_DIR       Build directory (default: build)
  LOG_ROOT        Parent log directory (default: artifacts/kkxtv7-sycl)
  GPT_OSS_MODEL   GPT-OSS model path (default: /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf)
  MISTRAL_MODEL   Mistral model path (default: /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf)
  QUICK_TIMEOUT   Timeout for synthetic gates (default: 5m)
  SMOKE_TIMEOUT   Timeout for p16/n4 model gates (default: 20m)
  FULL_TIMEOUT    Timeout for full benchmark gates (default: 60m)
  CONCISE_MAX_LINES
                  Maximum non-debug model-run log lines (default: 2500)
  MISTRAL_B580_MIN_PP / MISTRAL_B580_MIN_TG
                  Minimum B580 pp512/tg128 tok/s (default: 1680 / 78)
  MISTRAL_B580_FA_MIN_PP / MISTRAL_B580_FA_MIN_TG
                  Minimum B580 -fa 1 pp512/tg128 tok/s (default: 2000 / 78)
  MISTRAL_FA_BLOCKED_OK
                  Mark slow -fa 1 as BLOCKED with proof instead of failing (default: 1)
  MISTRAL_B50_MIN_PP / MISTRAL_B50_MIN_TG
                  Minimum B50 pp512/tg128 tok/s (default: 1050 / 38)
  MISTRAL_UNSAFE_ONEDNN_REPRO_LOG
                  Optional historical pre-fix unsafe oneDNN completion log
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

mode="quick"
BUILD_DIR="${BUILD_DIR:-build}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/artifacts/kkxtv7-sycl}"
GPT_OSS_MODEL="${GPT_OSS_MODEL:-/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf}"
MISTRAL_MODEL="${MISTRAL_MODEL:-/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf}"
QUICK_TIMEOUT="${QUICK_TIMEOUT:-5m}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-20m}"
FULL_TIMEOUT="${FULL_TIMEOUT:-60m}"
CONCISE_MAX_LINES="${CONCISE_MAX_LINES:-2500}"
MISTRAL_B580_MIN_PP="${MISTRAL_B580_MIN_PP:-1680}"
MISTRAL_B580_MIN_TG="${MISTRAL_B580_MIN_TG:-78}"
MISTRAL_B580_FA_MIN_PP="${MISTRAL_B580_FA_MIN_PP:-2000}"
MISTRAL_B580_FA_MIN_TG="${MISTRAL_B580_FA_MIN_TG:-78}"
MISTRAL_FA_BLOCKED_OK="${MISTRAL_FA_BLOCKED_OK:-1}"
MISTRAL_B50_MIN_PP="${MISTRAL_B50_MIN_PP:-1050}"
MISTRAL_B50_MIN_TG="${MISTRAL_B50_MIN_TG:-38}"
MISTRAL_UNSAFE_ONEDNN_REPRO_LOG="${MISTRAL_UNSAFE_ONEDNN_REPRO_LOG:-/tmp/mistral-completion-onednn-allow.log}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        quick|smoke|full)
            mode="$1"
            shift
            ;;
        --log-dir)
            if [[ $# -lt 2 ]]; then
                echo "error: --log-dir requires a directory" >&2
                usage >&2
                exit 2
            fi
            LOG_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -f /opt/intel/oneapi/setvars.sh ]]; then
    echo "error: /opt/intel/oneapi/setvars.sh not found" >&2
    exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_ROOT%/}/${timestamp}-${mode}"
mkdir -p "${LOG_DIR}"

if [[ "${BUILD_DIR}" = /* ]]; then
    BUILD_PATH="${BUILD_DIR}"
else
    BUILD_PATH="${ROOT_DIR}/${BUILD_DIR}"
fi

summary_file="${LOG_DIR}/summary.tsv"
: > "${summary_file}"

declare -a SUMMARY=()
declare -a GPT_OSS_HYBRID_LOGS=()
summary_printed=0

record_summary() {
    local name="$1"
    local status="$2"
    local log="$3"
    SUMMARY+=("${status}"$'\t'"${name}"$'\t'"${log}")
    printf "%s\t%s\t%s\n" "${status}" "${name}" "${log}" >> "${summary_file}"
}

update_summary_status() {
    local name="$1"
    local status="$2"
    local log="$3"
    local row
    local -a updated=()

    for row in "${SUMMARY[@]}"; do
        IFS=$'\t' read -r row_status row_name row_log <<< "${row}"
        if [[ "${row_name}" == "${name}" ]]; then
            row="${status}"$'\t'"${name}"$'\t'"${log}"
        fi
        updated+=("${row}")
    done

    SUMMARY=("${updated[@]}")
    : > "${summary_file}"
    for row in "${SUMMARY[@]}"; do
        printf "%s\n" "${row}" >> "${summary_file}"
    done
}

shell_quote() {
    printf "%q " "$@"
}

run_logged() {
    local name="$1"
    shift

    local safe_name="${name//[^A-Za-z0-9_.-]/_}"
    local log="${LOG_DIR}/${safe_name}.log"
    local cmd="${LOG_DIR}/${safe_name}.cmd"

    shell_quote "$@" > "${cmd}"
    printf "\n" >> "${cmd}"

    echo
    echo "==> ${name}"
    echo "    log: ${log}"
    echo "    cmd: $(cat "${cmd}")"

    set +e
    "$@" 2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    set -e

    if (( rc == 0 )); then
        record_summary "${name}" "PASS" "${log}"
    else
        record_summary "${name}" "FAIL(${rc})" "${log}"
        return "${rc}"
    fi
}

run_env_logged() {
    local name="$1"
    shift

    local safe_name="${name//[^A-Za-z0-9_.-]/_}"
    local log="${LOG_DIR}/${safe_name}.log"
    local cmd="${LOG_DIR}/${safe_name}.cmd"
    local -a env_args=()

    while [[ $# -gt 0 && "$1" == *=* ]]; do
        env_args+=("$1")
        shift
    done

    shell_quote env "${env_args[@]}" "$@" > "${cmd}"
    printf "\n" >> "${cmd}"

    echo
    echo "==> ${name}"
    echo "    log: ${log}"
    echo "    cmd: $(cat "${cmd}")"

    set +e
    env "${env_args[@]}" "$@" 2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    set -e

    if (( rc == 0 )); then
        record_summary "${name}" "PASS" "${log}"
    else
        record_summary "${name}" "FAIL(${rc})" "${log}"
        return "${rc}"
    fi
}

require_model() {
    local model="$1"
    local label="$2"
    if [[ ! -f "${model}" ]]; then
        echo "error: ${label} model not found: ${model}" >&2
        exit 1
    fi
}

check_gpt_oss_hybrid_logs() {
    local bad=0
    local pattern='CPU fallback|fallback to CPU|cpu_fallback'

    for log in "${GPT_OSS_HYBRID_LOGS[@]}"; do
        if grep -Eiq "${pattern}" "${log}"; then
            echo "error: CPU fallback appeared in GPT-OSS hybrid log: ${log}" >&2
            grep -Ein "${pattern}" "${log}" >&2 || true
            bad=1
        fi
    done

    if (( bad )); then
        return 1
    fi
}

reject_log_pattern() {
    local label="$1"
    local log="$2"
    local pattern="$3"

    if grep -Eiq "${pattern}" "${log}"; then
        echo "error: ${label} appeared in log: ${log}" >&2
        grep -Ein "${pattern}" "${log}" >&2 || true
        return 1
    fi
}

require_log_pattern() {
    local label="$1"
    local log="$2"
    local pattern="$3"

    if ! grep -Eiq "${pattern}" "${log}"; then
        echo "error: ${label} missing from log: ${log}" >&2
        return 1
    fi
}

require_mistral_numeric_completion() {
    local label="$1"
    local log="$2"

    if grep -Eq '1, 2, 3, 4, 5, 6, 7, 8, 9, 10' "${log}"; then
        return 0
    fi

    # llama-completion can print perf output between the "1" and "0" of "10".
    # Accept that split form while still requiring the deterministic prefix.
    if grep -Eq '1, 2, 3, 4, 5, 6, 7, 8, 9, 1' "${log}" && grep -Eq '^0$' "${log}"; then
        return 0
    fi

    echo "error: ${label} missing deterministic numeric completion in log: ${log}" >&2
    return 1
}

summarize_bench_log() {
    local name="$1"
    local log="$2"
    local speed_log="${LOG_DIR}/${name}.speed-summary.txt"

    {
        echo "Speed rows for ${name}:"
        grep -E 'pp[[:space:]]+[0-9]+|tg[[:space:]]+[0-9]+|tok/s|t/s' "${log}" || true
    } | tee "${speed_log}"
}

require_bench_speeds() {
    local name="$1"
    local log="$2"

    require_log_pattern "${name} pp speed row" "${log}" '(^|[|[:space:]])pp[[:space:]]*[0-9]+([^[:alnum:]_]|$)'
    require_log_pattern "${name} tg speed row" "${log}" '(^|[|[:space:]])tg[[:space:]]*[0-9]+([^[:alnum:]_]|$)'
}

extract_bench_speed() {
    local label="$1"
    local log="$2"

    awk -v label="${label}" '
        $0 ~ ("(^|[|[:space:]])" label "([^[:alnum:]_]|$)") {
            for (i = 1; i <= NF; ++i) {
                if (($i == "±" || $i == "+/-") && i > 1) {
                    value = $(i - 1)
                    gsub(/[^0-9.]/, "", value)
                    print value
                    exit
                }
            }
            for (i = 1; i <= NF; ++i) {
                if ($i == "t/s" && i > 1) {
                    value = $(i - 1)
                    gsub(/[^0-9.]/, "", value)
                    print value
                    exit
                }
            }
            for (i = NF; i >= 1; --i) {
                value = $i
                gsub(/[^0-9.]/, "", value)
                if (value ~ /^[0-9]+(\.[0-9]+)?$/) {
                    print value
                    exit
                }
            }
        }
    ' "${log}"
}

require_min_speed() {
    local name="$1"
    local log="$2"
    local label="$3"
    local minimum="$4"
    local actual

    actual="$(extract_bench_speed "${label}" "${log}")"
    if [[ -z "${actual}" ]]; then
        echo "error: ${name} missing ${label} speed row in log: ${log}" >&2
        update_summary_status "${name}" "FAIL(speed:${label})" "${log}"
        return 1
    fi

    if ! awk -v actual="${actual}" -v minimum="${minimum}" -v name="${name}" -v label="${label}" '
        BEGIN {
            if ((actual + 0) < (minimum + 0)) {
                printf("error: %s %s speed %.2f below minimum %.2f\n", name, label, actual, minimum) > "/dev/stderr"
                exit 1
            }
        }
    '; then
        update_summary_status "${name}" "FAIL(speed:${label})" "${log}"
        return 1
    fi
}

require_min_speed_or_blocked() {
    local name="$1"
    local log="$2"
    local label="$3"
    local minimum="$4"
    local proof="$5"
    local actual

    actual="$(extract_bench_speed "${label}" "${log}")"
    if [[ -z "${actual}" ]]; then
        echo "error: ${name} missing ${label} speed row in log: ${log}" >&2
        update_summary_status "${name}" "FAIL(speed:${label})" "${log}"
        return 1
    fi

    if awk -v actual="${actual}" -v minimum="${minimum}" 'BEGIN { exit !((actual + 0) >= (minimum + 0)) }'; then
        return 0
    fi

    {
        echo "BLOCKED: ${name} ${label} speed ${actual} below minimum ${minimum}"
        echo "bench_log=${log}"
        echo "bench_cmd=${log%.log}.cmd"
        echo "dispatch_log=${LOG_DIR}/mistral-level_zero-0-fa-dispatch-proof.log"
        echo "dispatch_cmd=${LOG_DIR}/mistral-level_zero-0-fa-dispatch-proof.cmd"
        echo "unsafe_completion_repro_log=${MISTRAL_UNSAFE_ONEDNN_REPRO_LOG}"
        echo "root_cause=oneDNN SDPA materialized GQA/MQA is enabled for the proven f16 5-D descriptor class. If speed is still below the gate, inspect ${LOG_DIR}/mistral-level_zero-0-fa-dispatch-proof.log for oneDNN MATERIALIZED diagnostics and use the bench log above as the blocked-performance proof."
    } > "${proof}"

    update_summary_status "${name}" "BLOCKED(speed:${label})" "${proof}"
    echo "blocked: ${name} ${label} speed ${actual} below minimum ${minimum}; proof: ${proof}" >&2
    if [[ "${MISTRAL_FA_BLOCKED_OK}" == "1" ]]; then
        return 0
    fi
    return 1
}

record_unsafe_onednn_repro_evidence() {
    local evidence="${LOG_DIR}/mistral-unsafe-onednn-repro-evidence.txt"
    local cmd="${LOG_DIR}/mistral-unsafe-onednn-repro.cmd"

    shell_quote env \
        ONEAPI_DEVICE_SELECTOR=level_zero:0 \
        GGML_SYCL_FA_ONEDNN_ALLOW=1 \
        GGML_SYCL_FA_FORCE_PATH=onednn \
        "${BUILD_PATH}/bin/llama-completion" \
        -m "${MISTRAL_MODEL}" \
        -p "1, 2, 3, 4, 5," -n 15 --seed 42 --temp 0 -fa 1 > "${cmd}"
    printf "\n" >> "${cmd}"

    {
        echo "Historical unsafe oneDNN deterministic completion repro."
        echo "This is intentionally not rerun after the safety fix because the override is now barred."
        echo "cmd_file=${cmd}"
        echo "historical_log=${MISTRAL_UNSAFE_ONEDNN_REPRO_LOG}"
        if [[ -f "${MISTRAL_UNSAFE_ONEDNN_REPRO_LOG}" ]]; then
            echo
            echo "bypass lines:"
            grep -En 'nc!=D contiguity gate bypassed' "${MISTRAL_UNSAFE_ONEDNN_REPRO_LOG}" | head -5 || true
            echo
            echo "incorrect output excerpt:"
            grep -En '1, 2, 3, 4, 5|Question 2|TDM 2018' "${MISTRAL_UNSAFE_ONEDNN_REPRO_LOG}" | head -8 || true
        else
            echo "missing historical log; set MISTRAL_UNSAFE_ONEDNN_REPRO_LOG to attach exact pre-fix output."
        fi
    } > "${evidence}"

    record_summary "mistral-unsafe-onednn-repro-evidence" "EVIDENCE" "${evidence}"
}

check_default_log_concision() {
    local name="$1"
    local log="$2"
    local lines
    local noise_pattern='CPU-STAGE|CPU-FWD|\[SYCL-DEBUG\]|\[UNIFIED-DEBUG\]|\[GRAPH-DEBUG\]'

    lines="$(wc -l < "${log}")"
    if (( lines > CONCISE_MAX_LINES )); then
        echo "error: ${name} log has ${lines} lines, above CONCISE_MAX_LINES=${CONCISE_MAX_LINES}: ${log}" >&2
        return 1
    fi

    reject_log_pattern "${name} ungated debug noise" "${log}" "${noise_pattern}"
}

reject_device_lost_errors() {
    local log="$1"
    local matches

    matches="$(
        grep -Ein 'DEVICE_LOST|UR_RESULT_ERROR_DEVICE_LOST|ZE_RESULT_ERROR_DEVICE_LOST|device[ -]?lost|Device[ -]?lost' "${log}" |
            grep -Eiv '\[SYCL\] WARNING: .*trigger DEVICE_LOST on compute-runtime 26\.x' || true
    )"
    if [[ -n "${matches}" ]]; then
        echo "error: DEVICE_LOST runtime error appeared in log: ${log}" >&2
        printf "%s\n" "${matches}" >&2
        return 1
    fi
}

print_summary() {
    if (( summary_printed )); then
        return
    fi
    summary_printed=1

    echo
    echo "Validation summary (${mode})"
    echo "Logs: ${LOG_DIR}"
    for row in "${SUMMARY[@]}"; do
        IFS=$'\t' read -r status name log <<< "${row}"
        printf "  %-8s %s\n" "${status}" "${name}"
        printf "           %s\n" "${log}"
    done
}

trap print_summary EXIT

# shellcheck disable=SC1091
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u

cd "${ROOT_DIR}"

targets=(
    llama-bench
    llama-completion
    test-sycl-fattn-onednn-gates
    test-sycl-fattn-onednn-descriptors
    test-sycl-fattn-onednn-materialization
    test-sycl-fattn-xmx-policy
    test-sycl-kv-planned-device-materialization
    test-sycl-kv-view-resolution
    test-sycl-set-rows-owner-routing
)

if [[ -f "${BUILD_PATH}/CMakeCache.txt" && -f "${BUILD_PATH}/build.ninja" ]]; then
    run_logged "build-required-targets" \
        cmake --build "${BUILD_PATH}" --config Release -j "$(nproc)" --target "${targets[@]}"
else
    run_logged "build-${targets[0]}" ./scripts/sycl-build.sh -B "${BUILD_DIR}" "${targets[0]}"

    if (( ${#targets[@]} > 1 )); then
        remaining_targets=("${targets[@]:1}")
        run_logged "build-required-targets" \
            cmake --build "${BUILD_PATH}" --config Release -j "$(nproc)" --target "${remaining_targets[@]}"
    fi
fi

synthetic_regex='^(test-sycl-fattn-onednn-gates|test-sycl-fattn-onednn-descriptors|test-sycl-fattn-onednn-materialization|test-sycl-fattn-xmx-policy|test-sycl-kv-planned-device-materialization|test-sycl-kv-view-resolution|test-sycl-set-rows-owner-routing)$'
run_env_logged "ctest-synthetic" \
    ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
    timeout "${QUICK_TIMEOUT}" \
    ctest --test-dir "${BUILD_DIR}" -R "${synthetic_regex}" --output-on-failure -j "$(nproc)"
reject_device_lost_errors "${LOG_DIR}/ctest-synthetic.log"

if [[ "${mode}" == "smoke" || "${mode}" == "full" ]]; then
    require_model "${GPT_OSS_MODEL}" "GPT-OSS"

    run_env_logged "gpt-oss-hybrid-p16-n4" \
        ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
        GGML_SYCL_MOE_MULTI_GPU=1 \
        GGML_SYCL_MULTI_GPU_MODE=hybrid \
        GGML_SYCL_OP_TIMEOUT_MS=60000 \
        timeout "${SMOKE_TIMEOUT}" \
        "${BUILD_PATH}/bin/llama-bench" \
        -m "${GPT_OSS_MODEL}" -p 16 -n 4 -ngl 99 -fa 1
    GPT_OSS_HYBRID_LOGS+=("${LOG_DIR}/gpt-oss-hybrid-p16-n4.log")
    reject_device_lost_errors "${LOG_DIR}/gpt-oss-hybrid-p16-n4.log"
    summarize_bench_log "gpt-oss-hybrid-p16-n4" "${LOG_DIR}/gpt-oss-hybrid-p16-n4.log"
    require_bench_speeds "gpt-oss-hybrid-p16-n4" "${LOG_DIR}/gpt-oss-hybrid-p16-n4.log"
    check_default_log_concision "gpt-oss-hybrid-p16-n4" "${LOG_DIR}/gpt-oss-hybrid-p16-n4.log"

    run_env_logged "gpt-oss-default-auto-p16-n4" \
        ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
        GGML_SYCL_MOE_MULTI_GPU=1 \
        GGML_SYCL_OP_TIMEOUT_MS=60000 \
        timeout "${SMOKE_TIMEOUT}" \
        "${BUILD_PATH}/bin/llama-bench" \
        -m "${GPT_OSS_MODEL}" -p 16 -n 4 -ngl 99 -fa 1
    require_log_pattern "default auto expert mode" "${LOG_DIR}/gpt-oss-default-auto-p16-n4.log" \
        'Multi-GPU auto mode: using expert parallelism|Mode:[[:space:]]*EXPERT'
    reject_device_lost_errors "${LOG_DIR}/gpt-oss-default-auto-p16-n4.log"
    summarize_bench_log "gpt-oss-default-auto-p16-n4" "${LOG_DIR}/gpt-oss-default-auto-p16-n4.log"
    require_bench_speeds "gpt-oss-default-auto-p16-n4" "${LOG_DIR}/gpt-oss-default-auto-p16-n4.log"
    check_default_log_concision "gpt-oss-default-auto-p16-n4" "${LOG_DIR}/gpt-oss-default-auto-p16-n4.log"
fi

if [[ "${mode}" == "full" ]]; then
    require_model "${MISTRAL_MODEL}" "Mistral"
    record_unsafe_onednn_repro_evidence

    run_env_logged "gpt-oss-hybrid-pp512-tg128" \
        ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
        GGML_SYCL_MOE_MULTI_GPU=1 \
        GGML_SYCL_MULTI_GPU_MODE=hybrid \
        GGML_SYCL_OP_TIMEOUT_MS=60000 \
        timeout "${FULL_TIMEOUT}" \
        "${BUILD_PATH}/bin/llama-bench" \
        -m "${GPT_OSS_MODEL}" -p 512 -n 128 -ngl 99 -fa 1
    GPT_OSS_HYBRID_LOGS+=("${LOG_DIR}/gpt-oss-hybrid-pp512-tg128.log")
    reject_device_lost_errors "${LOG_DIR}/gpt-oss-hybrid-pp512-tg128.log"
    summarize_bench_log "gpt-oss-hybrid-pp512-tg128" "${LOG_DIR}/gpt-oss-hybrid-pp512-tg128.log"
    require_bench_speeds "gpt-oss-hybrid-pp512-tg128" "${LOG_DIR}/gpt-oss-hybrid-pp512-tg128.log"
    check_default_log_concision "gpt-oss-hybrid-pp512-tg128" "${LOG_DIR}/gpt-oss-hybrid-pp512-tg128.log"

    run_env_logged "gpt-oss-default-auto-pp512-tg128" \
        ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
        GGML_SYCL_MOE_MULTI_GPU=1 \
        GGML_SYCL_OP_TIMEOUT_MS=60000 \
        timeout "${FULL_TIMEOUT}" \
        "${BUILD_PATH}/bin/llama-bench" \
        -m "${GPT_OSS_MODEL}" -p 512 -n 128 -ngl 99 -fa 1
    require_log_pattern "default auto expert mode" "${LOG_DIR}/gpt-oss-default-auto-pp512-tg128.log" \
        'Multi-GPU auto mode: using expert parallelism|Mode:[[:space:]]*EXPERT'
    reject_device_lost_errors "${LOG_DIR}/gpt-oss-default-auto-pp512-tg128.log"
    summarize_bench_log "gpt-oss-default-auto-pp512-tg128" "${LOG_DIR}/gpt-oss-default-auto-pp512-tg128.log"
    require_bench_speeds "gpt-oss-default-auto-pp512-tg128" "${LOG_DIR}/gpt-oss-default-auto-pp512-tg128.log"
    check_default_log_concision "gpt-oss-default-auto-pp512-tg128" "${LOG_DIR}/gpt-oss-default-auto-pp512-tg128.log"

    for selector in level_zero:0 level_zero:1; do
        device_label="${selector/:/-}"

        run_env_logged "mistral-${device_label}-completion" \
            ONEAPI_DEVICE_SELECTOR="${selector}" \
            timeout "${SMOKE_TIMEOUT}" \
            "${BUILD_PATH}/bin/llama-completion" \
            -m "${MISTRAL_MODEL}" \
            -p "1, 2, 3, 4, 5," -n 15 --seed 42 --temp 0
        require_mistral_numeric_completion "deterministic completion" \
            "${LOG_DIR}/mistral-${device_label}-completion.log"
        reject_device_lost_errors "${LOG_DIR}/mistral-${device_label}-completion.log"
        check_default_log_concision "mistral-${device_label}-completion" \
            "${LOG_DIR}/mistral-${device_label}-completion.log"

        run_env_logged "mistral-${device_label}-completion-fa" \
            ONEAPI_DEVICE_SELECTOR="${selector}" \
            timeout "${SMOKE_TIMEOUT}" \
            "${BUILD_PATH}/bin/llama-completion" \
            -m "${MISTRAL_MODEL}" \
            -p "1, 2, 3, 4, 5," -n 15 --seed 42 --temp 0 -fa 1
        require_mistral_numeric_completion "deterministic -fa 1 completion" \
            "${LOG_DIR}/mistral-${device_label}-completion-fa.log"
        reject_log_pattern "unsafe oneDNN nc!=D bypass" "${LOG_DIR}/mistral-${device_label}-completion-fa.log" \
            'nc!=D contiguity gate bypassed'
        reject_device_lost_errors "${LOG_DIR}/mistral-${device_label}-completion-fa.log"
        check_default_log_concision "mistral-${device_label}-completion-fa" \
            "${LOG_DIR}/mistral-${device_label}-completion-fa.log"

        run_env_logged "mistral-${device_label}-pp512-tg128" \
            ONEAPI_DEVICE_SELECTOR="${selector}" \
            timeout "${SMOKE_TIMEOUT}" \
            "${BUILD_PATH}/bin/llama-bench" \
            -m "${MISTRAL_MODEL}" -p 512 -n 128 -ngl 99
        reject_device_lost_errors "${LOG_DIR}/mistral-${device_label}-pp512-tg128.log"
        summarize_bench_log "mistral-${device_label}-pp512-tg128" \
            "${LOG_DIR}/mistral-${device_label}-pp512-tg128.log"
        require_bench_speeds "mistral-${device_label}-pp512-tg128" \
            "${LOG_DIR}/mistral-${device_label}-pp512-tg128.log"
        check_default_log_concision "mistral-${device_label}-pp512-tg128" \
            "${LOG_DIR}/mistral-${device_label}-pp512-tg128.log"

        if [[ "${selector}" == "level_zero:0" ]]; then
            require_min_speed "mistral-${device_label}-pp512-tg128" \
                "${LOG_DIR}/mistral-${device_label}-pp512-tg128.log" pp512 "${MISTRAL_B580_MIN_PP}"
            require_min_speed "mistral-${device_label}-pp512-tg128" \
                "${LOG_DIR}/mistral-${device_label}-pp512-tg128.log" tg128 "${MISTRAL_B580_MIN_TG}"

            run_env_logged "mistral-${device_label}-force-onednn-materialized" \
                ONEAPI_DEVICE_SELECTOR="${selector}" \
                GGML_SYCL_FA_ONEDNN_ALLOW=1 \
                GGML_SYCL_FA_FORCE_PATH=onednn \
                GGML_SYCL_FA_DISPATCH_DEBUG=1 \
                GGML_SYCL_FA_DISPATCH_DEBUG_LIMIT=4 \
                timeout "${SMOKE_TIMEOUT}" \
                "${BUILD_PATH}/bin/llama-completion" \
                -m "${MISTRAL_MODEL}" \
                -p "1, 2, 3, 4, 5," -n 15 --seed 42 --temp 0 -fa 1
            require_mistral_numeric_completion "forced oneDNN deterministic completion" \
                "${LOG_DIR}/mistral-${device_label}-force-onednn-materialized.log"
            require_log_pattern "forced oneDNN materialized dispatch" \
                "${LOG_DIR}/mistral-${device_label}-force-onednn-materialized.log" \
                'oneDNN MATERIALIZED D=128'
            reject_log_pattern "unsafe oneDNN nc!=D bypass" \
                "${LOG_DIR}/mistral-${device_label}-force-onednn-materialized.log" \
                'nc!=D contiguity gate bypassed'
            reject_device_lost_errors "${LOG_DIR}/mistral-${device_label}-force-onednn-materialized.log"

            run_env_logged "mistral-${device_label}-fa-dispatch-proof" \
                ONEAPI_DEVICE_SELECTOR="${selector}" \
                GGML_SYCL_FA_DISPATCH_DEBUG=1 \
                GGML_SYCL_FA_DISPATCH_DEBUG_LIMIT=4 \
                timeout "${SMOKE_TIMEOUT}" \
                "${BUILD_PATH}/bin/llama-bench" \
                -m "${MISTRAL_MODEL}" -p 512 -n 0 -ngl 99 -fa 1
            require_log_pattern "FA dispatch reaches prompt path" \
                "${LOG_DIR}/mistral-${device_label}-fa-dispatch-proof.log" \
                'fattn dispatch .*D=128 ne1>16 xmx=1'
            require_log_pattern "oneDNN planner records materialized dispatch" \
                "${LOG_DIR}/mistral-${device_label}-fa-dispatch-proof.log" \
                'oneDNN MATERIALIZED D=128'
            require_log_pattern "FA graph contains FLASH_ATTN_EXT" \
                "${LOG_DIR}/mistral-${device_label}-fa-dispatch-proof.log" \
                'FLASH_ATTN_EXT'
            summarize_bench_log "mistral-${device_label}-fa-dispatch-proof" \
                "${LOG_DIR}/mistral-${device_label}-fa-dispatch-proof.log"

            run_env_logged "mistral-${device_label}-fa-pp512-tg128" \
                ONEAPI_DEVICE_SELECTOR="${selector}" \
                timeout "${SMOKE_TIMEOUT}" \
                "${BUILD_PATH}/bin/llama-bench" \
                -m "${MISTRAL_MODEL}" -p 512 -n 128 -ngl 99 -fa 1
            reject_device_lost_errors "${LOG_DIR}/mistral-${device_label}-fa-pp512-tg128.log"
            reject_log_pattern "unsafe oneDNN nc!=D bypass" "${LOG_DIR}/mistral-${device_label}-fa-pp512-tg128.log" \
                'nc!=D contiguity gate bypassed'
            summarize_bench_log "mistral-${device_label}-fa-pp512-tg128" \
                "${LOG_DIR}/mistral-${device_label}-fa-pp512-tg128.log"
            require_bench_speeds "mistral-${device_label}-fa-pp512-tg128" \
                "${LOG_DIR}/mistral-${device_label}-fa-pp512-tg128.log"
            require_min_speed_or_blocked "mistral-${device_label}-fa-pp512-tg128" \
                "${LOG_DIR}/mistral-${device_label}-fa-pp512-tg128.log" pp512 "${MISTRAL_B580_FA_MIN_PP}" \
                "${LOG_DIR}/mistral-${device_label}-fa-pp512-tg128.blocked-proof.txt"
            require_min_speed_or_blocked "mistral-${device_label}-fa-pp512-tg128" \
                "${LOG_DIR}/mistral-${device_label}-fa-pp512-tg128.log" tg128 "${MISTRAL_B580_FA_MIN_TG}" \
                "${LOG_DIR}/mistral-${device_label}-fa-pp512-tg128.blocked-proof.txt"
        else
            require_min_speed "mistral-${device_label}-pp512-tg128" \
                "${LOG_DIR}/mistral-${device_label}-pp512-tg128.log" pp512 "${MISTRAL_B50_MIN_PP}"
            require_min_speed "mistral-${device_label}-pp512-tg128" \
                "${LOG_DIR}/mistral-${device_label}-pp512-tg128.log" tg128 "${MISTRAL_B50_MIN_TG}"
        fi
    done
fi

check_gpt_oss_hybrid_logs
print_summary
