#!/usr/bin/env bash
set -euo pipefail

B50_GPTOSS_MODEL="/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf"
B580_MISTRAL_MODEL="/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf"
B50_SELECTOR="level_zero:1"
B580_SELECTOR="level_zero:0"
MODEL="$B50_GPTOSS_MODEL"
DEVICE_SELECTOR="$B50_SELECTOR"
MODE="default"
DRY_RUN=0
USE_GROUPED_DECODE_CANDIDATE=0
LOGDIR="/tmp/llama_b50_moe_sequence_graphlet_$(date +%Y%m%d_%H%M%S)"
GROUPED_DECODE_DIAG_TIMEOUT_SECONDS="${GROUPED_DECODE_DIAG_TIMEOUT_SECONDS:-900}"
GROUPED_DECODE_PERF_TIMEOUT_SECONDS="${GROUPED_DECODE_PERF_TIMEOUT_SECONDS:-1800}"
GROUPED_DECODE_DIAG_TG32_MIN_TPS="${GROUPED_DECODE_DIAG_TG32_MIN_TPS:-5}"

usage() {
    cat <<'EOF'
Usage: scripts/sycl-b50-gptoss-moe-gates.sh [OPTIONS]

Lead-only B50 GPT-OSS MoE correctness/performance gates. Workers must use
--dry-run only. The script never probes devices with sycl-ls or DRM/fdinfo.

Options:
  --mode default|safe-optin|sequence-graphlet|profile|b50-profile-matrix|b50-aggressive-tg|aggressive-suite|down-dpas-direct-final|down-dpas-rank-parallel-atomic|down-dpas-scratch-reduce|down-dpas-same-expert-grouped|b50-pp-materialize-tg-safe|b50-default-candidate|b580-default-candidate|promotion-candidate|promotion-suite|all
      default                Run default canonical B50 GPT-OSS count + PP512/TG128 bench
      safe-optin             Run safe phase/down-XMX weighted-reduce B50 count + bench
      sequence-graphlet      Run active sequence graphlets (explicit unsafe replay+record gates) B50 count + bench
      profile                Run short profiled safe opt-in B50 bench
      b50-profile-matrix     Run B50 default/safe/direct/grouped decode perf and short profile matrix
      b50-aggressive-tg      Run B50 aggressive TG count/diag/perf gates with TG128>=45 and PP regression check
      aggressive-suite       Run B50 aggressive TG gates plus B580/Mistral aggressive count and no-regression checks
      down-dpas-direct-final Run B50 diagnostic down DPAS direct-final short profile and parser evidence check
      down-dpas-rank-parallel-atomic
                              Run B50 diagnostic down DPAS rank-parallel atomic short profile and parser evidence check
      down-dpas-scratch-reduce
                              Run B50 diagnostic down DPAS scratch+reduce short profile and parser evidence check
      down-dpas-same-expert-grouped
                              Run B50 diagnostic down DPAS same-expert grouped short profile and parser evidence check
      b50-pp-materialize-tg-safe
                              Run B50 PP selected-materialization with TG direct-final forbidden
      b50-default-candidate  Run B50 GPT-OSS default fast-path candidate count/diag + no-diag perf bench
      b580-default-candidate Run B580 Mistral default fast-path candidate count/diag + no-diag perf bench
      promotion-candidate    Run B50 and B580 default-candidate gates serially; grouped decode only with --grouped-decode-candidate
      promotion-suite        Run B50 and B580 default-candidate gates serially
      all                    Run legacy B50 gates, excluding unsafe sequence-graphlet mode
  --grouped-decode-candidate
                        Add GGML_SYCL_MOE_GROUPED_DECODE=1 to default-candidate envs
  --case CASE           Alias for --mode CASE, for diagnostic plan dry-runs
  --dry-run             Print and log commands without executing them
  --logdir DIR          Write logs to DIR
  -h, --help            Show this help
EOF
}

quote_cmd() {
    printf '%q ' "$@"
    printf '\n'
}

run_cmd() {
    local name="$1"
    shift
    mkdir -p "$LOGDIR"
    {
        printf '[%s] ' "$name"
        quote_cmd "$@"
    } | tee "$LOGDIR/${name}.cmd"
    if [[ "$DRY_RUN" == 1 ]]; then
        return 0
    fi
    "$@" >"$LOGDIR/${name}.stdout" 2>"$LOGDIR/${name}.stderr"
}

run_cmd_with_timeout() {
    local name="$1"
    local timeout_seconds="$2"
    shift 2
    mkdir -p "$LOGDIR"
    {
        printf '[%s] ' "$name"
        quote_cmd timeout --kill-after=30s "${timeout_seconds}s" "$@"
    } | tee "$LOGDIR/${name}.cmd"
    if [[ "$DRY_RUN" == 1 ]]; then
        return 0
    fi
    local old_opts="$-"
    set +e
    timeout --kill-after=30s "${timeout_seconds}s" "$@" >"$LOGDIR/${name}.stdout" 2>"$LOGDIR/${name}.stderr"
    local rc=$?
    case "$old_opts" in
        *e*) set -e ;;
        *) set +e ;;
    esac
    if [[ "$rc" == 124 || "$rc" == 137 ]]; then
        printf '[HARNESS-TIMEOUT] name=%s seconds=%s rc=%s\n' "$name" "$timeout_seconds" "$rc" >>"$LOGDIR/${name}.stderr"
    fi
    return "$rc"
}

capture_kernel_runtime_metadata() {
    local name="kernel_runtime_metadata"
    local out="$LOGDIR/${name}.kernel-runtime.log"
    mkdir -p "$LOGDIR"
    {
        printf '[%s] ' "$name"
        quote_cmd capture_kernel_runtime_metadata "$out"
    } | tee "$LOGDIR/${name}.cmd"
    if [[ "$DRY_RUN" == 1 ]]; then
        return 0
    fi

    {
        echo "[SYCL-KERNEL-RUNTIME] uname=$(uname -a)"
        if [[ -r /proc/cmdline ]]; then
            echo "[SYCL-KERNEL-RUNTIME] cmdline=$(tr '\0' ' ' < /proc/cmdline)"
        else
            echo "[SYCL-KERNEL-RUNTIME] cmdline=<unreadable>"
        fi

        if command -v modinfo >/dev/null 2>&1; then
            modinfo xe 2>/dev/null | sed 's/^/[SYCL-KERNEL-RUNTIME] modinfo.xe /' || true
        fi

        local cfg="/boot/config-$(uname -r)"
        local key
        if [[ -r "$cfg" ]]; then
            for key in CONFIG_DRM_XE CONFIG_DRM_XE_GPUSVM CONFIG_DRM_XE_PAGEMAP CONFIG_HMM_MIRROR CONFIG_DEVICE_PRIVATE; do
                local match
                match="$(grep -E "^${key}=" "$cfg" 2>/dev/null || true)"
                if [[ -n "$match" ]]; then
                    echo "[SYCL-KERNEL-RUNTIME] config.${match}"
                else
                    echo "[SYCL-KERNEL-RUNTIME] config.${key}=missing"
                fi
            done
        else
            echo "[SYCL-KERNEL-RUNTIME] config.path=$cfg readable=0"
        fi
    } >"$out"
}

check_idle() {
    if [[ "$DRY_RUN" == 1 ]]; then
        return 0
    fi
    local tmp="/tmp/llama-sycl-active.$$"
    if pgrep -af 'llama-(bench|cli|completion)|vtune|sycl-ls' >"$tmp" 2>/dev/null; then
        echo "Refusing to run while possible B50/SYCL jobs are active:" >&2
        cat "$tmp" >&2
        rm -f "$tmp"
        exit 3
    fi
    rm -f "$tmp"
}

source_oneapi_if_running() {
    if [[ "$DRY_RUN" == 1 ]]; then
        return 0
    fi
    # oneAPI setvars references optional environment variables before they are
    # defined on some installations, so do not source it under `set -u`.
    set +u
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null
    set -u
}

mode_selected() {
    local wanted="$1"
    [[ "$MODE" == "$wanted" || ( "$MODE" == "all" && "$wanted" != "sequence-graphlet" ) ]]
}

common_count_args() {
    common_b50_count_args
}

common_b50_count_args() {
    printf '%s\0' \
        ./build/bin/llama-cli \
        -m "$B50_GPTOSS_MODEL" -ngl 99 \
        -cnv -st --simple-io --no-display-prompt \
        --chat-template-kwargs '{"reasoning_effort":"medium"}' \
        --reasoning-format none --reasoning-budget 0 \
        -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' \
        -n 48 --seed 42 --temp 0
}

common_mistral_count_args() {
    printf '%s\0' \
        ./build/bin/llama-completion \
        -m "$B580_MISTRAL_MODEL" \
        -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
}

run_count_gate() {
    local name="$1"
    shift
    local -a env_args=("$@")
    local -a cmd
    mapfile -d '' -t cmd < <(common_count_args)
    run_cmd "$name" env "${env_args[@]}" ONEAPI_DEVICE_SELECTOR="$DEVICE_SELECTOR" "${cmd[@]}"
}

run_b50_count_gate() {
    local name="$1"
    shift
    local -a env_args=("$@")
    local -a cmd
    mapfile -d '' -t cmd < <(common_b50_count_args)
    run_cmd "$name" env "${env_args[@]}" ONEAPI_DEVICE_SELECTOR="$B50_SELECTOR" "${cmd[@]}"
}

run_b580_mistral_count_gate() {
    local name="$1"
    shift
    local -a env_args=("$@")
    local -a cmd
    mapfile -d '' -t cmd < <(common_mistral_count_args)
    run_cmd "$name" env "${env_args[@]}" ONEAPI_DEVICE_SELECTOR="$B580_SELECTOR" "${cmd[@]}"
}

run_bench_gate() {
    local name="$1"
    shift
    local -a env_args=("$@")
    run_cmd "$name" env "${env_args[@]}" ONEAPI_DEVICE_SELECTOR="$DEVICE_SELECTOR" \
        ./build/bin/llama-bench \
        -m "$MODEL" -ngl 99 -p 512 -n 128 -fa 1
}

run_b50_gptoss_bench() {
    local name="$1"
    shift
    local -a env_args=("$@")
    run_cmd "$name" env "${env_args[@]}" ONEAPI_DEVICE_SELECTOR="$B50_SELECTOR" \
        ./build/bin/llama-bench \
        -m "$B50_GPTOSS_MODEL" -ngl 99 -p 512 -n 128 -fa 1
}

run_b50_gptoss_grouped_perf_bench() {
    local name="$1"
    shift
    local -a env_args=("$@")
    run_cmd_with_timeout "$name" "$GROUPED_DECODE_PERF_TIMEOUT_SECONDS" \
        env "${env_args[@]}" ONEAPI_DEVICE_SELECTOR="$B50_SELECTOR" \
        ./build/bin/llama-bench \
        -m "$B50_GPTOSS_MODEL" -ngl 99 -p 512 -n 128 -fa 1
}

run_b50_gptoss_aggressive_perf_bench() {
    local name="$1"
    shift
    local -a env_args=("$@")
    run_cmd_with_timeout "$name" "$GROUPED_DECODE_PERF_TIMEOUT_SECONDS" \
        env "${env_args[@]}" ONEAPI_DEVICE_SELECTOR="$B50_SELECTOR" \
        ./build/bin/llama-bench \
        -m "$B50_GPTOSS_MODEL" -ngl 99 -p 512 -n 128 -fa 1
}

run_b580_mistral_bench() {
    local name="$1"
    shift
    local -a env_args=("$@")
    run_cmd "$name" env "${env_args[@]}" ONEAPI_DEVICE_SELECTOR="$B580_SELECTOR" \
        ./build/bin/llama-bench \
        -m "$B580_MISTRAL_MODEL" -p 512 -n 128 -fa 1
}

run_b50_gptoss_diag_bench() {
    local name="$1"
    shift
    local -a env_args=("$@")
    run_cmd "$name" env "${env_args[@]}" ONEAPI_DEVICE_SELECTOR="$B50_SELECTOR" \
        ./build/bin/llama-bench \
        -m "$B50_GPTOSS_MODEL" -ngl 99 -p 64 -n 32 -fa 1
}

run_b50_gptoss_grouped_diag_bench() {
    local name="$1"
    shift
    local -a env_args=("$@")
    run_cmd_with_timeout "$name" "$GROUPED_DECODE_DIAG_TIMEOUT_SECONDS" \
        env "${env_args[@]}" ONEAPI_DEVICE_SELECTOR="$B50_SELECTOR" \
        ./build/bin/llama-bench \
        -m "$B50_GPTOSS_MODEL" -ngl 99 -p 64 -n 32 -fa 1
}

run_b50_gptoss_aggressive_diag_bench() {
    local name="$1"
    shift
    local -a env_args=("$@")
    run_cmd_with_timeout "$name" "$GROUPED_DECODE_DIAG_TIMEOUT_SECONDS" \
        env "${env_args[@]}" ONEAPI_DEVICE_SELECTOR="$B50_SELECTOR" \
        ./build/bin/llama-bench \
        -m "$B50_GPTOSS_MODEL" -ngl 99 -p 64 -n 32 -fa 1
}

run_b580_mistral_diag_bench() {
    local name="$1"
    shift
    local -a env_args=("$@")
    run_cmd "$name" env "${env_args[@]}" ONEAPI_DEVICE_SELECTOR="$B580_SELECTOR" \
        ./build/bin/llama-bench \
        -m "$B580_MISTRAL_MODEL" -p 64 -n 32 -fa 1
}

run_default_candidate_activation_check() {
    local name="$1"
    local log_base="$2"
    run_cmd "$name" python3 scripts/parse-sycl-moe-profile.py --no-lines \
        --require-default-fast-path-optimized --require-no-fatal-markers \
        "$LOGDIR/${log_base}.stderr"
}

run_default_candidate_fatal_check() {
    local name="$1"
    local log_base="$2"
    run_cmd "$name" python3 scripts/parse-sycl-moe-profile.py --no-lines --require-no-fatal-markers \
        "$LOGDIR/${log_base}.stderr"
}

run_b50_count_output_check() {
    local name="$1"
    local log_base="$2"
    run_cmd "$name" python3 scripts/parse-sycl-moe-profile.py --no-lines \
        --require-generated-count-exact --require-no-fatal-markers \
        "$LOGDIR/${log_base}.stdout" "$LOGDIR/${log_base}.stderr"
}

run_b580_mistral_count_output_check() {
    local name="$1"
    local log_base="$2"
    run_cmd "$name" python3 scripts/parse-sycl-moe-profile.py --no-lines \
        --require-mistral-count-prefix --require-no-fatal-markers \
        "$LOGDIR/${log_base}.stdout" "$LOGDIR/${log_base}.stderr"
}

run_grouped_decode_binary_label_check() {
    run_cmd b50_grouped_decode_binary_label_check bash -lc \
        'strings ./build/bin/libggml-sycl.so | grep -q grouped-packed-q8-m2-device'
}

run_grouped_decode_partial_path_check() {
    local name="$1"
    local log_base="$2"
    run_cmd "$name" python3 scripts/parse-sycl-moe-profile.py --no-lines \
        --forbid-diag-path grouped-packed-q8-m2-device --require-no-fatal-markers \
        "$LOGDIR/${log_base}.stderr"
}

run_grouped_decode_diag_tg_floor_check() {
    local name="$1"
    local log_base="$2"
    run_cmd "$name" python3 scripts/parse-sycl-moe-profile.py --no-lines \
        --require-bench-min tg32 "${GROUPED_DECODE_DIAG_TG32_MIN_TPS}" \
        "$LOGDIR/${log_base}.stdout"
}

run_grouped_decode_perf_completion_check() {
    local name="$1"
    local log_base="$2"
    run_cmd "$name" python3 scripts/parse-sycl-moe-profile.py --no-lines \
        --require-bench-test tg128 --require-no-fatal-markers \
        "$LOGDIR/${log_base}.stdout" "$LOGDIR/${log_base}.stderr"
}

run_b50_grouped_decode_timed_diag_and_partial_path_check() {
    local bench_name="$1"
    local check_name="$2"
    shift 2
    set +e
    run_b50_gptoss_grouped_diag_bench "$bench_name" "$@"
    local bench_rc=$?
    set -e
    run_grouped_decode_partial_path_check "$check_name" "$bench_name"
    run_grouped_decode_diag_tg_floor_check "${bench_name}_tg_floor_check" "$bench_name"
    if [[ "$bench_rc" -ne 0 ]]; then
        return "$bench_rc"
    fi
}

run_b50_grouped_decode_timed_perf_and_completion_check() {
    local bench_name="$1"
    local check_name="$2"
    shift 2
    set +e
    run_b50_gptoss_grouped_perf_bench "$bench_name" "$@"
    local bench_rc=$?
    set -e
    run_grouped_decode_perf_completion_check "$check_name" "$bench_name"
    if [[ "$bench_rc" -ne 0 ]]; then
        return "$bench_rc"
    fi
}

run_aggressive_tg_diag_path_check() {
    local name="$1"
    local log_base="$2"
    run_cmd "$name" python3 scripts/parse-sycl-moe-profile.py --no-lines \
        --require-any-diag-path aggressive-partial,aggressive-partial-packed-q8-m4-artifact,aggressive-partial-fused-tg,aggressive-partial-soa-packed-q8-m4-artifact,direct-xmx \
        --require-aggressive-optimized-substrate \
        --require-xmx-original-clean \
        --forbid-diag-path split-sg16 \
        --forbid-diag-path grouped-packed-q8-m2-device --require-no-fatal-markers \
        "$LOGDIR/${log_base}.stderr"
}

run_aggressive_tg_perf_check() {
    local name="$1"
    local perf_base="$2"
    local safe_base="$3"
    run_cmd "$name" python3 scripts/parse-sycl-moe-profile.py --no-lines \
        --require-bench-test tg128 --require-bench-min tg128 45 \
        --require-bench-within-pct pp512 "$LOGDIR/${perf_base}.stdout" "$LOGDIR/${safe_base}.stdout" 5 \
        --require-no-fatal-markers \
        "$LOGDIR/${perf_base}.stdout" "$LOGDIR/${perf_base}.stderr"
}

run_down_dpas_direct_final_path_check() {
    local name="$1"
    local log_base="$2"
    run_cmd "$name" python3 scripts/parse-sycl-moe-profile.py --no-lines \
        --require-down-dpas-direct-final \
        --forbid-diag-path split-sg16 \
        --forbid-diag-path grouped-packed-q8-m2-device --require-no-fatal-markers \
        "$LOGDIR/${log_base}.stderr"
}

run_b50_pp_materialize_tg_safe_path_check() {
    local name="$1"
    local log_base="$2"
    run_cmd "$name" python3 scripts/parse-sycl-moe-profile.py --no-lines \
        --require-mxfp4-profile-evidence \
        --forbid-down-dpas-direct-final \
        --forbid-diag-path split-sg16 \
        --forbid-diag-path grouped-packed-q8-m2-device \
        --require-no-fatal-markers \
        "$LOGDIR/${log_base}.stderr"
}

run_b50_pp_materialize_tg_safe_perf_check() {
    local name="$1"
    local perf_base="$2"
    run_cmd "$name" python3 scripts/parse-sycl-moe-profile.py --no-lines \
        --require-bench-test pp512 --require-bench-min pp512 1100 \
        --require-bench-test tg128 --require-bench-min tg128 45 \
        --require-no-fatal-markers \
        "$LOGDIR/${perf_base}.stdout" "$LOGDIR/${perf_base}.stderr"
}


run_b580_mistral_perf_no_regression_check() {
    local name="$1"
    local perf_base="$2"
    local baseline_base="$3"
    run_cmd "$name" python3 scripts/parse-sycl-moe-profile.py --no-lines \
        --require-bench-test tg128 \
        --require-bench-within-pct pp512 "$LOGDIR/${perf_base}.stdout" "$LOGDIR/${baseline_base}.stdout" 5 \
        --require-bench-within-pct tg128 "$LOGDIR/${perf_base}.stdout" "$LOGDIR/${baseline_base}.stdout" 5 \
        --require-no-fatal-markers \
        "$LOGDIR/${perf_base}.stdout" "$LOGDIR/${perf_base}.stderr"
}

run_profile_gate() {
    local name="$1"
    shift
    local -a env_args=("$@")
    run_cmd "$name" env "${env_args[@]}" \
        ONEAPI_DEVICE_SELECTOR="$DEVICE_SELECTOR" \
        GGML_SYCL_MOE_PROFILE=1 \
        GGML_SYCL_MXFP4_TG_PROFILE=1 \
        GGML_SYCL_MXFP4_PP_PROFILE=1 \
        GGML_SYCL_MOE_PATH_TRACE=1 \
        GGML_SYCL_GRAPH_DIAG=1 \
        ./build/bin/llama-bench \
        -m "$MODEL" -ngl 99 -p 64 -n 32 -fa 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mode|--case)
                if [[ $# -lt 2 ]]; then
                    echo "$1 requires an argument" >&2
                    exit 2
                fi
                MODE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=1
                shift
                ;;
            --grouped-decode-candidate)
                USE_GROUPED_DECODE_CANDIDATE=1
                shift
                ;;
            --logdir)
                if [[ $# -lt 2 ]]; then
                    echo "--logdir requires an argument" >&2
                    exit 2
                fi
                LOGDIR="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "unknown arg: $1" >&2
                usage >&2
                exit 2
                ;;
        esac
    done

    case "$MODE" in
        default|safe-optin|sequence-graphlet|profile|b50-profile-matrix|b50-aggressive-tg|aggressive-suite|down-dpas-direct-final|down-dpas-rank-parallel-atomic|down-dpas-scratch-reduce|down-dpas-same-expert-grouped|b50-pp-materialize-tg-safe|b50-default-candidate|b580-default-candidate|promotion-candidate|promotion-suite|all) ;;
        *)
            echo "bad mode: $MODE" >&2
            usage >&2
            exit 2
            ;;
    esac
}

main() {
    parse_args "$@"
    check_idle
    source_oneapi_if_running
    mkdir -p "$LOGDIR"
    capture_kernel_runtime_metadata

    local -a safe_env=(
        GGML_SYCL_MOE_PHASE_MATERIALIZE=1
        GGML_SYCL_MOE_PHASE_BULK_XMX=1
        GGML_SYCL_MOE_DOWN_XMX_TILED=1
        GGML_SYCL_MOE_PHASE_DOWN_XMX=1
        GGML_SYCL_MOE_DOWN_SUM_DIRECT=1
    )
    local -a sequence_env=(
        "${safe_env[@]}"
        GGML_SYCL_MOE_SEQUENCE_GRAPHLETS=1
        GGML_SYCL_MOE_SEQUENCE_GRAPHLETS_UNSAFE_REPLAY=1
        GGML_SYCL_MOE_SEQUENCE_GRAPHLETS_ALLOW_UNSAFE_RECORD=1
        GGML_SYCL_GRAPH_DIAG=1
    )
    local -a default_candidate_base_env=(
        GGML_SYCL_MOE_DEFAULT_FAST_PATH=1
        GGML_SYCL_MOE_DEFAULT_FAST_PATH_PROMOTION_CANDIDATE=1
    )
    if [[ "$USE_GROUPED_DECODE_CANDIDATE" -eq 1 ]]; then
        default_candidate_base_env+=(GGML_SYCL_MOE_GROUPED_DECODE=1)
    fi
    local -a default_candidate_diag_env=(
        "${default_candidate_base_env[@]}"
        GGML_SYCL_GRAPH_DIAG=1
    )
    local -a default_candidate_perf_env=(
        "${default_candidate_base_env[@]}"
    )
    local -a direct_none_env=(
        GGML_SYCL_MOE_DEFAULT_FAST_PATH=1
        GGML_SYCL_MOE_DEFAULT_FAST_PATH_PROMOTION_CANDIDATE=1
        GGML_SYCL_MOE_AGGREGATION_DECISION=none
    )
    local -a grouped_decode_env=(
        GGML_SYCL_MOE_GROUPED_DECODE=1
        GGML_SYCL_MOE_DEFAULT_FAST_PATH=1
        GGML_SYCL_MOE_DEFAULT_FAST_PATH_PROMOTION_CANDIDATE=1
    )
    local -a grouped_decode_evidence_env=(
        GGML_SYCL_MOE_GLU_Q8_PUBLISH_DIAG=1
    )
    local -a aggressive_tg_env=(
        GGML_SYCL_MOE_AGGRESSIVE_TG=1
        GGML_SYCL_MOE_PARTIAL_DEVICE_GROUPING=1
        GGML_SYCL_MOE_AGGRESSIVE_XMX_TILED=1
    )
    local -a aggressive_tg_diag_env=(
        "${aggressive_tg_env[@]}"
        GGML_SYCL_MOE_GLU_Q8_PUBLISH_DIAG=1
        GGML_SYCL_XMX_TILED_VALIDATE_OUTPUT_ORIGINAL=8
        GGML_SYCL_XMX_TILED_VALIDATE_MATERIALIZATION_ORIGINAL=8
        GGML_SYCL_GRAPH_DIAG=1
    )
    local -a down_dpas_direct_final_env=(
        GGML_SYCL_MOE_DOWN_SUM_DIRECT=1
        GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL=1
        GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_I8=1
        GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_DPAS=1
        GGML_SYCL_MOE_DECODE_DOWN_I8_SELECTED=1
        GGML_SYCL_SELECTED_DPAS_MATERIALIZE=1
        GGML_SYCL_MXFP4_TG_PROFILE=1
    )
    local -a down_dpas_rank_parallel_atomic_env=(
        "${down_dpas_direct_final_env[@]}"
        GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_RANK_PARALLEL_ATOMIC=1
    )
    local -a down_dpas_scratch_reduce_env=(
        "${down_dpas_direct_final_env[@]}"
        GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_SCRATCH_REDUCE=1
    )
    local -a down_dpas_same_expert_grouped_env=(
        "${down_dpas_direct_final_env[@]}"
        GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_SAME_EXPERT_GROUPED=1
    )
    local -a kernel_profile_env=(
        GGML_SYCL_MOE_PROFILE=1
        GGML_SYCL_MXFP4_TG_PROFILE=1
        GGML_SYCL_MXFP4_PP_PROFILE=1
    )
    local -a pp_materialize_tg_safe_env=(
        GGML_SYCL_MOE_PHASE_MATERIALIZE=1
        GGML_SYCL_MOE_PHASE_BULK_XMX=1
        GGML_SYCL_MOE_DOWN_SUM_DIRECT=1
        GGML_SYCL_SELECTED_DPAS_MATERIALIZE=1
        GGML_SYCL_MXFP4_TG_PROFILE=1
        GGML_SYCL_MXFP4_PP_PROFILE=1
        GGML_SYCL_MOE_PROFILE=1
    )

    if mode_selected default; then
        run_count_gate default_count
        run_bench_gate default_bench
    fi

    if mode_selected safe-optin; then
        run_count_gate safe_optin_count "${safe_env[@]}"
        run_bench_gate safe_optin_bench "${safe_env[@]}"
    fi

    if mode_selected sequence-graphlet; then
        run_count_gate sequence_graphlet_count "${sequence_env[@]}"
        run_bench_gate sequence_graphlet_bench "${sequence_env[@]}"
    fi

    if mode_selected profile; then
        run_profile_gate safe_optin_profile "${safe_env[@]}"
    fi

    if [[ "$MODE" == "b50-default-candidate" || "$MODE" == "promotion-candidate" || "$MODE" == "promotion-suite" ]]; then
        run_b50_count_gate b50_default_candidate_count_diag "${default_candidate_diag_env[@]}"
        run_b50_gptoss_diag_bench b50_default_candidate_bench_diag "${default_candidate_diag_env[@]}"
        run_default_candidate_activation_check b50_default_candidate_activation_check b50_default_candidate_bench_diag
        run_b50_gptoss_bench b50_default_candidate_bench_perf "${default_candidate_perf_env[@]}"
    fi

    if [[ "$MODE" == "b580-default-candidate" || "$MODE" == "promotion-candidate" || "$MODE" == "promotion-suite" ]]; then
        run_b580_mistral_count_gate b580_default_candidate_count_diag "${default_candidate_diag_env[@]}"
        run_b580_mistral_diag_bench b580_default_candidate_bench_diag "${default_candidate_diag_env[@]}"
        run_default_candidate_fatal_check b580_default_candidate_fatal_check b580_default_candidate_bench_diag
        run_b580_mistral_bench b580_default_candidate_bench_perf "${default_candidate_perf_env[@]}"
    fi

    if [[ "$MODE" == "b50-profile-matrix" ]]; then
        run_grouped_decode_binary_label_check
        run_b50_grouped_decode_timed_diag_and_partial_path_check b50_grouped_decode_diag b50_grouped_decode_diag_partial_path_check \
            "${grouped_decode_env[@]}" "${grouped_decode_evidence_env[@]}" GGML_SYCL_GRAPH_DIAG=1
        run_b50_grouped_decode_timed_perf_and_completion_check b50_grouped_decode_perf b50_grouped_decode_perf_completion_check \
            "${grouped_decode_env[@]}"
        run_b50_gptoss_bench b50_default_perf
        run_b50_gptoss_bench b50_safe_env_perf "${safe_env[@]}"
        run_b50_gptoss_bench b50_direct_none_perf "${direct_none_env[@]}"
        run_b50_gptoss_diag_bench b50_direct_none_diag "${direct_none_env[@]}" GGML_SYCL_GRAPH_DIAG=1
        run_b50_gptoss_diag_bench b50_default_kernel_profile "${kernel_profile_env[@]}"
        run_b50_gptoss_diag_bench b50_safe_env_kernel_profile "${safe_env[@]}" "${kernel_profile_env[@]}"
        run_b50_grouped_decode_timed_diag_and_partial_path_check b50_grouped_decode_kernel_profile b50_grouped_decode_kernel_profile_partial_path_check \
            "${grouped_decode_env[@]}" "${grouped_decode_evidence_env[@]}" "${kernel_profile_env[@]}"
    fi

    if [[ "$MODE" == "b50-pp-materialize-tg-safe" ]]; then
        run_b50_count_gate b50_pp_materialize_tg_safe_count "${pp_materialize_tg_safe_env[@]}"
        run_b50_count_output_check b50_pp_materialize_tg_safe_count_output_check b50_pp_materialize_tg_safe_count
        run_b50_gptoss_diag_bench b50_pp_materialize_tg_safe_diag "${pp_materialize_tg_safe_env[@]}"
        run_b50_pp_materialize_tg_safe_path_check b50_pp_materialize_tg_safe_path_check b50_pp_materialize_tg_safe_diag
        run_b50_gptoss_bench b50_pp_materialize_tg_safe_perf "${pp_materialize_tg_safe_env[@]}"
        run_b50_pp_materialize_tg_safe_perf_check b50_pp_materialize_tg_safe_perf_check b50_pp_materialize_tg_safe_perf
    fi


    if [[ "$MODE" == "b50-aggressive-tg" || "$MODE" == "aggressive-suite" ]]; then
        run_b50_count_gate b50_aggressive_count "${aggressive_tg_env[@]}"
        run_b50_count_output_check b50_aggressive_count_output_check b50_aggressive_count
        run_b50_gptoss_bench b50_aggressive_safe_perf "${safe_env[@]}"
        run_b50_gptoss_aggressive_diag_bench b50_aggressive_diag "${aggressive_tg_diag_env[@]}"
        run_aggressive_tg_diag_path_check b50_aggressive_diag_path_check b50_aggressive_diag
        run_b50_gptoss_aggressive_perf_bench b50_aggressive_perf "${aggressive_tg_env[@]}"
        run_aggressive_tg_perf_check b50_aggressive_perf_check b50_aggressive_perf b50_aggressive_safe_perf
    fi

    if [[ "$MODE" == "down-dpas-direct-final" ]]; then
        run_b50_gptoss_aggressive_diag_bench down_dpas_direct_final_diag "${down_dpas_direct_final_env[@]}"
        run_down_dpas_direct_final_path_check down_dpas_direct_final_path_check down_dpas_direct_final_diag
    fi

    if [[ "$MODE" == "down-dpas-rank-parallel-atomic" ]]; then
        run_b50_gptoss_aggressive_diag_bench down_dpas_rank_parallel_atomic_diag "${down_dpas_rank_parallel_atomic_env[@]}"
        run_down_dpas_direct_final_path_check down_dpas_rank_parallel_atomic_path_check down_dpas_rank_parallel_atomic_diag
    fi

    if [[ "$MODE" == "down-dpas-scratch-reduce" ]]; then
        run_b50_gptoss_aggressive_diag_bench down_dpas_scratch_reduce_diag "${down_dpas_scratch_reduce_env[@]}"
        run_down_dpas_direct_final_path_check down_dpas_scratch_reduce_path_check down_dpas_scratch_reduce_diag
    fi

    if [[ "$MODE" == "down-dpas-same-expert-grouped" ]]; then
        run_b50_gptoss_aggressive_diag_bench down_dpas_same_expert_grouped_diag "${down_dpas_same_expert_grouped_env[@]}"
        run_down_dpas_direct_final_path_check down_dpas_same_expert_grouped_path_check down_dpas_same_expert_grouped_diag
    fi

    if [[ "$MODE" == "aggressive-suite" ]]; then
        run_b580_mistral_count_gate b580_aggressive_mistral_count "${aggressive_tg_env[@]}"
        run_b580_mistral_count_output_check b580_aggressive_mistral_count_output_check b580_aggressive_mistral_count
        run_b580_mistral_bench b580_aggressive_mistral_default_perf
        run_b580_mistral_bench b580_aggressive_mistral_perf "${aggressive_tg_env[@]}"
        run_b580_mistral_perf_no_regression_check b580_aggressive_mistral_perf_check \
            b580_aggressive_mistral_perf b580_aggressive_mistral_default_perf
    fi

    echo "logs: $LOGDIR"
}

main "$@"
