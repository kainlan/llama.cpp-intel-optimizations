#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_BENCH="${LLAMA_BENCH:-$ROOT_DIR/build/bin/llama-bench}"
LLAMA_CLI="${LLAMA_CLI:-$ROOT_DIR/build/bin/llama-cli}"
LLAMA_COMPLETION="${LLAMA_COMPLETION:-$ROOT_DIR/build/bin/llama-completion}"
GPTOSS_MODEL="${GPTOSS_MODEL:-/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf}"
MISTRAL_MODEL="${MISTRAL_MODEL:-/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/benchmark_results/sycl-dual-gpu-gate-$(date +%Y%m%d-%H%M%S)}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-900}"
GPTOSS_B50_MIN_PP512="${GPTOSS_B50_MIN_PP512:-1100}"
GPTOSS_B50_MIN_TG128="${GPTOSS_B50_MIN_TG128:-50}"
MISTRAL_B580_MIN_PP512="${MISTRAL_B580_MIN_PP512:-2000}"
MISTRAL_B580_MIN_TG128="${MISTRAL_B580_MIN_TG128:-85}"
RUN_CORRECTNESS="${RUN_CORRECTNESS:-1}"

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/sycl-gpu-preflight.sh"

set +u
source /opt/intel/oneapi/setvars.sh --force >/tmp/oneapi-setvars.log 2>&1
set -u

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.tsv"
printf "case\tselector\tmodel\tfa\trc\tpp512_tps\ttg128_tps\tmoe_ms\tgpu_expert_ms\tnon_moe_ms\troute_dev0\troute_secondary\troute_host\troute_unavailable\thost_routing\tlog\n" > "$SUMMARY"
CORRECTNESS="$OUT_DIR/correctness.tsv"
printf "case\tselector\trc\tpass\tlog\n" > "$CORRECTNESS"

if [[ "$RUN_CORRECTNESS" == "1" ]]; then
    if [[ ! -x "$LLAMA_CLI" ]]; then
        echo "llama-cli not executable: $LLAMA_CLI" >&2
        exit 2
    fi
    if [[ ! -x "$LLAMA_COMPLETION" ]]; then
        echo "llama-completion not executable: $LLAMA_COMPLETION" >&2
        exit 2
    fi
fi

parse_tps() {
    local log="$1"
    local test="$2"
    awk -F'|' -v test="$test" '
        index($0, test) {
            for (i = 1; i <= NF; ++i) {
                gsub(/^[ \t]+|[ \t]+$/, "", $i)
                if ($i ~ /^[0-9]+([.][0-9]+)?[ \t]*±/) {
                    split($i, a, /[ \t]+/)
                    print a[1]
                    exit
                }
            }
        }
    ' "$log"
}

parse_metric() {
    local log="$1"
    local label="$2"
    awk -F'|' -v label="$label" '
        index($0, label) {
            value = $2
            gsub(/^[ \t]+|[ \t]+$/, "", value)
            split(value, a, /[ \t]+/)
            print a[1]
            exit
        }
    ' "$log"
}

parse_host_routing() {
    local log="$1"
    awk '
        /Host routing:/ {
            print $3
            found = 1
        }
        END {
            if (!found) {
                print "NA"
            }
        }
    ' "$log" | tail -1
}

parse_route_metric() {
    local log="$1"
    local key="$2"
    awk -v key="$key" '
        /Per token entries:/ {
            for (i = 1; i <= NF; ++i) {
                if ($i == key) {
                    value = $(i + 1)
                    gsub(/,/, "", value)
                    print value
                    found = 1
                }
            }
        }
        END {
            if (!found) {
                print "NA"
            }
        }
    ' "$log" | tail -1
}

run_case() {
    local name="$1"
    local selector="$2"
    local model="$3"
    local fa="$4"
    local log="$OUT_DIR/${name}-fa${fa}.log"
    local -a env_args=("ONEAPI_DEVICE_SELECTOR=$selector")

    if [[ "$name" == gptoss-dual* ]]; then
        env_args+=("GGML_SYCL_MOE_PROFILE=1" "GGML_SYCL_MOE_PATH_TRACE=1")
    fi

    printf "[gate] %s selector=%s fa=%s\n" "$name" "$selector" "$fa" | tee "$log"
    sycl_gpu_preflight_check "$selector"
    set +e
    timeout "$BENCH_TIMEOUT" env "${env_args[@]}" "$LLAMA_BENCH" -m "$model" -p 512 -n 128 -fa "$fa" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    set -e

    local pp
    local tg
    local moe_ms
    local gpu_expert_ms
    local non_moe_ms
    local route_dev0
    local route_secondary
    local route_host
    local route_unavailable
    local host_routing
    pp="$(parse_tps "$log" "pp512" || true)"
    tg="$(parse_tps "$log" "tg128" || true)"
    moe_ms="$(parse_metric "$log" "MoE total" || true)"
    gpu_expert_ms="$(parse_metric "$log" "GPU expert compute" || true)"
    non_moe_ms="$(parse_metric "$log" "Non-MoE" || true)"
    route_dev0="$(parse_route_metric "$log" "dev0" || true)"
    route_secondary="$(parse_route_metric "$log" "secondary" || true)"
    route_host="$(parse_route_metric "$log" "host" || true)"
    route_unavailable="$(parse_route_metric "$log" "unavailable" || true)"
    host_routing="$(parse_host_routing "$log" || true)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$selector" "$(basename "$model")" "$fa" \
        "$rc" "${pp:-NA}" "${tg:-NA}" "${moe_ms:-NA}" "${gpu_expert_ms:-NA}" "${non_moe_ms:-NA}" \
        "${route_dev0:-NA}" "${route_secondary:-NA}" "${route_host:-NA}" "${route_unavailable:-NA}" \
        "${host_routing:-NA}" "$log" >> "$SUMMARY"
}

run_correctness_case() {
    local name="$1"
    local selector="$2"
    shift 2
    local log="$OUT_DIR/${name}.log"
    local -a cmd=("$@")

    printf "[correctness] %s selector=%s\n" "$name" "$selector" | tee "$log"
    sycl_gpu_preflight_check "$selector"
    set +e
    timeout "$BENCH_TIMEOUT" env ONEAPI_DEVICE_SELECTOR="$selector" "${cmd[@]}" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    set -e

    local pass="no"
    case "$name" in
        gptoss-b50-count)
            if grep -Eq '1, 2, 3, 4, 5' "$log"; then
                pass="yes"
            fi
            ;;
        mistral-b580-count)
            if grep -Eq '1, 2, 3, 4, 5, 6, 7, 8, 9, 10' "$log"; then
                pass="yes"
            fi
            ;;
    esac

    printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$selector" "$rc" "$pass" "$log" >> "$CORRECTNESS"
    if [[ "$rc" != "0" || "$pass" != "yes" ]]; then
        echo "correctness gate failed: $name" >&2
        return 1
    fi
}

run_correctness_gates() {
    if [[ "$RUN_CORRECTNESS" != "1" ]]; then
        return 0
    fi

    run_correctness_case "mistral-b580-count" "level_zero:0" \
        "$LLAMA_COMPLETION" \
        -m "$MISTRAL_MODEL" \
        -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

    run_correctness_case "gptoss-b50-count" "level_zero:1" \
        "$LLAMA_CLI" \
        -m "$GPTOSS_MODEL" \
        -cnv -st --simple-io --no-display-prompt \
        --reasoning-format none --reasoning-budget 0 \
        -p 'Count from 1 to 5. Answer with only: 1, 2, 3, 4, 5' \
        -n 48 --seed 42 --temp 0
}

write_comparison() {
    local comparison="$OUT_DIR/comparison.tsv"
    awk -F'\t' '
        BEGIN {
            OFS = "\t"
            print "fa", "best_single_case", "best_single_pp512", "dual_pp512", "dual_vs_best_pp512_pct", \
                  "target_110pct_pp512", "beats_best_by_10pct", "best_single_tg128", "dual_tg128", \
                  "dual_vs_best_tg128_pct"
        }
        NR == 1 {
            next
        }
        $3 == "gpt-oss-20b-mxfp4.gguf" {
            fa = $4
            if ($1 == "gptoss-b580" || $1 == "gptoss-b50") {
                if ($6 != "NA" && ($6 + 0) > best_pp[fa]) {
                    best_pp[fa] = $6 + 0
                    best_case[fa] = $1
                }
                if ($7 != "NA" && ($7 + 0) > best_tg[fa]) {
                    best_tg[fa] = $7 + 0
                }
            } else if ($1 == "gptoss-dual-b580-b50") {
                dual_pp[fa] = $6
                dual_tg[fa] = $7
                seen_dual[fa] = 1
            }
        }
        END {
            for (fa in seen_dual) {
                pp = dual_pp[fa]
                tg = dual_tg[fa]
                pp_pct = "NA"
                tg_pct = "NA"
                target = "NA"
                pass = "no"
                if (pp != "NA" && best_pp[fa] > 0) {
                    pp_pct = sprintf("%.1f", 100.0 * (pp + 0) / best_pp[fa])
                    target = sprintf("%.2f", 1.10 * best_pp[fa])
                    pass = ((pp + 0) >= 1.10 * best_pp[fa]) ? "yes" : "no"
                }
                if (tg != "NA" && best_tg[fa] > 0) {
                    tg_pct = sprintf("%.1f", 100.0 * (tg + 0) / best_tg[fa])
                }
                print fa, best_case[fa], best_pp[fa], pp, pp_pct, target, pass, best_tg[fa], tg, tg_pct
            }
        }
    ' "$SUMMARY" > "$comparison"
}

validate_guardrails() {
    local guardrails="$OUT_DIR/guardrails.tsv"
    awk -F'\t' \
        -v gpt_pp="$GPTOSS_B50_MIN_PP512" \
        -v gpt_tg="$GPTOSS_B50_MIN_TG128" \
        -v mistral_pp="$MISTRAL_B580_MIN_PP512" \
        -v mistral_tg="$MISTRAL_B580_MIN_TG128" '
        BEGIN {
            OFS = "\t"
            print "guard", "case", "fa", "rc", "pp512_tps", "min_pp512", "tg128_tps", "min_tg128", "pass"
        }
        NR == 1 {
            next
        }
        function numeric(value) {
            return value != "" && value != "NA" && value ~ /^[0-9]+([.][0-9]+)?$/
        }
        function check_guard(guard, min_pp, min_tg) {
            pass = (($5 + 0) == 0 && numeric($6) && numeric($7) && ($6 + 0) >= min_pp && ($7 + 0) >= min_tg) ? "yes" : "no"
            print guard, $1, $4, $5, $6, min_pp, $7, min_tg, pass
            if (pass != "yes") {
                failures++
            }
        }
        $1 == "gptoss-b50" && $4 == "1" {
            check_guard("gptoss_b50_fa1", gpt_pp, gpt_tg)
            seen_gpt = 1
        }
        $1 == "mistral-b580" && $4 == "1" {
            check_guard("mistral_b580_fa1", mistral_pp, mistral_tg)
            seen_mistral = 1
        }
        END {
            if (!seen_gpt) {
                print "gptoss_b50_fa1", "missing", "1", "NA", "NA", gpt_pp, "NA", gpt_tg, "no"
                failures++
            }
            if (!seen_mistral) {
                print "mistral_b580_fa1", "missing", "1", "NA", "NA", mistral_pp, "NA", mistral_tg, "no"
                failures++
            }
            exit failures ? 1 : 0
        }
    ' "$SUMMARY" > "$guardrails"
}

run_correctness_gates

for fa in 1 0; do
    run_case "gptoss-b580" "level_zero:0" "$GPTOSS_MODEL" "$fa"
    run_case "gptoss-b50" "level_zero:1" "$GPTOSS_MODEL" "$fa"
    run_case "gptoss-dual-b580-b50" "level_zero:0,1" "$GPTOSS_MODEL" "$fa"
done

for fa in 1 0; do
    run_case "mistral-b580" "level_zero:0" "$MISTRAL_MODEL" "$fa"
    run_case "mistral-b50" "level_zero:1" "$MISTRAL_MODEL" "$fa"
done

write_comparison
column -t -s $'\t' "$SUMMARY" | tee "$OUT_DIR/summary.txt"
if [[ "$RUN_CORRECTNESS" == "1" ]]; then
    printf "\nCorrectness:\n" | tee -a "$OUT_DIR/summary.txt"
    column -t -s $'\t' "$CORRECTNESS" | tee -a "$OUT_DIR/summary.txt"
fi
printf "\nComparison:\n" | tee -a "$OUT_DIR/summary.txt"
column -t -s $'\t' "$OUT_DIR/comparison.tsv" | tee -a "$OUT_DIR/summary.txt"
printf "\nGuardrails:\n" | tee -a "$OUT_DIR/summary.txt"
if ! validate_guardrails; then
    column -t -s $'\t' "$OUT_DIR/guardrails.tsv" | tee -a "$OUT_DIR/summary.txt"
    echo "gate failed: single-card performance guardrail missed" >&2
    exit 1
fi
column -t -s $'\t' "$OUT_DIR/guardrails.tsv" | tee -a "$OUT_DIR/summary.txt"
