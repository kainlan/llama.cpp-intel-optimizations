#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${BENCH:-$ROOT/build/bin/llama-bench}"
GGUF_TOOL="${GGUF_TOOL:-$ROOT/build/bin/llama-gguf}"
MODEL="${MODEL:-/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf}"
MISTRAL_MODEL="${MISTRAL_MODEL:-/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf}"
SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:1}"
B580_SELECTOR="${B580_SELECTOR:-level_zero:0}"
OUTDIR="${OUTDIR:-/tmp/sycl-moe-roofline.$(date +%Y%m%d-%H%M%S)}"

TOKENS="${TOKENS:-128}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
REPS="${REPS:-3}"
PROFILE_TOKENS="${PROFILE_TOKENS:-16}"
TG_BATCHES="${TG_BATCHES:-1 4 8 16 32}"
FA_VALUES="${FA_VALUES:-0 1}"
RUN_GPTOSS_PP="${RUN_GPTOSS_PP:-1}"
RUN_MISTRAL="${RUN_MISTRAL:-1}"
THEORETICAL_BW_GBPS="${THEORETICAL_BW_GBPS:-}"
MXFP4_ENTRY_BYTES="${MXFP4_ENTRY_BYTES:-}"
MXFP4_ENTRIES_PER_TOKEN="${MXFP4_ENTRIES_PER_TOKEN:-}"

usage() {
    cat <<'EOF'
Usage: scripts/sycl-moe-roofline-harness.sh

Sequential B50/B580 SYCL MoE profiling harness for GPT-OSS MXFP4 TG work.
It records clean TG throughput, TG batch-scaling, synchronized MXFP4/MoE
subphase timings, FA dispatch path, hardware capability log lines, an optional
memory roofline, selected-expert row aggregation telemetry, and planner
descriptor grouping telemetry.

Environment:
  BENCH                  llama-bench path, default build/bin/llama-bench
  GGUF_TOOL              llama-gguf path, default build/bin/llama-gguf
  MODEL                  GPT-OSS MXFP4 GGUF path
  MISTRAL_MODEL          Mistral regression model path
  ONEAPI_DEVICE_SELECTOR device selector, default level_zero:1
  B580_SELECTOR          Mistral guard selector, default level_zero:0
  OUTDIR                 log directory
  TOKENS                 clean TG tokens, default 128
  PROMPT_TOKENS          clean PP prompt tokens, default 512
  REPS                   clean TG reps, default 3
  PROFILE_TOKENS         synced profile TG tokens, default 16
  TG_BATCHES             space-separated tg-batch values, default "1 4 8 16 32"
  FA_VALUES              space-separated FA values, default "0 1"
  RUN_GPTOSS_PP=0        skip GPT-OSS PP FA on/off sweep
  RUN_MISTRAL=0          skip B580 Mistral FA guard
  THEORETICAL_BW_GBPS    optional board bandwidth for roofline math
  MXFP4_ENTRY_BYTES      optional bytes per selected expert role; inferred from GGUF when possible
  GGML_SYCL_MOE_ROW_AGG_DEBUG_LIMIT
                          max row-aggregation lines in profile runs, default 256 here

Notes:
  - Source oneAPI before running this script.
  - Do not run other GPU workloads while this script is measuring.
  - The script does not special-case a GPU model. It captures runtime caps from logs.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -x "$BENCH" ]]; then
    echo "llama-bench not executable: $BENCH" >&2
    exit 2
fi

mkdir -p "$OUTDIR"

summary="$OUTDIR/summary.tsv"
profile_summary="$OUTDIR/mxfp4-profile.tsv"
roofline_summary="$OUTDIR/roofline.tsv"
row_agg_summary="$OUTDIR/row-agg.tsv"
descriptor_summary="$OUTDIR/row-agg-descriptor.tsv"
caps_log="$OUTDIR/runtime-caps.log"

printf "case\tselector\tfa\ttg_batch\tpp_tok_s\ttg_tok_s\tlog\n" >"$summary"
printf "fa\tentries\tbatches\tavg_total_ms\tavg_kernel_ms\tavg_quant_ms\tavg_gateup_glu_ms\tavg_down_ms\tavg_per_entry_kernel_us\tlog\n" >"$profile_summary"
printf "case\tfa\ttg_batch\ttg_tok_s\tentry_bytes\tentries_per_token\tweight_gb_per_token\teffective_weight_gbps\ttheoretical_bw_gbps\troofline_tg_tok_s\n" >"$roofline_summary"
printf "fa\tpath\ttensor\tlayer\trole\tdevice\tlayout\tbatch\ttopk\tselected\tunique\tmax_per_expert\tavg_per_expert\tlog\n" >"$row_agg_summary"
printf "fa\tpath\ttensor\tlayer\trole\tsubmit_device\tlayout\tbatch\ttopk\tentries\tgroups\tdevice_rows\thost_rows\tmissing_rows\tlayout_mismatch\tunique_handles\tmax_rows\tavg_rows_per_group\tlog\n" >"$descriptor_summary"

extract_rate() {
    local log="$1"
    local kind="$2"
    awk -v kind="$kind" '
        $0 ~ "\\|[[:space:]]*" kind "[0-9]+[[:space:]]*\\|" {
            for (i = 1; i <= NF; ++i) {
                if ($i == "±" && i > 1) {
                    rate = $(i - 1)
                }
            }
        }
        END { if (rate != "") print rate; else print "NA" }
    ' "$log"
}

extract_caps() {
    local log="$1"
    rg -n '^\[UNIFIED-CACHE\] Device|^\[SYCL\] Device [0-9]+ caps|^\[SYCL\] Device [0-9]+ alloc caps|^\[XMX\]' "$log" \
        >>"$caps_log" || true
}

infer_mxfp4_entry_bytes() {
    if [[ -n "$MXFP4_ENTRY_BYTES" ]]; then
        printf "%s\n" "$MXFP4_ENTRY_BYTES"
        return
    fi
    if [[ ! -x "$GGUF_TOOL" ]]; then
        printf "0\n"
        return
    fi

    local meta
    set +o pipefail
    meta="$("$GGUF_TOOL" "$MODEL" r n 2>&1 | awk '
        /ffn_.*_exps\.weight, size =/ && weight_size == "" {
            line = $0
            sub(/.*size = /, "", line)
            sub(/,.*/, "", line)
            weight_size = line
        }
        /n_dims = 3/ && /ffn_.*_exps\.weight/ && n_expert == "" {
            line = $0
            sub(/.*ne = \(/, "", line)
            sub(/\).*/, "", line)
            gsub(/ /, "", line)
            split(line, dims, ",")
            n_expert = dims[3]
        }
        weight_size != "" && n_expert != "" {
            print weight_size, n_expert
            exit
        }
    ')"
    set -o pipefail

    local weight_size n_expert
    weight_size="$(awk '{ print $1 }' <<<"$meta")"
    n_expert="$(awk '{ print $2 }' <<<"$meta")"
    if [[ -n "$weight_size" && -n "$n_expert" && "$n_expert" != "0" ]]; then
        awk -v bytes="$weight_size" -v experts="$n_expert" 'BEGIN { printf "%.0f\n", bytes / experts }'
    else
        printf "0\n"
    fi
}

append_roofline() {
    local name="$1"
    local fa="$2"
    local tgb="$3"
    local tg="$4"
    local entries="$5"
    local entry_bytes="$6"

    if [[ "$tg" == "NA" || "$entry_bytes" == "0" || "$entries" == "0" ]]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\tNA\tNA\t%s\tNA\n" \
            "$name" "$fa" "$tgb" "$tg" "$entry_bytes" "$entries" "${THEORETICAL_BW_GBPS:-NA}" >>"$roofline_summary"
        return
    fi

    awk -v name="$name" -v fa="$fa" -v tgb="$tgb" -v tg="$tg" -v entry_bytes="$entry_bytes" \
        -v entries="$entries" -v bw="${THEORETICAL_BW_GBPS:-0}" '
        BEGIN {
            bytes_per_token = entry_bytes * entries
            gb_per_token = bytes_per_token / 1e9
            effective = gb_per_token * tg
            roofline = (bw > 0 && gb_per_token > 0) ? bw / gb_per_token : 0
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%.6f\t%.2f\t%s\t%s\n",
                name, fa, tgb, tg, entry_bytes, entries, gb_per_token, effective,
                (bw > 0 ? bw : "NA"), (roofline > 0 ? sprintf("%.2f", roofline) : "NA")
        }
    ' >>"$roofline_summary"
}

parse_profile() {
    local fa="$1"
    local log="$2"
    awk -v fa="$fa" -v logfile="$log" '
        /\[MXFP4-MOE-TG-PROFILE\]/ {
            entries = batches = 0
            mode = "main"
            for (i = 1; i <= NF; ++i) {
                if ($i == "per_call") {
                    mode = "per_call"
                    continue
                }
                if ($i == "per_entry") {
                    mode = "per_entry"
                    continue
                }
                split($i, kv, "=")
                if (kv[1] == "entries") entries = kv[2] + 0
                if (kv[1] == "batches") batches = kv[2] + 0
                if (mode == "main" && kv[1] == "total") total += kv[2] + 0
                if (mode == "main" && kv[1] == "kernel") kernel += kv[2] + 0
                if (mode == "main" && kv[1] == "quant") quant += kv[2] + 0
                if (mode == "main" && kv[1] == "gateup_glu") gateup += kv[2] + 0
                if (mode == "main" && kv[1] == "down") down += kv[2] + 0
                if (mode == "per_entry" && kv[1] == "kernel") per_entry += kv[2] + 0
            }
            last_entries = entries
            last_batches = batches
            count++
        }
        END {
            if (count == 0) {
                printf "%s\t0\t0\tNA\tNA\tNA\tNA\tNA\tNA\t%s\n", fa, logfile
            } else {
                printf "%s\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\n",
                    fa, last_entries, last_batches, total / count, kernel / count, quant / count,
                    gateup / count, down / count, per_entry / count, logfile
            }
        }
    ' "$log" >>"$profile_summary"
}

parse_row_agg() {
    local fa="$1"
    local log="$2"
    awk -v fa="$fa" -v logfile="$log" '
        /\[MOE-ROW-AGG\]/ && /stage=selected/ {
            path = tensor = layer = role = device = layout = batch = topk = selected = unique = max_per_expert = avg = "NA"
            for (i = 1; i <= NF; ++i) {
                split($i, kv, "=")
                if (kv[1] == "path") path = kv[2]
                if (kv[1] == "tensor") tensor = kv[2]
                if (kv[1] == "layer") layer = kv[2]
                if (kv[1] == "role") role = kv[2]
                if (kv[1] == "device") device = kv[2]
                if (kv[1] == "layout") layout = kv[2]
                if (kv[1] == "batch") batch = kv[2]
                if (kv[1] == "topk") topk = kv[2]
                if (kv[1] == "selected") selected = kv[2]
                if (kv[1] == "unique") unique = kv[2]
                if (kv[1] == "max_per_expert") max_per_expert = kv[2]
                if (kv[1] == "avg_per_expert") avg = kv[2]
            }
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                fa, path, tensor, layer, role, device, layout, batch, topk, selected,
                unique, max_per_expert, avg, logfile
        }
    ' "$log" >>"$row_agg_summary"
}

parse_descriptor_agg() {
    local fa="$1"
    local log="$2"
    awk -v fa="$fa" -v logfile="$log" '
        /\[MOE-ROW-AGG\]/ && /stage=descriptor/ {
            path = tensor = layer = role = device = layout = batch = topk = entries = groups = "NA"
            device_rows = host_rows = missing_rows = layout_mismatch = unique_handles = max_rows = avg = "NA"
            for (i = 1; i <= NF; ++i) {
                split($i, kv, "=")
                if (kv[1] == "path") path = kv[2]
                if (kv[1] == "tensor") tensor = kv[2]
                if (kv[1] == "layer") layer = kv[2]
                if (kv[1] == "role") role = kv[2]
                if (kv[1] == "submit_device") device = kv[2]
                if (kv[1] == "layout") layout = kv[2]
                if (kv[1] == "batch") batch = kv[2]
                if (kv[1] == "topk") topk = kv[2]
                if (kv[1] == "entries") entries = kv[2]
                if (kv[1] == "groups") groups = kv[2]
                if (kv[1] == "device_rows") device_rows = kv[2]
                if (kv[1] == "host_rows") host_rows = kv[2]
                if (kv[1] == "missing_rows") missing_rows = kv[2]
                if (kv[1] == "layout_mismatch") layout_mismatch = kv[2]
                if (kv[1] == "unique_handles") unique_handles = kv[2]
                if (kv[1] == "max_rows") max_rows = kv[2]
                if (kv[1] == "avg_rows_per_group") avg = kv[2]
            }
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                fa, path, tensor, layer, role, device, layout, batch, topk, entries, groups,
                device_rows, host_rows, missing_rows, layout_mismatch, unique_handles, max_rows, avg, logfile
        }
    ' "$log" >>"$descriptor_summary"
}

extract_profile_entries() {
    local log="$1"
    awk '
        /\[MXFP4-MOE-TG-PROFILE\]/ {
            for (i = 1; i <= NF; ++i) {
                split($i, kv, "=")
                if (kv[1] == "entries") entries = kv[2] + 0
            }
        }
        END { print entries + 0 }
    ' "$log"
}

write_roofline_from_summary() {
    local entries="$1"
    tail -n +2 "$summary" | while IFS=$'\t' read -r name selector fa tgb pp tg log; do
        case "$name" in
            gptoss-tg-fa*)
                append_roofline "$name" "$fa" "$tgb" "$tg" "$entries" "$entry_bytes"
                ;;
        esac
    done
}

run_case() {
    local name="$1"
    local selector="$2"
    local model="$3"
    local prompt="$4"
    local tokens="$5"
    local reps="$6"
    local fa="$7"
    local tg_batch="$8"
    shift 8
    local -a env_args=("$@")
    local log="$OUTDIR/${name}.log"

    echo "== $name =="
    env ONEAPI_DEVICE_SELECTOR="$selector" "${env_args[@]}" \
        "$BENCH" -m "$model" -p "$prompt" -n "$tokens" -r "$reps" -ngl 99 -fa "$fa" --tg-batch "$tg_batch" \
        >"$log" 2>&1

    extract_caps "$log"

    local pp tg
    pp="$(extract_rate "$log" pp)"
    tg="$(extract_rate "$log" tg)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$selector" "$fa" "$tg_batch" "$pp" "$tg" "$log" \
        | tee -a "$summary"
}

entry_bytes="$(infer_mxfp4_entry_bytes)"
echo "MXFP4_ENTRY_BYTES=$entry_bytes" >"$OUTDIR/model-mxfp4-bytes.env"
entries_per_token="${MXFP4_ENTRIES_PER_TOKEN:-0}"

for fa in $FA_VALUES; do
    if [[ "$RUN_GPTOSS_PP" == "1" ]]; then
        run_case "gptoss-pp-fa${fa}" "$SELECTOR" "$MODEL" "$PROMPT_TOKENS" 0 "$REPS" "$fa" 1
    fi

    for tgb in $TG_BATCHES; do
        run_case "gptoss-tg-fa${fa}-tgb${tgb}" "$SELECTOR" "$MODEL" 0 "$TOKENS" "$REPS" "$fa" "$tgb"
    done

    run_case "gptoss-profile-fa${fa}" "$SELECTOR" "$MODEL" 0 "$PROFILE_TOKENS" 1 "$fa" 1 \
        GGML_SYCL_MOE_PROFILE=1 GGML_SYCL_MXFP4_TG_PROFILE=1 \
        GGML_SYCL_MOE_ROW_AGG_DEBUG=1 GGML_SYCL_MOE_ROW_AGG_DEBUG_LIMIT=256
    parse_profile "$fa" "$OUTDIR/gptoss-profile-fa${fa}.log"
    parse_row_agg "$fa" "$OUTDIR/gptoss-profile-fa${fa}.log"
    parse_descriptor_agg "$fa" "$OUTDIR/gptoss-profile-fa${fa}.log"
    observed_entries="$(extract_profile_entries "$OUTDIR/gptoss-profile-fa${fa}.log")"
    if [[ "$observed_entries" != "0" ]]; then
        entries_per_token="$observed_entries"
    fi
done

write_roofline_from_summary "$entries_per_token"

run_case "gptoss-fa-dispatch-debug" "$SELECTOR" "$MODEL" 0 8 1 1 1 \
    GGML_SYCL_FA_DISPATCH_DEBUG=1 GGML_SYCL_FA_DISPATCH_DEBUG_LIMIT=32

if [[ "$RUN_MISTRAL" == "1" ]]; then
    run_case "mistral-b580-fa0" "$B580_SELECTOR" "$MISTRAL_MODEL" 512 128 "$REPS" 0 1
    run_case "mistral-b580-fa1" "$B580_SELECTOR" "$MISTRAL_MODEL" 512 128 "$REPS" 1 1
fi

echo "summary=$summary"
echo "profile_summary=$profile_summary"
echo "roofline_summary=$roofline_summary"
echo "row_agg_summary=$row_agg_summary"
echo "descriptor_summary=$descriptor_summary"
echo "caps_log=$caps_log"
