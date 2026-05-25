#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${BENCH:-$ROOT/build/bin/llama-bench}"
MODEL="${MODEL:-/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf}"
SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:1}"
OUTDIR="${OUTDIR:-/tmp/b50-gptoss-util.$(date +%Y%m%d-%H%M%S)}"

PP_TOKENS="${PP_TOKENS:-512}"
TG_TOKENS="${TG_TOKENS:-128}"
LONG_TG_TOKENS="${LONG_TG_TOKENS:-1024}"
PROFILE_TOKENS="${PROFILE_TOKENS:-32}"
REPS="${REPS:-1}"
FA_VALUES="${FA_VALUES:-1 0}"

XPU_DEVICE_ID="${XPU_DEVICE_ID:-}"
XPU_INTERVAL="${XPU_INTERVAL:-1}"
XPU_SAMPLES="${XPU_SAMPLES:-90}"
RUN_TELEMETRY="${RUN_TELEMETRY:-1}"
RUN_PROFILE="${RUN_PROFILE:-1}"

B50_BW_GBPS="${B50_BW_GBPS:-224}"
B50_INT8_TOPS="${B50_INT8_TOPS:-170}"
B50_TDP_W="${B50_TDP_W:-70}"

usage() {
    cat <<'EOF'
Usage: scripts/sycl-gptoss-b50-utilization.sh

Runs the current B50 GPT-OSS20B utilization evidence package sequentially:
  1. pp512/tg128 FA on/off llama-bench
  2. a longer FA-on TG run with sudo xpu-smi telemetry
  3. an optional MoE/OP_TIMING profile run

Environment:
  BENCH                 llama-bench path, default build/bin/llama-bench
  MODEL                 GPT-OSS MXFP4 model path
  ONEAPI_DEVICE_SELECTOR SYCL selector, default level_zero:1
  XPU_DEVICE_ID         xpu-smi device id; auto-detects Arc Pro B50 when empty
  OUTDIR                output directory, default /tmp/b50-gptoss-util.<timestamp>
  PP_TOKENS             prompt tokens, default 512
  TG_TOKENS             TG tokens, default 128
  LONG_TG_TOKENS        telemetry TG tokens, default 1024
  PROFILE_TOKENS        profile TG tokens, default 32
  FA_VALUES             FA values for pp/tg sweep, default "1 0"
  RUN_TELEMETRY=0       skip xpu-smi telemetry
  RUN_PROFILE=0         skip MoE/OP_TIMING profile
  B50_BW_GBPS           published/reference bandwidth, default 224
  B50_INT8_TOPS         published/reference INT8 TOPS, default 170
  B50_TDP_W             published/reference board power, default 70

Notes:
  - This script intentionally runs one GPU workload at a time.
  - sudo is used only for xpu-smi counters that require MEI telemetry access.
  - Published B50 values are reporting references only, not runtime policy inputs.
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

if [[ ! -f /opt/intel/oneapi/setvars.sh ]]; then
    echo "oneAPI setvars not found at /opt/intel/oneapi/setvars.sh" >&2
    exit 2
fi

set +u
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u

mkdir -p "$OUTDIR"

bench_summary="$OUTDIR/bench-summary.tsv"
telemetry_summary="$OUTDIR/telemetry-summary.tsv"
profile_summary="$OUTDIR/profile-summary.txt"
env_summary="$OUTDIR/environment.txt"

cat >"$env_summary" <<EOF
BENCH=$BENCH
MODEL=$MODEL
ONEAPI_DEVICE_SELECTOR=$SELECTOR
XPU_DEVICE_ID=${XPU_DEVICE_ID:-auto}
PP_TOKENS=$PP_TOKENS
TG_TOKENS=$TG_TOKENS
LONG_TG_TOKENS=$LONG_TG_TOKENS
PROFILE_TOKENS=$PROFILE_TOKENS
REPS=$REPS
B50_BW_GBPS=$B50_BW_GBPS
B50_INT8_TOPS=$B50_INT8_TOPS
B50_TDP_W=$B50_TDP_W
EOF

printf "case\tfa\tpp_tok_s\ttg_tok_s\tlog\n" >"$bench_summary"
printf "case\trows\tactive_rows\tsteady_rows\tavg_read_gbps\tavg_write_gbps\tavg_total_gbps\tpct_ref_bw\tpeak_total_gbps\tpeak_pct_ref_bw\tavg_compute_pct\tavg_eu_active_pct\tavg_eu_stall_pct\tavg_eu_idle_pct\tavg_power_w\tpct_ref_power\tavg_freq_mhz\tmax_freq_mhz\tlog\n" >"$telemetry_summary"

detect_xpu_device_id() {
    if [[ -n "$XPU_DEVICE_ID" ]]; then
        printf "%s\n" "$XPU_DEVICE_ID"
        return
    fi
    if ! command -v xpu-smi >/dev/null 2>&1; then
        printf "\n"
        return
    fi
    xpu-smi discovery 2>/dev/null | awk -F'|' '
        /Device Name:.*Arc.*Pro B50/ {
            id = $2
            gsub(/[ \t]/, "", id)
            print id
            exit
        }
    '
}

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

run_bench() {
    local name="$1"
    local fa="$2"
    local pp="$3"
    local tg="$4"
    shift 4
    local -a env_args=("$@")
    local log="$OUTDIR/${name}.log"

    echo "== $name =="
    env ONEAPI_DEVICE_SELECTOR="$SELECTOR" GGML_SYCL_OP_TIMEOUT_MS=120000 "${env_args[@]}" \
        "$BENCH" -m "$MODEL" -p "$pp" -n "$tg" -ngl 99 -fa "$fa" -r "$REPS" \
        >"$log" 2>&1

    local pp_rate tg_rate
    pp_rate="$(extract_rate "$log" pp)"
    tg_rate="$(extract_rate "$log" tg)"
    printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$fa" "$pp_rate" "$tg_rate" "$log" | tee -a "$bench_summary"
}

summarize_telemetry() {
    local name="$1"
    local log="$2"
    python3 - "$name" "$log" "$B50_BW_GBPS" "$B50_TDP_W" <<'PY' >>"$telemetry_summary"
import csv
import statistics
import sys

name, path, ref_bw, ref_power = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
rows = []
with open(path, newline="") as f:
    for raw in f:
        line = raw.strip()
        if not line or line.startswith("WARNING"):
            continue
        if line.startswith("Timestamp"):
            header = next(csv.reader([line]))
            continue
        try:
            parts = next(csv.reader([line]))
        except Exception:
            continue
        if len(parts) < 15:
            continue
        parts = [p.strip() for p in parts]
        if parts[2] == "N/A":
            continue
        try:
            row = {
                "gpu": float(parts[2]),
                "power": float(parts[3]),
                "freq": float(parts[4]),
                "memutil": float(parts[5]),
                "read_kbs": float(parts[6]),
                "write_kbs": float(parts[7]),
                "eu_active": float(parts[8]),
                "eu_stall": float(parts[9]),
                "eu_idle": float(parts[10]),
                "bw_util": float(parts[11]),
                "mem_used": float(parts[12]),
                "compute": float(parts[13]),
            }
        except ValueError:
            continue
        rows.append(row)

active = [r for r in rows if r["compute"] > 50 or r["memutil"] > 50 or r["read_kbs"] > 10_000_000]
steady = [r for r in active if r["compute"] > 90 and r["read_kbs"] > 50_000_000]
subset = steady or active

def mean(key):
    return statistics.mean(r[key] for r in subset) if subset else 0.0

def maxv(key):
    return max((r[key] for r in subset), default=0.0)

avg_read = mean("read_kbs") * 1000.0 / 1e9
avg_write = mean("write_kbs") * 1000.0 / 1e9
avg_total = avg_read + avg_write
peak_total = (maxv("read_kbs") + maxv("write_kbs")) * 1000.0 / 1e9
pct_bw = 100.0 * avg_total / ref_bw if ref_bw else 0.0
peak_pct_bw = 100.0 * peak_total / ref_bw if ref_bw else 0.0
pct_power = 100.0 * mean("power") / ref_power if ref_power else 0.0

print(
    f"{name}\t{len(rows)}\t{len(active)}\t{len(steady)}\t"
    f"{avg_read:.2f}\t{avg_write:.2f}\t{avg_total:.2f}\t{pct_bw:.2f}\t"
    f"{peak_total:.2f}\t{peak_pct_bw:.2f}\t"
    f"{mean('compute'):.2f}\t{mean('eu_active'):.2f}\t{mean('eu_stall'):.2f}\t{mean('eu_idle'):.2f}\t"
    f"{mean('power'):.2f}\t{pct_power:.2f}\t{mean('freq'):.0f}\t{maxv('freq'):.0f}\t{path}"
)
PY
}

run_bench_with_telemetry() {
    local name="$1"
    local fa="$2"
    local pp="$3"
    local tg="$4"
    local xpu_id="$5"
    local bench_log="$OUTDIR/${name}.log"
    local telemetry_log="$OUTDIR/${name}.xpu-smi.csv"

    echo "== $name with xpu-smi telemetry =="
    if [[ -z "$xpu_id" ]]; then
        echo "xpu-smi B50 device id not found; running benchmark without telemetry" >&2
        run_bench "$name" "$fa" "$pp" "$tg"
        return
    fi

    sudo -n xpu-smi dump -d "$xpu_id" -m 0,1,2,5,6,7,9,10,11,17,18,31,35 \
        -i "$XPU_INTERVAL" -n "$XPU_SAMPLES" >"$telemetry_log" 2>&1 &
    local mon_pid=$!
    sleep 2

    env ONEAPI_DEVICE_SELECTOR="$SELECTOR" GGML_SYCL_OP_TIMEOUT_MS=120000 \
        "$BENCH" -m "$MODEL" -p "$pp" -n "$tg" -ngl 99 -fa "$fa" -r "$REPS" \
        >"$bench_log" 2>&1

    sleep 2
    kill "$mon_pid" 2>/dev/null || true
    wait "$mon_pid" 2>/dev/null || true

    local pp_rate tg_rate
    pp_rate="$(extract_rate "$bench_log" pp)"
    tg_rate="$(extract_rate "$bench_log" tg)"
    printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$fa" "$pp_rate" "$tg_rate" "$bench_log" | tee -a "$bench_summary"
    summarize_telemetry "$name" "$telemetry_log"
}

for fa in $FA_VALUES; do
    run_bench "gptoss-pp${PP_TOKENS}-tg${TG_TOKENS}-fa${fa}" "$fa" "$PP_TOKENS" "$TG_TOKENS"
done

if [[ "$RUN_TELEMETRY" == "1" ]]; then
    xpu_id="$(detect_xpu_device_id)"
    echo "XPU_DEVICE_ID=${xpu_id:-not-found}" >>"$env_summary"
    run_bench_with_telemetry "gptoss-pp${PP_TOKENS}-tg${LONG_TG_TOKENS}-fa1-telemetry" 1 "$PP_TOKENS" "$LONG_TG_TOKENS" "$xpu_id"
fi

if [[ "$RUN_PROFILE" == "1" ]]; then
    run_bench "gptoss-profile-fa1" 1 1 "$PROFILE_TOKENS" \
        GGML_SYCL_MXFP4_TG_PROFILE=1 GGML_SYCL_MOE_PROFILE=1 GGML_SYCL_OP_TIMING=1

    profile_log="$OUTDIR/gptoss-profile-fa1.log"
    {
        echo "== MXFP4 profile tail =="
        grep '\[MXFP4-MOE-TG-PROFILE\]' "$profile_log" | tail -8 || true
        echo
        echo "== OP timing tail =="
        grep -A14 '\[OP-TIMING\] === Graph' "$profile_log" | tail -90 || true
        echo
        echo "== MoE profile summary =="
        grep -A45 'Averages across' "$profile_log" || true
    } >"$profile_summary"
fi

echo "OUTDIR=$OUTDIR"
echo "bench_summary=$bench_summary"
echo "telemetry_summary=$telemetry_summary"
echo "profile_summary=$profile_summary"
echo "environment=$env_summary"
