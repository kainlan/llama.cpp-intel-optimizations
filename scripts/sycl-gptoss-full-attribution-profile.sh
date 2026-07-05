#!/usr/bin/env bash
set -euo pipefail

EXECUTE=0
ACK=0
OUT_ROOT="${SYCL_GPTOSS_FULL_ATTRIBUTION_OUT:-/tmp/sycl_gptoss_full_attribution_$(date +%Y%m%d_%H%M%S)}"
DEVICE_SELECTOR="level_zero:1"
MODEL="${SYCL_GPTOSS_MODEL:-/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf}"
BENCH="${SYCL_LLAMA_BENCH:-./build/bin/llama-bench}"
PROMPT_TOKENS="512"
GEN_TOKENS="128"
REPEAT="1"
SOURCE_KERNEL="mxfp4_pair_glu_xmx_tiled"

usage() {
    printf 'usage: %s [--dry-run|--execute] [--i-understand-this-runs-gpu-models-and-profilers] [options]\n' "$0"
    printf 'default mode is dry-run; real execution requires --execute --i-understand-this-runs-gpu-models-and-profilers\n'
    printf 'options: --out-root DIR --device-selector SELECTOR --model GGUF --bench PATH --prompt-tokens N --gen-tokens N --repeat N --source-kernel NAME\n'
}

require_value() {
    local opt="$1"
    local value="${2-}"
    if [[ -z "${value}" || "${value}" == --* ]]; then
        printf 'error: %s requires a value\n' "${opt}" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            EXECUTE=0
            ;;
        --execute)
            EXECUTE=1
            ;;
        --i-understand-this-runs-gpu-models-and-profilers)
            ACK=1
            ;;
        --out-root)
            require_value "$1" "${2-}"
            OUT_ROOT="$2"
            shift
            ;;
        --device-selector)
            require_value "$1" "${2-}"
            DEVICE_SELECTOR="$2"
            shift
            ;;
        --model)
            require_value "$1" "${2-}"
            MODEL="$2"
            shift
            ;;
        --bench)
            require_value "$1" "${2-}"
            BENCH="$2"
            shift
            ;;
        --prompt-tokens)
            require_value "$1" "${2-}"
            PROMPT_TOKENS="$2"
            shift
            ;;
        --gen-tokens)
            require_value "$1" "${2-}"
            GEN_TOKENS="$2"
            shift
            ;;
        --repeat)
            require_value "$1" "${2-}"
            REPEAT="$2"
            shift
            ;;
        --source-kernel)
            require_value "$1" "${2-}"
            SOURCE_KERNEL="$2"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf 'unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ "${EXECUTE}" -eq 1 && "${ACK}" -ne 1 ]]; then
    printf 'error: --execute requires --i-understand-this-runs-gpu-models-and-profilers\n' >&2
    exit 2
fi

raw_timeline_dir=${OUT_ROOT}/raw/timeline
raw_kernel_dir=${OUT_ROOT}/raw/kernel
pti_dir=${OUT_ROOT}/pti
ur_dir=${OUT_ROOT}/ur
vtune_dir=${OUT_ROOT}/vtune
source_line_dir=${OUT_ROOT}/source-line
parsed_dir=${OUT_ROOT}/parsed
source_line_build_matrix=${source_line_dir}/build-matrix

raw_timeline=${raw_timeline_dir}/sycl-timeline.json
raw_kernel_base=${raw_kernel_dir}/sycl-kernels
raw_kernel_csv=${raw_kernel_dir}/sycl-kernels.csv
pti_l0_trace=${pti_dir}/level-zero-api.jsonl
ur_trace=${ur_dir}/sycl-ur-trace.log
vtune_result_dir=${vtune_dir}/result
vtune_kernel_csv=${vtune_dir}/exported-kernels.csv
vtune_source_csv=${vtune_dir}/exported-source-lines.csv
source_line_case=${source_line_build_matrix}/debug_full
source_line_sections=${source_line_case}/zebin-debug-sections.txt
source_line_dwarf_dump=${source_line_case}/zebin-debug-line.txt
source_line_dwarf_csv=${source_line_case}/dwarf-source-lines.csv
source_line_vtune_csv=${source_line_case}/vtune-gpu-source-line.csv
source_region_map=activation/sycl-source-region-map.json
source_line_probe_kernel=sycl_source_line_probe

env_args=(
    "ONEAPI_DEVICE_SELECTOR=${DEVICE_SELECTOR}"
    "GGML_SYCL_E2E_TG_PROFILE=1"
    "GGML_SYCL_TIMELINE=timeline+events"
    "GGML_SYCL_TIMELINE_OUTPUT=${raw_timeline}"
    "GGML_SYCL_KERNEL_PROFILE=1"
    "GGML_SYCL_KERNEL_PROFILE_OUTPUT=${raw_kernel_base}"
    "GGML_SYCL_KERNEL_PROFILE_FORMAT=both"
    "GGML_SYCL_KERNEL_PROFILE_RAW=1"
    "GGML_SYCL_KERNEL_PROFILE_TOP_N=80"
    "GGML_SYCL_VTUNE_ITT=1"
    "GGML_SYCL_MOE_PHASE_MATERIALIZE=1"
    "GGML_SYCL_MOE_PHASE_BULK_XMX=1"
    "GGML_SYCL_MOE_DOWN_SUM_DIRECT=1"
    "ZE_ENABLE_TRACING_LAYER=1"
    "PTI_L0_TRACE_OUTPUT=${pti_l0_trace}"
    "SYCL_UR_TRACE=2"
    "SYCL_UR_TRACE_LOG=${ur_trace}"
)

bench_args=(
    "${BENCH}"
    -m "${MODEL}"
    -ngl 99
    -fa 1
    -p "${PROMPT_TOKENS}"
    -n "${GEN_TOKENS}"
    -r "${REPEAT}"
)

print_cmd() {
    printf 'env'
    local item
    for item in "${env_args[@]}"; do
        printf ' %q' "${item}"
    done
    for item in "${bench_args[@]}"; do
        printf ' %q' "${item}"
    done
}

print_parse_plan() {
    printf 'python3 %q %q >%q\n' "scripts/parse-sycl-timeline.py" "${raw_timeline}" "${parsed_dir}/timeline.parse"
    printf 'python3 %q --top-kernels 40 %q >%q\n' "scripts/parse-sycl-kernel-profile.py" "${raw_kernel_csv}" "${parsed_dir}/kernel-cost.parse"
    printf 'python3 %q %q >%q\n' "scripts/parse-sycl-pti-l0.py" "${pti_l0_trace}" "${parsed_dir}/l0.parse"
    printf 'python3 %q %q >%q\n' "scripts/parse-sycl-ur-trace.py" "${ur_trace}" "${parsed_dir}/ur.parse"
    printf 'python3 %q --kernel-csv %q --source-csv %q >%q\n' "scripts/parse-sycl-vtune-exports.py" "${vtune_kernel_csv}" "${vtune_source_csv}" "${parsed_dir}/vtune.parse"
    printf 'python3 %q --timeline %q --kernel-profile %q --l0-summary %q --ur-summary %q --vtune-summary %q --bench-stderr %q >%q\n' "scripts/parse-sycl-layer-ledger.py" "${raw_timeline}" "${raw_kernel_csv}" "${parsed_dir}/l0.parse" "${parsed_dir}/ur.parse" "${parsed_dir}/vtune.parse" "${OUT_ROOT}/bench.stderr" "${parsed_dir}/layer-ledger.parse"
    printf 'bash %q --execute --i-understand-this-runs-gpu-source-probe --out-root %q --device-selector %q\n' "scripts/sycl-source-line-debug-matrix.sh" "${source_line_dir}" "${DEVICE_SELECTOR}"
    printf '# source-line matrix conversion: python3 %q --input %q --output %q --source-computing-task %q\n' "scripts/convert-sycl-zebin-line-table-to-source-csv.py" "${source_line_dwarf_dump}" "${source_line_dwarf_csv}" "${source_line_probe_kernel}"
    printf 'python3 %q --readelf-sections %q --vtune-csv %q --require-kernel %q --dwarf-line-dump %q --dwarf-source-lines-csv %q --allow-dwarf-line-table-only --vtune-stdout %q --vtune-stderr %q >%q\n' "scripts/check-sycl-vtune-source-lines.py" "${source_line_sections}" "${source_line_vtune_csv}" "${source_line_probe_kernel}" "${source_line_dwarf_dump}" "${source_line_dwarf_csv}" "${source_line_case}/probe.stdout" "${source_line_case}/probe.stderr" "${parsed_dir}/source-line.parse"
    printf 'if [[ -f %q && -f %q ]]; then python3 %q --cost-ranking %q --source-line %q --region-map %q >%q; else printf "source_attribution.status missing_parser\\nsource_attribution.blocker missing_parse_sycl_source_attribution_or_region_map\\n" >%q; fi\n' "scripts/parse-sycl-source-attribution.py" "${source_region_map}" "scripts/parse-sycl-source-attribution.py" "${parsed_dir}/kernel-cost.parse" "${parsed_dir}/source-line.parse" "${source_region_map}" "${parsed_dir}/source-attribution.parse" "${parsed_dir}/source-attribution.parse"
}

require_file() {
    local path="$1"
    if [[ ! -s "${path}" ]]; then
        printf 'error: required profiling artifact is missing or empty: %s\n' "${path}" >&2
        exit 1
    fi
}

if [[ "${EXECUTE}" -ne 1 ]]; then
    printf 'DRY RUN: would execute full-attribution GPT-OSS SYCL profile.\n'
    printf '# output root: %s\n' "${OUT_ROOT}"
    printf '# raw timeline: %s\n' "${raw_timeline}"
    printf '# raw kernel: %s\n' "${raw_kernel_csv}"
    printf '# PTI Level Zero trace: %s\n' "${pti_l0_trace}"
    printf '# UR trace: %s\n' "${ur_trace}"
    printf '# VTune kernel export: %s\n' "${vtune_kernel_csv}"
    printf '# VTune source export: %s\n' "${vtune_source_csv}"
    printf '# source-line build matrix: %s\n' "${source_line_build_matrix}"
    printf '# parsed layer ledger: %s\n' "${parsed_dir}/layer-ledger.parse"
    printf 'vtune -collect gpu-hotspots -knob gpu-profiling-mode=source-analysis -knob source-analysis=mem-latency -knob dump-compute-task-binaries=true -result-dir %q -- ' "${vtune_result_dir}"
    print_cmd
    printf ' >%q 2>%q\n' "${OUT_ROOT}/bench.stdout" "${OUT_ROOT}/bench.stderr"
    printf 'vtune -report gpu-compute-media-hotspots -r %q -format csv >%q\n' "${vtune_result_dir}" "${vtune_kernel_csv}"
    printf 'vtune -report hotspots -r %q -group-by gpu-source-line -format csv >%q\n' "${vtune_result_dir}" "${vtune_source_csv}"
    print_parse_plan
    printf '# execute with: %q --execute --i-understand-this-runs-gpu-models-and-profilers\n' "$0"
    exit 0
fi

mkdir -p "${raw_timeline_dir}" "${raw_kernel_dir}" "${pti_dir}" "${ur_dir}" "${vtune_dir}" "${source_line_dir}" "${parsed_dir}"
mkdir -p "${source_line_build_matrix}"
set +u
source /opt/intel/oneapi/setvars.sh --force >"${OUT_ROOT}/setvars.log" 2>&1
set -u
print_cmd >"${OUT_ROOT}/command.txt"
printf ' >%q 2>%q\n' "${OUT_ROOT}/bench.stdout" "${OUT_ROOT}/bench.stderr" >>"${OUT_ROOT}/command.txt"
vtune -collect gpu-hotspots \
    -knob gpu-profiling-mode=source-analysis \
    -knob source-analysis=mem-latency \
    -knob dump-compute-task-binaries=true \
    -result-dir "${vtune_result_dir}" \
    -- env "${env_args[@]}" "${bench_args[@]}" >"${OUT_ROOT}/bench.stdout" 2>"${OUT_ROOT}/bench.stderr"
if [[ ! -s "${ur_trace}" ]]; then
    grep '^UR_TRACE ' "${OUT_ROOT}/bench.stderr" >"${ur_trace}" || true
fi
vtune -report gpu-compute-media-hotspots -r "${vtune_result_dir}" -format csv >"${vtune_kernel_csv}"
vtune -report hotspots -r "${vtune_result_dir}" -group-by gpu-source-line -format csv >"${vtune_source_csv}"
bash scripts/sycl-source-line-debug-matrix.sh \
    --execute \
    --i-understand-this-runs-gpu-source-probe \
    --out-root "${source_line_dir}" \
    --device-selector "${DEVICE_SELECTOR}"

require_file "${raw_timeline_dir}/sycl-timeline.json"
require_file "${raw_kernel_dir}/sycl-kernels.csv"
require_file "${pti_dir}/level-zero-api.jsonl"
require_file "${ur_dir}/sycl-ur-trace.log"
require_file "${vtune_dir}/exported-kernels.csv"
require_file "${vtune_dir}/exported-source-lines.csv"
require_file "${source_line_sections}"
require_file "${source_line_vtune_csv}"

python3 scripts/parse-sycl-timeline.py "${raw_timeline_dir}/sycl-timeline.json" >"${parsed_dir}/timeline.parse"
python3 scripts/parse-sycl-kernel-profile.py --top-kernels 40 "${raw_kernel_dir}/sycl-kernels.csv" >"${parsed_dir}/kernel-cost.parse"
python3 scripts/parse-sycl-pti-l0.py "${pti_dir}/level-zero-api.jsonl" >"${parsed_dir}/l0.parse"
python3 scripts/parse-sycl-ur-trace.py "${ur_dir}/sycl-ur-trace.log" >"${parsed_dir}/ur.parse"
python3 scripts/parse-sycl-vtune-exports.py \
    --kernel-csv "${vtune_dir}/exported-kernels.csv" \
    --source-csv "${vtune_dir}/exported-source-lines.csv" >"${parsed_dir}/vtune.parse"
python3 scripts/parse-sycl-layer-ledger.py \
    --timeline "${raw_timeline_dir}/sycl-timeline.json" \
    --kernel-profile "${raw_kernel_dir}/sycl-kernels.csv" \
    --l0-summary "${parsed_dir}/l0.parse" \
    --ur-summary "${parsed_dir}/ur.parse" \
    --vtune-summary "${parsed_dir}/vtune.parse" \
    --bench-stderr "${OUT_ROOT}/bench.stderr" >"${parsed_dir}/layer-ledger.parse"
python3 scripts/check-sycl-vtune-source-lines.py \
    --readelf-sections "${source_line_sections}" \
    --vtune-csv "${source_line_vtune_csv}" \
    --require-kernel "${source_line_probe_kernel}" \
    --dwarf-line-dump "${source_line_dwarf_dump}" \
    --dwarf-source-lines-csv "${source_line_dwarf_csv}" \
    --allow-dwarf-line-table-only \
    --vtune-stdout "${source_line_case}/probe.stdout" \
    --vtune-stderr "${source_line_case}/probe.stderr" >"${parsed_dir}/source-line.parse"
if [[ -f scripts/parse-sycl-source-attribution.py && -f "${source_region_map}" ]]; then
    python3 scripts/parse-sycl-source-attribution.py \
        --cost-ranking "${parsed_dir}/kernel-cost.parse" \
        --source-line "${parsed_dir}/source-line.parse" \
        --region-map "${source_region_map}" >"${parsed_dir}/source-attribution.parse"
else
    printf 'source_attribution.status missing_parser\n' >"${parsed_dir}/source-attribution.parse"
    printf 'source_attribution.blocker missing_parse_sycl_source_attribution_or_region_map\n' >>"${parsed_dir}/source-attribution.parse"
fi
printf 'Artifacts: %s\n' "${OUT_ROOT}"
