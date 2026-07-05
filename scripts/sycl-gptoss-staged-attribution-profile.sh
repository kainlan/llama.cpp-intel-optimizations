#!/usr/bin/env bash
set -euo pipefail

EXECUTE=0
ACK=0
STAGE="all"
OUT_ROOT="${SYCL_GPTOSS_STAGED_OUT:-/tmp/sycl_gptoss_staged_$(date +%Y%m%d_%H%M%S)}"
DEVICE_SELECTOR="level_zero:1"
MODEL="/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf"
PROMPT_TOKENS=512
GEN_TOKENS=128
REPEAT=1
BENCH="./build/bin/llama-bench"
SOURCE_REGION_MAP="activation/sycl-source-region-map.json"
ABLATION_ROUTE="prepack"
ABLATION_KERNEL="mxfp4.gateup.xmx_tiled_dpas_m2"

usage() {
    printf 'usage: %s [--dry-run|--execute] [--i-understand-this-runs-staged-gpu-profiling] [options]\n' "$0"
    printf 'default mode is dry-run; real execution requires --execute and --i-understand-this-runs-staged-gpu-profiling\n'
    printf 'options:\n'
    printf '  --stage {base|ur|l0|vtune-source|ablation|merge|all}\n'
    printf '  --out-root DIR\n'
    printf '  --device-selector SELECTOR\n'
    printf '  --model GGUF\n'
    printf '  --prompt-tokens N --gen-tokens N --repeat N\n'
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
        --i-understand-this-runs-staged-gpu-profiling)
            ACK=1
            ;;
        --stage)
            require_value "$1" "${2-}"
            STAGE="$2"
            shift
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

case "${STAGE}" in
    base|ur|l0|vtune-source|ablation|merge|all) ;;
    *)
        printf 'error: unsupported stage: %s\n' "${STAGE}" >&2
        usage >&2
        exit 2
        ;;
esac

if [[ "${EXECUTE}" -eq 1 && "${ACK}" -ne 1 ]]; then
    printf 'error: --execute requires --i-understand-this-runs-staged-gpu-profiling\n' >&2
    exit 2
fi

base_root() { printf '%s/base' "${OUT_ROOT}"; }
ur_root() { printf '%s/ur' "${OUT_ROOT}"; }
l0_root() { printf '%s/l0' "${OUT_ROOT}"; }
vtune_source_root() { printf '%s/vtune-source' "${OUT_ROOT}"; }
ablation_root() { printf '%s/ablation' "${OUT_ROOT}"; }
merged_root() { printf '%s/merged' "${OUT_ROOT}"; }

safe_env_args=(
    "ONEAPI_DEVICE_SELECTOR=${DEVICE_SELECTOR}"
    "GGML_SYCL_MOE_PHASE_MATERIALIZE=1"
    "GGML_SYCL_MOE_PHASE_BULK_XMX=1"
    "GGML_SYCL_MOE_DOWN_SUM_DIRECT=1"
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

print_env_cmd() {
    printf 'env'
    local item
    for item in "$@"; do
        printf ' %q' "${item}"
    done
    for item in "${bench_args[@]}"; do
        printf ' %q' "${item}"
    done
}

write_manifest() {
    local stage="$1"
    local root="$2"
    local artifact_key="$3"
    local artifact_path="$4"
    mkdir -p "${root}"
    STAGED_STAGE="${stage}" \
    STAGED_ROOT="${root}" \
    STAGED_MODEL="${MODEL}" \
    STAGED_DEVICE_SELECTOR="${DEVICE_SELECTOR}" \
    STAGED_PROMPT_TOKENS="${PROMPT_TOKENS}" \
    STAGED_GEN_TOKENS="${GEN_TOKENS}" \
    STAGED_REPEAT="${REPEAT}" \
    STAGED_ARTIFACT_KEY="${artifact_key}" \
    STAGED_ARTIFACT_PATH="${artifact_path}" \
    python3 - "${root}/stage-manifest.json" <<'PY'
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

out = pathlib.Path(sys.argv[1])
model = pathlib.Path(os.environ["STAGED_MODEL"])
try:
    model_size = model.stat().st_size
except OSError:
    model_size = 0
try:
    build_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except (OSError, subprocess.CalledProcessError):
    build_sha = "unknown"
manifest = {
    "schema_version": 1,
    "stage": os.environ["STAGED_STAGE"],
    "artifact_root": os.environ["STAGED_ROOT"],
    "build_sha": build_sha,
    "model": {"path": str(model), "size": model_size},
    "device_selector": os.environ["STAGED_DEVICE_SELECTOR"],
    "fa": 1,
    "moe_knobs": {
        "GGML_SYCL_MOE_PHASE_MATERIALIZE": "1",
        "GGML_SYCL_MOE_PHASE_BULK_XMX": "1",
        "GGML_SYCL_MOE_DOWN_SUM_DIRECT": "1",
    },
    "prompt_tokens": int(os.environ["STAGED_PROMPT_TOKENS"]),
    "gen_tokens": int(os.environ["STAGED_GEN_TOKENS"]),
    "repeat": int(os.environ["STAGED_REPEAT"]),
    "artifacts": {os.environ["STAGED_ARTIFACT_KEY"]: os.environ["STAGED_ARTIFACT_PATH"]},
}
out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

print_stage_layout() {
    local root="$1"
    printf '#   root: %s\n' "${root}"
    printf '#   manifest: %s/stage-manifest.json\n' "${root}"
}

print_merge_command() {
    printf 'python3 %q' "scripts/merge-sycl-staged-ledger.py"
    printf ' --manifest %q' "$(base_root)/stage-manifest.json"
    printf ' --manifest %q' "$(ur_root)/stage-manifest.json"
    printf ' --manifest %q' "$(l0_root)/stage-manifest.json"
    printf ' --manifest %q' "$(vtune_source_root)/stage-manifest.json"
    printf ' --manifest %q' "$(ablation_root)/stage-manifest.json"
    printf ' --timeline %q' "$(base_root)/sycl-timeline.json"
    printf ' --kernel-profile %q' "$(base_root)/sycl-kernels.csv"
    printf ' --bench-stderr %q' "$(base_root)/bench.stderr"
    printf ' --l0-summary %q' "$(l0_root)/l0.parse"
    printf ' --ur-summary %q' "$(ur_root)/ur.parse"
    printf ' --vtune-summary %q' "$(vtune_source_root)/vtune.parse"
    printf ' --source-line %q' "$(vtune_source_root)/source-line.parse"
    printf ' --source-attribution %q' "$(ablation_root)/source-attribution.parse"
    printf ' >%q\n' "$(merged_root)/staged-ledger.parse"
}

run_base_stage() {
    local root
    root="$(base_root)"
    if [[ "${EXECUTE}" -ne 1 ]]; then
        printf 'stage=base\n'
        print_stage_layout "${root}"
        printf '#   raw timeline: %s/sycl-timeline.json\n' "${root}"
        printf '#   raw kernel profile: %s/sycl-kernels.csv\n' "${root}"
        print_env_cmd \
            "${safe_env_args[@]}" \
            "GGML_SYCL_TIMELINE=timeline+events" \
            "GGML_SYCL_TIMELINE_OUTPUT=${root}/sycl-timeline.json" \
            "GGML_SYCL_KERNEL_PROFILE=1" \
            "GGML_SYCL_KERNEL_PROFILE_OUTPUT=${root}/sycl-kernels" \
            "GGML_SYCL_KERNEL_PROFILE_FORMAT=both" \
            "GGML_SYCL_KERNEL_PROFILE_RAW=1"
        printf ' >%q 2>%q\n' "${root}/bench.stdout" "${root}/bench.stderr"
        printf 'python3 %q %q >%q\n' "scripts/parse-sycl-timeline.py" "${root}/sycl-timeline.json" "${root}/timeline.parse"
        printf 'python3 %q --top-kernels 40 %q >%q\n' "scripts/parse-sycl-kernel-profile.py" "${root}/sycl-kernels.csv" "${root}/cost-ranking.parse"
        return 0
    fi

    mkdir -p "${root}"
    env \
        "${safe_env_args[@]}" \
        "GGML_SYCL_TIMELINE=timeline+events" \
        "GGML_SYCL_TIMELINE_OUTPUT=${root}/sycl-timeline.json" \
        "GGML_SYCL_KERNEL_PROFILE=1" \
        "GGML_SYCL_KERNEL_PROFILE_OUTPUT=${root}/sycl-kernels" \
        "GGML_SYCL_KERNEL_PROFILE_FORMAT=both" \
        "GGML_SYCL_KERNEL_PROFILE_RAW=1" \
        "${bench_args[@]}" >"${root}/bench.stdout" 2>"${root}/bench.stderr"
    python3 scripts/parse-sycl-timeline.py "${root}/sycl-timeline.json" >"${root}/timeline.parse"
    python3 scripts/parse-sycl-kernel-profile.py "${root}/sycl-kernels.csv" >"${root}/kernels.parse"
    python3 scripts/parse-sycl-kernel-profile.py --top-kernels 40 "${root}/sycl-kernels.csv" >"${root}/cost-ranking.parse"
    write_manifest base "${root}" timeline "${root}/sycl-timeline.json"
}

run_ur_stage() {
    local root
    root="$(ur_root)"
    if [[ "${EXECUTE}" -ne 1 ]]; then
        printf 'stage=ur\n'
        print_stage_layout "${root}"
        print_env_cmd "${safe_env_args[@]}" "SYCL_UR_TRACE=2"
        printf ' >%q 2>%q\n' "${root}/bench.stdout" "${root}/bench.stderr"
        printf 'python3 %q %q >%q\n' "scripts/convert-sycl-ur-stderr.py" "${root}/bench.stderr" "${root}/sycl-ur-trace.log"
        printf 'python3 %q %q >%q\n' "scripts/parse-sycl-ur-trace.py" "${root}/sycl-ur-trace.log" "${root}/ur.parse"
        return 0
    fi

    mkdir -p "${root}"
    env "${safe_env_args[@]}" "SYCL_UR_TRACE=2" "${bench_args[@]}" >"${root}/bench.stdout" 2>"${root}/bench.stderr"
    python3 scripts/convert-sycl-ur-stderr.py "${root}/bench.stderr" >"${root}/sycl-ur-trace.log"
    python3 scripts/parse-sycl-ur-trace.py "${root}/sycl-ur-trace.log" >"${root}/ur.parse"
    write_manifest ur "${root}" ur_summary "${root}/ur.parse"
}

run_l0_stage() {
    local root
    root="$(l0_root)"
    if [[ "${EXECUTE}" -ne 1 ]]; then
        printf 'stage=l0\n'
        print_stage_layout "${root}"
        printf '#   accepted inputs: %s/level-zero-api.jsonl or %s/vtune-host-tasks.txt\n' "${root}" "${root}"
        printf 'python3 %q %q >%q\n' "scripts/parse-sycl-pti-l0.py" "${root}/level-zero-api.jsonl" "${root}/l0.parse"
        printf 'python3 %q %q >%q\n' "scripts/convert-sycl-vtune-l0-host-tasks.py" "${root}/vtune-host-tasks.txt" "${root}/level-zero-api.jsonl"
        return 0
    fi

    mkdir -p "${root}"
    if [[ -s "${root}/level-zero-api.jsonl" ]]; then
        python3 scripts/parse-sycl-pti-l0.py "${root}/level-zero-api.jsonl" >"${root}/l0.parse"
    elif [[ -s "${root}/vtune-host-tasks.txt" ]]; then
        python3 scripts/convert-sycl-vtune-l0-host-tasks.py "${root}/vtune-host-tasks.txt" >"${root}/level-zero-api.jsonl"
        python3 scripts/parse-sycl-pti-l0.py "${root}/level-zero-api.jsonl" >"${root}/l0.parse"
    else
        printf 'error: l0 stage requires %s/level-zero-api.jsonl or %s/vtune-host-tasks.txt\n' "${root}" "${root}" >&2
        return 2
    fi
    write_manifest l0 "${root}" l0_summary "${root}/l0.parse"
}

run_vtune_source_stage() {
    local root
    root="$(vtune_source_root)"
    if [[ "${EXECUTE}" -ne 1 ]]; then
        printf 'stage=vtune-source\n'
        print_stage_layout "${root}"
        printf 'bash %q --execute --i-understand-this-runs-gpu-source-probe --out-root %q --device-selector %q\n' "scripts/sycl-source-line-debug-matrix.sh" "${root}/source-line-matrix" "${DEVICE_SELECTOR}"
        printf '#   matrix outputs: %s/source-line-matrix/build-matrix/<case>/source-line-feasibility.parse\n' "${root}"
        printf '#   matrix outputs: %s/source-line-matrix/build-matrix/<case>/zebin-debug-sections.txt\n' "${root}"
        printf '#   matrix outputs: %s/source-line-matrix/build-matrix/<case>/vtune-gpu-source-line.csv\n' "${root}"
        printf '#   selection: prefer source_line.status pass; else source_line.blocker vtune_no_gpu_side_trace or vtune_unknown_source; else first source-line-feasibility.parse\n'
        printf 'cp %q %q\n' "${root}/source-line-matrix/build-matrix/<selected-case>/source-line-feasibility.parse" "${root}/source-line.parse"
        printf 'python3 %q --source-csv %q >%q\n' "scripts/parse-sycl-vtune-exports.py" "${root}/source-line-matrix/build-matrix/<selected-case>/vtune-gpu-source-line.csv" "${root}/vtune.parse"
        return 0
    fi

    mkdir -p "${root}"
    bash scripts/sycl-source-line-debug-matrix.sh \
        --execute \
        --i-understand-this-runs-gpu-source-probe \
        --out-root "${root}/source-line-matrix" \
        --device-selector "${DEVICE_SELECTOR}"

    local matrix_root
    local selected_parse=""
    local vtune_no_gpu_trace_parse=""
    local vtune_unknown_parse=""
    local fallback_parse=""
    local parse
    local selected_dir
    local selected_source_csv
    matrix_root="${root}/source-line-matrix/build-matrix"
    shopt -s nullglob
    local -a source_line_parses=("${matrix_root}"/*/source-line-feasibility.parse)
    shopt -u nullglob
    if [[ "${#source_line_parses[@]}" -eq 0 ]]; then
        printf 'error: vtune-source stage found no matrix source-line-feasibility.parse files under %s\n' "${matrix_root}" >&2
        return 2
    fi
    for parse in "${source_line_parses[@]}"; do
        if [[ -z "${fallback_parse}" ]]; then
            fallback_parse="${parse}"
        fi
        if grep -qx 'source_line.status pass' "${parse}"; then
            selected_parse="${parse}"
            break
        fi
        if [[ -z "${vtune_no_gpu_trace_parse}" ]] && grep -qx 'source_line.blocker vtune_no_gpu_side_trace' "${parse}"; then
            vtune_no_gpu_trace_parse="${parse}"
        fi
        if [[ -z "${vtune_unknown_parse}" ]] && grep -qx 'source_line.blocker vtune_unknown_source' "${parse}"; then
            vtune_unknown_parse="${parse}"
        fi
    done
    if [[ -z "${selected_parse}" ]]; then
        if [[ -n "${vtune_no_gpu_trace_parse}" ]]; then
            selected_parse="${vtune_no_gpu_trace_parse}"
        elif [[ -n "${vtune_unknown_parse}" ]]; then
            selected_parse="${vtune_unknown_parse}"
        else
            selected_parse="${fallback_parse}"
        fi
    fi

    selected_dir="$(dirname "${selected_parse}")"
    selected_source_csv="${selected_dir}/vtune-gpu-source-line.csv"
    if [[ ! -f "${selected_source_csv}" ]]; then
        printf 'error: vtune-source stage selected %s but missing %s\n' "${selected_parse}" "${selected_source_csv}" >&2
        return 2
    fi
    cp "${selected_parse}" "${root}/source-line.parse"
    python3 scripts/parse-sycl-vtune-exports.py \
        --source-csv "${selected_source_csv}" >"${root}/vtune.parse"
    write_manifest vtune-source "${root}" source_line "${root}/source-line.parse"
}

run_ablation_stage() {
    local root
    root="$(ablation_root)"
    if [[ "${EXECUTE}" -ne 1 ]]; then
        printf 'stage=ablation\n'
        print_stage_layout "${root}"
        printf 'python3 %q --microbench-jsonl %q --kernel %q --route %q >%q\n' "scripts/parse-sycl-ablation-deltas.py" "${root}/microbench.jsonl" "${ABLATION_KERNEL}" "${ABLATION_ROUTE}" "${root}/ablation.json"
        printf 'python3 %q --cost-ranking %q --source-line %q --region-map %q --ablation-json %q >%q\n' "scripts/parse-sycl-source-attribution.py" "$(base_root)/cost-ranking.parse" "$(vtune_source_root)/source-line.parse" "${SOURCE_REGION_MAP}" "${root}/ablation.json" "${root}/source-attribution.parse"
        return 0
    fi

    mkdir -p "${root}"
    if [[ ! -s "${root}/microbench.jsonl" ]]; then
        printf 'error: ablation stage requires %s/microbench.jsonl\n' "${root}" >&2
        return 2
    fi
    python3 scripts/parse-sycl-ablation-deltas.py \
        --microbench-jsonl "${root}/microbench.jsonl" \
        --kernel "${ABLATION_KERNEL}" \
        --route "${ABLATION_ROUTE}" >"${root}/ablation.json"
    python3 scripts/parse-sycl-source-attribution.py \
        --cost-ranking "$(base_root)/cost-ranking.parse" \
        --source-line "$(vtune_source_root)/source-line.parse" \
        --region-map "${SOURCE_REGION_MAP}" \
        --ablation-json "${root}/ablation.json" >"${root}/source-attribution.parse"
    write_manifest ablation "${root}" source_attribution "${root}/source-attribution.parse"
}

run_merge_stage() {
    local root
    root="$(merged_root)"
    if [[ "${EXECUTE}" -ne 1 ]]; then
        printf 'stage=merge\n'
        printf '#   root: %s\n' "${root}"
        print_merge_command
        return 0
    fi

    mkdir -p "${root}"
    python3 scripts/merge-sycl-staged-ledger.py \
        --manifest "$(base_root)/stage-manifest.json" \
        --manifest "$(ur_root)/stage-manifest.json" \
        --manifest "$(l0_root)/stage-manifest.json" \
        --manifest "$(vtune_source_root)/stage-manifest.json" \
        --manifest "$(ablation_root)/stage-manifest.json" \
        --timeline "$(base_root)/sycl-timeline.json" \
        --kernel-profile "$(base_root)/sycl-kernels.csv" \
        --bench-stderr "$(base_root)/bench.stderr" \
        --l0-summary "$(l0_root)/l0.parse" \
        --ur-summary "$(ur_root)/ur.parse" \
        --vtune-summary "$(vtune_source_root)/vtune.parse" \
        --source-line "$(vtune_source_root)/source-line.parse" \
        --source-attribution "$(ablation_root)/source-attribution.parse" >"${root}/staged-ledger.parse"
}

run_requested_stage() {
    case "$1" in
        base) run_base_stage ;;
        ur) run_ur_stage ;;
        l0) run_l0_stage ;;
        vtune-source) run_vtune_source_stage ;;
        ablation) run_ablation_stage ;;
        merge) run_merge_stage ;;
        all)
            run_base_stage
            run_ur_stage
            run_l0_stage
            run_vtune_source_stage
            run_ablation_stage
            run_merge_stage
            ;;
    esac
}

if [[ "${EXECUTE}" -ne 1 ]]; then
    printf 'DRY RUN: would execute staged GPT-OSS SYCL attribution profile.\n'
    printf '# output root: %s\n' "${OUT_ROOT}"
    printf '# stage=%s\n' "${STAGE}"
    run_requested_stage "${STAGE}"
    printf '# execute with: %q --execute --i-understand-this-runs-staged-gpu-profiling\n' "$0"
    exit 0
fi

mkdir -p "${OUT_ROOT}"
set +u
source /opt/intel/oneapi/setvars.sh --force >"${OUT_ROOT}/setvars.log" 2>&1
set -u
run_requested_stage "${STAGE}"
printf 'Artifacts: %s\n' "${OUT_ROOT}"
