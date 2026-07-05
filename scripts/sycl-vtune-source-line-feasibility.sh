#!/usr/bin/env bash
set -euo pipefail

EXECUTE=0
ACK=0
OUT_ROOT="/tmp/sycl_vtune_source_line_$(date +%Y%m%d_%H%M%S)"
BUILD_DIR="build-vtune-line"
DEVICE_SELECTOR="level_zero:1"
VTUNE_TARGET_GPU=""
TARGET_KERNEL="mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias"
TASK_GLOB=""
TASK_MATCH="mxfp4_pair_glu_xmx_tiled"
REQUIRE_MATRIX_PASS=""
IGA_PLATFORM="${SYCL_IGA_PLATFORM:-xe2}"

usage() {
    printf 'usage: %s [--dry-run|--execute] [--i-understand-this-runs-gpu-microbenchmarks] [--out-root DIR] [--build-dir DIR] [--device-selector SELECTOR] [--vtune-target-gpu VALUE] [--target-kernel NAME] [--task-glob GLOB] [--task-match TEXT] [--require-matrix-pass PATH] [--iga-platform PLATFORM]\n' "$0"
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
        --dry-run) EXECUTE=0 ;;
        --execute) EXECUTE=1 ;;
        --i-understand-this-runs-gpu-microbenchmarks) ACK=1 ;;
        --out-root) require_value "$1" "${2-}"; OUT_ROOT="$2"; shift ;;
        --build-dir) require_value "$1" "${2-}"; BUILD_DIR="$2"; shift ;;
        --device-selector) require_value "$1" "${2-}"; DEVICE_SELECTOR="$2"; shift ;;
        --vtune-target-gpu) require_value "$1" "${2-}"; VTUNE_TARGET_GPU="$2"; shift ;;
        --target-kernel) require_value "$1" "${2-}"; TARGET_KERNEL="$2"; TASK_MATCH="$2"; shift ;;
        --task-glob) require_value "$1" "${2-}"; TASK_GLOB="$2"; shift ;;
        --task-match) require_value "$1" "${2-}"; TASK_MATCH="$2"; shift ;;
        --require-matrix-pass) require_value "$1" "${2-}"; REQUIRE_MATRIX_PASS="$2"; shift ;;
        --iga-platform) require_value "$1" "${2-}"; IGA_PLATFORM="$2"; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if [[ "${EXECUTE}" -eq 1 && "${ACK}" -ne 1 ]]; then
    printf 'error: --execute requires --i-understand-this-runs-gpu-microbenchmarks\n' >&2
    exit 2
fi

find_asm_for_task() {
    local asm_dir="$1"
    local task="$2"
    local candidate
    while IFS= read -r candidate; do
        if [[ "$(basename "${candidate}")" == *"${task}"* ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done < <(find "${asm_dir}" -type f -name '*.asm' -print)
    find "${asm_dir}" -type f -name '*.asm' -print -quit
}

configure_cmd=(cmake -S . -B "${BUILD_DIR}" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_F16=ON -DGGML_SYCL_PROFILING_DEBUG=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx)
build_cmd=(cmake --build "${BUILD_DIR}" --config Release --target sycl-kernel-bench -j "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}")
bench_cmd=("${BUILD_DIR}/bin/sycl-kernel-bench" --kernel="${TARGET_KERNEL}" --quant=MXFP4 --dim_m=2880 --dim_n=4 --dim_k=2880 --iterations=100 --warmup=10 --validate --output=json)
vtune_dir="${OUT_ROOT}/vtune-source-line"

print_plan() {
    printf 'DRY RUN: would execute lead-only SYCL VTune source-line microbench feasibility.\n'
    printf '# output root: %s\n' "${OUT_ROOT}"
    printf '# IGA platform: %s (override with --iga-platform or SYCL_IGA_PLATFORM)\n' "${IGA_PLATFORM}"
    if [[ -n "${REQUIRE_MATRIX_PASS}" ]]; then
        printf 'matrix gate: require %q to contain source_line.status pass (VTune sampled exact), source_line.status asm-line-static-cost (ASM static source-line cost), or source_line.status dwarf-line-table-only (DWARF line-table fallback)\n' "${REQUIRE_MATRIX_PASS}"
        printf 'grep -Eq %q %q\n' "^source_line.status (pass|asm-line-static-cost|dwarf-line-table-only)$" "${REQUIRE_MATRIX_PASS}"
    fi
    printf '%q ' "${configure_cmd[@]}"; printf '> %q 2>&1\n' "${OUT_ROOT}/profiling-debug-build.log"
    printf '%q ' "${build_cmd[@]}"; printf '>> %q 2>&1\n' "${OUT_ROOT}/profiling-debug-build.log"
    printf 'env ONEAPI_DEVICE_SELECTOR=%q vtune -collect gpu-hotspots' "${DEVICE_SELECTOR}"
    if [[ -n "${VTUNE_TARGET_GPU}" ]]; then
        printf ' -knob %q' "target-gpu=${VTUNE_TARGET_GPU}"
    fi
    printf ' -knob gpu-profiling-mode=source-analysis -knob source-analysis=mem-latency -knob dump-compute-task-binaries=true'
    if [[ -n "${TASK_GLOB}" ]]; then
        printf ' -knob computing-tasks-of-interest=%q' "${TASK_GLOB}#1#1#20"
    fi
    printf ' -result-dir %q -- ' "${vtune_dir}"
    printf '%q ' "${bench_cmd[@]}"; printf '> %q 2> %q\n' "${OUT_ROOT}/bench.stdout" "${OUT_ROOT}/bench.stderr"
    printf 'vtune -report hotspots -r %q -group-by computing-task -format csv > %q\n' "${vtune_dir}" "${OUT_ROOT}/vtune-computing-tasks.csv"
    printf 'python3 scripts/parse-sycl-vtune-tasks.py %q --match %q > %q || printf %q %q >&2\n' "${OUT_ROOT}/vtune-computing-tasks.csv" "${TASK_MATCH}" "${OUT_ROOT}/vtune-task.parse" "warning: failed to parse VTune computing tasks for match %s\\n" "${TASK_MATCH}"
    printf 'first_zebin="$(find %q -name '\''*.zebin'\'' -type f -print -quit)"\n' "${vtune_dir}"
    printf 'llvm-readelf --sections --wide "${first_zebin}" > %q\n' "${OUT_ROOT}/zebin-debug-sections.txt"
    printf 'llvm-dwarfdump --debug-line %q > %q\n' "${vtune_dir}/data.0/<first-zebin>" "${OUT_ROOT}/zebin-debug-line.txt"
    printf 'python3 scripts/convert-sycl-zebin-line-table-to-source-csv.py --input %q --output %q --source-computing-task %q\n' "${OUT_ROOT}/zebin-debug-line.txt" "${OUT_ROOT}/dwarf-source-lines.csv" "${TARGET_KERNEL}"
    printf 'python3 scripts/prepare-sycl-iga-disasm-inputs.py --readelf-sections %q --zebin "${first_zebin}" --kernel-match %q --platform %q --out-dir %q || true\n' "${OUT_ROOT}/zebin-debug-sections.txt" "${TARGET_KERNEL}" "${IGA_PLATFORM}" "${OUT_ROOT}/iga-disasm"
    printf '(cd %q && bash run-iga-disasm.sh) || true  # emits kernel.iga.json using iga64 -Xprint-json -Xprint-pc\n' "${OUT_ROOT}/iga-disasm"
    printf 'python3 scripts/parse-sycl-iga-pc-disasm.py --input %q --format json --kernel %q > %q || true\n' "${OUT_ROOT}/iga-disasm/kernel.iga.json" "${TARGET_KERNEL}" "${OUT_ROOT}/iga-pc-instructions.csv"
    printf 'section_addr="$(python3 -c %q %q)"\n' 'import json,sys; print(json.load(open(sys.argv[1]))["extract.section_addr"])' "${OUT_ROOT}/iga-disasm/iga-disasm-manifest.json"
    printf 'python3 scripts/resolve-sycl-zebin-asm-source-lines.py --dwarf-line-dump %q --iga-instructions-csv %q --pc-base "${section_addr}" --output %q --summary-output %q --source-computing-task %q --require-source-path %q\n' "${OUT_ROOT}/zebin-debug-line.txt" "${OUT_ROOT}/iga-pc-instructions.csv" "${OUT_ROOT}/asm-source-lines.csv" "${OUT_ROOT}/asm-source-lines.parse" "${TARGET_KERNEL}" "mmvq.cpp"
    printf 'mkdir -p %q && cp %q %q && (cd %q && ocloc disasm -file kernel.zebin > ocloc.stdout 2> ocloc.stderr || true)\n' "${OUT_ROOT}/zebin-disasm" "${vtune_dir}/data.0/example.zebin" "${OUT_ROOT}/zebin-disasm/kernel.zebin" "${OUT_ROOT}/zebin-disasm"
    printf 'first_asm="$(find %q -type f -name '\''*%s*.asm'\'' -print -quit)"\n' "${OUT_ROOT}/zebin-disasm" "${TARGET_KERNEL}"
    printf 'if [[ -z "${first_asm}" ]]; then first_asm="$(find %q -type f -name '\''*.asm'\'' -print -quit)"; fi  # resolver rejects unmarked or mismatched ASM\n' "${OUT_ROOT}/zebin-disasm"
    printf 'python3 scripts/resolve-sycl-zebin-asm-source-lines.py --dwarf-line-dump %q --asm "${first_asm}" --output %q --summary-output %q --source-computing-task %q --require-source-path %q\n' "${OUT_ROOT}/zebin-debug-line.txt" "${OUT_ROOT}/asm-source-lines.csv" "${OUT_ROOT}/asm-source-lines.parse" "${TARGET_KERNEL}" "mmvq.cpp"
    printf 'vtune -report hotspots -r %q -group-by gpu-source-line -format csv > %q\n' "${vtune_dir}" "${OUT_ROOT}/vtune-gpu-source-line.csv"
    printf 'python3 scripts/check-sycl-vtune-source-lines.py --readelf-sections %q --vtune-csv %q --require-kernel %q --asm-source-lines-csv %q --allow-asm-line-static-cost --dwarf-line-dump %q --dwarf-source-lines-csv %q --allow-dwarf-line-table-only --require-source-path %q --vtune-stdout %q --vtune-stderr %q > %q\n' "${OUT_ROOT}/zebin-debug-sections.txt" "${OUT_ROOT}/vtune-gpu-source-line.csv" "${TARGET_KERNEL}" "${OUT_ROOT}/asm-source-lines.csv" "${OUT_ROOT}/zebin-debug-line.txt" "${OUT_ROOT}/dwarf-source-lines.csv" "mmvq.cpp" "${OUT_ROOT}/bench.stdout" "${OUT_ROOT}/bench.stderr" "${OUT_ROOT}/source-line-feasibility.parse"
}

if [[ "${EXECUTE}" -ne 1 ]]; then
    print_plan
    exit 0
fi

if [[ -n "${REQUIRE_MATRIX_PASS}" ]] && ! grep -Eq "^source_line.status (pass|asm-line-static-cost|dwarf-line-table-only)$" "${REQUIRE_MATRIX_PASS}"; then
    printf 'error: MXFP4 source-line matrix gate failed: %s must contain source_line.status pass (VTune sampled exact), source_line.status asm-line-static-cost (ASM static source-line cost), or source_line.status dwarf-line-table-only (DWARF line-table fallback)\n' "${REQUIRE_MATRIX_PASS}" >&2
    exit 2
fi

mkdir -p "${OUT_ROOT}"
set +u
source /opt/intel/oneapi/setvars.sh --force >"${OUT_ROOT}/setvars.log" 2>&1
set -u
"${configure_cmd[@]}" >"${OUT_ROOT}/profiling-debug-build.log" 2>&1
"${build_cmd[@]}" >>"${OUT_ROOT}/profiling-debug-build.log" 2>&1
vtune_collect_cmd=(
    env ONEAPI_DEVICE_SELECTOR="${DEVICE_SELECTOR}" vtune -collect gpu-hotspots
)
if [[ -n "${VTUNE_TARGET_GPU}" ]]; then
    vtune_collect_cmd+=(-knob "target-gpu=${VTUNE_TARGET_GPU}")
fi
vtune_collect_cmd+=(
    -knob gpu-profiling-mode=source-analysis
    -knob source-analysis=mem-latency
    -knob dump-compute-task-binaries=true
)
if [[ -n "${TASK_GLOB}" ]]; then
    vtune_collect_cmd+=(-knob "computing-tasks-of-interest=${TASK_GLOB}#1#1#20")
fi
vtune_collect_cmd+=(
    -result-dir "${vtune_dir}"
    -- "${bench_cmd[@]}"
)
"${vtune_collect_cmd[@]}" >"${OUT_ROOT}/bench.stdout" 2>"${OUT_ROOT}/bench.stderr"
if ! vtune -report hotspots -r "${vtune_dir}" -group-by computing-task -format csv >"${OUT_ROOT}/vtune-computing-tasks.csv"; then
    printf 'warning: VTune computing-task report failed; task parser will fail closed if the CSV is empty\n' >&2
fi
if ! python3 scripts/parse-sycl-vtune-tasks.py "${OUT_ROOT}/vtune-computing-tasks.csv" --match "${TASK_MATCH}" >"${OUT_ROOT}/vtune-task.parse"; then
    printf 'warning: failed to parse VTune computing tasks for match %s\n' "${TASK_MATCH}" >&2
fi
first_zebin="$(find "${vtune_dir}" -name '*.zebin' -type f -print -quit)"
if [[ -z "${first_zebin}" ]]; then
    printf 'error: no .zebin found in %s\n' "${vtune_dir}" >&2
    exit 1
fi
llvm-readelf --sections --wide "${first_zebin}" >"${OUT_ROOT}/zebin-debug-sections.txt"
llvm-dwarfdump --debug-line "${first_zebin}" >"${OUT_ROOT}/zebin-debug-line.txt"
rm -f "${OUT_ROOT}/dwarf-source-lines.csv"
if ! python3 scripts/convert-sycl-zebin-line-table-to-source-csv.py \
    --input "${OUT_ROOT}/zebin-debug-line.txt" \
    --output "${OUT_ROOT}/dwarf-source-lines.csv" \
    --source-computing-task "${TARGET_KERNEL}"; then
    printf 'warning: DWARF source-line CSV conversion failed; checker will fail closed unless %s exists\n' "${OUT_ROOT}/dwarf-source-lines.csv" >&2
fi
rm -f "${OUT_ROOT}/asm-source-lines.csv" "${OUT_ROOT}/asm-source-lines.parse"
iga_dir="${OUT_ROOT}/iga-disasm"
rm -rf "${iga_dir}"
rm -f "${OUT_ROOT}/iga-pc-instructions.csv"
if python3 scripts/prepare-sycl-iga-disasm-inputs.py \
    --readelf-sections "${OUT_ROOT}/zebin-debug-sections.txt" \
    --zebin "${first_zebin}" \
    --kernel-match "${TARGET_KERNEL}" \
    --platform "${IGA_PLATFORM}" \
    --out-dir "${iga_dir}" >&2; then
    if (cd "${iga_dir}" && bash run-iga-disasm.sh >>iga.stdout 2>>iga.stderr) && \
       python3 scripts/parse-sycl-iga-pc-disasm.py --input "${iga_dir}/kernel.iga.json" --format json --kernel "${TARGET_KERNEL}" >"${OUT_ROOT}/iga-pc-instructions.csv"; then
        section_addr="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["extract.section_addr"])' "${iga_dir}/iga-disasm-manifest.json")"
        if ! python3 scripts/resolve-sycl-zebin-asm-source-lines.py \
            --dwarf-line-dump "${OUT_ROOT}/zebin-debug-line.txt" \
            --iga-instructions-csv "${OUT_ROOT}/iga-pc-instructions.csv" \
            --pc-base "${section_addr}" \
            --output "${OUT_ROOT}/asm-source-lines.csv" \
            --summary-output "${OUT_ROOT}/asm-source-lines.parse" \
            --source-computing-task "${TARGET_KERNEL}" \
            --require-source-path "mmvq.cpp"; then
            rm -f "${OUT_ROOT}/asm-source-lines.csv"
            printf 'warning: IGA PC resolver failed; checker will use ocloc/DWARF evidence if available\n' >&2
        fi
    else
        printf 'warning: IGA PC disassembly failed; checker will use ocloc/DWARF evidence if available\n' >&2
    fi
else
    printf 'warning: IGA input preparation failed; checker will use ocloc/DWARF evidence if available\n' >&2
fi
if [[ ! -f "${OUT_ROOT}/asm-source-lines.parse" ]] || ! grep -qx 'asm_source.status ok' "${OUT_ROOT}/asm-source-lines.parse"; then
    rm -f "${OUT_ROOT}/asm-source-lines.csv"
    asm_dir="${OUT_ROOT}/zebin-disasm"
    rm -rf "${asm_dir}"
    mkdir -p "${asm_dir}"
    cp "${first_zebin}" "${asm_dir}/kernel.zebin"
    if ! (cd "${asm_dir}" && ocloc disasm -file kernel.zebin >ocloc.stdout 2>ocloc.stderr); then
        printf 'warning: ocloc disasm failed; checker will use VTune/DWARF evidence if available\n' >&2
    fi
    first_asm="$(find_asm_for_task "${asm_dir}" "${TARGET_KERNEL}")"
    if [[ -n "${first_asm}" ]]; then
        if ! python3 scripts/resolve-sycl-zebin-asm-source-lines.py \
            --dwarf-line-dump "${OUT_ROOT}/zebin-debug-line.txt" \
            --asm "${first_asm}" \
            --output "${OUT_ROOT}/asm-source-lines.csv" \
            --summary-output "${OUT_ROOT}/asm-source-lines.parse" \
            --source-computing-task "${TARGET_KERNEL}" \
            --require-source-path "mmvq.cpp"; then
            printf 'warning: ASM source-line resolver failed; checker will use VTune/DWARF evidence if available\n' >&2
        fi
    else
        printf 'warning: no .asm file found after ocloc disasm\n' >&2
    fi
fi
if ! vtune -report hotspots -r "${vtune_dir}" -group-by gpu-source-line -format csv >"${OUT_ROOT}/vtune-gpu-source-line.csv"; then
    printf 'warning: VTune gpu-source-line report failed; checker will use explicit blockers and DWARF fallback if available\n' >&2
fi
python3 scripts/check-sycl-vtune-source-lines.py \
    --readelf-sections "${OUT_ROOT}/zebin-debug-sections.txt" \
    --vtune-csv "${OUT_ROOT}/vtune-gpu-source-line.csv" \
    --require-kernel "${TARGET_KERNEL}" \
    --asm-source-lines-csv "${OUT_ROOT}/asm-source-lines.csv" \
    --allow-asm-line-static-cost \
    --dwarf-line-dump "${OUT_ROOT}/zebin-debug-line.txt" \
    --dwarf-source-lines-csv "${OUT_ROOT}/dwarf-source-lines.csv" \
    --allow-dwarf-line-table-only \
    --require-source-path "mmvq.cpp" \
    --vtune-stdout "${OUT_ROOT}/bench.stdout" \
    --vtune-stderr "${OUT_ROOT}/bench.stderr" >"${OUT_ROOT}/source-line-feasibility.parse"
printf 'Artifacts: %s\n' "${OUT_ROOT}"
