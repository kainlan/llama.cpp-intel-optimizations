#!/usr/bin/env bash
set -euo pipefail

EXECUTE=0
ACK=0
OUT_ROOT="source-line"
DEFAULT_MATRIX_ROOT="source-line/build-matrix"
DEVICE_SELECTOR="level_zero:1"
VTUNE_TARGET_GPU=""
TARGET_KERNEL="sycl_source_line_probe"
TASK_GLOB=""
TASK_MATCH="sycl_source_line_probe"
IGA_PLATFORM="${SYCL_IGA_PLATFORM:-xe2}"

CASE_NAMES=(release_split debug_line_tables debug_full debug_no_inline)
CASE_BUILD_TYPES=(Release Release RelWithDebInfo RelWithDebInfo)
CASE_CXX_FLAGS=(
    "-fsycl-device-code-split=per_kernel"
    "-gline-tables-only -fdebug-info-for-profiling -fsycl-instrument-device-code -fsycl-device-code-split=per_kernel"
    "-g -fdebug-info-for-profiling -fsycl-instrument-device-code -fsycl-device-code-split=per_kernel"
    "-g -fdebug-info-for-profiling -fsycl-instrument-device-code -fno-inline -fno-inline-functions -fsycl-device-code-split=per_kernel"
)
CASE_LINK_FLAGS=(
    "-fsycl-device-code-split=per_kernel"
    "-fsycl-instrument-device-code -fsycl-device-code-split=per_kernel"
    "-fsycl-instrument-device-code -fsycl-device-code-split=per_kernel"
    "-fsycl-instrument-device-code -fsycl-device-code-split=per_kernel"
)

usage() {
    printf 'usage: %s [--dry-run|--execute] [--i-understand-this-runs-gpu-source-probe] [--out-root DIR] [--device-selector SELECTOR] [--vtune-target-gpu VALUE] [--task-glob GLOB] [--task-match TEXT] [--iga-platform PLATFORM]\n' "$0"
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
        --i-understand-this-runs-gpu-source-probe) ACK=1 ;;
        --out-root) require_value "$1" "${2-}"; OUT_ROOT="$2"; shift ;;
        --device-selector) require_value "$1" "${2-}"; DEVICE_SELECTOR="$2"; shift ;;
        --vtune-target-gpu) require_value "$1" "${2-}"; VTUNE_TARGET_GPU="$2"; shift ;;
        --task-glob) require_value "$1" "${2-}"; TASK_GLOB="$2"; shift ;;
        --task-match) require_value "$1" "${2-}"; TASK_MATCH="$2"; shift ;;
        --iga-platform) require_value "$1" "${2-}"; IGA_PLATFORM="$2"; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if [[ "${EXECUTE}" -eq 1 && "${ACK}" -ne 1 ]]; then
    printf 'error: --execute requires --i-understand-this-runs-gpu-source-probe\n' >&2
    exit 2
fi

quote_cmd() {
    printf '%q ' "$@"
}

case_dir() {
    printf '%s/build-matrix/%s' "${OUT_ROOT}" "$1"
}

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

make_configure_cmd() {
    local name="$1"
    local build_type="$2"
    local cxx_flags="$3"
    local link_flags="$4"
    local dir
    dir="$(case_dir "${name}")"
    CONFIGURE_CMD=(
        cmake -S . -B "${dir}/build" -G Ninja
        "-DCMAKE_BUILD_TYPE=${build_type}"
        -DGGML_SYCL=ON
        -DGGML_SYCL_TARGET=INTEL
        -DGGML_SYCL_F16=ON
        -DGGML_SYCL_PROFILING_DEBUG=OFF
        -DCMAKE_C_COMPILER=icx
        -DCMAKE_CXX_COMPILER=icpx
        "-DCMAKE_CXX_FLAGS=${cxx_flags}"
        "-DCMAKE_EXE_LINKER_FLAGS=${link_flags}"
    )
}

make_build_cmd() {
    local name="$1"
    local dir
    dir="$(case_dir "${name}")"
    BUILD_CMD=(cmake --build "${dir}/build" --config "$2" --target sycl-source-line-probe -j "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}")
}

make_probe_cmd() {
    local name="$1"
    local dir
    dir="$(case_dir "${name}")"
    PROBE_CMD=("${dir}/build/bin/sycl-source-line-probe" --iterations 100 --size 1048576 --json "${dir}/probe.json")
}

print_vtune_collect_prefix() {
    local vtune_dir="$1"
    printf 'env ONEAPI_DEVICE_SELECTOR=%q vtune -collect gpu-hotspots' "${DEVICE_SELECTOR}"
    if [[ -n "${VTUNE_TARGET_GPU}" ]]; then
        printf ' -knob %q' "target-gpu=${VTUNE_TARGET_GPU}"
    fi
    printf ' -knob gpu-profiling-mode=source-analysis -knob source-analysis=mem-latency -knob dump-compute-task-binaries=true'
    if [[ -n "${TASK_GLOB}" ]]; then
        local task_knob
        task_knob="computing-tasks-of-interest=${TASK_GLOB}#1#1#20"
        printf ' -knob %q' "${task_knob}"
    fi
    printf ' -result-dir %q -- ' "${vtune_dir}"
}

print_plan() {
    printf 'DRY RUN: source-line debug-info matrix; no commands are executed.\n'
    printf '# output root: %s\n' "${OUT_ROOT}"
    printf '# default matrix root: %s\n' "${DEFAULT_MATRIX_ROOT}"
    printf '# IGA platform: %s (override with --iga-platform or SYCL_IGA_PLATFORM)\n' "${IGA_PLATFORM}"

    for index in "${!CASE_NAMES[@]}"; do
        local name="${CASE_NAMES[$index]}"
        local build_type="${CASE_BUILD_TYPES[$index]}"
        local cxx_flags="${CASE_CXX_FLAGS[$index]}"
        local link_flags="${CASE_LINK_FLAGS[$index]}"
        local dir
        local vtune_dir
        dir="$(case_dir "${name}")"
        vtune_dir="${dir}/vtune-source-line"

        make_configure_cmd "${name}" "${build_type}" "${cxx_flags}" "${link_flags}"
        make_build_cmd "${name}" "${build_type}"
        make_probe_cmd "${name}"

        printf '\n# matrix row: %s\n' "${name}"
        printf '# build_type=%s\n' "${build_type}"
        printf '# cxx_flags=%s\n' "${cxx_flags}"
        quote_cmd "${CONFIGURE_CMD[@]}"; printf '> %q 2>&1\n' "${dir}/configure.log"
        quote_cmd "${BUILD_CMD[@]}"; printf '> %q 2>&1\n' "${dir}/build.log"
        print_vtune_collect_prefix "${vtune_dir}"
        quote_cmd "${PROBE_CMD[@]}"; printf '> %q 2> %q\n' "${dir}/probe.stdout" "${dir}/probe.stderr"
        printf 'first_zebin="$(find %q -name '\''*.zebin'\'' -type f -print -quit)"\n' "${vtune_dir}"
        printf 'llvm-readelf --sections --wide "${first_zebin}" > %q\n' "${dir}/zebin-debug-sections.txt"
        printf 'vtune -report hotspots -r %q -group-by computing-task -format csv > %q\n' "${vtune_dir}" "${dir}/vtune-computing-tasks.csv"
        printf 'python3 scripts/parse-sycl-vtune-tasks.py %q --match %q > %q\n' "${dir}/vtune-computing-tasks.csv" "${TASK_MATCH}" "${dir}/vtune-task.parse"
        printf 'llvm-dwarfdump --debug-line %q > %q\n' "${vtune_dir}/data.0/<first-zebin>" "${dir}/zebin-debug-line.txt"
        printf 'python3 scripts/convert-sycl-zebin-line-table-to-source-csv.py --input %q --output %q --source-computing-task %q\n' "${dir}/zebin-debug-line.txt" "${dir}/dwarf-source-lines.csv" "${TARGET_KERNEL}"
        printf 'python3 scripts/prepare-sycl-iga-disasm-inputs.py --readelf-sections %q --zebin "${first_zebin}" --kernel-match %q --platform %q --out-dir %q || true\n' "${dir}/zebin-debug-sections.txt" "${TARGET_KERNEL}" "${IGA_PLATFORM}" "${dir}/iga-disasm"
        printf '(cd %q && bash run-iga-disasm.sh) || true  # emits kernel.iga.json using iga64 -Xprint-json -Xprint-pc\n' "${dir}/iga-disasm"
        printf 'python3 scripts/parse-sycl-iga-pc-disasm.py --input %q --format json --kernel %q > %q || true\n' "${dir}/iga-disasm/kernel.iga.json" "${TARGET_KERNEL}" "${dir}/iga-pc-instructions.csv"
        printf 'section_addr="$(python3 -c %q %q)"\n' 'import json,sys; print(json.load(open(sys.argv[1]))["extract.section_addr"])' "${dir}/iga-disasm/iga-disasm-manifest.json"
        printf 'python3 scripts/resolve-sycl-zebin-asm-source-lines.py --dwarf-line-dump %q --iga-instructions-csv %q --pc-base "${section_addr}" --output %q --summary-output %q --source-computing-task %q --require-source-path %q\n' "${dir}/zebin-debug-line.txt" "${dir}/iga-pc-instructions.csv" "${dir}/asm-source-lines.csv" "${dir}/asm-source-lines.parse" "${TARGET_KERNEL}" "main.cpp"
        printf 'mkdir -p %q && cp %q %q && (cd %q && ocloc disasm -file kernel.zebin > ocloc.stdout 2> ocloc.stderr || true)\n' "${dir}/zebin-disasm" "${vtune_dir}/data.0/example.zebin" "${dir}/zebin-disasm/kernel.zebin" "${dir}/zebin-disasm"
        printf 'first_asm="$(find %q -type f -name '\''*%s*.asm'\'' -print -quit)"\n' "${dir}/zebin-disasm" "${TARGET_KERNEL}"
        printf 'if [[ -z "${first_asm}" ]]; then first_asm="$(find %q -type f -name '\''*.asm'\'' -print -quit)"; fi  # resolver rejects unmarked or mismatched ASM\n' "${dir}/zebin-disasm"
        printf 'python3 scripts/resolve-sycl-zebin-asm-source-lines.py --dwarf-line-dump %q --asm "${first_asm}" --output %q --summary-output %q --source-computing-task %q --require-source-path %q\n' "${dir}/zebin-debug-line.txt" "${dir}/asm-source-lines.csv" "${dir}/asm-source-lines.parse" "${TARGET_KERNEL}" "main.cpp"
        printf 'vtune -report hotspots -r %q -group-by gpu-source-line -format csv > %q\n' "${vtune_dir}" "${dir}/vtune-gpu-source-line.csv"
        printf 'python3 scripts/check-sycl-vtune-source-lines.py --readelf-sections %q --vtune-csv %q --require-kernel %q --asm-source-lines-csv %q --allow-asm-line-static-cost --dwarf-line-dump %q --dwarf-source-lines-csv %q --allow-dwarf-line-table-only --require-source-path %q --vtune-stdout %q --vtune-stderr %q > %q\n' "${dir}/zebin-debug-sections.txt" "${dir}/vtune-gpu-source-line.csv" "${TARGET_KERNEL}" "${dir}/asm-source-lines.csv" "${dir}/zebin-debug-line.txt" "${dir}/dwarf-source-lines.csv" "main.cpp" "${dir}/probe.stdout" "${dir}/probe.stderr" "${dir}/source-line-feasibility.parse"
    done
}

if [[ "${EXECUTE}" -ne 1 ]]; then
    print_plan
    exit 0
fi

mkdir -p "${OUT_ROOT}/build-matrix"
set +u
source /opt/intel/oneapi/setvars.sh --force >"${OUT_ROOT}/setvars.log" 2>&1
set -u

for index in "${!CASE_NAMES[@]}"; do
    name="${CASE_NAMES[$index]}"
    build_type="${CASE_BUILD_TYPES[$index]}"
    cxx_flags="${CASE_CXX_FLAGS[$index]}"
    link_flags="${CASE_LINK_FLAGS[$index]}"
    dir="$(case_dir "${name}")"
    vtune_dir="${dir}/vtune-source-line"
    mkdir -p "${dir}"

    make_configure_cmd "${name}" "${build_type}" "${cxx_flags}" "${link_flags}"
    make_build_cmd "${name}" "${build_type}"
    make_probe_cmd "${name}"

    "${CONFIGURE_CMD[@]}" >"${dir}/configure.log" 2>&1
    "${BUILD_CMD[@]}" >"${dir}/build.log" 2>&1
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
    vtune_collect_cmd+=(-result-dir "${vtune_dir}" -- "${PROBE_CMD[@]}")
    "${vtune_collect_cmd[@]}" >"${dir}/probe.stdout" 2>"${dir}/probe.stderr"

    if ! vtune -report hotspots -r "${vtune_dir}" -group-by computing-task -format csv >"${dir}/vtune-computing-tasks.csv"; then
        printf 'warning: VTune computing-task report failed for matrix row %s\n' "${name}" >>"${dir}/probe.stderr"
    fi
    if ! python3 scripts/parse-sycl-vtune-tasks.py "${dir}/vtune-computing-tasks.csv" --match "${TASK_MATCH}" >"${dir}/vtune-task.parse"; then
        printf 'warning: VTune task selection failed for matrix row %s; see %s\n' "${name}" "${dir}/vtune-task.parse" >&2
    fi

    first_zebin="$(find "${vtune_dir}" -name '*.zebin' -type f -print -quit)"
    if [[ -z "${first_zebin}" ]]; then
        printf 'error: no .zebin found for matrix row %s in %s\n' "${name}" "${vtune_dir}" >&2
        exit 1
    fi

    llvm-readelf --sections --wide "${first_zebin}" >"${dir}/zebin-debug-sections.txt"
    llvm-dwarfdump --debug-line "${first_zebin}" >"${dir}/zebin-debug-line.txt"
    rm -f "${dir}/dwarf-source-lines.csv"
    if ! python3 scripts/convert-sycl-zebin-line-table-to-source-csv.py \
        --input "${dir}/zebin-debug-line.txt" \
        --output "${dir}/dwarf-source-lines.csv" \
        --source-computing-task "${TARGET_KERNEL}"; then
        printf 'warning: DWARF source-line CSV conversion failed for matrix row %s; checker will fail closed unless %s exists\n' "${name}" "${dir}/dwarf-source-lines.csv" >>"${dir}/probe.stderr"
    fi
    rm -f "${dir}/asm-source-lines.csv" "${dir}/asm-source-lines.parse"
    iga_dir="${dir}/iga-disasm"
    rm -rf "${iga_dir}"
    rm -f "${dir}/iga-pc-instructions.csv"
    if python3 scripts/prepare-sycl-iga-disasm-inputs.py \
        --readelf-sections "${dir}/zebin-debug-sections.txt" \
        --zebin "${first_zebin}" \
        --kernel-match "${TARGET_KERNEL}" \
        --platform "${IGA_PLATFORM}" \
        --out-dir "${iga_dir}" >>"${dir}/probe.stderr" 2>&1; then
        if (cd "${iga_dir}" && bash run-iga-disasm.sh >>iga.stdout 2>>iga.stderr) && \
           python3 scripts/parse-sycl-iga-pc-disasm.py --input "${iga_dir}/kernel.iga.json" --format json --kernel "${TARGET_KERNEL}" >"${dir}/iga-pc-instructions.csv"; then
            section_addr="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["extract.section_addr"])' "${iga_dir}/iga-disasm-manifest.json")"
            if ! python3 scripts/resolve-sycl-zebin-asm-source-lines.py \
                --dwarf-line-dump "${dir}/zebin-debug-line.txt" \
                --iga-instructions-csv "${dir}/iga-pc-instructions.csv" \
                --pc-base "${section_addr}" \
                --output "${dir}/asm-source-lines.csv" \
                --summary-output "${dir}/asm-source-lines.parse" \
                --source-computing-task "${TARGET_KERNEL}" \
                --require-source-path "main.cpp"; then
                rm -f "${dir}/asm-source-lines.csv"
                printf 'warning: IGA PC resolver failed for matrix row %s; checker will use ocloc/DWARF evidence if available\n' "${name}" >>"${dir}/probe.stderr"
            fi
        else
            printf 'warning: IGA PC disassembly failed for matrix row %s; checker will use ocloc/DWARF evidence if available\n' "${name}" >>"${dir}/probe.stderr"
        fi
    else
        printf 'warning: IGA input preparation failed for matrix row %s; checker will use ocloc/DWARF evidence if available\n' "${name}" >>"${dir}/probe.stderr"
    fi
    if [[ ! -f "${dir}/asm-source-lines.parse" ]] || ! grep -qx 'asm_source.status ok' "${dir}/asm-source-lines.parse"; then
        rm -f "${dir}/asm-source-lines.csv"
        asm_dir="${dir}/zebin-disasm"
        rm -rf "${asm_dir}"
        mkdir -p "${asm_dir}"
        cp "${first_zebin}" "${asm_dir}/kernel.zebin"
        if ! (cd "${asm_dir}" && ocloc disasm -file kernel.zebin >ocloc.stdout 2>ocloc.stderr); then
            printf 'warning: ocloc disasm failed for matrix row %s; checker will use VTune/DWARF evidence if available\n' "${name}" >>"${dir}/probe.stderr"
        fi
        first_asm="$(find_asm_for_task "${asm_dir}" "${TARGET_KERNEL}")"
        if [[ -n "${first_asm}" ]]; then
            if ! python3 scripts/resolve-sycl-zebin-asm-source-lines.py \
                --dwarf-line-dump "${dir}/zebin-debug-line.txt" \
                --asm "${first_asm}" \
                --output "${dir}/asm-source-lines.csv" \
                --summary-output "${dir}/asm-source-lines.parse" \
                --source-computing-task "${TARGET_KERNEL}" \
                --require-source-path "main.cpp"; then
                printf 'warning: ASM source-line resolver failed for matrix row %s; checker will use VTune/DWARF evidence if available\n' "${name}" >>"${dir}/probe.stderr"
            fi
        else
            printf 'warning: no .asm file found after ocloc disasm for matrix row %s\n' "${name}" >>"${dir}/probe.stderr"
        fi
    fi
    if ! vtune -report hotspots -r "${vtune_dir}" -group-by gpu-source-line -format csv >"${dir}/vtune-gpu-source-line.csv"; then
        printf 'warning: VTune gpu-source-line report failed for matrix row %s; writing fail-closed checker output\n' "${name}" >>"${dir}/probe.stderr"
    fi
    checker_args=(
        --readelf-sections "${dir}/zebin-debug-sections.txt"
        --vtune-csv "${dir}/vtune-gpu-source-line.csv"
        --require-kernel "${TARGET_KERNEL}"
        --dwarf-line-dump "${dir}/zebin-debug-line.txt"
        --dwarf-source-lines-csv "${dir}/dwarf-source-lines.csv"
        --allow-dwarf-line-table-only
        --require-source-path "main.cpp"
        --vtune-stdout "${dir}/probe.stdout"
        --vtune-stderr "${dir}/probe.stderr"
    )
    if [[ -f "${dir}/asm-source-lines.parse" ]] && grep -qx 'asm_source.status ok' "${dir}/asm-source-lines.parse"; then
        checker_args+=(--asm-source-lines-csv "${dir}/asm-source-lines.csv" --allow-asm-line-static-cost)
    fi
    if ! python3 scripts/check-sycl-vtune-source-lines.py "${checker_args[@]}" >"${dir}/source-line-feasibility.parse"; then
        printf 'warning: source-line checker reported failure for matrix row %s; see %s\n' "${name}" "${dir}/source-line-feasibility.parse" >&2
    fi
done

printf 'Artifacts: %s/build-matrix\n' "${OUT_ROOT}"
