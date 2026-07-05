#!/usr/bin/env bash
set -euo pipefail

EXECUTE=0
ACK=0
OUT_ROOT="pc-sampling-capability"
DEVICE_SELECTOR=""

usage() {
    printf 'usage: %s [--dry-run|--execute] [--i-understand-this-probes-intel-gpu-pc-sampling] [--out-root DIR] [--device-selector SELECTOR]\n' "$0"
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
        --i-understand-this-probes-intel-gpu-pc-sampling) ACK=1 ;;
        --out-root) require_value "$1" "${2-}"; OUT_ROOT="$2"; shift ;;
        --device-selector) require_value "$1" "${2-}"; DEVICE_SELECTOR="$2"; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if [[ "${EXECUTE}" -eq 1 && "${ACK}" -ne 1 ]]; then
    printf 'error: --execute requires --i-understand-this-probes-intel-gpu-pc-sampling\n' >&2
    exit 2
fi

print_probe_commands() {
    cat <<'EOF'
vtune -help collect gpu-hotspots >"${OUT_ROOT}/vtune-gpu-hotspots-help.txt" 2>&1 || true
vtune -help report gpu-source-line >"${OUT_ROOT}/vtune-gpu-source-line-help.txt" 2>&1 || true
command -v gtpin64 || command -v gtpin || true
if command -v gtpin64 >/dev/null 2>&1; then gtpin64 --help >"${OUT_ROOT}/gtpin-help.txt" 2>&1 || true; elif command -v gtpin >/dev/null 2>&1; then gtpin --help >"${OUT_ROOT}/gtpin-help.txt" 2>&1 || true; fi
find /opt/intel/oneapi -iname '*pti*' -o -iname 'libpti*' >"${OUT_ROOT}/pti-files.txt" 2>&1 || true
python3 - <<'PY' >"${OUT_ROOT}/level-zero-metric-groups.txt" 2>&1 || true
import ctypes
from ctypes import byref, c_uint32, c_void_p, POINTER
ze = ctypes.CDLL('libze_loader.so.1')
ZE_RESULT_SUCCESS = 0
if ze.zeInit(0) != ZE_RESULT_SUCCESS:
    raise SystemExit('zeInit failed')
ze.zeDriverGet.argtypes = [POINTER(c_uint32), POINTER(c_void_p)]
ze.zeDeviceGet.argtypes = [c_void_p, POINTER(c_uint32), POINTER(c_void_p)]
try:
    zetMetricGroupGet = ze.zetMetricGroupGet
except AttributeError:
    print('zetMetricGroupGet symbol missing')
    raise SystemExit(0)
zetMetricGroupGet.argtypes = [c_void_p, POINTER(c_uint32), POINTER(c_void_p)]
driver_count = c_uint32(0)
ze.zeDriverGet(byref(driver_count), None)
drivers = (c_void_p * driver_count.value)()
ze.zeDriverGet(byref(driver_count), drivers)
print('driver_count', driver_count.value)
for driver_index, driver in enumerate(drivers):
    device_count = c_uint32(0)
    ze.zeDeviceGet(driver, byref(device_count), None)
    devices = (c_void_p * device_count.value)()
    ze.zeDeviceGet(driver, byref(device_count), devices)
    print('driver', driver_index, 'device_count', device_count.value)
    for device_index, device in enumerate(devices):
        group_count = c_uint32(0)
        result = zetMetricGroupGet(device, byref(group_count), None)
        print('driver', driver_index, 'device', device_index, 'zetMetricGroupGet_count_result', result, 'metric_group_count', group_count.value)
        if result == ZE_RESULT_SUCCESS and group_count.value:
            groups = (c_void_p * group_count.value)()
            result = zetMetricGroupGet(device, byref(group_count), groups)
            print('driver', driver_index, 'device', device_index, 'zetMetricGroupGet_handles_result', result, 'metric_group_count', group_count.value)
# Metric groups are counters only; this probe never treats them as PC samples.
PY
EOF
}

print_dry_run() {
    printf 'DRY RUN: Intel GPU PC-sampling capability probe\n'
    printf 'Workers must not execute these lead-only probes. Use --execute plus --i-understand-this-probes-intel-gpu-pc-sampling only under lead ownership.\n'
    printf 'OUT_ROOT=%q\n' "${OUT_ROOT}"
    if [[ -n "${DEVICE_SELECTOR}" ]]; then
        printf 'ONEAPI_DEVICE_SELECTOR=%q\n' "${DEVICE_SELECTOR}"
    fi
    printf '\nLead-only commands that would run:\n'
    print_probe_commands
    printf '\nStatus rules:\n'
    printf 'pc_sampling.status available only when %s/pc-samples.csv exists with header kernel,pc,sample_count and positive sample_count rows.\n' "${OUT_ROOT}"
    printf 'pc_sampling.status metrics_only when VTune/PTI/level zero metric counters are found but no real PC sample CSV exists.\n'
    printf 'pc_sampling.status unavailable when no usable PC-sampling or metric evidence is found.\n'
    printf 'Level Zero metric groups are metric counters, not PC samples; the script never synthesizes pc-samples.csv.\n'
}

has_positive_pc_samples() {
    local csv_path="$1"
    python3 - "$csv_path" <<'PY'
import csv
import sys
path = sys.argv[1]
try:
    with open(path, newline='', encoding='utf-8', errors='replace') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if fields[:3] != ['kernel', 'pc', 'sample_count']:
            raise SystemExit(1)
        for row in reader:
            try:
                if int(row.get('sample_count') or '0') > 0:
                    raise SystemExit(0)
            except ValueError:
                pass
except OSError:
    raise SystemExit(1)
raise SystemExit(1)
PY
}

file_has_content() {
    local path="$1"
    [[ -s "${path}" ]]
}

pti_files_found() {
    local path="${OUT_ROOT}/pti-files.txt"
    file_has_content "${path}" && grep -Eiq '(^|/)lib?pti|pti' "${path}" && ! grep -Eq '^find: ' "${path}"
}

level_zero_metric_groups_found() {
    local path="${OUT_ROOT}/level-zero-metric-groups.txt"
    file_has_content "${path}" && grep -Eq 'metric_group_count [1-9][0-9]*' "${path}"
}

vtune_gpu_hotspots_supported() {
    local path="${OUT_ROOT}/vtune-gpu-hotspots-help.txt"
    file_has_content "${path}" && grep -Eiq 'gpu-hotspots|GPU Hotspots|hotspots' "${path}" && ! grep -Eiq 'command not found|not recognized' "${path}"
}

write_parse() {
    local parse_path="${OUT_ROOT}/pc-sampling-capability.parse"
    local status="unavailable"
    local -a blockers=()
    local metrics_evidence=0
    local pc_samples="${OUT_ROOT}/pc-samples.csv"

    if has_positive_pc_samples "${pc_samples}"; then
        status="available"
    else
        blockers+=("no_public_pc_sample_api_confirmed")
        blockers+=("vtune_source_rows_empty")

        if ! file_has_content "${OUT_ROOT}/gtpin-path.txt"; then
            blockers+=("gtpin_not_found")
        elif grep -Eiq 'unsupported|not supported' "${OUT_ROOT}/gtpin-help.txt" 2>/dev/null; then
            blockers+=("gtpin_unsupported")
        fi
        if grep -Eiq 'register[ -]?pressure|spill' "${OUT_ROOT}/gtpin-help.txt" 2>/dev/null; then
            blockers+=("gtpin_register_pressure_failure")
        fi

        if pti_files_found; then
            metrics_evidence=1
            blockers+=("pti_files_found_but_no_pc_sample_producer")
        fi
        if level_zero_metric_groups_found; then
            metrics_evidence=1
            blockers+=("level_zero_metrics_are_not_pc_samples")
        fi
        if vtune_gpu_hotspots_supported; then
            metrics_evidence=1
        fi
        if [[ "${metrics_evidence}" -eq 1 ]]; then
            status="metrics_only"
        fi
    fi

    {
        printf 'pc_sampling.status %s\n' "${status}"
        if [[ -n "${DEVICE_SELECTOR}" ]]; then
            printf 'pc_sampling.device_selector %s\n' "${DEVICE_SELECTOR}"
        fi
        printf 'pc_sampling.pc_samples_csv %s\n' "${pc_samples}"
        printf 'pc_sampling.note Level Zero metric groups are counters only, not PC samples.\n'
        local blocker
        local seen=""
        for blocker in "${blockers[@]}"; do
            case " ${seen} " in
                *" ${blocker} "*) ;;
                *) printf 'pc_sampling.blocker %s\n' "${blocker}"; seen+=" ${blocker}" ;;
            esac
        done
    } >"${parse_path}"
    cat "${parse_path}"
}

run_execute() {
    mkdir -p "${OUT_ROOT}"
    if [[ -n "${DEVICE_SELECTOR}" ]]; then
        export ONEAPI_DEVICE_SELECTOR="${DEVICE_SELECTOR}"
    fi

    vtune -help collect gpu-hotspots >"${OUT_ROOT}/vtune-gpu-hotspots-help.txt" 2>&1 || true
    vtune -help report gpu-source-line >"${OUT_ROOT}/vtune-gpu-source-line-help.txt" 2>&1 || true
    (command -v gtpin64 || command -v gtpin || true) >"${OUT_ROOT}/gtpin-path.txt"
    if command -v gtpin64 >/dev/null 2>&1; then gtpin64 --help >"${OUT_ROOT}/gtpin-help.txt" 2>&1 || true; elif command -v gtpin >/dev/null 2>&1; then gtpin --help >"${OUT_ROOT}/gtpin-help.txt" 2>&1 || true; fi
    find /opt/intel/oneapi -iname '*pti*' -o -iname 'libpti*' >"${OUT_ROOT}/pti-files.txt" 2>&1 || true
    python3 - <<'PY' >"${OUT_ROOT}/level-zero-metric-groups.txt" 2>&1 || true
import ctypes
from ctypes import byref, c_uint32, c_void_p, POINTER
ze = ctypes.CDLL('libze_loader.so.1')
ZE_RESULT_SUCCESS = 0
if ze.zeInit(0) != ZE_RESULT_SUCCESS:
    raise SystemExit('zeInit failed')
ze.zeDriverGet.argtypes = [POINTER(c_uint32), POINTER(c_void_p)]
ze.zeDeviceGet.argtypes = [c_void_p, POINTER(c_uint32), POINTER(c_void_p)]
try:
    zetMetricGroupGet = ze.zetMetricGroupGet
except AttributeError:
    print('zetMetricGroupGet symbol missing')
    raise SystemExit(0)
zetMetricGroupGet.argtypes = [c_void_p, POINTER(c_uint32), POINTER(c_void_p)]
driver_count = c_uint32(0)
ze.zeDriverGet(byref(driver_count), None)
drivers = (c_void_p * driver_count.value)()
ze.zeDriverGet(byref(driver_count), drivers)
print('driver_count', driver_count.value)
for driver_index, driver in enumerate(drivers):
    device_count = c_uint32(0)
    ze.zeDeviceGet(driver, byref(device_count), None)
    devices = (c_void_p * device_count.value)()
    ze.zeDeviceGet(driver, byref(device_count), devices)
    print('driver', driver_index, 'device_count', device_count.value)
    for device_index, device in enumerate(devices):
        group_count = c_uint32(0)
        result = zetMetricGroupGet(device, byref(group_count), None)
        print('driver', driver_index, 'device', device_index, 'zetMetricGroupGet_count_result', result, 'metric_group_count', group_count.value)
        if result == ZE_RESULT_SUCCESS and group_count.value:
            groups = (c_void_p * group_count.value)()
            result = zetMetricGroupGet(device, byref(group_count), groups)
            print('driver', driver_index, 'device', device_index, 'zetMetricGroupGet_handles_result', result, 'metric_group_count', group_count.value)
# Metric groups are counters only; this probe never treats them as PC samples.
PY
    write_parse
}

if [[ "${EXECUTE}" -eq 0 ]]; then
    print_dry_run
else
    run_execute
fi
