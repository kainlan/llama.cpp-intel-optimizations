#!/usr/bin/env bash
# Passive guard for Intel Level Zero workloads on this workstation.
# Source this file and call sycl_gpu_preflight_check "$selector" before each
# GPU run. It intentionally does not enumerate devices through oneAPI.

sycl_preflight_repo_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

sycl_preflight_selector_uses_level_zero() {
    local selector="${1:-}"
    [[ -z "$selector" || "$selector" == *level_zero* ]]
}

sycl_preflight_selector_may_use_b50() {
    local selector="${1:-}"
    [[ -z "$selector" ||
       "$selector" == *level_zero:1* ||
       "$selector" == *level_zero:0,1* ||
       "$selector" == *level_zero:1,0* ||
       "$selector" == *level_zero:gpu* ||
       "$selector" == *level_zero:*gpu* ]]
}

sycl_preflight_journal_has_current_boot_gpu_faults() {
    command -v journalctl >/dev/null 2>&1 || return 1
    journalctl -k -b --no-pager 2>/dev/null |
        grep -Eiq 'xe .*Engine reset|xe .*Timedout job|xe .*Kernel-submitted job timed out|Xe device coredump|guc_exec_queue_timedout_job|drm_sched_job_timedout|soft lockup|RCU.*stall|BUG:|Oops|ttm_resource_manager_usage|xe_drm_ioctl|xe_pt_zap_ptes'
}

sycl_preflight_journal_has_previous_boot_gpu_faults() {
    command -v journalctl >/dev/null 2>&1 || return 1
    journalctl -k -b -1 --no-pager 2>/dev/null |
        grep -Eiq 'xe .*Engine reset|xe .*Timedout job|xe .*Kernel-submitted job timed out|Xe device coredump|guc_exec_queue_timedout_job|drm_sched_job_timedout|soft lockup|RCU.*stall|BUG:|Oops|ttm_resource_manager_usage|xe_drm_ioctl|xe_pt_zap_ptes'
}

sycl_preflight_b50_sysfs_bad() {
    local b50="/sys/bus/pci/devices/0000:07:00.0"
    [[ -e "$b50" ]] || return 0

    local enable power_state runtime_status
    enable="$(cat "$b50/enable" 2>/dev/null || true)"
    power_state="$(cat "$b50/power_state" 2>/dev/null || true)"
    runtime_status="$(cat "$b50/power/runtime_status" 2>/dev/null || true)"

    # Runtime PM may park an idle B50 in D3hot with enable=0 after a clean boot.
    # Opening the Level Zero device should wake it; do not treat idle suspend as
    # a fault in the absence of current-boot xe/TTM errors.
    if [[ "$runtime_status" == "suspended" ]]; then
        return 1
    fi

    [[ "$enable" != "1" || "$power_state" != "D0" ]]
}

sycl_preflight_external_gpu_consumers() {
    pgrep -a 'llama-bench|llama-cli|sycl-ls|clinfo|vainfo|intel_gpu_top' 2>/dev/null |
        grep -Ev "^[[:space:]]*$$[[:space:]]" || true
}

sycl_preflight_uninterruptible_gpu_work() {
    ps -eo pid=,state=,wchan=,comm=,args= 2>/dev/null |
        awk '
            $2 == "D" &&
            $4 ~ /^(llama-bench|llama-cli|sycl-ls|clinfo|vainfo|intel_gpu_top)$/ &&
            $3 ~ /(drm|xe|ttm|dma_fence)/ {
                print
            }
        '
}

sycl_gpu_preflight_check() {
    local selector="${1:-${ONEAPI_DEVICE_SELECTOR:-level_zero:0}}"

    if ! sycl_preflight_selector_uses_level_zero "$selector"; then
        return 0
    fi

    if sycl_preflight_journal_has_current_boot_gpu_faults; then
        cat >&2 <<EOF
[sycl-preflight] refusing Level Zero selector '$selector': current boot has xe/TTM reset, timeout, oops, or lockup evidence.
[sycl-preflight] reboot before running more Level Zero workloads; sysfs or sycl-ls success is not a clean bill of health after this class of fault.
EOF
        return 4
    fi

    if [[ "${SYCL_PREFLIGHT_WARN_LEVEL_ZERO_PREVIOUS_BOOT:-0}" == "1" ]] &&
       sycl_preflight_journal_has_previous_boot_gpu_faults; then
        cat >&2 <<EOF
[sycl-preflight] warning for selector '$selector': previous boot contains xe/TTM fault evidence.
[sycl-preflight] This is postmortem context only; fresh-boot Level Zero runs are blocked by current-boot evidence, not previous-boot evidence.
EOF
    fi

    if sycl_preflight_selector_may_use_b50 "$selector" && sycl_preflight_b50_sysfs_bad; then
        cat >&2 <<EOF
[sycl-preflight] refusing B50 selector '$selector': B50 sysfs state is not enabled/D0/active.
EOF
        return 4
    fi

    local wedged_work
    wedged_work="$(sycl_preflight_uninterruptible_gpu_work)"
    if [[ -n "$wedged_work" ]]; then
        cat >&2 <<EOF
[sycl-preflight] refusing selector '$selector': GPU workload is stuck in uninterruptible kernel wait.
$wedged_work
[sycl-preflight] reboot before running more Level Zero workloads.
EOF
        return 4
    fi

    local consumers
    consumers="$(sycl_preflight_external_gpu_consumers)"
    if [[ -n "$consumers" && "${GGML_SYCL_ALLOW_EXTERNAL_GPU_CONSUMERS:-0}" != "1" ]]; then
        cat >&2 <<EOF
[sycl-preflight] refusing selector '$selector': external GPU-facing tools are active.
$consumers
EOF
        return 4
    fi

    return 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    sycl_gpu_preflight_check "${1:-${ONEAPI_DEVICE_SELECTOR:-level_zero:0}}"
fi
