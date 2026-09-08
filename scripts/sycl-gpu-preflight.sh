#!/usr/bin/env bash
# Passive guard for Intel Level Zero workloads on this workstation.
# Source this file and call sycl_gpu_preflight_check "$selector" before each
# GPU run. It intentionally does not enumerate devices through oneAPI.

sycl_preflight_repo_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

# Share derive_card_for_selector/is_top_level_card/find_card_by_pci with
# bench-guard.sh and sycl-decode-mode-capture.sh (llama.cpp-o4fs) instead of
# this file's own former hardcoded B50 PCI address (0000:07:00.0), which
# went stale at the 2026-09-05 boot when the discrete cards moved to
# 0000:04:00.0/0000:09:00.0 -- see CLAUDE.md. Sourcing has no side effects
# beyond defining those functions and defaulting $DRM_ROOT; see its own
# header comment for the full contract.
# Sourced relative to this script's own directory (BASH_SOURCE[0]), not the caller's cwd.
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sycl-gpu-sysfs.sh"

# sycl_preflight_b50_pci_address -- derive the PCI address of the SECOND
# (index 1) discrete Intel GPU under $SYCL_PREFLIGHT_DRM_ROOT (default
# /sys/class/drm, overridable for tests the same way bench-guard.sh's
# --drm-root is), i.e. the B50 by the same ascending-PCI-address convention
# bench-guard.sh documents (CLAUDE.md: "this assumes Level Zero orders the
# discrete cards by ascending PCI address -- true on every boot observed so
# far"). Prints nothing (and exits nonzero, though the caller below only
# checks for empty output) if a second discrete card cannot be derived --
# a single-GPU host, a topology error, or any of derive_card_for_selector's
# own loud-refusal cases.
#
# Runs derive_card_for_selector in a SUBSHELL with a local refuse() that
# merely exits that subshell, never the calling script -- this file is
# `source`d into arbitrary callers as a passive library, not something that
# owns the process, so a derivation failure here must never terminate
# whatever script sourced this file. This is exactly the kind of
# differently-shaped `refuse` sycl-gpu-sysfs.sh's own header contract
# anticipates: derive_card_for_selector does not care whether `refuse`
# exits the whole process or only a subshell, as long as it does not
# return normally back into it.
sycl_preflight_b50_pci_address() {
    local root="${SYCL_PREFLIGHT_DRM_ROOT:-/sys/class/drm}"
    (
        # shellcheck disable=SC2034  # read by derive_card_for_selector (sourced above)
        DRM_ROOT="$root"
        # shellcheck disable=SC2329  # invoked indirectly, from inside derive_card_for_selector
        refuse() { exit 1; }
        # NOT reset before the call, unlike bench-guard.sh/sycl-decode-mode-
        # capture.sh's own DERIVED_CARD="" DERIVED_PCI="" before their calls:
        # a stale DERIVED_PCI can never leak into this function's output
        # regardless, because a FAILED
        # derivation always reaches refuse()'s own `exit 1` -- which
        # terminates this SUBSHELL immediately -- before the `printf` line
        # below ever runs. Only a SUCCESSFUL derive_card_for_selector call
        # reaches printf, and success always means DERIVED_PCI was just
        # freshly assigned by that same call. The other two callers reset
        # defensively because they are NOT run in a subshell of their own --
        # a stale value there could in principle survive past a future
        # refactor that adds a path returning without calling refuse().
        derive_card_for_selector 1 >/dev/null 2>&1
        printf '%s\n' "$DERIVED_PCI"
    )
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

# Both journal-check functions below capture the full producer output into a
# variable FIRST and grep -c a herestring second, rather than piping straight
# into `grep -Eiq`. Piping directly is a fail-open SIGPIPE hazard: `grep -q`
# exits at the FIRST match without draining the rest of its input, so on a
# real fault (a match near the start of a long journal) the still-writing
# `journalctl` process gets SIGPIPE on its next write and exits 141. Under
# `set -o pipefail` (five of six sourcing callers run under it) bash reports
# a pipeline's status as the last command to exit non-zero -- `grep -q`
# itself exited 0 (it matched), so that 141 from `journalctl` becomes the
# pipeline's status, this function returns non-zero ("no fault"), and the
# gate does NOT refuse on a host with a current-boot CAT error or engine
# reset (llama.cpp-pqgl; reproduced with a fault line followed by 400k
# filler lines under `set -euo pipefail`). Capturing first sidesteps this
# entirely: the herestring grep never shares a pipe with a live writer.
#
# `|| return 1` on the capture preserves this file's existing behaviour for
# a MISSING or failing journalctl (same fail-open verdict the prior
# `command -v journalctl || return 1` guard gave): this preflight check is a
# passive advisory gate that already refuses independently on B50 sysfs
# state and uninterruptible GPU work, so "journal unreadable" here does not
# escalate to a refusal the way bench-guard.sh's postflight SUSPECT does --
# a deliberate difference, not an oversight, because this function's boolean
# result has no room to express a third "unreadable" state without changing
# every caller.
#
# Both captures below (this function's and its previous-boot sibling's) are
# deliberately UNBOUNDED (no `-n`, no `--since`): this is a
# current/previous-BOOT-wide fault check by design (`-b`/`-b -1`), not a
# windowed one like bench-guard.sh's postflight check (which has an actual
# window to bound -- the run it just guarded). There is no meaningful
# "recent enough" cutoff for "has this boot ever seen a GPU fault" short of
# re-deriving boot time from journalctl itself, which buys nothing over just
# reading the whole boot's journal. Measured size is modest in practice
# (~7.6 MB for a boot's kernel ring on this host) and this path already
# avoids the SIGPIPE hazard the capture-first rewrite above exists to fix.
sycl_preflight_journal_has_current_boot_gpu_faults() {
    command -v journalctl >/dev/null 2>&1 || return 1
    local jl_out n
    jl_out="$(journalctl -k -b --no-pager 2>/dev/null)" || return 1
    n="$(grep -ciE 'xe .*Engine reset|xe .*Schedule disable failed|xe .*reset (queued|started)|xe .*Timedout job|xe .*Kernel-submitted job timed out|Xe device coredump|guc_exec_queue_timedout_job|drm_sched_job_timedout|soft lockup|RCU.*stall|BUG:|Oops|ttm_resource_manager_usage|xe_drm_ioctl|xe_pt_zap_ptes' <<<"$jl_out" || true)"
    [ "${n:-0}" -gt 0 ]
}

sycl_preflight_journal_has_previous_boot_gpu_faults() {
    command -v journalctl >/dev/null 2>&1 || return 1
    local jl_out n
    jl_out="$(journalctl -k -b -1 --no-pager 2>/dev/null)" || return 1
    n="$(grep -ciE 'xe .*Engine reset|xe .*Schedule disable failed|xe .*reset (queued|started)|xe .*Timedout job|xe .*Kernel-submitted job timed out|Xe device coredump|guc_exec_queue_timedout_job|drm_sched_job_timedout|soft lockup|RCU.*stall|BUG:|Oops|ttm_resource_manager_usage|xe_drm_ioctl|xe_pt_zap_ptes' <<<"$jl_out" || true)"
    [ "${n:-0}" -gt 0 ]
}

# sycl_preflight_b50_sysfs_bad -- true (return 0) iff the B50's sysfs
# state looks bad (not confirmed enabled/D0/active) or its address could
# not be derived at all; false (return 1) iff it looks good, INCLUDING the
# idle-runtime-suspend exemption below. Bash return-code polarity: 0 means
# "yes, bad" here, matching how the caller uses it directly as an `if`
# condition (`sycl_preflight_selector_may_use_b50 "$selector" &&
# sycl_preflight_b50_sysfs_bad` refuses), NOT the ordinary "0 = success,
# entity is fine" sense a reader might otherwise assume from the name
# alone. Two env hooks, both overridable for hermetic tests the same way
# bench-guard.sh's own --drm-root is (see sycl_preflight_b50_pci_address
# above for the first): $SYCL_PREFLIGHT_DRM_ROOT (default /sys/class/drm)
# is where the B50's PCI address is derived FROM; $SYCL_PREFLIGHT_PCI_ROOT
# (default /sys/bus/pci/devices) is where that derived address is then
# looked UP to read enable/power_state/power/runtime_status.
sycl_preflight_b50_sysfs_bad() {
    local pci_root="${SYCL_PREFLIGHT_PCI_ROOT:-/sys/bus/pci/devices}"
    local pci b50
    # `|| true`: sycl_preflight_b50_pci_address legitimately returns nonzero
    # when it cannot derive a second discrete card (see its own docstring),
    # and that is an EXPECTED, checked outcome here (the `[[ -n "$pci" ]] ||
    # return 0` line right below), never a script-ending error. Without this,
    # a caller of this function that itself has `set -e` active and calls it
    # in a non-conditional context (not inside `if`/`&&`/`||`) would have the
    # whole calling script/shell terminated by THIS assignment the moment
    # derivation fails, before it ever reached the documented
    # "cannot derive -> bad" verdict below (the same class of bug this
    # file's own test suite hit and fixed with an identical `|| true` on its
    # own test-side calls).
    pci="$(sycl_preflight_b50_pci_address)" || true
    # Cannot derive a second discrete card's PCI address at all (single-GPU
    # host, topology error, ...): treat this the same conservative way the
    # old fixed-address form treated a genuinely-missing sysfs entry -- as
    # bad, not as "no B50 present, so nothing to check". Whether this
    # function's verdict is even consulted still depends on the caller's
    # own sycl_preflight_selector_may_use_b50 gate.
    [[ -n "$pci" ]] || return 0
    b50="$pci_root/$pci"
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
