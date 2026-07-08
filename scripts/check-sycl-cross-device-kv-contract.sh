#!/usr/bin/env bash
set -euo pipefail

doc="${1:-docs/backend/SYCL.md}"

section="$(
    awk '
        /^## Cross-device KV placement contract$/ { in_section = 1 }
        in_section { print }
        in_section && /^## / && $0 !~ /^## Cross-device KV placement contract$/ { exit }
    ' "$doc" | tr '\n' ' ' | sed -E 's/[[:space:]]+/ /g'
)"

require() {
    local pattern="$1"
    local description="$2"

    if ! grep -Eiq "$pattern" <<<"$section"; then
        printf 'missing SYCL cross-device KV contract clause: %s\n' "$description" >&2
        exit 1
    fi
}

require 'Cross-device KV placement contract' 'section heading'
require 'kv_device.*layer_device' 'planner kv_device/layer_device semantics'
require 'device_count.*total_gpu_count|total_gpu_count.*device_count' 'scheduler-visible and physical device counts'
require 'remote device-planned KV.*never.*ordinary host-pinned fallback' 'remote device-planned KV is not host fallback'
require 'CPU fallback.*invalid.*device-planned F16 attention|device-planned F16 attention.*CPU fallback.*invalid' 'CPU fallback invalid for device-planned F16 attention'
require 'safe default.*expert.*hybrid/layer.*explicit|expert.*safe default.*hybrid/layer.*explicit' 'expert-mode safe default and explicit hybrid/layer'
require 'planned device.*materialized owner.*buffer type.*root extra.*smart-handle.*routing decision' 'diagnostic fields'

printf 'SYCL cross-device KV contract documentation present in %s\n' "$doc"
