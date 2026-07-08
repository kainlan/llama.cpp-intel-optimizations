#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKER="$ROOT_DIR/scripts/check-sycl-alloc-usage.sh"

"$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/good"
"$CHECKER" "$ROOT_DIR/ggml/src/ggml-sycl"

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-alloc" >/dev/null 2>&1; then
    echo "expected bad alloc fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-host-alloc" >/dev/null 2>&1; then
    echo "expected bad host alloc fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-vm-alloc" >/dev/null 2>&1; then
    echo "expected bad vm alloc fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-dealloc" >/dev/null 2>&1; then
    echo "expected bad dealloc fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-cache-alloc" >/dev/null 2>&1; then
    echo "expected bad cache alloc fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-zone-alloc" >/dev/null 2>&1; then
    echo "expected bad zone alloc fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-zone-handle" >/dev/null 2>&1; then
    echo "expected bad zone handle fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-xmx-legacy" >/dev/null 2>&1; then
    echo "expected bad XMX legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-xmx-device-tmp-legacy" >/dev/null 2>&1; then
    echo "expected bad XMX device temp legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-sycl-ext-device-temp-legacy" >/dev/null 2>&1; then
    echo "expected bad sycl_ext device temp legacy helper fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-xmx-tiled-owner-legacy" >/dev/null 2>&1; then
    echo "expected bad XMX tiled owner legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-spec-verify-logits-legacy" >/dev/null 2>&1; then
    echo "expected bad speculative logits legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-pp-stage-legacy" >/dev/null 2>&1; then
    echo "expected bad PP stage legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-pp-pipeline-legacy" >/dev/null 2>&1; then
    echo "expected bad PP pipeline legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-fattn-split-legacy" >/dev/null 2>&1; then
    echo "expected bad FATTN split legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-fattn-packed-k-legacy" >/dev/null 2>&1; then
    echo "expected bad FATTN packed-K legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-fattn-onednn-materialized-raw-release" >/dev/null 2>&1; then
    echo "expected bad oneDNN FATTN materialized raw release fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-fattn-onednn-materialized-scoped-alloc" >/dev/null 2>&1; then
    echo "expected bad oneDNN FATTN materialized scoped alloc/as_mem_handle fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-fattn-buffers-kv-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad FATTN KV buffer legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-onednn-xmx-fill-scoped-stage" >/dev/null 2>&1; then
    echo "expected bad oneDNN/XMX fill scoped staging fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-ggml-sycl-cpp-scoped-unified-alloc" >/dev/null 2>&1; then
    echo "expected bad ggml-sycl.cpp scoped_unified_alloc fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-layer-stream-legacy" >/dev/null 2>&1; then
    echo "expected bad layer-stream legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-layer-stream-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad layer-stream legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-compute-buffer-legacy" >/dev/null 2>&1; then
    echo "expected bad compute-buffer legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-compute-buffer-manager-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad compute-buffer-manager legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-gpu-sampler-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad GPU sampler legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-backend-staging-legacy" >/dev/null 2>&1; then
    echo "expected bad backend staging legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-control-host-legacy" >/dev/null 2>&1; then
    echo "expected bad host-control legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-ffn-norm-legacy" >/dev/null 2>&1; then
    echo "expected bad FFN norm legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-tp-input-legacy" >/dev/null 2>&1; then
    echo "expected bad TP input legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-tp-input-capture-legacy" >/dev/null 2>&1; then
    echo "expected bad TP captured input legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-cpu-dispatch-host-copy-legacy" >/dev/null 2>&1; then
    echo "expected bad CPU-dispatch host-copy legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-cpu-dispatch-host-copy-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad CPU-dispatch host-copy legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-usm-pointer-type-cache-legacy" >/dev/null 2>&1; then
    echo "expected bad USM pointer-type cache legacy fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-weight-identity-pointer-cache-legacy" >/dev/null 2>&1; then
    echo "expected bad weight identity pointer-cache legacy fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-host-weight-extras-pointer-registry-legacy" >/dev/null 2>&1; then
    echo "expected bad host weight extras pointer-registry legacy fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-cpu-dispatch-host-ptr-split-registry-legacy" >/dev/null 2>&1; then
    echo "expected bad CPU-dispatch host pointer split-registry legacy fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-staging-cache-legacy" >/dev/null 2>&1; then
    echo "expected bad staging-cache legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-mem-ops-stage-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad mem-ops staging legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-cpy-host-stage-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad CPY host-stage legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-set-rows-host-stage-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad SET_ROWS host-stage legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-dmmv-stage-scratch-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad DMMV staging/scratch legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-convert-device-scratch-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad CONVERT device-scratch legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-mmq-xmx-device-scratch-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad MMQ_XMX device-scratch legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-dense-scheduler-slot-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad dense-scheduler slot legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-mmq-stage-counter-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad MMQ staging/counter legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-mmvq-stage-reuse-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad MMVQ staging/reuse legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-mmvq-graph-compact-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad MMVQ graph/compact legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-get-rows-stage-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad GET_ROWS staging legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-set-rows-staging-legacy" >/dev/null 2>&1; then
    echo "expected bad set_rows staging legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-set-rows-staging-scoped" >/dev/null 2>&1; then
    echo "expected bad SET_ROWS staging scoped allocation fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-binbcast-raw-host-stage-scoped" >/dev/null 2>&1; then
    echo "expected bad BINBCAST raw-host staging scoped allocation fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-common-host-staging-scoped" >/dev/null 2>&1; then
    echo "expected bad common.cpp host-staging scoped allocation fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-readback-fallback-scoped" >/dev/null 2>&1; then
    echo "expected bad readback fallback scoped staging fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-moe-readback-stage-scoped" >/dev/null 2>&1; then
    echo "expected bad MoE readback scoped staging fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-moe-phase2-d2h-scoped" >/dev/null 2>&1; then
    echo "expected bad MOE phase2 D2H scoped staging fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-split-weight-stage-scoped" >/dev/null 2>&1; then
    echo "expected bad split-weight D2H scoped staging fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-graph-q8-legacy" >/dev/null 2>&1; then
    echo "expected bad graph Q8 legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-mmvq-rmsnorm-scales-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad MMVQ RMSNorm scales legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-get-rows-host-stage-scoped" >/dev/null 2>&1; then
    echo "expected bad GET_ROWS host-stage scoped alloc/as_mem_handle fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-tp-reduce-legacy" >/dev/null 2>&1; then
    echo "expected bad TP reduce legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-tp-host-staging-legacy" >/dev/null 2>&1; then
    echo "expected bad TP host-staging legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-tp-quant-comm-legacy" >/dev/null 2>&1; then
    echo "expected bad TP quant-comm legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-tp-async-result-legacy" >/dev/null 2>&1; then
    echo "expected bad TP async-result legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-tp-alloc-tmp-legacy-out" >/dev/null 2>&1; then
    echo "expected bad TP alloc_tmp legacy alloc_handle out fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-tp-compute-buffer-legacy" >/dev/null 2>&1; then
    echo "expected bad TP compute-buffer legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-dev1-kv-cache-legacy" >/dev/null 2>&1; then
    echo "expected bad dev1 KV-cache legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-tp-column-output-legacy" >/dev/null 2>&1; then
    echo "expected bad TP column-output legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-routing-indices-legacy" >/dev/null 2>&1; then
    echo "expected bad routing-indices legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-moe-ids-legacy" >/dev/null 2>&1; then
    echo "expected bad MoE IDs legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-moe-ids-pack-legacy" >/dev/null 2>&1; then
    echo "expected bad MoE IDs packed staging legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-moe-fusion-legacy" >/dev/null 2>&1; then
    echo "expected bad MoE fusion legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-moe-ptr-table-legacy" >/dev/null 2>&1; then
    echo "expected bad MoE pointer-table legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-moe-gpu-probe-legacy" >/dev/null 2>&1; then
    echo "expected bad MoE GPU probe legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-gpu-reorder-temp-legacy" >/dev/null 2>&1; then
    echo "expected bad GPU reorder temp VRAM legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-reorder-d2h-stage-legacy" >/dev/null 2>&1; then
    echo "expected bad reorder D2H staging legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-payload-stage-legacy" >/dev/null 2>&1; then
    echo "expected bad payload staging legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-generic-owned-alloc-handoff" >/dev/null 2>&1; then
    echo "expected bad generic owned alloc handoff fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-runtime-lookup-owner-alloc" >/dev/null 2>&1; then
    echo "expected bad runtime lookup owner_alloc fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-backend-buffer-context-owner" >/dev/null 2>&1; then
    echo "expected bad backend buffer context alloc_handle owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-backend-buffer-context-assign" >/dev/null 2>&1; then
    echo "expected bad backend buffer context raw alloc_handle assignment fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-backend-buffer-context-as-mem-handle" >/dev/null 2>&1; then
    echo "expected bad backend buffer context as_mem_handle rebuild fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-backend-buffer-context-lookup-free" >/dev/null 2>&1; then
    echo "expected bad backend buffer context lookup/free fallback fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-sycl-pool-legacy" >/dev/null 2>&1; then
    echo "expected bad SYCL pool legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-tensor-extra-data-alloc-legacy" >/dev/null 2>&1; then
    echo "expected bad tensor-extra data_alloc legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-host-buffer-context-legacy" >/dev/null 2>&1; then
    echo "expected bad host buffer context legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-vram-pool-legacy" >/dev/null 2>&1; then
    echo "expected bad VRAM pool legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-allocate-direct-handle" >/dev/null 2>&1; then
    echo "expected bad unified_allocate direct handle fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-managed-host-pinned-legacy" >/dev/null 2>&1; then
    echo "expected bad managed host pinned legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-allocate-managed-host-pinned-legacy" >/dev/null 2>&1; then
    echo "expected bad allocate managed host pinned legacy handoff fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-staging-buffer-pool-legacy" >/dev/null 2>&1; then
    echo "expected bad staging buffer pool legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-cpu-fallback-host-copy-legacy" >/dev/null 2>&1; then
    echo "expected bad CPU fallback host-copy legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-tiered-kv-zone-h-legacy" >/dev/null 2>&1; then
    echo "expected bad tiered KV zone_h legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-onednn-scratch-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad oneDNN scratch legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-persistent-scratch-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad persistent scratch legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-compute-arena-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad compute arena legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-scratch-pool-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad scratch pool legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-staging-owner-legacy" >/dev/null 2>&1; then
    echo "expected bad staging owner legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-partial-row-cache-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad partial row cache legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-direct-alloc-owner-legacy" >/dev/null 2>&1; then
    echo "expected bad direct/deferred owner legacy fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-moe-prestage-cpu-reorder-legacy" >/dev/null 2>&1; then
    echo "expected bad MoE prestage CPU reorder legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-set-tensor-reorder-fallback-legacy" >/dev/null 2>&1; then
    echo "expected bad set_tensor reorder fallback legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-expert-prefetch-legacy" >/dev/null 2>&1; then
    echo "expected bad expert-prefetch legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-fp16-cache-legacy" >/dev/null 2>&1; then
    echo "expected bad FP16 cache legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-pinned-buffer-pool-legacy" >/dev/null 2>&1; then
    echo "expected bad pinned-buffer-pool legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-pinned-buffer-pool-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad pinned-buffer-pool legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-pinned-pool-chunk-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad pinned-pool chunk legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-pending-cpu-legacy" >/dev/null 2>&1; then
    echo "expected bad pending CPU scatter/pipeline legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-cpu-expert-pool-legacy" >/dev/null 2>&1; then
    echo "expected bad CPU expert pool legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-cpu-expert-pool-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad CPU expert pool legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-cont-batching-legacy-owner" >/dev/null 2>&1; then
    echo "expected bad continuous-batching legacy owner fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-micrograph-legacy" >/dev/null 2>&1; then
    echo "expected bad unified-kernel micro-graph legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-light-flags-legacy" >/dev/null 2>&1; then
    echo "expected bad unified-kernel light-flags legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-role-schedule-legacy" >/dev/null 2>&1; then
    echo "expected bad unified-kernel role-schedule legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-ops-pool-legacy" >/dev/null 2>&1; then
    echo "expected bad unified-kernel ops-pool legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-phase-schedule-legacy" >/dev/null 2>&1; then
    echo "expected bad unified-kernel phase-schedule legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-get-rows-legacy" >/dev/null 2>&1; then
    echo "expected bad unified-kernel get-rows legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-get-rows-shared-handle" >/dev/null 2>&1; then
    echo "expected bad unified-kernel shared GET_ROWS handle fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-scratch-pool-legacy" >/dev/null 2>&1; then
    echo "expected bad unified-kernel scratch-pool legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-deferred-copy-raw-scratch" >/dev/null 2>&1; then
    echo "expected bad unified-kernel deferred-copy raw scratch fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-deferred-copy-exec-raw-fallback" >/dev/null 2>&1; then
    echo "expected bad unified-kernel deferred-copy execution raw fallback fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-deferred-copy-raw-api-bridge" >/dev/null 2>&1; then
    echo "expected bad unified-kernel deferred-copy raw API bridge fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-final-copy-raw-dst" >/dev/null 2>&1; then
    echo "expected bad unified-kernel final copy raw destination fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-raw-handle-bridge-def" >/dev/null 2>&1; then
    echo "expected bad unified-kernel raw handle bridge definition fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-final-copy-op-index" >/dev/null 2>&1; then
    echo "expected bad unified-kernel final copy op-index fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-persistent-split-input-raw-rewrap" >/dev/null 2>&1; then
    echo "expected bad persistent split input raw rewrap fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-pending-device-token-raw-rewrap" >/dev/null 2>&1; then
    echo "expected bad pending device-token raw rewrap fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-pending-device-token-raw-state" >/dev/null 2>&1; then
    echo "expected bad pending device-token raw state fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-seq-ids-raw-state" >/dev/null 2>&1; then
    echo "expected bad seq_ids raw state fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-moe-weights-by-layer-pointer-cache" >/dev/null 2>&1; then
    echo "expected bad MoE weights-by-layer pointer-cache fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-moe-precomputed-skip-pointer-set" >/dev/null 2>&1; then
    echo "expected bad MoE precomputed-skip pointer-set fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-graph-prestage-pointer-dedupe" >/dev/null 2>&1; then
    echo "expected bad graph-prestage pointer dedupe fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-graph-input-discovery-pointer-dedupe" >/dev/null 2>&1; then
    echo "expected bad graph input-discovery pointer dedupe fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-prefetch-scheduler-pointer-active-set" >/dev/null 2>&1; then
    echo "expected bad PrefetchScheduler pointer active-set fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-moe-q8-pointer-cache" >/dev/null 2>&1; then
    echo "expected bad MoE Q8 pointer-cache fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-cpu-dispatch-expert-group-pointer" >/dev/null 2>&1; then
    echo "expected bad CPU-dispatch expert grouping pointer fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-backend-buffer-runtime-zone-reset" >/dev/null 2>&1; then
    echo "expected bad backend-buffer RUNTIME zone reset fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-persistent-tg-deferred-copy-raw" >/dev/null 2>&1; then
    echo "expected bad persistent TG deferred-copy raw registration fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-sync-block-legacy" >/dev/null 2>&1; then
    echo "expected bad unified-kernel sync-block legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-persistent-buffer-legacy" >/dev/null 2>&1; then
    echo "expected bad unified-kernel persistent-buffer legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-device-persistent-owner" >/dev/null 2>&1; then
    echo "expected bad unified-kernel device-persistent legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-pinned-persistent-owner" >/dev/null 2>&1; then
    echo "expected bad unified-kernel pinned-persistent legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-device-scratch-owner" >/dev/null 2>&1; then
    echo "expected bad unified-kernel device-scratch legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-dag-legacy" >/dev/null 2>&1; then
    echo "expected bad unified-kernel DAG legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-temp-device-legacy" >/dev/null 2>&1; then
    echo "expected bad unified-kernel temp-device legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-unified-kernel-graph-overhead-legacy" >/dev/null 2>&1; then
    echo "expected bad unified-kernel graph-overhead legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-split-persistent-q8-legacy" >/dev/null 2>&1; then
    echo "expected bad split persistent Q8 legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-split-persistent-sync-legacy" >/dev/null 2>&1; then
    echo "expected bad split persistent sync legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-mmvq-soa-bulk-legacy" >/dev/null 2>&1; then
    echo "expected bad MMVQ SoA bulk legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-split-secondary-gpu-legacy" >/dev/null 2>&1; then
    echo "expected bad split secondary GPU legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-split-secondary-output-legacy" >/dev/null 2>&1; then
    echo "expected bad split secondary output legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-split-weight-cache-legacy" >/dev/null 2>&1; then
    echo "expected bad split weight cache legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-mxfp4-tg-reuse-legacy" >/dev/null 2>&1; then
    echo "expected bad MXFP4 TG reuse legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-secondary-layer-tg-legacy" >/dev/null 2>&1; then
    echo "expected bad secondary layer TG legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-secondary-ring-legacy" >/dev/null 2>&1; then
    echo "expected bad secondary ring-buffer legacy ownership fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-persistent-set-rows-validate-scoped" >/dev/null 2>&1; then
    echo "expected bad persistent SET_ROWS validate scoped allocation fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-device-wrapper-alloc" >/dev/null 2>&1; then
    echo "expected bad device wrapper alloc fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-free-ptr" >/dev/null 2>&1; then
    echo "expected bad free ptr fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-copy" >/dev/null 2>&1; then
    echo "expected bad copy fixture to fail policy check" >&2
    exit 1
fi

if "$CHECKER" "$ROOT_DIR/tests/sycl-alloc-policy-fixtures/bad-fill" >/dev/null 2>&1; then
    echo "expected bad fill fixture to fail policy check" >&2
    exit 1
fi

echo "sycl alloc policy fixtures passed"
