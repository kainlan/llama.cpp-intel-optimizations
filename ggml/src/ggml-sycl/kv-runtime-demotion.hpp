#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "tlsf-allocator.hpp"

namespace ggml_sycl {

// Input snapshot of the placement state relevant to runtime KV demotion.
// Adapted from placement_plan (unified-cache.hpp) as plain values so this TU
// stays host-linkable with no unified-cache dependency.
//
// llama.cpp-3aos: layer_kv_bytes replaces what used to be a single
// kv_per_layer/kv_per_swa_layer scalar pair -- a uniform "one
// representative full-attention layer's bytes" figure applied to every
// full-attention layer regardless of its REAL per-layer width
// (placement_kv_info::kv_bytes_for_layer(), unified-cache.hpp) disagrees
// with refresh_kv_byte_totals() on the very same plan for a heterogeneous
// model (Gemma 4 E4B: full-attention layers are wider than its SWA layers).
// The caller (ggml_sycl_try_demote_runtime_kv(), ggml-sycl.cpp) now fills
// this per layer from that same shared formula, so the demotion decision
// and the byte-total refresh can no longer disagree about what any one
// layer costs. Named layer_kv_bytes, not kv_bytes_per_layer, to avoid
// colliding with placement_kv_info::kv_bytes_per_layer() (a scalar
// accessor, unified-cache.hpp) -- same name, unrelated shape.
struct kv_demotion_input {
    size_t               vram_budget = 0;
    size_t               vram_bytes  = 0;  // current total incl. device-resident KV
    // Per-layer device-resident KV bytes for THIS layer specifically (not a
    // uniform figure) -- 0 for a layer that holds no independent KV of its
    // own (e.g. a SHARED layer) or that this input does not track. Sized to
    // n_layers; kv_device and swa_layer_mask must be the same size.
    std::vector<size_t>  layer_kv_bytes;
    // Index = layer id; must be sized to n_layers. >=0 means device-resident on
    // that device id; -1 means already on the host tier. Layers absent from
    // placement_plan::kv_device are represented here as -1 (the caller adapts
    // the plan's sparse map into this dense vector before calling).
    std::vector<int>     kv_device;
    std::vector<uint8_t> swa_layer_mask;  // 1 = SWA layer
    // Demote SWA layers too, after every full-attention layer, when that is
    // not enough. Off: SWA layers are never demoted.
    bool                 demote_swa = false;
};

struct kv_demotion_result {
    std::vector<int> demoted_layers;  // in demotion order (latest full-attn first)
    size_t           vram_bytes_after    = 0;
    size_t           host_kv_bytes_added = 0;
    bool             fits                = false;  // vram_bytes_after <= vram_budget
};

// Pure decision: which device-resident KV layers must move to the host tier so
// vram_bytes fits vram_budget. Full-attention layers first, latest first; then,
// with demote_swa, SWA layers, latest first. Already-host layers and layers
// with 0 recorded bytes (nothing to move) are never touched. Does NOT mutate
// any plan -- the caller (ggml-sycl.cpp runtime update) applies the result to
// placement_plan::kv_device.
kv_demotion_result plan_runtime_kv_demotion(const kv_demotion_input & in);

// How many bytes of weights plus KV one device can hold before anything is
// materialized: with the VRAM arena active, weights and KV share one zone that
// is smaller than the device budget (the budget also pays for the SCRATCH,
// RUNTIME and ONEDNN zones). The planners pack weights against this. It is not
// what runtime KV admission asks: once weights are materialized, what is left
// for KV is the allocator's live headroom (unified_cache_kv_vram_available()),
// which also sees layout copies and zone growth the plan never charged.
// shared_zone_capacity == 0 means no arena zone: the budget is the limit.
size_t kv_weight_capacity(size_t vram_budget, size_t shared_zone_capacity);

// Live KV headroom from its two sources: with an active arena, the KV zone's
// free space, where 0 means the zone is full and never falls back to the budget
// path; without an arena, the budget-based compute headroom.
inline size_t kv_vram_available(bool has_arena, size_t zone_available, size_t budget_available) {
    return has_arena ? zone_available : budget_available;
}

// Whether a device's KV capacity and headroom may come from its own arena zones.
// A multi-device plan in GLOBAL cache mode has one cache, and so one KV zone, for
// every device, so neither KV question reads it per device. That makes
// unified_cache_kv_weight_capacity() correct, because its budget fallback is the
// plan's per-device vram_budget. It does NOT give unified_cache_kv_vram_available()
// a per-device answer: its budget fallback resolves every device to cache 0 and
// device 0's free VRAM, so each device is still admitted against one shared
// number. GLOBAL mode with a multi-device plan has no per-device KV headroom and
// is unsupported for runtime KV admission.
inline bool kv_reads_device_arena(bool multi_device, bool global_cache_mode) {
    return !(multi_device && global_cache_mode);
}

// Headroom runtime KV admission reserves per device-resident layer for the
// tiered KV allocator placing them one allocation at a time, possibly across
// two KV buffers: the arena allocator rounds every allocation up to its 256-byte
// block and the tiered allocator aligns each layer to 512 bytes
// (kv_layer_alloc_bytes); test-kv-runtime-demotion pins it against the
// allocator itself. Admission is a byte count over zone_available(); where free
// bytes are not whole-layer extents -- the holes an optional-layout yield
// leaves between live weights -- it is capped by what the zone can actually
// place (kv_residency_input::fit_capacity), because a device-planned layer the
// zone cannot place is not refused: it lands in raw device memory outside the
// arena (llama.cpp-moua).
constexpr size_t kv_alloc_slack_per_layer = 64 * 1024;

// The bytes one layer of KV asks the zone for: the tiered KV allocator aligns
// each layer's allocation to 512 bytes.
inline size_t kv_layer_alloc_bytes(size_t kv_bytes) {
    return (kv_bytes + 511) & ~size_t(511);
}

// The KV cache's shape as far as its size is concerned.
struct kv_shape {
    bool     runtime    = false;  // false: the load-time plan, before any runtime context
    uint32_t n_ctx      = 0;
    uint32_t n_seq_max  = 0;
    bool     kv_unified = false;
    bool     swa_full   = false;
};

// True when the KV shape differs from the published one (or the published plan
// is still the load-time one). n_ubatch is not part of the shape: it does not
// change KV size.
bool kv_shape_changed(const kv_shape & published, const kv_shape & next);

// True when a runtime-context update must decide KV residency again: the KV
// shape changed, or the context is not admitted yet. Every new context is
// fitted from the load-time residency against the live headroom, so none
// inherits an earlier context's residency, which may be more demoted than it
// needs.
//
// An admitted context never re-fits on a same-shape republish (the auto
// micro-batch trial): live available already excludes this context's own
// allocated KV, so fitting against it would demote KV that is already resident
// and contradict its own buffer.
//
// An admitted context whose shape changes would re-fit with its old KV still
// counted as used. llama_context fixes n_ctx, n_seq_max, kv_unified and
// swa_full before its first publish and never republishes another shape, so
// that case is unreachable for a single context; the runtime-context
// transaction WARNs if it ever happens. Constructing two contexts on one device
// with interleaved publishes can reach it: that is same-device concurrent
// contexts, which are unsupported (canonical memory contract §5).
bool kv_residency_needs_refit(const kv_shape & published, const kv_shape & next, bool context_admitted);

// True when `device`'s KV residency differs between the published plan and the
// next one: over the layers the device holds at load (load_kv_device), whether
// each is on the same device (or host tier, -1 or absent) in both. A runtime
// KV demotion on a device is news only when this is true for that device, so
// another device's change does not re-announce it.
bool kv_device_residency_changed(const std::unordered_map<int, int> & load_kv_device,
                                 const std::unordered_map<int, int> & published_kv_device,
                                 const std::unordered_map<int, int> & next_kv_device,
                                 int                                  device);

// GGML_SYCL_KV_HOT_LAYERS by value: a count >= 0 overrides the tier layout;
// unset or negative (-1) leaves it to the tier manager (kv-tier-manager.cpp).
inline bool kv_hot_layers_override_active(const char * value) {
    return value != nullptr && std::atoi(value) >= 0;
}

// The tiered KV allocator's backstop: device-planned KV for one buffer larger
// than the headroom it sees means admission and allocation disagreed. It
// refuses rather than silently demoting (llama.cpp-17ea).
inline bool kv_admission_mismatch(size_t planned_device_bytes, size_t kv_vram_cap) {
    return planned_device_bytes > kv_vram_cap;
}

// Where the tiered KV allocator puts one layer of a KV buffer created for
// `device`: the device whose VRAM holds it, or -1 for host memory. With a plan
// that is the plan's KV owner, so a layer another device owns is in that
// device's VRAM, not host memory; without one, the tier layout decides.
// force_host (GGML_SYCL_KV_HOST=1) puts every layer in host memory. The
// allocation loop, the host-memory refusal and the load summary all count by
// this.
inline int kv_buffer_layer_owner(bool have_plan, int plan_owner, bool layout_on_device, int device, bool force_host) {
    if (force_host) {
        return -1;
    }
    if (have_plan) {
        return plan_owner;
    }
    return layout_on_device ? device : -1;
}

// Runtime KV residency for a new KV shape.
struct kv_residency_input {
    // The planner's residency before any runtime demotion, so a smaller context
    // gets back the device residency a larger one gave up. Same conventions as
    // kv_demotion_input::kv_device.
    std::vector<int>     load_kv_device;
    std::vector<size_t>  layer_kv_bytes;  // at the new shape
    std::vector<uint8_t> swa_layer_mask;
    std::vector<int>     devices;         // devices to fit, in order
    std::vector<size_t>  available;       // live KV headroom of devices[i]
    // Bytes of optional layout copies devices[i] can release for its KV
    // (unified_cache_optional_layout_bytes()); empty means none.
    std::vector<size_t>  yieldable;
    // The most KV bytes devices[i] can hold whatever its headroom says: the
    // bytes of the leading layers its zone's free blocks can actually place
    // (unified_cache::yield_optional_layouts()). Empty, or SIZE_MAX, is no cap.
    std::vector<size_t>  fit_capacity;
    size_t               per_layer_slack = kv_alloc_slack_per_layer;
};

struct kv_residency_result {
    bool                            fits           = true;
    int                             refused_device = -1;  // the device that cannot fit even with KV demoted
    std::vector<int>                kv_device;            // the new residency
    std::vector<kv_demotion_result> per_device;           // same indexing as devices
    std::vector<size_t>             yield_bytes;          // optional layout bytes devices[i] must release
};

// Starts from load_kv_device. A device whose KV, plus per_layer_slack per
// resident layer, exceeds its headroom first counts up to its yieldable bytes
// as headroom (yield_bytes, which the caller must release before the KV is
// allocated): an optional layout copy never outranks KV for VRAM. Only what is
// still over then demotes the device's latest full-attention layers, then its
// latest SWA layers, to the host tier, with a device's KV also held to its
// fit_capacity. It refuses only when even that cannot fit.
kv_residency_result plan_runtime_kv_residency(const kv_residency_input & in);

// The allocators a device's KV is carved from, in the order zone_alloc(KV)
// tries them, as copies of their live state. Allocating on a copy is the
// allocator's own fit -- size-class rounding, coalescing on free -- which is
// what decides whether a KV layer lands, not a byte count: free bytes split
// into holes smaller than a layer are not KV headroom.
using kv_zone_model = std::vector<tlsf_allocator>;

// An allocation in a kv_zone_model: which allocator, and its offset there.
// allocator == SIZE_MAX: not in the model (it lives where KV is not carved).
struct kv_zone_block {
    size_t allocator = SIZE_MAX;
    size_t offset    = 0;
};

// How many of `layer_bytes`, allocated in order one allocation a layer (each
// from the first allocator with room, as the tiered KV allocator does), land
// before the first that does not. Works on its own copy of `zone`.
size_t kv_layers_allocatable(kv_zone_model zone, const std::vector<size_t> & layer_bytes);

// Which optional layout copies to release so more of `layer_bytes` land. Walks
// the copies from the zone's high end down, since copies staged together sit
// together and free into one extent with each other and with the free space
// above them; a copy is kept for release only once the extent it joins lets
// another layer land, and one that extent does not need is dropped again. A
// copy no layer can use stays resident. Returns indices into `copies`.
std::vector<size_t> select_optional_layout_yield(const kv_zone_model &              zone,
                                                 const std::vector<kv_zone_block> & copies,
                                                 const std::vector<size_t> &        layer_bytes);

// One device's view of a (possibly multi-device) plan for the zone-fit pass.
struct kv_device_fit_input {
    int                  device   = -1;
    size_t               capacity = 0;  // KV bytes the device can hold: its live KV headroom less the slack
    // Indexed by layer id, same conventions as kv_demotion_input. Layers owned
    // by other devices (or already on host) are never touched.
    std::vector<size_t>  layer_kv_bytes;
    std::vector<int>     kv_device;
    std::vector<uint8_t> swa_layer_mask;
    bool                 demote_swa = false;  // see kv_demotion_input
};

// Which of `device`'s KV layers must move to the host tier so its KV fits
// `capacity`. Same rules as plan_runtime_kv_demotion(), which it
// delegates to.
kv_demotion_result plan_device_kv_fit(const kv_device_fit_input & in);

// The KV owner of layers [start_layer, end_layer]: the device every layer that
// holds KV uses, -2 when they disagree, -1 when none holds KV. Layers with 0
// recorded bytes do not vote. The range is clamped to the vectors.
int layer_block_kv_device(const std::vector<int> &    kv_device,
                          const std::vector<size_t> & layer_kv_bytes,
                          int                         start_layer,
                          int                         end_layer);

}  // namespace ggml_sycl
