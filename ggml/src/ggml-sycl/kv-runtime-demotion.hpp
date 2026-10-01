#pragma once
#include "shared-zone-tags.hpp"
#include "tlsf-allocator.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <vector>

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

// The optional layout copies a device could release for its KV (an optional
// copy yields to runtime KV, unified_cache_entry::optional_layout), as its fit
// reads them: one copy of the device's KV zone and where each copy sits in it.
// This one snapshot is where the fit's yieldable bytes, the copies it picks and
// the layers it expects to land all come from. An empty zone is no zone:
// without an arena KV is not carved from one, so none limits it.
struct kv_optional_layouts {
    kv_zone_model              zone;
    std::vector<kv_zone_block> copies;
    std::vector<size_t>        bytes;  // same indexing as copies
};

// Which copies to release so more of `layer_bytes` land, as groups: a group is
// released whole or not at all, since part of one frees an extent no layer the
// group made room for fits. Walks the copies from the zone's high end down,
// since copies staged together sit together and free into one extent with each
// other and with the free space above them; a copy joins a group only once
// the extent it joins lets another layer land, and one that extent does not
// need is dropped again. A copy no layer can use stays resident.
struct kv_optional_layout_yield {
    std::vector<std::vector<size_t>> groups;         // indices into kv_optional_layouts::copies
    size_t                           kv_layers = 0;  // of layer_bytes, how many land once every group is released
};

kv_optional_layout_yield plan_optional_layout_yield(const kv_optional_layouts & copies,
                                                    const std::vector<size_t> & layer_bytes);

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
    // The optional layout copies devices[i] can release for its KV
    // (unified_cache_optional_layouts_snapshot()); empty means none.
    std::vector<kv_optional_layouts> optional_layouts;
    // The most KV bytes devices[i] can hold whatever its headroom says: the
    // bytes of the leading layers its zone's free blocks can actually place
    // (unified_cache::yield_optional_layouts_finish()). Empty, or SIZE_MAX, is
    // no cap.
    std::vector<size_t>  fit_capacity;
    size_t               per_layer_slack = kv_alloc_slack_per_layer;
};

struct kv_residency_result {
    bool                            fits           = true;
    int                             refused_device = -1;  // the device that cannot fit even with KV demoted
    std::vector<int>                kv_device;            // the new residency
    std::vector<kv_demotion_result> per_device;           // same indexing as devices
    std::vector<size_t>             yield_bytes;  // optional layout bytes the fit counted as devices[i]'s headroom
    // The copies devices[i] must release before its KV is allocated (empty
    // when yield_bytes[i] is 0), which the caller hands to the yield as its
    // pick list.
    std::vector<kv_optional_layout_yield> yields;
};

// Starts from load_kv_device. A device whose KV, plus per_layer_slack per
// resident layer, exceeds its headroom first counts up to its optional layout
// copies' bytes as headroom (yield_bytes): an optional layout copy never
// outranks KV for VRAM. Those bytes are headroom only where a layer lands in
// them, so the device's copies are then modelled on its zone
// (plan_optional_layout_yield()): the ones worth releasing become its pick list
// (yields), and its KV is held to the leading layers the zone places once they
// go. Only what is still over then demotes the device's latest full-attention
// layers, then its latest SWA layers, to the host tier, with a device's KV
// also held to its fit_capacity. It refuses only when even that cannot fit.
kv_residency_result plan_runtime_kv_residency(const kv_residency_input & in);

// devices' KV layers in allocation order (load_kv_device order), as the bytes
// each asks its zone for (kv_layer_alloc_bytes).
std::vector<size_t> kv_device_layer_alloc_bytes(const kv_residency_input & in, int device);

// The KV bytes of device's first n_layers layers in allocation order: what a
// device holds when n_layers of them land.
size_t kv_device_leading_layer_bytes(const kv_residency_input & in, int device, size_t n_layers);

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

// ---------------------------------------------------------------------------
// One per-layer KV byte function (llama.cpp-moua).
//
// The cell arithmetic below was kv_layer_bytes_for_kind()'s (unified-cache.hpp,
// llama.cpp-3aos and llama.cpp-uajm own the derivation of every branch there);
// it lives here, SYCL-free, so the load-time estimate and the context's region
// slots cannot size one layer two ways.  kv_layer_bytes_for_kind() calls
// kv_layer_cells() and kv_layer_tensor_bytes() with an f16 shape and no
// padding; the region fit sizes a slot from the shape llama actually publishes
// and the tiered buft's alignment.
// ---------------------------------------------------------------------------

// A layer's attention kind as far as its cell count is concerned.  The values
// are ggml_sycl_kv_layer_kind's (ggml-sycl.h); unified-cache.hpp static_asserts
// that, because this header takes no ggml-sycl.h include.
enum kv_cells_kind : uint8_t {
    KV_CELLS_FULL   = 0,
    KV_CELLS_SWA    = 1,
    KV_CELLS_SHARED = 2,  // no K/V of its own: 0 cells
};

// Cells one layer of `kind` holds, across all of its streams.  The derivation of
// each branch is the comment above kv_layer_bytes_for_kind() (unified-cache.hpp,
// llama.cpp-3aos and llama.cpp-uajm).  Inline: every target that reaches the
// load-time estimate through unified-cache.hpp needs it without linking this TU.
inline size_t kv_layer_cells(uint8_t  kind,
                             uint32_t n_ctx,
                             uint32_t n_ubatch,
                             uint32_t n_seq_max,
                             bool     kv_unified,
                             bool     swa_full,
                             uint32_t n_swa) {
    if (kind == KV_CELLS_SHARED) {
        return 0;
    }
    if (kind == KV_CELLS_SWA && !swa_full) {
        if (n_swa == 0) {
            return 0;
        }
        const uint32_t seqs = n_seq_max > 0 ? n_seq_max : 1;
        uint32_t       n_ctx_seq;    // cells per stream in the non-SWA (base) cache
        uint32_t       n_stream;     // number of independent KV streams
        uint32_t       window_seqs;  // the "unified ? n_seq_max : 1" term, llama_kv_cache_iswa::llama_kv_cache_iswa()
        auto           pad256 = [](uint32_t x) {
            return (x + 255u) & ~255u;
        };
        if (kv_unified) {
            n_ctx_seq   = n_ctx;
            n_stream    = 1;
            window_seqs = seqs;
        } else {
            // n_ctx is already adjusted to n_ctx_seq * n_seq_max, so dividing
            // seqs back out reproduces llama's own n_ctx_seq exactly.
            n_ctx_seq   = pad256(n_ctx / seqs);
            n_stream    = seqs;
            window_seqs = 1;
        }
        const uint32_t swa_cells_per_stream = pad256(std::min(n_ctx_seq, n_swa * window_seqs + n_ubatch));
        const uint32_t swa_cells            = swa_cells_per_stream * n_stream;
        return static_cast<size_t>(swa_cells);
    }
    // FULL, and SWA under swa_full: every cell of the context window.  Total
    // cells across streams is n_ctx_seq * n_stream, which llama_context's
    // n_ctx_seq invariant makes n_ctx in both modes.
    return static_cast<size_t>(n_ctx);
}

// One layer's widths, exactly the ones llama passes to ggml_new_tensor_3d for
// its K and V (n_embd_v_gqa taken after the [TAG_V_CACHE_VARIABLE] padding, 0
// when the model has no V).  has_kv == 0 marks a filtered, shared or reused
// layer, which holds no slot.
struct kv_layer_desc {
    uint32_t n_embd_k_gqa  = 0;
    uint32_t n_embd_v_gqa  = 0;
    uint32_t n_head_kv     = 0;
    uint32_t n_embd_head_k = 0;
    uint8_t  has_kv        = 0;
    uint8_t  is_swa        = 0;
};

// ggml_row_size, injected: this header is SYCL-free and its host test links no
// ggml-base.  Production passes ggml_row_size; a host test passes its own table.
typedef size_t (*kv_row_size_fn)(int32_t type, int64_t n_elements);

// `bytes` rounded up to a multiple of `pad_to`; 0 and 1 leave it alone.
inline size_t kv_pad_bytes(size_t bytes, size_t pad_to) {
    if (pad_to <= 1 || bytes == 0) {
        return bytes;
    }
    return ((bytes + pad_to - 1) / pad_to) * pad_to;
}

// Bytes of one layer's K and V tensors over `cells` cells:
// row_size(type, width) * cells for each, each padded to `pad_to` the way
// ggml_backend_alloc_ctx_tensors_from_buft pads every tensor it places.  A
// layer with has_kv == 0 is 0.  pad_to is the tiered buft's alignment for a
// region slot and 1 for the load-time estimate, which never sizes a region.
// That alignment is the whole of the buft's size rule today: get_alloc_size is
// NULL (identity) for the tiered buft.  A buft with a real get_alloc_size, such
// as a future recurrent-state buft, needs a second per-tensor hook applied here
// before the padding (L4's kv_layer_alloc_bytes is where it goes).
inline size_t kv_layer_tensor_bytes(const kv_layer_desc & layer,
                                    int32_t               type_k,
                                    int32_t               type_v,
                                    size_t                cells,
                                    size_t                pad_to,
                                    kv_row_size_fn        row_size) {
    if (!layer.has_kv || row_size == nullptr) {
        return 0;
    }
    const size_t k_bytes = layer.n_embd_k_gqa > 0 ? row_size(type_k, layer.n_embd_k_gqa) * cells : 0;
    const size_t v_bytes = layer.n_embd_v_gqa > 0 ? row_size(type_v, layer.n_embd_v_gqa) * cells : 0;
    return kv_pad_bytes(k_bytes, pad_to) + kv_pad_bytes(v_bytes, pad_to);
}

// One recurrent layer, exactly the arguments llama_memory_recurrent passes to
// ggml_new_tensor_2d for r_l and s_l (n_rows = mem_size * (1 + n_rs_seq)).
struct rs_layer_desc {
    uint32_t il       = 0;
    int32_t  type_r   = 0;
    int32_t  type_s   = 0;
    uint32_t n_embd_r = 0;
    uint32_t n_embd_s = 0;
    uint32_t n_rows   = 0;
};

// Bytes of one device's recurrent-state buffer: r_l then s_l of every layer in
// order, each tensor row_size(type, n_embd) * n_rows padded to `pad_to`, summed
// the way ggml_backend_alloc_ctx_tensors_from_buft lays one context out.
size_t rs_buffer_bytes(const std::vector<rs_layer_desc> & layers, size_t pad_to, kv_row_size_fn row_size);

// ---------------------------------------------------------------------------
// The context-side demand record (llama.cpp-moua §2.4.3).  A producer lists, by
// index, the allocations that can be live at once; every claim names its index.
// ---------------------------------------------------------------------------

// Whose lifetime a reservation follows.  There is no DEVICE scope: the u1bb
// ring's rows are per-context CONTEXT slots.
enum class demand_scope : uint8_t { MODEL, CONTEXT };

// How the fit places a record.  A HEAD_SLOT is mandatory and placed before any
// KV; an AFTER_KV record (23mk's oneDNN Graph scratch) is charged after the KV
// extents, never demotes KV and never refuses.
enum class demand_placement : uint8_t { HEAD_SLOT, AFTER_KV };

struct context_side_demand {
    int                  device   = -1;
    // CONTEXT, TRANSIENT or WEIGHT_SIDE_TRANSIENT, as SHARED_ZONE_LIFETIME_*
    shared_zone_lifetime lifetime = shared_zone_lifetime::SHARED_ZONE_LIFETIME_CONTEXT;
    demand_scope         scope    = demand_scope::CONTEXT;
    uint64_t             owner    = 0;                              // CONTEXT: the ContextId; MODEL: the ModelId
    const char *         cohort   = nullptr;                        // the cohort id its claims carry
    std::vector<size_t>  slots;                                     // slots[i] = cap of slot index i
    demand_placement     placement = demand_placement::HEAD_SLOT;
};

// A reserved slot an owner already holds: the key of a claim plus the cap.
struct held_slot {
    int          device = -1;
    demand_scope scope  = demand_scope::CONTEXT;
    uint64_t     owner  = 0;
    const char * cohort = nullptr;
    uint32_t     index  = 0;
    size_t       cap    = 0;
};

// Whether `held` is the slot of `demand`'s index `index` (same device, scope,
// owner, cohort and index), whatever its cap.
bool demand_slot_key_matches(const held_slot & held, const context_side_demand & demand, uint32_t index);

// The held slot that serves slot `index` of `demand` in place, or SIZE_MAX: the
// owner already holds the same (cohort, index) with a cap at least `need`.  A
// larger slot serves a smaller claim (size <= cap[index]), so a shrink is never
// a carve.
size_t demand_find_reusable_slot(const std::vector<held_slot> & held,
                                 const context_side_demand &    demand,
                                 uint32_t                       index,
                                 size_t                         need);

// What a transaction does with a context's records against what it holds.
struct demand_slot_ref {
    size_t   record = 0;         // index into the demands
    uint32_t index  = 0;         // slot index within the record
    size_t   size   = 0;         // the demanded cap
    size_t   held   = SIZE_MAX;  // index into held, or SIZE_MAX when none
};

struct demand_reconciliation {
    std::vector<demand_slot_ref> reused;      // served in place by a held slot of cap >= size
    std::vector<demand_slot_ref> carved;      // carved new, nothing held at that key
    std::vector<demand_slot_ref> superseded;  // carved new; `held` is the smaller slot the publish releases
    std::vector<size_t>          unused;      // held slots (indices) of a demanded owner no record names
};

// Reuse in place when the owner holds the same (cohort, index) with cap >= the
// new need; every other slot is carved new, and the old one it supersedes is
// released at the publish.  A claimed slot is never moved.  A held slot is
// `unused` when its owner has a record on its device and no record names its
// (cohort, index): it stays as held planned room (the tenant-only path) and is
// released only with its owner.  Precondition: `held` carries no two slots with
// the same (device, scope, owner, cohort, index); the registry never holds
// duplicates, and with a duplicate only the first would be reported superseded.
demand_reconciliation context_demand_reconcile(const std::vector<context_side_demand> & demands,
                                               const std::vector<held_slot> &           held);

// ---------------------------------------------------------------------------
// kv_region_fit (llama.cpp-moua §2.4.1): the pure fit of one context's KV
// region and head slots onto one device's shared-zone TLSFs.
//
// It reads a copy of the geometry (taken under the group mutex, §2.3.1) and a
// request, runs no allocator and touches no device, and its offsets are the
// carve's: a top carve below an anchor takes (end - size), and a gap smaller
// than size + MIN_BLOCK_SIZE is taken whole, exactly as tlsf_allocator's
// carve_gap does (§2.4.1 "Carve mirroring").
// ---------------------------------------------------------------------------

// One block of a TLSF, as the snapshot copies it out.  Every size and offset is
// in bytes within that TLSF's region.
struct zone_block {
    size_t offset          = 0;
    size_t size            = 0;
    bool   free            = false;  // an unallocated block
    bool   optional_tenant = false;  // an allocated block tagged SHARED_ZONE_TAG_OPTIONAL
    bool   yieldable       = false;  // optional_layout_yieldable_locked's answer (jehw); read only for optional_tenant
};

struct zone_range {
    size_t offset = 0;
    size_t size   = 0;
};

// A range some transaction has planned but not yet carved (§2.3.1).  The fit
// treats it as allocated.  `first_context` marks the model's FIRST_CONTEXT
// reservation (§2.3.3 A1), which the model's first context counts as free room.
struct zone_pending_range {
    size_t offset        = 0;
    size_t size          = 0;
    bool   first_context = false;
};

// A reserved slot already on the TLSF (§2.3.2).  It is an allocated block the
// fit never moves or frees; its only use is reuse in place.
struct zone_reservation {
    demand_scope scope  = demand_scope::CONTEXT;
    uint64_t     owner  = 0;
    const char * cohort = nullptr;
    uint32_t     index  = 0;
    size_t       offset = 0;
    size_t       size   = 0;
};

// One TLSF of the device's shared zone.
struct tlsf_geometry {
    // May this TLSF hold KV, and may it hold a head slot whose request names
    // WEIGHT (§2.4.1 N-chunk routing; H6): the last weight chunk takes the
    // WEIGHT-naming slots, the KV TLSF takes KV and the KV-naming slots.
    bool takes_kv           = true;
    bool takes_weight_named = true;

    // frontier_walk(anchor, SHARED_ZONE_TAG_OPTIONAL), highest block first: the
    // gap under the context side's lowest block, then the optional tenants and
    // free blocks below it up to the first block that is neither.  An optional
    // tenant that is not yieldable ends the ladder (§2.4.1 "Inputs").
    std::vector<zone_block> frontier;

    // The weight side outside the frontier walk, one run per maximal
    // low-to-high stretch of free blocks and lease-free optional tenants (the
    // whole-TLSF census of §2.4.1).  Its free blocks are the weight holes
    // (tier 3) and its optional tenants the buried ones (tier 4).
    std::vector<std::vector<zone_block>> side_runs;

    // RETAINED context-side runs: allocated to no live context, taken whole
    // with no carve (tier 1).
    std::vector<zone_range> retained_runs;

    std::vector<zone_pending_range> pending_ranges;  // other transactions', treated as allocated
    std::vector<zone_range>         own_ranges;      // this call's REGION ranges; used by a commit re-fit
    std::vector<zone_reservation>   reservations;
};

struct shared_zone_geometry {
    std::vector<tlsf_geometry> tlsfs;  // the device's shared-zone TLSFs, in zone order
};

enum kv_slot_group : uint8_t { KV_SLOT_FULL = 0, KV_SLOT_SWA = 1 };

// One KV layer's slot request (§2.4.1 "The request").  A layer's slot is
// kv_layer_alloc_bytes(kv_bytes) plus, with the persistent packed-K sidecar,
// kv_layer_alloc_bytes(sidecar_bytes): the sidecar is a slice of the same slot
// at sidecar_offset = kv_layer_alloc_bytes(kv_bytes).
struct kv_layer_slot_request {
    uint32_t      layer         = 0;
    kv_slot_group group         = KV_SLOT_FULL;
    size_t        kv_bytes      = 0;  // kv_layer_tensor_bytes over the published shape (§2.4.4)
    size_t        sidecar_bytes = 0;  // 0: no companion slot
};

// A head slot (§2.4.1): an indexed slot of a demand record, mandatory, placed
// before any KV and in tiers 1-3 only.
struct kv_head_slot_request {
    demand_scope scope        = demand_scope::CONTEXT;
    uint64_t     owner        = 0;
    const char * cohort       = nullptr;
    uint32_t     index        = 0;
    size_t       size         = 0;      // the planned size, before the 512 B slot rounding
    bool         names_weight = false;  // its zone request names WEIGHT: it goes to a takes_weight_named TLSF
};

// A device-resident attention layer's charge placed after the KV extents
// (23mk's oneDNN Graph scratch, rulings §M54): admitted in ascending order of
// size in tiers 1-3, declined when it does not fit; never a head slot, a
// demotion cause or a refusal.
struct kv_after_kv_request {
    uint32_t layer = 0;
    size_t   size  = 0;
};

// A KV slot this call already carved earlier in the same transaction (§2.4.1
// self_extents).  Its bytes are allocated in the geometry; the fit counts the
// layer device-resident, never re-places or demotes it, and copies it into the
// result.
struct kv_self_slot {
    uint32_t layer       = 0;
    size_t   slot_offset = 0;
    size_t   size        = 0;
};

struct kv_self_extent {
    size_t                    tlsf   = 0;
    size_t                    offset = 0;
    size_t                    size   = 0;
    std::vector<kv_self_slot> slots;
};

// A head slot this call already carved earlier in the transaction.
struct kv_self_head {
    size_t head   = 0;  // index into kv_region_request::head_slots
    size_t tlsf   = 0;
    size_t offset = 0;
    size_t size   = 0;
};

struct kv_region_fit_result;

struct kv_region_request {
    std::vector<kv_layer_slot_request> layers;  // any order; the fit orders them full first, then SWA, each by layer
    std::vector<kv_head_slot_request>  head_slots;   // placed first, in this order
    std::vector<kv_after_kv_request>   after_kv;
    std::vector<uint32_t>              forced_host;  // layers an earlier fit demoted; never promoted
    std::vector<kv_self_extent>        self_extents;
    std::vector<kv_self_head>          self_heads;
    bool                               first_context = false;  // the model's FIRST_CONTEXT ranges count as free room

    // §2.4.2 step 6, the commit re-fit.  Set, the call does not solve: it checks the
    // plan it points at against `geometry` (inside own_ranges only, with no yield) and
    // returns the plan itself when every planned extent is still free.  A planned
    // layer extent that is not free demotes that layer, every survivor keeping its
    // planned offset; a planned head slot that is not free is re-placed by best fit
    // in the room the survivors leave, demoting layers in the demotion order until it
    // fits.  It never adds a layer, admits a term the plan declined or moves anything
    // else, so what it returns is a subset of the plan at the plan's offsets.
    // The plan must be the result of a fit of this same request (same layers, head
    // slots and after-KV terms): anything else is misuse and aborts.  The pointer is
    // not owned and must outlive the call.
    const kv_region_fit_result * commit_plan = nullptr;
};

enum kv_demotion_cause : uint8_t {
    KV_DEMOTE_NONE        = 0,
    KV_DEMOTE_FORCED_HOST = 1,  // inherited from forced_host
    KV_DEMOTE_RING_GROWTH = 2,  // would have stayed on the device with the superseded slots counted free
    KV_DEMOTE_HEAD_SLOT   = 3,  // would have stayed on the device with this request's head slots removed
    KV_DEMOTE_CAPACITY    = 4,
};

enum kv_extent_kind : uint8_t {
    KV_EXTENT_FRONTIER = 0,  // the gap, grown by the yield prefix
    KV_EXTENT_RETAINED = 1,
    KV_EXTENT_HOLE     = 2,  // a weight hole, or the run formed by releasing buried optional tenants
    KV_EXTENT_SELF     = 3,
};

struct kv_region_extent {
    size_t         tlsf   = 0;
    size_t         offset = 0;
    size_t         size   = 0;
    kv_extent_kind kind   = KV_EXTENT_FRONTIER;
};

struct kv_layer_placement {
    uint32_t          layer          = 0;
    bool              device         = false;
    kv_demotion_cause cause          = KV_DEMOTE_NONE;  // set for a host layer
    size_t            extent         = SIZE_MAX;        // index into kv_region_fit_result::extents
    size_t            slot_offset    = 0;               // from the extent's base
    size_t            size           = 0;
    size_t            sidecar_offset = SIZE_MAX;        // from the extent's base; SIZE_MAX: no sidecar
};

struct kv_head_placement {
    size_t head        = 0;         // index into kv_region_request::head_slots
    bool   reused      = false;     // served in place by a reserved slot; nothing is carved
    size_t reservation = SIZE_MAX;  // index into the TLSF's reservations when reused
    size_t tlsf        = 0;
    size_t offset      = 0;
    size_t size        = 0;
};

struct kv_after_kv_placement {
    size_t term     = 0;  // index into kv_region_request::after_kv
    bool   admitted = false;
    size_t tlsf     = 0;
    size_t offset   = 0;
    size_t size     = 0;
};

struct kv_superseded_slot {
    size_t tlsf        = 0;
    size_t reservation = 0;  // index into that TLSF's reservations; the publish releases it
};

struct kv_buried_release {
    size_t tlsf   = 0;
    size_t offset = 0;  // a lease-free optional tenant the fit releases ahead of the carve
};

enum kv_carve_kind : uint8_t { KV_CARVE_HEAD = 0, KV_CARVE_EXTENT = 1, KV_CARVE_AFTER_KV = 2 };

// One step of the commit's carve, in the order the fit placed them.  `carve` is
// false when nothing goes through the allocator: a retained run is claimed, and
// an after-KV charge is only recorded as a range.
struct kv_carve_op {
    kv_carve_kind kind   = KV_CARVE_HEAD;
    size_t        index  = 0;  // head index, extent index or after-KV term
    size_t        tlsf   = 0;
    size_t        offset = 0;  // the block's base: what the carve returns
    size_t        size   = 0;  // the block's size, which a whole-gap take makes larger than `demand`
    size_t        demand = 0;  // the size asked of the allocator
    bool          carve  = true;
};

// A fit's result is the plan.  A commit re-fit's result (kv_region_request::commit_plan)
// is an assignment only: layers, extents, heads, after_kv, carve_order and refused_heads
// say what the commit carves, while free_after_full_kv, sub_slot_holes and superseded are
// empty (they describe the room and the reservations of the plan's own geometry, which a
// shortfall has changed; the plan's stay with the plan), yield_prefix and buried_released
// are zero and empty (a re-fit yields nothing), and tlsf_free is set only on a refusal.
struct kv_region_fit_result {
    bool fits = false;

    // Every requested layer, in slot-table order (full attention first, then
    // SWA, each by layer index), with its residency after the demotion loop.
    std::vector<kv_layer_placement>    layers;
    std::vector<kv_region_extent>      extents;
    std::vector<kv_head_placement>     heads;
    std::vector<kv_after_kv_placement> after_kv;
    std::vector<kv_superseded_slot>    superseded;

    // The strict LIFO yield prefix per TLSF: how many optional tenants the
    // frontier gives up, highest first; and the buried tenants released.
    std::vector<size_t>            yield_prefix;
    std::vector<kv_buried_release> buried_released;

    // free_after_full_kv (zhcn GA): per TLSF, the room the fit can place into
    // after the head slots (free, retained and yieldable bytes) minus every
    // requested KV slot with every KV layer device-resident, sidecars included.
    // A KV slot that no tier can place is charged to the last takes_kv TLSF.
    // Signed: a negative value is the deficit the demotion loop covers.
    std::vector<long long> free_after_full_kv;

    std::vector<kv_carve_op> carve_order;

    // Weight-side holes smaller than the smallest slot the loop demoted, listed when it
    // demoted: they are counted as room but no demoted layer fits one (§2.9 sub-slot WARN).
    std::vector<kv_buried_release> sub_slot_holes;

    // On a refusal: the head slots that did not fit (indices into
    // head_slots) and each TLSF's free bytes in tiers 1-3.
    std::vector<size_t> refused_heads;
    std::vector<size_t> tlsf_free;
};

kv_region_fit_result kv_region_fit(const shared_zone_geometry & geometry, const kv_region_request & request);

}  // namespace ggml_sycl
