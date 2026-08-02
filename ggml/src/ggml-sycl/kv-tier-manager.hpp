#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Forward-declare GGML_SYCL_MAX_DEVICES from ggml-sycl.h
#ifndef GGML_SYCL_MAX_DEVICES
#    define GGML_SYCL_MAX_DEVICES 48
#endif

namespace ggml_sycl {

struct placement_plan;

// Where a kv_slice_size's denominator came from.  Ranked: a mask pushed by
// llama_kv_cache states exactly which layers this buffer holds, so it outranks
// anything the placement planner inferred.
enum class kv_slice_source {
    UNKNOWN,                 // default-constructed; bytes() == 0
    LAYER_MASK,              // llama's explicit per-buffer layer mask (authoritative)
    PLANNER,                 // planner KV geometry (full-attn / SWA layer counts)
    MODEL_LAYERS_UNVERIFIED  // no mask, no planner geometry -- every layer assumed to hold KV
};

// Size of one layer's KV slice within a KV buffer.
//
// This type exists so that `total_bytes / n_layers` is not expressible where
// the slice size is set.  Only the layers that actually hold KV contribute to
// a KV buffer, so the model's *total* layer count is the wrong denominator: a
// hybrid model with 1 attention layer of 2 gets a 2x-undersized slice, and the
// KV cache is then written past its allocation (llama.cpp-2120 -- 10
// architectures ran with out-of-bounds KV pointers and 7 of them reported OK).
//
// The constructor is private; every slice is built through a named factory
// that states its denominator, and kv-tier-manager.cpp divides total_bytes by
// nothing at all.  Adding a new sizing path therefore means adding a factory
// here, in front of this comment, rather than another bare division at a call
// site.
class kv_slice_size {
  public:
    kv_slice_size() = default;

    // Divide the buffer by the layers llama said this buffer holds.
    static kv_slice_size from_layer_mask(size_t total_bytes, uint32_t n_kv_layers) {
        return kv_slice_size(total_bytes, n_kv_layers, kv_slice_source::LAYER_MASK);
    }

    // Take the planner's per-layer byte size directly (not a division).
    static kv_slice_size from_planner_bytes(size_t total_bytes, size_t bytes_per_layer, uint32_t n_kv_layers) {
        kv_slice_size s;
        s.total_bytes_ = total_bytes;
        s.bytes_       = bytes_per_layer;
        s.n_kv_layers_ = n_kv_layers;
        s.source_      = kv_slice_source::PLANNER;
        return s;
    }

    // Divide the buffer by a KV-bearing layer count derived from planner geometry.
    static kv_slice_size from_planner_layers(size_t total_bytes, uint32_t n_kv_layers) {
        return kv_slice_size(total_bytes, n_kv_layers, kv_slice_source::PLANNER);
    }

    // Last resort: no layer mask and no planner KV geometry, so every model
    // layer is *assumed* to hold KV.  This is the only legitimate use of the
    // model layer count as a denominator, and the name says it is unverified.
    static kv_slice_size from_model_layers_unverified(size_t total_bytes, uint32_t n_model_layers) {
        return kv_slice_size(total_bytes, n_model_layers, kv_slice_source::MODEL_LAYERS_UNVERIFIED);
    }

    size_t bytes() const { return bytes_; }

    uint32_t kv_layers() const { return n_kv_layers_; }

    size_t total_bytes() const { return total_bytes_; }

    kv_slice_source source() const { return source_; }

  private:
    // A zero KV-layer count means "nothing told us how this buffer is split",
    // so the whole buffer is one slice.  Never zero bytes: a zero slice makes
    // every layer allocation get skipped, which is worse than one big slice.
    kv_slice_size(size_t total_bytes, uint32_t n_kv_layers, kv_slice_source source) :
        bytes_(n_kv_layers > 0 ? total_bytes / n_kv_layers : total_bytes),
        total_bytes_(total_bytes),
        n_kv_layers_(n_kv_layers),
        source_(source) {}

    size_t          bytes_       = 0;
    size_t          total_bytes_ = 0;
    uint32_t        n_kv_layers_ = 0;
    kv_slice_source source_      = kv_slice_source::UNKNOWN;
};

// Per-layer region descriptor for KV cache placement.
struct layer_region {
    uint32_t layer_id;
    size_t   offset;     // Offset within device or host region
    size_t   size;       // Bytes for this layer's KV
    bool     on_device;  // true = VRAM, false = host-pinned
};

// Manages hot/cold tiering for KV cache memory on a per-layer basis.
// Hot layers: KV on device VRAM (fast GPU access, co-located with attention weights)
// Cold layers: KV on pinned host memory (PCIe access via USM)
class kv_tier_manager {
  public:
    kv_tier_manager() = default;

    // Configure the layer-based tier split for a device.
    // n_layers: total number of transformer layers (indexes the placement vectors)
    // kv_vram_cap: VRAM bytes available for KV cache
    // slice: per-layer KV byte size -- see kv_slice_size on why this is not a
    //        (total_bytes, n_layers) pair
    // Returns true if tiering is active (some layers on host)
    bool configure(int device, uint32_t n_layers, size_t kv_vram_cap, const kv_slice_size & slice);

    // Weight-aware configuration: queries unified cache for per-layer weight
    // residency and co-locates KV with device-resident weights.
    // Populates per-layer placement vector for non-contiguous placement.
    // Falls back to budget-based configure() when cache data is unavailable.
    void configure_with_weights(int device, uint32_t n_layers, size_t kv_vram_cap, const kv_slice_size & slice);

    // Plan-driven configuration: use authoritative planner KV residency.
    // The slice still wins over plan.kv_per_layer when it came from llama's
    // explicit layer mask -- the plan is a file-scope global and can carry a
    // stale cross-model value, whereas the mask describes *this* buffer.
    void configure_from_plan(int device, const placement_plan & plan, uint32_t n_layers, const kv_slice_size & slice);

    // Query tier state
    bool is_active() const { return active_; }

    uint32_t hot_layers() const { return hot_layers_; }

    uint32_t total_layers() const { return total_layers_; }

    int device_id() const { return device_; }

    size_t kv_per_layer() const { return kv_per_layer_; }

    // Per-layer KV byte size.  Returns the heterogeneous per-layer size when
    // configure_from_plan() populated per_layer_kv_bytes_; falls back to the
    // uniform kv_per_layer_ otherwise.
    size_t kv_layer_size(uint32_t layer_id) const {
        if (layer_id < per_layer_kv_bytes_.size()) {
            return per_layer_kv_bytes_[layer_id];
        }
        return kv_per_layer_;
    }

    // Returns true if the given layer should be placed in device VRAM (hot).
    // Supports non-contiguous placement when configure_with_weights() was used.
    bool is_hot(uint32_t layer_id) const;

    // Get hot/cold byte sizes for a given total buffer size.
    // hot_bytes: bytes for device-placed layers, cold_bytes: for host-placed layers.
    // Supports non-contiguous placement (sums device vs host layers).
    void get_region_sizes(size_t total_bytes, size_t & hot_bytes, size_t & cold_bytes) const;

    // Compute per-layer region layout with offsets within device/host regions.
    // Layer sizes are aligned to 512 bytes.
    std::vector<layer_region> compute_region_layout(size_t total_bytes) const;

    // Return the per-layer placement vector.
    // Empty when only budget-based configure() was used.
    const std::vector<bool> & get_layer_placement() const { return layer_on_device_; }

    // Return the number of device-placed layers.
    uint32_t get_device_layer_count() const { return hot_layers_; }

    // Override the hot layer count after allocation.  Called when device
    // allocation fails and the retry loop settles on fewer hot layers than
    // configure() initially computed (e.g. due to VRAM fragmentation or
    // env-var override that exceeds actual capacity).
    void set_actual_hot_layers(uint32_t n_hot);

    // Replace placement with the actual per-layer allocation result.
    // Deliberately does NOT touch the slice size: allocation reconciliation
    // moves layers between tiers, it does not resize them, and re-deriving the
    // slice from the layer-vector length is exactly the wrong-denominator bug
    // this class is shaped to prevent (llama.cpp-2120).
    void set_actual_layer_placement(int device, const std::vector<bool> & layer_on_device);

  private:
    bool              active_       = false;
    int               device_       = -1;
    uint32_t          hot_layers_   = 0;
    uint32_t          total_layers_ = 0;
    size_t            kv_per_layer_ = 0;
    std::vector<bool>   layer_on_device_;    // Per-layer: true = VRAM, false = host
    std::vector<size_t> per_layer_kv_bytes_; // Per-layer KV byte size (heterogeneous)
};

// Per-device singleton accessor
kv_tier_manager & get_kv_tier_manager(int device);

}  // namespace ggml_sycl
