#include "kv-tier-manager.hpp"

#include "ggml-impl.h"
#include "unified-cache.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <vector>

namespace ggml_sycl {

size_t unified_cache_get_layer_vram_bytes(int device, int layer_id);

static std::array<kv_tier_manager, GGML_SYCL_MAX_DEVICES> g_kv_tier_managers;

// Per-layer region alignment used by compute_region_layout(); also the
// per-layer slack a buffer may carry over its layers' raw byte sum (each K/V
// tensor is padded to the buffer type's alignment, which is smaller).
static constexpr size_t kv_layer_align_bytes = 512;

kv_tier_manager & get_kv_tier_manager(int device) {
    return g_kv_tier_managers[device];
}

bool kv_tier_manager::configure(int device, uint32_t n_layers, size_t kv_vram_cap, const kv_slice_size & slice) {
    device_       = device;
    total_layers_ = n_layers;

    if (n_layers == 0) {
        active_       = false;
        hot_layers_   = 0;
        kv_per_layer_ = 0;
        return false;
    }

    kv_per_layer_ = slice.bytes();

    // Check env var override: GGML_SYCL_KV_HOT_LAYERS=N
    const char * env = std::getenv("GGML_SYCL_KV_HOT_LAYERS");
    int          val = env ? std::atoi(env) : -1;
    if (val >= 0) {
        hot_layers_ = static_cast<uint32_t>(std::min(val, static_cast<int>(n_layers)));
    } else if (kv_vram_cap == 0 || kv_per_layer_ == 0) {
        hot_layers_ = 0;
    } else {
        hot_layers_ = static_cast<uint32_t>(std::min(static_cast<size_t>(n_layers), kv_vram_cap / kv_per_layer_));
    }

    if (hot_layers_ >= total_layers_) {
        active_ = false;
        return false;
    }

    active_ = true;
    return true;
}

bool kv_tier_manager::is_hot(uint32_t layer_id) const {
    if (!active_) {
        return true;  // All hot when tiering inactive
    }
    if (layer_id < layer_on_device_.size()) {
        return layer_on_device_[layer_id];
    }
    // Fallback for budget-only configure() path (no per-layer vector)
    return layer_id < hot_layers_;
}

void kv_tier_manager::set_actual_hot_layers(uint32_t n_hot) {
    if (n_hot > total_layers_) {
        n_hot = total_layers_;
    }
    hot_layers_ = n_hot;
    active_     = (hot_layers_ < total_layers_);

    // Sync per-layer vector if it was populated by configure_with_weights().
    // The allocation retry loop may have reduced the device budget, so trim
    // device-placed layers from the back until the count matches n_hot.
    if (!layer_on_device_.empty()) {
        uint32_t current_on_device = 0;
        for (bool v : layer_on_device_) {
            if (v) {
                current_on_device++;
            }
        }
        // Remove device-placed layers from the back until count matches
        for (uint32_t l = total_layers_; l > 0 && current_on_device > n_hot; l--) {
            if (layer_on_device_[l - 1]) {
                layer_on_device_[l - 1] = false;
                current_on_device--;
            }
        }
    }
}

void kv_tier_manager::set_actual_layer_placement(int device, const std::vector<bool> & layer_on_device) {
    device_          = device;
    total_layers_    = static_cast<uint32_t>(layer_on_device.size());
    // kv_per_layer_ and per_layer_kv_bytes_ are intentionally left alone: this
    // reconciles *placement* after allocation, and the slice sizes were already
    // established by the configure_*() call that preceded it.
    layer_on_device_ = layer_on_device;
    hot_layers_      = 0;
    for (bool on_device : layer_on_device_) {
        if (on_device) {
            hot_layers_++;
        }
    }
    active_ = (hot_layers_ < total_layers_);
}

void kv_tier_manager::get_region_sizes(size_t total_bytes, size_t & hot_bytes, size_t & cold_bytes) const {
    if (!active_ || total_layers_ == 0) {
        hot_bytes  = total_bytes;
        cold_bytes = 0;
        return;
    }
    // Use per-layer vector when available (non-contiguous / heterogeneous support)
    if (!layer_on_device_.empty()) {
        size_t sum_device = 0;
        size_t sum_host   = 0;
        for (uint32_t l = 0; l < total_layers_; l++) {
            const size_t layer_sz = kv_layer_size(l);
            if (l < layer_on_device_.size() && layer_on_device_[l]) {
                sum_device += layer_sz;
            } else {
                sum_host += layer_sz;
            }
        }
        hot_bytes  = std::min(sum_device, total_bytes);
        cold_bytes = total_bytes > hot_bytes ? total_bytes - hot_bytes : 0;
        return;
    }
    // Fallback: contiguous hot_layers_ count with uniform size
    hot_bytes  = std::min(static_cast<size_t>(hot_layers_) * kv_per_layer_, total_bytes);
    cold_bytes = total_bytes - hot_bytes;
}

std::vector<layer_region> kv_tier_manager::compute_region_layout(size_t total_bytes) const {
    GGML_UNUSED(total_bytes);

    std::vector<layer_region> regions(total_layers_);
    size_t                    device_offset = 0;
    size_t                    host_offset   = 0;
    for (uint32_t l = 0; l < total_layers_; l++) {
        const size_t raw_size     = kv_layer_size(l);
        const size_t aligned_size = (raw_size + kv_layer_align_bytes - 1) & ~(kv_layer_align_bytes - 1);
        regions[l].layer_id  = l;
        regions[l].size      = aligned_size;
        regions[l].on_device = is_hot(l);
        if (regions[l].on_device) {
            regions[l].offset = device_offset;
            device_offset += aligned_size;
        } else {
            regions[l].offset = host_offset;
            host_offset += aligned_size;
        }
    }
    return regions;
}

void kv_tier_manager::configure_with_weights(int                   device,
                                             uint32_t              n_layers,
                                             size_t                kv_vram_cap,
                                             const kv_slice_size & slice) {
    device_       = device;
    total_layers_ = n_layers;

    if (n_layers == 0 || slice.total_bytes() == 0) {
        active_       = false;
        hot_layers_   = 0;
        kv_per_layer_ = 0;
        layer_on_device_.clear();
        return;
    }

    kv_per_layer_ = slice.bytes();

    // PHASE 1: Query unified cache for per-layer weight residency
    std::vector<bool> layer_has_device_weights(n_layers, false);
    int               device_weight_count = 0;
    for (uint32_t l = 0; l < n_layers; l++) {
        if (unified_cache_get_layer_vram_bytes(device, static_cast<int>(l)) > 0) {
            layer_has_device_weights[l] = true;
            device_weight_count++;
        }
    }

    // Fall back to budget-based configure() when cache has no layer data.
    // Budget-only path uses contiguous hot_layers_ without per-layer vector.
    if (device_weight_count == 0) {
        layer_on_device_.clear();
        configure(device, n_layers, kv_vram_cap, slice);
        return;
    }

    // PHASE 2: Check env var override — takes precedence over weight residency
    const char * hot_layers_env = std::getenv("GGML_SYCL_KV_HOT_LAYERS");
    if (hot_layers_env) {
        int val = std::atoi(hot_layers_env);
        if (val >= 0) {
            // Env override: use contiguous placement (layers 0..N-1)
            hot_layers_ = std::min(static_cast<uint32_t>(val), n_layers);
            layer_on_device_.assign(n_layers, false);
            for (uint32_t l = 0; l < hot_layers_; l++) {
                layer_on_device_[l] = true;
            }
            active_ = (hot_layers_ < total_layers_);

            GGML_LOG_INFO(
                "[KV-TIER] Weight-aware (env override): %u/%u layers on device "
                "(%.1f MB device, %.1f MB host)\n",
                hot_layers_, total_layers_, (hot_layers_ * kv_per_layer_) / (1024.0 * 1024.0),
                ((total_layers_ - hot_layers_) * kv_per_layer_) / (1024.0 * 1024.0));
            return;
        }
    }

    // PHASE 3: Per-layer placement — co-locate KV with device-resident weights
    // within the VRAM budget.  Any layer with device weights gets placed on
    // device if the budget allows; this supports non-contiguous placement.
    layer_on_device_.assign(n_layers, false);
    size_t vram_used = 0;
    for (uint32_t l = 0; l < n_layers; l++) {
        if (layer_has_device_weights[l] && vram_used + kv_per_layer_ <= kv_vram_cap) {
            layer_on_device_[l] = true;
            vram_used += kv_per_layer_;
        }
    }

    // Update hot_layers_ for backward-compatible logging and fallback paths
    hot_layers_ = 0;
    for (bool on_dev : layer_on_device_) {
        if (on_dev) {
            hot_layers_++;
        }
    }

    active_ = (hot_layers_ < total_layers_);

    const size_t dev_bytes  = std::min(static_cast<size_t>(hot_layers_) * kv_per_layer_, slice.total_bytes());
    const size_t host_bytes = slice.total_bytes() - dev_bytes;
    GGML_LOG_INFO(
        "[KV-TIER] Weight-aware: %u/%u layers on device (%d with device weights, "
        "%u KV layers, %.1f MB device, %.1f MB host)\n",
        hot_layers_, total_layers_, device_weight_count, slice.kv_layers(), dev_bytes / (1024.0 * 1024.0),
        host_bytes / (1024.0 * 1024.0));
}

void kv_tier_manager::configure_from_plan(int                          device,
                                          const placement_plan &       plan,
                                          uint32_t                     n_layers,
                                          const kv_slice_size &        slice,
                                          const std::vector<uint8_t> * buffer_layer_mask) {
    device_       = device;
    total_layers_ = n_layers;

    if (n_layers == 0 || slice.total_bytes() == 0) {
        active_       = false;
        hot_layers_   = 0;
        kv_per_layer_ = 0;
        layer_on_device_.clear();
        return;
    }

    // Ranking, strongest first.  Not all layers hold KV (alternating SWA,
    // hybrid attention/recurrent), so any denominator that includes
    // non-attention layers undercounts and the cache is written out of bounds.
    //
    // 1. A slice derived from llama's explicit per-buffer layer mask wins
    //    outright: it describes *this* buffer, whereas plan is a file-scope
    //    global that can still hold a previous model's numbers.
    // 2. Otherwise the planner's own per-layer size, if it is sane.
    // 3. Otherwise whatever the slice carries (never a fresh division here --
    //    see kv_slice_size).
    if (slice.source() == kv_slice_source::LAYER_MASK && slice.bytes() > 0) {
        kv_per_layer_ = slice.bytes();
    } else if (plan.kv_per_layer > 0 && slice.total_bytes() >= plan.kv_per_layer) {
        kv_per_layer_ = plan.kv_per_layer;
    } else {
        kv_per_layer_ = slice.bytes();
    }

    // Explicit debug override remains higher priority than planned placement.
    const char * hot_layers_env = std::getenv("GGML_SYCL_KV_HOT_LAYERS");
    if (hot_layers_env) {
        int val = std::atoi(hot_layers_env);
        if (val >= 0) {
            hot_layers_ = std::min(static_cast<uint32_t>(val), n_layers);
            layer_on_device_.assign(n_layers, false);
            for (uint32_t l = 0; l < hot_layers_; l++) {
                layer_on_device_[l] = true;
            }
            active_ = (hot_layers_ < total_layers_);
            const size_t dev_bytes  = std::min(static_cast<size_t>(hot_layers_) * kv_per_layer_, slice.total_bytes());
            const size_t host_bytes = slice.total_bytes() - dev_bytes;
            GGML_LOG_INFO(
                "[KV-TIER] Plan-driven (env override): %u/%u layers on device "
                "(%.1f MB device, %.1f MB host)\n",
                hot_layers_, total_layers_, dev_bytes / (1024.0 * 1024.0), host_bytes / (1024.0 * 1024.0));
            return;
        }
    }

    // Build per-layer placement.
    layer_on_device_.assign(n_layers, false);
    // assign(), not resize(): kv_tier_manager is a per-device singleton reused
    // across models, and resize() would leave a shorter previous model's
    // per-layer sizes in place.
    per_layer_kv_bytes_.assign(n_layers, kv_per_layer_);
    hot_layers_ = 0;
    for (uint32_t l = 0; l < n_layers; ++l) {
        const bool on_device = plan.get_kv_device(static_cast<int>(l)) == device;
        layer_on_device_[l]  = on_device;
        if (on_device) {
            hot_layers_++;
        }
    }

    active_ = (hot_layers_ < total_layers_);

    // Per-layer KV sizes (llama.cpp-7yv9).  The plan's per-layer truth --
    // placement_plan::kv_size_for_layer(): FULL / SWA / SHARED kind and that
    // layer's own K/V width -- is what every other consumer of the KV budget
    // sizes from; sizing here from the uniform scalars instead gave every SWA
    // layer of a Gemma 4 E4B buffer plan.kv_per_swa_layer (computed at the
    // global full-attention width) and reserved 2x its real bytes.
    //
    // The buffer is the ground truth about what llama allocated, so the
    // truth is used only when it accounts for this buffer's layers: the sum
    // over the buffer's members must match slice.total_bytes() up to
    // per-layer alignment slack.  A sum that exceeds the buffer (a plan whose
    // fp16 formula does not match a quantized KV cache, or a stale plan) or
    // under-sizes it (which would under-allocate layers and overflow --
    // "[KV-REMAP] ERROR: ... overflows layer alloc!") is refused with a WARN
    // naming both numbers, and the slice -- exact by construction for a
    // homogeneous buffer -- sizes the layers as before.  Never a silent clamp.
    const auto in_buffer = [&](uint32_t l) {
        return buffer_layer_mask == nullptr || (l < buffer_layer_mask->size() && (*buffer_layer_mask)[l] != 0);
    };
    bool     truth_covers_buffer = true;
    size_t   truth_sum           = 0;
    uint32_t buffer_layers       = 0;
    for (uint32_t l = 0; l < n_layers; ++l) {
        if (!in_buffer(l)) {
            continue;
        }
        buffer_layers++;
        if (!plan.has_per_layer_kv_truth(l)) {
            truth_covers_buffer = false;
            break;
        }
        truth_sum += plan.kv_size_for_layer(l);
    }

    bool use_truth     = truth_covers_buffer && buffer_layers > 0 && truth_sum > 0;
    bool truth_refused = false;
    if (use_truth) {
        const size_t total = slice.total_bytes();
        const size_t slack = static_cast<size_t>(buffer_layers) * kv_layer_align_bytes;
        if (truth_sum > total) {
            GGML_LOG_WARN(
                "[KV-TIER] per-layer KV truth sum %zu B exceeds this buffer's %zu B (%u layers); "
                "sizing every layer from the slice (%zu B/layer) instead (llama.cpp-7yv9)\n",
                truth_sum, total, buffer_layers, kv_per_layer_);
            truth_refused = true;
        } else if (total - truth_sum > slack) {
            GGML_LOG_WARN(
                "[KV-TIER] per-layer KV truth sum %zu B under-sizes this buffer's %zu B by %zu B (> %u layers x %zu B "
                "alignment slack); sizing every layer from the slice (%zu B/layer) instead (llama.cpp-7yv9)\n",
                truth_sum, total, total - truth_sum, buffer_layers, kv_layer_align_bytes, kv_per_layer_);
            truth_refused = true;
        }
        use_truth = !truth_refused;
    }

    if (use_truth) {
        // Every layer with truth, not only this buffer's: kv_layer_size(l) is
        // then right for any layer asked about, and the caller zeroes the
        // regions of layers outside the buffer before allocating.
        for (uint32_t l = 0; l < n_layers; ++l) {
            if (plan.has_per_layer_kv_truth(l)) {
                per_layer_kv_bytes_[l] = plan.kv_size_for_layer(l);
            }
        }
    } else if (truth_refused) {
        // The plan's KV arithmetic does not describe this buffer, so its
        // other scalar (kv_per_swa_layer, same formula) is not trusted either:
        // every layer keeps the ranked slice already assigned above, which is
        // derived from the buffer llama actually allocated.
    } else if (plan.kv_per_swa_layer > 0 && !plan.swa_layer_mask.empty()) {
        // Legacy uniform-per-class split (no per-layer truth on the plan):
        // full-attention layers use the ranked kv_per_layer_, SWA layers
        // plan.kv_per_swa_layer.
        for (uint32_t l = 0; l < n_layers; ++l) {
            const bool is_swa      = l < plan.swa_layer_mask.size() && plan.swa_layer_mask[l];
            per_layer_kv_bytes_[l] = is_swa ? plan.kv_per_swa_layer : kv_per_layer_;
        }
    }

    // Byte totals for this buffer's layers, using the per-layer sizes.
    size_t dev_bytes  = 0;
    size_t host_bytes = 0;
    for (uint32_t l = 0; l < n_layers; ++l) {
        if (!in_buffer(l)) {
            continue;
        }
        if (layer_on_device_[l]) {
            dev_bytes += per_layer_kv_bytes_[l];
        } else {
            host_bytes += per_layer_kv_bytes_[l];
        }
    }
    // kv_layers/kv_per_layer are printed here on purpose: the sizing bug this
    // class guards against was invisible because the only logged per-layer
    // figure came from a value the allocator never used (llama.cpp-2120).
    // sizing= says which of the two paths above produced the per-layer sizes.
    GGML_LOG_INFO(
        "[KV-TIER] Plan-driven: %u/%u layers on device "
        "(planner_n_ctx=%u, kv_layers=%u, kv_per_layer=%zu, sizing=%s, %.1f MB device, %.1f MB host)\n",
        hot_layers_, total_layers_, plan.planner_n_ctx, slice.kv_layers(), kv_per_layer_,
        use_truth ? "per-layer" : "uniform", dev_bytes / (1024.0 * 1024.0), host_bytes / (1024.0 * 1024.0));
}

}  // namespace ggml_sycl
