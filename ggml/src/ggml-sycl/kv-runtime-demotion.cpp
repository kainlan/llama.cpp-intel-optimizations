#include "kv-runtime-demotion.hpp"

#include <algorithm>
#include <cstdint>

namespace ggml_sycl {

kv_demotion_result plan_runtime_kv_demotion(const kv_demotion_input & in) {
    kv_demotion_result r;
    r.vram_bytes_after = in.vram_bytes;

    if (in.vram_bytes <= in.vram_budget) {
        r.fits = true;
        return r;
    }

    // One pass over the device-resident layers of one kind, latest first.
    const int n_layers    = (int) in.kv_device.size();
    auto      demote_pass = [&](bool swa_pass) {
        for (int l = n_layers - 1; l >= 0; --l) {
            if (in.kv_device[l] < 0) {
                continue;  // already host
            }
            const bool is_swa = l < (int) in.swa_layer_mask.size() && in.swa_layer_mask[l] != 0;
            if (is_swa != swa_pass) {
                continue;
            }
            // llama.cpp-3aos: THIS layer's own recorded bytes, not a
            // uniform figure applied to every full-attention layer -- 0 means
            // nothing is recorded for it (untracked, or a SHARED layer with no
            // independent KV to move), so there is nothing to demote.
            const size_t layer_bytes = l < (int) in.layer_kv_bytes.size() ? in.layer_kv_bytes[l] : 0;
            if (layer_bytes == 0) {
                continue;
            }
            if (layer_bytes > r.vram_bytes_after) {
                return;  // demoting would underflow vram_bytes_after; refuse rather than wrap
            }
            r.demoted_layers.push_back(l);
            r.vram_bytes_after -= layer_bytes;
            r.host_kv_bytes_added += layer_bytes;
            if (r.vram_bytes_after <= in.vram_budget) {
                return;
            }
        }
    };
    demote_pass(false);
    // SWA KV is small (~1.5 MB/layer), so demoting it buys little and costs a
    // split: only as a last resort, and only when the caller asks.
    if (r.vram_bytes_after > in.vram_budget && in.demote_swa) {
        demote_pass(true);
    }

    r.fits = r.vram_bytes_after <= in.vram_budget;
    return r;
}

size_t kv_weight_capacity(size_t vram_budget, size_t shared_zone_capacity) {
    return shared_zone_capacity > 0 ? std::min(vram_budget, shared_zone_capacity) : vram_budget;
}

kv_demotion_result plan_device_kv_fit(const kv_device_fit_input & in) {
    const size_t       n_layers = in.kv_device.size();
    kv_demotion_input  one;
    size_t             kv_bytes = 0;
    kv_demotion_result overflow;
    one.vram_budget = in.capacity;
    one.kv_device.assign(n_layers, -1);
    one.layer_kv_bytes.assign(n_layers, 0);
    one.swa_layer_mask = in.swa_layer_mask;
    one.demote_swa     = in.demote_swa;
    for (size_t l = 0; l < n_layers; ++l) {
        if (in.kv_device[l] != in.device) {
            continue;
        }
        const size_t bytes = l < in.layer_kv_bytes.size() ? in.layer_kv_bytes[l] : 0;
        if (bytes > SIZE_MAX - kv_bytes) {
            return overflow;  // fits == false: an unrepresentable total is not a fit
        }
        one.kv_device[l]      = in.device;
        one.layer_kv_bytes[l] = bytes;
        kv_bytes += bytes;
    }
    one.vram_bytes = kv_bytes;
    return plan_runtime_kv_demotion(one);
}

bool kv_shape_changed(const kv_shape & published, const kv_shape & next) {
    return !published.runtime || published.n_ctx != next.n_ctx || published.n_seq_max != next.n_seq_max ||
           published.kv_unified != next.kv_unified || published.swa_full != next.swa_full;
}

bool kv_residency_needs_refit(const kv_shape & published, const kv_shape & next, bool context_admitted) {
    return kv_shape_changed(published, next) || !context_admitted;
}

bool kv_device_residency_changed(const std::unordered_map<int, int> & load_kv_device,
                                 const std::unordered_map<int, int> & published_kv_device,
                                 const std::unordered_map<int, int> & next_kv_device,
                                 int                                  device) {
    auto owner = [](const std::unordered_map<int, int> & kv_device, int layer) {
        const auto it = kv_device.find(layer);
        return it == kv_device.end() ? -1 : it->second;
    };
    for (const auto & entry : load_kv_device) {
        if (entry.second == device && owner(published_kv_device, entry.first) != owner(next_kv_device, entry.first)) {
            return true;
        }
    }
    return false;
}

kv_residency_result plan_runtime_kv_residency(const kv_residency_input & in) {
    kv_residency_result r;
    r.kv_device = in.load_kv_device;
    kv_device_fit_input fit;
    fit.layer_kv_bytes = in.layer_kv_bytes;
    fit.swa_layer_mask = in.swa_layer_mask;
    fit.demote_swa     = true;  // never shrink context: place all of its KV rather than refuse
    for (size_t i = 0; i < in.devices.size(); ++i) {
        fit.device    = in.devices[i];
        fit.kv_device = r.kv_device;

        size_t n_resident = 0;
        size_t kv_bytes   = 0;
        for (size_t l = 0; l < r.kv_device.size(); ++l) {
            if (r.kv_device[l] == fit.device && l < in.layer_kv_bytes.size() && in.layer_kv_bytes[l] > 0) {
                ++n_resident;
                kv_bytes += std::min(in.layer_kv_bytes[l], SIZE_MAX - kv_bytes);
            }
        }
        const bool   overflow  = in.per_layer_slack > 0 && n_resident > SIZE_MAX / in.per_layer_slack;
        const size_t slack     = overflow ? SIZE_MAX : n_resident * in.per_layer_slack;
        const size_t demand    = kv_bytes + std::min(slack, SIZE_MAX - kv_bytes);
        const size_t available = i < in.available.size() ? in.available[i] : 0;
        const size_t yieldable = i < in.yieldable.size() ? in.yieldable[i] : 0;
        const size_t yield     = demand > available ? std::min(demand - available, yieldable) : 0;
        const size_t headroom  = available + std::min(yield, SIZE_MAX - available);
        fit.capacity           = headroom > slack ? headroom - slack : 0;
        if (i < in.fit_capacity.size()) {
            fit.capacity = std::min(fit.capacity, in.fit_capacity[i]);
        }
        r.yield_bytes.push_back(yield);

        const kv_demotion_result demotion = plan_device_kv_fit(fit);
        for (int l : demotion.demoted_layers) {
            r.kv_device[static_cast<size_t>(l)] = -1;
        }
        r.per_device.push_back(demotion);
        if (!demotion.fits) {
            r.fits           = false;
            r.refused_device = fit.device;
            return r;
        }
    }
    return r;
}

size_t kv_layers_allocatable(kv_zone_model zone, const std::vector<size_t> & layer_bytes) {
    size_t placed = 0;
    for (size_t bytes : layer_bytes) {
        bool fits = false;
        for (tlsf_allocator & allocator : zone) {
            if (allocator.allocate(bytes) != SIZE_MAX) {
                fits = true;
                break;
            }
        }
        if (!fits) {
            break;
        }
        ++placed;
    }
    return placed;
}

std::vector<size_t> select_optional_layout_yield(const kv_zone_model &              zone,
                                                 const std::vector<kv_zone_block> & copies,
                                                 const std::vector<size_t> &        layer_bytes) {
    std::vector<size_t> order;
    for (size_t i = 0; i < copies.size(); ++i) {
        if (copies[i].allocator < zone.size()) {
            order.push_back(i);
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return copies[a].allocator != copies[b].allocator ? copies[a].allocator > copies[b].allocator :
                                                            copies[a].offset > copies[b].offset;
    });
    auto freed_model = [&](const std::vector<size_t> & picks) {
        kv_zone_model model = zone;
        for (size_t i : picks) {
            model[copies[i].allocator].free(copies[i].offset);
        }
        return model;
    };

    std::vector<size_t> picked;
    std::vector<size_t> pending;
    size_t              placed = kv_layers_allocatable(zone, layer_bytes);
    for (size_t i : order) {
        if (placed >= layer_bytes.size()) {
            break;
        }
        pending.push_back(i);
        std::vector<size_t> trial = picked;
        trial.insert(trial.end(), pending.begin(), pending.end());
        const size_t trial_placed = kv_layers_allocatable(freed_model(trial), layer_bytes);
        if (trial_placed <= placed) {
            continue;
        }
        // Keep only the pending copies this gain needs: one that no longer
        // coalesces into the block a layer took is dropped again.
        for (size_t p = 0; p + 1 < pending.size();) {
            std::vector<size_t> without = picked;
            for (size_t q = 0; q < pending.size(); ++q) {
                if (q != p) {
                    without.push_back(pending[q]);
                }
            }
            if (kv_layers_allocatable(freed_model(without), layer_bytes) >= trial_placed) {
                pending.erase(pending.begin() + (std::ptrdiff_t) p);
            } else {
                ++p;
            }
        }
        picked.insert(picked.end(), pending.begin(), pending.end());
        pending.clear();
        placed = trial_placed;
    }
    return picked;
}

int layer_block_kv_device(const std::vector<int> &    kv_device,
                          const std::vector<size_t> & layer_kv_bytes,
                          int                         start_layer,
                          int                         end_layer) {
    const int n_layers = (int) std::min(kv_device.size(), layer_kv_bytes.size());
    int       owner    = -1;
    bool      seen     = false;
    for (int l = std::max(start_layer, 0); l <= end_layer && l < n_layers; ++l) {
        if (layer_kv_bytes[l] == 0) {
            continue;
        }
        if (!seen) {
            owner = kv_device[l];
            seen  = true;
        } else if (owner != kv_device[l]) {
            return -2;
        }
    }
    return owner;
}

}  // namespace ggml_sycl
