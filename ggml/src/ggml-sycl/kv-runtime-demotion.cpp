#include "kv-runtime-demotion.hpp"

namespace ggml_sycl {

kv_demotion_result plan_runtime_kv_demotion(const kv_demotion_input & in) {
    kv_demotion_result r;
    r.vram_bytes_after = in.vram_bytes;

    if (in.vram_bytes <= in.vram_budget) {
        r.fits = true;
        return r;
    }

    const int n_layers = (int) in.kv_device.size();
    for (int l = n_layers - 1; l >= 0; --l) {
        if (in.kv_device[l] < 0) {
            continue;  // already host
        }
        const bool is_swa = l < (int) in.swa_layer_mask.size() && in.swa_layer_mask[l] != 0;
        if (is_swa) {
            continue;  // SWA KV is ~1.5 MB/layer; demoting it buys nothing and costs a split
        }
        // llama.cpp-3aos: THIS layer's own recorded bytes, not a
        // uniform figure applied to every full-attention layer -- 0 means
        // nothing is recorded for it (untracked, or a SHARED layer with no
        // independent KV to move), so there is nothing to demote.
        const size_t layer_bytes = l < (int) in.kv_bytes_per_layer.size() ? in.kv_bytes_per_layer[l] : 0;
        if (layer_bytes == 0) {
            continue;
        }
        if (layer_bytes > r.vram_bytes_after) {
            break;  // demoting would underflow vram_bytes_after; refuse rather than wrap
        }
        r.demoted_layers.push_back(l);
        r.vram_bytes_after -= layer_bytes;
        r.host_kv_bytes_added += layer_bytes;
        if (r.vram_bytes_after <= in.vram_budget) {
            break;
        }
    }

    r.fits = r.vram_bytes_after <= in.vram_budget;
    return r;
}

}  // namespace ggml_sycl
