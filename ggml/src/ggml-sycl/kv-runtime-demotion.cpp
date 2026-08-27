#include "kv-runtime-demotion.hpp"

namespace ggml_sycl {

kv_demotion_result plan_runtime_kv_demotion(const kv_demotion_input & in) {
    kv_demotion_result r;
    r.vram_bytes_after = in.vram_bytes;

    if (in.vram_bytes <= in.vram_budget) {
        r.fits = true;
        return r;
    }

    // Loop-invariant: kv_per_layer never changes across layers, so a caller
    // that supplied no per-layer byte figure cannot size any demotion --
    // decide nothing rather than evaluate the same false premise per layer.
    if (in.kv_per_layer == 0) {
        r.fits = false;
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
        if (in.kv_per_layer > r.vram_bytes_after) {
            break;  // demoting would underflow vram_bytes_after; refuse rather than wrap
        }
        r.demoted_layers.push_back(l);
        r.vram_bytes_after -= in.kv_per_layer;
        r.host_kv_bytes_added += in.kv_per_layer;
        if (r.vram_bytes_after <= in.vram_budget) {
            break;
        }
    }

    r.fits = r.vram_bytes_after <= in.vram_budget;
    return r;
}

}  // namespace ggml_sycl
