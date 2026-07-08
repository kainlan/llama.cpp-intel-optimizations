#include <unordered_map>

struct ggml_tensor {};

static thread_local std::unordered_map<int, const ggml_tensor *> g_moe_weights_by_layer;

void bad_record_moe_weights(int layer, const ggml_tensor * node) {
    g_moe_weights_by_layer[layer] = node;
}

const ggml_tensor * bad_get_moe_weights(int layer) {
    auto wit = g_moe_weights_by_layer.find(layer);
    if (wit == g_moe_weights_by_layer.end()) {
        return nullptr;
    }
    const ggml_tensor * wt = wit->second;
    return wt;
}
