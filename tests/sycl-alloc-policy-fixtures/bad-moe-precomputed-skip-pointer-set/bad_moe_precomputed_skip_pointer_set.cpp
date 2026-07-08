#include <unordered_set>

struct ggml_tensor;

static thread_local std::unordered_set<const ggml_tensor *> g_moe_precomputed_mmid_skip;
static thread_local std::unordered_set<const ggml_tensor *> g_moe_precomputed_node_skip;
static thread_local std::unordered_set<const ggml_tensor *> g_moe_down_sum_fusion_disabled;

bool bad_moe_skip_state(const ggml_tensor * node) {
    g_moe_precomputed_mmid_skip.insert(node);
    g_moe_precomputed_node_skip.erase(node);
    return g_moe_down_sum_fusion_disabled.find(node) != g_moe_down_sum_fusion_disabled.end();
}
