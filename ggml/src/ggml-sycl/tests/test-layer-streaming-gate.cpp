#include "../layer-streaming-gate.hpp"

#include <cstdio>
#include <unordered_map>
#include <vector>

// The build is -DNDEBUG (Release), so assert() would compile away and the
// test would pass vacuously. Use an explicit check that always runs, per the
// test-kv-runtime-demotion.cpp precedent in this directory.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

#define CHECK_GATE(got, want, msg)                                                                                     \
    do {                                                                                                               \
        const ggml_sycl::layer_streaming_gate got_v_  = (got);                                                         \
        const ggml_sycl::layer_streaming_gate want_v_ = (want);                                                        \
        if (got_v_ != want_v_) {                                                                                       \
            std::fprintf(stderr, "FAIL: %s:%d: %s (got %s, want %s)\n", __FILE__, __LINE__, msg,                       \
                         ggml_sycl::layer_streaming_gate_name(got_v_), ggml_sycl::layer_streaming_gate_name(want_v_)); \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

using ggml_sycl::layer_streaming_gate;
using ggml_sycl::layer_streaming_gate_decide;
using ggml_sycl::layer_streaming_gate_enabled;
using ggml_sycl::layer_streaming_gate_inputs;
using ggml_sycl::plan_layer_residency_count;

static constexpr size_t MB = 1024 * 1024;

// n_layers layers of tensors_per_layer tensors each, plus two layer-less
// tensors (token_embd, output) at the front.
static std::vector<int> make_layer_ids(int n_layers, int tensors_per_layer) {
    std::vector<int> ids = { -1, -1 };
    for (int l = 0; l < n_layers; ++l) {
        for (int t = 0; t < tensors_per_layer; ++t) {
            ids.push_back(l);
        }
    }
    return ids;
}

// Mistral 7B on the dense two-card split: B70 owns 0-29, B50 owns 30-31.
static std::unordered_map<int, int> make_split(int n_layers, int boundary) {
    std::unordered_map<int, int> ld;
    for (int l = 0; l < n_layers; ++l) {
        ld[l] = l < boundary ? 0 : 1;
    }
    return ld;
}

int main() {
    const std::vector<int> ids = make_layer_ids(32, 9);

    // 1. Residency count: layer-less tensors are not counted; a split with
    //    every layer on a device leaves nothing unresident.
    {
        const auto r = plan_layer_residency_count(ids, make_split(32, 30));
        CHECK(r.layer_tensors == 32 * 9, "case 1: layer tensors counted, layer-less skipped");
        CHECK(r.unresident_tensors == 0, "case 1: split is fully resident");
    }

    // 2. Residency count: a host layer and a layer the plan never assigned
    //    are both unresident.
    {
        auto ld = make_split(32, 30);
        ld[5]   = -1;
        ld.erase(7);
        const auto r = plan_layer_residency_count(ids, ld);
        CHECK(r.unresident_tensors == 2 * 9, "case 2: host layer + unplanned layer are unresident");
    }

    // 3. THE DEFECT (llama.cpp-40j4): the dense split at PCT=22. The whole
    //    model exceeds dev0's weight budget, but the plan holds every layer
    //    resident on its owner, so streaming must stay off.
    {
        layer_streaming_gate_inputs in;
        in.model_bytes       = 3918 * MB;
        in.weight_budget     = 3800 * MB;
        in.plan_multi_device = true;
        in.residency         = plan_layer_residency_count(ids, make_split(32, 30));
        const auto g         = layer_streaming_gate_decide(in);
        CHECK_GATE(g, layer_streaming_gate::OFF_PLAN_RESIDENT, "case 3: resident split must not stream");
        CHECK(!layer_streaming_gate_enabled(g), "case 3: not enabled");
    }

    // 4. Single-card over budget with planner host layers (B50 PCT=60):
    //    unchanged -- streaming stays on.
    {
        std::unordered_map<int, int> ld;
        for (int l = 0; l < 32; ++l) {
            ld[l] = l < 20 ? 0 : -1;
        }
        layer_streaming_gate_inputs in;
        in.model_bytes       = 3918 * MB;
        in.weight_budget     = 2500 * MB;
        in.plan_multi_device = false;
        in.residency         = plan_layer_residency_count(ids, ld);
        CHECK_GATE(layer_streaming_gate_decide(in), layer_streaming_gate::ON_OVER_BUDGET,
                   "case 4: single-card host layers keep streaming");
    }

    // 5. Single-card over budget with every layer on the device: the
    //    exemption is multi-device only, so the legacy answer stands.
    {
        std::unordered_map<int, int> ld;
        for (int l = 0; l < 32; ++l) {
            ld[l] = 0;
        }
        layer_streaming_gate_inputs in;
        in.model_bytes       = 3918 * MB;
        in.weight_budget     = 3800 * MB;
        in.plan_multi_device = false;
        in.residency         = plan_layer_residency_count(ids, ld);
        CHECK_GATE(layer_streaming_gate_decide(in), layer_streaming_gate::ON_OVER_BUDGET,
                   "case 5: single-device plan keeps legacy gate");
    }

    // 6. Split with one layer on the host: not resident, so streaming stays on.
    {
        auto ld = make_split(32, 30);
        ld[3]   = -1;
        layer_streaming_gate_inputs in;
        in.model_bytes       = 3918 * MB;
        in.weight_budget     = 3800 * MB;
        in.plan_multi_device = true;
        in.residency         = plan_layer_residency_count(ids, ld);
        CHECK_GATE(layer_streaming_gate_decide(in), layer_streaming_gate::ON_OVER_BUDGET,
                   "case 6: split with a host layer keeps streaming");
    }

    // 7. Split plan with no layer map at all (no layer tensors counted):
    //    residency is unproven, so the legacy answer stands.
    {
        layer_streaming_gate_inputs in;
        in.model_bytes       = 3918 * MB;
        in.weight_budget     = 3800 * MB;
        in.plan_multi_device = true;
        in.residency         = plan_layer_residency_count(ids, {});
        CHECK(in.residency.layer_tensors > 0 && in.residency.unresident_tensors == in.residency.layer_tensors,
              "case 7: empty layer map leaves every layer unresident");
        in.residency = {};
        CHECK_GATE(layer_streaming_gate_decide(in), layer_streaming_gate::ON_OVER_BUDGET,
                   "case 7: zero counted layers is not proof of residency");
    }

    // 8. FORCE_STREAMING wins even over a resident split.
    {
        layer_streaming_gate_inputs in;
        in.forced            = true;
        in.model_bytes       = 3918 * MB;
        in.weight_budget     = 3800 * MB;
        in.plan_multi_device = true;
        in.residency         = plan_layer_residency_count(ids, make_split(32, 30));
        CHECK_GATE(layer_streaming_gate_decide(in), layer_streaming_gate::ON_FORCED, "case 8: forced");
    }

    // 9. MoE over budget never auto-streams; a fitting model never streams.
    {
        layer_streaming_gate_inputs in;
        in.is_moe        = true;
        in.model_bytes   = 12000 * MB;
        in.weight_budget = 8000 * MB;
        CHECK_GATE(layer_streaming_gate_decide(in), layer_streaming_gate::OFF_MOE, "case 9: moe");
        in.is_moe      = false;
        in.model_bytes = 3000 * MB;
        CHECK_GATE(layer_streaming_gate_decide(in), layer_streaming_gate::OFF_FITS, "case 9: fits");
    }

    std::printf("test-layer-streaming-gate: all cases passed\n");
    return 0;
}
