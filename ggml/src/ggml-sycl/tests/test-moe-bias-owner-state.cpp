// Owner-keyed MoE expert-bias and fused-activation state (llama.cpp-nlww).
//
// Host-only: no SYCL queue, no device, no model. It drives the REAL
// ggml_sycl::moe_bias_activation_state through the REAL
// ggml_sycl::moe_discovery_registry, wired together by a live-state stand-in
// that parks, restores and clears exactly the way ggml-sycl.cpp's
// ggml_sycl_moe_live_discovery does. So the ownership and lifetime rules are
// proven without the GPU serialization the real cluster needs.
//
// What this test can and cannot see, stated plainly because it decides where a
// regression would be caught: it proves the state type and the ownership rules
// are correct when wired this way. It CANNOT see whether ggml-sycl.cpp is
// actually wired that way -- whether park() really moves, whether
// install_fresh() really clears the bias cluster, whether the graph scans really
// consult the owner-keyed guard. That is what the companion source contract
// (tests/test-sycl-moe-bias-owner-contract.py) covers.
//
// The four scenarios the task exists for are BIASED_TO_UNBIASED,
// UNBIASED_TO_BIASED, ACTIVATION_VARIANTS and OVERLAPPING_LAYER_IDS, plus the
// rule that binds them: no owner switch or teardown may clear another live
// model's state.

#include "moe-bias-state.hpp"
#include "moe-discovery-state.hpp"

#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using ggml_sycl::layer_expert_biases;
using ggml_sycl::moe_bias_activation_state;
using ggml_sycl::moe_discovery_parked_state;
using ggml_sycl::moe_discovery_registry;
using ggml_sycl::moe_owner_key;

static void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Mirrors ggml_sycl_moe_parked_discovery: the parked half holds the owner's
// bias working set by value, moved rather than copied.
struct parked_bias_state : public moe_discovery_parked_state {
    moe_bias_activation_state bias;
};

// Mirrors ggml_sycl_moe_live_discovery for the bias cluster only. `live` stands
// in for the process-global g_moe_bias_state.
struct bias_live_state : public ggml_sycl::moe_discovery_live_state {
    std::unique_ptr<moe_discovery_parked_state> park() noexcept override {
        parked_bias_state * parked = new (std::nothrow) parked_bias_state();
        if (!parked) {
            return nullptr;
        }
        parked->bias = std::move(live);
        live.clear();
        return std::unique_ptr<moe_discovery_parked_state>(parked);
    }

    void restore(std::unique_ptr<moe_discovery_parked_state> parked) noexcept override {
        install_fresh();
        parked_bias_state * state = static_cast<parked_bias_state *>(parked.get());
        if (!state) {
            return;
        }
        live = std::move(state->bias);
    }

    void install_fresh() noexcept override { live.clear(); }

    moe_bias_activation_state live;
};

static moe_owner_key owner_of(uint64_t model_id, uint32_t slot, uint64_t generation) {
    moe_owner_key key;
    key.model_id        = model_id;
    key.load_txn_id     = model_id * 10;
    key.slot            = slot;
    key.slot_generation = generation;
    return key;
}

// Stands in for one owner's graph scan: capture `n_experts` floats of bias data
// for `layer`, all set to `fill` so the test can tell whose data it is reading.
static void scan_layer_bias(moe_bias_activation_state & state, int layer, int n_experts, float fill) {
    float * gate = state.add_host_copy(static_cast<size_t>(n_experts));
    float * up   = state.add_host_copy(static_cast<size_t>(n_experts));
    for (int e = 0; e < n_experts; ++e) {
        gate[e] = fill;
        up[e]   = fill + 100.0f;
    }
    state.set_layer_stride(layer, sizeof(float));
    state.set_layer_bias(layer, ggml_sycl::MOE_BIAS_SLOT_GATE, gate);
    state.set_layer_bias(layer, ggml_sycl::MOE_BIAS_SLOT_UP, up);
}

// What the fusion path does: look the layer up and read expert 0's gate bias.
// Returns the sentinel when the layer is absent, which is the unbiased answer.
static float read_gate_bias(const moe_bias_activation_state & state, int layer, float absent) {
    const layer_expert_biases * entry = state.find_layer(layer);
    if (!entry || !entry->gate_bias) {
        return absent;
    }
    return entry->gate_bias[0];
}

// ---------------------------------------------------------------------------
// The invariant the whole design rests on.
// ---------------------------------------------------------------------------

static void case_state_is_move_only() {
    // A copy would leave the copy's layer_biases pointing into the ORIGINAL's
    // host_copies. Deleting the copy operations is what makes that unwritable
    // rather than merely discouraged, so it is asserted rather than assumed.
    static_assert(!std::is_copy_constructible<moe_bias_activation_state>::value,
                  "moe_bias_activation_state must not be copy-constructible: a copy dangles into the original");
    static_assert(!std::is_copy_assignable<moe_bias_activation_state>::value,
                  "moe_bias_activation_state must not be copy-assignable: a copy dangles into the original");
    static_assert(std::is_move_constructible<moe_bias_activation_state>::value,
                  "parking an owner's biases requires a move");
    static_assert(std::is_move_assignable<moe_bias_activation_state>::value,
                  "restoring an owner's biases requires a move");

    moe_bias_activation_state original;
    scan_layer_bias(original, 3, 4, 42.0f);
    const float * before = original.find_layer(3)->gate_bias;

    moe_bias_activation_state moved = std::move(original);
    check(moved.find_layer(3) != nullptr, "the moved-to state lost the layer entry");
    check(moved.find_layer(3)->gate_bias == before,
          "moving the state changed a recorded bias pointer -- the buffers were copied, not transferred");
    check(read_gate_bias(moved, 3, -1.0f) == 42.0f, "the moved-to state's bias pointer no longer reads its own data");
}

static void case_host_copy_pointers_survive_growth() {
    // The graph scan hands out a pointer into each host copy as it makes it,
    // then keeps making more. This is the positive control for the claim in
    // moe-bias-state.hpp that growing the outer vector moves rather than copies
    // the inner ones: without it, every pointer but the last would dangle.
    //
    // Deliberately no reserve_host_copies() here: the point is that the
    // pointers hold up when the outer vector reallocates, so the case must be
    // the one where it does.
    moe_bias_activation_state state;

    std::vector<float *> handed_out;
    for (int i = 0; i < 64; ++i) {
        float * buffer = state.add_host_copy(8);
        buffer[0]      = static_cast<float>(i);
        handed_out.push_back(buffer);
    }

    check(state.host_copy_count() == 64, "the state did not retain every host copy");
    for (int i = 0; i < 64; ++i) {
        check(handed_out[i][0] == static_cast<float>(i),
              "a pointer handed out before the host-copy list grew no longer reads its own data");
        check(state.host_copies[i].data() == handed_out[i],
              "growing the host-copy list relocated an earlier buffer -- every pointer already recorded "
              "in layer_biases would dangle");
    }
}

// ---------------------------------------------------------------------------
// The four required scenarios.
// ---------------------------------------------------------------------------

static void case_biased_to_unbiased() {
    bias_live_state        live;
    moe_discovery_registry registry(live);

    const moe_owner_key biased   = owner_of(1, 0, 1);
    const moe_owner_key unbiased = owner_of(2, 1, 1);

    check(registry.activate(biased) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating the biased model failed");
    scan_layer_bias(live.live, 3, 4, 42.0f);
    live.live.biases_scanned = true;

    check(registry.activate(unbiased) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating the unbiased model failed");
    check(!live.live.biases_scanned,
          "the unbiased model inherited the biased model's scanned flag and would never scan its own graph");
    check(live.live.layer_count() == 0, "the unbiased model inherited the biased model's per-layer bias entries");
    check(read_gate_bias(live.live, 3, -1.0f) == -1.0f, "the unbiased model read the biased model's layer 3 gate bias");

    // Its own scan finds nothing, and recording that must not resurrect the
    // other model's entries.
    live.live.biases_scanned = true;
    check(live.live.layer_count() == 0, "an unbiased scan produced bias entries");
}

static void case_unbiased_to_biased() {
    bias_live_state        live;
    moe_discovery_registry registry(live);

    const moe_owner_key unbiased = owner_of(1, 0, 1);
    const moe_owner_key biased   = owner_of(2, 1, 1);

    check(registry.activate(unbiased) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating the unbiased model failed");
    live.live.biases_scanned = true;  // scanned, found no ADD_ID nodes

    check(registry.activate(biased) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating the biased model failed");
    check(!live.live.biases_scanned,
          "the biased model inherited the unbiased model's scanned flag and would skip its own scan entirely");

    scan_layer_bias(live.live, 3, 4, 42.0f);
    live.live.biases_scanned = true;
    check(read_gate_bias(live.live, 3, -1.0f) == 42.0f, "the biased model could not record its own layer 3 bias");
}

static void case_activation_variants_do_not_leak() {
    bias_live_state        live;
    moe_discovery_registry registry(live);

    const moe_owner_key oai  = owner_of(1, 0, 1);
    const moe_owner_key silu = owner_of(2, 1, 1);

    check(registry.activate(oai) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating the SWIGLU_OAI model failed");
    live.live.act_variant = 1;  // CPU_EXPERT_FUSED_ACT_SWIGLU_OAI
    live.live.act_alpha   = 1.702f;
    live.live.act_limit   = 7.0f;

    check(registry.activate(silu) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating the SiLU model failed");
    check(!live.live.act_detected(),
          "the second model inherited a detected activation variant and would never scan its own GLU node");
    check(live.live.act_alpha == 0.0f && live.live.act_limit == 0.0f,
          "the second model inherited the first model's swiglu alpha/limit");

    live.live.act_variant = 0;  // CPU_EXPERT_FUSED_ACT_SILU

    // A -> B -> A hands the first model back its OWN variant and scalars, not
    // the second model's and not a rescan-from-scratch.
    check(registry.activate(oai) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "reactivating the SWIGLU_OAI model failed");
    check(live.live.act_variant == 1, "the first model did not get its own activation variant back");
    check(live.live.act_alpha == 1.702f && live.live.act_limit == 7.0f,
          "the first model did not get its own swiglu alpha/limit back");
}

static void case_overlapping_layer_ids() {
    bias_live_state        live;
    moe_discovery_registry registry(live);

    const moe_owner_key a = owner_of(1, 0, 1);
    const moe_owner_key b = owner_of(2, 1, 1);

    // Both models define layers 3 and 7. Layer ids are model-local, so this is
    // the ordinary case rather than a contrived one, and pointer identity alone
    // would not catch a mix-up -- the test reads the values through.
    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating A failed");
    scan_layer_bias(live.live, 3, 4, 42.0f);
    scan_layer_bias(live.live, 7, 4, 43.0f);
    live.live.biases_scanned = true;

    check(registry.activate(b) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating B failed");
    check(read_gate_bias(live.live, 3, -1.0f) == -1.0f, "B saw A's layer 3 bias");
    check(read_gate_bias(live.live, 7, -1.0f) == -1.0f, "B saw A's layer 7 bias");
    scan_layer_bias(live.live, 3, 4, 99.0f);
    scan_layer_bias(live.live, 7, 4, 98.0f);
    live.live.biases_scanned = true;

    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "reactivating A failed");
    check(read_gate_bias(live.live, 3, -1.0f) == 42.0f, "A did not get its own layer 3 bias back");
    check(read_gate_bias(live.live, 7, -1.0f) == 43.0f, "A did not get its own layer 7 bias back");
    check(live.live.biases_scanned, "A had to rescan a graph it had already scanned");

    check(registry.activate(b) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "reactivating B failed");
    check(read_gate_bias(live.live, 3, -1.0f) == 99.0f, "B did not get its own layer 3 bias back");
    check(read_gate_bias(live.live, 7, -1.0f) == 98.0f, "B did not get its own layer 7 bias back");
}

// ---------------------------------------------------------------------------
// No clear while another model is live.
// ---------------------------------------------------------------------------

static void case_releasing_a_parked_owner_leaves_the_live_one_alone() {
    bias_live_state        live;
    moe_discovery_registry registry(live);

    const moe_owner_key a = owner_of(1, 0, 1);
    const moe_owner_key b = owner_of(2, 1, 1);

    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating A failed");
    scan_layer_bias(live.live, 3, 4, 42.0f);
    live.live.biases_scanned = true;
    live.live.act_variant    = 1;

    check(registry.activate(b) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating B failed");
    scan_layer_bias(live.live, 3, 4, 99.0f);
    live.live.biases_scanned = true;
    live.live.act_variant    = 0;

    // Unloading A while B is live. A is parked, so only its snapshot goes.
    check(registry.release(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "releasing parked owner A failed");
    check(read_gate_bias(live.live, 3, -1.0f) == 99.0f,
          "unloading a parked model cleared the live model's expert biases");
    check(live.live.biases_scanned, "unloading a parked model cleared the live model's scanned flag");
    check(live.live.act_variant == 0, "unloading a parked model cleared the live model's activation variant");
    check(registry.is_active_owner(b), "unloading a parked model displaced the active owner");
}

static void case_releasing_the_active_owner_clears_only_its_own() {
    bias_live_state        live;
    moe_discovery_registry registry(live);

    const moe_owner_key a = owner_of(1, 0, 1);

    check(registry.activate(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "activating A failed");
    scan_layer_bias(live.live, 3, 4, 42.0f);
    live.live.biases_scanned = true;
    live.live.act_variant    = 1;
    live.live.act_alpha      = 1.702f;

    // The live half records pointers into buffers this owner owns, so retiring
    // the owner must take them with it.
    check(registry.release(a) == ggml_sycl::MOE_DISCOVERY_RESULT_OK, "releasing active owner A failed");
    check(live.live.clean(), "retiring the active owner left its bias/activation state behind");
    check(!registry.has_active_owner(), "retiring the active owner left it active");
}

// ---------------------------------------------------------------------------
// The two publish paths ggml-sycl.cpp uses.
// ---------------------------------------------------------------------------

static void case_adopting_biases_keeps_the_activation_half() {
    // The bias scan and the GLU scan run at different points in the same
    // owner's life, so installing the later one must not roll the earlier one
    // back. Move-assigning a whole freshly scanned state would do exactly that,
    // which is why adopt_biases exists.
    moe_bias_activation_state live;
    live.act_variant = 1;
    live.act_alpha   = 1.702f;
    live.act_limit   = 7.0f;

    moe_bias_activation_state scanned;
    scan_layer_bias(scanned, 3, 4, 42.0f);
    scanned.biases_scanned = true;

    live.adopt_biases(std::move(scanned));
    check(read_gate_bias(live, 3, -1.0f) == 42.0f, "adopting a scan did not install its biases");
    check(live.biases_scanned, "adopting a scan did not install its scanned flag");
    check(live.act_variant == 1 && live.act_alpha == 1.702f && live.act_limit == 7.0f,
          "adopting a bias scan rolled back the already-detected activation variant");
}

static void case_adopting_biases_drops_the_previous_map_first() {
    // A second adopt must not leave entries from the first pointing into
    // storage the second one freed.
    moe_bias_activation_state live;
    scan_layer_bias(live, 3, 4, 42.0f);
    live.biases_scanned = true;

    moe_bias_activation_state replacement;
    scan_layer_bias(replacement, 7, 4, 99.0f);
    replacement.biases_scanned = true;

    live.adopt_biases(std::move(replacement));
    check(live.find_layer(3) == nullptr, "adopting a scan left the previous owner's layer entry reachable");
    check(read_gate_bias(live, 7, -1.0f) == 99.0f, "adopting a scan did not install the new layer entry");
    check(live.host_copy_count() == 2, "adopting a scan retained the previous scan's host buffers");
}

static void case_clear_resets_every_field() {
    // clean() is what the module-reload gate reads, so it must not report clean
    // while any part of the cluster survives.
    moe_bias_activation_state state;
    scan_layer_bias(state, 3, 4, 42.0f);
    state.biases_scanned = true;
    state.act_variant    = 1;
    state.act_alpha      = 1.702f;
    state.act_limit      = 7.0f;
    check(!state.clean(), "a fully populated state reported itself clean");

    state.clear();
    check(state.clean(), "clear() left part of the cluster behind");
    check(state.find_layer(3) == nullptr, "clear() left a layer entry reachable");
    check(state.host_copy_count() == 0, "clear() left the host bias copies allocated");
    check(!state.act_detected(), "clear() left the activation variant detected");
}

static void case_absent_layer_reads_as_unbiased() {
    // A default-constructed entry would read as an all-zero bias, which is a
    // different model from an unbiased one. Absent must stay absent.
    moe_bias_activation_state state;
    scan_layer_bias(state, 3, 4, 42.0f);
    check(state.find_layer(4) == nullptr, "looking up an absent layer created an entry");
    check(state.layer_count() == 1, "looking up an absent layer grew the map");
}

int main() {
    struct test_case {
        const char * name;
        void (*run)();
    };

    const test_case cases[] = {
        { "state-is-move-only",         case_state_is_move_only                                 },
        { "host-copy-pointers-survive", case_host_copy_pointers_survive_growth                  },
        { "biased-to-unbiased",         case_biased_to_unbiased                                 },
        { "unbiased-to-biased",         case_unbiased_to_biased                                 },
        { "activation-variants",        case_activation_variants_do_not_leak                    },
        { "overlapping-layer-ids",      case_overlapping_layer_ids                              },
        { "release-parked-owner",       case_releasing_a_parked_owner_leaves_the_live_one_alone },
        { "release-active-owner",       case_releasing_the_active_owner_clears_only_its_own     },
        { "adopt-keeps-activation",     case_adopting_biases_keeps_the_activation_half          },
        { "adopt-drops-previous-map",   case_adopting_biases_drops_the_previous_map_first       },
        { "clear-resets-every-field",   case_clear_resets_every_field                           },
        { "absent-layer-is-unbiased",   case_absent_layer_reads_as_unbiased                     },
    };

    for (const test_case & entry : cases) {
        try {
            entry.run();
        } catch (const std::exception & error) {
            std::cerr << "FAIL: " << entry.name << ": " << error.what() << '\n';
            return 1;
        }
    }

    std::cout << "moe bias owner state: PASS (" << (sizeof(cases) / sizeof(cases[0])) << " cases)\n";
    return 0;
}
