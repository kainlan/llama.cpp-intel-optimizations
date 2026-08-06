#ifndef GGML_SYCL_MOE_BIAS_STATE_HPP
#define GGML_SYCL_MOE_BIAS_STATE_HPP

// Owner-keyed ownership for the MoE expert-bias and fused-activation state
// (llama.cpp-nlww, canonical §12 lifecycle contract).
//
// Two graph scans discover this cluster, both of them process-lifetime `once`
// before this change:
//
//   - The expert-bias scan walks ADD_ID(MUL_MAT_ID, bias) nodes, copies each
//     bias tensor from device memory to a host buffer, and records a per-layer
//     {gate, up, down} pointer triple that the CPU fused expert kernel reads.
//     It ran under a `std::call_once`, so the FIRST MoE model to reach it owned
//     those pointers until module shutdown. Layer ids are model-local, so a
//     second model with a layer 3 read the first model's layer 3 biases.
//   - The fused-activation scan reads the graph's first GLU node for the
//     variant and, for SWIGLU_OAI, its alpha and limit. It ran under a
//     `< 0` sentinel check on a process-global that nothing reset, so a SILU
//     model loaded after a SWIGLU_OAI model kept the earlier model's variant.
//
// `std::once_flag` cannot be reset, which is why this is a replacement rather
// than a wrapper: the "have we scanned?" bit moves into owner-keyed state that
// a model teardown or an owner switch clears, so each owner scans its own
// graph exactly once and never inherits another owner's answer.
//
// This header is host-only -- no SYCL, no ggml, no device -- so the ownership
// and lifetime rules can be proven without a GPU. ggml-sycl.cpp holds the one
// live instance, parks it through the llama.cpp-nn6z discovery registry, and is
// the only code that performs the device-to-host copies that fill it.

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ggml_sycl {

// Which of the three expert matmuls a captured bias belongs to. The graph scan
// derives it from the weight tensor's name.
enum moe_bias_slot {
    MOE_BIAS_SLOT_GATE = 0,
    MOE_BIAS_SLOT_UP   = 1,
    MOE_BIAS_SLOT_DOWN = 2,
};

// HOST copies of one layer's expert bias bases, plus the per-expert stride in
// bytes (the source tensor's nb[1]). The pointers address storage owned by the
// moe_bias_activation_state that holds this entry -- never the model's device
// memory, and never another owner's state.
struct layer_expert_biases {
    const float * gate_bias = nullptr;
    const float * up_bias   = nullptr;
    const float * down_bias = nullptr;
    size_t        nb        = 0;
};

// "This owner's graph has not been scanned for a GLU yet." The real variants
// are cpu_expert_fused_act, which starts at 0; keeping the sentinel here as a
// plain int is what lets this header stay free of cpu-dispatch.hpp.
static constexpr int moe_fused_act_undetected = -1;

// One model owner's complete bias/activation working set. Movable and not
// copyable, because a park/restore round trip must hand the owner back the very
// buffers its recorded pointers address.
struct moe_bias_activation_state {
    // Per-layer bias bases, keyed by the graph's layer id. Layer ids are
    // model-local, so two models routinely both define a layer 3 with entirely
    // different bias data. That overlap is exactly why this map must not
    // survive an owner switch.
    std::unordered_map<int, layer_expert_biases> layer_biases;

    // The storage `layer_biases` points into, one entry per captured bias
    // tensor. Vector-of-vectors is load-bearing rather than incidental: growing
    // the outer vector MOVES the inner ones, and moving a std::vector transfers
    // its buffer instead of copying it, so a pointer handed out by an earlier
    // add_host_copy() is still valid after later ones. The same property is
    // what makes this whole struct movable.
    std::vector<std::vector<float>> host_copies;

    // Set once this owner's graph has been scanned, whether or not it carried
    // any bias tensors. An unbiased model must record that it looked, or every
    // graph would rescan -- and it must record it under its OWN ownership, so
    // that the next owner still scans its own graph.
    bool biases_scanned = false;

    int   act_variant = moe_fused_act_undetected;
    float act_alpha   = 0.0f;
    float act_limit   = 0.0f;

    moe_bias_activation_state()                                         = default;
    moe_bias_activation_state(moe_bias_activation_state &&)             = default;
    moe_bias_activation_state & operator=(moe_bias_activation_state &&) = default;

    // Copying would leave the copy's layer_biases pointing into the ORIGINAL's
    // host_copies, so the copy would dangle the moment the original is cleared
    // -- which is precisely what an owner switch does. Moving is safe for the
    // same reason a copy is not: it transfers the buffers rather than
    // duplicating them.
    moe_bias_activation_state(const moe_bias_activation_state &)             = delete;
    moe_bias_activation_state & operator=(const moe_bias_activation_state &) = delete;

    bool act_detected() const noexcept { return act_variant != moe_fused_act_undetected; }

    // Hint the number of bias tensors about to be captured, so the scan below
    // does not repeatedly grow the outer vector.
    void reserve_host_copies(size_t n_tensors) { host_copies.reserve(n_tensors); }

    // Zeroed storage for one bias tensor, as a POINTER that stays valid as
    // later copies are added: growing the outer vector moves the inner ones,
    // which transfers their buffers rather than copying them.
    //
    // Returning the pointer rather than a reference to `host_copies.back()` is
    // the whole point, and is not a matter of taste. That reference names the
    // inner std::vector OBJECT, which lives in the outer vector's array and is
    // therefore relocated by the next call -- while the buffer it managed is
    // not. So the two have opposite lifetimes, and the reference form invites
    // holding the one that dies. Handing out only the stable one makes the
    // mistake unwritable.
    float * add_host_copy(size_t n_floats) {
        host_copies.emplace_back(n_floats, 0.0f);
        return host_copies.back().data();
    }

    void set_layer_stride(int layer, size_t nb) { layer_biases[layer].nb = nb; }

    void set_layer_bias(int layer, moe_bias_slot slot, const float * host_ptr) {
        layer_expert_biases & entry = layer_biases[layer];
        switch (slot) {
            case MOE_BIAS_SLOT_GATE:
                entry.gate_bias = host_ptr;
                break;
            case MOE_BIAS_SLOT_UP:
                entry.up_bias = host_ptr;
                break;
            case MOE_BIAS_SLOT_DOWN:
                entry.down_bias = host_ptr;
                break;
        }
    }

    // Null rather than a default-constructed entry: a caller that finds nothing
    // must run that layer unbiased, not read a zeroed triple as if the model
    // had declared biases.
    const layer_expert_biases * find_layer(int layer) const noexcept {
        std::unordered_map<int, layer_expert_biases>::const_iterator it = layer_biases.find(layer);
        return it == layer_biases.end() ? nullptr : &it->second;
    }

    size_t layer_count() const noexcept { return layer_biases.size(); }

    size_t host_copy_count() const noexcept { return host_copies.size(); }

    // Take over another state's captured biases, leaving this state's
    // fused-activation half alone. The two halves come from different graph
    // scans at different points in the same owner's life, so installing one
    // must not roll the other back.
    void adopt_biases(moe_bias_activation_state && scanned) {
        // Drop the outgoing map before the storage it points into, then take
        // the incoming pair in the same order, so no entry is ever reachable
        // while the buffer it addresses is gone.
        layer_biases.clear();
        host_copies.clear();
        host_copies    = std::move(scanned.host_copies);
        layer_biases   = std::move(scanned.layer_biases);
        biases_scanned = scanned.biases_scanned;
        scanned.clear();
    }

    // Back to "no owner has discovered anything here".
    void clear() noexcept {
        layer_biases.clear();
        host_copies.clear();
        biases_scanned = false;
        act_variant    = moe_fused_act_undetected;
        act_alpha      = 0.0f;
        act_limit      = 0.0f;
    }

    // Every field at its post-clear() value. The alpha/limit comparisons are
    // deliberate: a SILU model that inherited a SWIGLU_OAI model's alpha has
    // detected nothing wrong by the variant alone, so the scalars are part of
    // what "clean" means.
    bool clean() const noexcept {
        return layer_biases.empty() && host_copies.empty() && !biases_scanned && !act_detected() && act_alpha == 0.0f &&
               act_limit == 0.0f;
    }
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_MOE_BIAS_STATE_HPP
