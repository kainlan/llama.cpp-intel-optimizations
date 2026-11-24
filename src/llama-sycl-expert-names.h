#pragma once

#include <cstring>

// Whether a tensor belongs to a routed-expert family the batched prompt executor runs, by the names the model loaders
// create: the separate gate, up and down tensors and the fused gate+up tensor. The SYCL loader sizes the prompt MoE
// oneDNN ring from the widest of them. grovemoe's chunked families (ffn_*_chexps) are left out: the executor's claim
// does not take them (moe_classify_tensor in ggml-sycl.cpp reads them as no role), so the ring is not sized for them.
// The ring's other consumer, the per-expert path's MXFP4 f16 staging at SOA (pp_mxfp4_soa_f16_candidate), asks no role
// and so can meet a chunked MXFP4 expert; its admission compares the expert's f16 bytes with the planned slots and
// declines one they do not hold, so a chunked expert wider than the sized ring takes the per-expert path's other arm.
//
// It matches by substring, as the check it replaces did, so a family's ".bias" and ".scale" tensors match as well, and
// the sizing reads each one's ne[0] and ne[1] as a matrix's K and N. A bias {N, n_expert} gives K = its matrix's N,
// which another matrix of the separate families has as its K (down's K is gate's and up's N, and the reverse), and
// N = n_expert, which widens the output row only for a model with more experts than its widest expert matrix has
// rows. No loader creates a bias for the fused tensor, whose N (2 * n_ff) no matrix has as K. A ".scale" {n_expert}
// gives K = n_expert and N = 1.
static inline bool llama_sycl_is_moe_expert_weight_name(const char * name) {
    if (name == nullptr) {
        return false;
    }
    static const char * kinds[] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps" };
    for (const char * kind : kinds) {
        if (std::strstr(name, kind) != nullptr) {
            return true;
        }
    }
    return false;
}
