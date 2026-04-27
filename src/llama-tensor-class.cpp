#include "llama-tensor-class.h"

#include "ggml.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

// The classifier accepts canonical ggml/llama.cpp tensor names of the form:
//   <top>                        e.g. "output", "token_embd", "output_norm"
//   <top>.<suffix>               e.g. "output.weight", "token_embd.weight"
//   blk.<N>.<role>               e.g. "blk.3.attn_norm"
//   blk.<N>.<role>.<suffix>      e.g. "blk.3.attn_q.weight", "blk.0.attn_q.bias"
//
// The trailing suffix (typically "weight" or "bias") is a layout detail; the
// role drives the placement class. Bias tensors classify identically to their
// matching weight tensor.

namespace {

struct role_entry {
    const char *       role;
    llama_tensor_class cls;
};

// Top-level (non per-layer) roles.
const role_entry k_top_level[] = {
    { "token_embd",      LLAMA_TENSOR_CLASS_EMBD_TOKEN    },
    { "token_embd_norm", LLAMA_TENSOR_CLASS_EMBD_TOKEN    },
    { "output",          LLAMA_TENSOR_CLASS_EMBD_OUTPUT   },
    { "output_norm",     LLAMA_TENSOR_CLASS_OUTPUT_NORM   },
    { "position_embd",   LLAMA_TENSOR_CLASS_POSITION_EMBD },
    { "pos_embd",        LLAMA_TENSOR_CLASS_POSITION_EMBD },
};

// Per-layer roles (occur after "blk.<N>.").
const role_entry k_per_layer[] = {
    // Attention projections.
    { "attn_q",              LLAMA_TENSOR_CLASS_ATTN_Q            },
    { "attn_q_a",            LLAMA_TENSOR_CLASS_ATTN_Q            },
    { "attn_q_b",            LLAMA_TENSOR_CLASS_ATTN_Q            },
    { "attn_k",              LLAMA_TENSOR_CLASS_ATTN_K            },
    { "attn_k_b",            LLAMA_TENSOR_CLASS_ATTN_K            },
    { "attn_v",              LLAMA_TENSOR_CLASS_ATTN_V            },
    { "attn_v_b",            LLAMA_TENSOR_CLASS_ATTN_V            },
    { "attn_kv_a_mqa",       LLAMA_TENSOR_CLASS_ATTN_K            },
    { "attn_kv_b",           LLAMA_TENSOR_CLASS_ATTN_K            },
    { "attn_output",         LLAMA_TENSOR_CLASS_ATTN_O            },
    { "attn_gate",           LLAMA_TENSOR_CLASS_ATTN_O            },
    { "attn_qkv",            LLAMA_TENSOR_CLASS_ATTN_QKV          },

    // Attention norms / sinks (small, hot, attention-path).
    { "attn_norm",           LLAMA_TENSOR_CLASS_ATTN_NORM         },
    { "attn_norm_2",         LLAMA_TENSOR_CLASS_ATTN_NORM         },
    { "attn_q_norm",         LLAMA_TENSOR_CLASS_ATTN_NORM         },
    { "attn_k_norm",         LLAMA_TENSOR_CLASS_ATTN_NORM         },
    { "attn_q_a_norm",       LLAMA_TENSOR_CLASS_ATTN_NORM         },
    { "attn_kv_a_norm",      LLAMA_TENSOR_CLASS_ATTN_NORM         },
    { "attn_output_norm",    LLAMA_TENSOR_CLASS_ATTN_NORM         },
    { "attn_post_norm",      LLAMA_TENSOR_CLASS_ATTN_NORM         },
    { "post_attention_norm", LLAMA_TENSOR_CLASS_ATTN_NORM         },
    { "attn_sub_norm",       LLAMA_TENSOR_CLASS_ATTN_NORM         },
    { "attn_sinks",          LLAMA_TENSOR_CLASS_ATTN_NORM         },

    // Dense FFN per layer.
    { "ffn_gate",            LLAMA_TENSOR_CLASS_FFN_DENSE_GATE    },
    { "ffn_up",              LLAMA_TENSOR_CLASS_FFN_DENSE_UP      },
    { "ffn_down",            LLAMA_TENSOR_CLASS_FFN_DENSE_DOWN    },

    // FFN norms.
    { "ffn_norm",            LLAMA_TENSOR_CLASS_FFN_NORM          },
    { "ffn_norm_exps",       LLAMA_TENSOR_CLASS_FFN_NORM          },
    { "ffn_post_norm",       LLAMA_TENSOR_CLASS_FFN_NORM          },
    { "post_ffw_norm",       LLAMA_TENSOR_CLASS_FFN_NORM          },
    { "ffn_sub_norm",        LLAMA_TENSOR_CLASS_FFN_NORM          },
    { "layer_output_norm",   LLAMA_TENSOR_CLASS_FFN_NORM          },

    // MoE router and bias.
    { "ffn_gate_inp",        LLAMA_TENSOR_CLASS_FFN_GATE_INP      },
    { "ffn_gate_inp_shexp",  LLAMA_TENSOR_CLASS_FFN_GATE_INP      },
    { "exp_probs_b",         LLAMA_TENSOR_CLASS_FFN_GATE_INP      },

    // MoE merged-expert weights.
    { "ffn_gate_exps",       LLAMA_TENSOR_CLASS_FFN_MOE_GATE_EXPS },
    { "ffn_up_exps",         LLAMA_TENSOR_CLASS_FFN_MOE_UP_EXPS   },
    { "ffn_down_exps",       LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS },
    // Channel-major split (Grove MoE) — same logical role as the merged variant.
    { "ffn_gate_chexps",     LLAMA_TENSOR_CLASS_FFN_MOE_GATE_EXPS },
    { "ffn_up_chexps",       LLAMA_TENSOR_CLASS_FFN_MOE_UP_EXPS   },
    { "ffn_down_chexps",     LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS },

    // DeepSeek-style shared experts.
    { "ffn_gate_shexp",      LLAMA_TENSOR_CLASS_FFN_SHARED_GATE   },
    { "ffn_up_shexp",        LLAMA_TENSOR_CLASS_FFN_SHARED_UP     },
    { "ffn_down_shexp",      LLAMA_TENSOR_CLASS_FFN_SHARED_DOWN   },
};

const std::unordered_map<std::string, llama_tensor_class> & top_level_map() {
    static const auto map = [] {
        std::unordered_map<std::string, llama_tensor_class> m;
        for (const auto & e : k_top_level) {
            m.emplace(e.role, e.cls);
        }
        return m;
    }();
    return map;
}

const std::unordered_map<std::string, llama_tensor_class> & per_layer_map() {
    static const auto map = [] {
        std::unordered_map<std::string, llama_tensor_class> m;
        for (const auto & e : k_per_layer) {
            m.emplace(e.role, e.cls);
        }
        return m;
    }();
    return map;
}

// Strip a trailing ".<suffix>" if it is one of the known layout suffixes.
// Only "weight" and "bias" are stripped; an unknown trailing token is kept
// so it participates in the role lookup (and therefore lands in MISC if
// unrecognized, surfacing the new tensor name to the operator).
std::string strip_layout_suffix(const std::string & role) {
    const std::size_t dot = role.rfind('.');
    if (dot == std::string::npos) {
        return role;
    }
    const std::string suffix = role.substr(dot + 1);
    if (suffix == "weight" || suffix == "bias") {
        return role.substr(0, dot);
    }
    return role;
}

}  // namespace

// Note: called once per tensor at model load (not per token), so the few
// per-call string allocations below are intentional simplicity. If this ever
// becomes a hot path, switch to std::string_view + heterogeneous lookup.
llama_tensor_classification llama_tensor_classify(const char * name) {
    llama_tensor_classification result = { LLAMA_TENSOR_CLASS_MISC, -1 };
    if (name == nullptr) {
        return result;
    }

    const std::string s = name;

    if (s.rfind("blk.", 0) == 0) {
        // Per-layer: "blk.<N>.<role>[.<suffix>]"
        const std::size_t layer_start = 4;  // after "blk."
        const std::size_t layer_end   = s.find('.', layer_start);
        if (layer_end == std::string::npos) {
            return result;
        }

        const std::string layer_str = s.substr(layer_start, layer_end - layer_start);
        char *            end_ptr   = nullptr;
        const long        layer_val = std::strtol(layer_str.c_str(), &end_ptr, 10);
        if (end_ptr == layer_str.c_str() || *end_ptr != '\0' || layer_val < 0) {
            return result;
        }

        const std::string role = strip_layout_suffix(s.substr(layer_end + 1));
        const auto &      map  = per_layer_map();
        const auto        it   = map.find(role);
        if (it != map.end()) {
            result.cls       = it->second;
            result.layer_idx = static_cast<int>(layer_val);
        }
        return result;
    }

    // Top-level (non per-layer).
    const std::string role = strip_layout_suffix(s);
    const auto &      map  = top_level_map();
    const auto        it   = map.find(role);
    if (it != map.end()) {
        result.cls = it->second;
    }
    return result;
}

llama_tensor_classification llama_tensor_classify(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return { LLAMA_TENSOR_CLASS_MISC, -1 };
    }
    // tensor->name is char[64], never null — empty string falls through to MISC.
    return llama_tensor_classify(tensor->name);
}

const char * llama_tensor_class_name(llama_tensor_class cls) {
    switch (cls) {
        case LLAMA_TENSOR_CLASS_MISC:
            return "MISC";
        case LLAMA_TENSOR_CLASS_ATTN_Q:
            return "ATTN_Q";
        case LLAMA_TENSOR_CLASS_ATTN_K:
            return "ATTN_K";
        case LLAMA_TENSOR_CLASS_ATTN_V:
            return "ATTN_V";
        case LLAMA_TENSOR_CLASS_ATTN_O:
            return "ATTN_O";
        case LLAMA_TENSOR_CLASS_ATTN_QKV:
            return "ATTN_QKV";
        case LLAMA_TENSOR_CLASS_ATTN_NORM:
            return "ATTN_NORM";
        case LLAMA_TENSOR_CLASS_FFN_GATE_INP:
            return "FFN_GATE_INP";
        case LLAMA_TENSOR_CLASS_FFN_DENSE_GATE:
            return "FFN_DENSE_GATE";
        case LLAMA_TENSOR_CLASS_FFN_DENSE_UP:
            return "FFN_DENSE_UP";
        case LLAMA_TENSOR_CLASS_FFN_DENSE_DOWN:
            return "FFN_DENSE_DOWN";
        case LLAMA_TENSOR_CLASS_FFN_MOE_GATE_EXPS:
            return "FFN_MOE_GATE_EXPS";
        case LLAMA_TENSOR_CLASS_FFN_MOE_UP_EXPS:
            return "FFN_MOE_UP_EXPS";
        case LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS:
            return "FFN_MOE_DOWN_EXPS";
        case LLAMA_TENSOR_CLASS_FFN_SHARED_GATE:
            return "FFN_SHARED_GATE";
        case LLAMA_TENSOR_CLASS_FFN_SHARED_UP:
            return "FFN_SHARED_UP";
        case LLAMA_TENSOR_CLASS_FFN_SHARED_DOWN:
            return "FFN_SHARED_DOWN";
        case LLAMA_TENSOR_CLASS_FFN_NORM:
            return "FFN_NORM";
        case LLAMA_TENSOR_CLASS_EMBD_TOKEN:
            return "EMBD_TOKEN";
        case LLAMA_TENSOR_CLASS_EMBD_OUTPUT:
            return "EMBD_OUTPUT";
        case LLAMA_TENSOR_CLASS_OUTPUT_NORM:
            return "OUTPUT_NORM";
        case LLAMA_TENSOR_CLASS_POSITION_EMBD:
            return "POSITION_EMBD";
        case LLAMA_TENSOR_CLASS_COUNT:
            break;
    }
    return "UNKNOWN";
}

// PLACE-2 priority policy.
//
// Tier rationale (input to PLACE-3 greedy budget fitter):
//   P0 — every-token hot path, must be device-resident. Includes attention
//        projections (Q/K/V/O/QKV) and norms, MoE router, shared experts
//        (DeepSeek-style: shared experts run for every token, not gated by
//        the router), dense FFN (in dense or hybrid layers), output unembed
//        (full-vocab matmul each token), token + position embeddings, and
//        all *_NORM tensors. PLACE-3 hard-errors if P0 alone exceeds budget.
//   P1 — hot-half MoE expert weights (layer index < n_layers / 2). MoE-Infinity
//        (arXiv 2401.14361) and the HF llama.cpp MoE offload guide both report
//        skewed expert hit rates concentrated in earlier layers, so the early
//        half is a higher-value VRAM target than the late half.
//   P2 — cold-half MoE expert weights (layer index >= n_layers / 2). Compete
//        for VRAM after P0 + P1 are placed.
//   P3 — overflow from P1/P2 fitting (assigned by PLACE-3, not by this
//        function). PLACE-2 never returns P3.
//
// MISC -> P0: unknown roles default to "pin to device". A future quant adds
// a tensor we don't recognize, we'd rather see PLACE-3's hard-error if it
// blows the budget (a loud, actionable failure) than silently demote it to
// host and tank perf for what may turn out to be a hot tensor. Loud failure
// beats silent slowdown for the unknown-tensor case.
//
// POSITION_EMBD -> P0: only present on legacy arches without RoPE (GPT-2,
// some BERT variants); when present it's small and hot — same bucket as norms.
//
// L/2 split convention: integer division (floor). For odd n_layers (e.g. 33),
// boundary = 16, so P1 = [0,16) covers 16 layers and P2 = [16,33) covers 17.
// The asymmetry is at most one layer; floor is chosen for idiom simplicity,
// not for any hot-tier coverage argument.
llama_tensor_priority llama_tensor_priority_for(llama_tensor_class cls, int layer_idx, int n_layers) {
    switch (cls) {
        // Every-token hot path. All P0.
        case LLAMA_TENSOR_CLASS_ATTN_Q:
        case LLAMA_TENSOR_CLASS_ATTN_K:
        case LLAMA_TENSOR_CLASS_ATTN_V:
        case LLAMA_TENSOR_CLASS_ATTN_O:
        case LLAMA_TENSOR_CLASS_ATTN_QKV:
        case LLAMA_TENSOR_CLASS_ATTN_NORM:
        case LLAMA_TENSOR_CLASS_FFN_GATE_INP:
        case LLAMA_TENSOR_CLASS_FFN_DENSE_GATE:
        case LLAMA_TENSOR_CLASS_FFN_DENSE_UP:
        case LLAMA_TENSOR_CLASS_FFN_DENSE_DOWN:
        case LLAMA_TENSOR_CLASS_FFN_SHARED_GATE:
        case LLAMA_TENSOR_CLASS_FFN_SHARED_UP:
        case LLAMA_TENSOR_CLASS_FFN_SHARED_DOWN:
        case LLAMA_TENSOR_CLASS_FFN_NORM:
        case LLAMA_TENSOR_CLASS_EMBD_TOKEN:
        case LLAMA_TENSOR_CLASS_EMBD_OUTPUT:
        case LLAMA_TENSOR_CLASS_OUTPUT_NORM:
        case LLAMA_TENSOR_CLASS_POSITION_EMBD:
        case LLAMA_TENSOR_CLASS_MISC:
            return LLAMA_TENSOR_PRIORITY_P0;

        // Routed-expert weights: hot half (P1) vs cold half (P2) by layer.
        // Precondition: routed experts are always per-layer (PLACE-1 only emits
        // FFN_MOE_*_EXPS for blk.<N>.<role>) and a loaded model always has
        // n_layers > 0. Asserting at the violation site is louder than silent
        // re-bucketing, matching the MISC -> P0 rationale above.
        case LLAMA_TENSOR_CLASS_FFN_MOE_GATE_EXPS:
        case LLAMA_TENSOR_CLASS_FFN_MOE_UP_EXPS:
        case LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS:
            GGML_ASSERT(layer_idx >= 0 && n_layers > 0);
            return (layer_idx < n_layers / 2) ? LLAMA_TENSOR_PRIORITY_P1 : LLAMA_TENSOR_PRIORITY_P2;

        case LLAMA_TENSOR_CLASS_COUNT:
            break;
    }
    return LLAMA_TENSOR_PRIORITY_P0;
}

const char * llama_tensor_priority_name(llama_tensor_priority prio) {
    switch (prio) {
        case LLAMA_TENSOR_PRIORITY_P0:
            return "P0";
        case LLAMA_TENSOR_PRIORITY_P1:
            return "P1";
        case LLAMA_TENSOR_PRIORITY_P2:
            return "P2";
        case LLAMA_TENSOR_PRIORITY_P3:
            return "P3";
        case LLAMA_TENSOR_PRIORITY_COUNT:
            break;
    }
    return "UNKNOWN";
}
