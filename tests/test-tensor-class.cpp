// Unit tests for the tensor-name classifier (PLACE-1, llama.cpp-i7hhs).

#include "../src/llama-tensor-class.h"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

namespace {

struct case_expect {
    const char *       name;
    llama_tensor_class cls;
    int                layer_idx;
};

struct fail_log {
    int n_failures = 0;
};

void check_case(fail_log & log, const case_expect & c) {
    const llama_tensor_classification got = llama_tensor_classify(c.name);
    if (got.cls == c.cls && got.layer_idx == c.layer_idx) {
        return;
    }
    fprintf(stderr, "FAIL: %-50s -> got (%s, layer=%d), expected (%s, layer=%d)\n", c.name,
            llama_tensor_class_name(got.cls), got.layer_idx, llama_tensor_class_name(c.cls), c.layer_idx);
    ++log.n_failures;
}

// Per-arch tensor enumerator: produce the full tensor-name list a real GGUF
// would contain at the given layer count, matching the model loader's
// create_tensor calls. The tests classify each name and assert no MISC except
// names listed in 'expected_misc'.

std::vector<std::string> mistral7b_tensor_names(int n_layer) {
    std::vector<std::string> names;
    names.push_back("token_embd.weight");
    names.push_back("output_norm.weight");
    names.push_back("output.weight");

    for (int i = 0; i < n_layer; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        names.push_back(p + "attn_q.weight");
        names.push_back(p + "attn_k.weight");
        names.push_back(p + "attn_v.weight");
        names.push_back(p + "attn_output.weight");
        names.push_back(p + "attn_norm.weight");
        names.push_back(p + "ffn_gate.weight");
        names.push_back(p + "ffn_down.weight");
        names.push_back(p + "ffn_up.weight");
        names.push_back(p + "ffn_norm.weight");
    }
    return names;
}

std::vector<std::string> gpt_oss_tensor_names(int n_layer) {
    std::vector<std::string> names;
    names.push_back("token_embd.weight");
    names.push_back("output_norm.weight");
    names.push_back("output.weight");

    for (int i = 0; i < n_layer; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        // attention weights and biases
        names.push_back(p + "attn_q.weight");
        names.push_back(p + "attn_q.bias");
        names.push_back(p + "attn_k.weight");
        names.push_back(p + "attn_k.bias");
        names.push_back(p + "attn_v.weight");
        names.push_back(p + "attn_v.bias");
        names.push_back(p + "attn_output.weight");
        names.push_back(p + "attn_output.bias");
        names.push_back(p + "attn_norm.weight");
        names.push_back(p + "post_attention_norm.weight");
        names.push_back(p + "attn_sinks.weight");
        // MoE
        names.push_back(p + "ffn_gate_inp.weight");
        names.push_back(p + "ffn_gate_inp.bias");
        names.push_back(p + "ffn_gate_exps.weight");
        names.push_back(p + "ffn_gate_exps.bias");
        names.push_back(p + "ffn_up_exps.weight");
        names.push_back(p + "ffn_up_exps.bias");
        names.push_back(p + "ffn_down_exps.weight");
        names.push_back(p + "ffn_down_exps.bias");
    }
    return names;
}

std::vector<std::string> qwen3_moe_tensor_names(int n_layer) {
    std::vector<std::string> names;
    names.push_back("token_embd.weight");
    names.push_back("output_norm.weight");
    names.push_back("output.weight");

    for (int i = 0; i < n_layer; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        names.push_back(p + "attn_norm.weight");
        names.push_back(p + "attn_q.weight");
        names.push_back(p + "attn_q_norm.weight");
        names.push_back(p + "attn_k.weight");
        names.push_back(p + "attn_k_norm.weight");
        names.push_back(p + "attn_v.weight");
        names.push_back(p + "attn_output.weight");
        names.push_back(p + "ffn_norm.weight");
        names.push_back(p + "ffn_gate_inp.weight");
        names.push_back(p + "ffn_gate_exps.weight");
        names.push_back(p + "ffn_down_exps.weight");
        names.push_back(p + "ffn_up_exps.weight");
    }
    return names;
}

// DeepSeek-V2 covers shared-expert tensors and the MLA attention split.
std::vector<std::string> deepseek2_tensor_names(int n_layer) {
    std::vector<std::string> names;
    names.push_back("token_embd.weight");
    names.push_back("output_norm.weight");
    names.push_back("output.weight");

    for (int i = 0; i < n_layer; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        names.push_back(p + "attn_norm.weight");
        names.push_back(p + "attn_q_a.weight");
        names.push_back(p + "attn_q_a_norm.weight");
        names.push_back(p + "attn_q_b.weight");
        names.push_back(p + "attn_kv_a_mqa.weight");
        names.push_back(p + "attn_kv_a_norm.weight");
        names.push_back(p + "attn_kv_b.weight");
        names.push_back(p + "attn_output.weight");
        names.push_back(p + "ffn_norm.weight");
        names.push_back(p + "ffn_gate_inp.weight");
        names.push_back(p + "exp_probs_b.bias");
        names.push_back(p + "ffn_gate_exps.weight");
        names.push_back(p + "ffn_down_exps.weight");
        names.push_back(p + "ffn_up_exps.weight");
        names.push_back(p + "ffn_gate_shexp.weight");
        names.push_back(p + "ffn_down_shexp.weight");
        names.push_back(p + "ffn_up_shexp.weight");
    }
    return names;
}

bool check_arch(const char *                     arch_name,
                const std::vector<std::string> & names,
                const std::set<std::string> &    expected_misc) {
    int n_unexpected = 0;
    for (const auto & n : names) {
        const llama_tensor_classification c = llama_tensor_classify(n.c_str());
        if (c.cls != LLAMA_TENSOR_CLASS_MISC) {
            continue;
        }
        if (expected_misc.count(n) > 0) {
            continue;
        }
        fprintf(stderr, "FAIL: %s: unexpected MISC for tensor '%s'\n", arch_name, n.c_str());
        ++n_unexpected;
    }
    if (n_unexpected == 0) {
        printf("OK: %s: all %zu tensors classified (no unexpected MISC)\n", arch_name, names.size());
    }
    return n_unexpected == 0;
}

// Independent layer-index extraction check: every per-layer tensor in the
// supplied list should classify with a layer_idx matching the "blk.N." prefix.
bool check_layer_indices(const char * arch_name, const std::vector<std::string> & names) {
    int n_failed = 0;
    for (const auto & n : names) {
        if (n.rfind("blk.", 0) != 0) {
            continue;
        }
        const std::size_t end = n.find('.', 4);
        if (end == std::string::npos) {
            continue;
        }
        const int                         expected = std::atoi(n.substr(4, end - 4).c_str());
        const llama_tensor_classification c        = llama_tensor_classify(n.c_str());
        if (c.cls == LLAMA_TENSOR_CLASS_MISC) {
            continue;
        }
        if (c.layer_idx != expected) {
            fprintf(stderr, "FAIL: %s: tensor '%s' got layer_idx=%d expected %d\n", arch_name, n.c_str(), c.layer_idx,
                    expected);
            ++n_failed;
        }
    }
    if (n_failed == 0) {
        printf("OK: %s: layer indices correct\n", arch_name);
    }
    return n_failed == 0;
}

}  // namespace

int main() {
    fail_log log;

    // --- Direct unit cases ---------------------------------------------------
    const case_expect cases[] = {
        // Top-level tensors.
        { "token_embd.weight",                LLAMA_TENSOR_CLASS_EMBD_TOKEN,        -1  },
        { "output.weight",                    LLAMA_TENSOR_CLASS_EMBD_OUTPUT,       -1  },
        { "output_norm.weight",               LLAMA_TENSOR_CLASS_OUTPUT_NORM,       -1  },
        { "position_embd.weight",             LLAMA_TENSOR_CLASS_POSITION_EMBD,     -1  },
        { "token_types.weight",               LLAMA_TENSOR_CLASS_MISC,              -1  },

        // Attention QKV/O.
        { "blk.0.attn_q.weight",              LLAMA_TENSOR_CLASS_ATTN_Q,            0   },
        { "blk.0.attn_k.weight",              LLAMA_TENSOR_CLASS_ATTN_K,            0   },
        { "blk.0.attn_v.weight",              LLAMA_TENSOR_CLASS_ATTN_V,            0   },
        { "blk.0.attn_output.weight",         LLAMA_TENSOR_CLASS_ATTN_O,            0   },
        { "blk.0.attn_qkv.weight",            LLAMA_TENSOR_CLASS_ATTN_QKV,          0   },
        { "blk.5.attn_gate.weight",           LLAMA_TENSOR_CLASS_ATTN_O,            5   },

        // Bias follows weight role (option (a)).
        { "blk.7.attn_q.bias",                LLAMA_TENSOR_CLASS_ATTN_Q,            7   },
        { "blk.7.attn_output.bias",           LLAMA_TENSOR_CLASS_ATTN_O,            7   },
        { "blk.7.ffn_gate_inp.bias",          LLAMA_TENSOR_CLASS_FFN_GATE_INP,      7   },

        // Attention norms variants -> ATTN_NORM.
        { "blk.3.attn_norm.weight",           LLAMA_TENSOR_CLASS_ATTN_NORM,         3   },
        { "blk.3.attn_q_norm.weight",         LLAMA_TENSOR_CLASS_ATTN_NORM,         3   },
        { "blk.3.attn_k_norm.weight",         LLAMA_TENSOR_CLASS_ATTN_NORM,         3   },
        { "blk.3.attn_post_norm.weight",      LLAMA_TENSOR_CLASS_ATTN_NORM,         3   },
        { "blk.3.post_attention_norm.weight", LLAMA_TENSOR_CLASS_ATTN_NORM,         3   },
        { "blk.3.attn_sinks.weight",          LLAMA_TENSOR_CLASS_ATTN_NORM,         3   },

        // FFN dense and norm.
        { "blk.1.ffn_gate.weight",            LLAMA_TENSOR_CLASS_FFN_DENSE_GATE,    1   },
        { "blk.1.ffn_up.weight",              LLAMA_TENSOR_CLASS_FFN_DENSE_UP,      1   },
        { "blk.1.ffn_down.weight",            LLAMA_TENSOR_CLASS_FFN_DENSE_DOWN,    1   },
        { "blk.1.ffn_norm.weight",            LLAMA_TENSOR_CLASS_FFN_NORM,          1   },
        { "blk.1.post_ffw_norm.weight",       LLAMA_TENSOR_CLASS_FFN_NORM,          1   },

        // MoE router and merged experts.
        { "blk.0.ffn_gate_inp.weight",        LLAMA_TENSOR_CLASS_FFN_GATE_INP,      0   },
        { "blk.0.exp_probs_b.bias",           LLAMA_TENSOR_CLASS_FFN_GATE_INP,      0   },
        { "blk.0.ffn_gate_inp_shexp.weight",  LLAMA_TENSOR_CLASS_FFN_GATE_INP,      0   },
        { "blk.0.ffn_gate_exps.weight",       LLAMA_TENSOR_CLASS_FFN_MOE_GATE_EXPS, 0   },
        { "blk.0.ffn_up_exps.weight",         LLAMA_TENSOR_CLASS_FFN_MOE_UP_EXPS,   0   },
        { "blk.0.ffn_down_exps.weight",       LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS, 0   },

        // Shared experts (DeepSeek-style).
        { "blk.4.ffn_gate_shexp.weight",      LLAMA_TENSOR_CLASS_FFN_SHARED_GATE,   4   },
        { "blk.4.ffn_up_shexp.weight",        LLAMA_TENSOR_CLASS_FFN_SHARED_UP,     4   },
        { "blk.4.ffn_down_shexp.weight",      LLAMA_TENSOR_CLASS_FFN_SHARED_DOWN,   4   },

        // Two-digit and three-digit layer indices.
        { "blk.42.attn_q.weight",             LLAMA_TENSOR_CLASS_ATTN_Q,            42  },
        { "blk.123.ffn_norm.weight",          LLAMA_TENSOR_CLASS_FFN_NORM,          123 },

        // Missing-suffix forms (canonical without a trailing layout token).
        { "blk.0.attn_q",                     LLAMA_TENSOR_CLASS_ATTN_Q,            0   },
        { "output_norm",                      LLAMA_TENSOR_CLASS_OUTPUT_NORM,       -1  },

        // Unknown roles fall through to MISC, layer_idx stays -1 as designed.
        { "blk.0.foo_bar.weight",             LLAMA_TENSOR_CLASS_MISC,              -1  },
        { "totally_unknown",                  LLAMA_TENSOR_CLASS_MISC,              -1  },
        { "blk.notanumber.attn_q.weight",     LLAMA_TENSOR_CLASS_MISC,              -1  },
    };

    for (const auto & c : cases) {
        check_case(log, c);
    }

    // Null-pointer safety.
    {
        const llama_tensor_classification got = llama_tensor_classify(static_cast<const char *>(nullptr));
        if (got.cls != LLAMA_TENSOR_CLASS_MISC || got.layer_idx != -1) {
            fprintf(stderr, "FAIL: nullptr name -> (%s, layer=%d)\n", llama_tensor_class_name(got.cls), got.layer_idx);
            ++log.n_failures;
        }
    }

    // --- Per-arch full sweeps ------------------------------------------------
    bool ok = true;
    {
        const auto names = mistral7b_tensor_names(32);
        ok &= check_arch("Mistral 7B (LLAMA, 32 layers)", names, /*expected_misc*/ {});
        ok &= check_layer_indices("Mistral 7B", names);
    }
    {
        const auto names = gpt_oss_tensor_names(24);
        ok &= check_arch("GPT-OSS 20B (OPENAI_MOE, 24 layers)", names, /*expected_misc*/ {});
        ok &= check_layer_indices("GPT-OSS 20B", names);
    }
    {
        const auto names = qwen3_moe_tensor_names(48);
        ok &= check_arch("Qwen3-MoE (synthetic, 48 layers)", names, /*expected_misc*/ {});
        ok &= check_layer_indices("Qwen3-MoE", names);
    }
    {
        const auto names = deepseek2_tensor_names(60);
        ok &= check_arch("DeepSeek-V2 (synthetic, 60 layers)", names, /*expected_misc*/ {});
        ok &= check_layer_indices("DeepSeek-V2", names);
    }

    if (log.n_failures != 0 || !ok) {
        fprintf(stderr, "test-tensor-class: %d direct-case failure(s); arch sweep ok=%d\n", log.n_failures,
                static_cast<int>(ok));
        return 1;
    }

    printf("test-tensor-class: all checks passed\n");
    return 0;
}
