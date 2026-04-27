// Unit tests for the tensor-name classifier (PLACE-1) and priority policy
// (PLACE-2). Golden files live in tests/golden/place2-*.txt; the test
// program receives their paths via argv[1..] (any order) — set PLACE2_REGEN=1
// in env to overwrite the golden files from the live output instead of
// diffing.

#include "../src/llama-tensor-class.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
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
    // token_types.weight ships with some Mistral 7B GGUFs (input token-type IDs);
    // it has no role in placement so it must classify as MISC.
    names.push_back("token_types.weight");

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
    std::set<std::string> actual_misc;
    int                   n_unexpected = 0;
    for (const auto & n : names) {
        const llama_tensor_classification c = llama_tensor_classify(n.c_str());
        if (c.cls != LLAMA_TENSOR_CLASS_MISC) {
            continue;
        }
        actual_misc.insert(n);
        if (expected_misc.count(n) > 0) {
            continue;
        }
        fprintf(stderr, "FAIL: %s: unexpected MISC for tensor '%s'\n", arch_name, n.c_str());
        ++n_unexpected;
    }
    int n_missing = 0;
    for (const auto & e : expected_misc) {
        if (actual_misc.count(e) > 0) {
            continue;
        }
        fprintf(stderr, "FAIL: %s: expected MISC tensor '%s' was not classified MISC (or missing from input list)\n",
                arch_name, e.c_str());
        ++n_missing;
    }
    if (n_unexpected == 0 && n_missing == 0) {
        printf("OK: %s: all %zu tensors classified (MISC set matches exactly: %zu)\n", arch_name, names.size(),
               expected_misc.size());
    }
    return n_unexpected == 0 && n_missing == 0;
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
        // Mirror the impl's strtol+end-ptr validation rather than atoi: this
        // sweep enumerates synthetic names and a malformed entry should fail
        // the test loudly, not silently parse to 0.
        const std::string layer_str = n.substr(4, end - 4);
        char *            end_ptr   = nullptr;
        const long        layer_val = std::strtol(layer_str.c_str(), &end_ptr, 10);
        if (end_ptr == layer_str.c_str() || *end_ptr != '\0' || layer_val < 0) {
            fprintf(stderr, "FAIL: %s: malformed synthetic name '%s' (layer-extract failed)\n", arch_name, n.c_str());
            std::abort();
        }
        const int                         expected = static_cast<int>(layer_val);
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

// --- PLACE-2 priority policy: helpers ---------------------------------------

struct prio_case {
    llama_tensor_class    cls;
    int                   layer_idx;
    int                   n_layers;
    llama_tensor_priority expected;
};

void check_prio(fail_log & log, const prio_case & c) {
    const llama_tensor_priority got = llama_tensor_priority_for(c.cls, c.layer_idx, c.n_layers);
    if (got == c.expected) {
        return;
    }
    fprintf(stderr, "FAIL: prio(%s, layer=%d, n=%d) -> got %s, expected %s\n", llama_tensor_class_name(c.cls),
            c.layer_idx, c.n_layers, llama_tensor_priority_name(got), llama_tensor_priority_name(c.expected));
    ++log.n_failures;
}

// Render the (name | class | layer | tier) golden text for one arch by
// classifying every name and applying the priority policy. layer_idx is the
// classifier's output (-1 for non per-layer tensors) — the policy gets
// layer_idx + n_layers exactly as a real caller would supply them.
std::string render_golden(const std::vector<std::string> & names, int n_layers) {
    std::ostringstream oss;
    oss << "# name|class|layer|tier  (PLACE-2 golden, n_layers=" << n_layers << ")\n";
    for (const auto & n : names) {
        const llama_tensor_classification c    = llama_tensor_classify(n.c_str());
        const llama_tensor_priority       prio = llama_tensor_priority_for(c.cls, c.layer_idx, n_layers);
        oss << n << '|' << llama_tensor_class_name(c.cls) << '|' << c.layer_idx << '|'
            << llama_tensor_priority_name(prio) << '\n';
    }
    return oss.str();
}

bool read_file(const std::string & path, std::string & out) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

bool write_file(const std::string & path, const std::string & content) {
    std::ofstream out(path);
    if (!out.is_open()) {
        return false;
    }
    out << content;
    return out.good();
}

// Compare actual vs expected line-by-line; on mismatch, print the first few
// diverging lines with line numbers so the developer can fix the policy
// or refresh the golden file (PLACE2_REGEN=1).
bool diff_golden(const char * label, const std::string & path, const std::string & actual) {
    if (std::getenv("PLACE2_REGEN") != nullptr) {
        if (!write_file(path, actual)) {
            fprintf(stderr, "FAIL: %s: could not write golden file '%s'\n", label, path.c_str());
            return false;
        }
        printf("REGEN: %s: wrote %zu bytes to '%s'\n", label, actual.size(), path.c_str());
        return true;
    }

    std::string expected;
    if (!read_file(path, expected)) {
        fprintf(stderr, "FAIL: %s: could not read golden file '%s' (run with PLACE2_REGEN=1 to bootstrap)\n", label,
                path.c_str());
        return false;
    }

    if (expected == actual) {
        printf("OK: %s: golden matches '%s'\n", label, path.c_str());
        return true;
    }

    fprintf(stderr, "FAIL: %s: golden mismatch vs '%s'\n", label, path.c_str());
    std::istringstream a_in(actual);
    std::istringstream e_in(expected);
    std::string        a_line;
    std::string        e_line;
    int                lineno   = 0;
    int                printed  = 0;
    const int          max_show = 10;
    while (printed < max_show) {
        const bool a_ok = static_cast<bool>(std::getline(a_in, a_line));
        const bool e_ok = static_cast<bool>(std::getline(e_in, e_line));
        ++lineno;
        if (!a_ok && !e_ok) {
            break;
        }
        if (!a_ok) {
            fprintf(stderr, "  line %d:\n    actual:   <missing>\n    expected: %s\n", lineno, e_line.c_str());
            ++printed;
            continue;
        }
        if (!e_ok) {
            fprintf(stderr, "  line %d:\n    actual:   %s\n    expected: <missing>\n", lineno, a_line.c_str());
            ++printed;
            continue;
        }
        if (a_line != e_line) {
            fprintf(stderr, "  line %d:\n    actual:   %s\n    expected: %s\n", lineno, a_line.c_str(), e_line.c_str());
            ++printed;
        }
    }
    fprintf(stderr, "  (rerun with PLACE2_REGEN=1 to overwrite the golden file from current output)\n");
    return false;
}

// Pick the argv entry whose basename contains needle. argv has no defined
// order in the CMake call site; we match by substring so adding more golden
// files later doesn't require argv reshuffling.
const char * find_arg(int argc, char ** argv, const char * needle) {
    for (int i = 1; i < argc; ++i) {
        if (std::strstr(argv[i], needle) != nullptr) {
            return argv[i];
        }
    }
    return nullptr;
}

}  // namespace

int main(int argc, char ** argv) {
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
        // Layer-index edge cases: empty index between dots, and negative index.
        { "blk..attn_q.weight",               LLAMA_TENSOR_CLASS_MISC,              -1  },
        { "blk.-1.attn_q.weight",             LLAMA_TENSOR_CLASS_MISC,              -1  },
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
        ok &= check_arch("Mistral 7B (LLAMA, 32 layers)", names,
                         /*expected_misc*/ { "token_types.weight" });
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

    // --- PLACE-2 priority unit cases ----------------------------------------
    // Every class should map to its expected tier across two representative
    // n_layers (24 like GPT-OSS, 32 like Mistral). Routed-expert classes are
    // additionally exercised at the L/2 boundary and across odd n_layers (33)
    // to lock the floor convention.
    const prio_case prio_cases[] = {
        // P0 classes — tier independent of layer/n_layers.
        { LLAMA_TENSOR_CLASS_ATTN_Q,            0,  32, LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_ATTN_K,            0,  32, LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_ATTN_V,            0,  32, LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_ATTN_O,            0,  32, LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_ATTN_QKV,          0,  32, LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_ATTN_NORM,         0,  32, LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_FFN_GATE_INP,      0,  32, LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_FFN_DENSE_GATE,    0,  32, LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_FFN_DENSE_UP,      0,  32, LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_FFN_DENSE_DOWN,    0,  32, LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_FFN_SHARED_GATE,   0,  32, LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_FFN_SHARED_UP,     0,  32, LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_FFN_SHARED_DOWN,   0,  32, LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_FFN_NORM,          0,  32, LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_EMBD_TOKEN,        -1, 0,  LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_EMBD_OUTPUT,       -1, 0,  LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_OUTPUT_NORM,       -1, 0,  LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_POSITION_EMBD,     -1, 0,  LLAMA_TENSOR_PRIORITY_P0 },
        { LLAMA_TENSOR_CLASS_MISC,              0,  32, LLAMA_TENSOR_PRIORITY_P0 },

        // Routed experts at n_layers=24 (boundary=12).
        { LLAMA_TENSOR_CLASS_FFN_MOE_GATE_EXPS, 0,  24, LLAMA_TENSOR_PRIORITY_P1 },
        { LLAMA_TENSOR_CLASS_FFN_MOE_UP_EXPS,   11, 24, LLAMA_TENSOR_PRIORITY_P1 },
        { LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS, 12, 24, LLAMA_TENSOR_PRIORITY_P2 }, // boundary -> P2
        { LLAMA_TENSOR_CLASS_FFN_MOE_GATE_EXPS, 23, 24, LLAMA_TENSOR_PRIORITY_P2 },

        // Routed experts at n_layers=32 (boundary=16).
        { LLAMA_TENSOR_CLASS_FFN_MOE_GATE_EXPS, 0,  32, LLAMA_TENSOR_PRIORITY_P1 },
        { LLAMA_TENSOR_CLASS_FFN_MOE_UP_EXPS,   15, 32, LLAMA_TENSOR_PRIORITY_P1 },
        { LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS, 16, 32, LLAMA_TENSOR_PRIORITY_P2 }, // boundary -> P2
        { LLAMA_TENSOR_CLASS_FFN_MOE_GATE_EXPS, 31, 32, LLAMA_TENSOR_PRIORITY_P2 },

        // Odd n_layers=33 (boundary=16): P1 covers 16 layers, P2 covers 17.
        { LLAMA_TENSOR_CLASS_FFN_MOE_GATE_EXPS, 15, 33, LLAMA_TENSOR_PRIORITY_P1 },
        { LLAMA_TENSOR_CLASS_FFN_MOE_GATE_EXPS, 16, 33, LLAMA_TENSOR_PRIORITY_P2 },
    };
    for (const auto & c : prio_cases) {
        check_prio(log, c);
    }

    // --- PLACE-2 golden diff -------------------------------------------------
    const char * mistral_path = find_arg(argc, argv, "place2-mistral7b");
    const char * gptoss_path  = find_arg(argc, argv, "place2-gpt-oss-20b");
    if (mistral_path == nullptr || gptoss_path == nullptr) {
        fprintf(stderr, "FAIL: golden file paths missing from argv (mistral=%s, gpt-oss=%s)\n",
                mistral_path ? mistral_path : "<null>", gptoss_path ? gptoss_path : "<null>");
        ok = false;
    } else {
        ok &= diff_golden("Mistral 7B golden", mistral_path, render_golden(mistral7b_tensor_names(32), 32));
        ok &= diff_golden("GPT-OSS 20B golden", gptoss_path, render_golden(gpt_oss_tensor_names(24), 24));
    }

    if (log.n_failures != 0 || !ok) {
        fprintf(stderr, "test-tensor-class: %d direct-case failure(s); arch sweep ok=%d\n", log.n_failures,
                static_cast<int>(ok));
        return 1;
    }

    printf("test-tensor-class: all checks passed\n");
    return 0;
}
