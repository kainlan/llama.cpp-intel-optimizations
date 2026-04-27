// Unit + golden tests for the PLACE-3 greedy budget fitter
// (src/llama-tensor-fitter.{h,cpp}).
//
// Bytes are synthetic, fixed per (class) so the fitter behavior is the only
// thing under test — no GGUF dependency.  The goldens then exercise the full
// PLACE-1 (classify) + PLACE-2 (priority) + PLACE-3 (fit) pipeline against
// archetypal Mistral 7B and GPT-OSS 20B tensor lists at 100/50/30% budgets.
//
// Set PLACE3_REGEN=1 to overwrite the golden files from live output instead
// of diffing.  Golden file paths arrive on argv (any order); each test run
// matches by substring.

#include "../src/llama-tensor-class.h"
#include "../src/llama-tensor-fitter.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct fail_log {
    int n_failures = 0;
};

// Synthetic byte sizes per class.  These are NOT the real Mistral/GPT-OSS
// tensor sizes; they're stable round numbers chosen so the fitter's
// fill/spill order is easy to verify by inspection in goldens.  PLACE-4
// will pass real GGUF sizes at runtime.
size_t synth_bytes(llama_tensor_class cls) {
    switch (cls) {
        case LLAMA_TENSOR_CLASS_EMBD_TOKEN:
        case LLAMA_TENSOR_CLASS_EMBD_OUTPUT:
            return 64 * 1024 * 1024;  //  64 MB
        case LLAMA_TENSOR_CLASS_OUTPUT_NORM:
        case LLAMA_TENSOR_CLASS_ATTN_NORM:
        case LLAMA_TENSOR_CLASS_FFN_NORM:
        case LLAMA_TENSOR_CLASS_POSITION_EMBD:
            return 1 * 1024 * 1024;  //   1 MB
        case LLAMA_TENSOR_CLASS_ATTN_Q:
        case LLAMA_TENSOR_CLASS_ATTN_K:
        case LLAMA_TENSOR_CLASS_ATTN_V:
        case LLAMA_TENSOR_CLASS_ATTN_O:
            return 16 * 1024 * 1024;  //  16 MB each (dense attn)
        case LLAMA_TENSOR_CLASS_ATTN_QKV:
            return 48 * 1024 * 1024;  //  48 MB (fused QKV)
        case LLAMA_TENSOR_CLASS_FFN_DENSE_GATE:
        case LLAMA_TENSOR_CLASS_FFN_DENSE_UP:
        case LLAMA_TENSOR_CLASS_FFN_DENSE_DOWN:
            return 32 * 1024 * 1024;  //  32 MB each (dense FFN)
        case LLAMA_TENSOR_CLASS_FFN_GATE_INP:
            return 1 * 1024 * 1024;   //   1 MB (router gate)
        case LLAMA_TENSOR_CLASS_FFN_MOE_GATE_EXPS:
        case LLAMA_TENSOR_CLASS_FFN_MOE_UP_EXPS:
        case LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS:
            return 128 * 1024 * 1024;  // 128 MB each (MoE expert blocks)
        case LLAMA_TENSOR_CLASS_FFN_SHARED_GATE:
        case LLAMA_TENSOR_CLASS_FFN_SHARED_UP:
        case LLAMA_TENSOR_CLASS_FFN_SHARED_DOWN:
            return 16 * 1024 * 1024;  //  16 MB each (shared expert)
        case LLAMA_TENSOR_CLASS_MISC:
            return 1 * 1024 * 1024;   //   1 MB
        case LLAMA_TENSOR_CLASS_COUNT:
            break;
    }
    return 1 * 1024 * 1024;
}

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
        names.push_back(p + "attn_q.weight");
        names.push_back(p + "attn_k.weight");
        names.push_back(p + "attn_v.weight");
        names.push_back(p + "attn_output.weight");
        names.push_back(p + "attn_norm.weight");
        names.push_back(p + "ffn_gate_inp.weight");
        names.push_back(p + "ffn_gate_exps.weight");
        names.push_back(p + "ffn_up_exps.weight");
        names.push_back(p + "ffn_down_exps.weight");
    }
    return names;
}

// Build the fitter input list from a name list at a given layer count.
std::vector<llama_tensor_placement_input> build_inputs(const std::vector<std::string> & names, int n_layers) {
    std::vector<llama_tensor_placement_input> inputs;
    inputs.reserve(names.size());
    for (const auto & n : names) {
        const llama_tensor_classification c    = llama_tensor_classify(n.c_str());
        const llama_tensor_priority       prio = llama_tensor_priority_for(c.cls, c.layer_idx, n_layers);
        inputs.push_back({ n, c.cls, c.layer_idx, prio, synth_bytes(c.cls) });
    }
    return inputs;
}

// Total byte count across an input list.
size_t total_bytes(const std::vector<llama_tensor_placement_input> & inputs) {
    size_t s = 0;
    for (const auto & i : inputs) {
        s += i.bytes;
    }
    return s;
}

// --- Unit cases ------------------------------------------------------------

void check(fail_log & log, bool cond, const char * what) {
    if (cond) {
        printf("OK: %s\n", what);
        return;
    }
    fprintf(stderr, "FAIL: %s\n", what);
    ++log.n_failures;
}

// Empty tier list -> error.
void test_empty_tiers(fail_log & log) {
    std::string err;
    auto        out = llama_tensor_fit(
        {
            { "x", LLAMA_TENSOR_CLASS_MISC, -1, LLAMA_TENSOR_PRIORITY_P0, 1 }
    },
        {}, &err);
    check(log, !out.has_value(), "empty tier list: returns nullopt");
    check(log, err.find("empty") != std::string::npos, "empty tier list: err_msg mentions empty");
}

// Empty inputs + one tier -> success with zero placements / zero used.
void test_empty_inputs(fail_log & log) {
    auto out = llama_tensor_fit(
        {
    },
        { { "device", 1024 } });
    check(log, out.has_value(), "empty inputs: returns summary");
    if (!out.has_value()) {
        return;
    }
    check(log, out->placements.empty(), "empty inputs: no placements");
    check(log, out->per_tier.size() == 1 && out->per_tier[0].used_bytes == 0, "empty inputs: tier residual = budget");
}

// P0 alone exceeds tier[0] budget -> hard error.
void test_p0_overflow(fail_log & log) {
    std::vector<llama_tensor_placement_input> inputs = {
        { "embed", LLAMA_TENSOR_CLASS_EMBD_TOKEN,  -1, LLAMA_TENSOR_PRIORITY_P0, 100 },
        { "norm",  LLAMA_TENSOR_CLASS_OUTPUT_NORM, -1, LLAMA_TENSOR_PRIORITY_P0, 100 },
    };
    std::string err;
    auto        out = llama_tensor_fit(inputs,
                                       {
                                    { "device", 50   },
                                    { "host",   1000 }
    },
                                       &err);
    check(log, !out.has_value(), "P0 overflow: returns nullopt");
    check(log, err.find("P0") != std::string::npos && err.find("device") != std::string::npos,
          "P0 overflow: err_msg names P0 and tier");
}

// Exact-fit budget for P0 only -> P0 on device, all P1/P2 spill to host.
void test_exact_p0_fit(fail_log & log) {
    std::vector<llama_tensor_placement_input> inputs = {
        { "p0_a", LLAMA_TENSOR_CLASS_ATTN_Q,            0, LLAMA_TENSOR_PRIORITY_P0, 100 },
        { "p0_b", LLAMA_TENSOR_CLASS_ATTN_K,            0, LLAMA_TENSOR_PRIORITY_P0, 50  },
        { "p1_a", LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS, 0, LLAMA_TENSOR_PRIORITY_P1, 200 },
        { "p2_a", LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS, 8, LLAMA_TENSOR_PRIORITY_P2, 200 },
    };
    auto out = llama_tensor_fit(inputs, {
                                            { "device", 150   },
                                            { "host",   10000 }
    });
    check(log, out.has_value(), "exact P0 fit: success");
    if (!out.has_value()) {
        return;
    }
    int n_dev = 0, n_host = 0;
    for (const auto & p : out->placements) {
        if (p.tier_name == "device") {
            ++n_dev;
        }
        if (p.tier_name == "host") {
            ++n_host;
        }
    }
    check(log, n_dev == 2 && n_host == 2, "exact P0 fit: 2 P0 on device, 2 non-P0 on host");
    check(log, out->per_tier[0].used_bytes == 150 && out->per_tier[0].residual_bytes == 0,
          "exact P0 fit: device tier exhausted exactly");
}

// Two-tier: P0 on tier[0] entirely, P1 spills to tier[1] entirely.
void test_two_tier_spill(fail_log & log) {
    std::vector<llama_tensor_placement_input> inputs = {
        { "p0",   LLAMA_TENSOR_CLASS_ATTN_Q,            0, LLAMA_TENSOR_PRIORITY_P0, 100 },
        { "p1_a", LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS, 0, LLAMA_TENSOR_PRIORITY_P1, 200 },
        { "p1_b", LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS, 1, LLAMA_TENSOR_PRIORITY_P1, 200 },
    };
    auto out = llama_tensor_fit(inputs, {
                                            { "B580", 100  },
                                            { "B50",  1000 }
    });
    check(log, out.has_value(), "two-tier: success");
    if (!out.has_value()) {
        return;
    }
    check(log, out->placements[0].tier_name == "B580", "two-tier: P0 on B580");
    int n_b50 = 0;
    for (const auto & p : out->placements) {
        if (p.tier_name == "B50") {
            ++n_b50;
        }
    }
    check(log, n_b50 == 2, "two-tier: both P1 on B50");
}

// Three-tier with cross-tier ordering: P0 fits tier[0]; P1 fills tier[1];
// P2 overflows to tier[2].
void test_three_tier_cascade(fail_log & log) {
    std::vector<llama_tensor_placement_input> inputs = {
        { "p0",   LLAMA_TENSOR_CLASS_ATTN_Q,            0, LLAMA_TENSOR_PRIORITY_P0, 50  },
        { "p1_a", LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS, 0, LLAMA_TENSOR_PRIORITY_P1, 100 },
        { "p1_b", LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS, 1, LLAMA_TENSOR_PRIORITY_P1, 100 },
        { "p2_a", LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS, 8, LLAMA_TENSOR_PRIORITY_P2, 100 },
        { "p2_b", LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS, 9, LLAMA_TENSOR_PRIORITY_P2, 100 },
    };
    // tier[0] holds P0 only (50). tier[1] holds two P1 (200). tier[2] holds two P2.
    auto out = llama_tensor_fit(inputs, {
                                            { "A", 50   },
                                            { "B", 200  },
                                            { "C", 1000 }
    });
    check(log, out.has_value(), "three-tier cascade: success");
    if (!out.has_value()) {
        return;
    }
    check(log, out->per_tier[0].n_tensors == 1, "three-tier cascade: A holds 1 tensor (P0)");
    check(log, out->per_tier[1].n_tensors == 2, "three-tier cascade: B holds 2 tensors (P1)");
    check(log, out->per_tier[2].n_tensors == 2, "three-tier cascade: C holds 2 tensors (P2)");
}

// SIZE_MAX last-tier sentinel -> always fits.
void test_unbounded_last_tier(fail_log & log) {
    std::vector<llama_tensor_placement_input> inputs = {
        { "p0", LLAMA_TENSOR_CLASS_ATTN_Q,            0, LLAMA_TENSOR_PRIORITY_P0, 1                           },
        { "p2", LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS, 8, LLAMA_TENSOR_PRIORITY_P2, 100ULL * 1024 * 1024 * 1024 },
    };
    auto out = llama_tensor_fit(inputs, {
                                            { "device", 100                     },
                                            { "mmap",   static_cast<size_t>(-1) }
    });
    check(log, out.has_value(), "unbounded last tier: huge tensor fits");
}

// No spill capacity provided AND non-P0 tensor doesn't fit -> error.
void test_non_p0_unfittable(fail_log & log) {
    std::vector<llama_tensor_placement_input> inputs = {
        { "p0",      LLAMA_TENSOR_CLASS_ATTN_Q,            0, LLAMA_TENSOR_PRIORITY_P0, 50   },
        { "p2_huge", LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS, 8, LLAMA_TENSOR_PRIORITY_P2, 1000 },
    };
    std::string err;
    auto        out = llama_tensor_fit(inputs,
                                       {
                                    { "device", 100 },
                                    { "host",   100 }
    },
                                       &err);
    check(log, !out.has_value(), "non-P0 unfittable: returns nullopt");
    check(log, err.find("p2_huge") != std::string::npos, "non-P0 unfittable: err_msg names tensor");
}

// MoE half-experts test: budget large enough for hot half + P0 only.
// Input: 16-layer model, MoE expert in every layer (8 hot, 8 cold).
// Each expert = 100, P0 = 200 total. tier[0] budget = 200 + 8*100 = 1000.
// Expected: all 8 P1 (hot) on device; all 8 P2 (cold) on host.
void test_half_experts(fail_log & log) {
    std::vector<llama_tensor_placement_input> inputs;
    inputs.push_back({ "embed", LLAMA_TENSOR_CLASS_EMBD_TOKEN, -1, LLAMA_TENSOR_PRIORITY_P0, 100 });
    inputs.push_back({ "norm", LLAMA_TENSOR_CLASS_OUTPUT_NORM, -1, LLAMA_TENSOR_PRIORITY_P0, 100 });
    for (int i = 0; i < 16; ++i) {
        const auto prio = (i < 8) ? LLAMA_TENSOR_PRIORITY_P1 : LLAMA_TENSOR_PRIORITY_P2;
        inputs.push_back({ "blk." + std::to_string(i) + ".exp", LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS, i, prio, 100 });
    }
    auto out = llama_tensor_fit(inputs, {
                                            { "device", 1000  },
                                            { "host",   10000 }
    });
    check(log, out.has_value(), "half-experts: success");
    if (!out.has_value()) {
        return;
    }
    int n_dev_p1 = 0, n_host_p2 = 0, n_host_p1 = 0;
    for (size_t i = 0; i < out->placements.size(); ++i) {
        const auto & p   = out->placements[i];
        const auto & in  = inputs[i];
        const bool   dev = (p.tier_name == "device");
        if (in.priority == LLAMA_TENSOR_PRIORITY_P1) {
            if (dev) {
                ++n_dev_p1;
            } else {
                ++n_host_p1;
            }
        }
        if (in.priority == LLAMA_TENSOR_PRIORITY_P2 && !dev) {
            ++n_host_p2;
        }
    }
    // Note: out->placements is in ALGORITHM order (P0 first, then sorted
    // non-P0), not input order — the per-index alignment above is a hack
    // for this specific test where input order matches output order by
    // construction (P0 first, hot P1 0..7, cold P2 8..15).
    check(log, n_dev_p1 == 8 && n_host_p1 == 0 && n_host_p2 == 8, "half-experts: hot on device, cold on host");
}

// Determinism: shuffling inputs produces the same placements (by name).
void test_determinism(fail_log & log) {
    std::vector<llama_tensor_placement_input> inputs_a = {
        { "a", LLAMA_TENSOR_CLASS_ATTN_Q,            0, LLAMA_TENSOR_PRIORITY_P0, 10  },
        { "b", LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS, 0, LLAMA_TENSOR_PRIORITY_P1, 100 },
        { "c", LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS, 1, LLAMA_TENSOR_PRIORITY_P1, 100 },
    };
    std::vector<llama_tensor_placement_input> inputs_b = { inputs_a[2], inputs_a[0], inputs_a[1] };
    auto                                      out_a    = llama_tensor_fit(inputs_a, {
                                                { "d", 110  },
                                                { "h", 1000 }
    });
    auto                                      out_b    = llama_tensor_fit(inputs_b, {
                                                { "d", 110  },
                                                { "h", 1000 }
    });
    check(log, out_a.has_value() && out_b.has_value(), "determinism: both succeed");
    if (!out_a.has_value() || !out_b.has_value()) {
        return;
    }
    bool same = (out_a->placements.size() == out_b->placements.size());
    for (size_t i = 0; same && i < out_a->placements.size(); ++i) {
        same = (out_a->placements[i].name == out_b->placements[i].name) &&
               (out_a->placements[i].tier_name == out_b->placements[i].tier_name);
    }
    check(log, same, "determinism: same placement order regardless of input order");
}

// --- Golden harness --------------------------------------------------------

std::string render_golden(const char * label, const std::vector<std::string> & names, int n_layers, int budget_pct) {
    auto         inputs = build_inputs(names, n_layers);
    const size_t total  = total_bytes(inputs);
    const size_t budget = (total * static_cast<size_t>(budget_pct)) / 100;
    std::string  err;
    auto         out = llama_tensor_fit(inputs,
                                        {
                                    { "device", budget                  },
                                    { "host",   static_cast<size_t>(-1) }
    },
                                        &err);

    std::ostringstream oss;
    oss << "# name|class|layer|prio|bytes|tier  (PLACE-3 golden, " << label << ", n_layers=" << n_layers
        << ", budget=" << budget_pct << "% of " << total << " bytes = " << budget << " bytes)\n";
    if (!out.has_value()) {
        // Dense models with no spillable tier (per-tensor fitter cannot relocate
        // P0) hit this path at constrained budgets — that's the expected
        // behavior PLACE-6 will solve via whole-layer atomic packing.
        oss << "# fit_failed: " << err << '\n';
        return oss.str();
    }
    // Render in input order so the diff is human-readable (matches the
    // PLACE-2 golden's input-order convention).
    std::vector<std::string> assigned(inputs.size());
    for (const auto & p : out->placements) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i].name == p.name) {
                assigned[i] = p.tier_name;
                break;
            }
        }
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto & in = inputs[i];
        oss << in.name << '|' << llama_tensor_class_name(in.cls) << '|' << in.layer_idx << '|'
            << llama_tensor_priority_name(in.priority) << '|' << in.bytes << '|' << assigned[i] << '\n';
    }
    // Per-tier summary footer for at-a-glance visibility.
    oss << "# per-tier:";
    for (const auto & ts : out->per_tier) {
        oss << " " << ts.tier_name << "(used=" << ts.used_bytes << ",residual=" << ts.residual_bytes
            << ",n=" << ts.n_tensors << ")";
    }
    oss << '\n';
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

bool diff_golden(const char * label, const std::string & path, const std::string & actual) {
    const char * regen_env = std::getenv("PLACE3_REGEN");
    if (regen_env != nullptr && regen_env[0] != '\0' && regen_env[0] != '0') {
        if (!write_file(path, actual)) {
            fprintf(stderr, "FAIL: %s: could not write golden file '%s'\n", label, path.c_str());
            return false;
        }
        printf("REGEN: %s: wrote %zu bytes to '%s'\n", label, actual.size(), path.c_str());
        return true;
    }

    std::string expected;
    if (!read_file(path, expected)) {
        fprintf(stderr, "FAIL: %s: could not read golden file '%s' (run with PLACE3_REGEN=1 to bootstrap)\n", label,
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
        if (a_ok && e_ok && a_line == e_line) {
            continue;
        }
        fprintf(stderr, "  line %d:\n    actual:   %s\n    expected: %s\n", lineno, a_ok ? a_line.c_str() : "<missing>",
                e_ok ? e_line.c_str() : "<missing>");
        ++printed;
        if (!a_ok || !e_ok) {
            continue;
        }
    }
    fprintf(stderr, "  (rerun with PLACE3_REGEN=1 to overwrite the golden file from current output)\n");
    return false;
}

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

    test_empty_tiers(log);
    test_empty_inputs(log);
    test_p0_overflow(log);
    test_exact_p0_fit(log);
    test_two_tier_spill(log);
    test_three_tier_cascade(log);
    test_unbounded_last_tier(log);
    test_non_p0_unfittable(log);
    test_half_experts(log);
    test_determinism(log);

    bool ok = (log.n_failures == 0);

    // Golden tests: 3 budget percentages × 2 archs.  argv carries the golden
    // file paths; match by substring so the CMake call site can list them in
    // any order.
    struct {
        const char * label;
        const char * needle;
        int          n_layers;
        int          budget_pct;
        std::vector<std::string> (*names)(int);
    } golden_cases[] = {
        { "Mistral 7B 100%",  "place3-mistral7b-100pct",   32, 100, mistral7b_tensor_names },
        { "Mistral 7B 50%",   "place3-mistral7b-50pct",    32, 50,  mistral7b_tensor_names },
        { "Mistral 7B 30%",   "place3-mistral7b-30pct",    32, 30,  mistral7b_tensor_names },
        { "GPT-OSS 20B 100%", "place3-gpt-oss-20b-100pct", 24, 100, gpt_oss_tensor_names   },
        { "GPT-OSS 20B 50%",  "place3-gpt-oss-20b-50pct",  24, 50,  gpt_oss_tensor_names   },
        { "GPT-OSS 20B 30%",  "place3-gpt-oss-20b-30pct",  24, 30,  gpt_oss_tensor_names   },
    };

    for (const auto & gc : golden_cases) {
        const char * path = find_arg(argc, argv, gc.needle);
        if (path == nullptr) {
            fprintf(stderr, "FAIL: golden file path missing from argv (needle='%s')\n", gc.needle);
            ok = false;
            continue;
        }
        ok &= diff_golden(gc.label, path, render_golden(gc.label, gc.names(gc.n_layers), gc.n_layers, gc.budget_pct));
    }

    if (!ok || log.n_failures > 0) {
        fprintf(stderr, "place-3 fitter tests: FAIL (%d unit failures)\n", log.n_failures);
        return 1;
    }
    printf("place-3 fitter tests: PASS\n");
    return 0;
}
