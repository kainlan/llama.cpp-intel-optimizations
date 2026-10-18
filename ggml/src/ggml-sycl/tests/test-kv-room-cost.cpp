// test-kv-room-cost: the KV context room's per-layer cost (kv-room-cost.hpp; llama.cpp-w013, llama.cpp-7fmt) on the
// shapes it was measured on. Host only: no device, no model, no SYCL.
//
// Each arm is one attention layer of a real model at a real context, weighed with the header's own functions and the
// shipped defaults, so a change to a constant, the split mapping, the expert-rate rule or the arithmetic moves a
// verdict or a printed term here. The verdicts the measurements fixed (ticket comments c-ncj4, c-plz8, 2026-10-18):
//   - GPT-OSS 20B on the B50 without -c refuses its room (holding it decoded 9.9 t/s against 23.5 refused);
//   - Qwen3.8 IQ3_XXS on the B70 without -c holds it (12.1 t/s held against 9.9 refused);
//   - both hold at -c 4096, where the room is small;
//   - and both keep those verdicts at the gate's real fill, 128 cells, as at the planner's assumed 50 %.
// Those four constraints admit a split cost s with 175 < s < 1495 us; the window edges are pinned below.
//
// The w013 formula (one 40 GB/s rate, no split term) holds GPT-OSS's default room; a per-type rate without the split
// term refuses GPT-OSS's -c 4096 room. Both fail here.
//
// Qwen3.8's seven routed-expert types were measured as one mix, so the table gives each the mix's 22 GB/s; the mix's
// byte shares are pinned below as the catalog row states them, and a mix with one of them replaced by an unmeasured
// type refuses its default room, naming that type.

#include "kv-room-cost.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

using namespace ggml_sycl;

namespace {

int g_cases    = 0;
int g_failures = 0;

// The planning n_ctx every arm's load planned with: the room is the context's KV beyond it.
constexpr uint32_t PLANNER_N_CTX = 512;

// The fill the GPT-OSS and Qwen gates actually reach: a short prompt plus a short answer.
constexpr uint32_t SHORT_FILL_CELLS = 128;

struct model_shape {
    const char * name;
    size_t       kv_bytes_per_cell;  // one attention layer's KV per cell, its indexer key cache included
    bool         has_indexer;
    uint32_t     n_ctx_default;      // the model's default context: n_ctx_train padded to 256
    int          n_experts;
    int          n_expert_used;
    // The device's routed-expert weight bytes by type (from the GGUF tensor infos, biases excluded as
    // infer_tensor_usage excludes them).
    std::vector<std::pair<ggml_type, size_t>> experts;
};

// GPT-OSS 20B MXFP4: 8 KV heads x 64 x f16 x (K + V) per full-attention layer, 4 of 32 experts per token.
const model_shape GPTOSS_20B = {
    "gpt-oss-20b", 2048, false, 131072, 32, 4, { { GGML_TYPE_MXFP4, 10152345600ull } },
};

// GPT-OSS 120B MXFP4: the same attention, 4 of 128 experts per token. Its verdicts are the model's prediction, not
// a measurement. The expert bytes are a placeholder: with one type the rate is that type's, whatever the bytes.
const model_shape GPTOSS_120B = {
    "gpt-oss-120b", 2048, false, 131072, 128, 4, { { GGML_TYPE_MXFP4, 60000000000ull } },
};

// Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS: 2304 bytes per cell per attention layer, an indexer key cache in each, 10 of
// 512 experts per token. The file named IQ3_XXS mixes seven expert types; the bytes are the file's.
const model_shape QWEN38 = {
    "qwen3.8-iq3_xxs",
    2304,
    true,
    262144,
    512,
    10,
    {
      { GGML_TYPE_IQ3_S, 9371648000ull },
      { GGML_TYPE_IQ4_NL, 8493465600ull },
      { GGML_TYPE_Q2_0, 7077888000ull },
      { GGML_TYPE_IQ2_S, 5373952000ull },
      { GGML_TYPE_IQ2_XS, 4849664000ull },
      { GGML_TYPE_IQ2_XXS, 3892838400ull },
      { GGML_TYPE_IQ3_XXS, 3853516800ull },
      },
};

// GPT-OSS 20B's attention with Q8_0 experts, a type the table does not measure.
const model_shape Q8_EXPERTS = {
    "q8_0-experts", 2048, false, 131072, 32, 4, { { GGML_TYPE_Q8_0, 10152345600ull } },
};

// Half MXFP4, half Q8_0 bytes: the harmonic mean of 5.6 and the 40 GB/s fallback.
const model_shape MIXED_EXPERTS = {
    "mxfp4+q8_0-experts",
    2048,
    false,
    131072,
    32,
    4,
    { { GGML_TYPE_MXFP4, 5000000000ull }, { GGML_TYPE_Q8_0, 5000000000ull } },
};

// A dense model: no routed experts, so its room displaces nothing.
const model_shape DENSE = { "dense", 2048, false, 131072, 0, 0, {} };

// Qwen3.8's mix with its iq2_xxs gate/up bytes in q4_K, a type the table does not measure.
const model_shape QWEN38_Q4K = {
    "qwen3.8-with-q4_K",
    2304,
    true,
    262144,
    512,
    10,
    {
      { GGML_TYPE_IQ3_S, 9371648000ull },
      { GGML_TYPE_IQ4_NL, 8493465600ull },
      { GGML_TYPE_Q2_0, 7077888000ull },
      { GGML_TYPE_IQ2_S, 5373952000ull },
      { GGML_TYPE_IQ2_XS, 4849664000ull },
      { GGML_TYPE_Q4_K, 3892838400ull },
      { GGML_TYPE_IQ3_XXS, 3853516800ull },
      },
};

// GPT-OSS 20B's attention with half its expert bytes in iq3_xxs: two measurements in one source field.
const model_shape TWO_SOURCES = {
    "mxfp4+iq3_xxs-experts",
    2048,
    false,
    131072,
    32,
    4,
    { { GGML_TYPE_MXFP4, 5000000000ull }, { GGML_TYPE_IQ3_XXS, 5000000000ull } },
};

// ggml_type_name's names for the types the shapes use; the header takes the function, so this test links no ggml.
const char * test_type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:
            return "q8_0";
        case GGML_TYPE_Q4_K:
            return "q4_K";
        case GGML_TYPE_IQ2_XXS:
            return "iq2_xxs";
        case GGML_TYPE_IQ2_XS:
            return "iq2_xs";
        case GGML_TYPE_IQ3_XXS:
            return "iq3_xxs";
        case GGML_TYPE_IQ4_NL:
            return "iq4_nl";
        case GGML_TYPE_IQ3_S:
            return "iq3_s";
        case GGML_TYPE_IQ2_S:
            return "iq2_s";
        case GGML_TYPE_MXFP4:
            return "mxfp4";
        case GGML_TYPE_Q2_0:
            return "q2_0";
        default:
            return "?";
    }
}

struct arm {
    uint32_t n_ctx_context;
    bool     is_default;  // the caller named no n_ctx
    double   fill_pct        = PLACEMENT_CTX_FILL_PCT_DEFAULT;
    double   host_split_us   = PLACEMENT_HOST_SPLIT_US_DEFAULT;
    double   cpu_expert_gbps = PLACEMENT_CPU_EXPERT_GBPS_DEFAULT;
    bool     cpu_expert_set  = false;  // GGML_SYCL_PLAN_CPU_EXPERT_GBPS was given
};

arm arm_default(const model_shape & m) {
    return { m.n_ctx_default, true };
}

arm arm_explicit(uint32_t n_ctx) {
    return { n_ctx, false };
}

arm arm_short_fill(const model_shape & m) {
    arm a      = arm_default(m);
    a.fill_pct = SHORT_FILL_CELLS * 100.0 / m.n_ctx_default;
    return a;
}

struct weighed {
    kv_room_expert_rate rate;
    kv_room_cost_terms  cost;
    uint32_t            n_splits;
};

// What the planner does for one layer: the device's expert rate, then the layer's A and B.
weighed weigh(const model_shape & m, const arm & a) {
    kv_room_expert_types types;
    for (const auto & e : m.experts) {
        types.add(e.first, e.second);
    }
    weighed w;
    w.rate     = kv_room_expert_rate_for(types, a.cpu_expert_gbps, a.cpu_expert_set, a.is_default);
    w.n_splits = placement_host_attn_splits(m.has_indexer);

    kv_room_cost_inputs in;
    in.host_read_bytes = m.kv_bytes_per_cell * kv_room_filled_cells(a.n_ctx_context, a.fill_pct);
    in.n_host_splits   = w.n_splits;
    in.room_bytes      = m.kv_bytes_per_cell * (a.n_ctx_context - PLANNER_N_CTX);
    in.n_experts       = m.n_experts;
    in.n_expert_used   = m.n_expert_used;
    in.host_attn_gbps  = PLACEMENT_HOST_ATTN_GBPS_DEFAULT;
    in.host_split_us   = a.host_split_us;
    in.cpu_expert_gbps = w.rate.gbps;
    in.priced          = w.rate.priced;
    w.cost             = kv_room_weigh(in);
    return w;
}

bool near(double got, double want, double tol) {
    return std::isfinite(got) && std::fabs(got - want) <= tol;
}

// One arm: its verdict, and A and B to the 0.1 us the room's line prints (a negative want skips that term).
void expect(const char * name, const model_shape & m, const arm & a, bool want_held, double want_a, double want_b) {
    ++g_cases;
    const weighed w  = weigh(m, a);
    const bool    ok = w.cost.held == want_held && (want_a < 0.0 || near(w.cost.host_attn_us, want_a, 0.05)) &&
                    (want_b < 0.0 || near(w.cost.expert_us, want_b, 0.05));
    std::printf(
        "%s %s (%s, n_ctx_context=%u%s, fill %.4g%%, split %.4g us): %s, want %s; A %.1f us/token "
        "(%.1f fill + %u split(s)), want %.1f; B %.1f us/token (cpu-expert %.4g GB/s%s), want %.1f\n",
        ok ? "ok  " : "FAIL", name, m.name, a.n_ctx_context, a.is_default ? " default" : "", a.fill_pct,
        a.host_split_us, w.cost.held ? "held" : "refused", want_held ? "held" : "refused", w.cost.host_attn_us,
        w.cost.fill_us, w.n_splits, want_a, w.cost.expert_us, w.rate.gbps, w.rate.priced ? "" : ", unpriced", want_b);
    g_failures += ok ? 0 : 1;
}

void expect_true(const char * name, bool cond) {
    ++g_cases;
    std::printf("%s %s\n", cond ? "ok  " : "FAIL", name);
    g_failures += cond ? 0 : 1;
}

void expect_text(const char * name, const std::string & got, const char * want) {
    ++g_cases;
    const bool ok = got == want;
    if (ok) {
        std::printf("ok   %s: \"%s\"\n", name, got.c_str());
    } else {
        std::printf("FAIL %s: \"%s\", want \"%s\"\n", name, got.c_str(), want);
    }
    g_failures += ok ? 0 : 1;
}

std::string rate_source(const model_shape & m, const arm & a) {
    return kv_room_rate_source(weigh(m, a).rate, test_type_name);
}

void test_measured_arms() {
    // The four arms at the planner's assumed fill, then the two default arms at the gate's real fill.
    expect("gpt-oss-20b without -c refuses its room", GPTOSS_20B, arm_default(GPTOSS_20B), false, 4723.9, 5968.5);
    expect("gpt-oss-20b at -c 4096 holds its room", GPTOSS_20B, arm_explicit(4096), true, 389.8, 163.8);
    expect("qwen3.8 without -c holds its room", QWEN38, arm_default(QWEN38), true, 10816.3, 535.2);
    expect("qwen3.8 at -c 4096 holds its room", QWEN38, arm_explicit(4096), true, 907.3, 7.3);
    expect("qwen3.8 without -c holds its room at the gate's fill", QWEN38, arm_short_fill(QWEN38), true, 759.8, 535.2);
    expect("gpt-oss-20b without -c refuses its room at the gate's fill", GPTOSS_20B, arm_short_fill(GPTOSS_20B), false,
           258.7, 5968.5);
}

void test_gptoss_120b() {
    // A quarter of the 20B's hit rate: the room is worth its experts at the assumed fill, not at a short one.
    expect("gpt-oss-120b without -c holds its room at fill 50", GPTOSS_120B, arm_default(GPTOSS_120B), true, 4723.9,
           1492.1);
    expect("gpt-oss-120b without -c refuses its room at the gate's fill", GPTOSS_120B, arm_short_fill(GPTOSS_120B),
           false, 258.7, 1492.1);
}

void test_split_window() {
    // The lower edge is Qwen's default room at the gate's fill (s > 175.11), the upper edge GPT-OSS 20B's default room
    // at fill 50 (s < 1494.53).
    arm a           = arm_short_fill(QWEN38);
    a.host_split_us = 174.0;
    expect("qwen3.8 at the gate's fill refuses below the window (s = 174)", QWEN38, a, false, 531.8, 535.2);
    a.host_split_us = 176.0;
    expect("qwen3.8 at the gate's fill holds inside the window (s = 176)", QWEN38, a, true, 537.8, 535.2);
    arm g           = arm_default(GPTOSS_20B);
    g.host_split_us = 1494.0;
    expect("gpt-oss-20b at fill 50 refuses inside the window (s = 1494)", GPTOSS_20B, g, false, 5967.9, 5968.5);
    g.host_split_us = 1496.0;
    expect("gpt-oss-20b at fill 50 holds above the window (s = 1496)", GPTOSS_20B, g, true, 5969.9, 5968.5);
    expect_true("the shipped split cost is inside the window",
                PLACEMENT_HOST_SPLIT_US_DEFAULT > 175.11 && PLACEMENT_HOST_SPLIT_US_DEFAULT < 1494.53);
}

void test_split_mapping() {
    expect_true("a layer with an indexer key cache pays 3 splits", placement_host_attn_splits(true) == 3);
    expect_true("any other layer with its own KV pays 1 split", placement_host_attn_splits(false) == 1);
}

void test_expert_rates() {
    const placement_cpu_expert_rate mxfp4 = placement_cpu_expert_rate_measured(GGML_TYPE_MXFP4);
    expect_true("mxfp4 is measured at 5.6 GB/s", mxfp4.gbps == 5.6);
    expect_text("mxfp4's rate names its measurement", mxfp4.source, "gpt-oss-20b 2026-10-18 (c-ncj4)");
    const ggml_type mix[] = { GGML_TYPE_IQ3_S,  GGML_TYPE_IQ4_NL,  GGML_TYPE_Q2_0,   GGML_TYPE_IQ2_S,
                              GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS };
    for (const ggml_type t : mix) {
        const placement_cpu_expert_rate r      = placement_cpu_expert_rate_measured(t);
        const std::string               type   = test_type_name(t);
        const std::string               rate   = type + " is the qwen3.8 mix's 22 GB/s";
        const std::string               source = type + "'s rate names the mix";
        expect_true(rate.c_str(), r.gbps == 22.0);
        expect_text(source.c_str(), r.source != nullptr ? r.source : "", "qwen38-gsq-rco-mix 2026-10-18 (c-plz8)");
    }
    const placement_cpu_expert_rate q8 = placement_cpu_expert_rate_measured(GGML_TYPE_Q8_0);
    expect_true("q8_0 is not measured and names no measurement", q8.gbps == 0.0 && q8.source == nullptr);
    const placement_cpu_expert_rate q4k = placement_cpu_expert_rate_measured(GGML_TYPE_Q4_K);
    expect_true("q4_K is not measured and names no measurement", q4k.gbps == 0.0 && q4k.source == nullptr);

    const weighed gptoss = weigh(GPTOSS_20B, arm_default(GPTOSS_20B));
    expect_true("gpt-oss-20b's experts are read at mxfp4's rate", near(gptoss.rate.gbps, 5.6, 1e-9));
    expect_true("gpt-oss-20b's experts are all measured",
                gptoss.rate.types.unmeasured_mask == 0 && gptoss.rate.types.measured_mask == (1ull << GGML_TYPE_MXFP4));

    const weighed qwen = weigh(QWEN38, arm_default(QWEN38));
    expect_true("qwen3.8's seven expert types are read at the measured 22 GB/s", near(qwen.rate.gbps, 22.0, 1e-9));
    expect_true("qwen3.8's expert types are all measured", qwen.rate.priced && qwen.rate.types.unmeasured_mask == 0);

    // Half the bytes at 5.6 and half at the 40 GB/s fallback: 1 / (0.5 / 5.6 + 0.5 / 40).
    const weighed mixed = weigh(MIXED_EXPERTS, arm_explicit(131072));
    expect_true("a mixed device's rate is the byte-weighted harmonic mean",
                near(mixed.rate.gbps, 1.0 / (0.5 / 5.6 + 0.5 / 40.0), 1e-9));

    kv_room_expert_types untyped;
    untyped.add(GGML_TYPE_COUNT, 1000);
    expect_true("an untyped expert tensor counts as unmeasured",
                untyped.unmeasured_bytes == 1000 && untyped.unmeasured_mask == (1ull << GGML_TYPE_COUNT));
}

void test_qwen_mix() {
    // The mix's byte shares, as the catalog row states them (GGUF tensor infos, biases excluded); the down weights are
    // iq4_nl and q2_0, the gate/up weights the other five.
    const double shares[] = { 21.8, 19.8, 16.5, 12.5, 11.3, 9.1, 9.0 };
    size_t       total    = 0;
    for (const auto & e : QWEN38.experts) {
        total += e.second;
    }
    for (size_t i = 0; i < QWEN38.experts.size(); ++i) {
        const double      share = 100.0 * static_cast<double>(QWEN38.experts[i].second) / static_cast<double>(total);
        const std::string name  = std::string(test_type_name(QWEN38.experts[i].first)) + " is " +
                                 std::to_string(shares[i]).substr(0, 4) + " % of the mix's bytes";
        expect_true(name.c_str(), std::fabs(share - shares[i]) < 0.05);
    }

    // The mix's harmonic mean is its measured rate, and the default room is held: A/B is 20.2 with the splits, 18.8
    // on the fill alone.
    const weighed qwen = weigh(QWEN38, arm_default(QWEN38));
    expect_true("the qwen3.8 mix is read at 22 GB/s", near(qwen.rate.gbps, 22.0, 1e-9));
    expect_true("the qwen3.8 mix holds its default room by A/B 20.2 (18.8 on the fill alone)",
                qwen.cost.held && near(qwen.cost.host_attn_us / qwen.cost.expert_us, 20.21, 0.01) &&
                    near(qwen.cost.fill_us / qwen.cost.expert_us, 18.81, 0.01));

    // One of the seven replaced by an unmeasured type: without -c the room is refused unweighed, and the WARN names
    // that type alone; a named context prices it at the fallback.
    const weighed q4k = weigh(QWEN38_Q4K, arm_default(QWEN38_Q4K));
    expect("without -c the mix with a q4_K tensor refuses its room", QWEN38_Q4K, arm_default(QWEN38_Q4K), false, -1.0,
           -1.0);
    expect_true("without -c the mix with a q4_K tensor is unpriced", !q4k.rate.priced);
    expect_text("the unpriced WARN names q4_K", kv_room_type_names(q4k.rate.types.unmeasured_mask, test_type_name),
                "q4_K");
    expect("at -c 4096 the mix with a q4_K tensor is priced and holds", QWEN38_Q4K, arm_explicit(4096), true, 907.3,
           -1.0);
}

void test_rate_source() {
    // The room line's source field: each measurement after the types it priced, in type order.
    expect_text("gpt-oss-20b's rate source", rate_source(GPTOSS_20B, arm_default(GPTOSS_20B)),
                "mxfp4 via gpt-oss-20b 2026-10-18 (c-ncj4)");
    expect_text("qwen3.8's rate source", rate_source(QWEN38, arm_default(QWEN38)),
                "iq2_xxs/iq2_xs/iq3_xxs/iq4_nl/iq3_s/iq2_s/q2_0 via qwen38-gsq-rco-mix 2026-10-18 (c-plz8)");
    expect_text("two measurements, each after its types", rate_source(TWO_SOURCES, arm_explicit(4096)),
                "iq3_xxs via qwen38-gsq-rco-mix 2026-10-18 (c-plz8); mxfp4 via gpt-oss-20b 2026-10-18 (c-ncj4)");
    expect_text("unmeasured types follow the measured ones", rate_source(QWEN38_Q4K, arm_explicit(4096)),
                "iq2_xs/iq3_xxs/iq4_nl/iq3_s/iq2_s/q2_0 via qwen38-gsq-rco-mix 2026-10-18 (c-plz8); unmeasured q4_K");
    expect_text("only unmeasured types", rate_source(Q8_EXPERTS, arm_explicit(4096)), "unmeasured q8_0");
    arm o            = arm_default(GPTOSS_20B);
    o.cpu_expert_set = true;
    expect_text("the override", rate_source(GPTOSS_20B, o), "override");
    expect_text("no routed experts", rate_source(DENSE, arm_default(DENSE)), "no routed experts");
    kv_room_expert_types untyped;
    untyped.add(GGML_TYPE_COUNT, 1000);
    expect_text("an untyped tensor's name", kv_room_type_names(untyped.unmeasured_mask, test_type_name), "untyped");
}

void test_unmeasured_type_rule() {
    // At the 40 GB/s fallback this room is worth holding (A 4723.9 > B 835.6), so a refusal is the rule's alone.
    const weighed dflt = weigh(Q8_EXPERTS, arm_default(Q8_EXPERTS));
    expect_true("without -c an unmeasured expert type leaves the room unpriced", !dflt.rate.priced);
    expect("without -c an unmeasured expert type refuses the room", Q8_EXPERTS, arm_default(Q8_EXPERTS), false, 4723.9,
           835.6);
    expect("at an explicit -c an unmeasured expert type is priced at the fallback", Q8_EXPERTS, arm_explicit(131072),
           true, 4723.9, 835.6);
    expect("without -c one unmeasured type among measured ones refuses the room", MIXED_EXPERTS,
           arm_default(MIXED_EXPERTS), false, -1.0, -1.0);
}

void test_override() {
    arm q8            = arm_default(Q8_EXPERTS);
    q8.cpu_expert_set = true;
    expect("the override prices an unmeasured type without -c", Q8_EXPERTS, q8, true, 4723.9, 835.6);
    arm g            = arm_default(GPTOSS_20B);
    g.cpu_expert_set = true;
    expect("the override prices a measured type too", GPTOSS_20B, g, true, 4723.9, 835.6);
    expect_true("the override is reported", weigh(GPTOSS_20B, g).rate.overridden);
    expect_true("no override is reported without it", !weigh(GPTOSS_20B, arm_default(GPTOSS_20B)).rate.overridden);
}

void test_dense() {
    arm a      = arm_default(DENSE);
    a.fill_pct = 0.0;
    expect("a dense model holds its room at any fill", DENSE, a, true, 250.0, 0.0);
    expect_true("a dense model's room is priced", weigh(DENSE, a).rate.priced);
}

}  // namespace

int main() {
    test_measured_arms();
    test_gptoss_120b();
    test_split_window();
    test_split_mapping();
    test_expert_rates();
    test_qwen_mix();
    test_rate_source();
    test_unmeasured_type_rule();
    test_override();
    test_dense();
    std::printf("test-kv-room-cost: %d case(s), %d failure(s)\n", g_cases, g_failures);
    return g_failures == 0 ? 0 : 1;
}
