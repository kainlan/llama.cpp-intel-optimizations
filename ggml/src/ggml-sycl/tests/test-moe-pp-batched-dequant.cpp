// Live check of the type-keyed batched oneDNN MoE prompt executor (llama.cpp-348c).
//
// For each expert type below, one MUL_MAT_ID prompt op runs on a planned model loaded the way libllama loads one:
// the three expert tensors sit in the card's SYCL host buffer type, each is registered (file identity, weight
// identity, host weight, usage) while the load is open, and all are filled before it commits, so the commit's preload
// materializes every expert on the pinned card in the layout the planner chose for that type. On the B50 at this
// shape that is SOA for Q8_0 and Q6_K and AOS for the rest. The shape is test-backend-ops' MUL_MAT_ID (32 experts, 4
// used, k 2048, m 768), run at 64 and 300 tokens. The result is compared with the CPU backend at NMSE <= 5e-4, and the
// log must show the executor ran:
//   * the [PP-MOE] line, once per (device, type, layout), the first time the type-keyed arm runs it (each type
//     here runs in one layout, so once per type): it must appear during the type's first case, not during its
//     second, and exactly once over the run;
//   * a [MOE-PP-ONEDNN-F16-BATCHED] success line during each case (GGML_SYCL_MOE_ROUTE_LOG=1, set below). The
//     backend prints that line at most 96 times per process, so the cases must stay under that cap (static_assert
//     below). Every such line names the case's f16 groups (groups=, group_experts=, buffers=; see below);
//   * no refusal: no "[PP-MOE] refused" (traced under GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED_TRACE=1, set below), no
//     "[MOE-PROMPT-REFUSAL]", no "refusing unplanned PP MoE oneDNN batched scratch", and no "direct MMID route
//     re-entered" (an executor decline that sent a node back to the direct MMID route).
//
// The ring is planned the way libllama plans it by default (src/llama-model.cpp): weight slots sized for the MXFP4
// WOQ arm, about a quarter of every expert in f16. The f16 arm runs in that slot, never a wider one: it groups the
// active experts to fit it, alternating between the slot's two halves, each group's dequant chained on the GEMMs
// that last read its half. At this shape the slot holds 8 f16 experts, and every case activates all 32, so a case
// runs 8 groups of 4 and must name at least three groups.
// GGML_SYCL_MOE_PP_F16_GROUP_EXPERTS_MAX caps the experts per group. The backend reads it once per process, so the
// capped cases run in child processes: the test runs itself again as --group-cap N for each N in k_group_caps (1 and
// 3), each child runs one Q4_0 case at 300 tokens with the cap at N, and that case must name exactly ceil(32 / N)
// groups of N experts in both halves. A cap of 1 gives the longest chain, 32 groups of one expert.
//
// The skewed cases route as real prompts do, one expert taking every token and the rest fewer. Each runs in a child
// process, as --skew ROWS for each micro-batch in k_skew_rows (1024 and 2048), on a fixture planned for that
// micro-batch. Beyond what every case checks, the executor's activation scratch (the success line's act=) must be under
// half the rectangle's, the token count rounded up to the GEMM row alignment times every expert: the scratch is laid
// out per GEMM group, so a small group costs its own rows and not the busiest expert's.
//
// NVFP4 runs three cases more, each on a fixture of its own, at 64 tokens with the activation broadcast:
//   * every UE4M3 scale byte of the weight rewritten from a cycle of small ones (zero, subnormals, the NaN code, and
//     their sign-set forms), and again from a cycle of large ones (the largest finite value, its sign-set form, and
//     the NaN code), so each case's whole signal is scaled by those bytes and the NMSE sees how they decode. 0xFF is
//     never written: the CPU decode reads it as 240, the SYCL decode as 0, and no quantizer writes the sign bit;
//   * a fused gate_up tensor (blk.0.ffn_gate_up_exps) beside a down tensor, as Inkling 2 Mini's NVFP4 layers hold
//     them: the executor claims the fused weight as a plain matrix per expert.
// The executor's resident claim takes the fused role for every type it runs, so Q8_0 runs the fused fixture too, at
// 64 tokens with the activation broadcast. At this shape Q8_0 is materialized SOA where NVFP4 is AOS, so the fused
// role meets both of the executor's readers.
// NVFP4 and Q1_0 also have the direct MMID route, which runs ahead of the executor and steps aside for a prompt the
// executor claims while the ring is planned (moe_mmid_direct_defers_to_executor). The success line is what shows it
// stepped aside: had it run the node, no success line would print.
//
// The fixture models the default load only, GGML_SYCL_WEIGHTS_EVICTABLE on. With it off, libllama puts these
// device-planned experts in the device buffer instead, which this fixture does not model: that upload reorders an SOA
// type over the whole tensor and publishes no per-expert records for it, and the claim declines such a layout. So with
// it off the process exits 77 before any case.
//
// Every case runs, and each prints one CASE line naming what it saw; a capped case's exit status is its verdict. The
// process exits 1 if any case failed, 77 when there is no SYCL GPU or the weights are not evictable. Pin one card (the
// registration selects level_zero:1, the B50).
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "moe-planned-model-fixture.hpp"  // the planned model's shape, experts, inventory and lifecycle
#include "moe-pp-pack.hpp"  // moe_pp_gemm_row_align: the GEMM row rounding the rectangle is measured against

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

constexpr int      k_experts  = 32;
constexpr int      k_used     = 4;
constexpr int      k_in       = 2048;  // k: the expert weight's row length, the activation's width
constexpr int      k_out      = 768;   // m: output rows per expert
constexpr uint32_t k_n_ubatch = 512;
constexpr uint32_t k_n_ctx    = 512;
constexpr double   k_nmse_tol = 5e-4;

constexpr const char * k_tensor_names[3] = { "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                                             "blk.0.ffn_down_exps.weight" };
// The fused model's two tensors: the fused gate_up weight the case runs, and the down weight.
constexpr const char * k_fused_tensor_names[2] = { "blk.0.ffn_gate_up_exps.weight", "blk.0.ffn_down_exps.weight" };

// What a fixture loads. SMALL_SCALES and LARGE_SCALES are NVFP4 models; PLAIN and FUSED_GATE_UP take any type.
enum class fixture_variant {
    PLAIN,          // gate, up and down, one quantization shared by all three; the gate runs
    SMALL_SCALES,   // PLAIN with every UE4M3 scale byte rewritten from k_small_scales
    LARGE_SCALES,   // PLAIN with every UE4M3 scale byte rewritten from k_large_scales
    FUSED_GATE_UP,  // a fused gate_up weight and a down weight; the gate_up runs
};

// UE4M3 scale bytes both decodes read alike (ggml_ue4m3_to_fp32 on the CPU, ggml_sycl_ue4m3_to_fp32 on the device):
// zero, subnormals, the NaN code 0x7f (zero in both) and sign-set forms (both ignore the sign); then the largest
// finite value 0x7e, its sign-set form, and an exponent-15 value. Never 0xff, which the two decode differently.
constexpr uint8_t k_small_scales[] = { 0x00, 0x01, 0x02, 0x07, 0x7f, 0x80, 0x81, 0x87 };
constexpr uint8_t k_large_scales[] = { 0x7e, 0xfe, 0x78, 0x7f };

int fixture_tensor_count(fixture_variant variant) {
    return variant == fixture_variant::FUSED_GATE_UP ? 2 : 3;
}

const char * const * fixture_tensor_names(fixture_variant variant) {
    return variant == fixture_variant::FUSED_GATE_UP ? k_fused_tensor_names : k_tensor_names;
}

// Every expert type the batched executor's f16 table takes (moe_pp_batched_f16_supports_layout) that this fixture
// can quantize, each of which must run the executor. The fixture quantizes with no importance matrix, so the table's
// IQ1_S, IQ2_XXS and IQ2_XS, which need one (ggml_quantize_requires_imatrix), are not here; the source gate
// test-sycl-moe-pp-batched-types-source pins that the two lists cover the table between them.
constexpr ggml_type k_types[] = {
    GGML_TYPE_Q4_0,  GGML_TYPE_Q8_0,   GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,  GGML_TYPE_Q6_K,  GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0,  GGML_TYPE_Q5_1,  GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,  GGML_TYPE_IQ4_NL, GGML_TYPE_Q2_0, GGML_TYPE_IQ2_S, GGML_TYPE_IQ1_M, GGML_TYPE_NVFP4,
};
// The NVFP4 fixtures beyond the plain one, one case each.
constexpr fixture_variant k_nvfp4_variants[]    = { fixture_variant::SMALL_SCALES, fixture_variant::LARGE_SCALES,
                                                    fixture_variant::FUSED_GATE_UP };
constexpr int             k_nvfp4_variant_cases = int(sizeof(k_nvfp4_variants) / sizeof(k_nvfp4_variants[0]));

// The other types that run the fused gate_up fixture, one case each.
constexpr ggml_type k_fused_other_types[] = { GGML_TYPE_Q8_0 };
constexpr int       k_fused_other_cases   = int(sizeof(k_fused_other_types) / sizeof(k_fused_other_types[0]));

// 64 tokens with the activation broadcast to every used expert (gate/up, ne11 1), and 300 tokens with one
// activation row per used expert (down, ne11 k_used).
constexpr int k_cases_per_type = 2;
constexpr int k_cases =
    k_cases_per_type * int(sizeof(k_types) / sizeof(k_types[0])) + k_nvfp4_variant_cases + k_fused_other_cases;

// The backend prints its [MOE-PP-ONEDNN-F16-BATCHED] success line at most this many times per process (the
// batched_log counter in ggml-sycl.cpp). Each case runs one MUL_MAT_ID op and so prints one line, and a case past the
// cap would print none and read as an executor that did not run.
constexpr int k_route_log_cap = 96;
static_assert(k_cases <= k_route_log_cap, "a case past the route log's cap has no success line to count");

// The fewest f16 groups a case's success line may name at the slot's own grouping. At this shape the MXFP4-sized slot
// that every type but NVFP4 plans holds 8 f16 experts, so such a case runs 8 groups of 4. NVFP4 plans a larger slot for
// its f16 scale plane (983,040 bytes per expert, 10 f16 experts), so its cases run fewer groups, still at least 4 of
// the 32 experts. A slot that held every expert would run one group, and the case fails rather than pass without the
// chain.
constexpr size_t k_min_groups = 3;

// The group caps of the capped cases, one Q4_0 case each, each in its own child process (run_child_case).
constexpr int k_group_caps[] = { 1, 3 };
constexpr int k_capped_cases = int(sizeof(k_group_caps) / sizeof(k_group_caps[0]));

// The micro-batches of the skewed cases, one Q4_0 case each, each in its own child process (run_child_case).
constexpr int k_skew_rows[] = { 1024, 2048 };
constexpr int k_skew_cases  = int(sizeof(k_skew_rows) / sizeof(k_skew_rows[0]));

using moe_planned_fixture::require;

// Every log chunk the backend emits, in order, so a case can count what appeared while it ran. Also echoed to stderr,
// so the run's log carries the backend's own lines.
struct log_capture {
    std::mutex               mu;
    std::vector<std::string> chunks;

    size_t mark() {
        std::lock_guard<std::mutex> lock(mu);
        return chunks.size();
    }

    // Chunks from `from` on that contain every needle.
    size_t count_since(size_t from, std::initializer_list<const char *> needles) {
        std::lock_guard<std::mutex> lock(mu);
        size_t                      n = 0;
        for (size_t i = from; i < chunks.size(); ++i) {
            bool all = true;
            for (const char * needle : needles) {
                all = all && chunks[i].find(needle) != std::string::npos;
            }
            n += all ? 1 : 0;
        }
        return n;
    }

    // The smallest count that follows `field` (such as "groups=") in the chunks from `from` on that contain `needle`,
    // or SIZE_MAX when none of them carries the field.
    size_t min_field_since(size_t from, const char * needle, const char * field) {
        std::lock_guard<std::mutex> lock(mu);
        size_t                      least = SIZE_MAX;
        for (size_t i = from; i < chunks.size(); ++i) {
            const size_t at = chunks[i].find(needle) != std::string::npos ? chunks[i].find(field) : std::string::npos;
            if (at == std::string::npos) {
                continue;
            }
            const char * digits = chunks[i].c_str() + at + std::strlen(field);
            char *       end    = nullptr;
            const auto   value  = std::strtoull(digits, &end, 10);
            if (end != digits) {
                least = std::min(least, static_cast<size_t>(value));
            }
        }
        return least;
    }

    // The largest value that follows `field` (such as "act=") in the chunks from `from` on that contain `needle`, or
    // a negative number when none of them carries the field.
    double max_double_field_since(size_t from, const char * needle, const char * field) {
        std::lock_guard<std::mutex> lock(mu);
        double                      most = -1.0;
        for (size_t i = from; i < chunks.size(); ++i) {
            const size_t at = chunks[i].find(needle) != std::string::npos ? chunks[i].find(field) : std::string::npos;
            if (at == std::string::npos) {
                continue;
            }
            const char * digits = chunks[i].c_str() + at + std::strlen(field);
            char *       end    = nullptr;
            const double value  = std::strtod(digits, &end);
            if (end != digits) {
                most = std::max(most, value);
            }
        }
        return most;
    }
};

log_capture g_log;

void log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    if (!text) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_log.mu);
        g_log.chunks.emplace_back(text);
    }
    std::fputs(text, stderr);
}

// Rewrites every UE4M3 scale byte of NVFP4 `bytes` from `scales`, in turn. A block is 4 scale bytes, one per
// 16-element sub-block, then 32 bytes of E2M1 nibbles (block_nvfp4 in ggml-common.h).
void rewrite_nvfp4_scales(std::vector<uint8_t> & bytes, const uint8_t * scales, size_t n_scales) {
    constexpr size_t block_bytes = 36;
    constexpr size_t scale_bytes = 4;
    require(ggml_type_size(GGML_TYPE_NVFP4) == block_bytes && bytes.size() % block_bytes == 0,
            "the NVFP4 block is not 4 scale bytes and 32 nibble bytes");
    size_t next = 0;
    for (size_t block = 0; block < bytes.size(); block += block_bytes) {
        for (size_t j = 0; j < scale_bytes; ++j) {
            bytes[block + j] = scales[next++ % n_scales];
        }
    }
}

// The planned model a case runs (moe_planned_fixture::planned_model): the variant's expert tensors at this test's
// shape, planned for `n_ubatch` and a context of at least k_n_ctx. The one split is named per type and variant, so no
// two cases' weights share a file identity. The NVFP4 scale variants rewrite every scale byte of the quantized weights
// before they fill the tensors.
moe_planned_fixture::planned_shape fixture_shape(ggml_type type, uint32_t n_ubatch, fixture_variant variant) {
    return { type,
             k_experts,
             k_used,
             k_in,
             k_out,
             fixture_tensor_names(variant),
             fixture_tensor_count(variant),
             std::max(k_n_ctx, n_ubatch),
             n_ubatch };
}

std::string fixture_split_path(ggml_type type, fixture_variant variant) {
    return std::string("test-moe-pp-batched-dequant-") + ggml_type_name(type) + "-" +
           std::to_string(static_cast<int>(variant)) + ".gguf";
}

std::function<void(std::vector<uint8_t> &)> fixture_rewrite(fixture_variant variant) {
    if (variant == fixture_variant::SMALL_SCALES) {
        return [](std::vector<uint8_t> & bytes) {
            rewrite_nvfp4_scales(bytes, k_small_scales, sizeof(k_small_scales));
        };
    }
    if (variant == fixture_variant::LARGE_SCALES) {
        return [](std::vector<uint8_t> & bytes) {
            rewrite_nvfp4_scales(bytes, k_large_scales, sizeof(k_large_scales));
        };
    }
    return {};
}

struct lifecycle_fixture : moe_planned_fixture::planned_model {
    lifecycle_fixture(ggml_type       type,
                      std::mt19937 &  rng,
                      uint32_t        n_ubatch = k_n_ubatch,
                      fixture_variant variant  = fixture_variant::PLAIN) :
        planned_model(fixture_shape(type, n_ubatch, variant),
                      fixture_split_path(type, variant),
                      rng,
                      fixture_rewrite(variant)) {}
};

ggml_backend_buffer_t alloc_tensor(ggml_backend_buffer_type_t buft,
                                   ggml_tensor *              tensor,
                                   ggml_backend_buffer_usage  usage) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, tensor));
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

struct case_data {
    ggml_type            type;
    int                  n_tokens;
    int                  ne11;  // 1: the activation broadcasts to every used expert (gate/up); k_used: one row each
    std::vector<float>   input;
    std::vector<int32_t> ids;
};

case_data make_case(ggml_type type, int n_tokens, int ne11, std::mt19937 & rng) {
    case_data                             c{ type, n_tokens, ne11, {}, {} };
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    c.input.resize(size_t(k_in) * ne11 * n_tokens);
    for (float & v : c.input) {
        v = dist(rng);
    }
    // Distinct experts per token (offsets 0, 8, 16, 24 from a moving base), every expert active, uneven row counts.
    c.ids.resize(size_t(k_used) * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        const int base = (t * 3 + t / 16) % k_experts;
        for (int i = 0; i < k_used; ++i) {
            c.ids[size_t(t) * k_used + i] = (base + i * (k_experts / k_used)) % k_experts;
        }
    }
    return c;
}

// A skewed routing, as real prompts route: expert 0 takes every token, experts 1 to 3 a third each, experts 4 to 10 a
// seventh each, experts 11 to 31 a twenty-first each. Every expert is active and a token's four picks are distinct.
// The rectangle pads all 32 experts to the token count; the packed layout pads four groups to their own heads.
case_data make_skew_case(int n_tokens, std::mt19937 & rng) {
    case_data                             c{ GGML_TYPE_Q4_0, n_tokens, k_used, {}, {} };
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    c.input.resize(size_t(k_in) * k_used * n_tokens);
    for (float & v : c.input) {
        v = dist(rng);
    }
    c.ids.resize(size_t(k_used) * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        c.ids[size_t(t) * k_used + 0] = 0;
        c.ids[size_t(t) * k_used + 1] = 1 + t % 3;
        c.ids[size_t(t) * k_used + 2] = 4 + t % 7;
        c.ids[size_t(t) * k_used + 3] = 11 + t % 21;
    }
    return c;
}

// Builds MUL_MAT_ID over the case's input and ids on `backend` and computes it. Fills `out` on success. The SYCL run
// passes the fixture's loaded gate tensor as `loaded_weights`; the CPU reference passes null, and `weight_bytes` go
// into a fresh tensor on the CPU backend.
ggml_status run_backend(ggml_backend_t               backend,
                        ggml_tensor *                loaded_weights,
                        const std::vector<uint8_t> & weight_bytes,
                        const case_data &            c,
                        std::vector<float> &         out) {
    ggml_init_params                                    params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
    require(ctx != nullptr, "ggml_init failed");
    ggml_tensor * weights = loaded_weights;
    if (!weights) {
        weights = ggml_new_tensor_3d(ctx.get(), c.type, k_in, k_out, k_experts);
        ggml_set_name(weights, k_tensor_names[0]);
    }
    ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k_in, c.ne11, c.n_tokens);
    ggml_tensor * ids   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, k_used, c.n_tokens);
    ggml_set_name(input, "moe_input");
    ggml_set_name(ids, "moe_ids");
    ggml_tensor * dst = ggml_mul_mat_id(ctx.get(), weights, input, ids);
    ggml_set_name(dst, "moe_out");

    ggml_backend_buffer_type_t         buft    = ggml_backend_get_default_buffer_type(backend);
    std::vector<ggml_backend_buffer_t> buffers = {
        alloc_tensor(buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
        alloc_tensor(buft, ids, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
        alloc_tensor(buft, dst, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
    };
    if (!loaded_weights) {
        buffers.push_back(alloc_tensor(buft, weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS));
    }

    struct buffers_guard {
        std::vector<ggml_backend_buffer_t> & b;

        ~buffers_guard() {
            for (ggml_backend_buffer_t buffer : b) {
                if (buffer) {
                    ggml_backend_buffer_free(buffer);
                }
            }
        }
    } guard{ buffers };

    for (ggml_backend_buffer_t buffer : buffers) {
        require(buffer != nullptr, "tensor buffer allocation failed");
    }
    if (!loaded_weights) {
        ggml_backend_tensor_set(weights, weight_bytes.data(), 0, weight_bytes.size());
    }
    ggml_backend_tensor_set(input, c.input.data(), 0, c.input.size() * sizeof(float));
    ggml_backend_tensor_set(ids, c.ids.data(), 0, c.ids.size() * sizeof(int32_t));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    if (status == GGML_STATUS_SUCCESS) {
        out.resize(size_t(k_out) * k_used * c.n_tokens);
        ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
    }
    return status;
}

double nmse(const std::vector<float> & ref, const std::vector<float> & got) {
    double err = 0.0, norm = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = double(ref[i]) - double(got[i]);
        err += d * d;
        norm += double(ref[i]) * double(ref[i]);
    }
    return err / (norm > 0.0 ? norm : 1.0);
}

// Runs one case and prints its CASE line. Returns true when it passed. Every count is of the lines that appeared
// while this case ran, except the [PP-MOE] total, which is over the run so far. `first_of_type` is true for the
// type's first case, the only one in which its [PP-MOE] line may appear. `group_cap` is the process's
// GGML_SYCL_MOE_PP_F16_GROUP_EXPERTS_MAX, 0 when it is unset.
bool run_case(lifecycle_fixture & life, ggml_backend_t cpu, const case_data & c, bool first_of_type, int group_cap) {
    const std::string type_name  = ggml_type_name(c.type);
    const std::string type_field = "type=" + type_name + " ";

    std::vector<float> ref;
    require(run_backend(cpu, nullptr, life.gate_bytes, c, ref) == GGML_STATUS_SUCCESS,
            "CPU reference compute failed for " + type_name);

    const size_t       from = g_log.mark();
    std::vector<float> got;
    const ggml_status  status    = run_backend(life.backend, life.gate, life.gate_bytes, c, got);
    const double       err       = status == GGML_STATUS_SUCCESS ? nmse(ref, got) : NAN;
    const size_t       pp_moe    = g_log.count_since(from, { "[PP-MOE] device=", type_field.c_str() });
    const size_t       pp_total  = g_log.count_since(0, { "[PP-MOE] device=", type_field.c_str() });
    const size_t       batched   = g_log.count_since(from, { "[MOE-PP-ONEDNN-F16-BATCHED] tensor=" });
    const size_t       batch_all = g_log.count_since(0, { "[MOE-PP-ONEDNN-F16-BATCHED] tensor=" });
    const size_t       refused   = g_log.count_since(from, { "[PP-MOE] refused" });
    const size_t       prompt_rf = g_log.count_since(from, { "[MOE-PROMPT-REFUSAL]" });
    const size_t       unplanned = g_log.count_since(from, { "refusing unplanned PP MoE oneDNN batched scratch" });
    const size_t       reentered = g_log.count_since(from, { "direct MMID route re-entered" });
    const size_t       groups    = g_log.min_field_since(from, "[MOE-PP-ONEDNN-F16-BATCHED] tensor=", "groups=");

    // The rest of the group plan the success line names, and what a cap must give: every case activates all k_experts
    // experts, so a cap gives ceil(k_experts / cap) groups of that many experts, alternating between both halves.
    const char * success       = "[MOE-PP-ONEDNN-F16-BATCHED] tensor=";
    const size_t group_experts = g_log.min_field_since(from, success, "group_experts=");
    const size_t buffers       = g_log.min_field_since(from, success, "buffers=");
    const size_t want_groups   = group_cap > 0 ? (size_t(k_experts) + size_t(group_cap) - 1) / size_t(group_cap) : 0;

    std::string why;
    if (status != GGML_STATUS_SUCCESS) {
        why += " compute-status";
    } else if (!(std::isfinite(err) && err <= k_nmse_tol)) {
        why += " nmse";
    }
    if (pp_moe != (first_of_type ? 1u : 0u) || pp_total != 1) {
        why += " pp-moe-line";
    }
    if (batched == 0) {
        // Unreachable while the static_assert above holds, but a cap reached would otherwise read as a skip.
        why += batch_all >= size_t(k_route_log_cap) ? " route-log-cap" : " executor-did-not-run";
    }
    if (refused != 0 || prompt_rf != 0 || unplanned != 0 || reentered != 0) {
        why += " refusal";
    }
    // A success line with no groups= field fails too: the parse returns SIZE_MAX for it, which no minimum catches.
    if (batched != 0 && groups == SIZE_MAX) {
        why += " groups-field";
    } else if (batched != 0 && group_cap == 0 && groups < k_min_groups) {
        why += " groups";
    } else if (batched != 0 && group_cap > 0 &&
               (groups != want_groups || group_experts != size_t(group_cap) || buffers != 2)) {
        why += " group-cap";
    }
    auto shown = [](size_t field) {
        return field == SIZE_MAX ? size_t(0) : field;
    };
    std::fprintf(stderr,
                 "CASE type=%s tokens=%d ne11=%d cap=%d status=%d nmse=%.3e pp_moe_lines=%zu pp_moe_total=%zu "
                 "batched_lines=%zu groups=%zu group_experts=%zu buffers=%zu pp_moe_refused=%zu prompt_refusal=%zu "
                 "unplanned_scratch=%zu direct_reentry=%zu -> %s%s\n",
                 type_name.c_str(), c.n_tokens, c.ne11, group_cap, static_cast<int>(status), err, pp_moe, pp_total,
                 batched, shown(groups), shown(group_experts), shown(buffers), refused, prompt_rf, unplanned, reentered,
                 why.empty() ? "PASS" : "FAIL:", why.c_str());
    return why.empty();
}

// Runs the skewed case at `n_tokens` rows and checks, beyond what run_case checks, that the activation scratch the
// executor used is under half the rectangle's (the token count rounded up to the GEMM row alignment, times every
// expert).
bool run_skew_case(lifecycle_fixture & life, ggml_backend_t cpu, int n_tokens, std::mt19937 & rng) {
    const size_t from    = g_log.mark();
    const bool   passed  = run_case(life, cpu, make_skew_case(n_tokens, rng), true, 0);
    const double act_mb  = g_log.max_double_field_since(from, "[MOE-PP-ONEDNN-F16-BATCHED] tensor=", " act=");
    const size_t align   = ggml_sycl::moe_pp_gemm_row_align;
    const double rect_mb = double((size_t(n_tokens) + align - 1) / align * align) * k_experts * k_in *
                           sizeof(ggml_fp16_t) / (1024.0 * 1024.0);
    const bool packed = act_mb >= 0.0 && act_mb < rect_mb / 2.0;
    std::fprintf(stderr, "SKEW tokens=%d act=%.1fMB rectangle=%.1fMB -> %s\n", n_tokens, act_mb, rect_mb,
                 !passed ? "FAIL: case" :
                 packed  ? "PASS" :
                           "FAIL: act-not-packed");
    return passed && packed;
}

// Runs one case in a child process: this binary again, as `<flag> <value>`, so a setting the backend reads once per
// process (the group cap) or a fixture of another micro-batch (the skewed cases) starts from a fresh backend. The
// child's CASE line shares this process's stderr; its exit status is the verdict.
bool run_child_case(const char * flag, int value) {
    const std::string arg = std::to_string(value);
    std::fflush(stdout);
    std::fflush(stderr);
    const pid_t pid = fork();
    if (pid == 0) {
        execl("/proc/self/exe", "test-moe-pp-batched-dequant", flag, arg.c_str(), static_cast<char *>(nullptr));
        _exit(127);
    }
    if (pid < 0) {
        std::fprintf(stderr, "FAIL: fork for the %s %d case failed\n", flag, value);
        return false;
    }
    int        status = 0;
    const bool waited = waitpid(pid, &status, 0) == pid;
    if (waited && WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        return true;
    }
    if (!waited) {
        std::fprintf(stderr, "FAIL: waitpid for the %s %d case failed\n", flag, value);
    } else if (WIFEXITED(status)) {
        // 77 is a skip, which this process ruled out before it started the child; 127 is a failed exec.
        std::fprintf(stderr, "FAIL: the %s %d case exited %d\n", flag, value, WEXITSTATUS(status));
    } else {
        std::fprintf(stderr, "FAIL: the %s %d case died on signal %d\n", flag, value,
                     WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    }
    return false;
}

}  // namespace

int main(int argc, char ** argv) {
    // A child process (run_child_case) runs one Q4_0 case: with the group cap at argv[2], or skewed at argv[2] rows.
    int group_cap = 0;
    int skew_rows = 0;
    if (argc == 3 && std::strcmp(argv[1], "--group-cap") == 0) {
        group_cap = std::atoi(argv[2]);
        if (group_cap < 1) {
            std::fprintf(stderr, "FAIL: --group-cap takes a count of at least 1, not %s\n", argv[2]);
            return 1;
        }
        setenv("GGML_SYCL_MOE_PP_F16_GROUP_EXPERTS_MAX", argv[2], 1);
    } else if (argc == 3 && std::strcmp(argv[1], "--skew") == 0) {
        skew_rows = std::atoi(argv[2]);
        if (skew_rows < 64) {
            std::fprintf(stderr, "FAIL: --skew takes a row count of at least 64, not %s\n", argv[2]);
            return 1;
        }
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--group-cap N | --skew ROWS]\n", argv[0]);
        return 1;
    }
    // Read once by the backend, so they are set before anything touches it.
    setenv("GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED_TRACE", "1", 1);
    setenv("GGML_SYCL_MOE_ROUTE_LOG", "1", 1);
    ggml_log_set(log_callback, nullptr);
    try {
        bool have_gpu = false;
        for (const auto & device : sycl::device::get_devices()) {
            have_gpu |= device.is_gpu();
        }
        if (!have_gpu || ggml_backend_sycl_get_device_count() < 1) {
            std::fprintf(stderr, "SKIP: no usable SYCL GPU\n");
            return LLAMA_TEST_EXIT_SKIP;
        }
        if (!ggml_backend_sycl_weights_evictable()) {
            std::fprintf(stderr,
                         "SKIP: GGML_SYCL_WEIGHTS_EVICTABLE is off, so libllama would load these experts into the "
                         "device buffer, which this fixture does not model\n");
            return LLAMA_TEST_EXIT_SKIP;
        }
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(
            ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), &ggml_backend_free);
        require(cpu != nullptr, "CPU backend init failed");

        std::mt19937 rng(348);
        if (group_cap > 0) {
            lifecycle_fixture life(GGML_TYPE_Q4_0, rng);
            return run_case(life, cpu.get(), make_case(GGML_TYPE_Q4_0, 300, k_used, rng), true, group_cap) ? 0 : 1;
        }
        if (skew_rows > 0) {
            lifecycle_fixture life(GGML_TYPE_Q4_0, rng, static_cast<uint32_t>(skew_rows));
            return run_skew_case(life, cpu.get(), skew_rows, rng) ? 0 : 1;
        }
        // The capped and skewed cases first, each in its own process, so this one holds no model while a child runs.
        int failed = 0;
        for (int cap : k_group_caps) {
            if (!run_child_case("--group-cap", cap)) {
                ++failed;
            }
        }
        for (int rows : k_skew_rows) {
            if (!run_child_case("--skew", rows)) {
                ++failed;
            }
        }
        for (ggml_type type : k_types) {
            lifecycle_fixture life(type, rng);
            if (!run_case(life, cpu.get(), make_case(type, 64, 1, rng), true, 0)) {
                ++failed;
            }
            if (!run_case(life, cpu.get(), make_case(type, 300, k_used, rng), false, 0)) {
                ++failed;
            }
        }
        // After NVFP4's plain fixture, so its [PP-MOE] line has printed and none of these may print another.
        for (fixture_variant variant : k_nvfp4_variants) {
            lifecycle_fixture life(GGML_TYPE_NVFP4, rng, k_n_ubatch, variant);
            if (!run_case(life, cpu.get(), make_case(GGML_TYPE_NVFP4, 64, 1, rng), false, 0)) {
                ++failed;
            }
        }
        // After each type's plain fixture, for the same reason.
        for (ggml_type type : k_fused_other_types) {
            lifecycle_fixture life(type, rng, k_n_ubatch, fixture_variant::FUSED_GATE_UP);
            if (!run_case(life, cpu.get(), make_case(type, 64, 1, rng), false, 0)) {
                ++failed;
            }
        }
        const int total = k_cases + k_capped_cases + k_skew_cases;
        if (failed != 0) {
            std::fprintf(stderr, "FAIL: %d of %d cases failed\n", failed, total);
            return 1;
        }
        std::fprintf(stderr,
                     "PASS: batched MoE prompt executor, %d cases (%d at the slot's own grouping, %d capped, %d "
                     "skewed)\n",
                     total, k_cases, k_capped_cases, k_skew_cases);
        return 0;
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SKIP: no usable SYCL GPU: %s\n", e.what());
        return LLAMA_TEST_EXIT_SKIP;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
