// Live check of the BF16 grouped prompt arm of MUL_MAT_ID (Inkling 2 Mini plan, Task 24).
//
// A BF16 prompt node of at least 64 tokens (moe_bf16_grouped_min_tokens), on a card whose plan holds every expert at
// AOS and whose load planned the PP MoE oneDNN ring, runs as one oneDNN bf16 GEMM per active expert between one
// batched gather and one batched scatter (ggml_sycl_mul_mat_id_bf16_grouped); every other BF16 node runs on the BF16
// direct kernel. This test drives both through the public entry points on planned models loaded the way libllama loads
// one (moe_planned_fixture::planned_model): BF16 experts at Inkling 2 Mini's shapes, 128 experts, 8 used, beside a
// zero NVFP4 companion tensor of the same shape that makes the load plan the ring. A model of BF16 experts alone plans
// no ring, and the arm declines there; an Inkling load plans one for its NVFP4 layers.
//
// The cases, each compared with the CPU backend on every output row (the largest per-row NMSE must be <= 5e-4, the
// bound of the other planned-model prompt tests):
//   * gate_up (k 2304, m 4608), one activation row broadcast to every slot: 512 tokens over all 128 experts; 512 tokens
//     over 40 experts, so 88 are selected by no token; and 64 tokens, the arm's minimum. Each must show the arm's line,
//     "[MOE-ROUTE] bf16 grouped tensor=<name> ... experts=<the experts its ids select>", and no direct-route line;
//   * 63 tokens on the same tensor, one below the minimum: the arm's decline line with reason=tokens and the direct
//     route's line;
//   * down (k 2304, m 2304), one activation row per slot: 512 tokens over all 128 experts, on the arm;
//   * GGML_SYCL_MOE_BF16_GROUPED=0, which the backend reads once, so in a child process (this binary as --opt-out): the
//     gate_up 512-token case on the direct route, with the arm's decline line reason=opt_out;
//   * the gate_up 512-token case with the private failpoint armed that throws at the arm's GEMM stage, after its gather
//     was submitted: the compute must end GGML_STATUS_FAILED with the arm's WARN line and graph_compute's recoverable
//     fallback line naming "BF16 grouped arm: expert GEMM failed after submission", the failpoint consumed once, and
//     neither the arm's run line nor the direct route's; the same case then runs on the arm again in this process.
// Besides its lines, each case reads the arm's own counters (ggml_sycl_moe_bf16_grouped_test_counters_read): a node
// the arm takes adds exactly one run and no decline, a declined node exactly one decline, under its reason, and no
// run, and the thrown node neither. No case may show a refusal: the direct route's "[MoE] BF16 direct route refused",
// or the arm's "does not fit the planned PP MoE oneDNN ring". The counters and the failpoint exist only in the private
// backend objects, so this links ggml-sycl-private-fixtures, not the ordinary ggml-sycl.
//
// The activations are BF16-representable: the CPU rounds src1 to BF16 before its dot products and the arm's gather
// rounds it to BF16 to nearest even, so the two sides differ in summation order only. dst starts as NaN on both
// backends, so a row the arm leaves unwritten cannot pass.
//
// Every case prints one CASE line. The process exits 1 if any case failed, 77 when there is no SYCL GPU or the weights
// are not evictable (the fixture models the default load only). Pin one card (the registration selects level_zero:0,
// the B70).
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "moe-planned-model-fixture.hpp"            // the planned model's shape, experts, inventory and lifecycle
#include "q1-nvfp4-production-route-test-seam.hpp"  // the arm's counters, the failpoint and its consumed count

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

using moe_planned_fixture::require;

constexpr int      k_experts    = 128;
constexpr int      k_used       = 8;
constexpr int      k_in         = 2304;
constexpr int      k_gate_up    = 4608;  // m of the fused gate_up tensor
constexpr int      k_down       = 2304;  // m of the down tensor
constexpr int      k_subset     = 40;    // the experts the subset case selects
constexpr int      k_min_tokens = 64;    // moe_bf16_grouped_min_tokens
constexpr uint32_t k_n_ctx      = 512;
constexpr uint32_t k_n_ubatch   = 512;
constexpr double   k_nmse_tol   = 5e-4;

const char * const k_gate_up_names[] = { "blk.0.ffn_gate_up_exps.weight" };
const char * const k_down_names[]    = { "blk.0.ffn_down_exps.weight" };

// A model of one BF16 expert tensor of `n_out` rows per expert, `names[0]`, beside its NVFP4 companion `companion`, the
// tensor of the other role in the same block.
moe_planned_fixture::planned_shape bf16_shape(int n_out, const char * const * names, const char * companion) {
    moe_planned_fixture::planned_shape shape = { GGML_TYPE_BF16, k_experts, k_used,  k_in,      n_out,
                                                 names,          1,         k_n_ctx, k_n_ubatch };
    shape.companion_type                     = GGML_TYPE_NVFP4;
    shape.companion_name                     = companion;
    return shape;
}

const moe_planned_fixture::planned_shape k_gate_up_shape =
    bf16_shape(k_gate_up, k_gate_up_names, "blk.0.ffn_down_exps.weight");
const moe_planned_fixture::planned_shape k_down_shape =
    bf16_shape(k_down, k_down_names, "blk.0.ffn_gate_up_exps.weight");

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

// The CPU backend's copy of a model's BF16 expert tensor, allocated once: every case of the model reads it.
struct cpu_weights {
    std::unique_ptr<ggml_context, decltype(&ggml_free)>                       ctx{ nullptr, &ggml_free };
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{ nullptr,
                                                                                      &ggml_backend_buffer_free };
    ggml_tensor *                                                             tensor = nullptr;

    cpu_weights(ggml_backend_t cpu, const moe_planned_fixture::planned_model & model) {
        ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
        ctx.reset(ggml_init(params));
        require(ctx != nullptr, "ggml_init for the CPU weights failed");
        const ggml_tensor * gate = model.gate;
        tensor                   = ggml_new_tensor_3d(ctx.get(), gate->type, gate->ne[0], gate->ne[1], gate->ne[2]);
        ggml_set_name(tensor, gate->name);
        buffer.reset(
            alloc_tensor(ggml_backend_get_default_buffer_type(cpu), tensor, GGML_BACKEND_BUFFER_USAGE_WEIGHTS));
        require(buffer != nullptr, "CPU weight buffer allocation failed");
        ggml_backend_tensor_set(tensor, model.gate_bytes.data(), 0, model.gate_bytes.size());
    }
};

struct case_data {
    const char *         label;
    int                  n_tokens;
    int                  ne11;    // 1: one activation row broadcast to every slot; k_used: one row per slot
    int                  active;  // the distinct experts the ids select
    std::vector<float>   input;
    std::vector<int32_t> ids;
};

// BF16-representable activations, and k_used distinct experts per token drawn from `n_select` experts spread over all
// k_experts: slots stride the selected experts from a moving base, as the BF16 direct device test routes.
case_data make_case(const char * label, int n_tokens, int ne11, int n_select, std::mt19937 & rng) {
    require(n_select % k_used == 0 && n_select <= k_experts, "the selected experts must split evenly over the slots");
    case_data                             c{ label, n_tokens, ne11, 0, {}, {} };
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    c.input.resize(size_t(k_in) * ne11 * n_tokens);
    for (float & v : c.input) {
        v = ggml_bf16_to_fp32(ggml_fp32_to_bf16(dist(rng)));
    }
    const int         stride = n_select / k_used;
    const int         spread = k_experts / n_select;
    std::set<int32_t> selected;
    c.ids.resize(size_t(k_used) * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        const int base = (t * 3 + t / 4) % n_select;
        for (int i = 0; i < k_used; ++i) {
            const int32_t expert          = ((base + i * stride) % n_select) * spread;
            c.ids[size_t(t) * k_used + i] = expert;
            selected.insert(expert);
        }
    }
    c.active = int(selected.size());
    return c;
}

// The MUL_MAT_ID of one case over `weights` in `ctx`; its src1 and ids are unallocated.
ggml_tensor * build_op(ggml_context * ctx, ggml_tensor * weights, const case_data & c) {
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, weights->ne[0], c.ne11, c.n_tokens);
    ggml_tensor * ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k_used, c.n_tokens);
    ggml_set_name(input, "moe_input");
    ggml_set_name(ids, "moe_ids");
    ggml_tensor * dst = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_set_name(dst, "moe_out");
    return dst;
}

bool supports(ggml_backend_t backend, ggml_tensor * weights, const case_data & c) {
    ggml_init_params                                    params = { 64 * ggml_tensor_overhead(), nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
    require(ctx != nullptr, "ggml_init failed");
    return ggml_backend_supports_op(backend, build_op(ctx.get(), weights, c));
}

// Builds the case's MUL_MAT_ID over `weights` (a tensor `backend` can read) and computes it on `backend`. Fills `out`
// on success.
ggml_status run_backend(ggml_backend_t backend, ggml_tensor * weights, const case_data & c, std::vector<float> & out) {
    ggml_init_params                                    params = { 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
    require(ctx != nullptr, "ggml_init failed");
    ggml_tensor * dst   = build_op(ctx.get(), weights, c);
    ggml_tensor * input = dst->src[1];
    ggml_tensor * ids   = dst->src[2];

    ggml_backend_buffer_type_t         buft    = ggml_backend_get_default_buffer_type(backend);
    std::vector<ggml_backend_buffer_t> buffers = {
        alloc_tensor(buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
        alloc_tensor(buft, ids, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
        alloc_tensor(buft, dst, GGML_BACKEND_BUFFER_USAGE_COMPUTE),
    };

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
    ggml_backend_tensor_set(input, c.input.data(), 0, c.input.size() * sizeof(float));
    ggml_backend_tensor_set(ids, c.ids.data(), 0, c.ids.size() * sizeof(int32_t));
    const std::vector<float> poison(size_t(weights->ne[1]) * k_used * c.n_tokens,
                                    std::numeric_limits<float>::quiet_NaN());
    ggml_backend_tensor_set(dst, poison.data(), 0, poison.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    if (status == GGML_STATUS_SUCCESS) {
        out.resize(poison.size());
        ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
    }
    return status;
}

// The largest NMSE over the output rows (one row per token and slot, `row` values each). A non-finite value in a row
// makes that row's NMSE infinite.
double max_row_nmse(const std::vector<float> & ref, const std::vector<float> & got, size_t row) {
    double worst = 0.0;
    for (size_t r = 0; r + row <= ref.size(); r += row) {
        double err = 0.0, norm = 0.0;
        for (size_t i = r; i < r + row; ++i) {
            if (!std::isfinite(got[i])) {
                return INFINITY;
            }
            const double d = double(ref[i]) - double(got[i]);
            err += d * d;
            norm += double(ref[i]) * double(ref[i]);
        }
        worst = std::max(worst, err / (norm > 0.0 ? norm : 1.0));
    }
    return worst;
}

// The arm's counters, with `reason` the decline reason whose count is read (any reason when only the runs and the total
// matter).
ggml_sycl_moe_bf16_grouped_test_counters arm_counters(const char * reason) {
    ggml_sycl_moe_bf16_grouped_test_counters counters{};
    require(ggml_sycl_moe_bf16_grouped_test_counters_read(reason, &counters),
            std::string("the BF16 grouped arm has no decline reason named ") + reason);
    return counters;
}

// How the arm's counters moved over one case.
struct counter_moves {
    uint64_t runs;
    uint64_t declines;
    uint64_t reason_declines;
};

counter_moves counters_since(const ggml_sycl_moe_bf16_grouped_test_counters & before, const char * reason) {
    const ggml_sycl_moe_bf16_grouped_test_counters after = arm_counters(reason);
    return { after.runs - before.runs, after.declines - before.declines,
             after.reason_declines - before.reason_declines };
}

uint64_t failures_consumed() {
    ggml_sycl_q1_nvfp4_test_counters counters{};
    ggml_sycl_q1_nvfp4_test_counters_read(&counters);
    return counters.failure_consumed;
}

// Runs one case and prints its CASE line. Returns true when it passed. Every count is of the lines that appeared
// while this case ran, and of the arm's counters across it. With `decline` null the arm must run the node: one run,
// no decline. Otherwise the arm must decline it for that reason, one decline and no run, and the direct route must run
// it.
bool run_case(const moe_planned_fixture::planned_model & model,
              ggml_backend_t                             cpu,
              const cpu_weights &                        reference,
              const case_data &                          c,
              const char *                               decline) {
    const char * name     = model.gate->name;
    const size_t m        = size_t(model.gate->ne[1]);
    const bool   admitted = supports(model.backend, model.gate, c);
    const char * reason   = decline ? decline : "tokens";

    std::vector<float> ref;
    require(run_backend(cpu, reference.tensor, c, ref) == GGML_STATUS_SUCCESS,
            std::string("CPU reference compute failed for ") + c.label);

    const ggml_sycl_moe_bf16_grouped_test_counters before = arm_counters(reason);
    const size_t                                   from   = g_log.mark();
    std::vector<float>                             got;
    ggml_status                                    status = GGML_STATUS_FAILED;
    double                                         err    = NAN;
    if (admitted) {
        status = run_backend(model.backend, model.gate, c, got);
        err    = status == GGML_STATUS_SUCCESS ? max_row_nmse(ref, got, m) : NAN;
    }
    const counter_moves moved         = counters_since(before, reason);
    const std::string   experts_field = "experts=" + std::to_string(c.active) + " ";
    const std::string   reason_field  = std::string("reason=") + (decline ? decline : "") + " ";
    const size_t        arm_lines     = g_log.count_since(from, { "[MOE-ROUTE] bf16 grouped tensor=", name });
    const size_t        arm_experts =
        g_log.count_since(from, { "[MOE-ROUTE] bf16 grouped tensor=", name, experts_field.c_str() });
    const size_t direct_lines = g_log.count_since(from, { "[MOE-ROUTE] bf16 direct tensor=", name });
    const size_t decline_lines =
        g_log.count_since(from, { "[MOE-ROUTE] bf16 grouped declined tensor=", name, reason_field.c_str() });
    const size_t refusals = g_log.count_since(from, { "[MoE] BF16 direct route refused" }) +
                            g_log.count_since(from, { "does not fit the planned PP MoE oneDNN ring" });

    std::string why;
    if (!admitted) {
        why += " supports-op";
    } else if (status != GGML_STATUS_SUCCESS) {
        why += " compute-status";
    } else if (!(std::isfinite(err) && err <= k_nmse_tol)) {
        why += " nmse";
    }
    if (admitted && decline == nullptr) {
        if (arm_lines == 0) {
            why += " arm-did-not-run";
        } else if (arm_experts == 0) {
            why += " arm-experts";
        }
        if (direct_lines != 0) {
            why += " direct-route-ran";
        }
        if (moved.runs != 1 || moved.declines != 0) {
            why += " arm-counters";
        }
    } else if (admitted) {
        if (arm_lines != 0) {
            why += " arm-ran";
        }
        if (decline_lines == 0) {
            why += " no-decline-line";
        }
        if (direct_lines == 0) {
            why += " direct-route-did-not-run";
        }
        if (moved.runs != 0 || moved.declines != 1 || moved.reason_declines != 1) {
            why += " decline-counters";
        }
    }
    if (refusals != 0) {
        why += " refusal";
    }
    std::fprintf(stderr,
                 "CASE %s tensor=%s k=%d m=%zu experts=%d used=%d ne11=%d tokens=%d active=%d want=%s supports_op=%d "
                 "status=%d max_row_nmse=%.3e arm_lines=%zu arm_experts_lines=%zu direct_lines=%zu decline_lines=%zu "
                 "refusals=%zu runs+=%llu declines+=%llu %s+=%llu -> %s%s\n",
                 c.label, name, k_in, m, k_experts, k_used, c.ne11, c.n_tokens, c.active, decline ? decline : "arm",
                 admitted ? 1 : 0, static_cast<int>(status), err, arm_lines, arm_experts, direct_lines, decline_lines,
                 refusals, (unsigned long long) moved.runs, (unsigned long long) moved.declines, reason,
                 (unsigned long long) moved.reason_declines, why.empty() ? "PASS" : "FAIL:", why.c_str());
    return why.empty();
}

// The failpoint case: the arm throws at its GEMM stage once, after its gather was submitted. The compute must fail
// with the arm's WARN line and graph_compute's recoverable fallback line, both naming the stage, with the failpoint
// consumed once, no run or decline counted, and neither the arm's run line nor the direct route's line: a thrown
// submission does not fall through to the direct kernel.
bool run_thrown_case(const moe_planned_fixture::planned_model & model, const case_data & c) {
    const char * name     = model.gate->name;
    const bool   admitted = supports(model.backend, model.gate, c);

    const ggml_sycl_moe_bf16_grouped_test_counters before          = arm_counters("submit");
    const uint64_t                                 consumed_before = failures_consumed();
    const size_t                                   from            = g_log.mark();
    ggml_sycl_q1_nvfp4_test_failure_once(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_BF16_GROUPED_AFTER_GATHER);
    std::vector<float> got;
    const ggml_status  status   = admitted ? run_backend(model.backend, model.gate, c, got) : GGML_STATUS_FAILED;
    const uint64_t     consumed = failures_consumed() - consumed_before;
    if (consumed == 0) {
        // The node never reached the failpoint; disarm it so the next case measures only itself.
        ggml_sycl_q1_nvfp4_test_failure_once(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_NONE);
    }
    const counter_moves moved = counters_since(before, "submit");

    const std::string named_warn  = std::string("tensor=") + name + " device=0 stage=expert GEMM error=";
    const size_t      warns       = g_log.count_since(from, { "[MoE] BF16 grouped arm failed after submission: " });
    const size_t      named_warns = g_log.count_since(
        from, { "[MoE] BF16 grouped arm failed after submission: ", named_warn.c_str(), "private-testing failpoint" });
    const size_t fallbacks    = g_log.count_since(from, { "recoverable runtime fallback failed" });
    const size_t named_errors = g_log.count_since(
        from, { "recoverable runtime fallback failed: BF16 grouped arm: expert GEMM failed after submission: "
                "private-testing failpoint after the gather" });
    const size_t arm_lines    = g_log.count_since(from, { "[MOE-ROUTE] bf16 grouped tensor=", name });
    const size_t direct_lines = g_log.count_since(from, { "[MOE-ROUTE] bf16 direct tensor=", name });

    std::string why;
    if (!admitted) {
        why += " supports-op";
    } else if (status != GGML_STATUS_FAILED) {
        why += " compute-status";
    }
    if (consumed != 1) {
        why += " failpoint-consumed";
    }
    if (warns != 1 || named_warns != 1) {
        why += " arm-warn";
    }
    if (fallbacks != 1 || named_errors != 1) {
        why += " named-error";
    }
    if (arm_lines != 0) {
        why += " arm-ran";
    }
    if (direct_lines != 0) {
        why += " direct-route-ran";
    }
    if (moved.runs != 0 || moved.declines != 0) {
        why += " counters";
    }
    std::fprintf(stderr,
                 "CASE %s tensor=%s tokens=%d want=thrown supports_op=%d status=%d failpoint_consumed=%llu warns=%zu "
                 "named_warns=%zu fallback_lines=%zu named_errors=%zu arm_lines=%zu direct_lines=%zu runs+=%llu "
                 "declines+=%llu -> %s%s\n",
                 c.label, name, c.n_tokens, admitted ? 1 : 0, static_cast<int>(status), (unsigned long long) consumed,
                 warns, named_warns, fallbacks, named_errors, arm_lines, direct_lines, (unsigned long long) moved.runs,
                 (unsigned long long) moved.declines, why.empty() ? "PASS" : "FAIL:", why.c_str());
    return why.empty();
}

// Runs the opt-out case in a child process: this binary again, as --opt-out, so the backend reads the opt-out from a
// fresh start. The child's CASE line shares this process's stderr; its exit status is the verdict.
bool run_opt_out_child() {
    std::fflush(stdout);
    std::fflush(stderr);
    const pid_t pid = fork();
    if (pid == 0) {
        execl("/proc/self/exe", "test-moe-bf16-grouped-device", "--opt-out", static_cast<char *>(nullptr));
        _exit(127);
    }
    if (pid < 0) {
        std::fprintf(stderr, "FAIL: fork for the opt-out case failed\n");
        return false;
    }
    int        status = 0;
    const bool waited = waitpid(pid, &status, 0) == pid;
    if (waited && WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        return true;
    }
    if (!waited) {
        std::fprintf(stderr, "FAIL: waitpid for the opt-out case failed\n");
    } else if (WIFEXITED(status)) {
        // 77 is a skip, which this process ruled out before it started the child; 127 is a failed exec.
        std::fprintf(stderr, "FAIL: the opt-out case exited %d\n", WEXITSTATUS(status));
    } else {
        std::fprintf(stderr, "FAIL: the opt-out case died on signal %d\n", WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    }
    return false;
}

}  // namespace

int main(int argc, char ** argv) {
    const bool opt_out = argc == 2 && std::strcmp(argv[1], "--opt-out") == 0;
    if (argc != 1 && !opt_out) {
        std::fprintf(stderr, "usage: %s [--opt-out]\n", argv[0]);
        return 1;
    }
    // Read once by the backend, so they are set before anything touches it.
    if (opt_out) {
        setenv("GGML_SYCL_MOE_BF16_GROUPED", "0", 1);
    }
    setenv("GGML_SYCL_MOE_ROUTE_LOG", "1", 1);
    ggml_log_set(log_callback, nullptr);
    // The device probe is the one place a SYCL exception means "no usable GPU here"; after it, any exception fails.
    try {
        bool have_gpu = false;
        for (const auto & device : sycl::device::get_devices()) {
            have_gpu |= device.is_gpu();
        }
        if (!have_gpu || ggml_backend_sycl_get_device_count() < 1) {
            std::fprintf(stderr, "SKIP: no usable SYCL GPU\n");
            return LLAMA_TEST_EXIT_SKIP;
        }
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SKIP: no usable SYCL GPU: %s\n", e.what());
        return LLAMA_TEST_EXIT_SKIP;
    }
    try {
        if (!ggml_backend_sycl_weights_evictable()) {
            std::fprintf(stderr,
                         "SKIP: GGML_SYCL_WEIGHTS_EVICTABLE is off, so libllama would load these experts into the "
                         "device buffer, which this fixture does not model\n");
            return LLAMA_TEST_EXIT_SKIP;
        }
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(
            ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), &ggml_backend_free);
        require(cpu != nullptr, "CPU backend init failed");

        std::mt19937 rng(24);
        if (opt_out) {
            moe_planned_fixture::planned_model model(k_gate_up_shape, "test-moe-bf16-grouped-device-gate_up.gguf", rng);
            const cpu_weights                  reference(cpu.get(), model);
            const case_data                    c = make_case("opt-out", 512, 1, k_experts, rng);
            return run_case(model, cpu.get(), reference, c, "opt_out") ? 0 : 1;
        }

        int        failed = 0;
        int        cases  = 0;
        const auto tally  = [&](bool passed) {
            ++cases;
            failed += passed ? 0 : 1;
        };
        // The opt-out case first, in its own process, so this one holds no model while the child runs.
        tally(run_opt_out_child());
        {
            moe_planned_fixture::planned_model model(k_gate_up_shape, "test-moe-bf16-grouped-device-gate_up.gguf", rng);
            const cpu_weights                  reference(cpu.get(), model);
            const auto run = [&](const char * label, int n_tokens, int n_select, const char * decline) {
                tally(run_case(model, cpu.get(), reference, make_case(label, n_tokens, 1, n_select, rng), decline));
            };
            run("gate_up-all", 512, k_experts, nullptr);
            run("gate_up-subset", 512, k_subset, nullptr);
            run("gate_up-min-tokens", k_min_tokens, k_experts, nullptr);
            run("gate_up-below-min", k_min_tokens - 1, k_experts, "tokens");
            // A thrown submission fails the graph by name; the process and the arm go on.
            tally(run_thrown_case(model, make_case("gate_up-thrown-after-gather", 512, 1, k_experts, rng)));
            run("gate_up-after-thrown", 512, k_experts, nullptr);
        }
        {
            moe_planned_fixture::planned_model model(k_down_shape, "test-moe-bf16-grouped-device-down.gguf", rng);
            const cpu_weights                  reference(cpu.get(), model);
            tally(run_case(model, cpu.get(), reference, make_case("down-all", 512, k_used, k_experts, rng), nullptr));
        }
        if (failed != 0) {
            std::fprintf(stderr, "FAIL: %d of %d cases failed\n", failed, cases);
            return 1;
        }
        std::fprintf(stderr, "PASS: BF16 grouped prompt arm, %d cases\n", cases);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
