// Mixed-residency MUL_MAT_ID for Q1_0 and NVFP4 experts on one card: in ONE op, the MMID direct route runs the experts
// the plan keeps on the card in its device kernel and the experts it puts on the host in its host leg (the HOST_CPU
// recipe), and the op's result matches the CPU backend.
//
// For each type, a planned model is loaded the way libllama loads one (test-moe-pp-batched-dequant.cpp describes the
// sequence): its three expert tensors sit in the card's SYCL host buffer type and are registered while the load is
// open. Between the inventory stage and the weight load, the fixture copies the load's candidate plan, moves the
// experts in k_host_experts of every expert tensor to the host (placement_plan::demote_expert_placement_to_host keeps
// the plan's byte terms consistent), and stages the copy as the candidate again. The commit's preload then
// materializes the other experts on the card and registers the moved ones as host experts. The cases take the side
// each expert runs on from the plan the commit published. The fixture, the log capture and the route line parser live
// in moe-mixed-residency-fixture.hpp, which test-moe-q1-nvfp4-dispatch-refusal.cpp includes too.
//
// Each case runs one MUL_MAT_ID op at 1 token (decode), 7 tokens and 512 tokens (prompt), on the gate tensor with the
// activation broadcast to every used expert (ne11 1) and on the down tensor with one activation row per used expert
// (ne11 k_used). Every build runs a Q1_0/NVFP4 op on the MMID direct route, which submits the op's card occurrences to
// its device kernel and its host occurrences to its host leg in one admission, so the witness of the split is the
// route's own: its route-testing counters (q1-nvfp4-production-route-test-seam.hpp). They exist only in the private
// backend objects, so this test links ggml-sycl-private-fixtures, not the shipped DSO. A case passes when:
//   * the compute succeeds and every output value is finite: dst is filled with NaN before the op, so a row that
//     neither side wrote fails;
//   * the rows of the device-planned experts and the rows of the host-planned experts each match the CPU backend at
//     NMSE <= 5e-4, measured apart, so a side that computed nothing or the wrong thing fails on its own;
//   * the direct route took the op once and finished it: its candidate, admit, submit, terminal and recycle counters
//     each moved by one, and quarantine by none;
//   * the route's card leg counted the occurrences of device-planned experts in the case's ids, and its host leg the
//     occurrences of host-planned experts in one submission (none when there are none), so the two sides sum to every
//     occurrence and none went to another card or nowhere;
//   * nothing refused, declined or ran another way: no line in k_refusal_needles, no [MOE-DIRECT-DECLINE] line, and no
//     [MOE-ROUTE] partition line for the tensor appeared while it ran. The retained route core prints that line for
//     every op it runs (GGML_SYCL_MOE_ROUTE_LOG=1, set below), so its absence says the core did not run the op too.
// The route's kernel is the one executor of a card occurrence here and its host leg the one executor of a host
// occurrence, and either one's refusal fails the op by name, so correct rows on both sides of a successful op are what
// each computed.
//
// The process exits 1 if any case failed. Otherwise it exits 77 when a case ran experts on one side only (a plan that
// left every picked expert on one side measures nothing here), and 0 when every case ran both sides. It also exits 77
// when there is no SYCL GPU, when more than one card is visible, or when the weights are not evictable (libllama would
// then load the device-planned experts into the device buffer, which this fixture does not model). Pin one card: the
// registration selects level_zero:1, the B50.
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "moe-mixed-residency-fixture.hpp"  // the shape, the log capture, the route parser and the model's lifecycle
#include "q1-nvfp4-production-route-test-seam.hpp"  // the direct route's counters

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

using namespace moe_mixed_residency_fixture;

namespace {

constexpr ggml_type k_types[]  = { GGML_TYPE_Q1_0, GGML_TYPE_NVFP4 };
constexpr int       k_tokens[] = { 1, 7, 512 };
constexpr int       k_roles[]  = { k_gate, k_down };
constexpr int       k_cases    = int(std::size(k_types) * std::size(k_tokens) * std::size(k_roles));

struct case_data {
    ggml_type            type;
    int                  role;  // the expert tensor the op runs: k_gate or k_down
    int                  n_tokens;
    int                  ne11;  // 1: the activation broadcasts to every used expert (gate); k_used: one row each
    std::vector<float>   input;
    std::vector<int32_t> ids;
};

// Token t picks experts b, b+3, b+6 and b+9 (mod k_experts) from the base b = 5t + t/16 + 1 (mod k_experts), so a
// token's picks are distinct. Over a run of 16 tokens 5t visits every residue once (5 and 16 are coprime), and the t/16
// term shifts each later run by one, so every base recurs and the 512-token op picks every expert; the 7-token op's
// seven bases (1, 6, 11, 0, 5, 10, 15) with their four offsets pick every expert too. Against k_host_experts the picks
// fall on both sides in every op here: 1 card and 3 host occurrences at 1 token, 15 and 13 at 7 tokens, 1152 and 896
// at 512 tokens.
case_data make_case(ggml_type type, int role, int n_tokens, std::mt19937 & rng) {
    const int                             ne11 = role == k_gate ? 1 : k_used;
    case_data                             c{ type, role, n_tokens, ne11, {}, {} };
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    c.input.resize(size_t(k_in) * ne11 * n_tokens);
    for (float & v : c.input) {
        v = dist(rng);
    }
    c.ids.resize(size_t(k_used) * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        const int base = (t * 5 + t / 16 + 1) % k_experts;
        for (int i = 0; i < k_used; ++i) {
            c.ids[size_t(t) * k_used + i] = (base + i * 3) % k_experts;
        }
    }
    return c;
}

// Builds MUL_MAT_ID over the case's input and ids on `backend` and computes it, with dst filled with NaN first. Fills
// `out` on success. The SYCL run passes the fixture's loaded tensor as `loaded_weights`; the CPU reference passes null,
// and `weight_bytes` go into a fresh tensor on the CPU backend.
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
        ggml_set_name(weights, k_tensor_names[c.role]);
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
    const std::vector<float> poison(size_t(k_out) * k_used * c.n_tokens, std::numeric_limits<float>::quiet_NaN());
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

enum class case_result { PASS, FAIL, ONE_SIDED };

// Runs one case and prints its CASE line. Every log count is of the lines that appeared while the SYCL op ran.
case_result run_case(lifecycle_fixture & life, ggml_backend_t cpu, const case_data & c) {
    const char *       tensor_name = k_tensor_names[c.role];
    std::vector<float> ref;
    require(run_backend(cpu, nullptr, life.weight_bytes, c, ref) == GGML_STATUS_SUCCESS,
            std::string("CPU reference compute failed for ") + ggml_type_name(c.type));

    // What the op must route: each occurrence (token, slot) to the side the published plan puts its expert on.
    const auto & on_host    = life.on_host[c.role];
    size_t       want_local = 0;
    size_t       want_host  = 0;
    for (int32_t expert : c.ids) {
        ++(on_host[expert] ? want_host : want_local);
    }

    const size_t                     from = g_log.mark();
    ggml_sycl_q1_nvfp4_test_counters before{};
    ggml_sycl_q1_nvfp4_test_counters after{};
    ggml_sycl_q1_nvfp4_test_counters_read(&before);
    std::vector<float> got;
    const ggml_status  status = run_backend(life.backend, life.tensors[c.role], life.weight_bytes, c, got);
    // run_backend synchronized the backend, so the route's completion task has run and its counters are final.
    ggml_sycl_q1_nvfp4_test_counters_read(&after);

    // The device rows ([0]) and the host rows ([1]) against the CPU reference, apart.
    double err[2]    = {};
    double norm[2]   = {};
    size_t nonfinite = 0;
    if (status == GGML_STATUS_SUCCESS) {
        for (size_t o = 0; o < c.ids.size(); ++o) {
            const int side = on_host[c.ids[o]] ? 1 : 0;
            for (size_t n = 0; n < size_t(k_out); ++n) {
                const size_t i = o * k_out + n;
                if (!std::isfinite(got[i])) {
                    ++nonfinite;
                    continue;
                }
                const double d = double(ref[i]) - double(got[i]);
                err[side] += d * d;
                norm[side] += double(ref[i]) * double(ref[i]);
            }
        }
    }
    auto nmse = [&](int side) {
        return status != GGML_STATUS_SUCCESS ? NAN : err[side] / (norm[side] > 0.0 ? norm[side] : 1.0);
    };

    // The route's own account of the op: how far each counter moved while it ran.
    const auto moved = [&](uint64_t ggml_sycl_q1_nvfp4_test_counters::* counter) {
        return static_cast<unsigned long long>(after.*counter - before.*counter);
    };
    const unsigned long long candidate   = moved(&ggml_sycl_q1_nvfp4_test_counters::candidate);
    const unsigned long long admit       = moved(&ggml_sycl_q1_nvfp4_test_counters::admit);
    const unsigned long long submit      = moved(&ggml_sycl_q1_nvfp4_test_counters::submit);
    const unsigned long long terminal    = moved(&ggml_sycl_q1_nvfp4_test_counters::terminal);
    const unsigned long long recycle     = moved(&ggml_sycl_q1_nvfp4_test_counters::recycle);
    const unsigned long long quarantine  = moved(&ggml_sycl_q1_nvfp4_test_counters::quarantine);
    const unsigned long long host_submit = moved(&ggml_sycl_q1_nvfp4_test_counters::host_submit);
    const unsigned long long card        = moved(&ggml_sycl_q1_nvfp4_test_counters::card_occurrences);
    const unsigned long long host        = moved(&ggml_sycl_q1_nvfp4_test_counters::host_occurrences);
    const bool               route_took_op =
        candidate == 1 && admit == 1 && submit == 1 && terminal == 1 && recycle == 1 && quarantine == 0;
    const bool split_counted = card == want_local && host == want_host && host_submit == (want_host != 0 ? 1u : 0u);

    // The retained route core's partition line, which it prints for every op it runs: the core must not have run it.
    const std::string route_key   = std::string("[MOE-ROUTE] tensor=") + tensor_name + " layer=";
    const size_t      route_lines = g_log.since(from, { route_key.c_str() }).size();
    const size_t      declines    = g_log.since(from, { "[MOE-DIRECT-DECLINE]" }).size();
    size_t            refusals    = 0;
    for (const char * needle : k_refusal_needles) {
        refusals += g_log.since(from, { needle }).size();
    }

    std::string why;
    if (status != GGML_STATUS_SUCCESS) {
        why += " compute-status";
    } else {
        if (nonfinite != 0) {
            why += " nonfinite";
        }
        if (!(nmse(0) <= k_nmse_tol)) {
            why += " nmse-device";
        }
        if (!(nmse(1) <= k_nmse_tol)) {
            why += " nmse-host";
        }
    }
    if (!route_took_op) {
        why += " route-stages";
    }
    if (!split_counted) {
        why += " route-split";
    }
    if (route_lines != 0) {
        why += " retained-core";
    }
    if (declines != 0) {
        why += " decline";
    }
    if (refusals != 0) {
        why += " refusal";
    }
    // A passing case measured both executors only when the op had occurrences on both sides.
    const case_result result = !why.empty()                        ? case_result::FAIL :
                               (want_local == 0 || want_host == 0) ? case_result::ONE_SIDED :
                                                                     case_result::PASS;
    std::fprintf(stderr,
                 "CASE type=%s tensor=%s tokens=%d ne11=%d status=%d want_local=%zu want_host=%zu card=%llu host=%llu "
                 "candidate=%llu admit=%llu submit=%llu terminal=%llu recycle=%llu quarantine=%llu host_submit=%llu "
                 "route_lines=%zu declines=%zu nmse_device=%.3e nmse_host=%.3e nonfinite=%zu refusals=%zu -> %s%s\n",
                 ggml_type_name(c.type), tensor_name, c.n_tokens, c.ne11, static_cast<int>(status), want_local,
                 want_host, card, host, candidate, admit, submit, terminal, recycle, quarantine, host_submit,
                 route_lines, declines, nmse(0), nmse(1), nonfinite, refusals,
                 result == case_result::PASS      ? "PASS" :
                 result == case_result::ONE_SIDED ? "ONE-SIDED" :
                                                    "FAIL:",
                 why.c_str());
    return result;
}

}  // namespace

int main() {
    // Read once by the backend, so it is set before anything touches it.
    setenv("GGML_SYCL_MOE_ROUTE_LOG", "1", 1);
    ggml_log_set(log_callback, nullptr);
    try {
        bool have_gpu = false;
        try {
            for (const auto & device : sycl::device::get_devices()) {
                have_gpu |= device.is_gpu();
            }
        } catch (const sycl::exception & e) {
            throw test_skip("no-gpu", std::string("SYCL device enumeration failed: ") + e.what());
        }
        if (!have_gpu || ggml_backend_sycl_get_device_count() < 1) {
            throw test_skip("no-gpu", "no usable SYCL GPU");
        }
        if (ggml_backend_sycl_get_device_count() > 1) {
            throw test_skip("multi-card",
                            "more than one SYCL card is visible; the expected counts are for one card "
                            "(pin ONEAPI_DEVICE_SELECTOR to one card)");
        }
        if (!ggml_backend_sycl_weights_evictable()) {
            throw test_skip("not-evictable",
                            "GGML_SYCL_WEIGHTS_EVICTABLE is off, so libllama would load the device-planned experts "
                            "into the device buffer, which this fixture does not model");
        }
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(
            ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), &ggml_backend_free);
        require(cpu != nullptr, "CPU backend init failed");

        std::mt19937 rng(1607);
        int          failed    = 0;
        int          one_sided = 0;
        for (ggml_type type : k_types) {
            require(k_in % ggml_blck_size(type) == 0,
                    std::string("k_in is not whole blocks of ") + ggml_type_name(type));
            const std::string split_path = std::string("test-moe-mixed-residency-") + ggml_type_name(type) + ".gguf";
            lifecycle_fixture life(type, split_path, rng);
            for (int n_tokens : k_tokens) {
                for (int role : k_roles) {
                    const case_result result = run_case(life, cpu.get(), make_case(type, role, n_tokens, rng));
                    failed += result == case_result::FAIL ? 1 : 0;
                    one_sided += result == case_result::ONE_SIDED ? 1 : 0;
                }
            }
        }
        if (failed != 0) {
            std::fprintf(stderr, "FAIL: %d of %d cases failed\n", failed, k_cases);
            return 1;
        }
        if (one_sided != 0) {
            std::fprintf(stderr,
                         "SKIP[one-sided]: %d of %d cases ran experts on one side only, so they measured no mixed op\n",
                         one_sided, k_cases);
            return LLAMA_TEST_EXIT_SKIP;
        }
        std::fprintf(stderr, "PASS: %d mixed-residency cases, each with device and host experts in one op\n", k_cases);
        return 0;
    } catch (const test_skip & e) {
        std::fprintf(stderr, "SKIP[%s]: %s\n", e.tag, e.what());
        return LLAMA_TEST_EXIT_SKIP;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
