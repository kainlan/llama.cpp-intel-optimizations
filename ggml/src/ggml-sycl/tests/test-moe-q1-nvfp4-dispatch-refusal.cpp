// A Q1_0/NVFP4 AOS MUL_MAT_ID whose batched dispatch refuses ends the op with a named refusal and a failed compute,
// and the process goes on: the same op, run again in the same process, succeeds and matches the CPU backend.
//
// On the card such an operand has one executor, mmvq_moe_batched_dispatch, and the op refuses when it returns false.
// No ordinary run makes it return false, so this test arms the private-testing failpoint
// GGML_SYCL_Q1_NVFP4_TEST_FAILURE_BATCHED_DISPATCH, which makes the dispatch refuse once with the reason
// test-failpoint, just before its submit. It therefore links the private backend objects (ggml-sycl-private-fixtures).
// Every build runs the MMID direct route, which takes such an op before the batched dispatch sees it;
// ggml_sycl::test_moe_mmid_route_reachable_override_guard(false) closes that route for the whole run, so the model
// plans no MMID workspace, the direct block declines each op for want of the workspace's queue capability
// ([MOE-DIRECT-DECLINE] ... step=queue_capability, a decline and not a refusal), and the op falls through to the
// retained route core, where the batched dispatch and its failpoint are.
//
// The vehicle is the mixed-residency fixture's (moe-mixed-residency-fixture.hpp, which both tests include;
// test-moe-mixed-residency-q1-nvfp4.cpp describes the load): a planned NVFP4 model of three expert tensors whose plan
// the fixture edits to put 7 of 16 experts on the host. The op is the gate tensor at one token with ids {1, 4, 7, 10}:
// expert 7 runs on the card and the other three on the host.
//
// The first run, with the failpoint armed, passes when:
//   * the compute returns GGML_STATUS_FAILED;
//   * the failpoint was consumed exactly once (failure_consumed +1);
//   * exactly one WARN says the Q1_0/NVFP4 AOS batched dispatch of this tensor returned false, with
//     reason=test-failpoint requested_bytes=0, and no other batched-dispatch refusal appeared;
//   * graph_compute logged the recoverable fallback with the exception text that names test-failpoint.
// The second run, with nothing armed, on the same graph, passes when it meets the mixed-residency case's bar: the
// compute succeeds, every value is finite (dst is filled with NaN first), the device rows and the host rows each match
// the CPU backend at NMSE <= 5e-4, one [MOE-ROUTE] partition line has local=1 host=3 and nothing else, nothing refused,
// and the failpoint was not consumed again.
//
// The process exits 1 if either run failed, and 0 when both passed. It exits 77 when there is no SYCL GPU, when more
// than one card is visible, when the weights are not evictable, or when the published plan leaves the op's experts on
// one side only. Pin one card: the registration selects level_zero:1, the B50.
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl-test.hpp"  // test_moe_mmid_route_reachable_override_guard
#include "ggml-sycl.h"
#include "ggml.h"
#include "moe-mixed-residency-fixture.hpp"  // the shape, the log capture, the route parser and the model's lifecycle
#include "q1-nvfp4-production-route-test-seam.hpp"  // the batched-dispatch failpoint and its counter

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

using namespace moe_mixed_residency_fixture;

namespace {

constexpr ggml_type k_type = GGML_TYPE_NVFP4;

// The op's picks: 7 on the card, 1, 4 and 10 on the host (k_host_experts), so a route line that swapped the two
// counts fails.
constexpr int32_t k_ids[k_used] = { 1, 4, 7, 10 };

constexpr const char * k_refused_warn  = "[MOE-ROUTE] Q1_0/NVFP4 AOS batched dispatch returned false";
constexpr const char * k_refused_cause = "reason=test-failpoint requested_bytes=0";
constexpr const char * k_fallback_line =
    "recoverable runtime fallback failed: MUL_MAT_ID batched dispatch for a Q1_0/NVFP4 AOS operand refused: "
    "test-failpoint";

uint64_t failures_consumed() {
    ggml_sycl_q1_nvfp4_test_counters counters{};
    ggml_sycl_q1_nvfp4_test_counters_read(&counters);
    return counters.failure_consumed;
}

// One MUL_MAT_ID op on the gate tensor at one token, its activation broadcast to every used expert, built once on
// `backend` so it can be computed more than once on the same graph. The SYCL op passes the fixture's loaded tensor as
// `loaded_weights`; the CPU reference passes null, and `weight_bytes` go into a fresh tensor on the CPU backend.
struct gate_op {
    ggml_backend_t                                      backend;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ nullptr, &ggml_free };
    std::vector<ggml_backend_buffer_t>                  buffers;
    ggml_tensor *                                       dst   = nullptr;
    ggml_cgraph *                                       graph = nullptr;

    gate_op(ggml_backend_t               on_backend,
            ggml_tensor *                loaded_weights,
            const std::vector<uint8_t> & weight_bytes,
            const std::vector<float> &   input_values) :
        backend(on_backend) {
        ggml_init_params params = { 1024 * 1024, nullptr, true };
        ctx.reset(ggml_init(params));
        require(ctx != nullptr, "ggml_init failed");
        ggml_tensor * weights = loaded_weights;
        if (!weights) {
            weights = ggml_new_tensor_3d(ctx.get(), k_type, k_in, k_out, k_experts);
            ggml_set_name(weights, k_tensor_names[k_gate]);
        }
        ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k_in, 1, 1);
        ggml_tensor * ids   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, k_used, 1);
        ggml_set_name(input, "moe_input");
        ggml_set_name(ids, "moe_ids");
        dst = ggml_mul_mat_id(ctx.get(), weights, input, ids);
        ggml_set_name(dst, "moe_out");

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        buffers.push_back(alloc_tensor(buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE));
        buffers.push_back(alloc_tensor(buft, ids, GGML_BACKEND_BUFFER_USAGE_COMPUTE));
        buffers.push_back(alloc_tensor(buft, dst, GGML_BACKEND_BUFFER_USAGE_COMPUTE));
        if (!loaded_weights) {
            buffers.push_back(alloc_tensor(buft, weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS));
        }
        for (ggml_backend_buffer_t buffer : buffers) {
            require(buffer != nullptr, "tensor buffer allocation failed");
        }
        if (!loaded_weights) {
            ggml_backend_tensor_set(weights, weight_bytes.data(), 0, weight_bytes.size());
        }
        ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size() * sizeof(float));
        ggml_backend_tensor_set(ids, k_ids, 0, sizeof(k_ids));
        graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, dst);
    }

    ~gate_op() {
        for (ggml_backend_buffer_t buffer : buffers) {
            if (buffer) {
                ggml_backend_buffer_free(buffer);
            }
        }
    }

    gate_op(const gate_op &)             = delete;
    gate_op & operator=(const gate_op &) = delete;

    // Fills dst with NaN, computes the graph and, on success, reads dst into `out`.
    ggml_status compute(std::vector<float> & out) {
        const std::vector<float> poison(size_t(k_out) * k_used, std::numeric_limits<float>::quiet_NaN());
        ggml_backend_tensor_set(dst, poison.data(), 0, poison.size() * sizeof(float));
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        out.clear();
        if (status == GGML_STATUS_SUCCESS) {
            out.resize(poison.size());
            ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
        }
        return status;
    }
};

size_t count_needles(size_t from) {
    size_t n = 0;
    for (const char * needle : k_refusal_needles) {
        n += g_log.since(from, { needle }).size();
    }
    return n;
}

// The first run: the failpoint armed, the op must end with the named refusal and a failed compute.
bool run_refused(gate_op & op) {
    const uint64_t consumed_before = failures_consumed();
    const size_t   from            = g_log.mark();
    ggml_sycl_q1_nvfp4_test_failure_once(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_BATCHED_DISPATCH);
    std::vector<float> got;
    const ggml_status  status   = op.compute(got);
    const uint64_t     consumed = failures_consumed() - consumed_before;
    if (consumed == 0) {
        // The dispatch never reached the failpoint; disarm it so the second run measures only itself.
        ggml_sycl_q1_nvfp4_test_failure_once(GGML_SYCL_Q1_NVFP4_TEST_FAILURE_NONE);
    }
    const size_t named_warns = g_log.since(from, { k_refused_warn, k_tensor_names[k_gate], k_refused_cause }).size();
    const size_t any_warns   = g_log.since(from, { "batched dispatch returned false" }).size();
    const size_t fallbacks   = g_log.since(from, { k_fallback_line }).size();

    std::string why;
    if (status != GGML_STATUS_FAILED) {
        why += " compute-status";
    }
    if (consumed != 1) {
        why += " failpoint-consumed";
    }
    if (named_warns != 1 || any_warns != 1) {
        why += " refusal-warn";
    }
    if (fallbacks != 1) {
        why += " fallback-line";
    }
    std::fprintf(stderr,
                 "RUN name=refused status=%d failpoint_consumed=%llu named_warns=%zu any_warns=%zu fallback_lines=%zu "
                 "-> %s%s\n",
                 static_cast<int>(status), (unsigned long long) consumed, named_warns, any_warns, fallbacks,
                 why.empty() ? "PASS" : "FAIL:", why.c_str());
    return why.empty();
}

// The second run: nothing armed, the same graph must compute what the CPU backend computes, on both sides.
bool run_recovered(gate_op &                  op,
                   const std::vector<float> & ref,
                   const lifecycle_fixture &  life,
                   size_t                     want_local,
                   size_t                     want_host) {
    const uint64_t     consumed_before = failures_consumed();
    const size_t       from            = g_log.mark();
    std::vector<float> got;
    const ggml_status  status   = op.compute(got);
    const uint64_t     consumed = failures_consumed() - consumed_before;

    // The device rows ([0]) and the host rows ([1]) against the CPU reference, apart.
    double err[2]    = {};
    double norm[2]   = {};
    size_t nonfinite = 0;
    if (status == GGML_STATUS_SUCCESS) {
        for (size_t o = 0; o < size_t(k_used); ++o) {
            const int side = life.on_host[k_gate][k_ids[o]] ? 1 : 0;
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

    const std::string  route_key   = std::string("[MOE-ROUTE] tensor=") + k_tensor_names[k_gate] + " layer=";
    const auto         route_lines = g_log.since(from, { route_key.c_str() });
    const route_counts route       = route_lines.size() == 1 ? parse_route_line(route_lines[0]) : route_counts{};
    const size_t       refusals    = count_needles(from);

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
    if (route_lines.size() != 1) {
        why += " route-lines";
    } else if (route.local != (long long) want_local || route.host != (long long) want_host || route.other_gpu != 0 ||
               route.missing != 0 || route.layout_mismatch != 0) {
        why += " route-counts";
    }
    if (refusals != 0) {
        why += " refusal";
    }
    if (consumed != 0) {
        why += " failpoint-consumed";
    }
    std::fprintf(stderr,
                 "RUN name=recovered status=%d want_local=%zu want_host=%zu route_lines=%zu local=%lld other_gpu=%lld "
                 "host=%lld missing=%lld layout_mismatch=%lld nmse_device=%.3e nmse_host=%.3e nonfinite=%zu "
                 "refusals=%zu failpoint_consumed=%llu -> %s%s\n",
                 static_cast<int>(status), want_local, want_host, route_lines.size(), route.local, route.other_gpu,
                 route.host, route.missing, route.layout_mismatch, nmse(0), nmse(1), nonfinite, refusals,
                 (unsigned long long) consumed, why.empty() ? "PASS" : "FAIL:", why.c_str());
    return why.empty();
}

}  // namespace

int main() {
    // Read once by the backend, so it is set before anything touches it.
    setenv("GGML_SYCL_MOE_ROUTE_LOG", "1", 1);
    ggml_log_set(log_callback, nullptr);
    try {
        check_route_parser();
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
        // Plan, bind and run as an ordinary build does, for the whole run: the fixture's lifecycle is destroyed first.
        ggml_sycl::test_moe_mmid_route_reachable_override_guard ordinary_route(false);

        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(
            ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), &ggml_backend_free);
        require(cpu != nullptr, "CPU backend init failed");

        std::mt19937      rng(1931);
        lifecycle_fixture life(k_type, "test-moe-q1-nvfp4-dispatch-refusal.gguf", rng);

        size_t want_local = 0;
        size_t want_host  = 0;
        for (int32_t expert : k_ids) {
            ++(life.on_host[k_gate][expert] ? want_host : want_local);
        }
        if (want_local == 0 || want_host == 0) {
            throw test_skip("one-sided", "the published plan puts the op's experts on one side only (local=" +
                                             std::to_string(want_local) + " host=" + std::to_string(want_host) + ")");
        }

        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float>                    input(k_in);
        for (float & v : input) {
            v = dist(rng);
        }
        std::vector<float> ref;
        {
            gate_op reference(cpu.get(), nullptr, life.weight_bytes, input);
            require(reference.compute(ref) == GGML_STATUS_SUCCESS, "CPU reference compute failed");
        }

        bool ok = true;
        {
            gate_op op(life.backend, life.tensors[k_gate], life.weight_bytes, input);
            ok = run_refused(op) && ok;
            ok = run_recovered(op, ref, life, want_local, want_host) && ok;
        }
        if (!ok) {
            std::fprintf(stderr, "FAIL: the refused run or the recovered run failed\n");
            return 1;
        }
        std::fprintf(stderr, "PASS: the refused op failed by name and the same op then matched the CPU backend\n");
        return 0;
    } catch (const test_skip & e) {
        std::fprintf(stderr, "SKIP[%s]: %s\n", e.tag, e.what());
        return LLAMA_TEST_EXIT_SKIP;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
