// Canary 4: oneDNN 3.11 graph SDPA fusion smoke test
// Verifies that the 5-op SDPA pattern fuses into a single partition on Arc B580.

#include <iostream>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

using namespace dnnl::graph;
using layout_type = logical_tensor::layout_type;
using dim         = logical_tensor::dim;
using dims        = logical_tensor::dims;
using dt          = logical_tensor::data_type;

struct sdpa_dims {
    dims q;
    dims kv;
    dims mask;
    dims score;
};

// -------------------------------------------------------------------
// Helper: print one partition's summary line
// (graph API exposes op IDs after finalisation, not op kinds)
// -------------------------------------------------------------------
static void describe_partition(size_t idx, const partition & p) {
    std::cout << "  Partition " << idx
              << ": ops=" << p.get_ops_num()
              << " is_supported=" << (p.is_supported() ? "YES" : "NO")
              << " id=" << p.get_id() << "\n";
}

// -------------------------------------------------------------------
// Build the 5-op SDPA graph, report partitions, and attempt compile.
// Pattern: MatMul(Q,K^T) -> Divide(scale) -> Add(mask) -> SoftMax -> MatMul(V)
// -------------------------------------------------------------------
static int run_sdpa(const sycl::queue & q, const sdpa_dims & sh,
        const char * label) {
    std::cout << "\n=== " << label << " ===\n";

    dnnl::engine eng = dnnl::sycl_interop::make_engine(
            q.get_device(), q.get_context());

    size_t id   = 0;
    const dims sc_sz = {1};

    auto lt_q    = logical_tensor(id++, dt::f16, sh.q,     layout_type::strided);
    auto lt_k    = logical_tensor(id++, dt::f16, sh.kv,    layout_type::strided);
    auto lt_scr  = logical_tensor(id++, dt::f32, sh.score, layout_type::strided);
    auto lt_sc   = logical_tensor(id++, dt::f16, sc_sz,    layout_type::strided);
    auto lt_ssc  = logical_tensor(id++, dt::f32, sh.score, layout_type::strided);
    auto lt_msk  = logical_tensor(id++, dt::f16, sh.mask,  layout_type::strided);
    auto lt_mssc = logical_tensor(id++, dt::f32, sh.score, layout_type::strided);
    auto lt_prob = logical_tensor(id++, dt::f16, sh.score, layout_type::strided);
    auto lt_v    = logical_tensor(id++, dt::f16, sh.kv,    layout_type::strided);
    auto lt_out  = logical_tensor(id++, dt::f16, sh.q,     layout_type::strided);

    op bmm1(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({lt_q, lt_k});
    bmm1.add_outputs({lt_scr});

    op div_op(id++, op::kind::Divide, "scale_div");
    div_op.add_inputs({lt_scr, lt_sc});
    div_op.add_outputs({lt_ssc});

    op add_op(id++, op::kind::Add, "mask_add");
    add_op.add_inputs({lt_ssc, lt_msk});
    add_op.add_outputs({lt_mssc});

    op sfmx(id++, op::kind::SoftMax, "softmax");
    sfmx.set_attr<int64_t>(op::attr::axis, -1);
    sfmx.add_inputs({lt_mssc});
    sfmx.add_outputs({lt_prob});

    op bmm2(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({lt_prob, lt_v});
    bmm2.add_outputs({lt_out});

    graph sdpa_graph(engine::kind::gpu);
    sdpa_graph.add_op(bmm1);
    sdpa_graph.add_op(div_op);
    sdpa_graph.add_op(add_op);
    sdpa_graph.add_op(sfmx);
    sdpa_graph.add_op(bmm2);
    sdpa_graph.finalize();

    auto partitions = sdpa_graph.get_partitions();
    std::cout << "Partition count: " << partitions.size() << "\n";

    bool all_supported = true;
    for (size_t i = 0; i < partitions.size(); ++i) {
        describe_partition(i, partitions[i]);
        if (!partitions[i].is_supported()) all_supported = false;
    }

    // 0=PASS, 1=SOFT-FAIL, 2=HARD-FAIL; CI checks exit code for fusion health.
    int result;
    if (partitions.size() == 1 && partitions[0].get_ops_num() == 5
            && partitions[0].is_supported()) {
        std::cout << "Graph-build verdict: PASS (single fused partition, 5 ops)\n";
        result = 0;
    } else if (partitions.size() > 1 && all_supported) {
        std::cout << "Graph-build verdict: SOFT-FAIL (split into "
                  << partitions.size() << " partitions, all supported)\n";
        result = 1;
    } else {
        std::cout << "Graph-build verdict: HARD-FAIL (unsupported partition(s))\n";
        result = 2;
    }

    std::cout << "Compile attempts:\n";
    for (size_t i = 0; i < partitions.size(); ++i) {
        if (!partitions[i].is_supported()) {
            std::cout << "  Partition " << i << ": SKIP (not supported)\n";
            continue;
        }
        try {
            auto in_ports  = partitions[i].get_input_ports();
            auto out_ports = partitions[i].get_output_ports();
            compiled_partition cp = partitions[i].compile(in_ports, out_ports, eng);
            std::cout << "  Partition " << i << ": compile OK\n";
        } catch (dnnl::error & e) {
            std::cout << "  Partition " << i << ": compile FAILED — " << e.what()
                      << " (status=" << e.status << ")\n";
        } catch (std::exception & e) {
            std::cout << "  Partition " << i << ": compile FAILED — " << e.what() << "\n";
        }
    }

    return result;
}

int main() {
    try {
        // Select Arc B580 via ONEAPI_DEVICE_SELECTOR=level_zero:0 in env.
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
        std::cout << "Device: "
                  << q.get_device().get_info<sycl::info::device::name>() << "\n";

        // 4-D non-GQA: Q and KV share the same head count.
        int r1 = run_sdpa(q, {
            {1, 32, 512, 128},  // q
            {1, 32, 512, 128},  // kv
            {1,  1, 512, 512},  // mask
            {1, 32, 512, 512},  // score
        }, "4-D SDPA non-GQA 32/32");

        // 4-D GQA: Q has 32 heads, KV has 8; 4:1 broadcast not supported at
        // compile time in the standard 4-D MatMul — expect compile FAILED.
        // Fusion itself succeeds (PASS), so r2=0; compile failure is not a
        // fusion regression and does not make main() exit nonzero.
        int r2 = run_sdpa(q, {
            {1, 32, 512, 128},  // q
            {1,  8, 512, 128},  // kv
            {1,  1, 512, 512},  // mask
            {1, 32, 512, 512},  // score
        }, "4-D SDPA GQA 32q/8kv");

        // 5-D GQA: oneDNN-native layout (mb, kv_heads, head_rep, seq, head_size).
        // Per-column mask; oneDNN broadcasts along the seq-len (rows) dim.
        int r3 = run_sdpa(q, {
            {1, 8, 4, 512, 128},  // q  (head_rep=4 = 32/8)
            {1, 8, 1, 512, 128},  // kv
            {1, 1, 1,   1, 512},  // mask — broadcast over all but last dim
            {1, 8, 4, 512, 512},  // score
        }, "5-D GQA layout");

        std::cout << "\nDone.\n";
        return std::max({r1, r2, r3});
    } catch (dnnl::error & e) {
        std::cerr << "oneDNN error: " << e.what() << " (status " << e.status << ")\n";
        return 1;
    } catch (sycl::exception & e) {
        std::cerr << "SYCL error: " << e.what() << "\n";
        return 1;
    } catch (std::exception & e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
