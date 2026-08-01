// Follow-up probe #4 for llama.cpp-403s.
//
// probe-device-init, probe-kernel-launch (256 distinct JIT kernels x 3
// devices), and probe-concurrent (8 concurrent threads x 3 devices) are ALL
// flat -- no TTM shmem growth. The mechanism must be specific to something
// ggml-sycl's own code path does that raw SYCL kernel dispatch doesn't.
//
// ggml/src/ggml-sycl/fattn-onednn.cpp builds a 5-op oneDNN GRAPH (not the
// classic primitive API) for flash attention SDPA: MatMul->Divide->[Add
// mask]->SoftMax->MatMul, then calls partition.compile(). This is a
// materially different code path from raw SYCL/L0 kernel dispatch -- it
// goes through oneDNN's own graph-partitioning/codegen backend, which is
// its own JIT with its own internal (unified-cache-invisible) buffer
// management. This probe reproduces exactly that build+compile pattern,
// standalone, with small dims representative of a tiny model, and samples
// /proc/meminfo before/after graph.finalize()/get_partitions()/compile().
//
// Usage: ./probe-sdpa-compile <ndevices> <ncompiles_per_device>
//   ncompiles_per_device > 1 varies ne11 (KV length) per compile, mimicking
//   sdpa_shape_key's cache growing one entry per distinct KV length --
//   exactly what happens during autoregressive decode if the PP-only
//   eligibility gate (ncols>=8) does not actually block it.

#include <sycl/sycl.hpp>
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace dnnl::graph;
using lt       = logical_tensor;
using dim_t    = lt::dim;
using dims_t   = lt::dims;
using dt       = lt::data_type;

static void sample_meminfo(const char * label, bool * abort_flag) {
    std::ifstream f("/proc/meminfo");
    std::string line;
    long shmem_kb = -1, avail_kb = -1;
    while (std::getline(f, line)) {
        if (line.rfind("Shmem:", 0) == 0) sscanf(line.c_str(), "Shmem: %ld kB", &shmem_kb);
        else if (line.rfind("MemAvailable:", 0) == 0) sscanf(line.c_str(), "MemAvailable: %ld kB", &avail_kb);
    }
    fprintf(stderr, "[meminfo] %-32s Shmem=%.2f GB  MemAvailable=%.2f GB\n",
            label, shmem_kb / 1048576.0, avail_kb / 1048576.0);
    fflush(stderr);
    if (abort_flag && avail_kb > 0 && avail_kb < 60L * 1048576L) {
        fprintf(stderr, "GUARD TRIP: MemAvailable < 60GB, aborting\n");
        *abort_flag = true;
    }
}

// Build + compile one non-GQA, no-mask SDPA partition, mirroring
// build_and_compile_sdpa() in fattn-onednn.cpp (simplified: dense
// contiguous strides, no mask, no GQA -- just enough to exercise
// graph.finalize()/get_partitions()/compile()).
static compiled_partition build_and_compile(const dnnl::engine & eng,
                                             dim_t batch, dim_t H, dim_t ncols, dim_t ne11, dim_t D) {
    size_t id = 0;

    dims_t q_dims = { batch, H, ncols, D };
    dims_t k_dims = { batch, H, ne11, D };
    dims_t v_dims = { batch, H, ne11, D };
    dims_t score_dims = { batch, H, ncols, ne11 };
    dims_t out_dims = { batch, H, ncols, D };

    dims_t q_strides = { H * ncols * D, ncols * D, D, 1 };
    dims_t k_strides = { H * ne11 * D, ne11 * D, D, 1 };
    dims_t v_strides = { H * ne11 * D, ne11 * D, D, 1 };
    dims_t score_strides = { H * ncols * ne11, ncols * ne11, ne11, 1 };
    dims_t out_strides = { H * ncols * D, D, H * D, 1 };

    dims_t scale_dims = { 1 };
    dims_t scale_strides = { 1 };

    lt lt_q(id++, dt::f16, q_dims, q_strides);
    lt lt_k(id++, dt::f16, k_dims, k_strides);
    lt lt_scale(id++, dt::f16, scale_dims, scale_strides);
    lt lt_v(id++, dt::f16, v_dims, v_strides);

    lt lt_score(id++, dt::f32, score_dims, score_strides);
    lt lt_scaled(id++, dt::f32, score_dims, score_strides);
    lt lt_probs(id++, dt::f16, score_dims, score_strides);
    lt lt_out(id++, dt::f32, out_dims, out_strides);

    op bmm1(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({ lt_q, lt_k });
    bmm1.add_outputs({ lt_score });

    op div_op(id++, op::kind::Divide, "scale_div");
    div_op.add_inputs({ lt_score, lt_scale });
    div_op.add_outputs({ lt_scaled });

    op sfmx(id++, op::kind::SoftMax, "softmax");
    sfmx.set_attr<int64_t>(op::attr::axis, -1);
    sfmx.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    sfmx.add_inputs({ lt_scaled });
    sfmx.add_outputs({ lt_probs });

    op bmm2(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({ lt_probs, lt_v });
    bmm2.add_outputs({ lt_out });

    dnnl::graph::graph g(dnnl::engine::kind::gpu);
    g.add_op(bmm1);
    g.add_op(div_op);
    g.add_op(sfmx);
    g.add_op(bmm2);
    g.finalize();

    auto parts = g.get_partitions();
    if (parts.empty() || !parts[0].is_supported()) {
        throw std::runtime_error("no supported partition");
    }
    std::vector<lt> in_ports  = { lt_q, lt_k, lt_scale, lt_v };
    std::vector<lt> out_ports = { lt_out };
    return parts[0].compile(in_ports, out_ports, eng);
}

int main(int argc, char ** argv) {
    int want_n     = argc > 1 ? atoi(argv[1]) : 1;
    int ncompiles  = argc > 2 ? atoi(argv[2]) : 1;

    bool stop = false;
    sample_meminfo("start", &stop);

    std::vector<sycl::device> devices;
    auto platforms = sycl::platform::get_platforms();
    for (auto & p : platforms) {
        std::string pname = p.get_info<sycl::info::platform::name>();
        if (pname.find("Level-Zero") == std::string::npos && pname.find("Level Zero") == std::string::npos) continue;
        for (auto & d : p.get_devices()) if (d.is_gpu()) devices.push_back(d);
    }
    if ((int) devices.size() < want_n) {
        fprintf(stderr, "requested %d devices but only %zu visible\n", want_n, devices.size());
        return 1;
    }
    devices.resize(want_n);

    // Tiny-model-representative dims: D=64, H=8 (e.g. stories15M-ish head
    // config), ncols=8 (PP batch, matches the ncols>=8 eligibility gate),
    // ne11 varied per compile to mimic KV-length-keyed cache growth.
    const dim_t D = 64, H = 8, batch = 1, ncols = 8;

    std::vector<compiled_partition> keep_alive; // don't let compiled partitions get destroyed
    std::vector<sycl::queue> queues;

    for (int d = 0; d < want_n && !stop; d++) {
        sycl::queue q(devices[d], sycl::property::queue::in_order());
        dnnl::engine eng = dnnl::sycl_interop::make_engine(devices[d], q.get_context());
        queues.push_back(q);

        char label[80];
        snprintf(label, sizeof(label), "dev%d: before any compile", d);
        sample_meminfo(label, &stop);

        for (int c = 0; c < ncompiles && !stop; c++) {
            dim_t ne11 = 16 + c * 16; // distinct KV length per compile -> distinct shape key
            try {
                auto cp = build_and_compile(eng, batch, H, ncols, ne11, D);
                keep_alive.push_back(std::move(cp));
            } catch (std::exception & e) {
                fprintf(stderr, "dev%d compile %d (ne11=%lld) FAILED: %s\n", d, c, (long long) ne11, e.what());
            }
            snprintf(label, sizeof(label), "dev%d: after compile %d (ne11=%lld)", d, c, (long long) ne11);
            sample_meminfo(label, &stop);
        }
    }

    sample_meminfo("all done, holding compiled partitions", &stop);
    keep_alive.clear();
    sample_meminfo("after partitions destroyed", nullptr);

    return stop ? 2 : 0;
}
