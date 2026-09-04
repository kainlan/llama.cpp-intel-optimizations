// GPU numerics gate for llama.cpp-lis9 (plan Task G2, parent llama.cpp-30ak7
// lever 2): opt-in CU-scaled K-split for the MXFP4 gate/up decode DPAS
// kernel (mmvq.cpp: mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl /
// mxfp4_pair_glu_xmx_tiled_dpas_m2_submit, profile label
// "mxfp4.gateup.xmx_tiled_dpas_m2"), gated by GGML_SYCL_MXFP4_GATEUP_KSPLIT.
//
// llama.cpp-ulp9 (Task G1) found the B70 launch for this kernel (k=4
// experts, 720 ESIMD threads) sitting on the flat part of its occupancy
// curve -- 8x the K-tile bytes costs only +14% time, elbow between k=8 and
// k=16 -- so splitting each thread's K reduction across S sub-threads and
// combining the partial sums should move the launch toward that elbow. This
// test proves the split is numerically sound, not that it is fast (the lead
// session owns the GPU profile/A-B measurement per CLAUDE.md's GPU
// serialization rule).
//
// STRATEGY: call the exported ggml_sycl::ggml_sycl_mxfp4_pair_glu_bench_launch
// raw-kernel launcher (ggml-sycl-bench.hpp) directly with
// xmx_tiled_m_tiles=2, xmx_tiled_pack_q8=true, xmx_tiled_grouped=false,
// xmx_tiled_prefetch=false -- the exact precondition combination that routes
// to mxfp4_pair_glu_xmx_tiled_dpas_m2_submit<8>() in
// ggml_sycl_mxfp4_pair_glu_bench_launch's own dispatch (mmvq.cpp), bypassing
// the full MoE graph-dispatch heuristics (SOA/XMX_TILED layout selection,
// singlecol/aggressive-partial routing, etc.) that would otherwise make
// "did this run reach kernel X" a second thing to prove blind. This mirrors
// ggml/src/ggml-sycl/tests/test-xmx-moe-mxfp4.cpp's own
// mxfp4_pair_glu_bench_args usage (its xmx_tiled_m_tiles=1 grouped-path
// tests), generalized to the ungrouped xmx_tiled_m_tiles=2 shape and a real
// GPT-OSS gate/up size (K=2880, N=2880, 4 experts, 1 token -- total_batches=4,
// matching the 720-thread launch llama.cpp-ulp9 profiled).
//
// Run S in {1, 2, 4} and compare S=2/S=4 against S=1 (not a CPU reference:
// the plan's acceptance criterion is internal agreement -- DPAS int8
// accumulation is exact, so any drift is purely how the K-split float
// combine reorders the final scale-and-sum, bounded by
// 1e-4 + 1e-3*|ref| per element).
//
// ⚠️ SEPARATE PROCESSES ARE REQUIRED, NOT REPEATED setenv() IN ONE. Every
// GGML_SYCL_* accessor in this fork (common.cpp, mmvq.cpp) reads its env var
// through a `static const ... = []{ ... }();` immediately-invoked lambda --
// read once per process, cached forever after (see e.g.
// mxfp4_moe_gateup_m2_tg1_index_enabled in mmvq.cpp). A second setenv() in
// the same process would be silently ignored by
// ggml_sycl_mxfp4_gateup_ksplit(). So each S value gets fork()+execv() of
// this very binary with a "--child" argv flag (the pattern
// tests/test-sycl-mmvq-q8-0-soa-numerics.cpp uses for
// ONEAPI_DEVICE_SELECTOR, generalized here to a second env var): the setenv()
// calls happen in the forked child BEFORE execv() replaces its process image,
// so the fresh image's static initializers (this one and libccl's, which is
// why ONEAPI_DEVICE_SELECTOR itself needs the same re-exec treatment --
// llama.cpp-2x3m) observe the correct environment from their first read.
//
// Usage:
//   source /opt/intel/oneapi/setvars.sh --force
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-mxfp4-gateup-ksplit-numerics

#include "test-skip.h"

#if !defined(GGML_USE_SYCL)
int main() {
    std::fprintf(stderr, "SKIP: GGML_USE_SYCL not enabled; this run proves NOTHING about the ksplit kernel.\n");
    return LLAMA_TEST_EXIT_SKIP;
}
#else

#    include "ggml-common.h"
#    include "ggml-sycl/common.hpp"
#    include "ggml-sycl/ggml-sycl-bench.hpp"
#    include "ggml-sycl/sycl-kernel-profiler.hpp"
#    include "ggml.h"

#    include <sys/wait.h>
#    include <unistd.h>

#    include <cerrno>
#    include <cmath>
#    include <cstdint>
#    include <cstdio>
#    include <cstdlib>
#    include <cstring>
#    include <fstream>
#    include <optional>
#    include <random>
#    include <string>
#    include <sycl/sycl.hpp>
#    include <vector>

namespace {

// GPT-OSS gate/up shape (K=2880, N=2880), 4 experts used, 1 token: the exact
// shape llama.cpp-ulp9 profiled at k=4 (720 tiles = total_batches(4) *
// m_tile_pairs(180), Repeat=8).
constexpr int   kNcols          = 2880;
constexpr int   kNrowsPerExpert = 2880;
constexpr int   kNExperts       = 4;
constexpr int   kNIds           = 4;
constexpr int   kNTokens        = 1;
constexpr float kAlpha          = 1.702f;  // same alpha/limit GPT-OSS's SWIGLU_OAI FFN uses
constexpr float kLimit          = 7.0f;    // (src/llama-graph.cpp, LLM_FFN_SWIGLU_OAI_MOE)

constexpr int kExecN = static_cast<int>(GGML_SYCL_MXFP4_MOE_XMX_N);
constexpr int kKPer  = static_cast<int>(GGML_SYCL_MXFP4_MOE_XMX_K);

struct shape_layout {
    int64_t q8_row_size             = 0;
    size_t  weight_bytes_per_expert = 0;
    size_t  activation_bytes        = 0;
    size_t  output_floats           = 0;
    int     k_tiles                 = 0;
    size_t  b_packed_elems          = 0;
    size_t  y_scales_elems          = 0;
};

shape_layout compute_shape() {
    shape_layout s{};
    s.q8_row_size = static_cast<int64_t>(kNcols) * static_cast<int64_t>(sizeof(block_q8_1)) / QK8_1;

    const int packed_bytes    = kKPer / 2;
    const int tile_n_total    = kExecN;
    const int group_bytes     = tile_n_total * (1 + packed_bytes);
    const int n_tile_groups_n = (kNrowsPerExpert + tile_n_total - 1) / tile_n_total;
    s.k_tiles                 = kNcols / kKPer;

    s.weight_bytes_per_expert =
        static_cast<size_t>(s.k_tiles) * static_cast<size_t>(n_tile_groups_n) * static_cast<size_t>(group_bytes);
    s.activation_bytes = static_cast<size_t>(kNIds) * static_cast<size_t>(s.q8_row_size);
    s.output_floats = static_cast<size_t>(kNIds) * static_cast<size_t>(kNTokens) * static_cast<size_t>(kNrowsPerExpert);

    const int total_batches = kNIds * kNTokens;
    s.b_packed_elems =
        static_cast<size_t>(total_batches) * static_cast<size_t>(s.k_tiles) * static_cast<size_t>(kKPer * kExecN);
    s.y_scales_elems =
        static_cast<size_t>(total_batches) * static_cast<size_t>(s.k_tiles) * static_cast<size_t>(kExecN);
    return s;
}

// Fills one expert's gate/up weight buffer in the XMX-tiled MXFP4 SOA group
// layout mxfp4_xmx_tiled_load_a_vec_from_group() (mmvq.cpp) reads: K-tile
// major, row-group minor -- for k-tile kt and row-group group_n, group_bytes
// = tile_n_total * (1 + K/2) bytes at offset (kt*n_tile_groups_n+group_n)*
// group_bytes, laid out as [tile_n_total e8m0 exponent bytes][tile_n_total *
// (K/2) packed 4-bit qs bytes, row_in_group-major]. Mirrors
// ggml/src/ggml-sycl/tests/test-xmx-moe-mxfp4.cpp's fill_tiled_weight lambda
// (test_grouped_pair_glu_can_emit_down_q8_artifact), generalized from that
// test's single K-tile/single row-group case to the full GPT-OSS shape.
void fill_tiled_weight(std::vector<uint8_t> & bytes, uint32_t seed) {
    const int packed_bytes    = kKPer / 2;
    const int tile_n_total    = kExecN;
    const int group_bytes     = tile_n_total * (1 + packed_bytes);
    const int n_tile_groups_n = (kNrowsPerExpert + tile_n_total - 1) / tile_n_total;
    const int k_tiles         = kNcols / kKPer;

    std::mt19937                       rng(seed);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    // Keep the e8m0 exponent near 1.0x scale (128 == 2^0) so quantized *
    // scale products stay in a well-conditioned range for the tolerance
    // check below -- this test is about K-split summation order, not about
    // exercising extreme dynamic range.
    std::uniform_int_distribution<int> exp_dist(120, 136);

    for (int kt = 0; kt < k_tiles; ++kt) {
        for (int group_n = 0; group_n < n_tile_groups_n; ++group_n) {
            uint8_t * group = bytes.data() + (static_cast<size_t>(kt) * static_cast<size_t>(n_tile_groups_n) +
                                              static_cast<size_t>(group_n)) *
                                                 static_cast<size_t>(group_bytes);
            for (int row_in_group = 0; row_in_group < tile_n_total; ++row_in_group) {
                const int row = group_n * tile_n_total + row_in_group;
                if (row >= kNrowsPerExpert) {
                    continue;
                }
                group[row_in_group] = static_cast<uint8_t>(exp_dist(rng));
                uint8_t * qs        = group + tile_n_total + row_in_group * packed_bytes;
                for (int i = 0; i < packed_bytes; ++i) {
                    qs[i] = static_cast<uint8_t>(byte_dist(rng));
                }
            }
        }
    }
}

// Fills one Q8 SOA activation row read by mxfp4_dpas_pack_q8_single_col_groups_sycl
// (mmvq.cpp): `ncols` int8 quantized values followed by one (d, s)
// sycl::half pair per QK8_1-sized block (the packer reads only `d`, at
// offset ncols + kt*sizeof(sycl::half2); `s` is filled anyway so the row
// matches block_q8_1's real on-disk layout instead of leaving padding
// uninitialized).
void fill_activation_row(uint8_t * row, uint32_t seed) {
    std::mt19937                       rng(seed);
    std::uniform_int_distribution<int> q_dist(-8, 8);
    for (int i = 0; i < kNcols; ++i) {
        row[i] = static_cast<uint8_t>(static_cast<int8_t>(q_dist(rng)));
    }
    const int    n_blocks = kNcols / QK8_1;
    sycl::half * ds       = reinterpret_cast<sycl::half *>(row + kNcols);
    for (int b = 0; b < n_blocks; ++b) {
        ds[b * 2 + 0] = sycl::half(1.0f / 16.0f);  // d
        ds[b * 2 + 1] = sycl::half(0.0f);          // s (unused by this kernel's packer read)
    }
}

// Exception-safe device selection (sycl::device's default ctor throws
// uncaught on a device-less host -- see
// ggml/src/ggml-sycl/tests/sycl-test-skip.hpp's rationale, not reachable
// from this top-level tests/ directory by relative include, so replicated
// minimally here: the throw stays inside the try).
std::optional<sycl::queue> make_queue() {
    try {
        sycl::device dev(sycl::gpu_selector_v);
        if (!dev.has(sycl::aspect::ext_intel_matrix)) {
            return std::nullopt;
        }
        return sycl::queue(dev);
    } catch (const sycl::exception &) {
        return std::nullopt;
    }
}

// One child process: runs the m2 gate/up decode kernel once at whatever
// GGML_SYCL_MXFP4_GATEUP_KSPLIT this process's environment already carries
// (set by the parent before its execv() -- see main()), then writes the raw
// F32 GLU output and a kernel-profile CSV for the parent to read back.
// Returns 0 on success, LLAMA_TEST_EXIT_SKIP (77) if no XMX GPU is present,
// 1 on any other failure.
int run_child(const std::string & out_prefix) {
    std::optional<sycl::queue> q_opt = make_queue();
    if (!q_opt) {
        std::fprintf(stderr, "SKIP: no XMX-capable SYCL GPU device available\n");
        return LLAMA_TEST_EXIT_SKIP;
    }
    sycl::queue & q = *q_opt;

    const shape_layout shape = compute_shape();

    std::vector<std::vector<uint8_t>> h_gate(kNExperts);
    std::vector<std::vector<uint8_t>> h_up(kNExperts);
    for (int e = 0; e < kNExperts; ++e) {
        h_gate[e].assign(shape.weight_bytes_per_expert, 0);
        h_up[e].assign(shape.weight_bytes_per_expert, 0);
        fill_tiled_weight(h_gate[e], 1000u + static_cast<uint32_t>(e));
        fill_tiled_weight(h_up[e], 2000u + static_cast<uint32_t>(e));
    }
    std::vector<uint8_t> h_activation(shape.activation_bytes, 0);
    for (int slot = 0; slot < kNIds; ++slot) {
        fill_activation_row(h_activation.data() + static_cast<size_t>(slot) * static_cast<size_t>(shape.q8_row_size),
                            3000u + static_cast<uint32_t>(slot));
    }
    const int32_t h_ids[kNIds] = { 0, 1, 2, 3 };

    std::vector<uint8_t *> d_gate(kNExperts, nullptr);
    std::vector<uint8_t *> d_up(kNExperts, nullptr);
    for (int e = 0; e < kNExperts; ++e) {
        d_gate[e] = sycl::malloc_device<uint8_t>(shape.weight_bytes_per_expert, q);
        d_up[e]   = sycl::malloc_device<uint8_t>(shape.weight_bytes_per_expert, q);
    }
    uint8_t *     d_activation = sycl::malloc_device<uint8_t>(shape.activation_bytes, q);
    float *       d_output     = sycl::malloc_device<float>(shape.output_floats, q);
    const void ** d_gate_ptrs  = sycl::malloc_shared<const void *>(kNExperts, q);
    const void ** d_up_ptrs    = sycl::malloc_shared<const void *>(kNExperts, q);
    int32_t *     d_ids        = sycl::malloc_device<int32_t>(kNIds, q);
    int8_t *      d_b_packed   = sycl::malloc_device<int8_t>(shape.b_packed_elems, q);
    float *       d_y_scales   = sycl::malloc_device<float>(shape.y_scales_elems, q);

    auto free_all = [&]() {
        for (int e = 0; e < kNExperts; ++e) {
            if (d_gate[e]) {
                sycl::free(d_gate[e], q);
            }
            if (d_up[e]) {
                sycl::free(d_up[e], q);
            }
        }
        if (d_activation) {
            sycl::free(d_activation, q);
        }
        if (d_output) {
            sycl::free(d_output, q);
        }
        if (d_gate_ptrs) {
            sycl::free(d_gate_ptrs, q);
        }
        if (d_up_ptrs) {
            sycl::free(d_up_ptrs, q);
        }
        if (d_ids) {
            sycl::free(d_ids, q);
        }
        if (d_b_packed) {
            sycl::free(d_b_packed, q);
        }
        if (d_y_scales) {
            sycl::free(d_y_scales, q);
        }
    };

    bool alloc_ok = d_activation && d_output && d_gate_ptrs && d_up_ptrs && d_ids && d_b_packed && d_y_scales;
    for (int e = 0; e < kNExperts; ++e) {
        alloc_ok = alloc_ok && d_gate[e] && d_up[e];
    }
    if (!alloc_ok) {
        std::fprintf(stderr, "FAIL: device allocation failed\n");
        free_all();
        return 1;
    }

    try {
        for (int e = 0; e < kNExperts; ++e) {
            q.memcpy(d_gate[e], h_gate[e].data(), shape.weight_bytes_per_expert).wait();
            q.memcpy(d_up[e], h_up[e].data(), shape.weight_bytes_per_expert).wait();
            d_gate_ptrs[e] = d_gate[e];
            d_up_ptrs[e]   = d_up[e];
        }
        q.memcpy(d_activation, h_activation.data(), shape.activation_bytes).wait();
        q.memcpy(d_ids, h_ids, sizeof(h_ids)).wait();
        q.memset(d_output, 0, shape.output_floats * sizeof(float)).wait();

        ggml_sycl::mxfp4_pair_glu_bench_args args{};
        args.stream             = &q;
        args.gate_ptrs          = d_gate_ptrs;
        args.up_ptrs            = d_up_ptrs;
        args.activations_q8_soa = d_activation;
        args.output             = d_output;
        args.ids                = d_ids;
        args.dpas_b_packed      = d_b_packed;
        args.dpas_y_scales      = d_y_scales;
        args.ncols              = kNcols;
        args.ncols_y            = kNcols;
        args.nrows_per_expert   = kNrowsPerExpert;
        args.num_experts        = kNExperts;
        args.n_ids              = kNIds;
        args.n_tokens           = kNTokens;
        args.ne11               = kNIds;
        args.ids_nb0            = sizeof(int32_t);
        args.ids_nb1            = static_cast<int64_t>(kNIds) * sizeof(int32_t);
        // ne11 == n_ids, n_tokens == 1: mxfp4_dpas_pack_q8_single_col_groups_sycl
        // indexes q8_src as i11*nb11 + i12*nb12 with i11 = id % ne11 (id is the
        // expert-slot index, 0..n_ids-1) and i12 = iid1 = 0 always, so nb11 must
        // be the stride BETWEEN the n_ids activation rows this test lays out
        // (q8_row_size) and nb12's value is never read (always multiplied by 0).
        args.nb11               = shape.q8_row_size;
        args.nb12               = shape.q8_row_size;
        args.dst_nb1           = static_cast<int64_t>(kNTokens) * static_cast<int64_t>(kNrowsPerExpert) * sizeof(float);
        args.dst_nb2           = static_cast<int64_t>(kNrowsPerExpert) * sizeof(float);
        args.rows_per_wg       = 8;
        args.xmx_tiled         = true;
        args.xmx_tiled_pack_q8 = true;
        args.xmx_tiled_m_tiles = 2;
        args.xmx_tiles_n       = 1;
        args.glu_op            = GGML_GLU_OP_SWIGLU_OAI;
        args.alpha             = kAlpha;
        args.limit             = kLimit;

        if (!ggml_sycl::ggml_sycl_mxfp4_pair_glu_bench_launch(args)) {
            std::fprintf(stderr, "FAIL: m2 gate/up launch rejected test args\n");
            free_all();
            return 1;
        }
        q.wait_and_throw();

        std::vector<float> h_output(shape.output_floats, 0.0f);
        q.memcpy(h_output.data(), d_output, shape.output_floats * sizeof(float)).wait();

        ggml_sycl_kernel_profile_flush(/*wait_for_events=*/true, "test-sycl-mxfp4-gateup-ksplit-numerics");

        free_all();

        std::ofstream fout(out_prefix + ".floats", std::ios::binary | std::ios::trunc);
        if (!fout) {
            std::fprintf(stderr, "FAIL: could not open %s.floats for writing\n", out_prefix.c_str());
            return 1;
        }
        fout.write(reinterpret_cast<const char *>(h_output.data()),
                   static_cast<std::streamsize>(h_output.size() * sizeof(float)));
        if (!fout) {
            std::fprintf(stderr, "FAIL: short write to %s.floats\n", out_prefix.c_str());
            return 1;
        }
        return 0;
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "FAIL: SYCL exception: %s\n", e.what());
        free_all();
        return 1;
    }
}

// Extracts the "metadata" CSV column (3rd field: name,category,metadata,...,
// sycl-kernel-profiler.cpp format_csv_rows) of the row whose "name" column is
// EXACTLY row_name -- not a prefix match, since
// "mxfp4.gateup.xmx_tiled_dpas_m2" is itself a prefix of the sibling
// "mxfp4.gateup.xmx_tiled_dpas_m2_tg1_index" row. Returns false if no such
// row is present (kernel profiling produced no matching row at all).
bool find_csv_metadata(const std::string & csv_text, const std::string & row_name, std::string & metadata_out) {
    size_t line_start = 0;
    while (line_start < csv_text.size()) {
        size_t line_end = csv_text.find('\n', line_start);
        if (line_end == std::string::npos) {
            line_end = csv_text.size();
        }
        const std::string line = csv_text.substr(line_start, line_end - line_start);
        line_start             = line_end + 1;

        const size_t comma1 = line.find(',');
        if (comma1 == std::string::npos) {
            continue;
        }
        if (line.compare(0, comma1, row_name) != 0) {
            continue;
        }
        const size_t comma2 = line.find(',', comma1 + 1);
        if (comma2 == std::string::npos) {
            continue;
        }
        const size_t comma3 = line.find(',', comma2 + 1);
        if (comma3 == std::string::npos) {
            continue;
        }
        metadata_out = line.substr(comma2 + 1, comma3 - comma2 - 1);
        return true;
    }
    return false;
}

std::string read_text_file(const std::string & path, bool & ok) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        ok = false;
        return {};
    }
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    ok = true;
    return content;
}

bool read_float_file(const std::string & path, size_t expected_count, std::vector<float> & out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    out.assign(expected_count, 0.0f);
    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(expected_count * sizeof(float)));
    return static_cast<size_t>(in.gcount()) == expected_count * sizeof(float);
}

struct child_result {
    int                rc = -1;
    std::vector<float> floats;
    std::string        csv;
    bool               floats_ok = false;
    bool               csv_ok    = false;
};

child_result run_one_ksplit(const char * argv0, int ksplit, size_t expected_floats, const std::string & out_prefix) {
    child_result result;

    const pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "FAIL: fork() failed for ksplit=%d (%s)\n", ksplit, std::strerror(errno));
        return result;
    }
    if (pid == 0) {
        // Child: set env BEFORE execv() replaces this process's image, so the
        // fresh image's static initializers (this test's accessor cache and
        // libccl's device-selector memoization -- llama.cpp-2x3m) see it on
        // their first read.
        const std::string ksplit_str = std::to_string(ksplit);
        setenv("GGML_SYCL_MXFP4_GATEUP_KSPLIT", ksplit_str.c_str(), 1);
        setenv("GGML_SYCL_KERNEL_PROFILE", "1", 1);
        setenv("GGML_SYCL_KERNEL_PROFILE_FORMAT", "csv", 1);
        setenv("GGML_SYCL_KERNEL_PROFILE_OUTPUT", (out_prefix + ".csv").c_str(), 1);
        if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
            setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
        }

        std::string         child_flag = "--child";
        std::vector<char *> child_argv = { const_cast<char *>(argv0), const_cast<char *>(child_flag.c_str()),
                                           const_cast<char *>(ksplit_str.c_str()),
                                           const_cast<char *>(out_prefix.c_str()), nullptr };
        execv("/proc/self/exe", child_argv.data());
        std::fprintf(stderr, "FAIL: re-exec failed for ksplit=%d (%s)\n", ksplit, std::strerror(errno));
        _exit(1);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        std::fprintf(stderr, "FAIL: waitpid() failed for ksplit=%d (%s)\n", ksplit, std::strerror(errno));
        return result;
    }
    result.rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (result.rc != 0) {
        return result;
    }

    result.floats_ok = read_float_file(out_prefix + ".floats", expected_floats, result.floats);
    result.csv       = read_text_file(out_prefix + ".csv", result.csv_ok);
    return result;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc >= 4 && std::strcmp(argv[1], "--child") == 0) {
        return run_child(argv[3]);
    }

    const shape_layout shape  = compute_shape();
    const pid_t        my_pid = getpid();

    const int    ksplits[] = { 1, 2, 4 };
    child_result results[3];
    std::string  prefixes[3];

    for (int i = 0; i < 3; ++i) {
        prefixes[i] = "/tmp/test-sycl-mxfp4-gateup-ksplit-numerics-" + std::to_string(static_cast<long>(my_pid)) +
                      "-s" + std::to_string(ksplits[i]);
        results[i] = run_one_ksplit(argv[0], ksplits[i], shape.output_floats, prefixes[i]);
    }

    auto cleanup = [&]() {
        for (int i = 0; i < 3; ++i) {
            std::remove((prefixes[i] + ".floats").c_str());
            std::remove((prefixes[i] + ".csv").c_str());
        }
    };

    // A baseline (S=1) skip means no XMX GPU is present at all -- the whole
    // run proves nothing, matching this fork's "a SKIP with status 0 is not
    // a pass" convention (CLAUDE.md).
    if (results[0].rc == LLAMA_TEST_EXIT_SKIP) {
        std::fprintf(stderr,
                     "SKIP: baseline (ksplit=1) child reported no XMX GPU device; "
                     "this run proves NOTHING about the ksplit kernel.\n");
        cleanup();
        return LLAMA_TEST_EXIT_SKIP;
    }
    for (int i = 0; i < 3; ++i) {
        if (results[i].rc == LLAMA_TEST_EXIT_SKIP) {
            std::fprintf(stderr,
                         "SKIP: ksplit=%d child reported no XMX GPU device after ksplit=1 succeeded; "
                         "inconsistent device availability across forks.\n",
                         ksplits[i]);
            cleanup();
            return LLAMA_TEST_EXIT_SKIP;
        }
        if (results[i].rc != 0) {
            std::fprintf(stderr, "FAIL: ksplit=%d child exited with status %d\n", ksplits[i], results[i].rc);
            cleanup();
            return 1;
        }
        if (!results[i].floats_ok) {
            std::fprintf(stderr, "FAIL: ksplit=%d child did not produce %zu floats at %s.floats\n", ksplits[i],
                         shape.output_floats, prefixes[i].c_str());
            cleanup();
            return 1;
        }
        if (!results[i].csv_ok) {
            std::fprintf(stderr, "FAIL: ksplit=%d child did not produce a profile CSV at %s.csv\n", ksplits[i],
                         prefixes[i].c_str());
            cleanup();
            return 1;
        }
    }

    bool        all_ok = true;
    std::string metadata;

    // S=1 must be byte-for-byte the current behaviour: no combine kernel
    // runs, so the profile row must carry no "ksplit=" tag at all.
    if (!find_csv_metadata(results[0].csv, "mxfp4.gateup.xmx_tiled_dpas_m2", metadata)) {
        std::fprintf(stderr,
                     "FAIL: no 'mxfp4.gateup.xmx_tiled_dpas_m2' profile row for ksplit=1 "
                     "(metadata=<absent>)\n");
        all_ok = false;
    } else if (metadata.find("ksplit=") != std::string::npos) {
        std::fprintf(stderr,
                     "FAIL: ksplit=1 profile row unexpectedly carries a 'ksplit=' tag (metadata=\"%s\") "
                     "-- the S=1 fast path must stay untouched\n",
                     metadata.c_str());
        all_ok = false;
    }

    // S=2 and S=4: the profile row must name the split factor. THIS IS THE
    // RED SIGNAL before GREEN: today GGML_SYCL_MXFP4_GATEUP_KSPLIT is
    // unknown to the kernel, S is silently 1 regardless of the env var, and
    // no row ever carries "ksplit=2" or "ksplit=4".
    for (int i = 1; i < 3; ++i) {
        const std::string tag = "ksplit=" + std::to_string(ksplits[i]);
        if (!find_csv_metadata(results[i].csv, "mxfp4.gateup.xmx_tiled_dpas_m2", metadata)) {
            std::fprintf(stderr, "FAIL: no 'mxfp4.gateup.xmx_tiled_dpas_m2' profile row for ksplit=%d\n", ksplits[i]);
            all_ok = false;
            continue;
        }
        if (metadata.find(tag) == std::string::npos) {
            std::fprintf(stderr, "FAIL: ksplit=%d profile row metadata=\"%s\" does not contain \"%s\"\n", ksplits[i],
                         metadata.c_str(), tag.c_str());
            all_ok = false;
        }
    }

    // Numerics: S=2/S=4 must agree with S=1 per element. DPAS int8
    // accumulation is exact; the float combine across k-parts reorders
    // additions, so this is a tolerance check, not bit-equality.
    for (int i = 1; i < 3; ++i) {
        size_t mismatches   = 0;
        float  max_abs_diff = 0.0f;
        size_t worst_idx    = 0;
        for (size_t j = 0; j < shape.output_floats; ++j) {
            const float ref  = results[0].floats[j];
            const float got  = results[i].floats[j];
            const float diff = std::fabs(ref - got);
            const float tol  = 1e-4f + 1e-3f * std::fabs(ref);
            if (!std::isfinite(got) || diff > tol) {
                ++mismatches;
                if (diff > max_abs_diff) {
                    max_abs_diff = diff;
                    worst_idx    = j;
                }
            }
        }
        if (mismatches > 0) {
            std::fprintf(stderr,
                         "FAIL: ksplit=%d vs ksplit=1 numerics: %zu/%zu elements outside tolerance "
                         "(worst idx=%zu ref=%.6f got=%.6f diff=%.6f)\n",
                         ksplits[i], mismatches, shape.output_floats, worst_idx, results[0].floats[worst_idx],
                         results[i].floats[worst_idx], max_abs_diff);
            all_ok = false;
        }
    }

    // Sanity: the baseline output should not be trivially all-zero (a
    // silently-skipped or no-op launch would still "pass" the tolerance
    // checks above by comparing zeros to zeros).
    float abs_sum = 0.0f;
    for (float v : results[0].floats) {
        abs_sum += std::fabs(v);
    }
    if (!(abs_sum > 1e-6f)) {
        std::fprintf(stderr, "FAIL: ksplit=1 baseline output was all zero -- nothing was verified\n");
        all_ok = false;
    }

    cleanup();

    if (!all_ok) {
        std::fprintf(stderr, "FAILED: test-sycl-mxfp4-gateup-ksplit-numerics\n");
        return 1;
    }
    std::fprintf(stderr,
                 "PASS: test-sycl-mxfp4-gateup-ksplit-numerics (ksplit=1,2,4 agree; ksplit= metadata "
                 "present for 2 and 4)\n");
    return 0;
}

#endif  // GGML_USE_SYCL
