// GPU numerics guard: a flash-attention KV position whose mask is -inf must not
// influence the output, whatever bytes its K and V rows hold.
//
// WHY: llama.cpp keeps freed and stale KV cells inside the n_kv window and hides
// them from attention with a -inf mask. The CPU oracle never reads such a cell
// (ggml-cpu/ops.cpp, flash_attn_ext_f16_one_chunk: `if (mv == -INFINITY)
// continue;`), so a non-finite value left in one stale cell is harmless there.
// A kernel that instead multiplies a masked V row by softmax weight 0, or adds
// the -inf mask to a NaN QK^T score, turns that one cell into NaN for every
// query row that shares the tile. Behind an `isfinite(val) ? val : 0` output
// write this surfaces as an all-zero attention row: finite, invisible to any
// NaN probe, and wrong.
//
// WHAT: every case builds one GGML_OP_FLASH_ATTN_EXT op, runs it on the SYCL
// backend and on the CPU backend in reference mode, and compares. Each case runs
// in three variants:
//   clean  -- the dead cells hold ordinary finite values (positive control: it
//             must pass on any correct kernel, so a failure in the poisoned
//             variants is attributable to the poison, not to the harness);
//   k_poison / v_poison -- NaN, +Inf and -Inf written into whole K (resp. V)
//             rows at cells whose mask is -inf for EVERY query row. Poisoning K
//             and V separately tells a QK^T-side defect from a P*V-side one.
// Dead cells are both interleaved with visible ones and in the tail pad
// (n_kv=256 with only 72 cells in use, the shape of a short prompt).
//
// Multi-row cases add two PARTIAL variants, k_partial / v_partial: NaN in the
// whole K (resp. V) row of one cell that the causal mask hides from the first
// half of the query rows and shows to the second half. The CPU writes NaN rows
// for the second half and finite rows for the first. A kernel must not hide the
// visible NaN (a finite SYCL row where the CPU row is non-finite fails), and
// the finite rows are scored as usual.
//   Documented deviation: the XMX v1 and v2 tile kernels multiply a whole V
// tile on the matrix engine, and "dead" is decided per work-group tile, so a
// cell visible to any row of the tile keeps its V and 0 * NaN reaches the rows
// of that tile that mask it. For those kernels, and for the V variants only
// (v_partial, and v_underflow below), a SYCL-non-finite row where the CPU row
// is finite is reported and allowed.
// This differs from the CPU there, and it is acceptable: the NaN is a genuine
// non-finite visible input, so the batch already reports NaN for the rows that
// see it; zeroing it instead would hide it from those rows as well. K needs no
// such allowance, because the mask is a per-element select that runs before
// the softmax.
//
// Every case except the fully masked set also runs v_underflow: NaN in the V
// row of the same cell, which the rows that see it see through a finite mask
// of -60000 instead of 0. Its softmax weight underflows to exactly 0, but the
// cell is VISIBLE, so the CPU still adds 0 * V and reports NaN for those rows.
// A kernel that skips a cell because its weight is 0, rather than because its
// mask is -inf, hides that NaN (hidden_rows > 0). Only a -inf mask means the
// cell does not exist. The XMX tile allowance above applies here too.
//
// The CPU backend runs with use_ref: its tiled multi-query path adds the mask to
// the score and multiplies masked V by 0 exactly like the kernels under test, so
// it is only an oracle in reference mode.
//
// A case fails on any of: SYCL non-finite values, an all-zero SYCL row where the
// CPU row is non-zero (the sanitizer signature), a row left holding the dst
// sentinel (an unwritten row), or NMSE above test-backend-ops' FLASH_ATTN_EXT
// bound (reported but not gated for the nondeterministic XMX v1 kernel). The
// CPU output must itself be finite, or the case is void.
//
// KERNEL COVERAGE: the kernel is chosen by the dispatcher from the shape and
// the environment. The test sets GGML_SYCL_FA_DISPATCH_DEBUG (unless already
// set), captures the dispatcher's "[SYCL] fattn selected" line for every
// compute, prints it as kernel=<name>, and fails a case that printed no
// dispatch line. It also fails a case whose last dispatch line is a
// "*_rejected" one: a rejected forced path falls back to a tile kernel
// without printing a second line, so the name would claim coverage of a
// kernel that never ran.
//
// ARMS. Each arm is an environment that reaches a different set of kernels.
// GGML_SYCL_FATTN_TEST_EXPECT=<arm> makes the test verify the arm's
// environment, run only the cases the arm covers, and FAIL any case whose
// kernel differs from the arm's table (expected_kernels below), so a routing
// change cannot silently drop coverage. Without it every case runs and the
// kernel is printed, not asserted.
//   default                    no FA overrides (oneDNN PP, ESIMD decode)
//   noonednn                   GGML_SYCL_FA_ONEDNN=0 (native XMX v2 for PP)
//   v1pp                       GGML_SYCL_FA_ONEDNN=0 GGML_SYCL_FA_XMX_V1_PP=1
//   noesimd                    GGML_SYCL_FA_ESIMD=0 (VEC decode, tile_d512); skips
//                              the cases oneDNN still routes
//   force-v2-decode            GGML_SYCL_FA_FORCE_PATH=xmx-v2-decode (m1n64)
//   force-v2-gqa               GGML_SYCL_FA_FORCE_PATH=xmx-v2-gqa
//   force-v2-gqa-split-packed  GGML_SYCL_FA_FORCE_PATH=xmx-v2-gqa-split-packed
//                              (split first pass plus the merge kernel)
//   force-tile                 GGML_SYCL_FA_FORCE_PATH=tile (tile_f16_ncols8)
// The XMX v2 decode kernels are reachable only through their force paths and
// only at D=64 with one query row, which is the d64_sinks_decode shape.
//
// NOT COVERED, by construction: the FP8 K/V branches of XMX v1/v2 and ESIMD.
// This tree has no FP8 ggml type (ggml_sycl_type_is_fp8_e4m3() returns false
// for every type), so no FLASH_ATTN_EXT graph can reach them and the CPU
// backend has no FP8 oracle. They are verified by code reading only. The paged
// V loads belong to the opt-in paged v2 path and are out of scope here.
//
// Exit 77 when no SYCL device is present. GPU test: never run it in a loop.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "sycl-selector-fallback.hpp"
#include "test-skip.h"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_failures = 0;

// Written into dst before the SYCL compute; a row still holding it was never
// written by the kernel.
static constexpr float DST_SENTINEL = 12345.0f;

// test-backend-ops' test_flash_attn_ext::max_nmse_err().
static constexpr double NMSE_MAX = 5e-4;

enum poison_kind {
    POISON_NONE,
    POISON_K,
    POISON_V,
    POISON_K_PARTIAL,
    POISON_V_PARTIAL,
    POISON_V_UNDERFLOW,
};

// The finite mask value the v_underflow variant gives its visible cell: large
// enough that exp(score - max) is exactly 0 in f32, still representable in f16.
static constexpr float UNDERFLOW_MASK = -60000.0f;

static const char * poison_name(poison_kind p) {
    switch (p) {
        case POISON_NONE:
            return "clean";
        case POISON_K:
            return "k_poison";
        case POISON_V:
            return "v_poison";
        case POISON_K_PARTIAL:
            return "k_partial";
        case POISON_V_PARTIAL:
            return "v_partial";
        case POISON_V_UNDERFLOW:
            return "v_underflow";
    }
    return "?";
}

struct fa_case {
    const char * name;
    int          D;
    int          H_q;
    int          H_kv;
    int          ne01;
    int          n_kv;
    int          n_used;            // cells [0, n_used) are live; [n_used, n_kv) are the tail pad
    bool         sinks;
    bool         q_f32;             // D=512 routes require F32 Q; elsewhere F16 matches production
    bool         fully_masked_row;  // row 1 (or the only row) masked everywhere
};

// Interleaved dead cells inside the live window: masked for every query row, the
// way another sequence's cells or a freed cell look to this ubatch.
static bool is_interleaved_dead(int cell, int n_used) {
    return cell < n_used && cell % 12 == 5;
}

// Cells that receive NaN/+Inf/-Inf in the poisoned variants. All are dead for
// every query row: two interleaved dead cells, the seven cells right after the
// live window (a previous request's decode cells), and two deep in the pad.
static std::vector<int> poison_cells(int n_kv, int n_used) {
    std::vector<int> cells;
    for (int c = 0; c < n_used; ++c) {
        if (is_interleaved_dead(c, n_used) && (c == 17 || c == 41)) {
            cells.push_back(c);
        }
    }
    for (int c = n_used; c < std::min(n_kv, n_used + 7); ++c) {
        cells.push_back(c);
    }
    if (n_kv > n_used + 60) {
        cells.push_back(n_used + 58);
    }
    cells.push_back(n_kv - 1);
    return cells;
}

// The cell the partial variants poison: visible to query rows >= ne01 / 2 and
// masked for the rows before them (see build_mask). If a shape ever made it an
// interleaved dead cell, the CPU output would be finite and the case VOID.
static int partial_cell(const fa_case & c) {
    return c.n_used - c.ne01 + c.ne01 / 2;
}

static bool poison_is_partial(poison_kind p) {
    return p == POISON_K_PARTIAL || p == POISON_V_PARTIAL;
}

// Variants whose CPU output has NaN rows by design, scored row by row.
static bool poison_is_rowwise(poison_kind p) {
    return poison_is_partial(p) || p == POISON_V_UNDERFLOW;
}

static float poison_value(size_t i) {
    switch (i % 3) {
        case 0:
            return NAN;
        case 1:
            return INFINITY;
        default:
            return -INFINITY;
    }
}

// Query row r is the r-th of the last ne01 positions of the live window and sees
// every live cell up to and including its own position, except the interleaved
// dead cells. With fully_masked_row, one row sees nothing at all.
static void build_mask(const fa_case & c, std::vector<ggml_fp16_t> & mask) {
    mask.assign((size_t) c.n_kv * c.ne01, ggml_fp32_to_fp16(-INFINITY));
    const int fully_masked = c.fully_masked_row ? (c.ne01 > 1 ? 1 : 0) : -1;
    for (int r = 0; r < c.ne01; ++r) {
        if (r == fully_masked) {
            continue;
        }
        const int last_visible = c.n_used - c.ne01 + r;
        for (int t = 0; t <= last_visible; ++t) {
            if (!is_interleaved_dead(t, c.n_used)) {
                mask[(size_t) r * c.n_kv + t] = ggml_fp32_to_fp16(0.0f);
            }
        }
    }
}

static void fill_f16(std::vector<ggml_fp16_t> & out, size_t n, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    out.resize(n);
    for (ggml_fp16_t & v : out) {
        v = ggml_fp32_to_fp16(dist(rng));
    }
}

static void fill_f32(std::vector<float> & out, size_t n, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    out.resize(n);
    for (float & v : out) {
        v = dist(rng);
    }
}

struct fa_inputs {
    std::vector<float>       q_f32;
    std::vector<ggml_fp16_t> q_f16;
    std::vector<ggml_fp16_t> k;
    std::vector<ggml_fp16_t> v;
    std::vector<ggml_fp16_t> mask;
    std::vector<float>       sinks;
};

static fa_inputs make_inputs(const fa_case & c, poison_kind poison) {
    fa_inputs    in;
    std::mt19937 rng(0xdd82u ^ ((unsigned) c.D * 2654435761u) ^ ((unsigned) c.ne01 * 40503u) ^
                     ((unsigned) c.H_q * 0x9e3779b9u));
    const size_t n_q  = (size_t) c.D * c.ne01 * c.H_q;
    const size_t n_kv = (size_t) c.D * c.n_kv * c.H_kv;
    if (c.q_f32) {
        fill_f32(in.q_f32, n_q, rng, -1.0f, 1.0f);
    } else {
        fill_f16(in.q_f16, n_q, rng, -1.0f, 1.0f);
    }
    fill_f16(in.k, n_kv, rng, -1.0f, 1.0f);
    fill_f16(in.v, n_kv, rng, -1.0f, 1.0f);
    build_mask(c, in.mask);
    if (c.sinks) {
        fill_f32(in.sinks, (size_t) c.H_q, rng, -1.0f, 1.0f);
    }

    if (poison != POISON_NONE) {
        const bool                 is_k    = poison == POISON_K || poison == POISON_K_PARTIAL;
        const bool                 rowwise = poison_is_rowwise(poison);
        std::vector<ggml_fp16_t> & dst     = is_k ? in.k : in.v;
        const std::vector<int> cells = rowwise ? std::vector<int>{ partial_cell(c) } : poison_cells(c.n_kv, c.n_used);
        if (poison == POISON_V_UNDERFLOW) {
            for (int r = 0; r < c.ne01; ++r) {
                ggml_fp16_t & m = in.mask[(size_t) r * c.n_kv + cells[0]];
                if (ggml_fp16_to_fp32(m) == 0.0f) {
                    m = ggml_fp32_to_fp16(UNDERFLOW_MASK);
                }
            }
        }
        for (size_t i = 0; i < cells.size(); ++i) {
            const ggml_fp16_t val = ggml_fp32_to_fp16(rowwise ? NAN : poison_value(i));
            for (int h = 0; h < c.H_kv; ++h) {
                for (int d = 0; d < c.D; ++d) {
                    dst[((size_t) h * c.n_kv + cells[i]) * c.D + d] = val;
                }
            }
        }
    }
    return in;
}

struct fa_graph {
    ggml_context * ctx   = nullptr;
    ggml_tensor *  q     = nullptr;
    ggml_tensor *  k     = nullptr;
    ggml_tensor *  v     = nullptr;
    ggml_tensor *  mask  = nullptr;
    ggml_tensor *  sinks = nullptr;
    ggml_tensor *  out   = nullptr;
    ggml_cgraph *  graph = nullptr;
};

static bool build_graph(const fa_case & c, fa_graph & g) {
    ggml_init_params params = { 16 * 1024 * 1024, nullptr, /*no_alloc=*/true };
    g.ctx                   = ggml_init(params);
    if (!g.ctx) {
        return false;
    }
    g.q    = ggml_new_tensor_4d(g.ctx, c.q_f32 ? GGML_TYPE_F32 : GGML_TYPE_F16, c.D, c.ne01, c.H_q, 1);
    g.k    = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, c.D, c.n_kv, c.H_kv, 1);
    g.v    = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, c.D, c.n_kv, c.H_kv, 1);
    g.mask = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, c.n_kv, c.ne01, 1, 1);
    ggml_set_name(g.q, "masked_nonfinite_q");
    ggml_set_name(g.k, "masked_nonfinite_k");
    ggml_set_name(g.v, "masked_nonfinite_v");
    ggml_set_name(g.mask, "masked_nonfinite_mask");

    const float scale = 1.0f / std::sqrt((float) c.D);
    g.out             = ggml_flash_attn_ext(g.ctx, g.q, g.k, g.v, g.mask, scale, /*max_bias=*/0.0f,
                                            /*logit_softcap=*/0.0f);
    if (c.sinks) {
        g.sinks = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, c.H_q);
        ggml_set_name(g.sinks, "masked_nonfinite_sinks");
        ggml_flash_attn_ext_add_sinks(g.out, g.sinks);
    }
    // llama-graph pins every FA op to F32 precision; the dispatcher picks the
    // accumulator variant from it, so match production.
    ggml_flash_attn_ext_set_prec(g.out, GGML_PREC_F32);
    ggml_set_name(g.out, "masked_nonfinite_out");
    g.graph = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.graph, g.out);
    return true;
}

static void upload_inputs(const fa_case & c, const fa_graph & g, const fa_inputs & in) {
    if (c.q_f32) {
        ggml_backend_tensor_set(g.q, in.q_f32.data(), 0, in.q_f32.size() * sizeof(float));
    } else {
        ggml_backend_tensor_set(g.q, in.q_f16.data(), 0, in.q_f16.size() * sizeof(ggml_fp16_t));
    }
    ggml_backend_tensor_set(g.k, in.k.data(), 0, in.k.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(g.v, in.v.data(), 0, in.v.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(g.mask, in.mask.data(), 0, in.mask.size() * sizeof(ggml_fp16_t));
    if (g.sinks) {
        ggml_backend_tensor_set(g.sinks, in.sinks.data(), 0, in.sinks.size() * sizeof(float));
    }
}

static bool run_cpu(ggml_backend_t cpu, const fa_case & c, const fa_inputs & in, std::vector<float> & out) {
    fa_graph g;
    if (!build_graph(c, g)) {
        return false;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(g.ctx, cpu);
    if (!buf) {
        ggml_free(g.ctx);
        return false;
    }
    upload_inputs(c, g, in);
    const bool ok = ggml_backend_graph_compute(cpu, g.graph) == GGML_STATUS_SUCCESS;
    if (ok) {
        out.resize((size_t) ggml_nelements(g.out));
        ggml_backend_tensor_get(g.out, out.data(), 0, out.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(g.ctx);
    return ok;
}

// Redirects fd 2 into a temporary file for the duration of one SYCL compute so
// the dispatcher's "[SYCL] fattn selected" lines can be read back. Everything
// captured is replayed to the real stderr afterwards.
struct stderr_capture {
    FILE * tmp   = nullptr;
    int    saved = -1;

    bool begin() {
        std::fflush(stderr);
        tmp = std::tmpfile();
        if (!tmp) {
            return false;
        }
        saved = dup(STDERR_FILENO);
        if (saved < 0 || dup2(fileno(tmp), STDERR_FILENO) < 0) {
            std::fclose(tmp);
            tmp = nullptr;
            return false;
        }
        return true;
    }

    std::string end() {
        std::string text;
        if (!tmp) {
            return text;
        }
        std::fflush(stderr);
        dup2(saved, STDERR_FILENO);
        close(saved);
        std::rewind(tmp);
        char   chunk[4096];
        size_t n = 0;
        while ((n = std::fread(chunk, 1, sizeof(chunk), tmp)) > 0) {
            text.append(chunk, n);
        }
        std::fclose(tmp);
        tmp = nullptr;
        std::fputs(text.c_str(), stderr);
        return text;
    }
};

// The last kernel the dispatcher reported: "[SYCL] fattn selected [N/L] <name> ..."
// or "[SYCL] fattn selected [d512] <name> ...". The small-multiquery D=128 guard
// reports itself and then the tile kernel it hands over to, so the last line wins.
static std::string last_selected_kernel(const std::string & text) {
    static const char * tag = "fattn selected [";
    std::string         kernel;
    size_t              pos = 0;
    while ((pos = text.find(tag, pos)) != std::string::npos) {
        const size_t close = text.find("] ", pos);
        if (close == std::string::npos) {
            break;
        }
        const size_t start = close + 2;
        const size_t stop  = text.find_first_of(" \n", start);
        kernel             = text.substr(start, stop == std::string::npos ? std::string::npos : stop - start);
        pos                = start;
    }
    return kernel;
}

static ggml_backend_buffer_t alloc_tensor_buffer(ggml_backend_buffer_type_t buft, ggml_tensor * tensor) {
    const size_t          size   = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

static bool run_sycl(ggml_backend_t       sycl,
                     const fa_case &      c,
                     const fa_inputs &    in,
                     std::vector<float> & out,
                     std::string &        kernel) {
    fa_graph g;
    if (!build_graph(c, g)) {
        return false;
    }
    ggml_backend_buffer_type_t         buft    = ggml_backend_get_default_buffer_type(sycl);
    ggml_tensor *                      owned[] = { g.q, g.k, g.v, g.mask, g.sinks, g.out };
    std::vector<ggml_backend_buffer_t> bufs;
    bool                               ok = true;
    for (ggml_tensor * t : owned) {
        if (!t) {
            continue;
        }
        ggml_backend_buffer_t b = alloc_tensor_buffer(buft, t);
        if (!b) {
            ok = false;
            break;
        }
        bufs.push_back(b);
    }

    if (ok) {
        upload_inputs(c, g, in);
        std::vector<float> sentinel((size_t) ggml_nelements(g.out), DST_SENTINEL);
        ggml_backend_tensor_set(g.out, sentinel.data(), 0, sentinel.size() * sizeof(float));

        stderr_capture cap;
        const bool     capturing = cap.begin();
        ok                       = ggml_backend_graph_compute(sycl, g.graph) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(sycl);
        if (capturing) {
            kernel = last_selected_kernel(cap.end());
        }
        if (ok) {
            out.resize((size_t) ggml_nelements(g.out));
            ggml_backend_tensor_get(g.out, out.data(), 0, out.size() * sizeof(float));
        }
    }
    for (ggml_backend_buffer_t b : bufs) {
        ggml_backend_buffer_free(b);
    }
    ggml_free(g.ctx);
    return ok;
}

// An arm's required environment; nullptr means the variable must be unset.
struct expect_arm {
    const char * name;
    const char * fa_onednn;
    const char * fa_xmx_v1_pp;
    const char * fa_esimd;
    const char * fa_force_path;
    bool         inherit_default;  // cases absent from the arm's rows use the default arm's
};

static const expect_arm expect_arms[] = {
    { "default",                   nullptr, nullptr, nullptr, nullptr,                   false },
    { "noonednn",                  "0",     nullptr, nullptr, nullptr,                   true  },
    { "v1pp",                      "0",     "1",     nullptr, nullptr,                   true  },
    { "noesimd",                   nullptr, nullptr, "0",     nullptr,                   false },
    { "force-v2-decode",           nullptr, nullptr, nullptr, "xmx-v2-decode",           false },
    { "force-v2-gqa",              nullptr, nullptr, nullptr, "xmx-v2-gqa",              false },
    { "force-v2-gqa-split-packed", nullptr, nullptr, nullptr, "xmx-v2-gqa-split-packed", false },
    { "force-tile",                nullptr, nullptr, nullptr, "tile",                    false },
};

struct expect_kernel {
    const char * arm;
    const char * case_name;
    const char * kernel;
};

// The kernel each arm must reach per case, as observed on the B50 and B70. A
// case missing from an arm (after inheritance) is not run under that arm.
static const expect_kernel expected_kernels[] = {
    { "default",                   "d128_decode",                 "esimd_f16"                            },
    { "default",                   "d128_pp72",                   "onednn"                               },
    { "default",                   "d128_mq4",                    "tile_f16_ncols4"                      },
    { "default",                   "d64_sinks_decode",            "esimd_f16"                            },
    { "default",                   "d64_sinks_mq4",               "esimd_f16_batched"                    },
    { "default",                   "d64_sinks_pp72",              "xmx_v2_f16_pp_ncols32"                },
    { "default",                   "d256_pp16",                   "onednn"                               },
    { "default",                   "d512_decode",                 "esimd_partitioned"                    },
    { "default",                   "d512_mq4",                    "tile_d512"                            },
    { "default",                   "d128_decode_all_masked",      "esimd_f16"                            },
    { "default",                   "d64_sinks_decode_all_masked", "esimd_f16"                            },
    { "default",                   "d512_decode_all_masked",      "esimd_partitioned"                    },
    { "default",                   "d128_pp72_row1_masked",       "onednn"                               },
    { "default",                   "d128_mq4_row1_masked",        "tile_f16_ncols4"                      },
    { "default",                   "d64_sinks_mq4_row1_masked",   "esimd_f16_batched"                    },
    { "default",                   "d64_sinks_pp72_row1_masked",  "xmx_v2_f16_pp_ncols32"                },

    { "noonednn",                  "d128_pp72",                   "xmx_v2_f16_ncols16_large"             },
    { "noonednn",                  "d128_pp72_row1_masked",       "xmx_v2_f16_ncols16_large"             },
    { "noonednn",                  "d256_pp16",                   "xmx_v2_f16_ncols16"                   },

    { "v1pp",                      "d128_pp72",                   "xmx_v1_f16_ncols8_large_kv"           },
    { "v1pp",                      "d128_pp72_row1_masked",       "xmx_v1_f16_ncols8_large_kv"           },
    { "v1pp",                      "d256_pp16",                   "xmx_v2_f16_ncols16"                   },

    { "noesimd",                   "d128_decode",                 "vec_f16"                              },
    { "noesimd",                   "d64_sinks_decode",            "vec_f16"                              },
    { "noesimd",                   "d128_decode_all_masked",      "vec_f16"                              },
    { "noesimd",                   "d64_sinks_decode_all_masked", "vec_f16"                              },
    { "noesimd",                   "d512_decode",                 "tile_d512"                            },
    { "noesimd",                   "d512_decode_all_masked",      "tile_d512"                            },
    { "noesimd",                   "d64_sinks_mq4",               "xmx_v2_f16_ncols8"                    },
    { "noesimd",                   "d64_sinks_mq4_row1_masked",   "xmx_v2_f16_ncols8"                    },
    // oneDNN still routes the D=128 and D=256 prefill cases, which stay red
    // until llama.cpp-t0f4; noonednn and v1pp cover their native kernels.
    { "noesimd",                   "d128_mq4",                    "tile_f16_ncols4"                      },
    { "noesimd",                   "d128_mq4_row1_masked",        "tile_f16_ncols4"                      },
    { "noesimd",                   "d64_sinks_pp72",              "xmx_v2_f16_pp_ncols32"                },
    { "noesimd",                   "d64_sinks_pp72_row1_masked",  "xmx_v2_f16_pp_ncols32"                },
    { "noesimd",                   "d512_mq4",                    "tile_d512"                            },

    { "force-v2-decode",           "d64_sinks_decode",            "force_xmx_v2_decode_m1n64"            },
    { "force-v2-decode",           "d64_sinks_decode_all_masked", "force_xmx_v2_decode_m1n64"            },

    { "force-v2-gqa",              "d64_sinks_decode",            "force_xmx_v2_decode_gqa"              },
    { "force-v2-gqa",              "d64_sinks_decode_all_masked", "force_xmx_v2_decode_gqa"              },

    { "force-v2-gqa-split-packed", "d64_sinks_decode",            "force_xmx_v2_decode_gqa_split_packed" },
    { "force-v2-gqa-split-packed", "d64_sinks_decode_all_masked", "force_xmx_v2_decode_gqa_split_packed" },

    // The forced tile path does not apply to the separate D=512 dispatcher.
    { "force-tile",                "d128_decode",                 "force_tile_f16"                       },
    { "force-tile",                "d128_pp72",                   "force_tile_f16"                       },
    { "force-tile",                "d128_mq4",                    "force_tile_f16"                       },
    { "force-tile",                "d64_sinks_decode",            "force_tile_f16"                       },
    { "force-tile",                "d64_sinks_mq4",               "force_tile_f16"                       },
    { "force-tile",                "d64_sinks_pp72",              "force_tile_f16"                       },
    { "force-tile",                "d256_pp16",                   "force_tile_f16"                       },
    { "force-tile",                "d128_decode_all_masked",      "force_tile_f16"                       },
    { "force-tile",                "d64_sinks_decode_all_masked", "force_tile_f16"                       },
    { "force-tile",                "d128_pp72_row1_masked",       "force_tile_f16"                       },
    { "force-tile",                "d128_mq4_row1_masked",        "force_tile_f16"                       },
    { "force-tile",                "d64_sinks_mq4_row1_masked",   "force_tile_f16"                       },
    { "force-tile",                "d64_sinks_pp72_row1_masked",  "force_tile_f16"                       },
};

static const char * lookup_expected(const char * arm, const char * case_name) {
    for (const expect_kernel & e : expected_kernels) {
        if (std::strcmp(e.arm, arm) == 0 && std::strcmp(e.case_name, case_name) == 0) {
            return e.kernel;
        }
    }
    return nullptr;
}

// The selected arm (nullptr: routing is printed but not asserted).
static const expect_arm * g_arm = nullptr;

// The kernel case_name must reach under the selected arm, or nullptr when the
// arm does not cover it. Only meaningful when g_arm is set.
static const char * expected_kernel_for(const char * case_name) {
    const char * k = lookup_expected(g_arm->name, case_name);
    if (!k && g_arm->inherit_default) {
        k = lookup_expected("default", case_name);
    }
    return k;
}

static bool env_matches(const char * name, const char * required) {
    const char * val = std::getenv(name);
    if (!required) {
        return val == nullptr;
    }
    return val != nullptr && std::strcmp(val, required) == 0;
}

// Selects the arm named by GGML_SYCL_FATTN_TEST_EXPECT and checks that the
// environment is the one the arm's table was observed under.
static bool select_arm() {
    const char * name = std::getenv("GGML_SYCL_FATTN_TEST_EXPECT");
    if (!name) {
        std::printf("expect: GGML_SYCL_FATTN_TEST_EXPECT unset, kernel routing is printed but not asserted\n");
        return true;
    }
    for (const expect_arm & a : expect_arms) {
        if (std::strcmp(a.name, name) != 0) {
            continue;
        }

        struct {
            const char * var;
            const char * required;
        } const checks[] = {
            { "GGML_SYCL_FA_ONEDNN",     a.fa_onednn     },
            { "GGML_SYCL_FA_XMX_V1_PP",  a.fa_xmx_v1_pp  },
            { "GGML_SYCL_FA_ESIMD",      a.fa_esimd      },
            { "GGML_SYCL_FA_FORCE_PATH", a.fa_force_path },
        };

        bool ok = true;
        for (const auto & chk : checks) {
            if (!env_matches(chk.var, chk.required)) {
                std::printf("FAIL: arm %s requires %s=%s\n", a.name, chk.var, chk.required ? chk.required : "(unset)");
                ok = false;
            }
        }
        g_arm = &a;
        std::printf("expect: arm %s\n", a.name);
        return ok;
    }
    std::printf("FAIL: unknown GGML_SYCL_FATTN_TEST_EXPECT=%s\n", name);
    return false;
}

static bool ends_with(const std::string & s, const char * suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

static void run_case(ggml_backend_t  sycl,
                     ggml_backend_t  cpu,
                     const fa_case & c,
                     poison_kind     poison,
                     const char *    expected) {
    char label[160];
    std::snprintf(label, sizeof(label), "%s %s D=%d H_q=%d H_kv=%d ne01=%d n_kv=%d used=%d sinks=%d", c.name,
                  poison_name(poison), c.D, c.H_q, c.H_kv, c.ne01, c.n_kv, c.n_used, (int) c.sinks);

    const fa_inputs    in = make_inputs(c, poison);
    std::vector<float> ref;
    std::vector<float> got;
    std::string        kernel;

    if (!run_cpu(cpu, c, in, ref)) {
        std::printf("FAIL [%s]: CPU reference compute failed\n", label);
        ++g_failures;
        return;
    }
    if (!run_sycl(sycl, c, in, got, kernel)) {
        std::printf("FAIL [%s] kernel=%s: SYCL compute failed\n", label, kernel.empty() ? "?" : kernel.c_str());
        ++g_failures;
        return;
    }
    if (got.size() != ref.size()) {
        std::printf("FAIL [%s]: output size mismatch (sycl=%zu cpu=%zu)\n", label, got.size(), ref.size());
        ++g_failures;
        return;
    }

    // dst is [D, H_q, ne01]: one row per (query, head).
    const bool   rowwise       = poison_is_rowwise(poison);
    const size_t n_rows        = ref.size() / (size_t) c.D;
    size_t       cpu_nonfinite = 0;
    size_t       gpu_nonfinite = 0;
    size_t       cpu_nf_rows   = 0;  // rows the CPU reports non-finite
    size_t       hidden_rows   = 0;  // SYCL finite where the CPU row is not: a visible NaN hidden
    size_t       spill_rows    = 0;  // SYCL non-finite where the CPU row is finite
    size_t       zeroed_rows   = 0;  // SYCL all-zero where CPU is not
    size_t       stray_rows    = 0;  // SYCL non-zero where CPU is all-zero (fully masked row)
    size_t       sentinel_rows = 0;
    double       mse_diff      = 0.0;
    double       mse_ref       = 0.0;
    double       max_diff      = 0.0;
    double       cmp_diff      = 0.0;  // row-wise variants: over rows finite on both sides
    double       cmp_ref       = 0.0;
    double       cmp_max_diff  = 0.0;
    for (size_t r = 0; r < n_rows; ++r) {
        bool   ref_zero     = true;
        bool   got_zero     = true;
        bool   sentinel     = false;
        bool   ref_finite   = true;
        bool   got_finite   = true;
        double row_diff     = 0.0;
        double row_ref      = 0.0;
        double row_max_diff = 0.0;
        for (int d = 0; d < c.D; ++d) {
            const float a = ref[r * c.D + d];
            const float b = got[r * c.D + d];
            cpu_nonfinite += std::isfinite(a) ? 0 : 1;
            gpu_nonfinite += std::isfinite(b) ? 0 : 1;
            ref_finite = ref_finite && std::isfinite(a);
            got_finite = got_finite && std::isfinite(b);
            ref_zero   = ref_zero && a == 0.0f;
            got_zero   = got_zero && b == 0.0f;
            sentinel   = sentinel || b == DST_SENTINEL;
            // The reference norm must not depend on the SYCL values: summing
            // it only where both sides are finite made a fully non-finite
            // SYCL result look like an all-zero oracle.
            if (std::isfinite(a)) {
                mse_ref += (double) a * (double) a;
                row_ref += (double) a * (double) a;
            }
            if (std::isfinite(a) && std::isfinite(b)) {
                const double diff = (double) b - (double) a;
                mse_diff += diff * diff;
                row_diff += diff * diff;
                max_diff     = std::max(max_diff, std::fabs(diff));
                row_max_diff = std::max(row_max_diff, std::fabs(diff));
            }
        }
        cpu_nf_rows += ref_finite ? 0 : 1;
        hidden_rows += (!ref_finite && got_finite) ? 1 : 0;
        spill_rows += (ref_finite && !got_finite) ? 1 : 0;
        zeroed_rows += (got_zero && !ref_zero) ? 1 : 0;
        stray_rows += (ref_zero && !got_zero && !sentinel) ? 1 : 0;
        sentinel_rows += sentinel ? 1 : 0;
        if (ref_finite && got_finite) {
            cmp_diff += row_diff;
            cmp_ref += row_ref;
            cmp_max_diff = std::max(cmp_max_diff, row_max_diff);
        }
    }
    // A non-finite SYCL value has no finite distance to the reference, so it
    // must never be summarised as a small error. The row-wise variants expect
    // non-finite rows, check them row by row above, and measure the error over
    // the rows both sides report finite.
    double nmse = 0.0;
    if (rowwise) {
        max_diff = cmp_max_diff;
        if (cmp_ref > 0.0) {
            nmse = cmp_diff / cmp_ref;
        } else if (cmp_diff > 0.0) {
            nmse = INFINITY;
        }
    } else if (gpu_nonfinite > 0) {
        nmse     = INFINITY;
        max_diff = INFINITY;
    } else if (mse_ref > 0.0) {
        nmse = mse_diff / mse_ref;
    } else if (mse_diff > 0.0) {
        nmse = INFINITY;
    }

    // XMX v1 is nondeterministic: bit-identical inputs give a different dst on
    // every call, and its clean control lands at the NMSE bound on its own. Its
    // error magnitude therefore cannot tell poison from noise, so for v1 only
    // the structural signatures of this defect gate (non-finite values, zeroed
    // rows, unwritten rows); the NMSE is still printed.
    const bool nmse_gated = kernel.rfind("xmx_v1", 0) != 0;

    // The documented tile-kernel deviation (see the file header): a V cell
    // visible to some rows of an XMX work-group tile reaches the rows of that
    // tile that mask it.
    const bool xmx_tile = kernel.rfind("xmx_v1", 0) == 0 || kernel.rfind("xmx_v2_f16_ncols", 0) == 0 ||
                          kernel.rfind("xmx_v2_f16_pp_ncols", 0) == 0;
    const bool spill_allowed = (poison == POISON_V_PARTIAL || poison == POISON_V_UNDERFLOW) && xmx_tile;

    // With no visible cell at all the CPU output is legitimately all zero. A
    // row-wise variant needs NaN rows (the cell is visible somewhere), and with
    // several query rows also finite rows (it is masked somewhere), or it tests
    // nothing.
    const bool expect_all_zero = c.n_used == 0;
    const bool rowwise_oracle =
        cpu_nf_rows > 0 && (cpu_nf_rows == n_rows || mse_ref > 0.0) && (c.ne01 == 1 || cpu_nf_rows < n_rows);
    const bool oracle_ok =
        rowwise ? rowwise_oracle : (cpu_nonfinite == 0 && (expect_all_zero ? mse_ref == 0.0 : mse_ref > 0.0));
    const bool finite_ok  = rowwise ? (hidden_rows == 0 && (spill_rows == 0 || spill_allowed)) : gpu_nonfinite == 0;
    // A rejected forced path runs a fallback without naming it.
    const bool rejected   = ends_with(kernel, "_rejected");
    const bool routing_ok = !kernel.empty() && !rejected && (!expected || kernel == expected);
    const bool ok = oracle_ok && routing_ok && finite_ok && zeroed_rows == 0 && stray_rows == 0 && sentinel_rows == 0 &&
                    (!nmse_gated || nmse <= NMSE_MAX);
    // A pass that relies on the documented XMX tile deviation says so.
    const char * status = !ok ? "FAIL" : (spill_rows > 0 ? "OK-DEVIATION" : "OK");

    std::printf(
        "%s [%s] kernel=%s nmse=%.3e (max %.1e%s) max_diff=%.3e sycl_nonfinite=%zu zeroed_rows=%zu/%zu "
        "stray_rows=%zu sentinel_rows=%zu cpu_nonfinite=%zu\n",
        status, label, kernel.empty() ? "<none: no dispatch line captured>" : kernel.c_str(), nmse, NMSE_MAX,
        nmse_gated ? "" : ", not gated", max_diff, gpu_nonfinite, zeroed_rows, n_rows, stray_rows, sentinel_rows,
        cpu_nonfinite);
    if (rejected) {
        std::printf(
            "  REJECTED [%s]: the dispatcher refused %s and fell back without naming the fallback, so this run "
            "proves nothing about %s\n",
            label, kernel.c_str(), kernel.c_str());
    }
    if (expected && !kernel.empty() && kernel != expected) {
        std::printf("  ROUTING [%s]: arm %s expects kernel=%s\n", label, g_arm->name, expected);
    }
    if (rowwise) {
        std::printf("  rows [%s]: cpu_nonfinite_rows=%zu hidden_rows=%zu spill_rows=%zu%s\n", label, cpu_nf_rows,
                    hidden_rows, spill_rows,
                    spill_rows == 0 ? "" : (spill_allowed ? " (allowed: XMX tile deviation)" : " (not allowed)"));
    }
    if (!oracle_ok) {
        std::printf(
            "  VOID [%s]: the CPU reference is non-finite or has the wrong all-zero state, so this case proves "
            "nothing\n",
            label);
    }
    if (!ok) {
        ++g_failures;
    }
}

int main(int, char ** argv) {
    // The dispatcher writes to stderr while results go to stdout; when both
    // land in one file a block-buffered stdout is flushed mid-line and the
    // result lines are torn. Line buffering makes each result one write.
    setvbuf(stdout, nullptr, _IOLBF, 1 << 16);

    sycl_test_selector_fallback(argv, "level_zero:1");

    // The kernel line is the proof of which path ran; without it a pass is vacuous.
    if (!std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG")) {
        setenv("GGML_SYCL_FA_DISPATCH_DEBUG", "1", 1);
    }
    if (!std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG_LIMIT")) {
        setenv("GGML_SYCL_FA_DISPATCH_DEBUG_LIMIT", "1000000", 1);
    }

    if (ggml_backend_sycl_get_device_count() <= 0) {
        std::printf("SKIP: no SYCL GPU device available\n");
        return LLAMA_TEST_EXIT_SKIP;
    }
    ggml_backend_t sycl = ggml_backend_sycl_init(0);
    if (!sycl) {
        std::printf("SKIP: no SYCL GPU device available\n");
        return LLAMA_TEST_EXIT_SKIP;
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::printf("FAIL: CPU backend init failed\n");
        ggml_backend_free(sycl);
        return 1;
    }
    ggml_backend_cpu_set_use_ref(cpu, true);

    static const char * env_names[] = { "ONEAPI_DEVICE_SELECTOR", "GGML_SYCL_FA_ONEDNN", "GGML_SYCL_FA_XMX_V1_PP",
                                        "GGML_SYCL_FA_ESIMD", "GGML_SYCL_FA_FORCE_PATH" };
    for (const char * name : env_names) {
        const char * val = std::getenv(name);
        std::printf("env %s=%s\n", name, val ? val : "(unset)");
    }
    if (!select_arm()) {
        ggml_backend_free(cpu);
        ggml_backend_free(sycl);
        return 1;
    }
    int n_run = 0;

    // name, D, H_q, H_kv, ne01, n_kv, n_used, sinks, q_f32, fully_masked_row
    const fa_case cases[] = {
        // Decode on a Mistral-shaped layer (ESIMD by default, VEC under GGML_SYCL_FA_ESIMD=0).
        { "d128_decode",      128, 32, 8, 1,  256, 72, false, false, false },
        // Prefill of a 72-token prompt into a 256-cell window (oneDNN by default,
        // XMX v2 under GGML_SYCL_FA_ONEDNN=0, XMX v1 with GGML_SYCL_FA_XMX_V1_PP=1).
        { "d128_pp72",        128, 32, 8, 72, 256, 72, false, false, false },
        // Small multi-query batch: the D=128 guard sends it to the tile kernel.
        { "d128_mq4",         128, 32, 8, 4,  256, 72, false, false, false },
        // GPT-OSS-shaped layers with attention sinks.
        { "d64_sinks_decode", 64,  64, 8, 1,  256, 72, true,  false, false },
        { "d64_sinks_mq4",    64,  64, 8, 4,  256, 72, true,  false, false },
        { "d64_sinks_pp72",   64,  64, 8, 72, 256, 72, true,  false, false },
        { "d256_pp16",        256, 16, 4, 16, 256, 72, false, false, false },
        // gemma4-shaped global layers (ESIMD decode by default, tile_d512 otherwise).
        { "d512_decode",      512, 8,  2, 1,  256, 72, false, true,  false },
        { "d512_mq4",         512, 8,  2, 4,  256, 72, false, true,  false },
    };
    const poison_kind variants[] = { POISON_NONE,      POISON_K,         POISON_V,
                                     POISON_K_PARTIAL, POISON_V_PARTIAL, POISON_V_UNDERFLOW };

    for (const fa_case & c : cases) {
        const char * expected = g_arm ? expected_kernel_for(c.name) : nullptr;
        if (g_arm && !expected) {
            std::printf("== %s == (not covered by arm %s)\n", c.name, g_arm->name);
            continue;
        }
        std::printf("== %s ==\n", c.name);
        ++n_run;
        for (poison_kind p : variants) {
            // A single query row sees every cell or none: no partial mask.
            if (poison_is_partial(p) && c.ne01 == 1) {
                continue;
            }
            run_case(sycl, cpu, c, p, expected);
        }
    }

    // A query row with every cell masked: the CPU writes 0 (S == 0 guard), or
    // 0 / S when a sink supplies the whole denominator, and so must every
    // kernel. Kept apart from the poisoned cases so a failure here is not
    // mistaken for a poison failure. Clean K/V only.
    std::printf("== fully masked rows ==\n");
    const fa_case fully_masked[] = {
        { "d128_decode_all_masked",      128, 32, 8, 1,  256, 0,  false, false, false },
        // A sink supplies the whole denominator: 0 / S.
        { "d64_sinks_decode_all_masked", 64,  64, 8, 1,  256, 0,  true,  false, false },
        { "d512_decode_all_masked",      512, 8,  2, 1,  256, 0,  false, true,  false },
        { "d128_pp72_row1_masked",       128, 32, 8, 72, 256, 72, false, false, true  },
        { "d128_mq4_row1_masked",        128, 32, 8, 4,  256, 72, false, false, true  },
        { "d64_sinks_mq4_row1_masked",   64,  64, 8, 4,  256, 72, true,  false, true  },
        { "d64_sinks_pp72_row1_masked",  64,  64, 8, 72, 256, 72, true,  false, true  },
    };
    for (const fa_case & c : fully_masked) {
        const char * expected = g_arm ? expected_kernel_for(c.name) : nullptr;
        if (g_arm && !expected) {
            std::printf("-- %s (not covered by arm %s)\n", c.name, g_arm->name);
            continue;
        }
        ++n_run;
        run_case(sycl, cpu, c, POISON_NONE, expected);
    }

    ggml_backend_free(cpu);
    ggml_backend_free(sycl);

    if (n_run == 0) {
        std::printf("FAIL: no case ran, so this run proves nothing\n");
        return 1;
    }
    if (g_failures) {
        std::printf("FAILED: %d case(s)\n", g_failures);
        return 1;
    }
    std::printf("PASS: masked non-finite K/V cells do not reach the flash-attention output\n");
    return 0;
}
