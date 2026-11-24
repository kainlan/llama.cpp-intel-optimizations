// Probe: can this device's oneDNN run a matmul whose weights stay in NVFP4's stored bits, f4_e2m1 values with one
// f8_e4m3 scale per 16 elements along K? The answer decides whether the batched MoE prompt executor can hand NVFP4
// experts to oneDNN without expanding them to f16 (its weight-only-quantised arm), and in which form.
//
// The installed oneDNN headers document the form: set_scales takes "block sizes for the last two tensor dimensions"
// (dnnl.hpp, primitive_attr::set_scales) and the C API says the group count "must be 0 or 2"
// (dnnl.h, dnnl_primitive_attr_set_scales_v3). The headers declare the types; only a primitive_desc on the device
// says whether an implementation takes them, so each arm reports what it found instead of aborting.
//
// Arms, each printing one line "PROBE <name> <PASS|FAIL|UNSUPPORTED> <detail>" on stdout:
//   nvfp4-2d-create       primitive_desc for f16 src [M,K] x f4_e2m1 weights [K,N], f8_e4m3 scales, mask 3,
//                         groups {16,1}, f32 dst.
//   nvfp4-2d-numerics     that primitive run on NVFP4 blocks repacked on the host, against a reference that decodes
//                         the stored blocks the way the SYCL backend does.
//   nvfp4-3d              [B,K,N] weights, mask 7, groups {16,1}: B experts in one execute.
//   mxfp4-3d-control      MXFP4 (f4_e2m1, e8m0, groups {32,1}) in the same 3-D form, as the control.
//   mxfp4-3d-prod-groups  the 3-D MXFP4 form with the group dims the backend's opt-in 3-D arm passes
//                         (woq_gemm_batch_mxfp4 in ggml-sycl/gemm.hpp under GGML_SYCL_MOE_PP_WOQ_3D: {1,32,1}, three
//                         dims), whose comment records wrong numbers. Diagnostic: it shows whether the group dims are
//                         the cause.
//
// Exit status: 0 once every arm has run, whatever each found; 77 without a SYCL GPU; 1 only when the probe itself
// breaks outside an arm. Device buffers come from sycl::malloc_device, as in test-onednn-woq-mxfp4.cpp: the probe
// calls oneDNN directly and links no backend, so there is no unified cache to allocate from.
#include "test-skip.h"
// The attribute production builds with, so the answer is about the primitive the backend would run.
#include "ggml-sycl/dnnl-deterministic.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <random>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <vector>

using dt = dnnl::memory::data_type;

namespace {

constexpr int k_nvfp4_block       = 64;  // QK_NVFP4
constexpr int k_nvfp4_sub         = 16;  // QK_NVFP4_SUB: one UE4M3 scale per 16 elements
constexpr int k_nvfp4_block_bytes = 36;  // sizeof(block_nvfp4): d[4], then qs[32]
constexpr int k_mxfp4_group       = 32;  // QK_MXFP4

// The per-column bound. With f16 accumulation over K = 2304 the error of a column is near sqrt(K) * 2^-11 of a typical
// term while its largest output is near sqrt(K) terms, so a correct arm reads around 1e-3 at most; a misread scale
// byte or nibble order puts the whole column off, which reads near 1 or above.
constexpr double k_column_tol = 1e-2;

// ggml-common.h kvalues_mxfp4: the E2M1 values doubled. NVFP4 and MXFP4 share it.
constexpr int8_t k_kvalues_doubled[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

// A copy of the SYCL backend's UE4M3 decode (ggml-sycl/common.hpp, ggml_sycl_ue4m3_to_fp32), so the reference is the
// decode every SYCL NVFP4 path shares: half the E4M3 value (kvalues are doubled), the sign bit ignored, and 0x7F and
// 0xFF read as 0. The CPU decode (ggml-impl.h, ggml_ue4m3_to_fp32) reads only 0x7F as 0 and decodes 0xFF as 240; no
// quantizer writes a sign bit, so the two agree on every block ggml produces.
float ue4m3_half_sycl(uint8_t x) {
    if ((x & 0x7f) == 0x7f) {
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xf;
    const int man = x & 0x7;
    if (exp == 0) {
        return std::ldexp((float) man, -10);
    }
    return std::ldexp(1.0f + (float) man / 8.0f, exp - 8);
}

// The scale byte oneDNN must see so that f8_e4m3 times true E2M1 equals the SYCL decode: f8_e4m3 honours the sign bit
// and reads 0x7F and 0xFF as NaN, so the sign is cleared and both NaN patterns become 0.
uint8_t ue4m3_sanitize(uint8_t x) {
    const uint8_t mag = x & 0x7f;
    return mag == 0x7f ? 0 : mag;
}

float e8m0_to_f32(uint8_t e) {
    return std::ldexp(1.0f, (int) e - 127);
}

float round_to_f16(float x) {
    return (float) sycl::half(x);
}

uint16_t f16_bits(float x) {
    return sycl::bit_cast<uint16_t>(sycl::half(x));
}

// Each output column draws every scale of its weights from one class, so a misread kind of scale byte shows as a
// whole column off instead of being diluted by the others. The last class mixes every kind within one column.
enum scale_class : int {
    SCALE_CLASS_MODERATE,
    SCALE_CLASS_ZERO,
    SCALE_CLASS_MIN_SUBNORMAL,
    SCALE_CLASS_MAX_FINITE,
    SCALE_CLASS_NAN_PATTERN,
    SCALE_CLASS_SIGNED_NAN_PATTERN,
    SCALE_CLASS_SIGN_SET,
    SCALE_CLASS_MIXED,
    SCALE_CLASS_COUNT,
};

constexpr const char * k_scale_class_names[SCALE_CLASS_COUNT] = {
    "moderate", "0x00", "0x01", "0x7e", "0x7f", "0xff", "sign-set", "mixed",
};

uint8_t scale_moderate(std::mt19937 & rng) {
    // Exponent field 4..9 with any mantissa: 2^-3 up to 7.5 as E4M3.
    const int exp = 4 + (int) (rng() % 6);
    const int man = (int) (rng() % 8);
    return (uint8_t) ((exp << 3) | man);
}

uint8_t scale_byte(int cls, std::mt19937 & rng) {
    switch (cls) {
        case SCALE_CLASS_MODERATE:
            return scale_moderate(rng);
        case SCALE_CLASS_ZERO:
            return 0x00;
        case SCALE_CLASS_MIN_SUBNORMAL:
            return 0x01;
        case SCALE_CLASS_MAX_FINITE:
            return 0x7e;
        case SCALE_CLASS_NAN_PATTERN:
            return 0x7f;
        case SCALE_CLASS_SIGNED_NAN_PATTERN:
            return 0xff;
        case SCALE_CLASS_SIGN_SET:
            return (uint8_t) (0x80 | scale_moderate(rng));
        case SCALE_CLASS_MIXED:
            return scale_byte((int) (rng() % SCALE_CLASS_MIXED), rng);
        default:
            return 0x00;
    }
}

// One matmul problem: the bytes oneDNN reads, and the reference it must reproduce.
struct woq_problem {
    int                   B = 1;
    int                   M = 0;
    int                   K = 0;
    int                   N = 0;
    std::vector<uint8_t>  weights;    // B planes of K x N f4_e2m1, (k, n) at k * N + n, even offset in the low nibble
    std::vector<uint8_t>  scales;     // B planes of (K / group) x N, (g, n) at g * N + n
    std::vector<uint16_t> src;        // B x M x K f16
    std::vector<double>   ref;        // B x M x N
    std::vector<int>      col_class;  // N entries; -1 when the problem has no scale classes
};

void put_nibble(std::vector<uint8_t> & plane, size_t base, size_t off, uint8_t code) {
    plane[base + off / 2] |= (uint8_t) ((code & 0xf) << ((off & 1) * 4));
}

void fill_src(woq_problem & p, std::mt19937 & rng, std::vector<float> & src_f) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const size_t                          n_src = (size_t) p.B * p.M * p.K;
    p.src.resize(n_src);
    src_f.resize(n_src);
    for (size_t i = 0; i < n_src; ++i) {
        const float v = dist(rng);
        p.src[i]      = f16_bits(v);
        src_f[i]      = round_to_f16(v);
    }
}

// ref[b][m][n] = sum_k src[b][m][k] * w[b][n][k], in double over the f16-rounded activations.
void compute_ref(woq_problem & p, const std::vector<float> & src_f, const std::vector<float> & w_nk) {
    p.ref.assign((size_t) p.B * p.M * p.N, 0.0);
    for (int b = 0; b < p.B; ++b) {
        for (int m = 0; m < p.M; ++m) {
            const float * a = src_f.data() + ((size_t) b * p.M + m) * p.K;
            for (int n = 0; n < p.N; ++n) {
                const float * w   = w_nk.data() + ((size_t) b * p.N + n) * p.K;
                double        acc = 0.0;
                for (int k = 0; k < p.K; ++k) {
                    acc += (double) a[k] * (double) w[k];
                }
                p.ref[((size_t) b * p.M + m) * p.N + n] = acc;
            }
        }
    }
}

// NVFP4 as stored (ggml-common.h, block_nvfp4): row n of an expert holds its K weights in blocks of 64; in sub-block s
// of a block, byte qs[s * 8 + j] holds element s * 16 + j in its low nibble and s * 16 + j + 8 in its high nibble
// (dequantize_row_nvfp4 in ggml-quants.c). The nibble is an E2M1 code. Because kvalues are the E2M1 values doubled
// and the decode halves the scale, the stored product equals true E2M1 times true E4M3, so the codes pass to oneDNN
// unchanged and only their order and the scale bytes change. The repack writes the row-major [K][N] plane, the only
// sub-byte weight layout oneDNN was found to read correctly (test-onednn-woq-mxfp4.cpp, its capability tripwire).
woq_problem make_nvfp4_problem(int B, int M, int K, int N, uint32_t seed) {
    woq_problem p;
    p.B = B;
    p.M = M;
    p.K = K;
    p.N = N;
    std::mt19937 rng(seed);

    const int    nb           = K / k_nvfp4_block;
    const int    groups       = K / k_nvfp4_sub;
    const size_t plane_bytes  = (size_t) K * N / 2;
    const size_t scale_plane  = (size_t) groups * N;
    const size_t expert_bytes = (size_t) N * nb * k_nvfp4_block_bytes;
    p.weights.assign(plane_bytes * B, 0);
    p.scales.assign(scale_plane * B, 0);
    p.col_class.resize(N);
    for (int n = 0; n < N; ++n) {
        p.col_class[n] = n % SCALE_CLASS_COUNT;
    }

    std::vector<uint8_t> stored(expert_bytes);
    std::vector<float>   w_nk((size_t) B * N * K);
    for (int b = 0; b < B; ++b) {
        for (int n = 0; n < N; ++n) {
            for (int ib = 0; ib < nb; ++ib) {
                uint8_t * blk = stored.data() + ((size_t) n * nb + ib) * k_nvfp4_block_bytes;
                for (int s = 0; s < k_nvfp4_block / k_nvfp4_sub; ++s) {
                    blk[s] = scale_byte(p.col_class[n], rng);
                }
                for (int i = 0; i < k_nvfp4_block / 2; ++i) {
                    blk[4 + i] = (uint8_t) (rng() & 0xff);
                }
            }
        }
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                const uint8_t * blk = stored.data() + ((size_t) n * nb + k / k_nvfp4_block) * k_nvfp4_block_bytes;
                const int       e   = k % k_nvfp4_block;
                const int       s   = e / k_nvfp4_sub;
                const int       j   = e % k_nvfp4_sub;
                const uint8_t   q   = blk[4 + s * (k_nvfp4_sub / 2) + (j & 7)];
                const uint8_t   c   = j < 8 ? (uint8_t) (q & 0xf) : (uint8_t) (q >> 4);
                w_nk[((size_t) b * N + n) * K + k] = (float) k_kvalues_doubled[c] * ue4m3_half_sycl(blk[s]);
                put_nibble(p.weights, plane_bytes * b, (size_t) k * N + n, c);
            }
            for (int g = 0; g < groups; ++g) {
                const uint8_t * blk = stored.data() + ((size_t) n * nb + g / 4) * k_nvfp4_block_bytes;
                p.scales[scale_plane * b + (size_t) g * N + n] = ue4m3_sanitize(blk[g % 4]);
            }
        }
    }
    std::vector<float> src_f;
    fill_src(p, rng, src_f);
    compute_ref(p, src_f, w_nk);
    return p;
}

// MXFP4 control: codes and e8m0 scales written straight into the oneDNN planes; the reference uses true E2M1 values
// (half the doubled table) times 2^(e - 127).
woq_problem make_mxfp4_problem(int B, int M, int K, int N, uint32_t seed) {
    woq_problem p;
    p.B = B;
    p.M = M;
    p.K = K;
    p.N = N;
    std::mt19937 rng(seed);

    const int    groups      = K / k_mxfp4_group;
    const size_t plane_bytes = (size_t) K * N / 2;
    const size_t scale_plane = (size_t) groups * N;
    p.weights.assign(plane_bytes * B, 0);
    p.scales.resize(scale_plane * B);
    p.col_class.assign(N, -1);
    for (size_t i = 0; i < p.scales.size(); ++i) {
        p.scales[i] = (uint8_t) (120 + rng() % 15);
    }
    std::vector<float> w_nk((size_t) B * N * K);
    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < N; ++n) {
                const uint8_t c = (uint8_t) (rng() & 0xf);
                put_nibble(p.weights, plane_bytes * b, (size_t) k * N + n, c);
                const uint8_t e                    = p.scales[scale_plane * b + (size_t) (k / k_mxfp4_group) * N + n];
                w_nk[((size_t) b * N + n) * K + k] = 0.5f * (float) k_kvalues_doubled[c] * e8m0_to_f32(e);
            }
        }
    }
    std::vector<float> src_f;
    fill_src(p, rng, src_f);
    compute_ref(p, src_f, w_nk);
    return p;
}

// How the scales are declared to oneDNN.
struct woq_form {
    dt                 scale_dt;
    int                group_k;  // elements of K per scale
    int                mask;
    dnnl::memory::dims groups;
};

struct dev_buffer {
    sycl::queue & q;
    void *        ptr = nullptr;

    dev_buffer(sycl::queue & queue, size_t bytes) : q(queue) { ptr = sycl::malloc_device(bytes, q); }

    ~dev_buffer() {
        if (ptr != nullptr) {
            sycl::free(ptr, q);
        }
    }

    dev_buffer(const dev_buffer &)             = delete;
    dev_buffer & operator=(const dev_buffer &) = delete;
};

struct memory_descs {
    dnnl::memory::desc src;
    dnnl::memory::desc weights;
    dnnl::memory::desc scales;
    dnnl::memory::desc dst;
};

memory_descs make_descs(const woq_problem & p, const woq_form & f, dt weight_dt) {
    const int64_t M = p.M;
    const int64_t K = p.K;
    const int64_t N = p.N;
    const int64_t G = p.K / f.group_k;
    memory_descs  d;
    if (p.B == 1) {
        d.src     = dnnl::memory::desc({ M, K }, dt::f16, { K, 1 });
        d.weights = dnnl::memory::desc({ K, N }, weight_dt, { N, 1 });
        d.scales  = dnnl::memory::desc({ G, N }, f.scale_dt, { N, 1 });
        d.dst     = dnnl::memory::desc({ M, N }, dt::f32, { N, 1 });
    } else {
        const int64_t B = p.B;
        d.src           = dnnl::memory::desc({ B, M, K }, dt::f16, { M * K, K, 1 });
        d.weights       = dnnl::memory::desc({ B, K, N }, weight_dt, { K * N, N, 1 });
        d.scales        = dnnl::memory::desc({ B, G, N }, f.scale_dt, { G * N, N, 1 });
        d.dst           = dnnl::memory::desc({ B, M, N }, dt::f32, { M * N, N, 1 });
    }
    return d;
}

// Builds the primitive descriptor with the attribute production uses. Throws dnnl::error when oneDNN refuses the
// attribute or has no implementation; the caller reports that as UNSUPPORTED.
dnnl::matmul::primitive_desc make_pd(const dnnl::engine & eng, const memory_descs & d, const woq_form & f) {
    dnnl::primitive_attr attr;
    ggml_sycl_dnnl_attr_set_reproducible(attr);
    attr.set_scales(DNNL_ARG_WEIGHTS, f.mask, f.groups, f.scale_dt);
    attr.set_fpmath_mode(dnnl::fpmath_mode::f16, /*apply_to_int=*/true);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);
    return dnnl::matmul::primitive_desc(eng, d.src, d.weights, d.dst, attr);
}

std::string dims_str(const woq_problem & p) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "B=%d M=%d K=%d N=%d", p.B, p.M, p.K, p.N);
    return buf;
}

void report(const char * name, const char * verdict, const std::string & detail) {
    std::printf("PROBE %s %s %s\n", name, verdict, detail.c_str());
    std::fflush(stdout);
}

// Per column (and expert): the largest error over its rows divided by its largest reference value. A column whose
// reference is exactly 0 (every scale 0x00, 0x7F or 0xFF) must read exactly 0.
struct column_check {
    bool        ok        = true;
    double      worst_rel = 0.0;
    int         worst_col = -1;
    int         nonfinite = 0;
    double      nmse      = 0.0;
    std::string class_worst;
};

column_check check_columns(const woq_problem & p, const std::vector<float> & out) {
    column_check r;
    double       class_worst[SCALE_CLASS_COUNT] = {};
    double       err2                           = 0.0;
    double       ref2                           = 0.0;
    for (int b = 0; b < p.B; ++b) {
        for (int n = 0; n < p.N; ++n) {
            double max_ref = 0.0;
            double max_err = 0.0;
            for (int m = 0; m < p.M; ++m) {
                const size_t i   = ((size_t) b * p.M + m) * p.N + n;
                const double o   = out[i];
                const double ref = p.ref[i];
                if (!std::isfinite(o)) {
                    r.nonfinite++;
                    continue;
                }
                max_ref = std::max(max_ref, std::fabs(ref));
                max_err = std::max(max_err, std::fabs(o - ref));
                err2 += (o - ref) * (o - ref);
                ref2 += ref * ref;
            }
            double rel = 0.0;
            if (max_ref == 0.0) {
                rel = max_err == 0.0 ? 0.0 : INFINITY;
            } else {
                rel = max_err / max_ref;
            }
            if (rel > r.worst_rel || r.worst_col < 0) {
                r.worst_rel = rel;
                r.worst_col = n;
            }
            const int cls = p.col_class[n];
            if (cls >= 0) {
                class_worst[cls] = std::max(class_worst[cls], rel);
            }
        }
    }
    r.nmse = ref2 > 0.0 ? err2 / ref2 : err2;
    r.ok   = r.nonfinite == 0 && r.worst_rel <= k_column_tol;
    if (!p.col_class.empty() && p.col_class[0] >= 0) {
        for (int c = 0; c < SCALE_CLASS_COUNT; ++c) {
            char buf[48];
            std::snprintf(buf, sizeof(buf), "%s%s:%.3g", c == 0 ? "" : ",", k_scale_class_names[c], class_worst[c]);
            r.class_worst += buf;
        }
    }
    return r;
}

// Runs one numerics arm: builds the primitive (UNSUPPORTED when oneDNN refuses), executes it on the problem's bytes,
// compares against the reference and times ten more executes so a reference implementation shows as slow.
void run_numerics_arm(const char *         name,
                      sycl::queue &        q,
                      const dnnl::engine & eng,
                      dnnl::stream &       stream,
                      const woq_problem &  p,
                      const woq_form &     f) {
    const memory_descs           d = make_descs(p, f, dt::f4_e2m1);
    dnnl::matmul::primitive_desc pd;
    try {
        pd = make_pd(eng, d, f);
    } catch (const dnnl::error & e) {
        report(name, "UNSUPPORTED", dims_str(p) + " refused: " + e.what());
        return;
    }
    const std::string impl = pd.impl_info_str() ? pd.impl_info_str() : "?";
    try {
        const size_t out_count = (size_t) p.B * p.M * p.N;
        dev_buffer   src_dev(q, p.src.size() * sizeof(uint16_t));
        dev_buffer   w_dev(q, p.weights.size());
        dev_buffer   s_dev(q, p.scales.size());
        dev_buffer   dst_dev(q, out_count * sizeof(float));
        if (!src_dev.ptr || !w_dev.ptr || !s_dev.ptr || !dst_dev.ptr) {
            report(name, "FAIL", dims_str(p) + " impl=" + impl + " device allocation failed");
            return;
        }
        q.memcpy(src_dev.ptr, p.src.data(), p.src.size() * sizeof(uint16_t));
        q.memcpy(w_dev.ptr, p.weights.data(), p.weights.size());
        q.memcpy(s_dev.ptr, p.scales.data(), p.scales.size());
        q.memset(dst_dev.ptr, 0, out_count * sizeof(float));
        q.wait();

        dnnl::memory                          src_m(d.src, eng, src_dev.ptr);
        dnnl::memory                          w_m(d.weights, eng, w_dev.ptr);
        dnnl::memory                          s_m(d.scales, eng, s_dev.ptr);
        dnnl::memory                          dst_m(d.dst, eng, dst_dev.ptr);
        dnnl::matmul                          prim(pd);
        std::unordered_map<int, dnnl::memory> args{
            { DNNL_ARG_SRC,                            src_m },
            { DNNL_ARG_WEIGHTS,                        w_m   },
            { DNNL_ARG_DST,                            dst_m },
            { DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, s_m   },
        };
        prim.execute(stream, args);
        stream.wait();
        std::vector<float> out(out_count);
        q.memcpy(out.data(), dst_dev.ptr, out_count * sizeof(float)).wait();

        constexpr int reps = 10;
        const auto    t0   = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            prim.execute(stream, args);
        }
        stream.wait();
        const double us =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;

        const column_check c = check_columns(p, out);
        char               buf[256];
        std::snprintf(buf, sizeof(buf), " impl=%s worst_rel=%.3g (col %d) nonfinite=%d nmse=%.3g us_per_exec=%.1f",
                      impl.c_str(), c.worst_rel, c.worst_col, c.nonfinite, c.nmse, us);
        std::string detail = dims_str(p) + buf;
        if (!c.class_worst.empty()) {
            detail += " class_worst=" + c.class_worst;
        }
        report(name, c.ok ? "PASS" : "FAIL", detail);
    } catch (const dnnl::error & e) {
        report(name, "FAIL", dims_str(p) + " impl=" + impl + " execute: " + e.what());
    } catch (const sycl::exception & e) {
        report(name, "FAIL", dims_str(p) + " impl=" + impl + " sycl: " + e.what());
    }
}

}  // namespace

int main() {
    std::unique_ptr<sycl::queue> q;
    try {
        q = std::make_unique<sycl::queue>(sycl::gpu_selector_v);
    } catch (const sycl::exception & e) {
        std::printf("SKIP: no SYCL GPU (%s)\n", e.what());
        return LLAMA_TEST_EXIT_SKIP;
    }
    if (!q->get_device().has(sycl::aspect::fp16)) {
        std::printf("SKIP: the SYCL GPU has no fp16\n");
        return LLAMA_TEST_EXIT_SKIP;
    }
    try {
        const dnnl_version_t * v = dnnl_version();
        std::printf("[probe] device=%s oneDNN=%d.%d.%d\n", q->get_device().get_info<sycl::info::device::name>().c_str(),
                    v->major, v->minor, v->patch);
        std::fflush(stdout);

        dnnl::engine eng    = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
        dnnl::stream stream = dnnl::sycl_interop::make_stream(eng, *q);

        const woq_form nvfp4_2d = {
            dt::f8_e4m3, k_nvfp4_sub, (1 << 0) | (1 << 1), { k_nvfp4_sub, 1 }
        };
        const woq_form nvfp4_3d = {
            dt::f8_e4m3, k_nvfp4_sub, (1 << 0) | (1 << 1) | (1 << 2), { k_nvfp4_sub, 1 }
        };
        const woq_form mxfp4_3d = {
            dt::e8m0, k_mxfp4_group, (1 << 0) | (1 << 1) | (1 << 2), { k_mxfp4_group, 1 }
        };
        const woq_form mxfp4_3d_prod = {
            dt::e8m0, k_mxfp4_group, (1 << 0) | (1 << 1) | (1 << 2), { 1, k_mxfp4_group, 1 }
        };

        // An Inkling gate_up expert: K = N = 2304, 32 rows (the average rows per expert of a 512-token micro-batch
        // with 8 of 128 experts used).
        const woq_problem p2d = make_nvfp4_problem(1, 32, 2304, 2304, 0x4e5646u);
        try {
            const dnnl::matmul::primitive_desc pd = make_pd(eng, make_descs(p2d, nvfp4_2d, dt::f4_e2m1), nvfp4_2d);
            report("nvfp4-2d-create", "PASS", std::string("impl=") + (pd.impl_info_str() ? pd.impl_info_str() : "?"));
        } catch (const dnnl::error & e) {
            report("nvfp4-2d-create", "UNSUPPORTED", std::string("refused: ") + e.what());
        }
        run_numerics_arm("nvfp4-2d-numerics", *q, eng, stream, p2d, nvfp4_2d);

        // Four Inkling down experts in one execute: K = 1152, N = 2304.
        const woq_problem p3d = make_nvfp4_problem(4, 32, 1152, 2304, 0x335644u);
        run_numerics_arm("nvfp4-3d", *q, eng, stream, p3d, nvfp4_3d);

        const woq_problem pmx = make_mxfp4_problem(4, 32, 1152, 2304, 0x4d5846u);
        run_numerics_arm("mxfp4-3d-control", *q, eng, stream, pmx, mxfp4_3d);
        run_numerics_arm("mxfp4-3d-prod-groups", *q, eng, stream, pmx, mxfp4_3d_prod);
    } catch (const std::exception & e) {
        std::printf("PROBE harness FAIL %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
    return 0;
}
