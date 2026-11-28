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
//   nvfp4-2d-f16-scales   the 2-D and 3-D NVFP4 forms with each sanitised scale widened to f16, the fallback when
//   nvfp4-3d-f16-scales   f8_e4m3 scales are refused: 0.125 instead of 0.0625 bytes of scale per element.
//
// Exit status: 0 once every arm has run, whatever each found; 77 without a SYCL GPU; 1 only when the probe itself
// breaks outside an arm. So it is built and not registered as a ctest, which must be able to fail: it measures what
// oneDNN offers, and no shipped path relies on an arm yet (the NVFP4 prompt executor runs its F16 arm). The lead runs
// it by hand: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-onednn-woq-nvfp4. Device buffers come from
// sycl::malloc_device, as in test-onednn-woq-mxfp4.cpp: the probe calls oneDNN directly and links no backend, so there
// is no unified cache to allocate from.
//
// --bench (argv[1]) runs, instead of the probe arms, the timing that decides how the executor's NVFP4 experts go to a
// GEMM (plan Task 26a): today's path (f16-dequant, a SYCL kernel, then f16-gemm) against woq-repack (a tiled SYCL
// kernel to the planes above) then a WOQ form (woq-3d-e4m3, woq-3d-f16s, woq-2d-e4m3, woq-2d-f16s), plus the diagnostic
// woq-3d-ba-diag, at Inkling's gate_up and down shapes, B in {8, 32, 128} experts and M in {32, 64, 128} rows. Every
// arm is checked once per shape at B 8, M 32 ("NUMERICS ..." lines), then timed in device time on an in-order profiling
// queue between two marker kernels: one "BENCH <arm> shape= B= M= K= N= dev_us_per_expert= wall_us_per_expert= GBps=
// numerics= impl=" line per arm and grid point, then one "PIPELINE shape= B= M= f16= woq_best= form= ratio=" line per
// shape, B and M.
//
// The bench's output vocabulary:
//   NUMERICS <arm> ... PASS|FAIL      the check at B 8, M 32; its verdict is BENCH's numerics= for the whole shape:
//                                     PASS, BADNUM (ran, check failed) or FAIL (could not be built or run).
//   BENCH ... timing=BADSPAN          appended when marker2 did not start after marker1 ended: dev_us_per_expert=0.00.
//   BENCH ... timing=FAIL             appended when the arm produced no time (refused, unallocated, or threw while
//                                     timed); its numerics= is still the check's.
//   WARN ... device time exceeds wall time   the timestamps of that BENCH line are not trustworthy.
//   PIPELINE ... ratio=BADNUM         f16-dequant, f16-gemm or woq-repack did not pass at that shape.
//   PIPELINE ... form=none ratio=nan  no WOQ form passed, or a sum is not a finite time.
//   XCHECK <arm> shape= B= M= worst_rel= nonfinite= ref_nonfinite= zero_cols=<n> of <B*N> PASS|FAIL impl=
//                                     woq-3d-ba-diag and woq-3d-f16s against f16-gemm's dst at every grid point and at
//                                     (B 1, M 64), (B 32, M 192), (B 32, M 256): the same weights and src, worst column
//                                     error over the column's largest value. nonfinite= counts the arm's non-finite
//                                     output, ref_nonfinite= f16-gemm's; zero_cols= counts the columns both read as all
//                                     zero, 3 of every 8 when healthy (the 0x00, 0x7f and 0xff scale classes). PASS at
//                                     <= 1e-3, both counts 0 and no more zero columns than the class table holds.
//   XCHECK-ERROR <arm> shape= B= M= <why>   before an XCHECK line: the arm, or f16-gemm, could not run
//                                     (f16-gemm:nonfinite when the reference itself held non-finite output).
//   SCRATCHPAD woq-3d-ba-diag shape= B= M= user_bytes= impl=   that primitive built in user scratchpad mode
//                                     (production's; the bench times library mode), not executed.
//   XCHECK_SUMMARY pass= fail=        the XCHECK totals. XCHECK failures do not change the exit status.
// woq-repack writes both scale planes, so woq_best pays for the plane its form does not read: about 10% of the repack's
// bytes when woq_best is an e4m3 form, about 5% when it is an f16-scale form.
#include "test-skip.h"
// The attribute production builds with, so the answer is about the primitive the backend would run.
#include "ggml-sycl/dnnl-deterministic.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <vector>

using dt = dnnl::memory::data_type;

// The --bench kernels' names, declared at namespace scope as SYCL kernel names must be.
class nvfp4_bench_dequant;
class nvfp4_bench_repack;
class nvfp4_bench_marker;

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
    std::vector<uint8_t>  stored;     // NVFP4 only: B experts of N rows of block_nvfp4, the bytes the backend holds
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

    p.stored.assign(expert_bytes * B, 0);
    std::vector<float> w_nk((size_t) B * N * K);
    for (int b = 0; b < B; ++b) {
        uint8_t * const stored = p.stored.data() + expert_bytes * b;
        for (int n = 0; n < N; ++n) {
            for (int ib = 0; ib < nb; ++ib) {
                uint8_t * blk = stored + ((size_t) n * nb + ib) * k_nvfp4_block_bytes;
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
                const uint8_t * blk = stored + ((size_t) n * nb + k / k_nvfp4_block) * k_nvfp4_block_bytes;
                const int       e   = k % k_nvfp4_block;
                const int       s   = e / k_nvfp4_sub;
                const int       j   = e % k_nvfp4_sub;
                const uint8_t   q   = blk[4 + s * (k_nvfp4_sub / 2) + (j & 7)];
                const uint8_t   c   = j < 8 ? (uint8_t) (q & 0xf) : (uint8_t) (q >> 4);
                w_nk[((size_t) b * N + n) * K + k] = (float) k_kvalues_doubled[c] * ue4m3_half_sycl(blk[s]);
                put_nibble(p.weights, plane_bytes * b, (size_t) k * N + n, c);
            }
            for (int g = 0; g < groups; ++g) {
                const uint8_t * blk = stored + ((size_t) n * nb + g / 4) * k_nvfp4_block_bytes;
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

// The same problem with every sanitised E4M3 scale byte replaced by its value as f16 (two bytes, little-endian). A
// sanitised byte has no sign and no NaN pattern, and every E4M3 value is exact in f16, so the reference stands.
woq_problem with_f16_scales(const woq_problem & p) {
    woq_problem out = p;
    out.scales.resize(p.scales.size() * 2);
    for (size_t i = 0; i < p.scales.size(); ++i) {
        const uint16_t h      = f16_bits(2.0f * ue4m3_half_sycl(p.scales[i]));
        out.scales[2 * i]     = (uint8_t) (h & 0xff);
        out.scales[2 * i + 1] = (uint8_t) (h >> 8);
    }
    return out;
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

    // Waits for the queue first, so a buffer left behind by an exception is not freed under a kernel still using it. A
    // destructor must not throw, so nothing the wait or the free raises escapes.
    ~dev_buffer() {
        if (ptr != nullptr) {
            try {
                q.wait();
                sycl::free(ptr, q);
            } catch (...) {
            }
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

// The 2-D form for one expert, the 3-D batched form otherwise; force_3d keeps the 3-D form at B 1 (the bench's).
memory_descs make_descs(const woq_problem & p, const woq_form & f, dt weight_dt, bool force_3d = false) {
    const int64_t M = p.M;
    const int64_t K = p.K;
    const int64_t N = p.N;
    const int64_t G = p.K / f.group_k;
    memory_descs  d;
    if (p.B == 1 && !force_3d) {
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
// attribute or has no implementation; the caller reports that as UNSUPPORTED. The bench's SCRATCHPAD line asks for user
// mode, production's; everything that executes uses library mode.
dnnl::matmul::primitive_desc make_pd(const dnnl::engine &  eng,
                                     const memory_descs &  d,
                                     const woq_form &      f,
                                     dnnl::scratchpad_mode scratchpad = dnnl::scratchpad_mode::library) {
    dnnl::primitive_attr attr;
    ggml_sycl_dnnl_attr_set_reproducible(attr);
    attr.set_scales(DNNL_ARG_WEIGHTS, f.mask, f.groups, f.scale_dt);
    attr.set_fpmath_mode(dnnl::fpmath_mode::f16, /*apply_to_int=*/true);
    attr.set_scratchpad_mode(scratchpad);
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

// ---- --bench: dequant + f16 GEMM against repack + WOQ GEMM at Inkling's expert shapes (plan Task 26a) ----

constexpr int    k_bench_wg          = 256;  // work-items per group, both kernels
constexpr int    k_bench_tile        = 64;   // the repack's tile: 64 n by 64 k, one block of each of 64 rows
constexpr int    k_bench_block_words = k_nvfp4_block_bytes / 4;
constexpr int    k_bench_warmup      = 3;
constexpr double k_bench_min_span_ns = 50e6;
constexpr int    k_bench_max_reps    = 1 << 16;
constexpr int    k_bench_check_b     = 8;
constexpr int    k_bench_check_m     = 32;
constexpr int    k_bench_bs[]        = { 8, 32, 128 };
constexpr int    k_bench_ms[]        = { 32, 64, 128 };
constexpr double k_xcheck_tol        = 1e-3;

// XCHECK-only points beyond the grid: one expert, and the executor's taller row groups. No BENCH line, no timing.
struct bench_extent {
    int B;
    int M;
};

constexpr bench_extent k_xcheck_extra[] = {
    { 1,  64  },
    { 32, 192 },
    { 32, 256 },
};

// The numerics check runs at the first grid point of a shape, and every point's verdicts start as FAIL until it has.
static_assert(k_bench_bs[0] == k_bench_check_b && k_bench_ms[0] == k_bench_check_m,
              "the check point must be the grid's first point");

// The scale step of the repack: one work-item per (group, n) of the tile.
static_assert(k_bench_wg == (k_nvfp4_block / k_nvfp4_sub) * k_bench_tile, "repack scale step needs 4 x 64 work-items");

// ue4m3_half_sycl for device code, built from bits as common.hpp's ggml_sycl_ue4m3_to_fp32 does, so the kernels call no
// libm function. run_bench checks it against ue4m3_half_sycl for all 256 bytes before anything runs.
float ue4m3_half_device(uint8_t x) {
    const uint32_t exp    = (x >> 3) & 0xfu;
    const uint32_t man    = x & 0x7u;
    const float    normal = sycl::bit_cast<float>(((exp + 119u) << 23) | (man << 20));
    const float    small  = (float) man * 0x1p-10f;
    const float    value  = exp == 0 ? small : normal;
    return (x & 0x7fu) == 0x7fu ? 0.0f : value;
}

// f16-dequant: B experts of stored NVFP4 to f16, expert b row n holding its K halves contiguously at
// (b * N + n) * K + k, the layout the executor's dequant writes and f16-gemm reads. A stand-in for convert.cpp's NVFP4
// converter at equal bytes. One work-item per output pair of halves, consecutive work-items consecutive along K; the
// block indexing is make_nvfp4_problem's. run_bench checks that B * N * K / 2 fills whole work-groups.
sycl::event launch_dequant(sycl::queue & q, const uint8_t * stored, uint32_t * w16_pairs, int B, int K, int N) {
    const int    nb           = K / k_nvfp4_block;
    const size_t pairs        = (size_t) K / 2;
    const size_t expert_bytes = (size_t) N * nb * k_nvfp4_block_bytes;
    const size_t total        = (size_t) B * N * pairs;
    return q.parallel_for<nvfp4_bench_dequant>(sycl::nd_range<1>(total, k_bench_wg), [=](sycl::nd_item<1> it) {
        const size_t    i      = it.get_global_id(0);
        const size_t    row    = i / pairs;  // b * N + n
        const size_t    n      = row % N;
        const size_t    b      = row / N;
        const int       k0     = (int) (i % pairs) * 2;
        const uint8_t * blk    = stored + b * expert_bytes + (n * nb + k0 / k_nvfp4_block) * k_nvfp4_block_bytes;
        uint32_t        packed = 0;
        for (int h = 0; h < 2; ++h) {
            const int     e  = (k0 + h) % k_nvfp4_block;
            const int     s  = e / k_nvfp4_sub;
            const int     j  = e % k_nvfp4_sub;
            const uint8_t qb = blk[4 + s * (k_nvfp4_sub / 2) + (j & 7)];
            const uint8_t c  = j < 8 ? (uint8_t) (qb & 0xf) : (uint8_t) (qb >> 4);
            const float   w  = (float) k_kvalues_doubled[c] * ue4m3_half_device(blk[s]);
            packed |= (uint32_t) sycl::bit_cast<uint16_t>(sycl::half(w)) << (16 * h);
        }
        w16_pairs[i] = packed;
    });
}

// woq-repack: B experts of stored NVFP4 to the planes make_descs declares: nibble (k, n) at k * N + n with the even
// offset in the low nibble, the ue4m3_sanitize byte of scale group g at g * N + n, and that scale widened to f16 as
// with_f16_scales does, at the same index. It is a transpose from n-major blocks to k-major nibbles, so it goes
// through local memory: a work-group owns 64 n by 64 k (block kb of 64 rows), reads those blocks with consecutive
// work-items on consecutive words, and writes each of its 64 output rows' 32 nibble bytes as 8 consecutive words. It
// has no tail: run_bench checks that K and N are multiples of 64.
sycl::event launch_repack(sycl::queue &   q,
                          const uint8_t * stored,
                          uint8_t *       nibbles,
                          uint8_t *       scales_e4m3,
                          uint16_t *      scales_f16,
                          int             B,
                          int             K,
                          int             N) {
    const int    nb           = K / k_nvfp4_block;
    const int    G            = K / k_nvfp4_sub;
    const int    n_tiles      = N / k_bench_tile;
    const size_t expert_bytes = (size_t) N * nb * k_nvfp4_block_bytes;
    const size_t plane_bytes  = (size_t) K * N / 2;
    const size_t n_groups     = (size_t) B * nb * n_tiles;
    return q.submit([&](sycl::handler & h) {
        sycl::local_accessor<uint32_t, 1> tile(sycl::range<1>(k_bench_tile * k_bench_block_words), h);
        h.parallel_for<nvfp4_bench_repack>(
            sycl::nd_range<1>(n_groups * k_bench_wg, k_bench_wg), [=](sycl::nd_item<1> it) {
                const size_t grp = it.get_group(0);
                const int    lid = (int) it.get_local_id(0);
                const int    n0  = (int) (grp % n_tiles) * k_bench_tile;
                const int    kb  = (int) (grp / n_tiles % nb);
                const size_t b   = grp / n_tiles / nb;
                const int    k0  = kb * k_nvfp4_block;

                // Tile row r is block kb of row n0 + r: d[4] then qs[32], 9 words.
                const uint32_t * src = reinterpret_cast<const uint32_t *>(stored + b * expert_bytes);
                for (int i = lid; i < k_bench_tile * k_bench_block_words; i += k_bench_wg) {
                    const int r = i / k_bench_block_words;
                    const int w = i % k_bench_block_words;
                    tile[i]     = src[((size_t) (n0 + r) * nb + kb) * k_bench_block_words + w];
                }
                sycl::group_barrier(it.get_group());
                const auto tile_byte = [&](int r, int byte) {
                    return (uint8_t) (tile[r * k_bench_block_words + byte / 4] >> (8 * (byte % 4)));
                };

                // Word wn of output row k holds n0 + 8 wn .. n0 + 8 wn + 7, nibble t at bits 4t: n0 is even, so an even
                // n lands in the low nibble of its byte.
                uint32_t * dst = reinterpret_cast<uint32_t *>(nibbles + b * plane_bytes);
                for (int i = lid; i < k_bench_tile * (k_bench_tile / 8); i += k_bench_wg) {
                    const int kk   = i / (k_bench_tile / 8);
                    const int wn   = i % (k_bench_tile / 8);
                    const int k    = k0 + kk;
                    const int e    = k % k_nvfp4_block;
                    const int s    = e / k_nvfp4_sub;
                    const int j    = e % k_nvfp4_sub;
                    uint32_t  word = 0;
                    for (int t = 0; t < 8; ++t) {
                        const uint8_t qb = tile_byte(wn * 8 + t, 4 + s * (k_nvfp4_sub / 2) + (j & 7));
                        const uint8_t c  = j < 8 ? (uint8_t) (qb & 0xf) : (uint8_t) (qb >> 4);
                        word |= (uint32_t) c << (4 * t);
                    }
                    dst[((size_t) k * N + n0 + wn * 8) / 8] = word;
                }

                // Scale group g of row n is blk[g % 4] of the row's block g / 4, as make_nvfp4_problem reads it.
                const int     gl = lid / k_bench_tile;
                const int     nl = lid % k_bench_tile;
                const int     g  = k0 / k_nvfp4_sub + gl;
                const uint8_t s8 = ue4m3_sanitize(tile_byte(nl, g % 4));
                const size_t  o  = b * G * N + (size_t) g * N + n0 + nl;
                scales_e4m3[o]   = s8;
                scales_f16[o]    = sycl::bit_cast<uint16_t>(sycl::half(2.0f * ue4m3_half_device(s8)));
            });
    });
}

// Waits for everything queued, swallowing what the wait raises: for exception paths and destructors.
void queue_drain_nothrow(sycl::queue & q) {
    try {
        q.wait();
    } catch (...) {
    }
}

sycl::event submit_marker(sycl::queue & q) {
    return q.single_task<nvfp4_bench_marker>([]() {});
}

struct bench_timing {
    bool   ok       = false;
    bool   bad_span = false;  // ran, but marker2 command_start did not follow marker1 command_end; dev_us is 0
    double dev_us   = NAN;    // per expert
    double wall_us  = NAN;    // per expert
};

// Device time per expert of rep(), one rep being B experts' work: 3 warm-up reps, then a marker, R reps and a second
// marker on the in-order profiling queue, (marker2 command_start - marker1 command_end) / R / B. R grows until the
// timed span is at least 50 ms. Host wall-clock is taken over the same R, from the first marker's submission to the
// wait. A span of zero or less means the timestamps cannot time this arm: it stops there, marked bad_span, and R does
// not grow.
template <typename F> bench_timing time_arm(sycl::queue & q, int B, F && rep) {
    for (int i = 0; i < k_bench_warmup; ++i) {
        rep();
    }
    q.wait_and_throw();
    int reps = 1;
    for (;;) {
        const auto  t0 = std::chrono::steady_clock::now();
        sycl::event m1 = submit_marker(q);
        for (int r = 0; r < reps; ++r) {
            rep();
        }
        sycl::event m2 = submit_marker(q);
        q.wait_and_throw();
        const auto     t1     = std::chrono::steady_clock::now();
        const uint64_t end1   = m1.get_profiling_info<sycl::info::event_profiling::command_end>();
        const uint64_t start2 = m2.get_profiling_info<sycl::info::event_profiling::command_start>();
        const double   span   = start2 > end1 ? (double) (start2 - end1) : 0.0;
        if (span <= 0.0 || span >= k_bench_min_span_ns || reps >= k_bench_max_reps) {
            bench_timing t;
            t.ok       = span > 0.0;
            t.bad_span = span <= 0.0;
            t.dev_us   = span / 1e3 / reps / B;
            t.wall_us  = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps / B;
            return t;
        }
        const double per_rep = std::max(span / reps, 1e3);
        const double wanted  = std::ceil(1.2 * k_bench_min_span_ns / per_rep);
        reps                 = (int) std::min<double>(k_bench_max_reps, std::max<double>(2.0 * reps, wanted));
    }
}

enum bench_arm : int {
    BENCH_ARM_F16_DEQUANT,
    BENCH_ARM_F16_GEMM,
    BENCH_ARM_WOQ_REPACK,
    BENCH_ARM_WOQ_3D_E4M3,
    BENCH_ARM_WOQ_3D_F16S,
    BENCH_ARM_WOQ_2D_E4M3,
    BENCH_ARM_WOQ_2D_F16S,
    BENCH_ARM_WOQ_3D_BA_DIAG,
    BENCH_ARM_COUNT,
};

constexpr const char * k_bench_arm_names[BENCH_ARM_COUNT] = {
    "f16-dequant", "f16-gemm",    "woq-repack",  "woq-3d-e4m3",
    "woq-3d-f16s", "woq-2d-e4m3", "woq-2d-f16s", "woq-3d-ba-diag",
};

// The forms PIPELINE picks woq_best from; the diagnostic arm is not one of them.
constexpr int k_bench_woq_forms[] = { BENCH_ARM_WOQ_3D_E4M3, BENCH_ARM_WOQ_3D_F16S, BENCH_ARM_WOQ_2D_E4M3,
                                      BENCH_ARM_WOQ_2D_F16S };

// The arms XCHECK holds to f16-gemm: the diagnostic form (n-major, K-contiguous nibbles; its scale plane stays
// k-major) and the transposed 3-D f16-scale form.
constexpr int k_xcheck_arms[] = { BENCH_ARM_WOQ_3D_BA_DIAG, BENCH_ARM_WOQ_3D_F16S };

bool bench_arm_f16_scales(int arm) {
    return arm == BENCH_ARM_WOQ_3D_F16S || arm == BENCH_ARM_WOQ_2D_F16S || arm == BENCH_ARM_WOQ_3D_BA_DIAG;
}

bool bench_arm_2d(int arm) {
    return arm == BENCH_ARM_WOQ_2D_E4M3 || arm == BENCH_ARM_WOQ_2D_F16S;
}

// What the numerics check at B 8, M 32 found for an arm, carried to every grid point of its shape.
enum bench_verdict : int {
    BENCH_VERDICT_PASS,
    BENCH_VERDICT_BADNUM,  // ran, and its check failed
    BENCH_VERDICT_FAIL,    // could not be built or run
};

const char * bench_verdict_str(int v) {
    return v == BENCH_VERDICT_PASS ? "PASS" : v == BENCH_VERDICT_BADNUM ? "BADNUM" : "FAIL";
}

struct bench_shape {
    const char * name;
    int          K;
    int          N;
    uint32_t     seed;
};

// One grid point's device buffers, freed when the point ends so peak device memory is one point's worth.
struct bench_point {
    int        B;
    int        M;
    int        K;
    int        N;
    int        G;
    size_t     stored_bytes;
    size_t     w16_bytes;
    size_t     nib_bytes;
    size_t     s8_bytes;
    size_t     src_bytes;
    size_t     dst_bytes;
    dev_buffer stored;  // B experts of block_nvfp4 rows
    dev_buffer w16;     // f16-dequant's output, f16-gemm's weights
    dev_buffer nib;     // woq-repack's nibble plane
    dev_buffer nib_nk;  // woq-3d-ba-diag's nibbles, n-major and K-contiguous, built on the host
    dev_buffer s8;      // woq-repack's f8_e4m3 scale plane
    dev_buffer s16;     // woq-repack's f16 scale plane
    dev_buffer src;
    dev_buffer dst;

    bench_point(sycl::queue & q, int B_, int M_, int K_, int N_) :
        B(B_),
        M(M_),
        K(K_),
        N(N_),
        G(K_ / k_nvfp4_sub),
        stored_bytes((size_t) B_ * N_ * (K_ / k_nvfp4_block) * k_nvfp4_block_bytes),
        w16_bytes((size_t) B_ * K_ * N_ * sizeof(uint16_t)),
        nib_bytes((size_t) B_ * K_ * N_ / 2),
        s8_bytes((size_t) B_ * G * N_),
        src_bytes((size_t) B_ * M_ * K_ * sizeof(uint16_t)),
        dst_bytes((size_t) B_ * M_ * N_ * sizeof(float)),
        stored(q, stored_bytes),
        w16(q, w16_bytes),
        nib(q, nib_bytes),
        nib_nk(q, nib_bytes),
        s8(q, s8_bytes),
        s16(q, s8_bytes * 2),
        src(q, src_bytes),
        dst(q, dst_bytes) {}

    bool allocated() const {
        return stored.ptr && w16.ptr && nib.ptr && nib_nk.ptr && s8.ptr && s16.ptr && src.ptr && dst.ptr;
    }
};

// Bytes one expert's work reads plus writes, for GBps.
double bench_arm_bytes(int arm, const bench_point & pt) {
    const double stored = (double) pt.stored_bytes / pt.B;
    const double kn     = (double) pt.K * pt.N;
    const double gn     = (double) pt.G * pt.N;
    const double act    = (double) pt.M * pt.K * sizeof(uint16_t) + (double) pt.M * pt.N * sizeof(float);
    switch (arm) {
        case BENCH_ARM_F16_DEQUANT:
            return stored + kn * sizeof(uint16_t);
        case BENCH_ARM_F16_GEMM:
            return kn * sizeof(uint16_t) + act;
        case BENCH_ARM_WOQ_REPACK:
            return stored + kn / 2 + gn + gn * sizeof(uint16_t);
        default:
            return kn / 2 + gn * (bench_arm_f16_scales(arm) ? sizeof(uint16_t) : 1) + act;
    }
}

std::string one_token(std::string s) {
    for (char & c : s) {
        if (c == ' ' || c == '\t' || c == '\n') {
            c = '_';
        }
    }
    return s;
}

// A oneDNN arm built for one grid point: one rep is every entry of execs, one execute each (B for the 2-D forms).
struct onednn_arm {
    bool                                               ok = false;
    std::string                                        impl;
    dnnl::matmul                                       prim;
    std::vector<std::unordered_map<int, dnnl::memory>> execs;
};

void onednn_arm_rep(onednn_arm & a, dnnl::stream & stream) {
    for (auto & args : a.execs) {
        a.prim.execute(stream, args);
    }
}

// A WOQ arm's scales: f8_e4m3 or f16, one per 16 elements of K, mask 3 for the 2-D form and 7 for the 3-D form.
woq_form bench_woq_form(int arm) {
    const bool two_d = bench_arm_2d(arm);
    return {
        bench_arm_f16_scales(arm) ? dt::f16 : dt::f8_e4m3,
        k_nvfp4_sub,
        two_d ? (1 << 0) | (1 << 1) : (1 << 0) | (1 << 1) | (1 << 2),
        { k_nvfp4_sub, 1 }
    };
}

// A WOQ arm's descriptors, as make_descs declares them: the 2-D form for one expert, the 3-D form for B experts (B 1
// included). The diagnostic arm alone declares its nibbles n-major and K-contiguous, nibble (b, k, n) at
// b * K * N + n * K + k, while its scale plane stays k-major like every other form's.
memory_descs bench_woq_descs(int arm, const bench_point & pt, const woq_form & f) {
    const bool  two_d = bench_arm_2d(arm);
    woq_problem dims;
    dims.B         = two_d ? 1 : pt.B;
    dims.M         = pt.M;
    dims.K         = pt.K;
    dims.N         = pt.N;
    memory_descs d = make_descs(dims, f, dt::f4_e2m1, /*force_3d=*/!two_d);
    if (arm == BENCH_ARM_WOQ_3D_BA_DIAG) {
        const int64_t B = pt.B;
        const int64_t K = pt.K;
        const int64_t N = pt.N;
        d.weights       = dnnl::memory::desc({ B, K, N }, dt::f4_e2m1, { K * N, 1, K });
    }
    return d;
}

onednn_arm build_onednn_arm(int arm, const dnnl::engine & eng, bench_point & pt) {
    onednn_arm    a;
    const int64_t B = pt.B;
    const int64_t M = pt.M;
    const int64_t K = pt.K;
    const int64_t N = pt.N;
    try {
        dnnl::matmul::primitive_desc pd;
        if (arm == BENCH_ARM_F16_GEMM) {
            // Weights as the executor's dequant stores them: expert b, row n holds K contiguous halves.
            const dnnl::memory::desc src({ B, M, K }, dt::f16, { M * K, K, 1 });
            const dnnl::memory::desc wei({ B, K, N }, dt::f16, { K * N, 1, K });
            const dnnl::memory::desc dst({ B, M, N }, dt::f32, { M * N, N, 1 });
            dnnl::primitive_attr     attr;  // make_pd's attribute, minus set_scales
            ggml_sycl_dnnl_attr_set_reproducible(attr);
            attr.set_fpmath_mode(dnnl::fpmath_mode::f16, /*apply_to_int=*/true);
            attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);
            pd     = dnnl::matmul::primitive_desc(eng, src, wei, dst, attr);
            a.prim = dnnl::matmul(pd);
            a.execs.push_back({
                { DNNL_ARG_SRC,     dnnl::memory(src, eng, pt.src.ptr) },
                { DNNL_ARG_WEIGHTS, dnnl::memory(wei, eng, pt.w16.ptr) },
                { DNNL_ARG_DST,     dnnl::memory(dst, eng, pt.dst.ptr) },
            });
        } else {
            const bool         f16s  = bench_arm_f16_scales(arm);
            const bool         two_d = bench_arm_2d(arm);
            const woq_form     f     = bench_woq_form(arm);
            const memory_descs d     = bench_woq_descs(arm, pt, f);
            const bool         diag  = arm == BENCH_ARM_WOQ_3D_BA_DIAG;
            pd                       = make_pd(eng, d, f);
            a.prim                   = dnnl::matmul(pd);
            uint8_t *    w           = static_cast<uint8_t *>(diag ? pt.nib_nk.ptr : pt.nib.ptr);
            uint8_t *    src         = static_cast<uint8_t *>(pt.src.ptr);
            uint8_t *    s           = static_cast<uint8_t *>(f16s ? pt.s16.ptr : pt.s8.ptr);
            uint8_t *    dst         = static_cast<uint8_t *>(pt.dst.ptr);
            // The 2-D forms run once per expert, each on its expert's slice of every buffer.
            const int    n_exec      = two_d ? pt.B : 1;
            const size_t src_step    = (size_t) M * K * sizeof(uint16_t);
            const size_t w_step      = (size_t) K * N / 2;
            const size_t s_step      = (size_t) pt.G * N * (f16s ? sizeof(uint16_t) : 1);
            const size_t dst_step    = (size_t) M * N * sizeof(float);
            const int    scales_arg  = DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS;
            for (int e = 0; e < n_exec; ++e) {
                a.execs.push_back({
                    { DNNL_ARG_SRC,     dnnl::memory(d.src,     eng, src + e * src_step) },
                    { DNNL_ARG_WEIGHTS, dnnl::memory(d.weights, eng, w + e * w_step)     },
                    { DNNL_ARG_DST,     dnnl::memory(d.dst,     eng, dst + e * dst_step) },
                    { scales_arg,       dnnl::memory(d.scales,  eng, s + e * s_step)     },
                });
            }
        }
        a.impl = one_token(pd.impl_info_str() ? pd.impl_info_str() : "?");
        a.ok   = true;
    } catch (const dnnl::error & e) {
        a.ok   = false;
        a.impl = one_token(std::string("refused:") + e.what());
    }
    return a;
}

// Runs one rep of a oneDNN arm into a NaN-filled dst and reads dst back, so an output the arm never writes fails.
std::vector<float> onednn_arm_read(onednn_arm & a, dnnl::stream & stream, sycl::queue & q, const bench_point & pt) {
    q.memset(pt.dst.ptr, 0xff, pt.dst_bytes);
    onednn_arm_rep(a, stream);
    q.wait_and_throw();
    std::vector<float> out(pt.dst_bytes / sizeof(float));
    q.memcpy(out.data(), pt.dst.ptr, pt.dst_bytes).wait_and_throw();
    return out;
}

void report_numerics(int arm, const bench_shape & sh, const bench_point & pt, int verdict, const std::string & detail) {
    std::printf("NUMERICS %s shape=%s B=%d M=%d K=%d N=%d %s %s\n", k_bench_arm_names[arm], sh.name, pt.B, pt.M, pt.K,
                pt.N, verdict == BENCH_VERDICT_PASS ? "PASS" : "FAIL", detail.c_str());
    std::fflush(stdout);
}

int report_column_check(int                        arm,
                        const bench_shape &        sh,
                        const bench_point &        pt,
                        const woq_problem &        chk,
                        const std::vector<float> & out,
                        const std::string &        impl) {
    const column_check c = check_columns(chk, out);
    char               buf[192];
    std::snprintf(buf, sizeof(buf), "worst_rel=%.3g (col %d) nonfinite=%d nmse=%.3g impl=", c.worst_rel, c.worst_col,
                  c.nonfinite, c.nmse);
    std::string detail = buf + impl;
    if (!c.class_worst.empty()) {
        detail += " class_worst=" + c.class_worst;
    }
    const int v = c.ok ? BENCH_VERDICT_PASS : BENCH_VERDICT_BADNUM;
    report_numerics(arm, sh, pt, v, detail);
    return v;
}

// The numerics check, once per shape at B 8, M 32 on make_nvfp4_problem's bytes (pt holds chk.stored, chk.src and the
// host-built diagnostic plane). The two kernels are compared bit for bit with the host's planes; each oneDNN arm runs
// on the planes the kernels wrote and is held to check_columns against the double reference, which decodes the stored
// blocks independently of both kernels. f16-gemm's check therefore covers f16-dequant's output as well.
void bench_check(sycl::queue &        q,
                 const dnnl::engine & eng,
                 dnnl::stream &       stream,
                 const bench_shape &  sh,
                 bench_point &        pt,
                 const woq_problem &  chk,
                 const woq_problem &  chk_f16,
                 int                  verdict[BENCH_ARM_COUNT]) {
    const size_t plane  = (size_t) pt.K * pt.N / 2;
    const size_t splane = (size_t) pt.G * pt.N;

    // f16-dequant against the host decode of make_nvfp4_problem's planes.
    try {
        q.memset(pt.w16.ptr, 0xff, pt.w16_bytes);
        launch_dequant(q, static_cast<const uint8_t *>(pt.stored.ptr), static_cast<uint32_t *>(pt.w16.ptr), pt.B, pt.K,
                       pt.N);
        q.wait_and_throw();
        std::vector<uint16_t> got(pt.w16_bytes / sizeof(uint16_t));
        q.memcpy(got.data(), pt.w16.ptr, pt.w16_bytes).wait_and_throw();
        size_t bad   = 0;
        size_t first = SIZE_MAX;
        for (int b = 0; b < pt.B; ++b) {
            for (int n = 0; n < pt.N; ++n) {
                for (int k = 0; k < pt.K; ++k) {
                    const size_t  kn   = (size_t) k * pt.N + n;
                    const uint8_t byte = chk.weights[plane * b + kn / 2];
                    const uint8_t c    = (kn & 1) ? (uint8_t) (byte >> 4) : (uint8_t) (byte & 0xf);
                    const uint8_t sc   = chk.scales[splane * b + (size_t) (k / k_nvfp4_sub) * pt.N + n];
                    const size_t  i    = ((size_t) b * pt.N + n) * pt.K + k;
                    if (got[i] != f16_bits((float) k_kvalues_doubled[c] * ue4m3_half_sycl(sc))) {
                        first = bad++ == 0 ? i : first;
                    }
                }
            }
        }
        char buf[128];
        std::snprintf(buf, sizeof(buf), "mismatched_halves=%zu of %zu first=%lld impl=kernel:nvfp4_bench_dequant", bad,
                      got.size(), bad ? (long long) first : -1LL);
        verdict[BENCH_ARM_F16_DEQUANT] = bad == 0 ? BENCH_VERDICT_PASS : BENCH_VERDICT_BADNUM;
        report_numerics(BENCH_ARM_F16_DEQUANT, sh, pt, verdict[BENCH_ARM_F16_DEQUANT], buf);
    } catch (const sycl::exception & e) {
        verdict[BENCH_ARM_F16_DEQUANT] = BENCH_VERDICT_FAIL;
        report_numerics(BENCH_ARM_F16_DEQUANT, sh, pt, BENCH_VERDICT_FAIL, one_token(std::string("sycl:") + e.what()));
    }

    // woq-repack against make_nvfp4_problem's planes and with_f16_scales' scales, byte for byte.
    try {
        q.memset(pt.nib.ptr, 0xff, pt.nib_bytes);
        q.memset(pt.s8.ptr, 0xff, pt.s8_bytes);
        q.memset(pt.s16.ptr, 0xff, pt.s8_bytes * 2);
        launch_repack(q, static_cast<const uint8_t *>(pt.stored.ptr), static_cast<uint8_t *>(pt.nib.ptr),
                      static_cast<uint8_t *>(pt.s8.ptr), static_cast<uint16_t *>(pt.s16.ptr), pt.B, pt.K, pt.N);
        q.wait_and_throw();
        std::vector<uint8_t> nib(pt.nib_bytes);
        std::vector<uint8_t> s8(pt.s8_bytes);
        std::vector<uint8_t> s16(pt.s8_bytes * 2);
        q.memcpy(nib.data(), pt.nib.ptr, nib.size());
        q.memcpy(s8.data(), pt.s8.ptr, s8.size());
        q.memcpy(s16.data(), pt.s16.ptr, s16.size());
        q.wait_and_throw();
        const auto count_bad = [](const std::vector<uint8_t> & got, const std::vector<uint8_t> & want) {
            size_t bad = got.size() == want.size() ? 0 : SIZE_MAX;
            for (size_t i = 0; bad != SIZE_MAX && i < got.size(); ++i) {
                bad += got[i] != want[i];
            }
            return bad;
        };
        const size_t bad_nib = count_bad(nib, chk.weights);
        const size_t bad_s8  = count_bad(s8, chk.scales);
        const size_t bad_s16 = count_bad(s16, chk_f16.scales);
        char         buf[160];
        std::snprintf(buf, sizeof(buf),
                      "mismatched_bytes nibbles=%zu e4m3_scales=%zu f16_scales=%zu impl=kernel:nvfp4_bench_repack",
                      bad_nib, bad_s8, bad_s16);
        const bool ok                 = bad_nib == 0 && bad_s8 == 0 && bad_s16 == 0;
        verdict[BENCH_ARM_WOQ_REPACK] = ok ? BENCH_VERDICT_PASS : BENCH_VERDICT_BADNUM;
        report_numerics(BENCH_ARM_WOQ_REPACK, sh, pt, verdict[BENCH_ARM_WOQ_REPACK], buf);
    } catch (const sycl::exception & e) {
        verdict[BENCH_ARM_WOQ_REPACK] = BENCH_VERDICT_FAIL;
        report_numerics(BENCH_ARM_WOQ_REPACK, sh, pt, BENCH_VERDICT_FAIL, one_token(std::string("sycl:") + e.what()));
    }

    // The oneDNN arms, on what the kernels just wrote (and the host-built diagnostic plane).
    for (int arm = BENCH_ARM_F16_GEMM; arm < BENCH_ARM_COUNT; ++arm) {
        if (arm == BENCH_ARM_WOQ_REPACK) {
            continue;
        }
        onednn_arm a = build_onednn_arm(arm, eng, pt);
        if (!a.ok) {
            verdict[arm] = BENCH_VERDICT_FAIL;
            report_numerics(arm, sh, pt, BENCH_VERDICT_FAIL, "impl=" + a.impl);
            continue;
        }
        try {
            verdict[arm] = report_column_check(arm, sh, pt, chk, onednn_arm_read(a, stream, q, pt), a.impl);
        } catch (const dnnl::error & e) {
            verdict[arm] = BENCH_VERDICT_FAIL;
            report_numerics(arm, sh, pt, BENCH_VERDICT_FAIL, one_token("impl=" + a.impl + " execute:" + e.what()));
        } catch (const sycl::exception & e) {
            verdict[arm] = BENCH_VERDICT_FAIL;
            report_numerics(arm, sh, pt, BENCH_VERDICT_FAIL, one_token("impl=" + a.impl + " sycl:" + e.what()));
        }
    }
}

void report_bench(int                  arm,
                  const bench_shape &  sh,
                  const bench_point &  pt,
                  const bench_timing & t,
                  int                  verdict,
                  const std::string &  impl) {
    // numerics= is always the check's verdict; a timing that did not produce a figure says why in timing=.
    const double gbps   = t.ok ? bench_arm_bytes(arm, pt) / t.dev_us / 1e3 : NAN;
    const char * timing = t.ok ? "" : t.bad_span ? " timing=BADSPAN" : " timing=FAIL";
    std::printf(
        "BENCH %s shape=%s B=%d M=%d K=%d N=%d dev_us_per_expert=%.2f wall_us_per_expert=%.2f GBps=%.1f numerics=%s "
        "impl=%s%s\n",
        k_bench_arm_names[arm], sh.name, pt.B, pt.M, pt.K, pt.N, t.dev_us, t.wall_us, gbps, bench_verdict_str(verdict),
        impl.c_str(), timing);
    if (t.ok && t.dev_us > t.wall_us) {
        std::printf("WARN %s shape=%s B=%d M=%d: device time exceeds wall time, the timestamps are not trustworthy\n",
                    k_bench_arm_names[arm], sh.name, pt.B, pt.M);
    }
    std::fflush(stdout);
}

struct bench_result {
    int          shape;
    int          B;
    int          M;
    bench_timing t[BENCH_ARM_COUNT];
};

// Times every arm at one grid point. The planes the WOQ arms read are the ones the repack arm writes here.
void bench_time_point(sycl::queue &        q,
                      const dnnl::engine & eng,
                      dnnl::stream &       stream,
                      const bench_shape &  sh,
                      bench_point &        pt,
                      const int            verdict[BENCH_ARM_COUNT],
                      bench_result &       res) {
    const uint8_t * stored = static_cast<const uint8_t *>(pt.stored.ptr);
    for (int arm = 0; arm < BENCH_ARM_COUNT; ++arm) {
        bench_timing & t = res.t[arm];
        std::string    impl;
        // Declared outside the try, so a throw mid-timing drains the queue (below) before the primitive and its memory
        // objects are destroyed at the end of this iteration.
        onednn_arm     a;
        try {
            if (arm == BENCH_ARM_F16_DEQUANT) {
                impl = "kernel:nvfp4_bench_dequant";
                t    = time_arm(q, pt.B, [&]() {
                    launch_dequant(q, stored, static_cast<uint32_t *>(pt.w16.ptr), pt.B, pt.K, pt.N);
                });
            } else if (arm == BENCH_ARM_WOQ_REPACK) {
                impl = "kernel:nvfp4_bench_repack";
                t    = time_arm(q, pt.B, [&]() {
                    launch_repack(q, stored, static_cast<uint8_t *>(pt.nib.ptr), static_cast<uint8_t *>(pt.s8.ptr),
                                     static_cast<uint16_t *>(pt.s16.ptr), pt.B, pt.K, pt.N);
                });
            } else {
                a    = build_onednn_arm(arm, eng, pt);
                impl = a.impl;
                if (a.ok) {
                    t = time_arm(q, pt.B, [&]() { onednn_arm_rep(a, stream); });
                }
            }
        } catch (const dnnl::error & e) {
            queue_drain_nothrow(q);
            t    = bench_timing();
            impl = one_token(impl + " execute:" + e.what());
        } catch (const sycl::exception & e) {
            queue_drain_nothrow(q);
            t    = bench_timing();
            impl = one_token(impl + " sycl:" + e.what());
        }
        report_bench(arm, sh, pt, t, verdict[arm], impl);
    }
}

// Copies the 8 check experts' bytes into a point's B experts: whole groups of 8 for B a multiple of 8, or the first B
// experts for B below 8 (the B 1 XCHECK point). Any other B would leave a partial group, so it is refused.
void upload_tiled(sycl::queue & q, void * dst, const std::vector<uint8_t> & experts8, int B) {
    if (B <= 0 || (B >= k_bench_check_b && B % k_bench_check_b != 0) || experts8.size() % k_bench_check_b != 0) {
        throw std::runtime_error("upload_tiled: B=" + std::to_string(B) + " is neither below 8 nor a multiple of 8");
    }
    const size_t expert_bytes = experts8.size() / k_bench_check_b;
    for (int e = 0; e < B; e += k_bench_check_b) {
        const int n = std::min(B - e, k_bench_check_b);
        q.memcpy(static_cast<uint8_t *>(dst) + (size_t) e * expert_bytes, experts8.data(), (size_t) n * expert_bytes);
    }
}

// Per column (and expert), the largest |out - ref| over its rows divided by the column's largest |ref|; a column whose
// ref is exactly 0 must read exactly 0. Entries where either side is not finite are left out, and the output's are
// counted (the caller counts the reference's). A column whose ref and out are both exactly 0 on every row, with no
// entry left out, is a zero column.
struct xcheck_result {
    double    worst_rel     = NAN;
    int       nonfinite     = -1;  // the WOQ arm's output; -1: not compared
    int       ref_nonfinite = -1;  // f16-gemm's output; -1: not read
    long long zero_cols     = -1;
};

xcheck_result xcheck_columns(const std::vector<float> & ref, const std::vector<float> & out, const bench_point & pt) {
    xcheck_result       r;
    std::vector<double> max_ref(pt.N);
    std::vector<double> max_out(pt.N);
    std::vector<double> max_err(pt.N);
    std::vector<char>   skipped(pt.N);
    r.worst_rel = 0.0;
    r.nonfinite = 0;
    r.zero_cols = 0;
    for (int b = 0; b < pt.B; ++b) {
        std::fill(max_ref.begin(), max_ref.end(), 0.0);
        std::fill(max_out.begin(), max_out.end(), 0.0);
        std::fill(max_err.begin(), max_err.end(), 0.0);
        std::fill(skipped.begin(), skipped.end(), 0);
        for (int m = 0; m < pt.M; ++m) {
            const size_t row = ((size_t) b * pt.M + m) * pt.N;
            for (int n = 0; n < pt.N; ++n) {
                const double o = out[row + n];
                const double f = ref[row + n];
                r.nonfinite += !std::isfinite(o);
                if (!std::isfinite(o) || !std::isfinite(f)) {
                    skipped[n] = 1;
                    continue;
                }
                max_ref[n] = std::max(max_ref[n], std::fabs(f));
                max_out[n] = std::max(max_out[n], std::fabs(o));
                max_err[n] = std::max(max_err[n], std::fabs(o - f));
            }
        }
        for (int n = 0; n < pt.N; ++n) {
            const double rel = max_ref[n] == 0.0 ? (max_err[n] == 0.0 ? 0.0 : INFINITY) : max_err[n] / max_ref[n];
            r.worst_rel      = std::max(r.worst_rel, rel);
            r.zero_cols += !skipped[n] && max_ref[n] == 0.0 && max_out[n] == 0.0;
        }
    }
    return r;
}

struct xcheck_tally {
    int pass = 0;
    int fail = 0;
};

// The scale classes whose every byte decodes to 0 (0x00, and 0x7f and 0xff, read as 0), so a column drawn from one
// has all-zero weights and an all-zero output.
bool scale_class_reads_zero(int cls) {
    return cls == SCALE_CLASS_ZERO || cls == SCALE_CLASS_NAN_PATTERN || cls == SCALE_CLASS_SIGNED_NAN_PATTERN;
}

// The zero columns a point's B experts hold: every expert repeats one of the 8 checked experts, whose columns all take
// their scale class from col_class.
long long xcheck_expected_zero_cols(const std::vector<int> & col_class, const bench_point & pt) {
    long long per_expert = 0;
    for (int n = 0; n < pt.N; ++n) {
        per_expert += scale_class_reads_zero(col_class[n]);
    }
    return per_expert * pt.B;
}

// One XCHECK line (after an XCHECK-ERROR line naming what could not run, if anything). More zero columns than the
// class table holds means some output was never written, so the line fails.
void report_xcheck(int                   arm,
                   const bench_shape &   sh,
                   const bench_point &   pt,
                   const xcheck_result & r,
                   long long             expected_zero_cols,
                   const std::string &   error,
                   const std::string &   impl,
                   xcheck_tally &        tally) {
    if (!error.empty()) {
        std::printf("XCHECK-ERROR %s shape=%s B=%d M=%d %s\n", k_bench_arm_names[arm], sh.name, pt.B, pt.M,
                    error.c_str());
    }
    if (r.ref_nonfinite > 0) {
        std::printf("XCHECK-ERROR %s shape=%s B=%d M=%d f16-gemm:nonfinite\n", k_bench_arm_names[arm], sh.name, pt.B,
                    pt.M);
    }
    const bool pass = error.empty() && r.nonfinite == 0 && r.ref_nonfinite == 0 && r.worst_rel <= k_xcheck_tol &&
                      r.zero_cols >= 0 && r.zero_cols <= expected_zero_cols;
    std::printf(
        "XCHECK %s shape=%s B=%d M=%d worst_rel=%.2e nonfinite=%d ref_nonfinite=%d zero_cols=%lld of %lld %s "
        "impl=%s\n",
        k_bench_arm_names[arm], sh.name, pt.B, pt.M, r.worst_rel, r.nonfinite, r.ref_nonfinite, r.zero_cols,
        (long long) pt.B * pt.N, pass ? "PASS" : "FAIL", impl.c_str());
    std::fflush(stdout);
    (pass ? tally.pass : tally.fail)++;
}

// Cross-checks the two 3-D WOQ forms Task 26b chooses between against f16-gemm at one point. All three read the 8
// checked experts repeated and the same src, so every column must agree with f16-gemm's. The dequant and the repack
// run first, into poisoned buffers (NaN halves, 0xff planes, whose scales read as NaN), so a point that was not timed
// has its planes too and a region either kernel skips shows as non-finite output. Then each arm runs once into a
// NaN-filled dst that is read back before the next arm runs, so no arm's output stands in for another's.
void bench_xcheck_point(sycl::queue &            q,
                        const dnnl::engine &     eng,
                        dnnl::stream &           stream,
                        const bench_shape &      sh,
                        bench_point &            pt,
                        const std::vector<int> & col_class,
                        xcheck_tally &           tally) {
    const long long    expected_zero_cols = xcheck_expected_zero_cols(col_class, pt);
    std::vector<float> ref;
    int                ref_nonfinite = -1;
    std::string        ref_error;
    {
        onednn_arm g;
        try {
            q.memset(pt.w16.ptr, 0xff, pt.w16_bytes);
            q.memset(pt.nib.ptr, 0xff, pt.nib_bytes);
            q.memset(pt.s8.ptr, 0xff, pt.s8_bytes);
            q.memset(pt.s16.ptr, 0xff, pt.s8_bytes * 2);
            launch_dequant(q, static_cast<const uint8_t *>(pt.stored.ptr), static_cast<uint32_t *>(pt.w16.ptr), pt.B,
                           pt.K, pt.N);
            launch_repack(q, static_cast<const uint8_t *>(pt.stored.ptr), static_cast<uint8_t *>(pt.nib.ptr),
                          static_cast<uint8_t *>(pt.s8.ptr), static_cast<uint16_t *>(pt.s16.ptr), pt.B, pt.K, pt.N);
            q.wait_and_throw();
            g = build_onednn_arm(BENCH_ARM_F16_GEMM, eng, pt);
            if (g.ok) {
                ref           = onednn_arm_read(g, stream, q, pt);
                ref_nonfinite = (int) std::count_if(ref.begin(), ref.end(), [](float v) { return !std::isfinite(v); });
            } else {
                ref_error = "f16-gemm:" + g.impl;
            }
        } catch (const dnnl::error & e) {
            queue_drain_nothrow(q);
            ref_error = one_token(std::string("f16-gemm:execute:") + e.what());
        } catch (const sycl::exception & e) {
            queue_drain_nothrow(q);
            ref_error = one_token(std::string("f16-gemm:sycl:") + e.what());
        }
    }
    for (const int arm : k_xcheck_arms) {
        xcheck_result r;
        std::string   error = ref_error;
        std::string   impl  = "-";
        if (error.empty()) {
            onednn_arm a;
            try {
                a    = build_onednn_arm(arm, eng, pt);
                impl = a.impl;
                if (a.ok) {
                    r = xcheck_columns(ref, onednn_arm_read(a, stream, q, pt), pt);
                } else {
                    error = a.impl;
                }
            } catch (const dnnl::error & e) {
                queue_drain_nothrow(q);
                error = one_token(std::string("execute:") + e.what());
            } catch (const sycl::exception & e) {
                queue_drain_nothrow(q);
                error = one_token(std::string("sycl:") + e.what());
            }
        }
        r.ref_nonfinite = ref_nonfinite;
        report_xcheck(arm, sh, pt, r, expected_zero_cols, error, impl, tally);
    }
}

// A point whose buffers could not be allocated still prints its two XCHECK lines, as failures.
void bench_xcheck_unallocated(const bench_shape & sh, const bench_point & pt, xcheck_tally & tally) {
    for (const int arm : k_xcheck_arms) {
        report_xcheck(arm, sh, pt, xcheck_result(), 0, "device-allocation-failed", "-", tally);
    }
}

// The diagnostic arm's primitive_desc in user scratchpad mode, production's: its scratchpad size and implementation.
// Built, not executed.
void bench_report_scratchpad(const dnnl::engine & eng, const bench_shape & sh, const bench_point & pt) {
    long long   user_bytes = -1;
    std::string impl;
    try {
        const woq_form                     f = bench_woq_form(BENCH_ARM_WOQ_3D_BA_DIAG);
        const dnnl::matmul::primitive_desc pd =
            make_pd(eng, bench_woq_descs(BENCH_ARM_WOQ_3D_BA_DIAG, pt, f), f, dnnl::scratchpad_mode::user);
        user_bytes = (long long) pd.scratchpad_desc().get_size();
        impl       = one_token(pd.impl_info_str() ? pd.impl_info_str() : "?");
    } catch (const dnnl::error & e) {
        impl = one_token(std::string("refused:") + e.what());
    }
    std::printf("SCRATCHPAD %s shape=%s B=%d M=%d user_bytes=%lld impl=%s\n",
                k_bench_arm_names[BENCH_ARM_WOQ_3D_BA_DIAG], sh.name, pt.B, pt.M, user_bytes, impl.c_str());
    std::fflush(stdout);
}

// Uploads a point's inputs: the 8 checked experts' stored bytes and diagnostic plane repeated over B, and src.
void bench_upload_point(sycl::queue &                q,
                        bench_point &                pt,
                        const std::vector<uint8_t> & stored8,
                        const std::vector<uint8_t> & nk8,
                        const uint16_t *             src) {
    upload_tiled(q, pt.stored.ptr, stored8, pt.B);
    upload_tiled(q, pt.nib_nk.ptr, nk8, pt.B);
    q.memcpy(pt.src.ptr, src, pt.src_bytes);
    q.wait_and_throw();
}

int run_bench() {
    std::unique_ptr<sycl::queue> q;
    try {
        q = std::make_unique<sycl::queue>(
            sycl::gpu_selector_v,
            sycl::property_list{ sycl::property::queue::in_order(), sycl::property::queue::enable_profiling() });
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
        std::printf("[bench] device=%s oneDNN=%d.%d.%d\n", q->get_device().get_info<sycl::info::device::name>().c_str(),
                    v->major, v->minor, v->patch);
        std::fflush(stdout);
        if (q->get_device().get_info<sycl::info::device::max_work_group_size>() < (size_t) k_bench_wg) {
            std::printf("BENCH harness FAIL the device's max work-group size is below %d\n", k_bench_wg);
            return 1;
        }
        // The kernels' decode must be the reference's, bit for bit, on every byte.
        for (int x = 0; x < 256; ++x) {
            if (sycl::bit_cast<uint32_t>(ue4m3_half_device((uint8_t) x)) !=
                sycl::bit_cast<uint32_t>(ue4m3_half_sycl((uint8_t) x))) {
                std::printf("BENCH harness FAIL ue4m3_half_device(0x%02x) differs from ue4m3_half_sycl\n", x);
                return 1;
            }
        }

        const bench_shape shapes[] = {
            { "gate_up", 2304, 2304, 0x475550u },
            { "down",    1152, 2304, 0x444f57u },
        };
        constexpr int n_shapes = (int) (sizeof(shapes) / sizeof(shapes[0]));

        // Every B and M any point uses: the grid and the XCHECK-only points.
        std::vector<int> all_b(std::begin(k_bench_bs), std::end(k_bench_bs));
        std::vector<int> all_m(std::begin(k_bench_ms), std::end(k_bench_ms));
        for (const bench_extent & x : k_xcheck_extra) {
            all_b.push_back(x.B);
            all_m.push_back(x.M);
        }
        const int max_b = *std::max_element(all_b.begin(), all_b.end());
        const int max_m = *std::max_element(all_m.begin(), all_m.end());

        // The kernels have no tails: the repack's tiles need K and N in multiples of 64, and the dequant's nd_range
        // needs B * N * K / 2 in whole work-groups at every B.
        for (const bench_shape & sh : shapes) {
            if (sh.K % k_nvfp4_block != 0 || sh.N % k_bench_tile != 0) {
                std::printf("BENCH harness FAIL shape %s: K=%d and N=%d must be multiples of %d\n", sh.name, sh.K, sh.N,
                            k_bench_tile);
                return 1;
            }
            for (const int B : all_b) {
                if ((size_t) B * sh.N * (sh.K / 2) % k_bench_wg != 0) {
                    std::printf("BENCH harness FAIL shape %s B=%d: B*N*K/2 is not a multiple of %d\n", sh.name, B,
                                k_bench_wg);
                    return 1;
                }
            }
        }

        dnnl::engine eng    = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
        dnnl::stream stream = dnnl::sycl_interop::make_stream(eng, *q);

        int                       verdicts[n_shapes][BENCH_ARM_COUNT];
        std::vector<bench_result> results;
        xcheck_tally              tally;
        for (int si = 0; si < n_shapes; ++si) {
            const bench_shape & sh      = shapes[si];
            const woq_problem   chk     = make_nvfp4_problem(k_bench_check_b, k_bench_check_m, sh.K, sh.N, sh.seed);
            const woq_problem   chk_f16 = with_f16_scales(chk);

            // The diagnostic arm's plane: chk's nibbles with (b, k, n) at b * K * N + n * K + k.
            const size_t         plane = (size_t) sh.K * sh.N / 2;
            std::vector<uint8_t> nk8(chk.weights.size(), 0);
            for (int b = 0; b < k_bench_check_b; ++b) {
                for (int k = 0; k < sh.K; ++k) {
                    for (int n = 0; n < sh.N; ++n) {
                        const size_t  kn   = (size_t) k * sh.N + n;
                        const uint8_t byte = chk.weights[plane * b + kn / 2];
                        put_nibble(nk8, plane * b, (size_t) n * sh.K + k,
                                   (kn & 1) ? (uint8_t) (byte >> 4) : (uint8_t) (byte & 0xf));
                    }
                }
            }
            // Activations for every point but the check point; their weights repeat chk's 8 experts.
            std::mt19937                          rng(sh.seed ^ 0x5a5a5a5au);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::vector<uint16_t>                 src_all((size_t) max_b * max_m * sh.K);
            for (uint16_t & h : src_all) {
                h = f16_bits(dist(rng));
            }

            // Verdicts start as FAIL; the check at the grid's first point (static_assert above) sets them.
            int * verdict = verdicts[si];
            std::fill(verdict, verdict + BENCH_ARM_COUNT, (int) BENCH_VERDICT_FAIL);
            for (const int B : k_bench_bs) {
                for (const int M : k_bench_ms) {
                    bench_result res = { si, B, M, {} };
                    bench_point  pt(*q, B, M, sh.K, sh.N);
                    if (!pt.allocated()) {
                        for (int arm = 0; arm < BENCH_ARM_COUNT; ++arm) {
                            report_bench(arm, sh, pt, bench_timing(), BENCH_VERDICT_FAIL, "device-allocation-failed");
                        }
                        bench_xcheck_unallocated(sh, pt, tally);
                        bench_report_scratchpad(eng, sh, pt);
                        results.push_back(res);
                        continue;
                    }
                    const bool check = B == k_bench_check_b && M == k_bench_check_m;
                    bench_upload_point(*q, pt, chk.stored, nk8, check ? chk.src.data() : src_all.data());
                    if (check) {
                        bench_check(*q, eng, stream, sh, pt, chk, chk_f16, verdict);
                    }
                    bench_time_point(*q, eng, stream, sh, pt, verdict, res);
                    bench_xcheck_point(*q, eng, stream, sh, pt, chk.col_class, tally);
                    bench_report_scratchpad(eng, sh, pt);
                    results.push_back(res);
                }
            }

            // XCHECK-only points: no BENCH line and no timing, so the BENCH and PIPELINE counts stay 144 and 18.
            for (const bench_extent & x : k_xcheck_extra) {
                bench_point pt(*q, x.B, x.M, sh.K, sh.N);
                if (!pt.allocated()) {
                    bench_xcheck_unallocated(sh, pt, tally);
                    bench_report_scratchpad(eng, sh, pt);
                    continue;
                }
                bench_upload_point(*q, pt, chk.stored, nk8, src_all.data());
                bench_xcheck_point(*q, eng, stream, sh, pt, chk.col_class, tally);
                bench_report_scratchpad(eng, sh, pt);
            }
        }

        // f16 = dequant + gemm; woq_best = repack + the fastest WOQ form whose check passed at this shape. woq-repack
        // writes both scale planes, so woq_best pays for a plane its form does not read: about 10% of the repack's
        // bytes when woq_best is an e4m3 form, about 5% when it is an f16-scale form.
        for (const bench_result & r : results) {
            const bench_timing & dq   = r.t[BENCH_ARM_F16_DEQUANT];
            const bench_timing & gemm = r.t[BENCH_ARM_F16_GEMM];
            const bench_timing & rp   = r.t[BENCH_ARM_WOQ_REPACK];
            const double         f16  = dq.ok && gemm.ok ? dq.dev_us + gemm.dev_us : NAN;
            int                  best = -1;
            for (const int arm : k_bench_woq_forms) {
                if (verdicts[r.shape][arm] == BENCH_VERDICT_PASS && r.t[arm].ok &&
                    (best < 0 || r.t[arm].dev_us < r.t[best].dev_us)) {
                    best = arm;
                }
            }
            const double woq         = best >= 0 && rp.ok ? rp.dev_us + r.t[best].dev_us : NAN;
            // The ratio is a number only when both pipelines' inputs passed and both sums are finite times.
            const int *  v           = verdicts[r.shape];
            const bool   inputs_pass = v[BENCH_ARM_F16_DEQUANT] == BENCH_VERDICT_PASS &&
                                     v[BENCH_ARM_F16_GEMM] == BENCH_VERDICT_PASS &&
                                     v[BENCH_ARM_WOQ_REPACK] == BENCH_VERDICT_PASS;
            char ratio[32];
            if (!inputs_pass) {
                std::snprintf(ratio, sizeof(ratio), "BADNUM");
            } else if (!std::isfinite(woq) || !std::isfinite(f16) || f16 <= 0.0) {
                std::snprintf(ratio, sizeof(ratio), "nan");
            } else {
                std::snprintf(ratio, sizeof(ratio), "%.3f", woq / f16);
            }
            std::printf("PIPELINE shape=%s B=%d M=%d f16=%.2f woq_best=%.2f form=%s ratio=%s\n", shapes[r.shape].name,
                        r.B, r.M, f16, woq, best >= 0 ? k_bench_arm_names[best] : "none", ratio);
        }
        std::printf("XCHECK_SUMMARY pass=%d fail=%d\n", tally.pass, tally.fail);
        std::fflush(stdout);
    } catch (const std::exception & e) {
        std::printf("BENCH harness FAIL %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        return run_bench();
    }
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

        const woq_form nvfp4_2d_f16 = {
            dt::f16, k_nvfp4_sub, (1 << 0) | (1 << 1), { k_nvfp4_sub, 1 }
        };
        const woq_form nvfp4_3d_f16 = {
            dt::f16, k_nvfp4_sub, (1 << 0) | (1 << 1) | (1 << 2), { k_nvfp4_sub, 1 }
        };
        run_numerics_arm("nvfp4-2d-f16-scales", *q, eng, stream, with_f16_scales(p2d), nvfp4_2d_f16);
        run_numerics_arm("nvfp4-3d-f16-scales", *q, eng, stream, with_f16_scales(p3d), nvfp4_3d_f16);
    } catch (const std::exception & e) {
        std::printf("PROBE harness FAIL %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
    return 0;
}
