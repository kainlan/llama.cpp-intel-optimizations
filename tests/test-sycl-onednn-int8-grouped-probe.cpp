// oneDNN int8-with-grouped-scales matmul probe + microbench (llama.cpp-ovkn,
// track prefill of epic llama.cpp-xihy).
//
// QUESTION THIS ANSWERS: can the oneDNN linked into this build (3.11.x,
// oneAPI 2026.0) run, ON THE GPU, a matmul whose SOURCE is int8 with per-row
// K/32-group scales (the Q8_1 activation format the mmq/mmvq kernels already
// use) and whose WEIGHTS are int8 (Q8_0) or int4 (Q4_0) with K/32-group scales
// -- i.e. an INT8-XMX GEMM that consumes the stored quantized layouts directly
// -- and if so, how fast is it against today's f16 x f16 ceiling
// (ggml-sycl.cpp's dequant-to-FP16 + DnnlGemmWrapper::row_gemm arm)? It also
// asks the same question for the MoE PP GEMM shape: can an int8 activation
// feed the existing f4_e2m1/e8m0 WOQ-MXFP4 weight path (V6), against the
// f16-activation control that is today's mxfp4.pp.gemm.execute arm (V7).
//
// It is deliberately STANDALONE: dnnl + sycl only, no ggml, no backend, no
// unified cache. Raw sycl::malloc_device here is a probe fixture, not a
// backend allocation (the unified-cache rule in CLAUDE.md governs backend
// code; this binary never links it). Nothing here is production code.
//
// Per variant and shape it prints ONE line:
//   variant=<id> shape=M<M>xN<N>xK<K> created=<Y|N> [reason=...]
//           [any_plain=<Y|N> any_bytes=<weights_desc size>] scratch=<bytes>
//           us=<avg wall per exec> ev_us=<last-kernel event>
//           tops=<2MNK/t> maxerr=<max abs err vs double ref> ref=<ref scale>
// and exits 77 when it cannot probe (no GPU / no oneDNN), 1 when a variant
// that CREATED produced numbers that do not match the reference (that is a
// real defect in either the probe or oneDNN and must not read as "unsupported"),
// 0 otherwise. No support answer is "wrong": this is a capability/perf probe,
// not a regression gate. Build:
//   ./scripts/sycl-build.sh test-sycl-onednn-int8-grouped-probe
// Run (lead only -- it touches the GPU; pin the selector, CLAUDE.md):
//   source /opt/intel/oneapi/setvars.sh --force
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-sycl-onednn-int8-grouped-probe
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-onednn-int8-grouped-probe
// Env: PROBE_ITERS (default 20), PROBE_SHAPES=small (only the 4096x4096 shape),
//      PROBE_VARIANTS=comma list of ids to restrict.
//
// V6/V7 e2m1 decode: kvalues_mxfp4 / e8m0_to_f32 below are copied from
// tests/test-onednn-woq-mxfp4.cpp (not included -- this binary must stay
// standalone) rather than shared, per llama.cpp-ovkn task comment c-idw3.
// dt::f4_e2m1 / dt::e8m0 are compile-time enumerators in this build's
// dnnl.hpp (verified: oneapi/dnnl/dnnl_common_types.h defines
// dnnl_f4_e2m1/dnnl_e8m0 unconditionally, no version guard), so no
// dtype-missing fallback is reachable here; if a future relink drops them,
// this file will fail to compile rather than silently skip -- update this
// comment and add a guard if that ever happens.

#if GGML_SYCL_DNNL
#    include <oneapi/dnnl/dnnl.hpp>
#    include <oneapi/dnnl/dnnl_sycl.hpp>
#    include <sycl/sycl.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr int EXIT_SKIP = 77;

#if GGML_SYCL_DNNL

namespace {

using dt  = dnnl::memory::data_type;
using dim = dnnl::memory::dim;

constexpr int64_t GROUP = 32;  // Q8_0 / Q8_1 / Q4_0 / MXFP4 block size

// Copied from tests/test-onednn-woq-mxfp4.cpp (llama.cpp-4nlr). Deliberately
// HALVED vs. ggml's own kvalues_mxfp4 (ggml-common.h:1126), paired with an
// un-halved e8m0_to_f32 -- algebraically identical to ggml's doubled-int
// table paired with its own e8m0_to_fp32_half. Not shared via #include: this
// binary must stay standalone (dnnl + sycl only, no ggml/backend deps).
const float kvalues_mxfp4[16] = { 0, .5f, 1, 1.5f, 2, 3, 4, 6, -0, -.5f, -1, -1.5f, -2, -3, -4, -6 };

// Mirrors ggml_e8m0_to_fp32 (ggml-impl.h), including its e==0 special case.
float e8m0_to_f32(uint8_t e) {
    union {
        uint32_t u;
        float    f;
    } v;

    v.u = (e == 0) ? 0x00400000u : ((uint32_t) e << 23);
    return v.f;
}

struct shape {
    int64_t m, n, k;
};

// Which side is quantized and how the scales are laid out.
enum class src_kind { f16, s8_grouped, s8_per_row };
enum class wei_kind { f16, s8_grouped, s4_grouped, s4_grouped_zp, e2m1_grouped };
enum class wei_scale_layout { plain, strided_soa };  // plain = [K/32][N]; strided_soa = stored [N][K/32]

struct variant {
    const char *     id;
    src_kind         src;
    wei_kind         wei;
    dt               src_scale_dt;
    dt               wei_scale_dt;
    wei_scale_layout wei_scales;
    bool             wei_any;  // ask for format_tag::any and report whether it stays plain
    bool             fpmath_f16_int;
};

const variant VARIANTS[] = {
    // V0: today's arm. f16 x f16 -> f32.
    { "V0_f16xf16",              src_kind::f16,        wei_kind::f16,           dt::undef, dt::undef, wei_scale_layout::plain, false, false },
    // V1: exact Q8_1 x Q8_0: s8 src grouped + s8 wei grouped, f16 scales (as stored) / f32 scales.
    { "V1_s8g_x_s8g_f16sc",      src_kind::s8_grouped, wei_kind::s8_grouped,    dt::f16,   dt::f16,   wei_scale_layout::plain,
     false,                                                                                                                           false },
    { "V1_s8g_x_s8g_f32sc",      src_kind::s8_grouped, wei_kind::s8_grouped,    dt::f32,   dt::f32,   wei_scale_layout::plain,
     false,                                                                                                                           false },
    // V1any: same, weights format_tag::any -- does oneDNN want a blocked (repacked) layout?
    { "V1any_s8g_x_s8g_f32sc",   src_kind::s8_grouped, wei_kind::s8_grouped,    dt::f32,   dt::f32,   wei_scale_layout::plain,
     true,                                                                                                                            false },
    // V5: V1 with weight scales strided the way Q8_0 SOA stores them ([N][K/32]).
    { "V5_s8g_x_s8g_soa_scales", src_kind::s8_grouped, wei_kind::s8_grouped,    dt::f16,   dt::f16,
     wei_scale_layout::strided_soa,                                                                                            false, false },
    // V2: per-row (per-token) src scale, grouped weights.
    { "V2_s8row_x_s8g",          src_kind::s8_per_row, wei_kind::s8_grouped,    dt::f32,   dt::f32,   wei_scale_layout::plain, false,
     false                                                                                                                                  },
    // V3: Q8_1 x Q4_0: s8 src grouped x s4 weights grouped (symmetric, no zp) and with s8 zp as the WoQ arm sets.
    { "V3_s8g_x_s4g",            src_kind::s8_grouped, wei_kind::s4_grouped,    dt::f32,   dt::f32,   wei_scale_layout::plain, false,
     false                                                                                                                                  },
    { "V3z_s8g_x_s4g_zp",        src_kind::s8_grouped, wei_kind::s4_grouped_zp, dt::f32,   dt::f32,   wei_scale_layout::plain,
     false,                                                                                                                           false },
    // V4: WoQ-int8 with f16 activations (FP16 compute; control for V1).
    { "V4_f16_x_s8g_fpmath",     src_kind::f16,        wei_kind::s8_grouped,    dt::undef, dt::f32,   wei_scale_layout::plain, false,
     true                                                                                                                                   },
    { "V4b_f16_x_s4g_fpmath",    src_kind::f16,        wei_kind::s4_grouped,    dt::undef, dt::f32,   wei_scale_layout::plain, false,
     true                                                                                                                                   },
    // V6: MoE PP GEMM shape -- s8 grouped src feeding the production
    // f4_e2m1/e8m0 WOQ-MXFP4 weight format (gemm.hpp's woq_gemm_batch_mxfp4
    // 2-D fallback layout: weights (K,N) strides {N,1}, scales (K/32,N)
    // strides {N,1}). No fpmath. Answers whether the MoE PP GEMM can take
    // int8 activations.
    { "V6_s8g_x_e2m1",           src_kind::s8_grouped, wei_kind::e2m1_grouped,  dt::f32,   dt::e8m0,  wei_scale_layout::plain, false,
     false                                                                                                                                  },
    // V7: f16 src x the same e2m1/e8m0 weights with fpmath f16 apply_to_int
    // -- today's mxfp4.pp.gemm.execute arm, the control for V6.
    { "V7_f16_x_e2m1_woq",       src_kind::f16,        wei_kind::e2m1_grouped,  dt::undef, dt::e8m0,  wei_scale_layout::plain, false,
     true                                                                                                                                   },
};

const shape SHAPES_FULL[] = {
    { 512,  4096,  4096  },
    { 512,  14336, 4096  },
    { 512,  4096,  14336 },
    { 512,  1024,  4096  },
    { 2048, 4096,  4096  },
};
const shape SHAPES_SMALL[] = {
    { 512, 4096, 4096 }
};

struct host_data {
    // logical A (M,K), B (K,N) stored [N][K]; scales as described per kind.
    std::vector<int8_t>   a_s8;   // [M][K]
    std::vector<float>    a_sc;   // [M][K/32] (grouped) or [M] (per row)
    std::vector<uint16_t> a_f16;  // [M][K]
    std::vector<int8_t>   b_s8;   // [N][K]
    std::vector<uint8_t>  b_s4;   // [N][K/2] packed nibbles, low nibble = even k
    std::vector<int8_t>   b_zp;   // [K/32][N] (only for zp variant)
    std::vector<float>    b_sc;   // [N][K/32] as stored (SOA order)
    std::vector<uint16_t> b_f16;  // [N][K]
    // e2m1_grouped only: logical (K,N) strides {N,1} -- offset(k,n)=k*N+n,
    // matching gemm.hpp's woq_gemm_batch_mxfp4 2-D fallback weight desc.
    std::vector<uint8_t>  b_nib;   // packed e2m1 codes, el=k*N+n, byte=el/2, even el = low nibble
    std::vector<uint8_t>  b_e8m0;  // e8m0 scale bytes, logical (K/32,N) strides {N,1}: (kb*N+n)
    // effective real values used by the reference (after f16 rounding where applicable)
    std::vector<float>    a_real;  // [M][K]
    std::vector<float>    b_real;  // [N][K]
};

uint16_t f32_to_f16_bits(float f) {
    return sycl::bit_cast<uint16_t>(sycl::half(f));
}

float f16_round(float f) {
    return static_cast<float>(sycl::half(f));
}

// Build one dataset per (variant, shape). Values are chosen so that every
// arm's REFERENCE is the same real-valued product of the effective A and B
// (after whatever rounding that arm's storage implies), so a mismatch is a
// compute defect, not a data-representation difference.
host_data make_data(const variant & v, const shape & s, std::mt19937 & rng) {
    host_data                             d;
    const int64_t                         M = s.m, N = s.n, K = s.k, KB = K / GROUP;
    std::uniform_int_distribution<int>    q8(-127, 127);
    std::uniform_int_distribution<int>    q4(-8, 7);
    std::uniform_real_distribution<float> sc(0.5f, 1.5f);

    d.a_real.assign(M * K, 0.f);
    d.b_real.assign(N * K, 0.f);

    // ---- A side
    if (v.src == src_kind::f16) {
        d.a_f16.resize(M * K);
        for (int64_t i = 0; i < M * K; ++i) {
            const float x = q8(rng) * 0.01f;
            d.a_f16[i]    = f32_to_f16_bits(x);
            d.a_real[i]   = f16_round(x);
        }
    } else {
        d.a_s8.resize(M * K);
        const bool grouped = v.src == src_kind::s8_grouped;
        d.a_sc.assign(grouped ? M * KB : M, 0.f);
        for (int64_t m = 0; m < M; ++m) {
            const float row_scale = sc(rng) * 0.01f;
            for (int64_t kb = 0; kb < KB; ++kb) {
                float g = grouped ? sc(rng) * 0.01f : row_scale;
                if (v.src_scale_dt == dt::f16) {
                    g = f16_round(g);
                }
                if (grouped) {
                    d.a_sc[m * KB + kb] = g;
                } else if (kb == 0) {
                    d.a_sc[m] = g;
                }
                for (int64_t j = 0; j < GROUP; ++j) {
                    const int64_t k     = kb * GROUP + j;
                    const int     q     = q8(rng);
                    d.a_s8[m * K + k]   = static_cast<int8_t>(q);
                    d.a_real[m * K + k] = q * g;
                }
            }
        }
    }

    // ---- B side
    if (v.wei == wei_kind::f16) {
        d.b_f16.resize(N * K);
        for (int64_t i = 0; i < N * K; ++i) {
            const float x = q8(rng) * 0.01f;
            d.b_f16[i]    = f32_to_f16_bits(x);
            d.b_real[i]   = f16_round(x);
        }
    } else if (v.wei == wei_kind::e2m1_grouped) {
        // Stored per the WOQ-order desc: weights (K,N) strides {N,1}, i.e.
        // offset(k,n)=k*N+n; scales (K/32,N) strides {N,1}: offset(kb,n)=kb*N+n.
        // This is gemm.hpp's woq_gemm_batch_mxfp4 2-D fallback layout, not
        // the Q8_0-SOA layout the s8/s4 branch below uses.
        d.b_nib.assign(N * K / 2, 0);
        d.b_e8m0.assign(KB * N, 0);
        std::uniform_int_distribution<int> nibsel(0, 15);
        std::uniform_int_distribution<int> e8sel(123, 127);  // small positive-exponent range
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t kb = 0; kb < KB; ++kb) {
                const uint8_t se     = static_cast<uint8_t>(e8sel(rng));
                d.b_e8m0[kb * N + n] = se;
                const float g        = e8m0_to_f32(se);
                for (int64_t j = 0; j < GROUP; ++j) {
                    const int64_t k   = kb * GROUP + j;
                    const int     nib = nibsel(rng);
                    const int64_t el  = k * N + n;
                    if (el & 1) {
                        d.b_nib[el / 2] |= static_cast<uint8_t>(nib << 4);
                    } else {
                        d.b_nib[el / 2] |= static_cast<uint8_t>(nib);
                    }
                    d.b_real[n * K + k] = kvalues_mxfp4[nib] * g;
                }
            }
        }
    } else {
        const bool is4 = v.wei == wei_kind::s4_grouped || v.wei == wei_kind::s4_grouped_zp;
        const bool zp  = v.wei == wei_kind::s4_grouped_zp;
        if (is4) {
            d.b_s4.assign(N * K / 2, 0);
        } else {
            d.b_s8.resize(N * K);
        }
        if (zp) {
            d.b_zp.assign(KB * N, 0);
        }
        d.b_sc.assign(N * KB, 0.f);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t kb = 0; kb < KB; ++kb) {
                float g = sc(rng) * 0.01f;
                if (v.wei_scale_dt == dt::f16) {
                    g = f16_round(g);
                }
                d.b_sc[n * KB + kb] = g;
                // dnnl_s4 is SIGNED (dnnl_common_types.h), and the WoQ arm
                // this mirrors (onednn-woq.cpp's pack_q4_0_aos_to_s4) stores
                // the already-signed dequant value's two's-complement nibble
                // directly -- nib = (uint8_t)v & 0xF for v in [-8,7] -- with
                // gemm.hpp's zero_points ALWAYS filled with 0
                // (onednn-woq.cpp: out.zero_points.assign(..., 0)), never a
                // non-zero offset. V3z therefore differs from V3 only in
                // that the zero-point attribute/arg is set with value 0, not
                // in the stored nibble or a genuine non-zero zp (that would
                // need dt::u4, which the WoQ arm does not use either).
                if (zp) {
                    d.b_zp[kb * N + n] = 0;
                }
                for (int64_t j = 0; j < GROUP; ++j) {
                    const int64_t k = kb * GROUP + j;
                    if (is4) {
                        const int     q   = q4(rng);  // signed value in [-8,7]
                        const int     nib = static_cast<uint8_t>(q) & 0xF;
                        const int64_t idx = n * K + k;
                        d.b_s4[idx / 2] |= static_cast<uint8_t>(nib << ((idx & 1) ? 4 : 0));
                        d.b_real[idx] = q * g;
                    } else {
                        const int q         = q8(rng);
                        d.b_s8[n * K + k]   = static_cast<int8_t>(q);
                        d.b_real[n * K + k] = q * g;
                    }
                }
            }
        }
    }
    return d;
}

template <typename T> T * dev_upload(sycl::queue & q, const std::vector<T> & h, std::vector<void *> & owned) {
    if (h.empty()) {
        return nullptr;
    }
    T * p = sycl::malloc_device<T>(h.size(), q);
    if (!p) {
        throw std::runtime_error("malloc_device failed");
    }
    q.memcpy(p, h.data(), h.size() * sizeof(T)).wait();
    owned.push_back(p);
    return p;
}

void * dev_upload_scales(sycl::queue & q, const std::vector<float> & sc, dt scale_dt, std::vector<void *> & owned) {
    if (sc.empty()) {
        return nullptr;
    }
    if (scale_dt == dt::f16) {
        std::vector<uint16_t> h(sc.size());
        for (size_t i = 0; i < sc.size(); ++i) {
            h[i] = f32_to_f16_bits(sc[i]);
        }
        return dev_upload(q, h, owned);
    }
    return dev_upload(q, sc, owned);
}

std::string env_or(const char * name, const char * dflt) {
    const char * v = std::getenv(name);
    return v ? std::string(v) : std::string(dflt);
}

bool variant_selected(const char * id) {
    const std::string sel = env_or("PROBE_VARIANTS", "");
    if (sel.empty()) {
        return true;
    }
    return sel.find(id) != std::string::npos;
}

// Returns: 0 ok/unsupported, 1 mismatch.
int run_one(sycl::queue &   q,
            dnnl::engine &  eng,
            dnnl::stream &  strm,
            const variant & v,
            const shape &   s,
            std::mt19937 &  rng,
            int             iters) {
    const int64_t M = s.m, N = s.n, K = s.k, KB = K / GROUP;
    std::printf("variant=%s shape=M%lldxN%lldxK%lld ", v.id, (long long) M, (long long) N, (long long) K);
    std::fflush(stdout);

    // ---- descriptors
    const dt a_dt = v.src == src_kind::f16 ? dt::f16 : dt::s8;
    dt       b_dt = dt::f16;
    if (v.wei == wei_kind::s8_grouped) {
        b_dt = dt::s8;
    } else if (v.wei == wei_kind::e2m1_grouped) {
        b_dt = dt::f4_e2m1;
    } else if (v.wei != wei_kind::f16) {
        b_dt = dt::s4;
    }
    const dnnl::memory::desc a_md({ M, K }, a_dt, { K, 1 });
    // B logical (K, N). Two layouts in play:
    //  - s8/s4 arms: stored [N][K] (Q8_0 SOA plane), strides {1, K}.
    //  - e2m1_grouped: stored per gemm.hpp's WOQ-order desc, strides {N, 1}.
    const bool               wei_woq_order = v.wei == wei_kind::e2m1_grouped;
    const dnnl::memory::desc b_plain_md =
        wei_woq_order ? dnnl::memory::desc({ K, N }, b_dt, { N, 1 }) : dnnl::memory::desc({ K, N }, b_dt, { 1, K });
    const dnnl::memory::desc b_md =
        v.wei_any ? dnnl::memory::desc({ K, N }, b_dt, dnnl::memory::format_tag::any) : b_plain_md;
    const dnnl::memory::desc c_md({ M, N }, dt::f32, { N, 1 });

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);
    try {
        if (v.src == src_kind::s8_grouped) {
            attr.set_scales(DNNL_ARG_SRC, (1 << 0) | (1 << 1), { 1, GROUP }, v.src_scale_dt);
        } else if (v.src == src_kind::s8_per_row) {
            attr.set_scales(DNNL_ARG_SRC, (1 << 0), {}, v.src_scale_dt);
        }
        if (v.wei != wei_kind::f16) {
            attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), { GROUP, 1 }, v.wei_scale_dt);
        }
        if (v.wei == wei_kind::s4_grouped_zp) {
            attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), { GROUP, 1 }, dt::s8);
        }
        if (v.fpmath_f16_int) {
            attr.set_fpmath_mode(dnnl::fpmath_mode::f16, /* apply_to_int = */ true);
        }
    } catch (const dnnl::error & e) {
        std::printf("created=N reason=attr:%s\n", e.what());
        return 0;
    }

    dnnl::matmul::primitive_desc pd;
    try {
        pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
    } catch (const dnnl::error & e) {
        std::printf("created=N reason=%s\n", e.what());
        return 0;
    } catch (const std::exception & e) {
        std::printf("created=N reason=%s\n", e.what());
        return 0;
    }
    std::printf("created=Y ");
    if (v.wei_any) {
        const bool plain = pd.weights_desc() == b_plain_md;
        std::printf("any_plain=%c any_bytes=%zu ", plain ? 'Y' : 'N', pd.weights_desc().get_size());
        if (!plain) {
            // A repacked weight is a second layout (ruling 5); report and stop here.
            std::printf("(would need reorder; not benched)\n");
            return 0;
        }
    }
    std::printf("scratch=%zu ", pd.scratchpad_desc().get_size());
    std::fflush(stdout);

    // ---- data. make_data() and the dnnl::matmul ctor (which triggers oneDNN
    // kernel generation) both belong inside the try: an exception from
    // either must be reported as exec_failed and terminate this line rather
    // than escaping to main() and truncating every remaining row.
    std::vector<void *> owned;
    int                 rc = 0;
    try {
        host_data d = make_data(v, s, rng);

        dnnl::matmul prim(pd);

        void * a_dev =
            v.src == src_kind::f16 ? (void *) dev_upload(q, d.a_f16, owned) : (void *) dev_upload(q, d.a_s8, owned);
        void * b_dev = nullptr;
        if (v.wei == wei_kind::f16) {
            b_dev = dev_upload(q, d.b_f16, owned);
        } else if (v.wei == wei_kind::s8_grouped) {
            b_dev = dev_upload(q, d.b_s8, owned);
        } else if (v.wei == wei_kind::e2m1_grouped) {
            b_dev = dev_upload(q, d.b_nib, owned);
        } else {
            b_dev = dev_upload(q, d.b_s4, owned);
        }
        float * c_dev = sycl::malloc_device<float>(M * N, q);
        owned.push_back(c_dev);
        q.memset(c_dev, 0, M * N * sizeof(float)).wait();

        std::unordered_map<int, dnnl::memory> args;
        args.insert({ DNNL_ARG_SRC, dnnl::memory(a_md, eng, a_dev) });
        args.insert({ DNNL_ARG_WEIGHTS, dnnl::memory(b_plain_md, eng, b_dev) });
        args.insert({ DNNL_ARG_DST, dnnl::memory(c_md, eng, c_dev) });

        if (v.src == src_kind::s8_grouped) {
            void *             sc_dev = dev_upload_scales(q, d.a_sc, v.src_scale_dt, owned);
            dnnl::memory::desc sc_md({ M, KB }, v.src_scale_dt, { KB, 1 });
            args.insert({ DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dnnl::memory(sc_md, eng, sc_dev) });
        } else if (v.src == src_kind::s8_per_row) {
            void *             sc_dev = dev_upload_scales(q, d.a_sc, v.src_scale_dt, owned);
            dnnl::memory::desc sc_md({ M }, v.src_scale_dt, { 1 });
            args.insert({ DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dnnl::memory(sc_md, eng, sc_dev) });
        }
        if (v.wei == wei_kind::e2m1_grouped) {
            // e8m0 scale bytes are already the wire format (not a float to
            // round/convert), so upload them directly rather than routing
            // through dev_upload_scales.
            void *             wsc_dev = dev_upload(q, d.b_e8m0, owned);
            dnnl::memory::desc wsc_md({ KB, N }, dt::e8m0, { N, 1 });
            args.insert({ DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(wsc_md, eng, wsc_dev) });
        } else if (v.wei != wei_kind::f16) {
            // logical scales shape (K/32, N). Stored order: plain => [KB][N]; strided_soa => [N][KB].
            std::vector<float> wsc;
            dnnl::memory::desc wsc_md;
            if (v.wei_scales == wei_scale_layout::plain) {
                wsc.assign(KB * N, 0.f);
                for (int64_t n = 0; n < N; ++n) {
                    for (int64_t kb = 0; kb < KB; ++kb) {
                        wsc[kb * N + n] = d.b_sc[n * KB + kb];
                    }
                }
                wsc_md = dnnl::memory::desc({ KB, N }, v.wei_scale_dt, { N, 1 });
            } else {
                wsc    = d.b_sc;  // already [N][KB]
                wsc_md = dnnl::memory::desc({ KB, N }, v.wei_scale_dt, { 1, KB });
            }
            void * wsc_dev = dev_upload_scales(q, wsc, v.wei_scale_dt, owned);
            args.insert({ DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(wsc_md, eng, wsc_dev) });
        }
        if (v.wei == wei_kind::s4_grouped_zp) {
            void *             zp_dev = dev_upload(q, d.b_zp, owned);
            dnnl::memory::desc zp_md({ KB, N }, dt::s8, { N, 1 });
            args.insert({ DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, dnnl::memory(zp_md, eng, zp_dev) });
        }

        // warmup (also the correctness run)
        sycl::event ev0 = dnnl::sycl_interop::execute(prim, strm, args, {});
        ev0.wait_and_throw();

        std::vector<float> c_host(M * N);
        q.memcpy(c_host.data(), c_dev, M * N * sizeof(float)).wait();

        // reference on a random sample of outputs (double accumulation)
        std::uniform_int_distribution<int64_t> um(0, M - 1), un(0, N - 1);
        double                                 maxerr = 0.0, maxref = 0.0;
        for (int t = 0; t < 512; ++t) {
            const int64_t m = um(rng), n = un(rng);
            double        acc = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                acc += (double) d.a_real[m * K + k] * (double) d.b_real[n * K + k];
            }
            maxerr = std::max(maxerr, std::fabs(acc - (double) c_host[m * N + n]));
            maxref = std::max(maxref, std::fabs(acc));
        }
        // f16-accumulating arms (V0/V4/V4b/V7) legitimately drift ~1e-3
        // relative; int8-src arms accumulate in int32 and should be within
        // f32 rounding of the reference.
        const double tol = (v.src == src_kind::f16 ? 2e-2 : 2e-3) * std::max(1.0, maxref);
        const bool   ok  = maxerr <= tol;

        // timing: N back-to-back executes, host wall (in-order stream) + last event
        sycl::event last;
        const auto  t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            last = dnnl::sycl_interop::execute(prim, strm, args, {});
        }
        last.wait_and_throw();
        const auto   t1      = std::chrono::steady_clock::now();
        const double wall_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        double       ev_us   = -1.0;
        try {
            const auto st = last.get_profiling_info<sycl::info::event_profiling::command_start>();
            const auto en = last.get_profiling_info<sycl::info::event_profiling::command_end>();
            ev_us         = (double) (en - st) / 1000.0;
        } catch (...) {
        }
        const double flop = 2.0 * (double) M * (double) N * (double) K;
        std::printf("us=%.1f ev_us=%.1f tops=%.1f maxerr=%.3g ref=%.3g %s\n", wall_us, ev_us, flop / wall_us / 1e6,
                    maxerr, maxref, ok ? "match=Y" : "match=N");
        if (!ok) {
            rc = 1;
        }
    } catch (const dnnl::error & e) {
        std::printf("exec_failed reason=%s\n", e.what());
    } catch (const std::exception & e) {
        std::printf("exec_failed reason=%s\n", e.what());
    }
    for (void * p : owned) {
        sycl::free(p, q);
    }
    return rc;
}

}  // namespace
#endif  // GGML_SYCL_DNNL

int main() {
#if GGML_SYCL_DNNL
    try {
        sycl::device  dev{ sycl::gpu_selector_v };
        sycl::context ctx{ dev };
        sycl::queue   q{
            ctx, dev, { sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{} }
        };
        std::printf("device=%s driver=%s dnnl=%d.%d.%d\n", dev.get_info<sycl::info::device::name>().c_str(),
                    dev.get_info<sycl::info::device::driver_version>().c_str(), DNNL_VERSION_MAJOR, DNNL_VERSION_MINOR,
                    DNNL_VERSION_PATCH);
        dnnl::engine eng  = dnnl::sycl_interop::make_engine(dev, ctx);
        dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);

        const int    iters = std::atoi(env_or("PROBE_ITERS", "20").c_str());
        const bool   small = env_or("PROBE_SHAPES", "") == "small";
        std::mt19937 rng(1234);
        int          rc = 0;
        for (const variant & v : VARIANTS) {
            if (!variant_selected(v.id)) {
                continue;
            }
            const shape * begin = small ? SHAPES_SMALL : SHAPES_FULL;
            const size_t  count = small ? sizeof(SHAPES_SMALL) / sizeof(shape) : sizeof(SHAPES_FULL) / sizeof(shape);
            for (size_t i = 0; i < count; ++i) {
                rc |= run_one(q, eng, strm, v, begin[i], rng, iters > 0 ? iters : 20);
            }
        }
        std::puts(rc == 0 ? "PASS: onednn.int8_grouped.probe completed" : "FAIL: onednn.int8_grouped.probe mismatch");
        return rc;
    } catch (const sycl::exception & e) {
        std::printf("SKIP: onednn.int8_grouped.probe setup_failed reason=%s\n", e.what());
        return EXIT_SKIP;
    } catch (const dnnl::error & e) {
        std::printf("SKIP: onednn.int8_grouped.probe setup_failed reason=%s\n", e.what());
        return EXIT_SKIP;
    } catch (const std::exception & e) {
        std::printf("SKIP: onednn.int8_grouped.probe setup_failed reason=%s\n", e.what());
        return EXIT_SKIP;
    }
#else
    std::puts("SKIP: onednn.int8_grouped.probe GGML_SYCL_DNNL=0");
    return EXIT_SKIP;
#endif
}
