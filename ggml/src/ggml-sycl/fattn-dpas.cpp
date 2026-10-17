//
// MIT license
// SPDX-License-Identifier: MIT
//
// Flash attention over a q4_0 / q8_0 KV cache on the Xe2 matrix engine (DPAS).
//
// The kernel is Torchit1's, from https://github.com/Torchit1/llama.cpp, branch sycl-fa-q4-xmx, commit
// 24a3a71896cdf1d21b625c13c63f98b05bdde8e7, file ggml/src/ggml-sycl/fattn-dpas.cpp. That fork is MIT licensed, as are
// upstream llama.cpp and this fork. The kernel body, its tile shapes and the tuning measurements in its comments are
// his and are kept as he wrote them. What this fork changed:
// - The operands are the resolved pointers of ggml_sycl_flash_attn_ext's fattn_params, never tensor data fields, and
//   the split-K workspace is a unified-cache allocation the caller owns and passes in, in place of a pool allocation.
// - Q may be F16: under GGML_SYCL_F16 the graph casts Q to f16 before FLASH_ATTN_EXT. Q is read as F16 or F32 at its
//   one load and quantized to s8 there as before.
// - A dead KV cell (mask -inf) may hold NaN or Inf. Its score is set with fattn_mask_apply rather than by adding the
//   mask, and each dequantized V value passes fattn_v_zero_if_dead before the P.V multiply (fattn-common.hpp says why).
// - No query-row cap: prefill runs here as well. Work-groups multiply with the token chunks, and the key split shrinks
//   to keep one wave.
// - The device gate, the knob and the supports_op predicate are in fattn.cpp. The Xe-core count is passed in: this
//   fork's nsm is the compute-unit count, eight per Xe2 core.
// - Every launch returns its event and goes through ggml_sycl_profile_submit. The split combine is a kernel of this
//   file, and the 256-register mode is instantiated only where it is launched.
// - GGML_SYCL_FA_DPAS_NO_KERNELS compiles the kernels out and leaves a launcher that aborts; fattn.cpp then never
//   admits the route, so the knob and the hook stay in place.
// - The GGML_SYCL_XMX_AOT_SG build guard (GGML_SYCL_FA_DPAS_BUILD 0 for a sub-group-8 or mixed AOT list) is dropped.
//   No such macro exists here. This fork's AOT targets are bmg_g21 and bmg_g31, both sub-group 16, beside the spir64
//   JIT image, and the fattn.cpp gate admits only Battlemage. Each kernel carries its sub_group_size<16> property.
//

#include "fattn-dpas.hpp"

#include "sycl-kernel-profiler.hpp"

#include <algorithm>
#include <cstdint>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/prefetch.hpp>
#include <sycl/sycl.hpp>

#ifndef GGML_SYCL_FA_DPAS_NO_KERNELS

namespace syclex = sycl::ext::oneapi::experimental;

static constexpr int   FA_DPAS_WG   = 256;       // work-group size
static constexpr int   FA_DPAS_NSG  = 16;        // sub-groups of 16 lanes
static constexpr float FA_DPAS_MINF = -1.0e30f;  // finite start value of the running max

typedef short fa_dpas_short8 __attribute__((ext_vector_type(8)));
typedef int   fa_dpas_int4 __attribute__((ext_vector_type(4)));
typedef int   fa_dpas_int8 __attribute__((ext_vector_type(8)));
typedef float fa_dpas_float8 __attribute__((ext_vector_type(8)));

// Xe2 DPAS (SIMD16, systolic depth 8, repeat count 8) as inline vISA: joint_matrix has no s4 type.
// d = a x b. a: 8 rows x 32 s8, 2 per lane per row. b: 32 x 16 s4, 16 bytes per lane, one column per lane.
static inline void fa_dpas_s8_s4(const fa_dpas_short8 & a, const fa_dpas_int4 & b, fa_dpas_int8 & d) {
#    ifdef __SYCL_DEVICE_ONLY__
    __asm__(
        "{\n"
        ".decl DB v_type=G type=ud num_elts=64 align=GRF alias=<%1,0>\n"
        ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
        "dpas.s4.s8.8.8 (M1, 16) %0.0 %%null.0 DB.0 DA(0,0)\n"
        "}\n"
        : "=rw"(d)
        : "rw"(b), "rw"(a));
#    else
    d = 0;
    GGML_UNUSED(a);
    GGML_UNUSED(b);
#    endif
}

// d = a x b. a: 8 rows x 32 s8, 2 per lane per row. b: 32 x 16 s8, 32 bytes per lane, one column per lane.
static inline void fa_dpas_s8_s8(const fa_dpas_short8 & a, const fa_dpas_int8 & b, fa_dpas_int8 & d) {
#    ifdef __SYCL_DEVICE_ONLY__
    __asm__(
        "{\n"
        ".decl DB v_type=G type=ud num_elts=128 align=GRF alias=<%1,0>\n"
        ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
        "dpas.s8.s8.8.8 (M1, 16) %0.0 %%null.0 DB.0 DA(0,0)\n"
        "}\n"
        : "=rw"(d)
        : "rw"(b), "rw"(a));
#    else
    d = 0;
    GGML_UNUSED(a);
    GGML_UNUSED(b);
#    endif
}

// acc += a x b. a: 8 rows x 16 f16, one per lane per row. b: 16 x 16 f16 as pairs, 8 dwords per lane, one column per
// lane.
static inline void fa_dpas_f16(const fa_dpas_short8 & a, const fa_dpas_int8 & b, fa_dpas_float8 & acc) {
#    ifdef __SYCL_DEVICE_ONLY__
    __asm__(
        "{\n"
        ".decl DB v_type=G type=ud num_elts=128 align=GRF alias=<%1,0>\n"
        ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
        "dpas.hf.hf.8.8 (M1, 16) %0.0 %0.0 DB.0 DA(0,0)\n"
        "}\n"
        : "+rw"(acc)
        : "rw"(b), "rw"(a));
#    else
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    GGML_UNUSED(acc);
#    endif
}

// Block b of a row = f16 scale + NQ dwords of quants (q4_0: 18 bytes, NQ 4; q8_0: 34 bytes, NQ 8), at byte (2 + 4 NQ) b.
// w = the NQ + 1 dwords from that byte / 4. An odd block starts 2 bytes into w[0].
template <int NQ> static inline void fa_dpas_split_block(const uint32_t * w, bool odd, uint32_t & sc, uint32_t * q) {
    if (!odd) {
        sc = w[0] & 0xFFFFu;
#    pragma unroll
        for (int i = 0; i < NQ; ++i) {
            q[i] = (w[i] >> 16) | (w[i + 1] << 16);
        }
    } else {
        sc = w[0] >> 16;
#    pragma unroll
        for (int i = 0; i < NQ; ++i) {
            q[i] = w[i + 1];
        }
    }
}

struct fa_dpas_params {
    const char *   Q;
    const char *   K;
    const char *   V;
    const char *   mask;
    float *        dst;
    float *        parts;  // per-split partial results, nsplit > 1 only
    sycl::float2 * meta;   // per-split (max, sum), nsplit > 1 only
    float          scale;
    const float *  sinks;  // per query head softmax sink logit, or null
    bool           q_f16;  // Q is F16 (else F32)
    int            G;      // query heads per KV head
    int            nq;     // query tokens per work-group
    int            nqc;    // token chunks
    int            ne01;
    int            ne02;
    int            ne11;
    int            ne33;
    int            nsplit;
    int            chunk;  // keys per split
    int64_t        nb01, nb02, nb03;
    int64_t        nb11, nb12, nb13;
    int64_t        nb21, nb22, nb23;
    int64_t        nb31, nb33;
};

// The kernel lambda with its launch properties (sub-group size, register file), which reach the kernel only through the
// functor's get(properties_tag).
template <typename KernelFunc, typename PropertiesT> struct fa_dpas_kernel_with_properties {
    KernelFunc  func;
    PropertiesT props;

    void operator()(sycl::nd_item<3> it) const { func(it); }

    auto get(syclex::properties_tag) const { return props; }
};

static ggml_sycl_profile_label fa_dpas_profile_label(int device, const char * name, const char * metadata) {
    ggml_sycl_profile_label label{};
    label.name       = name;
    label.category   = "fattn";
    label.queue_kind = "compute";
    label.metadata   = metadata;
    label.device     = device;
    label.bytes      = 0;
    return label;
}

// One work-group = one KV head x nq query tokens x one slice of the keys.
// Rows r = c * G + g: token c, query head g of the KV head. At most 16 * NR16 rows.
// Per tile of 16 * KG keys:
// - S = Q K^T: s8 x s4 (q4_0) or s8 x s8 (q8_0) DPAS on the raw K blocks, one key per lane. Q is quantized to s8 once
//   per work-group.
// - online softmax per row, P stored as f16.
// - O^T += V^T P^T: f16 DPAS, V dequantized in registers. Sub-group w owns head dims D / 16 * w .. D / 16 * (w + 1) - 1.
// ST: copy each K and V tile to SLM first, 1: coalesced 16-byte loads (single token decode, D 256), 2: coalesced dword
// loads (q8_0, D 128: its 136-byte rows gather poorly per lane).
// LG: 256-register file. The 32-row launch runs 2 work-groups x 16 sub-groups per Xe core, which is what that mode
// allows, so it costs no occupancy and removes spills. grf_size_automatic does not pick it for this kernel. Every
// Battlemage part has the large register file.
// Q8: K/V are q8_0 (else q4_0).
template <int D, int NR16, int KG, int TC, int ST, bool LG, bool Q8>
static sycl::event fattn_dpas_kernel(const fa_dpas_params            p,
                                     const int                       nkvh,
                                     const int                       ne03,
                                     dpct::queue_ptr                 stream,
                                     const ggml_sycl_profile_label & label) {
    static_assert(QK4_0 == QK8_0, "q4_0 and q8_0 blocks hold the same number of dims");
    // RTMAX: 8-row tiles in Q.K. TK: keys per tile. BB: bytes per block. NQ: quant dwords per block. NB: blocks per
    // row. DPS: head dims per sub-group in P.V, inside one 32-dim block. NPV: sub-groups doing P.V, the rest only
    // reach its barrier. ALLPV: every sub-group does P.V, no runtime check in the loops. XT: 8-dim tiles per sub-group
    // in P.V. PS: f16 row stride of P. RW: dwords per row. ODD: odd block count, rows may start 2 bytes into a dword.
    // SW: odd SLM row stride, no bank conflicts for lane = key.
    constexpr int  RMAX  = 16 * NR16;
    constexpr int  RTMAX = RMAX / 8;
    constexpr int  TK    = 16 * KG;
    constexpr int  BB    = Q8 ? (int) sizeof(block_q8_0) : (int) sizeof(block_q4_0);
    constexpr int  NQ    = (BB - 2) / 4;
    constexpr int  NB    = D / QK4_0;
    constexpr int  DPS   = D <= 128 ? 8 : D <= 256 ? 16 : 32;
    constexpr int  NPV   = D / DPS;
    constexpr bool ALLPV = NPV == FA_DPAS_NSG;
    constexpr int  XT    = DPS / 8;
    constexpr int  PS    = TK + 16;
    constexpr int  RW    = NB * BB / 4;
    constexpr bool ODD   = (NB * BB) % 4 != 0;
    constexpr int  SW    = RW + 1;
    static_assert(D % QK4_0 == 0 && D % DPS == 0 && NPV <= FA_DPAS_NSG && QK4_0 % DPS == 0, "unsupported head size");
    static_assert(ST != 1 || RW % 4 == 0, "16-byte staging needs rows of 16-byte multiples");
    static_assert(!LG || NR16 == 2, "the 256-register mode is for the 32-row launch");

    return ggml_sycl_profile_submit(*stream, label, [&](sycl::queue & q) {
        return q.submit([&](sycl::handler & cgh) {
            // sQ8: Q s8 [tile][block][lane]. sDQ: Q scales [tile][block]. sS: S [row][key]. sP: P f16 [row][key].
            sycl::local_accessor<fa_dpas_short8, 1> sQ8(sycl::range<1>(RTMAX * NB * 16), cgh);
            sycl::local_accessor<fa_dpas_float8, 1> sDQ(sycl::range<1>(RTMAX * NB), cgh);
            sycl::local_accessor<float, 1>          sS(sycl::range<1>(RMAX * TK), cgh);
            sycl::local_accessor<fa_dpas_int8, 1>   sP(sycl::range<1>(RMAX * PS / 16), cgh);
            sycl::local_accessor<float, 1>          sM(sycl::range<1>(RMAX), cgh);
            sycl::local_accessor<float, 1>          sL(sycl::range<1>(RMAX), cgh);
            sycl::local_accessor<float, 1>          sA(sycl::range<1>(RMAX), cgh);
            sycl::local_accessor<uint32_t, 1>       sKV(sycl::range<1>(ST ? TK * SW : 1), cgh);
            const sycl::nd_range<3> ndr(sycl::range<3>((size_t) ne03 * p.nqc, nkvh, (size_t) p.nsplit * FA_DPAS_WG),
                                        sycl::range<3>(1, 1, FA_DPAS_WG));

            auto kern = [=](sycl::nd_item<3> it) {
                const int tid   = it.get_local_id(2);
                const int split = it.get_group(2);
                const int kvh   = it.get_group(1);
                const int seq   = it.get_group(0) / p.nqc;
                const int c0    = (it.get_group(0) % p.nqc) * p.nq;
                const int G     = p.G;
                const int R     = p.nq * G;
                const int RT    = (R + 7) / 8;
                auto      sg    = it.get_sub_group();
                const int w     = tid / 16;
                const int lane  = tid % 16;
                short *   sQs   = (short *) &sQ8[0];
                short *   sPs   = (short *) &sP[0];

                // Q -> s8, one scale per row and block. Lane l holds the 2 dims of K column elements 2 l, 2 l + 1:
                // q4_0 packs dims l and l + 16 into byte l, q8_0 stores dims in order
                for (int e = tid; e < RT * 8 * NB; e += FA_DPAS_WG) {
                    const int    r    = e / NB;
                    const int    b    = e % NB;
                    const int    c    = c0 + r / G;
                    const bool   real = r < R && c < p.ne01;
                    const char * qrow = p.Q + seq * p.nb03 + (int64_t) (real ? c : 0) * p.nb01 +
                                        (int64_t) (kvh * G + (real ? r % G : 0)) * p.nb02;
                    float x[QK4_0];
                    float amax = 0.0f;
                    if (p.q_f16) {
                        const sycl::half * qr = (const sycl::half *) qrow + b * QK4_0;
#    pragma unroll
                        for (int i = 0; i < QK4_0; ++i) {
                            x[i] = real ? static_cast<float>(qr[i]) * p.scale : 0.0f;
                            amax = sycl::fmax(amax, sycl::fabs(x[i]));
                        }
                    } else {
                        const float * qr = (const float *) qrow + b * QK4_0;
#    pragma unroll
                        for (int i = 0; i < QK4_0; ++i) {
                            x[i] = real ? qr[i] * p.scale : 0.0f;
                            amax = sycl::fmax(amax, sycl::fabs(x[i]));
                        }
                    }
                    const float dq = amax / 127.0f;
                    const float iq = amax > 0.0f ? 127.0f / amax : 0.0f;
                    const int   t  = r / 8;
                    const int   m  = r % 8;

                    ((float *) &sDQ[0])[(t * NB + b) * 8 + m] = dq;
#    pragma unroll
                    for (int l = 0; l < 16; ++l) {
                        const int lo = (int) sycl::rint(x[Q8 ? 2 * l : l] * iq);
                        const int hi = (int) sycl::rint(x[Q8 ? 2 * l + 1 : l + 16] * iq);

                        sQs[((t * NB + b) * 16 + l) * 8 + m] = (short) ((lo & 0xFF) | ((hi & 0xFF) << 8));
                    }
                }
                // rows without a query keep P = 0
                for (int e = tid; e < RMAX * PS; e += FA_DPAS_WG) {
                    sPs[e] = 0;
                }
                if (tid < RMAX) {
                    // the sink is one more key with logit sinks[head] and no value; the first key slice owns it
                    const bool sink = p.sinks && split == 0 && tid < R;
                    sM[tid]         = sink ? p.sinks[kvh * G + tid % G] : FA_DPAS_MINF;
                    sL[tid]         = sink ? 1.0f : 0.0f;
                    sA[tid]         = 1.0f;
                }
                it.barrier(sycl::access::fence_space::local_space);

                const int          k_begin = split * p.chunk;
                const int          k_end   = sycl::min(k_begin + p.chunk, p.ne11);
                const char *       Kh      = p.K + seq * p.nb13 + kvh * p.nb12;
                const char *       Vh      = p.V + seq * p.nb23 + kvh * p.nb22;
                const sycl::half * mrow    = p.mask ? (const sycl::half *) (p.mask + (seq % p.ne33) * p.nb33) : nullptr;
                // the mask rows of this work-group's tokens, for the dead-cell test on V
                const int64_t      mstride = p.nb31 / (int64_t) sizeof(sycl::half);
                const sycl::half * mrows   = mrow ? mrow + (int64_t) c0 * mstride : nullptr;
                const int          mnrows  = sycl::min(p.nq, p.ne01 - c0);

                // O^T: lane = query row, element m = head dim
                fa_dpas_float8 o[NR16][XT];
#    pragma unroll
                for (int t = 0; t < NR16; ++t) {
#    pragma unroll
                    for (int x = 0; x < XT; ++x) {
                        o[t][x] = 0.0f;
                    }
                }
                // the dims of this sub-group lie in one V block
                const int vb  = (DPS * w) / QK4_0;
                const int vd0 = (DPS * w) % QK4_0;

                // keys past k_end repeat key k_begin, their scores are -inf
                auto stage = [&](const char * base, int64_t nb, int t0) {
                    if constexpr (ST == 2) {
                        for (int e = tid; e < TK * RW; e += FA_DPAS_WG) {
                            const int key     = e / RW;
                            const int q       = e - key * RW;
                            const int j       = t0 + key < k_end ? t0 + key : k_begin;
                            sKV[key * SW + q] = ((const uint32_t *) (base + (int64_t) j * nb))[q];
                        }
                        return;
                    }
                    constexpr int RV  = ST == 1 ? RW / 4 : 1;
                    constexpr int NV  = TK * RV;
                    constexpr int PER = (NV + FA_DPAS_WG - 1) / FA_DPAS_WG;
                    sycl::uint4   v[PER];
#    pragma unroll
                    for (int i = 0; i < PER; ++i) {
                        const int e = tid + i * FA_DPAS_WG;
                        if (e < NV) {
                            const int key = e / RV;
                            const int q   = e - key * RV;
                            const int j   = t0 + key < k_end ? t0 + key : k_begin;
                            v[i]          = ((const sycl::uint4 *) (base + (int64_t) j * nb))[q];
                        }
                    }
#    pragma unroll
                    for (int i = 0; i < PER; ++i) {
                        const int e = tid + i * FA_DPAS_WG;
                        if (e < NV) {
                            const int key = e / RV;
                            const int q   = e - key * RV;
                            const int o   = key * SW + 4 * q;
                            sKV[o + 0]    = v[i][0];
                            sKV[o + 1]    = v[i][1];
                            sKV[o + 2]    = v[i][2];
                            sKV[o + 3]    = v[i][3];
                        }
                    }
                };

                // the NQ + 1 dwords of block b: of tile key kk in SLM, or of a row that starts 2 ph bytes into dword
                // row[0]
                auto load_block = [&](const uint32_t * row, int ph, int kk, int b, uint32_t * wv) {
                    const int wd = (2 * ph + BB * b) / 4;
                    if constexpr (ST) {
                        GGML_UNUSED(row);
#    pragma unroll
                        for (int i = 0; i < NQ + 1; ++i) {
                            wv[i] = sKV[kk * SW + wd + i];
                        }
                    } else {
                        GGML_UNUSED(kk);
#    pragma unroll
                        for (int i = 0; i < NQ + 1; ++i) {
                            wv[i] = row[wd + i];
                        }
                    }
                };

                // start loading the rows of keys t0 .. t0 + TK - 1 into L1 while the current phase computes
                auto prefetch_rows = [&](const char * base, int64_t nb, int t0) {
                    constexpr int NL = RW * 4 / 64 + 1;  // cache lines per row
                    for (int e = tid; e < TK * NL; e += FA_DPAS_WG) {
                        const int j = t0 + e / NL;
                        if (j < k_end) {
                            char * pa = const_cast<char *>(base) + (int64_t) j * nb + (e % NL) * 64;
                            syclex::prefetch(pa, syclex::properties{ syclex::prefetch_hint_L1 });
                        }
                    }
                };

                for (int t0 = k_begin; t0 < k_end; t0 += TK) {
                    if constexpr (ST) {
                        stage(Kh, p.nb11, t0);
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    prefetch_rows(Vh, p.nb21, t0);

                    // S = Q K^T, job = (key group, TC row tiles)
                    const int nch = (RT + TC - 1) / TC;
                    for (int job = w; job < KG * nch; job += FA_DPAS_NSG) {
                        const int        kg = job % KG;
                        const int        ch = job / KG;
                        const int        j  = t0 + kg * 16 + lane;
                        const bool       jv = j < k_end;
                        const char *     kb = Kh + (int64_t) (jv ? j : k_begin) * p.nb11;
                        // row starts 2 bytes into a dword
                        const int        kp = (ST || !ODD) ? 0 : (int) (((uintptr_t) kb >> 1) & 1);
                        const uint32_t * kr = (const uint32_t *) (kb - 2 * kp);
                        fa_dpas_float8   s[TC];
#    pragma unroll
                        for (int tt = 0; tt < TC; ++tt) {
                            s[tt] = 0.0f;
                        }
#    pragma unroll
                        for (int b = 0; b < NB; ++b) {
                            uint32_t wv[NQ + 1];
                            uint32_t sc;
                            uint32_t q[NQ];
                            load_block(kr, kp, kg * 16 + lane, b, wv);
                            fa_dpas_split_block<NQ>(wv, (b + kp) % 2 != 0, sc, q);
                            const float dk = static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) sc));
#    pragma unroll
                            for (int tt = 0; tt < TC; ++tt) {
                                const int t = ch * TC + tt;
                                if (t < RT) {
                                    fa_dpas_int8 ia;
                                    if constexpr (Q8) {
                                        const fa_dpas_int8 bv = { (int) q[0], (int) q[1], (int) q[2], (int) q[3],
                                                                  (int) q[4], (int) q[5], (int) q[6], (int) q[7] };
                                        fa_dpas_s8_s8(sQ8[(t * NB + b) * 16 + lane], bv, ia);
                                    } else {
                                        // u4 (n + 8) -> s4 n
                                        const fa_dpas_int4 bv = { (int) (q[0] ^ 0x88888888u),
                                                                  (int) (q[1] ^ 0x88888888u),
                                                                  (int) (q[2] ^ 0x88888888u),
                                                                  (int) (q[3] ^ 0x88888888u) };
                                        fa_dpas_s8_s4(sQ8[(t * NB + b) * 16 + lane], bv, ia);
                                    }
                                    const fa_dpas_float8 ds = sDQ[t * NB + b] * dk;
                                    s[tt] += __builtin_convertvector(ia, fa_dpas_float8) * ds;
                                }
                            }
                        }
#    pragma unroll
                        for (int tt = 0; tt < TC; ++tt) {
                            const int t = ch * TC + tt;
                            if (t < RT) {
#    pragma unroll
                                for (int m = 0; m < 8; ++m) {
                                    const int r = t * 8 + m;
                                    const int c = c0 + r / G;
                                    if (r < R) {
                                        // a cell the mask kills is exactly -inf whatever its K row holds
                                        float score = -INFINITY;
                                        if (jv && c < p.ne01) {
                                            score = mrow ? fattn_mask_apply(s[tt][m], 1.0f,
                                                                            mrow[(int64_t) c * mstride + j]) :
                                                           s[tt][m];
                                        }
                                        sS[r * TK + kg * 16 + lane] = score;
                                    }
                                }
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if constexpr (ST) {
                        stage(Vh, p.nb21, t0);  // the softmax does not use sKV
                    }

                    // online softmax per row -> P
                    for (int r = w; r < R; r += FA_DPAS_NSG) {
                        float sv[KG];
                        float mt = -INFINITY;
#    pragma unroll
                        for (int i = 0; i < KG; ++i) {
                            sv[i] = sS[r * TK + i * 16 + lane];
                            mt    = sycl::fmax(mt, sv[i]);
                        }
                        mt                = sycl::reduce_over_group(sg, mt, sycl::maximum<float>());
                        const float m_old = sM[r];
                        const float m_new = sycl::fmax(m_old, mt);
                        float       lt    = 0.0f;
#    pragma unroll
                        for (int i = 0; i < KG; ++i) {
                            const float pv              = sycl::native::exp(sv[i] - m_new);
                            sPs[r * PS + i * 16 + lane] = (short) sycl::bit_cast<uint16_t>(sycl::half(pv));
                            lt += pv;
                        }
                        lt = sycl::reduce_over_group(sg, lt, sycl::plus<float>());
                        if (lane == 0) {
                            const float a = sycl::native::exp(m_old - m_new);
                            sA[r]         = a;
                            sL[r]         = sL[r] * a + lt;
                            sM[r]         = m_new;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    prefetch_rows(Kh, p.nb11, t0 + TK);

                    // O^T = diag(alpha) O^T + V^T P^T. A = V^T (8 dims x 16 keys, lane = key), B = P^T (16 keys x 16 rows)
#    pragma unroll
                    for (int t = 0; t < NR16; ++t) {
                        const float a = sA[t * 16 + lane];
#    pragma unroll
                        for (int x = 0; x < XT; ++x) {
                            o[t][x] *= a;
                        }
                    }
#    pragma unroll 1
                    for (int kg = 0; kg < ((ALLPV || w < NPV) ? KG : 0); ++kg) {
                        const int        j   = t0 + kg * 16 + lane;
                        const bool       jv  = j < k_end;
                        const char *     vbp = Vh + (int64_t) (jv ? j : k_begin) * p.nb21;
                        const int        vp  = (ST || !ODD) ? 0 : (int) (((uintptr_t) vbp >> 1) & 1);
                        const uint32_t * vr  = (const uint32_t *) (vbp - 2 * vp);
                        uint32_t         wv[NQ + 1];
                        uint32_t         sc;
                        uint32_t         q[NQ];
                        load_block(vr, vp, kg * 16 + lane, vb, wv);
                        fa_dpas_split_block<NQ>(wv, (vb + vp) % 2 != 0, sc, q);
                        const float    dv = jv ? static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) sc)) : 0.0f;
                        fa_dpas_short8 va[XT];
#    pragma unroll
                        for (int x = 0; x < XT; ++x) {
#    pragma unroll
                            for (int m = 0; m < 8; ++m) {
                                const int d = vd0 + x * 8 + m;
                                int       v;
                                if constexpr (Q8) {
                                    v = (int8_t) ((q[d / 4] >> (8 * (d % 4))) & 0xFF);
                                } else {
                                    const int byte = d % 16;
                                    v              = (int) ((q[byte / 4] >> (8 * (byte % 4) + 4 * (d / 16))) & 0xF) - 8;
                                }
                                // weight 0 times a dead cell's non-finite V is NaN on the matrix engine; such a V is
                                // zeroed when the cell is dead for every token of the work-group
                                const sycl::half vh =
                                    fattn_v_zero_if_dead(sycl::half((float) v * dv), mrows, mstride, mnrows, j);
                                va[x][m] = (short) sycl::bit_cast<uint16_t>(vh);
                            }
                        }
#    pragma unroll
                        for (int t = 0; t < NR16; ++t) {
                            const fa_dpas_int8 pb = sP[((t * 16 + lane) * PS + kg * 16) / 16];
#    pragma unroll
                            for (int x = 0; x < XT; ++x) {
                                fa_dpas_f16(va[x], pb, o[t][x]);
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

#    pragma unroll
                for (int t = 0; t < NR16; ++t) {
                    const int r = t * 16 + lane;
                    const int c = c0 + r / G;
                    if ((ALLPV || w < NPV) && r < R && c < p.ne01) {
                        const int64_t jdu = ((int64_t) seq * p.ne01 + c) * p.ne02 + kvh * G + r % G;
                        const float   l   = sL[r];
#    pragma unroll
                        for (int x = 0; x < XT; ++x) {
#    pragma unroll
                            for (int m = 0; m < 8; ++m) {
                                const int d = w * DPS + x * 8 + m;
                                if (p.nsplit == 1) {
                                    p.dst[jdu * D + d] = l > 0.0f ? o[t][x][m] / l : 0.0f;
                                } else {
                                    p.parts[(jdu * p.nsplit + split) * D + d] = o[t][x][m];
                                }
                            }
                        }
                    }
                }
                if (p.nsplit > 1 && tid < R && c0 + tid / G < p.ne01) {
                    const int64_t jdu = ((int64_t) seq * p.ne01 + c0 + tid / G) * p.ne02 + kvh * G + tid % G;
                    p.meta[jdu * p.nsplit + split] = sycl::float2(sM[tid], sL[tid]);
                }
            };
            if constexpr (LG) {
                auto props =
                    syclex::properties{ sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<16> };
                cgh.parallel_for(ndr, fa_dpas_kernel_with_properties<decltype(kern), decltype(props)>{ kern, props });
            } else {
                auto props = syclex::properties{ syclex::sub_group_size<16> };
                cgh.parallel_for(ndr, fa_dpas_kernel_with_properties<decltype(kern), decltype(props)>{ kern, props });
            }
        });
    });
}

// Merges the key slices of a split launch: out = sum_s exp(m_s - M) O_s / sum_s exp(m_s - M) l_s, M = max_s m_s, with
// the unnormalised O_s and the (m_s, l_s) of each slice. A row no key reached (every l_s 0) is 0, as the unsplit kernel
// writes it.
static sycl::event fattn_dpas_combine(dpct::queue_ptr                 stream,
                                      const ggml_sycl_profile_label & label,
                                      const float *                   parts,
                                      const sycl::float2 *            meta,
                                      float *                         dst,
                                      int64_t                         nrows,
                                      int                             D,
                                      int                             nsplit) {
    const size_t n = (size_t) nrows * (size_t) D;
    return ggml_sycl_profile_submit(*stream, label, [&](sycl::queue & q) {
        return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const int64_t        i   = (int64_t) idx[0];
            const int64_t        row = i / D;
            const int            d   = (int) (i - row * D);
            const sycl::float2 * mr  = meta + row * nsplit;
            float                M   = -INFINITY;
            for (int s = 0; s < nsplit; ++s) {
                M = sycl::fmax(M, mr[s].x());
            }
            float num = 0.0f;
            float den = 0.0f;
            for (int s = 0; s < nsplit; ++s) {
                const float wgt = sycl::native::exp(mr[s].x() - M);
                num += wgt * parts[(row * nsplit + s) * D + d];
                den += wgt * mr[s].y();
            }
            dst[i] = den > 0.0f ? num / den : 0.0f;
        });
    });
}

template <int D, int NR16, int KG, int TC, int ST, bool Q8>
static sycl::event fattn_dpas_case(dpct::queue_ptr                    stream,
                                   int                                device,
                                   const fattn_params &               params,
                                   float *                            parts,
                                   sycl::float2 *                     meta,
                                   size_t                             parts_capacity,
                                   size_t                             meta_capacity,
                                   int                                xe_cores,
                                   ggml_sycl_fattn_dpas_launch_info * info) {
    constexpr int TK = 16 * KG;

    const int ne01 = params.ne01;
    const int ne11 = params.ne11;
    const int nkvh = params.ne12;
    const int ne03 = params.ne03;

    fa_dpas_params p;
    p.Q     = params.Q;
    p.K     = params.K;
    p.V     = params.V;
    p.mask  = params.mask;
    p.sinks = (const float *) params.sinks;
    p.dst   = params.dst;
    p.scale = params.scale;
    p.q_f16 = params.Q_type == GGML_TYPE_F16;
    p.G     = params.ne02 / params.ne12;
    p.nq    = std::max(1, std::min(ne01, 16 * NR16 / p.G));
    p.nqc   = (ne01 + p.nq - 1) / p.nq;
    p.ne01  = ne01;
    p.ne02  = params.ne02;
    p.ne11  = ne11;
    p.ne33  = params.mask ? std::max(1, params.ne33) : 1;
    p.nb01  = params.nb01;
    p.nb02  = params.nb02;
    p.nb03  = params.nb03;
    p.nb11  = params.nb11;
    p.nb12  = params.nb12;
    p.nb13  = params.nb13;
    p.nb21  = params.nb21;
    p.nb22  = params.nb22;
    p.nb23  = params.nb23;
    p.nb31  = params.mask ? params.nb31 : 0;
    p.nb33  = params.mask ? params.nb33 : 0;

    // split the keys so that all work-groups run in one wave
    // SLM fits about 4 work-groups per Xe2 core with 16 rows, 2 with 32.
    const int wave    = std::max(1, xe_cores) * (NR16 == 1 ? 4 : 2);
    const int ntiles  = (ne11 + TK - 1) / TK;
    int       nsplit  = std::max(1, std::min(wave / (nkvh * ne03 * p.nqc), ntiles));
    p.chunk           = ((ntiles + nsplit - 1) / nsplit) * TK;
    nsplit            = (ne11 + p.chunk - 1) / p.chunk;
    // ggml_sycl_fattn_dpas_parts_capacity() bounds every split this computes; a missing or smaller workspace runs the
    // keys unsplit instead
    const size_t rows = (size_t) ne03 * (size_t) ne01 * (size_t) params.ne02;
    if (nsplit > 1 && (parts == nullptr || meta == nullptr || (size_t) nsplit * rows * (size_t) D > parts_capacity ||
                       (size_t) nsplit * rows > meta_capacity)) {
        nsplit  = 1;
        p.chunk = ntiles * TK;
    }
    p.nsplit = nsplit;
    p.parts  = nsplit > 1 ? parts : nullptr;
    p.meta   = nsplit > 1 ? meta : nullptr;

    // B580, us, 4 / 8 / 16 tokens: D 256 G 6 at 128K keys 1281 / 2460 / 6077 -> 1209 / 2300 / 5757,
    // D 128 G 4 (8 / 16 tokens) 2429 / 5483 -> 2098 / 4754. Below ~4K keys the launch overhead wins (512 keys: 16 -> 19).
    const bool lg = NR16 == 2 && ne11 >= 4096;

    info->nsplit = nsplit;
    info->nr16   = NR16;
    info->st     = ST;
    info->lg     = lg;

    const ggml_sycl_profile_label label = fa_dpas_profile_label(device, "fattn.compute.dpas", "role=compute;path=dpas");
    sycl::event                   ev;
    if constexpr (NR16 == 2) {
        ev = lg ? fattn_dpas_kernel<D, NR16, KG, TC, ST, true, Q8>(p, nkvh, ne03, stream, label) :
                  fattn_dpas_kernel<D, NR16, KG, TC, ST, false, Q8>(p, nkvh, ne03, stream, label);
    } else {
        ev = fattn_dpas_kernel<D, NR16, KG, TC, ST, false, Q8>(p, nkvh, ne03, stream, label);
    }
    if (nsplit > 1) {
        const ggml_sycl_profile_label combine_label =
            fa_dpas_profile_label(device, "fattn.compute.dpas_combine", "role=compute;path=dpas");
        ev = fattn_dpas_combine(stream, combine_label, parts, meta, params.dst, (int64_t) rows, D, nsplit);
    }
    return ev;
}

#    define FATTN_DPAS_ARGS stream, device, params, parts, meta, parts_capacity, meta_capacity, xe_cores, info

template <bool Q8>
static sycl::event fattn_dpas_dispatch(dpct::queue_ptr                    stream,
                                       int                                device,
                                       const fattn_params &               params,
                                       float *                            parts,
                                       sycl::float2 *                     meta,
                                       size_t                             parts_capacity,
                                       size_t                             meta_capacity,
                                       int                                xe_cores,
                                       ggml_sycl_fattn_dpas_launch_info * info) {
    const int D    = params.ne00;
    const int G    = params.ne02 / params.ne12;
    const int ne01 = params.ne01;

    const uintptr_t k_addr = (uintptr_t) params.K;
    const uintptr_t v_addr = (uintptr_t) params.V;
    const bool aligned16   = k_addr % 16 == 0 && v_addr % 16 == 0 && params.nb11 % 16 == 0 && params.nb12 % 16 == 0 &&
                           params.nb13 % 16 == 0 && params.nb21 % 16 == 0 && params.nb22 % 16 == 0 &&
                           params.nb23 % 16 == 0;

    // single token decode: staged K/V tiles are faster at long context (B580, 48K keys: 302 -> 247 us per layer)
    if (D == 256 && ne01 == 1 && G <= 16 && aligned16) {
        return fattn_dpas_case<256, 1, 8, 1, 1, Q8>(FATTN_DPAS_ARGS);
    }
    const bool rows16 = ne01 * G <= 16;
    // q8_0 at head size 128: K/V tiles staged in SLM by coalesced dword loads. B580, Llama 3.1 8B tg128 at 16K / 64K:
    // 41.0 / 18.8 -> 45.5 / 22.5 t/s. Not for q4_0 (72-byte rows: per-lane reads win, staging -25..-40%) or other sizes.
    if constexpr (Q8) {
        const bool aligned4 = k_addr % 4 == 0 && v_addr % 4 == 0 && params.nb11 % 4 == 0 && params.nb12 % 4 == 0 &&
                              params.nb13 % 4 == 0 && params.nb21 % 4 == 0 && params.nb22 % 4 == 0 &&
                              params.nb23 % 4 == 0;
        if (D == 128 && aligned4) {
            return rows16 ? fattn_dpas_case<128, 1, 16, 1, 2, Q8>(FATTN_DPAS_ARGS) :
                            fattn_dpas_case<128, 2, 8, 2, 2, Q8>(FATTN_DPAS_ARGS);
        }
    }
    // 256-key tiles for one row tile; 128 at head size 512 so S and the Q copy stay in SLM for the planned waves
#    define FATTN_DPAS_D(D_, KG16_)                                                    \
        case D_:                                                                       \
            return rows16 ? fattn_dpas_case<D_, 1, KG16_, 1, 0, Q8>(FATTN_DPAS_ARGS) : \
                            fattn_dpas_case<D_, 2, 8, 2, 0, Q8>(FATTN_DPAS_ARGS);
    switch (D) {
        FATTN_DPAS_D(64, 16)
        FATTN_DPAS_D(96, 16)
        FATTN_DPAS_D(128, 16)
        FATTN_DPAS_D(192, 16)
        FATTN_DPAS_D(256, 16)
        FATTN_DPAS_D(512, 8)
        default:
            GGML_ABORT("fattn-dpas: unsupported head size %d", D);
    }
#    undef FATTN_DPAS_D
}

#    undef FATTN_DPAS_ARGS

sycl::event ggml_sycl_fattn_dpas_launch(dpct::queue_ptr                    stream,
                                        int                                device,
                                        const fattn_params &               params,
                                        float *                            parts,
                                        sycl::float2 *                     meta,
                                        size_t                             parts_capacity,
                                        size_t                             meta_capacity,
                                        int                                xe_cores,
                                        ggml_sycl_fattn_dpas_launch_info * info) {
    if (params.K_type == GGML_TYPE_Q8_0) {
        return fattn_dpas_dispatch<true>(stream, device, params, parts, meta, parts_capacity, meta_capacity, xe_cores,
                                         info);
    }
    return fattn_dpas_dispatch<false>(stream, device, params, parts, meta, parts_capacity, meta_capacity, xe_cores,
                                      info);
}

#else   // GGML_SYCL_FA_DPAS_NO_KERNELS

sycl::event ggml_sycl_fattn_dpas_launch(dpct::queue_ptr                    stream,
                                        int                                device,
                                        const fattn_params &               params,
                                        float *                            parts,
                                        sycl::float2 *                     meta,
                                        size_t                             parts_capacity,
                                        size_t                             meta_capacity,
                                        int                                xe_cores,
                                        ggml_sycl_fattn_dpas_launch_info * info) {
    GGML_UNUSED(stream);
    GGML_UNUSED(device);
    GGML_UNUSED(params);
    GGML_UNUSED(parts);
    GGML_UNUSED(meta);
    GGML_UNUSED(parts_capacity);
    GGML_UNUSED(meta_capacity);
    GGML_UNUSED(xe_cores);
    GGML_UNUSED(info);
    GGML_ABORT("fattn-dpas: this build compiled the DPAS kernels out (GGML_SYCL_FA_DPAS_NO_KERNELS)");
}

#endif  // GGML_SYCL_FA_DPAS_NO_KERNELS
