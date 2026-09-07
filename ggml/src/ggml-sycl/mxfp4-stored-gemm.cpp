// See mxfp4-stored-gemm.hpp for the file-level design note (llama.cpp-vtfs,
// option C step 2 -- plan Task G4). No production dispatch is touched here;
// ggml-sycl.cpp does not reference this file.

#include "mxfp4-stored-gemm.hpp"

#include "common.hpp"
#include "ggml-quants.h"
#include "ggml.h"
#include "mem-handle.hpp"      // ggml_sycl::mem_handle, retain_handles_until_event
#include "unified-cache.hpp"   // ggml_sycl::unified_allocate, alloc_request, alloc_role, runtime_category
#include "unified-kernel.hpp"  // GGML_SYCL_ESIMD_AVAILABLE, esimd/xmx aliases, SYCL_ESIMD_KERNEL

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ggml_sycl_mxfp4_stored_gemm {

q8_1_activation_pack quantize_activations_q8_1(const float * x, int64_t M, int64_t K) {
    GGML_ASSERT(x != nullptr && M > 0 && K > 0 && K % QK8_1 == 0);
    const int64_t nb = K / QK8_1;

    q8_1_activation_pack pack;
    pack.qs.resize((size_t) (M * K));
    pack.scales.resize((size_t) (M * nb));

    std::vector<block_q8_1> blocks((size_t) nb);
    for (int64_t m = 0; m < M; ++m) {
        quantize_row_q8_1_ref(x + m * K, blocks.data(), K);
        for (int64_t b = 0; b < nb; ++b) {
            const block_q8_1 & blk = blocks[(size_t) b];
            std::memcpy(pack.qs.data() + (size_t) (m * K + b * QK8_1), blk.qs, (size_t) QK8_1);
            // block_q8_1's `d`/`s` union member names depend on which
            // GGML_COMMON_DECL_* branch won in THIS translation unit (see
            // ggml-common.h's GGML_COMMON_AGGR_S/_U) -- this backend's own
            // headers (pulled in via common.hpp above) can already have
            // resolved ggml-common.h under a different branch than this
            // file's own `#include "ggml-quants.h"` (GGML_COMMON_DECL_C)
            // requests, since ggml-common.h is included-once keyed on
            // GGML_COMMON_DECL. Read the leading 2 bytes by LAYOUT instead
            // of by member name: `d` (ggml_half, fp16) is always the first
            // 2 bytes of block_q8_1 regardless of which branch won -- only
            // the accessor name changes, never the memory layout. Already
            // pinned tree-wide by ggml-common.h:269's own
            // `static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_half) +
            // QK8_1, "wrong q8_1 block size/padding")` -- a duplicate
            // assert here would check nothing that assert does not already
            // guarantee build-wide (llama.cpp-6f73 c-py5n nit, correcting
            // round 1's redundant local static_assert).
            ggml_fp16_t d_bits;
            std::memcpy(&d_bits, &blk, sizeof(d_bits));
            pack.scales[(size_t) (m * nb + b)] = ggml_fp16_to_fp32(d_bits);
        }
    }
    return pack;
}

}  // namespace ggml_sycl_mxfp4_stored_gemm

#if GGML_SYCL_ESIMD_AVAILABLE

// SYCL kernel-name types: deliberately declared at FILE SCOPE (external
// linkage), NOT inside the anonymous namespace below, matching mmvq.cpp's
// own convention (e.g. `template <int Repeat> struct
// mxfp4_dpas_down_single_col_kernel;`) -- a kernel-name type needs external
// linkage for the SYCL runtime to identify it, and llama.cpp-6cgq is the
// precedent for what goes wrong with kernel-name handling: there, two TUs
// defined the SAME kernel name identically, causing a link-time collision.
// Both names are unique to this file (verified against the rest of the
// tree), so external linkage here does not reintroduce that hazard.
//
// TWO kernel names now (llama.cpp-kcya round 6, replacing the single
// K_PARTITIONS/SLM kernel rounds 3-5 used): a "partial" kernel that streams
// one contiguous K-tile sub-range per independent work-item with no
// barrier/SLM, and a tiny "combine" kernel that sums each N-tile's
// partitions and applies the boundary-checked store -- the exact
// partial/combine split mxfp4_pair_glu_xmx_tiled_dpas_m2_ksplit_sycl /
// _combine_sycl (mmvq.cpp) already uses in production, for the same reason
// (see the design note above mxfp4_soa_gemm_int8_dpas_partial_launch below).
template <int M_TILE> struct mxfp4_soa_gemm_int8_dpas_partial_kernel;
template <int M_TILE> struct mxfp4_soa_gemm_int8_dpas_combine_kernel;

namespace {

// Private re-derivations of mmvq.cpp's mxfp4_e8m0_to_fp32_half_esimd /
// mxfp4_code_values_esimd bit-tricks (mmvq.cpp:~7370-7405), renamed and
// confined to this TU's anonymous namespace. The two symbols do NOT share
// one linkage story (llama.cpp-6f73 c-py5n should-fix 1, correcting the
// round-1 comment): mxfp4_e8m0_to_fp32_half_esimd is a plain, non-static,
// non-template file-scope function at mmvq.cpp:7368, outside that file's
// own anonymous namespace at :823-957 -- ordinary external linkage, and this
// file COULD declare an extern prototype and link against the exact symbol
// mmvq.cpp's object file emits. mxfp4_code_values_esimd (mmvq.cpp:7390-7391)
// is instead an `inline` function TEMPLATE: its instantiations are formally
// external linkage too, but the compiler is free to discard (COMDAT-fold or
// simply never emit) any instantiation mmvq.cpp's own TU does not itself
// use with matching template arguments -- there is no single, guaranteed
// object-file symbol here to link against the way there is for the plain
// function, only the template DEFINITION, which is not declared in any
// shared header. Either way, the reason for a private copy is the same:
// neither symbol is declared in any shared header, so reaching them from
// here would mean hand-declaring a prototype (or including this file's
// private definition) for another TU's internal helper -- worse than a
// small, self-contained re-derivation this TU owns outright. Confining the
// copy to this TU's own anonymous namespace rules out any duplicate-symbol
// collision either way, regardless of which linkage story applies to which
// original. The formulas are reproduced verbatim -- E8M0 "halved"
// convention to match GGML_E8M0_TO_FP32_HALF, and the same e2m1
// magnitude/sign decomposition for the 4-bit code table.
//
// The scalar E8M0 decode CANNOT reuse this backend's own
// sycl_e8m0_to_fp32_half (common.hpp) here: that helper's `memcpy` is a
// non-ESIMD SYCL device call, and this kernel runs in an ESIMD context
// (SYCL_ESIMD_KERNEL) -- "SYCL device function cannot be called from an
// ESIMD context" is a hard compile error, not a style choice. ESIMD's own
// `simd<uint32_t,1>::bit_cast_view<float>()` is the ESIMD-legal
// reinterpret-bits path, exactly what mmvq.cpp's own scalar ESIMD decode
// already uses.
SYCL_ESIMD_FUNCTION inline float mxfp4_stored_gemm_e8m0_half_esimd(uint8_t e) {
    using namespace sycl::ext::intel::esimd;
    uint32_t bits;
    if (e < 2) {
        bits = 0x00200000u << e;
    } else {
        bits = static_cast<uint32_t>(e - 1) << 23;
    }
    simd<uint32_t, 1> packed(bits);
    return packed.template bit_cast_view<float>()[0];
}

template <int N>
SYCL_ESIMD_FUNCTION inline sycl::ext::intel::esimd::simd<int8_t, N> mxfp4_stored_gemm_code_values_esimd(
    sycl::ext::intel::esimd::simd<uint8_t, N> codes) {
    using namespace sycl::ext::intel::esimd;
    simd<uint8_t, N> base_mag = codes & uint8_t{ 7 };
    simd<uint8_t, N> extra    = base_mag - uint8_t{ 4 };
    extra.merge(simd<uint8_t, N>(0), base_mag <= uint8_t{ 4 });
    simd<uint8_t, N> mag = base_mag + extra;
    mag.merge(simd<uint8_t, N>(12), base_mag == uint8_t{ 7 });

    simd<int8_t, N> values = mag;
    simd<int8_t, N> neg    = -values;
    values.merge(neg, (codes & uint8_t{ 8 }) != uint8_t{ 0 });
    return values;
}

// Software-prefetch a 64B-aligned cache line ahead of a streaming read.
// Private re-derivation of mmvq.cpp's mxfp4_xmx_tiled_prefetch_line
// (mmvq.cpp:7662), same reasoning as the two helpers above for why this is
// a copy rather than a shared declaration: not declared in any header, and
// an ESIMD-context function, so it cannot be reached from outside its own
// TU's anonymous namespace without one.
SYCL_ESIMD_FUNCTION inline void mxfp4_stored_gemm_prefetch_line(const uint8_t * ptr) {
    using namespace sycl::ext::intel::esimd;
    const uintptr_t  aligned_addr = reinterpret_cast<uintptr_t>(ptr) & ~uintptr_t{ 63 };
    const uint32_t * aligned_ptr  = reinterpret_cast<const uint32_t *>(aligned_addr);
    constexpr auto   props =
        properties{ cache_hint_L1<cache_hint::streaming>, cache_hint_L2<cache_hint::uncached>, alignment<64> };
    prefetch<uint32_t, 16>(aligned_ptr, 0, simd_mask<1>(1), props);
}

// llama.cpp-kcya: how many independent work-items jointly stream the K
// reduction for ONE N-tile. Deliberately NOT reusing mmvq.cpp's
// ggml_sycl_mxfp4_gateup_ksplit(): that accessor clamps to [1,4] because its
// OWN base geometry (total_batches * m_tile_pairs, the MoE gate/up decode
// launch size) is already large for realistic traffic -- ksplit there only
// compensates a wider B70 vs the B50 baseline. This kernel's base geometry
// is a SINGLE expert's N-tiles (n_tiles = ceil(n_out/16), 180 at N=2880)
// with no batch axis to multiply by, so reaching the same occupancy floor
// needs an order of magnitude more split -- reusing the [1,4]-clamped
// accessor here would reproduce exactly the "~1.4 threads/CU" starvation
// round 1 already diagnosed (llama.cpp-kcya description; round 3's
// K_PARTITIONS=8-per-work-group design was the last attempt to compensate
// for it without touching this number).
//
// target_threads = compute_units * THREADS_PER_CU * WAVES: this is
// llama.cpp-kcya's own framing ("the B50's 128 CUs x 8 threads have >= 2
// waves of work") turned into arithmetic, generalised to whatever device
// this call actually lands on (256 CU on the B70) rather than hardcoding
// the B50 figure. ksplit is the smallest split of n_tiles that reaches that
// many independent work-items -- floored at 1 (no split, direct write) and
// capped at k_tiles (a partition with zero k-tiles to reduce contributes
// nothing but launch overhead).
//
// GGML_SYCL_STORED_GEMM_KSPLIT overrides the computed value outright (any
// positive integer, clamped to [1, k_tiles]) -- the fast, no-rebuild knob
// this task's hardware round-trips are expected to sweep; see this
// commit's body for the values already tried.
int mxfp4_stored_gemm_ksplit_for(sycl::queue & queue, int64_t n_tiles, int64_t k_tiles) {
    if (n_tiles <= 0 || k_tiles <= 0) {
        return 1;
    }
    if (const char * env = std::getenv("GGML_SYCL_STORED_GEMM_KSPLIT")) {
        char *     end    = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end != env && parsed > 0) {
            return (int) std::min<long>(parsed, k_tiles);
        }
    }

    constexpr int THREADS_PER_CU = 8;
    constexpr int WAVES          = 2;
    constexpr int FALLBACK_CU    = 128;  // B50 CU count -- used only if the device query fails

    const int device        = ggml_sycl_get_device_id_from_queue(queue);
    uint32_t  compute_units = 0;
    if (device >= 0 && device < ggml_sycl_info().device_count) {
        compute_units = ggml_sycl_info().devices[device].xmx_caps.compute_units;
    }
    const int64_t target_threads = (int64_t) (compute_units > 0 ? compute_units : FALLBACK_CU) * THREADS_PER_CU * WAVES;

    int64_t ksplit = (target_threads + n_tiles - 1) / n_tiles;  // ceil
    ksplit         = std::max<int64_t>(ksplit, 1);
    ksplit         = std::min<int64_t>(ksplit, k_tiles);
    return (int) ksplit;
}

// Thread-local scratch cache for the partial pass's per-partition sums.
// Mirrors mxfp4_moe_gateup_ksplit_get_or_alloc_scratch /
// _scratch_mark_ready (mmvq.cpp, llama.cpp-lis9) exactly, renamed to this
// file's own symbols: SYCL Memory Ownership (CLAUDE.md) forbids
// sycl::malloc_device / side caches outside the unified-cache allocator, so
// this cache's mem_handle is the allocation's sole owner for as long as it
// is kept, and growth retires the old handle against the outstanding
// combine kernel's event (via retain_handles_until_event) rather than
// dropping it while that kernel may still be reading it.
struct mxfp4_stored_gemm_ksplit_scratch {
    size_t                capacity     = 0;
    int                   owner_device = -1;
    ggml_sycl::mem_handle handle;
    sycl::event           ready_event     = {};
    bool                  ready_event_set = false;
};

thread_local mxfp4_stored_gemm_ksplit_scratch g_mxfp4_stored_gemm_ksplit_scratch;

float * mxfp4_stored_gemm_ksplit_get_or_alloc_scratch(sycl::queue * stream, int device, size_t required_floats) {
    const size_t required_bytes = required_floats * sizeof(float);
    if (required_bytes == 0) {
        return nullptr;
    }
    auto & cache = g_mxfp4_stored_gemm_ksplit_scratch;
    if (cache.handle.valid() && cache.capacity >= required_bytes && cache.owner_device == device) {
        // Safe without an explicit dependency on cache.ready_event for the
        // same reason the mmvq.cpp original's identical reuse path is:
        // `stream` is this backend's in-order queue, so THIS call's own
        // partial-kernel submission is already ordered after the prior
        // combine kernel that produced ready_event.
        auto resolved = cache.handle.resolve(device);
        return resolved ? reinterpret_cast<float *>(resolved.ptr) : nullptr;
    }

    ggml_sycl::mem_handle retired_owner     = cache.handle;
    sycl::event           retired_event     = cache.ready_event;
    const bool            retired_event_set = cache.ready_event_set;
    cache.handle                            = {};
    cache.capacity                          = 0;
    cache.owner_device                      = -1;
    cache.ready_event                       = {};
    cache.ready_event_set                   = false;
    if (retired_owner.valid() && retired_event_set) {
        std::vector<ggml_sycl::mem_handle> retired_handles;
        retired_handles.push_back(std::move(retired_owner));
        ggml_sycl::retain_handles_until_event(std::move(retired_handles), std::move(retired_event));
    }

    ggml_sycl::alloc_request req{};
    req.queue                          = stream;
    req.device                         = device;
    req.size                           = required_bytes;
    req.intent.role                    = ggml_sycl::alloc_role::EXPERT_STAGING;
    req.intent.category                = ggml_sycl::runtime_category::STAGING;
    req.intent.constraints.must_device = true;
    req.alignment = 64;  // 64 divides acc_width*sizeof(float) (64*M_TILE bytes) for every M_TILE this file
                         // uses, so every partition's offset into the scratch buffer stays 64-byte aligned
    cache.handle  = ggml_sycl::unified_allocate(req);
    if (!cache.handle.valid()) {
        cache.handle = {};
        return nullptr;
    }
    auto resolved = cache.handle.resolve(device);
    if (!resolved || !resolved.ptr || !resolved.on_device) {
        cache.handle = {};
        return nullptr;
    }
    cache.capacity     = cache.handle.size();
    cache.owner_device = cache.handle.device();
    return reinterpret_cast<float *>(resolved.ptr);
}

void mxfp4_stored_gemm_ksplit_scratch_mark_ready(int device, const sycl::event & event) {
    auto & cache = g_mxfp4_stored_gemm_ksplit_scratch;
    if (!cache.handle.valid() || cache.owner_device != device) {
        return;
    }
    cache.ready_event     = event;
    cache.ready_event_set = true;
}

// llama.cpp-kcya round 6 (parent llama.cpp-6f73, plan Task G4): replaces
// round 3-5's K_PARTITIONS-per-work-group + SLM tree reduction with the
// structure mxfp4_pair_glu_xmx_tiled_dpas_m2_ksplit_sycl /
// mxfp4_pair_glu_xmx_tiled_dpas_m2_combine_sycl (mmvq.cpp) already proves in
// production: single-item work-groups (no barrier, no SLM), each streaming
// ONE contiguous K-tile sub-range with deep software prefetch, writing a
// PARTIAL sum; a separate tiny "combine" kernel sums the partitions per
// N-tile and applies the boundary-checked store.
//
// WHY round 3-5's design plateaued at 15% instead of hitting >=50%: its
// launch geometry (n_tiles work-groups of K_PARTITIONS=8 work-items each)
// gave 1440 threads total on the B50, which sounds like enough -- but every
// one of those 8 threads in a work-group had to reach a `barrier()` every
// iteration before any of them could proceed (the SLM tree-reduction body
// this design deletes). A barrier forces the SLOWEST of the 8 threads'
// current memory request to complete before ANY of the 8 can issue their
// next one -- so the work-group's effective memory-level parallelism (MLP)
// was capped at whatever ONE iteration's worth of outstanding loads looks
// like, repeated k_tiles/K_PARTITIONS times, rather than the full k_tiles
// depth all 8 threads could otherwise have in flight independently. Rounds
// 4-5 (batched loads, then software prefetch) both tried to widen that
// single barrier-bounded window and both REGRESSED -- consistent with the
// window, not raw per-load latency, being the actual ceiling.
//
// This design removes the barrier and the SLM entirely: every work-item is
// now single-item (local_range=1) and independent end-to-end, so the GPU's
// memory subsystem can have as many outstanding requests in flight as there
// are RESIDENT threads across all CUs, not just within one 8-thread
// work-group. Software prefetch (distance=10, mirroring
// mxfp4_pair_glu_xmx_tiled_dpas_m2_k_reduce's own Prefetch=true arm) then
// hides each thread's OWN load latency on top of that -- the same two
// techniques the ~195 GB/s reference kernel combines, adapted to this
// file's STORED SOA addressing (16 independent per-row byte streams instead
// of m2's single interleaved "group" stream -- see this file's header
// comment and mxfp4_soa_load_a_vec, mmvq.cpp:7566, for why that addressing
// is unchanged here: this kernel exists specifically to prove DPAS
// generalises to the layout production dispatch will actually need to
// read, not to a decode-optimised repacking of it).
//
// Occupancy arithmetic (see mxfp4_stored_gemm_ksplit_for above) and the
// hardware numbers this round measures against are in the commit body.
template <int M_TILE>
static sycl::event mxfp4_soa_gemm_int8_dpas_partial_launch(sycl::queue &                    queue,
                                                           const uint8_t *                  soa_base,
                                                           const int8_t *                   act_qs,
                                                           const float *                    act_scales,
                                                           float *                          out,
                                                           int                              n_out,
                                                           int                              n_k,
                                                           int64_t                          n_tiles,
                                                           int64_t                          k_tiles,
                                                           int                              ksplit,
                                                           bool                             write_direct,
                                                           const std::vector<sycl::event> & deps) {
    constexpr int exec_n            = (int) GGML_SYCL_MXFP4_MOE_XMX_N;  // 16, DPAS execution-size (weight/N tile)
    constexpr int k_per             = (int) GGML_SYCL_MXFP4_MOE_XMX_K;  // 32, QK_MXFP4
    constexpr int packed_bytes      = k_per / 2;                        // 16 nibble-packed bytes per block
    constexpr int an                = M_TILE * k_per;                   // activation "A" operand size
    constexpr int bn                = k_per * exec_n;                   // weight "B" operand size (VNNI-packed)
    constexpr int acc_width         = M_TILE * exec_n;                  // per-partition accumulator width
    constexpr int prefetch_distance = 10;  // matches mxfp4_pair_glu_xmx_tiled_dpas_m2_k_reduce's Prefetch arm

    const int64_t row_qs_stride    = k_tiles * packed_bytes;
    const int64_t total_qs_size    = (static_cast<int64_t>(n_k) / 2) * static_cast<int64_t>(n_out);
    const int64_t act_row_stride_q = static_cast<int64_t>(n_k);
    const int64_t act_row_stride_s = k_tiles;
    const int64_t k_tiles_base     = k_tiles / ksplit;
    const int64_t k_tiles_rem      = k_tiles % ksplit;
    const int64_t launch_size      = n_tiles * static_cast<int64_t>(ksplit);

    ggml_sycl_profile_label profile_label{};
    profile_label.name         = "mxfp4.stored_gemm.soa.partial";
    profile_label.category     = "mxfp4";
    profile_label.queue_kind   = "compute";
    const std::string metadata = "M=" + std::to_string(M_TILE) + ";n_out=" + std::to_string(n_out) +
                                 ";n_k=" + std::to_string(n_k) + ";ksplit=" + std::to_string(ksplit);
    profile_label.metadata    = metadata.c_str();
    profile_label.device      = ggml_sycl_get_device_id_from_queue(queue);
    // Traffic this pass touches: the whole expert weight buffer and the
    // whole M x n_k activation, read once each (same DRAM-traffic reasoning
    // as ggml_sycl_mxfp4_soa_gemm_dpas's own accounting further below --
    // the activation is assumed cache-resident across the launch, not
    // re-fetched per work-item); plus EXACTLY ONE of the two things this
    // pass's own epilogue can write (mxfp4_soa_gemm_int8_dpas_launch's
    // write_direct branch decides which): when write_direct (ksplit <= 1,
    // or the scratch-alloc-refusal fallback), the M_TILE x n_out f32 write
    // straight to the caller's dst; otherwise (ksplit > 1) the
    // launch_size x acc_width f32 partial write into scratch.
    //
    // Reconciling this row against ggml_sycl_mxfp4_soa_gemm_dpas's own
    // test-facing bytes_moved figure (llama.cpp-kcya c-amir should-fix 3):
    // on the write_direct path they now match exactly (weight + act + dst,
    // the useful traffic, nothing else). On the ksplit > 1 path they do
    // NOT match by design -- this row plus
    // mxfp4_soa_gemm_int8_dpas_combine_launch's own row (below) sum to the
    // useful weight+act+dst traffic PLUS the scratch round-trip (written
    // here, read there) that only exists BECAUSE of the K-split. That
    // round-trip is real DRAM traffic this pass causes, so it belongs in
    // the profiler's bandwidth accounting; it is NOT "useful" work a
    // caller cares about, so it is deliberately excluded from
    // bytes_moved. Do not sum the two profiler rows' bytes and expect
    // ggml_sycl_mxfp4_soa_gemm_dpas's bytes_moved to fall out -- see this
    // file's own env-vars doc entry (docs/backend/sycl-env-vars.md,
    // GGML_SYCL_STORED_GEMM_DEBUG) for the same reconciliation restated
    // for a reader who has not opened this file.
    const size_t weight_bytes = (size_t) n_out * ((size_t) (n_k / 2) + (size_t) (n_k / 32));
    const size_t act_bytes    = (size_t) M_TILE * ((size_t) n_k + (size_t) (n_k / 32) * sizeof(float));
    const size_t dst_bytes    = (size_t) M_TILE * (size_t) n_out * sizeof(float);
    const size_t scratch_write_bytes =
        write_direct ? (size_t) 0 : (size_t) launch_size * (size_t) acc_width * sizeof(float);
    profile_label.bytes = weight_bytes + act_bytes + scratch_write_bytes + (write_direct ? dst_bytes : (size_t) 0);

    return ggml_sycl_profile_submit(queue, profile_label, [&](sycl::queue & profiled_queue) {
        return profiled_queue.submit([&](sycl::handler & h) {
            if (!deps.empty()) {
                h.depends_on(deps);
            }
            h.parallel_for<mxfp4_soa_gemm_int8_dpas_partial_kernel<M_TILE>>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(launch_size)), sycl::range<1>(1)),
                [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                    using namespace sycl::ext::intel::esimd;
                    const int64_t global_idx = static_cast<int64_t>(item.get_global_id(0));
                    const int64_t n_tile     = global_idx / ksplit;
                    const int     k_part     = static_cast<int>(global_idx - n_tile * static_cast<int64_t>(ksplit));
                    const int64_t n_base     = n_tile * exec_n;

                    // Near-equal contiguous partition of [0, k_tiles) into
                    // `ksplit` parts; the first (k_tiles % ksplit) parts
                    // absorb the remainder tile -- same scheme
                    // mxfp4_pair_glu_xmx_tiled_dpas_m2_ksplit_sycl uses.
                    const int64_t kt_start =
                        static_cast<int64_t>(k_part) * k_tiles_base + std::min<int64_t>(k_part, k_tiles_rem);
                    const int64_t kt_count = k_tiles_base + (k_part < k_tiles_rem ? 1 : 0);
                    const int64_t kt_end   = kt_start + kt_count;

                    simd<float, acc_width> acc = 0.0f;

                    for (int64_t kt = kt_start; kt < kt_end; ++kt) {
                        const int64_t kt_prefetch = kt + prefetch_distance;
                        if (kt_prefetch < kt_end) {
                        // Activation: M_TILE independent streams, each
                        // row's qs bytes contiguous across kt.
#    pragma unroll
                            for (int r = 0; r < M_TILE; ++r) {
                                mxfp4_stored_gemm_prefetch_line(reinterpret_cast<const uint8_t *>(
                                    act_qs + r * act_row_stride_q + kt_prefetch * k_per));
                            }
                        // Weight: exec_n independent streams (rows),
                        // each row's packed qs bytes AND its E8M0 scale
                        // bytes both contiguous across kt (block_index =
                        // row * n_k_blocks + k_block) -- so one line per
                        // row per pointer is the whole prefetch, no
                        // cross-row coalescing to attempt.
#    pragma unroll
                            for (int n = 0; n < exec_n; ++n) {
                                const int64_t row = n_base + n;
                                if (row < n_out) {
                                    mxfp4_stored_gemm_prefetch_line(soa_base + row * row_qs_stride +
                                                                    kt_prefetch * packed_bytes);
                                    mxfp4_stored_gemm_prefetch_line(soa_base + total_qs_size + row * k_tiles +
                                                                    kt_prefetch);
                                }
                            }
                        }

                        // Activation "A" operand: M_TILE rows x 32 int8
                        // q8_1 codes, plain row-major (no VNNI packing for
                        // A). y_scale stays a plain C array (not simd<>) for
                        // the same reason the round-3 kernel's did: it is
                        // only ever read back one scalar element at a time.
                        simd<int8_t, an> a_vec;
                        float            y_scale[M_TILE];
#    pragma unroll
                        for (int r = 0; r < M_TILE; ++r) {
                            const int8_t * qs_ptr                      = act_qs + r * act_row_stride_q + kt * k_per;
                            a_vec.template select<k_per, 1>(r * k_per) = block_load<int8_t, k_per>(qs_ptr);
                            y_scale[r]                                 = act_scales[r * act_row_stride_s + kt];
                        }

                        // Weight "B" operand: 16 output rows x 32 MXFP4
                        // elements, decoded from the raw stored SOA bytes
                        // and VNNI-packed ((kk/4)*exec_n*4 + n*4 + (kk%4)).
                        // Rows past n_out are left as zero (both codes and
                        // scale), matching mxfp4_soa_load_a_vec's boundary
                        // check.
                        simd<int8_t, bn>    b_vec   = 0;
                        simd<float, exec_n> w_scale = 0.0f;
#    pragma unroll
                        for (int n = 0; n < exec_n; ++n) {
                            const int64_t row = n_base + n;
                            if (row < n_out) {
                                const uint8_t * packed_ptr         = soa_base + row * row_qs_stride + kt * packed_bytes;
                                simd<uint8_t, packed_bytes> packed = block_load<uint8_t, packed_bytes>(packed_ptr);
                                simd<uint8_t, k_per>        codes;
                                codes.template select<packed_bytes, 1>(0)            = packed & uint8_t{ 0x0f };
                                codes.template select<packed_bytes, 1>(packed_bytes) = packed >> 4;
                                simd<int8_t, k_per> vals = mxfp4_stored_gemm_code_values_esimd<k_per>(codes);
#    pragma unroll
                                for (int kk = 0; kk < k_per; ++kk) {
                                    b_vec[(kk / 4) * exec_n * 4 + n * 4 + (kk % 4)] = vals[kk];
                                }
                                // Plain scalar dereference, not block_load:
                                // see the identical reasoning at
                                // llama.cpp-6f73 c-gngd (kept from round 3 --
                                // this address is not guaranteed 4-byte
                                // aligned, so block_load<uint8_t,1> would be
                                // the questionable pattern here too).
                                const uint8_t scale_byte = soa_base[total_qs_size + row * k_tiles + kt];
                                w_scale[n]               = mxfp4_stored_gemm_e8m0_half_esimd(scale_byte);
                            }
                        }

                        // Both weight and activation scales vary per K
                        // block, so the multiply happens INSIDE this loop,
                        // per k-tile.
                        simd<int, acc_width> part = 0;
                        part                      = xmx::dpas<8, M_TILE, int, int, int8_t, int8_t>(part, b_vec, a_vec);

#    pragma unroll
                        for (int r = 0; r < M_TILE; ++r) {
                            simd<int, exec_n>   row_i = part.template select<exec_n, 1>(r * exec_n);
                            simd<float, exec_n> row_f = convert<float>(row_i) * (w_scale * y_scale[r]);
                            acc.template select<exec_n, 1>(r * exec_n) =
                                acc.template select<exec_n, 1>(r * exec_n) + row_f;
                        }
                    }

                    if (write_direct) {
                    // ksplit == 1: this partition alone covers the
                    // whole K reduction for its N-tile, so its own
                    // accumulator IS the final answer -- write straight
                    // to the caller's dst, matching round 3's original
                    // (unsplit) epilogue exactly.
#    pragma unroll
                        for (int r = 0; r < M_TILE; ++r) {
#    pragma unroll
                            for (int n = 0; n < exec_n; ++n) {
                                const int64_t row = n_base + n;
                                if (row < n_out) {
                                    simd<float, 1> value = acc.template select<1, 1>(r * exec_n + n);
                                    block_store<float, 1>(out + (int64_t) r * n_out + row, value);
                                }
                            }
                        }
                    } else {
                        block_store<float, acc_width>(out + global_idx * acc_width, acc);
                    }
                });
        });
    });
}

// Combine pass: launches `n_tiles` single-item work-groups (one per N-tile,
// the same geometry the ksplit==1 direct-write path uses) that sum each
// N-tile's `ksplit` partials and apply the boundary-checked store. Mirrors
// mxfp4_pair_glu_xmx_tiled_dpas_m2_combine_sycl's structure.
template <int M_TILE>
static sycl::event mxfp4_soa_gemm_int8_dpas_combine_launch(sycl::queue &       queue,
                                                           const float *       partial,
                                                           float *             dst,
                                                           int                 n_out,
                                                           int64_t             n_tiles,
                                                           int                 ksplit,
                                                           const sycl::event & partial_event) {
    constexpr int exec_n    = (int) GGML_SYCL_MXFP4_MOE_XMX_N;
    constexpr int acc_width = M_TILE * exec_n;

    ggml_sycl_profile_label profile_label{};
    profile_label.name       = "mxfp4.stored_gemm.soa.combine";
    profile_label.category   = "mxfp4";
    profile_label.queue_kind = "compute";
    const std::string metadata =
        "M=" + std::to_string(M_TILE) + ";n_out=" + std::to_string(n_out) + ";ksplit=" + std::to_string(ksplit);
    profile_label.metadata          = metadata.c_str();
    profile_label.device            = ggml_sycl_get_device_id_from_queue(queue);
    // See mxfp4_soa_gemm_int8_dpas_partial_launch's own bytes comment
    // (above) for the full reconciliation: this row's scratch_read_bytes
    // is the other half of that pass's scratch_write_bytes, together the
    // K-split's real DRAM round-trip -- traffic this dispatch causes but
    // that ggml_sycl_mxfp4_soa_gemm_dpas's own bytes_moved deliberately
    // excludes as not "useful" work.
    const size_t scratch_read_bytes = (size_t) n_tiles * (size_t) ksplit * (size_t) acc_width * sizeof(float);
    const size_t dst_bytes          = (size_t) M_TILE * (size_t) n_out * sizeof(float);
    profile_label.bytes             = scratch_read_bytes + dst_bytes;

    return ggml_sycl_profile_submit(queue, profile_label, [&](sycl::queue & profiled_queue) {
        return profiled_queue.submit([&](sycl::handler & h) {
            h.depends_on(partial_event);
            h.parallel_for<mxfp4_soa_gemm_int8_dpas_combine_kernel<M_TILE>>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(n_tiles)), sycl::range<1>(1)),
                [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                    using namespace sycl::ext::intel::esimd;
                    const int64_t n_tile = static_cast<int64_t>(item.get_global_id(0));
                    const int64_t n_base = n_tile * exec_n;

                    simd<float, acc_width> acc       = 0.0f;
                    const float *          part_base = partial + n_tile * static_cast<int64_t>(ksplit) * acc_width;
                    for (int k_part = 0; k_part < ksplit; ++k_part) {
                        simd<float, acc_width> p =
                            block_load<float, acc_width>(part_base + (int64_t) k_part * acc_width);
                        acc = acc + p;
                    }

#    pragma unroll
                    for (int r = 0; r < M_TILE; ++r) {
#    pragma unroll
                        for (int n = 0; n < exec_n; ++n) {
                            const int64_t row = n_base + n;
                            if (row < n_out) {
                                simd<float, 1> value = acc.template select<1, 1>(r * exec_n + n);
                                block_store<float, 1>(dst + (int64_t) r * n_out + row, value);
                            }
                        }
                    }
                });
        });
    });
}

}  // namespace

// Top-level dispatcher: computes the launch geometry, then either runs a
// single direct-write partial pass (ksplit == 1, or scratch allocation
// failed) or the two-pass partial+combine split. Kept in this file's own
// (non-anonymous-namespace) scope like the rest of this section -- still
// `static`, internal linkage, not part of this file's public surface.
template <int M_TILE>
static sycl::event mxfp4_soa_gemm_int8_dpas_launch(sycl::queue &                    queue,
                                                   const uint8_t *                  soa_base,
                                                   const int8_t *                   act_qs,
                                                   const float *                    act_scales,
                                                   float *                          dst,
                                                   int                              n_out,
                                                   int                              n_k,
                                                   const std::vector<sycl::event> & deps) {
    constexpr int exec_n = (int) GGML_SYCL_MXFP4_MOE_XMX_N;
    constexpr int k_per  = (int) GGML_SYCL_MXFP4_MOE_XMX_K;

    const int64_t k_tiles = n_k / k_per;
    const int64_t n_tiles = (static_cast<int64_t>(n_out) + exec_n - 1) / exec_n;
    const int     ksplit  = mxfp4_stored_gemm_ksplit_for(queue, n_tiles, k_tiles);

    if (std::getenv("GGML_SYCL_STORED_GEMM_DEBUG")) {
        const int device        = ggml_sycl_get_device_id_from_queue(queue);
        uint32_t  compute_units = 0;
        if (device >= 0 && device < ggml_sycl_info().device_count) {
            compute_units = ggml_sycl_info().devices[device].xmx_caps.compute_units;
        }
        // combine_work_items is 0 when ksplit <= 1: the ksplit <= 1 branch
        // below never submits a combine kernel at all, so printing n_tiles
        // here would advertise a launch that does not happen (llama.cpp-kcya
        // c-amir nit 6) -- this print exists specifically to read geometry
        // without running the binary, so it should not misstate it. A
        // scratch-allocation refusal on the ksplit > 1 path is a SEPARATE,
        // not-yet-known-here fallback to the same no-combine shape -- see
        // the GGML_LOG_WARN a few lines below, which fires only if that
        // actually happens.
        const long long combine_work_items = (ksplit <= 1) ? 0 : (long long) n_tiles;
        std::fprintf(stderr,
                     "[mxfp4-stored-gemm] M_TILE=%d n_out=%d n_k=%d compute_units=%u n_tiles=%lld k_tiles=%lld "
                     "ksplit=%d partial_work_items=%lld combine_work_items=%lld k_tiles_per_partition(max)=%lld\n",
                     M_TILE, n_out, n_k, compute_units, (long long) n_tiles, (long long) k_tiles, ksplit,
                     (long long) (n_tiles * ksplit), combine_work_items, (long long) ((k_tiles + ksplit - 1) / ksplit));
    }

    if (ksplit <= 1) {
        return mxfp4_soa_gemm_int8_dpas_partial_launch<M_TILE>(queue, soa_base, act_qs, act_scales, dst, n_out, n_k,
                                                               n_tiles, k_tiles, /*ksplit=*/1, /*write_direct=*/true,
                                                               deps);
    }

    const int    device          = ggml_sycl_get_device_id_from_queue(queue);
    const size_t required_floats = (size_t) n_tiles * (size_t) ksplit * (size_t) (M_TILE * exec_n);
    float *      scratch         = mxfp4_stored_gemm_ksplit_get_or_alloc_scratch(&queue, device, required_floats);
    if (!scratch) {
        // Allocator refusal / out of budget: fall back to a single,
        // unsplit pass writing straight to dst rather than losing the
        // dispatch outright -- mirrors
        // mxfp4_pair_glu_xmx_tiled_dpas_m2_ksplit_submit's identical
        // fallback in mmvq.cpp. WARN (not INFO -- INFO is dropped at
        // default verbosity, CLAUDE.md) because this is a MEASUREMENT
        // instrument, not a production dispatch path: the fallback is
        // ~2.2x slower than the split path (llama.cpp-kcya c-amir nit 7),
        // and a silent degrade here would otherwise read as a mystery
        // regression in whoever's next bandwidth table, discoverable only
        // by cross-checking the profiler metadata's ksplit=1 against the
        // GGML_SYCL_STORED_GEMM_DEBUG print's own (higher) resolved ksplit.
        GGML_LOG_WARN(
            "[mxfp4-stored-gemm] scratch allocation failed (required_floats=%zu); falling back to the unsplit "
            "direct-write path (ksplit=1) -- this launch is measurably slower than the requested ksplit=%d\n",
            required_floats, ksplit);
        return mxfp4_soa_gemm_int8_dpas_partial_launch<M_TILE>(queue, soa_base, act_qs, act_scales, dst, n_out, n_k,
                                                               n_tiles, k_tiles, /*ksplit=*/1, /*write_direct=*/true,
                                                               deps);
    }

    sycl::event partial_event =
        mxfp4_soa_gemm_int8_dpas_partial_launch<M_TILE>(queue, soa_base, act_qs, act_scales, scratch, n_out, n_k,
                                                        n_tiles, k_tiles, ksplit, /*write_direct=*/false, deps);
    sycl::event combine_event =
        mxfp4_soa_gemm_int8_dpas_combine_launch<M_TILE>(queue, scratch, dst, n_out, n_tiles, ksplit, partial_event);
    mxfp4_stored_gemm_ksplit_scratch_mark_ready(device, combine_event);
    return combine_event;
}

#endif  // GGML_SYCL_ESIMD_AVAILABLE

namespace ggml_sycl_mxfp4_stored_gemm {

sycl::event ggml_sycl_mxfp4_soa_gemm_dpas(sycl::queue &                    queue,
                                          const void *                     soa_weight_device,
                                          const int8_t *                   act_qs_device,
                                          const float *                    act_scales_device,
                                          float *                          dst_device,
                                          int                              M,
                                          int                              n_out,
                                          int                              n_k,
                                          const std::vector<sycl::event> & deps) {
    GGML_ASSERT(soa_weight_device != nullptr);
    GGML_ASSERT(act_qs_device != nullptr);
    GGML_ASSERT(act_scales_device != nullptr);
    GGML_ASSERT(dst_device != nullptr);
    GGML_ASSERT(n_out > 0);
    GGML_ASSERT(n_k > 0 && n_k % (int) GGML_SYCL_MXFP4_MOE_XMX_K == 0);
    // 32-byte-aligned base pointers, for the two device pointers the kernel
    // actually vector-loads: `block_load<uint8_t, 16>` off `soa_weight_device`
    // (the weight load) and `block_load<int8_t, 32>` off `act_qs_device`
    // (the activation load) both carry ESIMD vector-alignment requirements,
    // satisfied only because every offset the kernel computes off these two
    // bases is itself a multiple of the vector width (`row_qs_stride` and
    // `act_row_stride_q` are both multiples of 32) -- a caller passing a
    // mid-buffer sub-pointer that is not itself 32-byte aligned would get a
    // silent misaligned load. `act_scales_device` is deliberately NOT
    // asserted here (llama.cpp-6f73 c-py5n should-fix 2, correcting round
    // 1): it is only ever read by scalar subscript, never block_loaded, so
    // it carries no such alignment requirement and an assert on it would
    // guard nothing.
    GGML_ASSERT(reinterpret_cast<uintptr_t>(soa_weight_device) % 32 == 0);
    GGML_ASSERT(reinterpret_cast<uintptr_t>(act_qs_device) % 32 == 0);

#if GGML_SYCL_ESIMD_AVAILABLE
    const auto * soa_base = static_cast<const uint8_t *>(soa_weight_device);

    // llama.cpp-kcya round 6: NO outer ggml_sycl_profile_submit wrapping
    // here any more (round 3-5 had one, profile_label.name =
    // "mxfp4.stored_gemm.soa"). The kernel-profiler records a returned
    // sycl::event's OWN device command_start/command_end
    // (sycl-kernel-profiler.cpp's drain_pending_events) -- for a
    // multi-kernel dispatch that only covers the LAST kernel submitted, so
    // wrapping the switch below and returning whichever event
    // mxfp4_soa_gemm_int8_dpas_launch produces would silently under-time
    // the ksplit > 1 path (missing the partial kernel's, by far the larger,
    // share of the work) exactly the way this file's own header comments
    // warn against for the CSV's aggregate `bytes` column. The two real
    // per-launch profiler rows are "mxfp4.stored_gemm.soa.partial" and
    // "mxfp4.stored_gemm.soa.combine" (mxfp4_soa_gemm_int8_dpas_partial_launch
    // / _combine_launch above), each wrapping its own single queue.submit
    // and therefore each carrying an honest device-timed row; sum their
    // mean_ns for one logical launch's total time. When ksplit resolves to
    // 1, only the ".partial" row exists for that launch (no combine kernel
    // is submitted at all -- see mxfp4_soa_gemm_int8_dpas_launch's
    // ksplit <= 1 branch).
    //
    // Sum mean_ns, but do NOT sum the two rows' `bytes` and expect this
    // function's own test-facing bytes_moved figure to fall out
    // (llama.cpp-kcya c-amir should-fix 3): on the ksplit <= 1 path they
    // match (weight + activation + dst, the useful traffic and nothing
    // else); on the ksplit > 1 path the two rows' bytes ALSO include the
    // K-split's scratch round-trip (written by .partial, read back by
    // .combine) -- real DRAM traffic this dispatch causes, so it belongs
    // in the profiler's own accounting, but not "useful" work a caller of
    // this function cares about, so bytes_moved deliberately excludes it.
    // See mxfp4_soa_gemm_int8_dpas_partial_launch's own bytes comment for
    // the full reconciliation and docs/backend/sycl-env-vars.md
    // (GGML_SYCL_STORED_GEMM_DEBUG) for the same note restated where a
    // profiler-CSV reader is more likely to look first.
    switch (M) {
        case 1:
            return mxfp4_soa_gemm_int8_dpas_launch<1>(queue, soa_base, act_qs_device, act_scales_device, dst_device,
                                                      n_out, n_k, deps);
        case 2:
            return mxfp4_soa_gemm_int8_dpas_launch<2>(queue, soa_base, act_qs_device, act_scales_device, dst_device,
                                                      n_out, n_k, deps);
        case 4:
            return mxfp4_soa_gemm_int8_dpas_launch<4>(queue, soa_base, act_qs_device, act_scales_device, dst_device,
                                                      n_out, n_k, deps);
        case 8:
            return mxfp4_soa_gemm_int8_dpas_launch<8>(queue, soa_base, act_qs_device, act_scales_device, dst_device,
                                                      n_out, n_k, deps);
        default:
            GGML_ABORT("ggml_sycl_mxfp4_soa_gemm_dpas: unsupported M=%d (must be one of {1,2,4,8})", M);
    }
#else
    GGML_UNUSED(soa_weight_device);
    GGML_UNUSED(act_qs_device);
    GGML_UNUSED(act_scales_device);
    GGML_UNUSED(dst_device);
    GGML_UNUSED(M);
    GGML_UNUSED(n_out);
    GGML_UNUSED(n_k);
    GGML_UNUSED(deps);
    GGML_ABORT(
        "ggml_sycl_mxfp4_soa_gemm_dpas: ESIMD/DPAS not available in this SYCL toolchain (GGML_SYCL_ESIMD_AVAILABLE=0)");
#endif
}

}  // namespace ggml_sycl_mxfp4_stored_gemm
