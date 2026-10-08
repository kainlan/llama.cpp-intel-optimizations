// Host-only: the aliasing gate for fused kernels needs no SYCL device (llama.cpp-rb2h).
//
// The fused ADD+RMS_NORM kernel reads its second ADD operand while it writes the RMS_NORM output.
// ggml_gallocr places that output in whatever the unfused order left free, which can be the dead
// operand's block merged with a freed neighbour, so the output lands a few KB off the operand and
// workgroup r overwrites cols of row r-1 that workgroup r-1 is still reading. The addresses below
// are the layout a failing Qwen3.6-27B B50 perplexity run logged at layer 47 (RMS_NORM output
// 8192 bytes below the ADD operand it overlaps).
//
// Every case names the property it pins; a gate that admitted everything, ignored that property, or
// resolved a tensor twice fails it.

#include "ggml-sycl/fusion-alias.hpp"
#include "ggml.h"

#include <cstdint>
#include <cstdio>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static const int64_t n_embd   = 5120;
static const int64_t n_tokens = 512;
static const size_t  n_bytes  = n_embd * n_tokens * sizeof(float);

static void * at(uintptr_t addr) {
    return reinterpret_cast<void *>(addr);
}

// The chain starts at the first node that is not a VIEW: a view is a graph node of its own.
static int first_chain_node(const ggml_cgraph * gf) {
    int i = 0;
    while (i < gf->n_nodes && gf->nodes[i]->op == GGML_OP_VIEW) {
        ++i;
    }
    return i;
}

static ggml_sycl_fusion_alias_result check(const ggml_cgraph * gf, ggml_sycl_fusion_alias_site site) {
    return ggml_sycl_fusion_chain_alias_check(gf, first_chain_node(gf), site,
                                              [](const ggml_tensor * t, bool) { return t->data; });
}

static bool admits(const ggml_cgraph * gf, ggml_sycl_fusion_alias_site site) {
    return check(gf, site).verdict == GGML_SYCL_FUSION_ALIAS_SAFE;
}

static ggml_cgraph * finish(ggml_context * ctx, ggml_tensor * last) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, last);
    return gf;
}

// ADD(x, add) -> RMS_NORM, the bit5 chain; the kernel writes both the sum and the norm.
static ggml_cgraph * build_add_rms(ggml_context * ctx, uintptr_t x, uintptr_t add, uintptr_t sum, uintptr_t rms) {
    ggml_tensor * tx   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * tadd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * tsum = ggml_add(ctx, tx, tadd);
    ggml_tensor * trms = ggml_rms_norm(ctx, tsum, 1e-6f);
    tx->data           = at(x);
    tadd->data         = at(add);
    tsum->data         = at(sum);
    trms->data         = at(rms);
    return finish(ctx, trms);
}

// RMS_NORM(x) -> MUL(w), the bit4 chain; w is the broadcast norm weight. The kernel writes only the MUL.
static ggml_cgraph * build_rms_mul(ggml_context * ctx, uintptr_t x, uintptr_t w, uintptr_t rms, uintptr_t mul) {
    ggml_tensor * tx   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * tw   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor * trms = ggml_rms_norm(ctx, tx, 1e-6f);
    ggml_tensor * tmul = ggml_mul(ctx, trms, tw);
    tx->data           = at(x);
    tw->data           = at(w);
    trms->data         = at(rms);
    tmul->data         = at(mul);
    return finish(ctx, tmul);
}

// RMS_NORM(x) -> MUL(w) -> ADD(b), the bit1 chain. The kernel writes only the ADD.
static ggml_cgraph * build_rms_mul_add(ggml_context * ctx,
                                       uintptr_t      x,
                                       uintptr_t      w,
                                       uintptr_t      b,
                                       uintptr_t      rms,
                                       uintptr_t      mul,
                                       uintptr_t      add) {
    ggml_tensor * tx   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * tw   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor * tb   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor * trms = ggml_rms_norm(ctx, tx, 1e-6f);
    ggml_tensor * tmul = ggml_mul(ctx, trms, tw);
    ggml_tensor * tadd = ggml_add(ctx, tmul, tb);
    tx->data           = at(x);
    tw->data           = at(w);
    tb->data           = at(b);
    trms->data         = at(rms);
    tmul->data         = at(mul);
    tadd->data         = at(add);
    return finish(ctx, tadd);
}

// MUL(x, scale) -> ADD(bias), the bit6 chain. The kernel writes only the ADD.
static ggml_cgraph * build_mul_add(ggml_context * ctx,
                                   uintptr_t      x,
                                   uintptr_t      scale,
                                   uintptr_t      bias,
                                   uintptr_t      mul,
                                   uintptr_t      add) {
    ggml_tensor * tx    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * tsc   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor * tbias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor * tmul  = ggml_mul(ctx, tx, tsc);
    ggml_tensor * tadd  = ggml_add(ctx, tmul, tbias);
    tx->data            = at(x);
    tsc->data           = at(scale);
    tbias->data         = at(bias);
    tmul->data          = at(mul);
    tadd->data          = at(add);
    return finish(ctx, tadd);
}

int main() {
    ggml_init_params params = { 4 * 1024 * 1024, nullptr, true };

    const uintptr_t base = 0xffffe392d6800880ull;
    const uintptr_t lo   = base + 0x2000;      // l_out-46, the ADD's second operand
    const uintptr_t hi   = lo + n_bytes;       // attn_output-47, and in place attn_residual-47
    const uintptr_t far  = hi + n_bytes + 0x2000;
    const uintptr_t vec  = far + 4 * n_bytes;  // a norm weight / bias vector, far from everything

    // ---- ADD+RMS_NORM (bit5): both outputs are written -------------------------------------------------
    // The failing layout: the RMS_NORM output starts 8192 bytes before the operand it overlaps.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_add_rms(ctx, hi, lo, hi, base);
        CHECK(gf->n_nodes == 2, "the chain is the ADD and the RMS_NORM");
        const ggml_sycl_fusion_alias_result r = check(gf, GGML_SYCL_FUSION_SITE_ADD_RMS_NORM);
        CHECK(r.verdict == GGML_SYCL_FUSION_ALIAS_OVERLAP, "RMS_NORM output partially overlapping the ADD operand");
        CHECK(r.write == gf->nodes[1] && r.other != nullptr, "the decline names the output and what it overlaps");
        ggml_free(ctx);
    }
    // The output straddles the end of one operand and the start of the in-place sum.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_add_rms(ctx, hi, lo, hi, hi - 0x2000);
        CHECK(!admits(gf, GGML_SYCL_FUSION_SITE_ADD_RMS_NORM),
              "RMS_NORM output straddling two live blocks must not fuse");
        ggml_free(ctx);
    }
    // The ADD output off its own operand by a few KB, in the other direction.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_add_rms(ctx, hi, lo, hi + 0x1000, far);
        CHECK(!admits(gf, GGML_SYCL_FUSION_SITE_ADD_RMS_NORM),
              "ADD output partially overlapping its operand must not fuse");
        ggml_free(ctx);
    }
    // The two outputs overlap each other without touching an input.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_add_rms(ctx, hi, lo, far, far + 0x1000);
        CHECK(!admits(gf, GGML_SYCL_FUSION_SITE_ADD_RMS_NORM), "two outputs overlapping each other must not fuse");
        ggml_free(ctx);
    }
    // The layouts every earlier layer had: in place on the same address, or disjoint.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_add_rms(ctx, hi, lo, hi, lo);
        CHECK(admits(gf, GGML_SYCL_FUSION_SITE_ADD_RMS_NORM),
              "ADD in place on its first operand, RMS_NORM in place on the second, must fuse");
        ggml_free(ctx);
    }
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_add_rms(ctx, hi, lo, hi, far);
        CHECK(admits(gf, GGML_SYCL_FUSION_SITE_ADD_RMS_NORM), "a disjoint RMS_NORM output must fuse");
        ggml_free(ctx);
    }
    // Fail closed: a tensor the caller could not resolve declines, whichever role it has.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_add_rms(ctx, hi, lo, hi, far);
        gf->nodes[1]->data = nullptr;  // an unresolvable output
        CHECK(check(gf, GGML_SYCL_FUSION_SITE_ADD_RMS_NORM).verdict == GGML_SYCL_FUSION_ALIAS_UNRESOLVED,
              "an unresolvable output must decline");
        ggml_free(ctx);
    }
    {
        ggml_context * ctx         = ggml_init(params);
        ggml_cgraph *  gf          = build_add_rms(ctx, hi, lo, hi, far);
        gf->nodes[0]->src[1]->data = nullptr;  // an unresolvable input
        CHECK(check(gf, GGML_SYCL_FUSION_SITE_ADD_RMS_NORM).verdict == GGML_SYCL_FUSION_ALIAS_UNRESOLVED,
              "an unresolvable input must decline");
        ggml_free(ctx);
    }
    // A malformed request declines rather than reading past the graph.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_add_rms(ctx, hi, lo, hi, far);
        CHECK(ggml_sycl_fusion_chain_alias_check(
                  gf, 1, GGML_SYCL_FUSION_SITE_ADD_RMS_NORM,
                  [](const ggml_tensor * t, bool) {
                      return t->data;
                  }).verdict == GGML_SYCL_FUSION_ALIAS_UNRESOLVED,
              "a chain that runs past the last node must decline");
        ggml_free(ctx);
    }
    // Each distinct tensor is resolved once: the sum, the norm and the two inputs.
    {
        ggml_context * ctx   = ggml_init(params);
        ggml_cgraph *  gf    = build_add_rms(ctx, hi, lo, hi, far);
        int            calls = 0;
        ggml_sycl_fusion_chain_alias_check(gf, 0, GGML_SYCL_FUSION_SITE_ADD_RMS_NORM, [&](const ggml_tensor * t, bool) {
            ++calls;
            return t->data;
        });
        CHECK(calls == 4, "four distinct tensors resolve four times, not once per pair");
        ggml_free(ctx);
    }

    // ---- RMS_NORM+MUL (bit4): the RMS_NORM result is never written -------------------------------------
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_rms_mul(ctx, hi, vec, far, hi - 0x2000);
        CHECK(gf->n_nodes == 2, "the chain is the RMS_NORM and the MUL");
        CHECK(check(gf, GGML_SYCL_FUSION_SITE_RMS_NORM_MUL).verdict == GGML_SYCL_FUSION_ALIAS_OVERLAP,
              "MUL output partially overlapping the RMS_NORM input must not fuse");
        ggml_free(ctx);
    }
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_rms_mul(ctx, hi, vec, hi, hi);
        CHECK(admits(gf, GGML_SYCL_FUSION_SITE_RMS_NORM_MUL), "RMS_NORM and MUL both in place on the input must fuse");
        ggml_free(ctx);
    }
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_rms_mul(ctx, hi, vec, far, far);
        CHECK(admits(gf, GGML_SYCL_FUSION_SITE_RMS_NORM_MUL), "a disjoint MUL output must fuse");
        ggml_free(ctx);
    }
    // The RMS_NORM intermediate straddles the input but is never written, so the chain still fuses.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_rms_mul(ctx, hi, vec, hi - 0x2000, far);
        CHECK(admits(gf, GGML_SYCL_FUSION_SITE_RMS_NORM_MUL),
              "an unwritten RMS_NORM intermediate overlapping the input must not decline the chain");
        ggml_free(ctx);
    }
    // The broadcast norm weight at the MUL output's address has a different shape: same address, not in place.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_rms_mul(ctx, hi, far, hi, far);
        CHECK(check(gf, GGML_SYCL_FUSION_SITE_RMS_NORM_MUL).verdict == GGML_SYCL_FUSION_ALIAS_LAYOUT,
              "a broadcast operand at the output's address (same address, different shape) must decline");
        ggml_free(ctx);
    }
    // The weight overlapping the output at a different address.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_rms_mul(ctx, hi, far + 0x1000, hi, far);
        CHECK(check(gf, GGML_SYCL_FUSION_SITE_RMS_NORM_MUL).verdict == GGML_SYCL_FUSION_ALIAS_OVERLAP,
              "a weight vector inside the output block must decline");
        ggml_free(ctx);
    }
    // The weight tensor resolves like the kernel resolves it: the role is passed to the resolver.
    {
        ggml_context * ctx         = ggml_init(params);
        ggml_cgraph *  gf          = build_rms_mul(ctx, hi, vec, far, far);
        int            writes_seen = 0, reads_seen = 0;
        ggml_sycl_fusion_chain_alias_check(gf, 0, GGML_SYCL_FUSION_SITE_RMS_NORM_MUL,
                                           [&](const ggml_tensor * t, bool w) {
                                               (w ? writes_seen : reads_seen)++;
                                               return t->data;
                                           });
        CHECK(writes_seen == 1 && reads_seen == 2, "bit4 writes the MUL only and reads the input and the weight");
        ggml_free(ctx);
    }

    // ---- RMS_NORM+MUL+ADD (bit1): a 3-node chain, only the ADD is written --------------------------------
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_rms_mul_add(ctx, hi, vec, vec + 0x4000, far, far + n_bytes, far + 2 * n_bytes);
        CHECK(gf->n_nodes == 3, "the chain is the RMS_NORM, the MUL and the ADD");
        CHECK(admits(gf, GGML_SYCL_FUSION_SITE_RMS_NORM_MUL_ADD), "a disjoint 3-node chain must fuse");
        ggml_free(ctx);
    }
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_rms_mul_add(ctx, hi, vec, vec + 0x4000, far, far + n_bytes, hi - 0x2000);
        CHECK(check(gf, GGML_SYCL_FUSION_SITE_RMS_NORM_MUL_ADD).verdict == GGML_SYCL_FUSION_ALIAS_OVERLAP,
              "the ADD output partially overlapping the RMS_NORM input must not fuse");
        ggml_free(ctx);
    }
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_rms_mul_add(ctx, hi, vec, vec + 0x4000, far, far + n_bytes, hi);
        CHECK(admits(gf, GGML_SYCL_FUSION_SITE_RMS_NORM_MUL_ADD), "the ADD in place on the input must fuse");
        ggml_free(ctx);
    }
    // Neither intermediate is written, so both may straddle the input.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_rms_mul_add(ctx, hi, vec, vec + 0x4000, hi - 0x2000, hi + 0x1000, far);
        CHECK(admits(gf, GGML_SYCL_FUSION_SITE_RMS_NORM_MUL_ADD),
              "unwritten RMS_NORM and MUL intermediates overlapping the input must not decline the chain");
        ggml_free(ctx);
    }
    // The ADD operand (a bias vector) inside the ADD output block.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_rms_mul_add(ctx, hi, vec, far + 0x1000, far - n_bytes, far - 2 * n_bytes, far);
        CHECK(check(gf, GGML_SYCL_FUSION_SITE_RMS_NORM_MUL_ADD).verdict == GGML_SYCL_FUSION_ALIAS_OVERLAP,
              "the ADD operand inside the ADD output block must decline");
        ggml_free(ctx);
    }

    // ---- MUL+ADD (bit6): the MUL result is never written -----------------------------------------------
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_mul_add(ctx, hi, vec, vec + 0x4000, far, far + n_bytes);
        CHECK(gf->n_nodes == 2, "the chain is the MUL and the ADD");
        CHECK(admits(gf, GGML_SYCL_FUSION_SITE_MUL_ADD), "a disjoint MUL+ADD chain must fuse");
        ggml_free(ctx);
    }
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_mul_add(ctx, hi, vec, vec + 0x4000, far, hi - 0x2000);
        CHECK(check(gf, GGML_SYCL_FUSION_SITE_MUL_ADD).verdict == GGML_SYCL_FUSION_ALIAS_OVERLAP,
              "the ADD output partially overlapping x must not fuse");
        ggml_free(ctx);
    }
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_mul_add(ctx, hi, vec, vec + 0x4000, far, hi);
        CHECK(admits(gf, GGML_SYCL_FUSION_SITE_MUL_ADD), "the ADD in place on x must fuse");
        ggml_free(ctx);
    }
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_mul_add(ctx, hi, vec, vec + 0x4000, hi - 0x2000, far);
        CHECK(admits(gf, GGML_SYCL_FUSION_SITE_MUL_ADD),
              "an unwritten MUL intermediate overlapping x must not decline the chain");
        ggml_free(ctx);
    }
    // A scale vector at the ADD output's address (a different shape) is not an in-place pair.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_mul_add(ctx, hi, far, vec + 0x4000, far - n_bytes, far);
        CHECK(check(gf, GGML_SYCL_FUSION_SITE_MUL_ADD).verdict == GGML_SYCL_FUSION_ALIAS_LAYOUT,
              "a scale vector at the output's address must decline");
        ggml_free(ctx);
    }

    // ---- views and strides --------------------------------------------------------------------------
    // A contiguous view at a non-zero offset into a larger tensor: the in-place pair is admitted, an
    // output straddling its start is not.
    {
        ggml_context *  ctx   = ggml_init(params);
        ggml_tensor *   big   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 2 * n_tokens);
        ggml_tensor *   view  = ggml_view_2d(ctx, big, n_embd, n_tokens, big->nb[1], n_tokens / 2 * big->nb[1]);
        ggml_tensor *   tadd  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_tensor *   tsum  = ggml_add(ctx, view, tadd);
        ggml_tensor *   trms  = ggml_rms_norm(ctx, tsum, 1e-6f);
        const uintptr_t vaddr = hi + n_tokens / 2 * big->nb[1];
        big->data             = at(hi);
        view->data            = at(vaddr);
        tadd->data            = at(far + 3 * n_bytes);
        tsum->data            = at(vaddr);
        trms->data            = at(far + n_bytes);
        ggml_cgraph * gf      = finish(ctx, trms);
        CHECK(admits(gf, GGML_SYCL_FUSION_SITE_ADD_RMS_NORM),
              "a view at a non-zero offset, summed in place, must fuse");
        trms->data = at(vaddr - 0x2000);
        CHECK(check(gf, GGML_SYCL_FUSION_SITE_ADD_RMS_NORM).verdict == GGML_SYCL_FUSION_ALIAS_OVERLAP,
              "an output straddling the start of a view must decline");
        ggml_free(ctx);
    }
    // A non-contiguous view (rows twice the width): its extent includes the gaps, so an output inside them
    // declines, and an output at its address has other strides.
    {
        ggml_context * ctx  = ggml_init(params);
        ggml_tensor *  big  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * n_embd, n_tokens);
        ggml_tensor *  view = ggml_view_2d(ctx, big, n_embd, n_tokens, big->nb[1], 0);
        ggml_tensor *  tadd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_tensor *  tsum = ggml_add(ctx, view, tadd);
        ggml_tensor *  trms = ggml_rms_norm(ctx, tsum, 1e-6f);
        big->data           = at(hi);
        view->data          = at(hi);
        tadd->data          = at(far + 3 * n_bytes);
        tsum->data          = at(far + n_bytes);
        trms->data          = at(far + 2 * n_bytes);
        ggml_cgraph * gf    = finish(ctx, trms);
        CHECK(admits(gf, GGML_SYCL_FUSION_SITE_ADD_RMS_NORM), "outputs disjoint from a strided view must fuse");
        tsum->data = at(hi + 4 * n_embd);  // inside the gap the view's rows leave
        CHECK(check(gf, GGML_SYCL_FUSION_SITE_ADD_RMS_NORM).verdict == GGML_SYCL_FUSION_ALIAS_OVERLAP,
              "an output inside the extent of a strided view must decline");
        tsum->data = at(hi);
        CHECK(check(gf, GGML_SYCL_FUSION_SITE_ADD_RMS_NORM).verdict == GGML_SYCL_FUSION_ALIAS_LAYOUT,
              "a contiguous output at a strided view's address (different strides) must decline");
        ggml_free(ctx);
    }

    // ---- the two bit0 sites hand their resolved operands to the check directly ------------------------
    {
        // MUL_MAT+ADD: the addend is read in the MMVQ epilogue as the same row is written, so an in-place
        // addend is admitted and a straddling one is not. The activation is quantised to scratch first.
        ggml_context *           ctx    = ggml_init(params);
        ggml_tensor *            addend = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        ggml_tensor *            out    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        const size_t             vbytes = n_embd * sizeof(float);
        ggml_sycl_fusion_operand w      = { out, at(hi), true };
        ggml_sycl_fusion_operand r      = { addend, at(hi), true };
        CHECK(ggml_sycl_fusion_alias_check(&w, 1, &r, 1).verdict == GGML_SYCL_FUSION_ALIAS_SAFE,
              "an addend in place on the output must fuse");
        r.ptr = at(hi + vbytes);
        CHECK(ggml_sycl_fusion_alias_check(&w, 1, &r, 1).verdict == GGML_SYCL_FUSION_ALIAS_SAFE,
              "a disjoint addend must fuse");
        r.ptr = at(hi + vbytes / 2);
        CHECK(ggml_sycl_fusion_alias_check(&w, 1, &r, 1).verdict == GGML_SYCL_FUSION_ALIAS_OVERLAP,
              "an addend overlapping the output at another offset must decline");
        r.ptr = nullptr;
        CHECK(ggml_sycl_fusion_alias_check(&w, 1, &r, 1).verdict == GGML_SYCL_FUSION_ALIAS_UNRESOLVED,
              "an unresolved addend must decline");
        ggml_free(ctx);
    }
    {
        // Router: other subgroups still read the activation while a lane writes probs, so the activation is
        // never in place; the bias is read and written by the same lane; the argsort is written last.
        ggml_context *           ctx       = ggml_init(params);
        ggml_tensor *            act       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        ggml_tensor *            bias      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        ggml_tensor *            probs     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        ggml_tensor *            sort      = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 64);
        ggml_sycl_fusion_operand writes[2] = {
            { probs, at(far),          true },
            { sort,  at(far + 0x1000), true }
        };
        ggml_sycl_fusion_operand reads[2] = {
            { act,  at(hi),           false },
            { bias, at(far - 0x1000), true  }
        };
        CHECK(ggml_sycl_fusion_alias_check(writes, 2, reads, 2).verdict == GGML_SYCL_FUSION_ALIAS_SAFE,
              "a disjoint router must fuse");
        reads[1].ptr = at(far);
        CHECK(ggml_sycl_fusion_alias_check(writes, 2, reads, 2).verdict == GGML_SYCL_FUSION_ALIAS_SAFE,
              "a bias in place on the probs must fuse");
        reads[0].ptr = at(far);
        CHECK(ggml_sycl_fusion_alias_check(writes, 2, reads, 2).verdict == GGML_SYCL_FUSION_ALIAS_INPLACE,
              "an activation identical to the probs is read by other lanes after they are written: decline");
        reads[0].ptr = at(far + 0x40);
        CHECK(ggml_sycl_fusion_alias_check(writes, 2, reads, 2).verdict == GGML_SYCL_FUSION_ALIAS_OVERLAP,
              "an activation overlapping the probs at another offset must decline");
        reads[0].ptr  = at(hi);
        writes[1].ptr = at(far);  // argsort on the probs: same address, different type
        CHECK(ggml_sycl_fusion_alias_check(writes, 2, reads, 2).verdict == GGML_SYCL_FUSION_ALIAS_LAYOUT,
              "the argsort on the probs (same address, int32 vs f32) must decline");
        ggml_free(ctx);
    }

    std::printf("test-sycl-fusion-alias: PASS\n");
    return 0;
}
