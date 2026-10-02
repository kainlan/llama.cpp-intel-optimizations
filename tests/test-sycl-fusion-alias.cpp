// Host-only: the aliasing gate for fused chain kernels needs no SYCL device (llama.cpp-rb2h).
//
// The fused ADD+RMS_NORM kernel reads its second ADD operand while it writes the RMS_NORM output.
// ggml_gallocr places that output in whatever the unfused order left free, which can be the dead
// operand's block merged with a freed neighbour, so the output lands a few KB off the operand and
// workgroup r overwrites cols of row r-1 that workgroup r-1 is still reading. The addresses below
// are the layout a failing Qwen3.6-27B B50 perplexity run logged at layer 47 (RMS_NORM output
// 8192 bytes below the ADD operand it overlaps).

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

static bool admits(const ggml_cgraph * gf) {
    return ggml_sycl_fusion_chain_alias_safe(gf, 0, gf->n_nodes, [](const ggml_tensor * t) { return t->data; });
}

// ADD(x, add) -> RMS_NORM, the bit5 chain.
static ggml_cgraph * build_add_rms(ggml_context * ctx, uintptr_t x, uintptr_t add, uintptr_t sum, uintptr_t rms) {
    ggml_tensor * tx   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * tadd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * tsum = ggml_add(ctx, tx, tadd);
    ggml_tensor * trms = ggml_rms_norm(ctx, tsum, 1e-6f);
    tx->data           = at(x);
    tadd->data         = at(add);
    tsum->data         = at(sum);
    trms->data         = at(rms);
    ggml_cgraph * gf   = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, trms);
    return gf;
}

// RMS_NORM(x) -> MUL(w), the bit4 chain; w is the broadcast norm weight.
static ggml_cgraph * build_rms_mul(ggml_context * ctx, uintptr_t x, uintptr_t w, uintptr_t rms, uintptr_t mul) {
    ggml_tensor * tx   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * tw   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor * trms = ggml_rms_norm(ctx, tx, 1e-6f);
    ggml_tensor * tmul = ggml_mul(ctx, trms, tw);
    tx->data           = at(x);
    tw->data           = at(w);
    trms->data         = at(rms);
    tmul->data         = at(mul);
    ggml_cgraph * gf   = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, tmul);
    return gf;
}

int main() {
    ggml_init_params params = { 4 * 1024 * 1024, nullptr, true };

    const uintptr_t base = 0xffffe392d6800880ull;
    const uintptr_t lo   = base + 0x2000;  // l_out-46, the ADD's second operand
    const uintptr_t hi   = lo + n_bytes;   // attn_output-47, and in place attn_residual-47
    const uintptr_t far  = hi + n_bytes + 0x2000;

    // The failing layout: the RMS_NORM output starts 8192 bytes before the operand it overlaps.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_add_rms(ctx, hi, lo, hi, base);
        CHECK(gf->n_nodes == 2, "the chain is the ADD and the RMS_NORM");
        CHECK(!admits(gf), "RMS_NORM output partially overlapping the ADD operand must not fuse");
        ggml_free(ctx);
    }
    // The output straddles the end of one operand and the start of the in-place sum.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_add_rms(ctx, hi, lo, hi, hi - 0x2000);
        CHECK(!admits(gf), "RMS_NORM output straddling two live blocks must not fuse");
        ggml_free(ctx);
    }
    // The ADD output off its own operand by a few KB.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_add_rms(ctx, hi, lo, hi + 0x1000, far);
        CHECK(!admits(gf), "ADD output partially overlapping its operand must not fuse");
        ggml_free(ctx);
    }
    // The layouts every earlier layer had: in place on the same address, or disjoint.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_add_rms(ctx, hi, lo, hi, lo);
        CHECK(admits(gf), "ADD in place on its first operand, RMS_NORM in place on the second, must fuse");
        ggml_free(ctx);
    }
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_add_rms(ctx, hi, lo, hi, far);
        CHECK(admits(gf), "a disjoint RMS_NORM output must fuse");
        ggml_free(ctx);
    }
    // The RMS_NORM+MUL chain: the MUL output off its own input by a few KB, then the safe layouts.
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_rms_mul(ctx, hi, far + n_bytes, far, hi - 0x2000);
        CHECK(gf->n_nodes == 2, "the chain is the RMS_NORM and the MUL");
        CHECK(!admits(gf), "MUL output partially overlapping the RMS_NORM input must not fuse");
        ggml_free(ctx);
    }
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_rms_mul(ctx, hi, far + n_bytes, hi, hi);
        CHECK(admits(gf), "RMS_NORM and MUL both in place on the input must fuse");
        ggml_free(ctx);
    }
    {
        ggml_context * ctx = ggml_init(params);
        ggml_cgraph *  gf  = build_rms_mul(ctx, hi, far + n_bytes, far, far);
        CHECK(admits(gf), "a disjoint MUL output must fuse");
        ggml_free(ctx);
    }

    std::printf("test-sycl-fusion-alias: PASS\n");
    return 0;
}
