// oneDNN packed-layout pack_m propagation gate.
//
// SUBJECT: ggml_sycl_update_layout_from_cache() must copy a cached
// ONEDNN_PACKED entry's onednn_pack_m onto the tensor's layout record, so a
// weight staged with a non-default pack_m is dispatched with that pack_m
// rather than silently falling back to 0.  That assignment is the ternary in
// ggml-sycl.cpp guarded by `onednn_pack_m > 0`, reached from
// ggml_sycl_get_weight_layout_ptr() on a cache hit.
//
// HISTORY (llama.cpp-vexg).  This file replaces test-sycl-onednn-packed-cache,
// which asserted three behaviours that no longer exist and therefore could not
// fail: on-demand layout materialization, on-demand repack when pack_m
// changes, and AoS eviction on layout switch.  All three lived in
// unified_cache::ensure_cached_layout(), which ggml_sycl_get_weight_layout_ptr()
// stopped calling at 38c02b099 / b8afde8e9 -- it is a pure lookup now, and
// staging belongs to S1-PRELOAD.  The old test skipped at its first
// precondition and exited 0, so every historical green was a skip-pass.
//
// WHY THE NAME CHANGED.  The old PASS line claimed "AoS evicted", which is
// unsatisfiable here: any host-registered weight is swept by S1-PRELOAD at
// load-transaction close, which direct-stages it under
// make_direct_stage_key(DENSE_WEIGHT, key, layout) -- a key that folds the
// layout in precisely so a later layout cannot displace an earlier one -- and
// unified_cache::is_cached() probes that key first.  So is_cached(key, AOS)
// stays true no matter what is staged afterwards, and the only ways to falsify
// it are for the test to call remove() itself (tautological) or to skip
// registration (which forfeits the tensor extra this test needs).
//
// SCOPE.  This gate asserts cache metadata propagation, not numerical
// correctness of the packed bytes: the fixture stages with a plain copy
// because no production site produces a real ONEDNN_PACKED layout today.
// That absence is tracked separately.

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#ifndef GGML_SYCL_WARP_SIZE
#    define GGML_SYCL_WARP_SIZE 32
#endif
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/unified-cache.hpp"

// ctest SKIP_RETURN_CODE, matching the test-sycl-*-policy.sh family.  Only a
// genuinely absent device skips; every other precondition FAILs, because a
// precondition that skips is how this file spent months reporting green.
static const int GATE_PASS = 0;
static const int GATE_FAIL = 1;
static const int GATE_SKIP = 77;

// The tensor-level override under test.  Deliberately different from the
// GGML_SYCL_ONEDNN_PACK_M env default below, so that a lookup which ignored
// the override would resolve 16, mismatch the staged entry's 32, and return
// null instead of quietly agreeing.
static const int64_t PACK_M_OVERRIDE = 32;

struct fixture {
    ggml_backend_t        backend    = nullptr;
    ggml_context *        ctx        = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    ggml_backend_buffer_t input_buf  = nullptr;
    ggml_backend_buffer_t output_buf = nullptr;
    ggml_sycl_model_token model{};
    bool                  model_owned = false;

    ~fixture() {
        if (weight_buf) {
            ggml_backend_buffer_free(weight_buf);
        }
        if (input_buf) {
            ggml_backend_buffer_free(input_buf);
        }
        if (output_buf) {
            ggml_backend_buffer_free(output_buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
        if (backend) {
            ggml_backend_free(backend);
        }
        if (model_owned) {
            (void) ggml_backend_sycl_model_unloaded_token(model);
        }
    }
};

static int run_pack_m_propagation_test() {
    setenv("GGML_SYCL_ONEDNN_PACK_M", "16", 1);

    fixture fx;

    fx.backend = ggml_backend_sycl_init(0);
    if (!fx.backend) {
        std::printf("SKIP: SYCL backend unavailable\n");
        return GATE_SKIP;
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();
    ggml_backend_buffer_type_t dev_buft  = ggml_backend_get_default_buffer_type(fx.backend);
    if (!host_buft || !dev_buft) {
        std::printf("SKIP: buffer types unavailable\n");
        return GATE_SKIP;
    }

    ggml_init_params params = {
        16 * 1024 * 1024,
        nullptr,
        true,
    };
    fx.ctx = ggml_init(params);
    if (!fx.ctx) {
        std::printf("FAIL: ggml_init failed\n");
        return GATE_FAIL;
    }

    const int ncols   = 256;
    const int nrows   = 128;
    const int ntokens = 1;

    ggml_tensor * weight = ggml_new_tensor_2d(fx.ctx, GGML_TYPE_F16, ncols, nrows);
    ggml_set_name(weight, "onednn_packed_weight");
    ggml_tensor * input = ggml_new_tensor_2d(fx.ctx, GGML_TYPE_F16, ncols, ntokens);
    ggml_set_name(input, "onednn_packed_input");
    ggml_tensor * output = ggml_mul_mat(fx.ctx, weight, input);
    ggml_set_name(output, "onednn_packed_output");

    fx.weight_buf = ggml_backend_buft_alloc_buffer(host_buft, ggml_backend_buft_get_alloc_size(host_buft, weight));
    fx.input_buf  = ggml_backend_buft_alloc_buffer(dev_buft, ggml_backend_buft_get_alloc_size(dev_buft, input));
    fx.output_buf = ggml_backend_buft_alloc_buffer(dev_buft, ggml_backend_buft_get_alloc_size(dev_buft, output));
    if (!fx.weight_buf || !fx.input_buf || !fx.output_buf) {
        std::printf("FAIL: buffer allocation failed\n");
        return GATE_FAIL;
    }

    ggml_backend_buffer_set_usage(fx.weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(fx.input_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_set_usage(fx.output_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    ggml_backend_tensor_alloc(fx.weight_buf, weight, ggml_backend_buffer_get_base(fx.weight_buf));
    ggml_backend_tensor_alloc(fx.input_buf, input, ggml_backend_buffer_get_base(fx.input_buf));
    ggml_backend_tensor_alloc(fx.output_buf, output, ggml_backend_buffer_get_base(fx.output_buf));

    std::vector<ggml_fp16_t> weight_data(ncols * nrows);
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = ggml_fp32_to_fp16(0.01f * static_cast<float>(i % 97));
    }

    // Host-weight registration is only honoured inside a model-load
    // transaction: outside one, ggml_backend_sycl_register_host_weight_tensor
    // cannot acquire a load-effect lease and silently registers nothing, so the
    // tensor never gets an extra (and ggml_backend_sycl_set_onednn_pack_m,
    // which needs one, becomes a no-op).  S1-PRELOAD -- the only producer of a
    // dense-weight cache entry -- runs when the outermost transaction closes.
    // Same bracket as tests/test-q8-0-layout-cache-path.cpp; RCA llama.cpp-43uy.
    ggml_sycl::test_clear_host_weight_registry();

    ggml_sycl_load_txn load{};
    if (ggml_backend_sycl_model_load_begin(&load) != GGML_SYCL_LIFECYCLE_OK) {
        std::printf("FAIL: could not begin SYCL model-load transaction\n");
        return GATE_FAIL;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(fx.backend);
    if (dev) {
        ggml_backend_sycl_register_host_weight_tensor(dev, weight);
    }
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sycl_model_load_end(load, true, &fx.model) != GGML_SYCL_LIFECYCLE_OK) {
        std::printf("FAIL: could not end SYCL model-load transaction\n");
        return GATE_FAIL;
    }
    fx.model_owned = true;

    if (!weight->extra || !weight->layout) {
        std::printf("FAIL: host-weight registration produced no tensor extra/layout\n");
        return GATE_FAIL;
    }

    sycl::queue &              q     = ggml_sycl_get_device(0).default_queue();
    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache(q);
    if (!cache) {
        std::printf("FAIL: unified cache unavailable after model load\n");
        return GATE_FAIL;
    }

    ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, 0);
    if (!key.valid) {
        std::printf("FAIL: weight cache key invalid\n");
        return GATE_FAIL;
    }

    // PROBE, never a failure.  S1-PRELOAD stages an F16 dense weight as AoS
    // (layout_policy returns AOS for every non-quantized type), but it may
    // instead host-route it under VRAM pressure, in which case no device entry
    // appears.  Either outcome is fine for this gate -- the packed entry below
    // is staged explicitly and the non-vacuity control is the layout-record
    // check, not this line -- so report it and move on rather than going red
    // for a reason that is not this test's subject.
    std::printf("NOTE: S1-PRELOAD AoS entry after load: %s\n",
                cache->is_cached(key, GGML_LAYOUT_AOS) ? "present (device-staged)" : "absent (host-routed)");

    // Re-assert the override after the load: staging the AoS layout resets the
    // tensor's onednn_pack_m (the same ternary this test gates writes 0 for any
    // non-oneDNN layout), so the value must be re-established before it can be
    // asked to survive a packed lookup.
    ggml_backend_sycl_set_onednn_pack_m(weight, PACK_M_OVERRIDE);

    const size_t packed_bytes = ggml_sycl::test_layout_bytes(weight, GGML_LAYOUT_ONEDNN_PACKED, 0);
    if (packed_bytes == 0) {
        std::printf("FAIL: packed layout size resolved to 0\n");
        return GATE_FAIL;
    }

    auto stage = cache->direct_stage_weight(key, weight->data, ggml_nbytes(weight), packed_bytes,
                                            GGML_LAYOUT_ONEDNN_PACKED, nullptr, nullptr, &q);
    if (!stage.ok || !stage.ptr) {
        std::printf("FAIL: direct_stage_weight(ONEDNN_PACKED) refused (%zu bytes)\n", packed_bytes);
        return GATE_FAIL;
    }
    stage.event.wait();

    // direct_stage_weight carries no pack_m parameter, so stamp the entry the
    // way a producer would.  register_ready resolves the entry it just staged
    // through id_to_key_ and matches on pointer+layout+type.
    cache->register_ready(key, stage.ptr, GGML_LAYOUT_ONEDNN_PACKED, packed_bytes,
                          ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, weight->data, PACK_M_OVERRIDE);

    if (!cache->is_cached(key, GGML_LAYOUT_ONEDNN_PACKED)) {
        std::printf("FAIL: packed entry not cached after staging\n");
        return GATE_FAIL;
    }

    void * packed_ptr = ggml_sycl_get_weight_layout_ptr(weight, 0, GGML_LAYOUT_ONEDNN_PACKED);
    if (!packed_ptr) {
        std::printf("FAIL: lookup missed the staged packed entry (pack_m override not honoured?)\n");
        return GATE_FAIL;
    }

    // Non-vacuity control for the assertion below.  mode/data_ptr are written
    // ONLY by ggml_sycl_update_layout_from_cache, so checking them first
    // distinguishes "the propagation ran and produced the wrong pack_m" from
    // "the propagation never ran and pack_m merely retained its seeded value".
    if (weight->layout->mode != GGML_LAYOUT_ONEDNN_PACKED || weight->layout->data_ptr != packed_ptr) {
        std::printf("FAIL: layout record not updated from cache (mode=%d data_ptr=%p expected %p)\n",
                    (int) weight->layout->mode, weight->layout->data_ptr, packed_ptr);
        return GATE_FAIL;
    }

    if (weight->layout->onednn_pack_m != PACK_M_OVERRIDE) {
        std::printf("FAIL: expected onednn pack_m=%lld, got=%lld\n", (long long) PACK_M_OVERRIDE,
                    (long long) weight->layout->onednn_pack_m);
        return GATE_FAIL;
    }

    std::printf("PASS: onednn packed layout cached and pack_m=%lld propagated\n", (long long) PACK_M_OVERRIDE);
    return GATE_PASS;
}

int main() {
    return run_pack_m_propagation_test();
}
