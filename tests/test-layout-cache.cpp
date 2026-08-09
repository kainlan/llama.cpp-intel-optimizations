// Unit tests for SYCL layout selection and unified cache behavior
//
// Usage (the registration pins level_zero:1; a DIRECT run does not inherit it):
//   source /opt/intel/oneapi/setvars.sh --force
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-layout-cache
//
// Registered at tests/CMakeLists.txt (restored per llama.cpp-b8uv), currently
// carrying DISABLED TRUE pending llama.cpp-0tkh -- see that block's comment for
// the remaining blockers.  It loads models, so it belongs to the OOM-hazard
// family: run once, never in a loop.
//
// ⚠️ ONE HAZARD LIVES IN THIS FILE: main() setenv()s ONEAPI_DEVICE_SELECTOR to
// level_zero:0 -- the B70 BENCHMARK card -- when nothing is set.  The setenv
// only fires when the variable is unset, so the registration's pinned
// level_zero:1 defeats it; a direct run must pin it by hand.
//
// ---------------------------------------------------------------------------
// llama.cpp-0tkh ADJUDICATION: STALE ORACLE, not a dropped override.
//
// The restored test_mul_mat_layout_choice_coalesced allocated its weight in
// ggml_backend_sycl_host_buffer_type(), forced GGML_LAYOUT_COALESCED via
// ggml_sycl::test_layout_override_guard, and asserted the resolved entry
// recorded COALESCED.  It got AOS.  The override is not at fault -- the op
// never reaches layout selection at all:
//
//  - The override is a layout-SELECTION hint with no residency authority.
//    ggml-sycl-test.hpp:16-17 says exactly that ("temporarily force a layout
//    during a scoped operation"), test_set/get/clear_layout_override
//    (ggml-sycl.cpp:8217-8233) are a bare atomic pair, and EVERY consumer is a
//    selection site: 8821 (reorder gate, reacts to AOS only), 28584
//    (S1-PRELOAD layout pick), 53083 (UnifiedMatmulOrchestrator::select),
//    56692 (dispatch-failure path), 61083 (MoE XMX layout gate).  None of them
//    moves a tensor between host and device.
//  - Routing is decided upstream and independently.  should_dispatch_to_cpu()
//    (ggml-sycl.cpp:75918) asks ggml_sycl_mul_mat_weight_resolves_to_host()
//    (~75874), which tests resolve().on_device, resolve_allocation() and
//    ggml_backend_buffer_is_host() -- and never consults the override.  A
//    host-buft weight therefore leaves the SYCL backend before any layout is
//    chosen, which is the first run's "[SYCL-CPU] MUL_MAT routed to CPU from
//    resolved-host-handle: weight=attn_q.weight" (artifacts/b8uv/first-run.txt).
//
// So the fixture encoded the wrong axis, and the product is behaving as
// CLAUDE.md describes ("Host-resident weights -> CPU dispatch, not GPU PCIe
// zero-copy").  The cases below split the two axes apart instead.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml-sycl/ggml-sycl-test.hpp"

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

#include "ggml-quants.h"
#include "ggml-sycl/common.hpp"

static const char * layout_name(ggml_layout_mode mode) {
    switch (mode) {
        case GGML_LAYOUT_AOS:       return "AOS";
        case GGML_LAYOUT_SOA:       return "SOA";
        case GGML_LAYOUT_COALESCED: return "COALESCED";
        case GGML_LAYOUT_XMX_TILED: return "XMX_TILED";
        case GGML_LAYOUT_XMX_GEMM_TILED: return "XMX_GEMM_TILED";
        default:                    return "UNKNOWN";
    }
}

static const char * usage_name(tensor_usage usage) {
    switch (usage) {
        case tensor_usage::UNKNOWN:           return "UNKNOWN";
        case tensor_usage::ATTENTION_WEIGHT:  return "ATTENTION_WEIGHT";
        case tensor_usage::FFN_WEIGHT:        return "FFN_WEIGHT";
        case tensor_usage::MOE_EXPERT_WEIGHT: return "MOE_EXPERT_WEIGHT";
        case tensor_usage::MOE_GATE:          return "MOE_GATE";
        case tensor_usage::EMBEDDING:         return "EMBEDDING";
        case tensor_usage::NORM:              return "NORM";
        default:                              return "UNKNOWN";
    }
}

static void reset_layout_choices() {
    ggml_sycl::test_clear_host_weight_registry();
    ggml_sycl_load_txn load{};
    ggml_sycl_model_token model{};
    if (ggml_backend_sycl_model_load_begin(&load) == GGML_SYCL_LIFECYCLE_OK &&
        ggml_backend_sycl_model_load_end(load, true, &model) == GGML_SYCL_LIFECYCLE_OK) {
        (void) ggml_backend_sycl_model_unloaded_token(model);
    }
}

static bool expect_usage(const char * label, tensor_usage got, tensor_usage expected) {
    if (got != expected) {
        fprintf(stderr, "%s: expected usage %s, got %s\n",
                label, usage_name(expected), usage_name(got));
        return false;
    }
    return true;
}

static bool expect_layout(const char * label, ggml_layout_mode got, ggml_layout_mode expected) {
    if (got != expected) {
        fprintf(stderr, "%s: expected layout %s, got %s\n",
                label, layout_name(expected), layout_name(got));
        return false;
    }
    return true;
}

static void fill_pattern(std::vector<uint8_t> & data) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i * 131) ^ 0x5a);
    }
}

// Splits the two axes the old single "Failed to cache <layout> layout" message
// conflated: a null return means the layout could not be produced at all, while
// a false is_cached() means it WAS produced but is not registered under the key
// the test asserts on.  Those are different bugs and the old message named
// neither.
static bool expect_layout_cached(const char *               label,
                                 ggml_sycl::unified_cache * cache,
                                 const ggml_sycl_cache_id & key,
                                 const ggml_tensor *        weight,
                                 int                        device_id,
                                 ggml_layout_mode           layout) {
    void * ptr = ggml_sycl_get_weight_layout_ptr(weight, device_id, layout);
    if (!ptr) {
        fprintf(stderr, "%s: ggml_sycl_get_weight_layout_ptr(%s) returned null (layout not produced)\n", label,
                layout_name(layout));
        return false;
    }
    if (!cache->is_cached(key, layout)) {
        fprintf(stderr, "%s: %s produced at %p but is_cached(key, %s) is false (not registered under the key)\n", label,
                layout_name(layout), ptr, layout_name(layout));
        return false;
    }
    return true;
}

// Conditional counterpart to ggml_sycl::test_layout_override_guard.  The
// no-override control cases below must run with the override genuinely absent,
// so the guard cannot simply be constructed and then ignored.
struct scoped_layout_override {
    bool active;

    scoped_layout_override(bool active_arg, ggml_layout_mode layout) : active(active_arg) {
        if (active) {
            ggml_sycl::test_set_layout_override(layout);
        }
    }

    ~scoped_layout_override() {
        if (active) {
            ggml_sycl::test_clear_layout_override();
        }
    }
};

struct scoped_env {
    const char * name;
    std::string  prev;
    bool         had_prev;

    scoped_env(const char * name_arg, const char * value) : name(name_arg), had_prev(false) {
        const char * current = std::getenv(name_arg);
        if (current) {
            prev     = current;
            had_prev = true;
        }
        setenv(name_arg, value, 1);
    }

    ~scoped_env() {
        if (had_prev) {
            setenv(name, prev.c_str(), 1);
        } else {
            unsetenv(name);
        }
    }
};

static bool test_layout_selection(int device_id, bool xmx_supported) {
    bool ok = true;

    // The MoE expectation below is derived from the device's XMX capability, so
    // an "expected XMX_TILED, got SOA" line is ambiguous without the inputs:
    // print them, so the failure names whether it is a capability axis or a
    // policy axis.
    const auto & caps = ggml_sycl_info().devices[device_id].xmx_caps;
    printf("[0tkh] layout-selection inputs: device=%d xmx.supported=%d xmx.supports_int8=%d xmx_supported=%d\n",
           device_id, caps.supported ? 1 : 0, caps.supports_int8 ? 1 : 0, xmx_supported ? 1 : 0);

    tensor_usage usage = infer_tensor_usage("attn_q.weight");
    ok &= expect_usage("infer_tensor_usage(attn_q.weight)", usage, tensor_usage::ATTENTION_WEIGHT);
    ok &= expect_layout("layout_policy(attn_q, Q8_0)",
                        layout_policy::get_with_override(GGML_TYPE_Q8_0, usage, device_id),
                        GGML_LAYOUT_COALESCED);

    usage = infer_tensor_usage("ffn_gate_exps.weight");
    ok &= expect_usage("infer_tensor_usage(ffn_gate_exps.weight)", usage, tensor_usage::MOE_EXPERT_WEIGHT);
    // llama.cpp-no1t: layout_policy no longer returns XMX_TILED for MoE experts
    // on ANY branch, so the old `xmx_supported ? XMX_TILED : SOA` was stale in
    // both directions.  The MOE_EXPERT_WEIGHT branch (common.hpp:1359-1374) now
    // reads: XMX+int8 device -> SOA (":1360-1366", "the conservative GPU MoE
    // layout ... XMX_TILED is selected only by the tensor-aware planner"),
    // otherwise MXFP4 -> COALESCED (":1367-1369").  XMX_TILED is the planner's
    // output (common.hpp:1070), not layout_policy's -- the struct contains no
    // GGML_LAYOUT_XMX_TILED identifier at all.  3c8f296fd made that change; the
    // oracle dates from 06c735e93, where the branch really did return XMX_TILED
    // under GGML_SYCL_XMX_MOE_TILED (which is why main() still sets it -- that
    // env is inert HERE but still read elsewhere, so it is left alone).
    const ggml_layout_mode expected_moe = xmx_supported ? GGML_LAYOUT_SOA : GGML_LAYOUT_COALESCED;
    ok &= expect_layout("layout_policy(ffn_gate_exps, MXFP4)",
                        layout_policy::get_with_override(GGML_TYPE_MXFP4, usage, device_id),
                        expected_moe);

    usage = infer_tensor_usage("ffn_up.weight");
    ok &= expect_usage("infer_tensor_usage(ffn_up.weight)", usage, tensor_usage::FFN_WEIGHT);
    ok &= expect_layout("layout_policy(ffn_up, Q4_0)",
                        layout_policy::get_with_override(GGML_TYPE_Q4_0, usage, device_id),
                        GGML_LAYOUT_COALESCED);

    usage = infer_tensor_usage("tok_norm.weight");
    ok &= expect_usage("infer_tensor_usage(tok_norm.weight)", usage, tensor_usage::NORM);
    ok &= expect_layout("layout_policy(tok_norm, F32)",
                        layout_policy::get_with_override(GGML_TYPE_F32, usage, device_id),
                        GGML_LAYOUT_AOS);

    return ok;
}

static bool test_aos_drop(int device_id) {
    reset_layout_choices();

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        fprintf(stderr, "Failed to init CPU backend\n");
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu_backend);
    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    const int64_t ncols = QK8_0 * MMVQ_COALESCED_TILE_BLOCKS;
    const int64_t nrows = 4;
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);
    ggml_set_name(weight, "attn_q.weight");

    const size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to allocate CPU weight buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, ggml_backend_buffer_get_base(weight_buffer));

    const size_t weight_bytes = ggml_nbytes(weight);
    std::vector<uint8_t> host_data(weight_bytes);
    fill_pattern(host_data);
    ggml_backend_tensor_set(weight, host_data.data(), 0, host_data.size());

    ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, device_id);
    if (!key.valid) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to get cache key\n");
        return false;
    }

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(device_id);
    if (!cache) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to get unified cache\n");
        return false;
    }

    if (!expect_layout_cached("test_aos_drop", cache, key, weight, device_id, GGML_LAYOUT_AOS)) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        return false;
    }

    if (!expect_layout_cached("test_aos_drop", cache, key, weight, device_id, GGML_LAYOUT_COALESCED)) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        return false;
    }

    if (cache->is_cached(key, GGML_LAYOUT_AOS)) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "test_aos_drop: AOS cache entry not dropped after COALESCED cache\n");
        return false;
    }

    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);

    return true;
}

// What one MUL_MAT fixture observed about its weight after the graph ran.
struct layout_case_result {
    bool             has_entry = false;  // ggml_sycl_resolve() produced a pointer
    ggml_layout_mode layout    = GGML_LAYOUT_AOS;
    bool             on_device = false;
};

// Runs one MUL_MAT with the weight placed on either the SYCL host buft or the
// device buft, with the layout override either active or absent, and reports
// BOTH axes (routing and layout).  Returns false only when the fixture itself
// could not be built or run -- the oracle lives in the caller, so a single run
// yields every case's reading instead of stopping at the first disagreement.
static bool run_mul_mat_layout_case(int                  device_id,
                                    const char *         label,
                                    bool                 device_buft,
                                    bool                 use_override,
                                    ggml_layout_mode     override_layout,
                                    layout_case_result & out) {
    out = layout_case_result();

    reset_layout_choices();

    scoped_layout_override override_guard(use_override, override_layout);
    scoped_env             disable_graph("GGML_SYCL_DISABLE_GRAPH", "1");

    // main() already returned 77 when no device exists, so a failure here is a
    // real anomaly, not a CPU-only runner.
    ggml_backend_t backend = ggml_backend_sycl_init(device_id);
    if (!backend) {
        printf("FAIL %s: ggml_backend_sycl_init(%d) failed\n", label, device_id);
        return false;
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();
    ggml_backend_buffer_type_t dev_buft  = ggml_backend_get_default_buffer_type(backend);
    if (!host_buft || !dev_buft) {
        printf("FAIL %s: buffer types unavailable (host=%p dev=%p)\n", label, (void *) host_buft, (void *) dev_buft);
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_buffer_type_t weight_buft = device_buft ? dev_buft : host_buft;

    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAIL %s: ggml_init failed\n", label);
        ggml_backend_free(backend);
        return false;
    }

    const int64_t ncols   = QK4_0 * MMVQ_COALESCED_TILE_BLOCKS;
    const int64_t nrows   = 4;
    const int64_t ntokens = 1;

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, ncols, nrows);
    ggml_set_name(weight, "attn_q.weight");
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, ntokens);
    ggml_set_name(input, "layout_choice_input");
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(output, "layout_choice_output");

    ggml_backend_buffer_t weight_buf =
        ggml_backend_buft_alloc_buffer(weight_buft, ggml_backend_buft_get_alloc_size(weight_buft, weight));
    ggml_backend_buffer_t input_buf =
        ggml_backend_buft_alloc_buffer(dev_buft, ggml_backend_buft_get_alloc_size(dev_buft, input));
    ggml_backend_buffer_t output_buf =
        ggml_backend_buft_alloc_buffer(dev_buft, ggml_backend_buft_get_alloc_size(dev_buft, output));

    bool ok = true;
    if (!weight_buf || !input_buf || !output_buf) {
        printf("FAIL %s: buffer allocation failed\n", label);
        ok = false;
    }

    if (ok) {
        ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_set_usage(input_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_buffer_set_usage(output_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

        ggml_backend_tensor_alloc(weight_buf, weight, ggml_backend_buffer_get_base(weight_buf));
        ggml_backend_tensor_alloc(input_buf, input, ggml_backend_buffer_get_base(input_buf));
        ggml_backend_tensor_alloc(output_buf, output, ggml_backend_buffer_get_base(output_buf));

        // Only the host-buft fixture registers a host weight; doing it for the
        // device fixture would re-introduce the host residency this case exists
        // to avoid.
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev && !device_buft) {
            ggml_backend_sycl_register_host_weight_tensor(dev, weight);
        }

        std::vector<uint8_t> weight_data(ggml_nbytes(weight), 0);
        std::vector<float>   input_data(ncols * ntokens, 0.25f);
        ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size());
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            printf("FAIL %s: mul_mat graph compute failed\n", label);
            ok = false;
        }
    }

    if (ok) {
        auto resolved = ggml_sycl_resolve(weight, device_id);
        out.has_entry = static_cast<bool>(resolved);
        if (resolved) {
            out.layout    = resolved.layout;
            out.on_device = resolved.on_device;
        }
        printf("[0tkh] case=%-14s buft=%-6s override=%-9s entry=%-3s layout=%-9s on_device=%s\n", label,
               device_buft ? "device" : "host", use_override ? layout_name(override_layout) : "none",
               out.has_entry ? "yes" : "no", out.has_entry ? layout_name(out.layout) : "-",
               out.on_device ? "yes" : "no");
    }

    if (weight_buf) {
        ggml_backend_buffer_free(weight_buf);
    }
    if (input_buf) {
        ggml_backend_buffer_free(input_buf);
    }
    if (output_buf) {
        ggml_backend_buffer_free(output_buf);
    }
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
}

// Drives the four fixtures the 0tkh adjudication separated, plus one probe.
//
// The no-override CONTROLS are load-bearing, not decoration.  On a freshly
// allocated DEVICE weight ggml_sycl_can_use_layout_for_kernel()
// (ggml-sycl.cpp:53014) has no layout info to match against and falls through
// to its "AOS only" tail, so the orchestrator's override branch (53083) is
// refused and the default policy reaches COALESCED for Q4_0 on its own.
// Without dev/auto printed beside it, dev/coalesced would read as a passing
// override test while proving nothing about the override.
static bool test_mul_mat_layout_choice(int device_id) {
    layout_case_result host_auto;
    layout_case_result host_coalesced;
    layout_case_result dev_auto;
    layout_case_result dev_coalesced;
    layout_case_result dev_soa;

    bool ran = true;
    ran &= run_mul_mat_layout_case(device_id, "host/auto", false, false, GGML_LAYOUT_AOS, host_auto);
    ran &= run_mul_mat_layout_case(device_id, "host/coalesced", false, true, GGML_LAYOUT_COALESCED, host_coalesced);
    ran &= run_mul_mat_layout_case(device_id, "dev/auto", true, false, GGML_LAYOUT_AOS, dev_auto);
    ran &= run_mul_mat_layout_case(device_id, "dev/coalesced", true, true, GGML_LAYOUT_COALESCED, dev_coalesced);
    ran &= run_mul_mat_layout_case(device_id, "dev/soa", true, true, GGML_LAYOUT_SOA, dev_soa);
    if (!ran) {
        return false;
    }

    bool ok = true;

    // Contract 1 -- ROUTING.  A host-buft weight resolves to a host handle and
    // is dispatched to CPU, where AOS is the correct layout.  This is the case
    // the old oracle mistook for a layout-selection test.
    if (!host_coalesced.has_entry) {
        printf("FAIL host/coalesced: no cache entry for the weight after mul_mat\n");
        ok = false;
    } else if (host_coalesced.on_device) {
        printf(
            "FAIL host/coalesced [routing axis]: expected a host handle (on_device=no), got on_device=yes -- the "
            "host-buft weight is no longer CPU-routed, so re-adjudicate llama.cpp-0tkh\n");
        ok = false;
    } else if (host_coalesced.layout != GGML_LAYOUT_AOS) {
        printf("FAIL host/coalesced [layout axis]: expected AOS on the CPU-routed host path, got %s\n",
               layout_name(host_coalesced.layout));
        ok = false;
    }

    // Contract 2 -- OVERRIDE INERTNESS on the host path.  host_auto is the
    // control for host_coalesced: if the COALESCED override really has no
    // authority once the op is CPU-routed, the two must agree on both axes.
    // This is the adjudication's central claim made empirical, instead of
    // resting on the call-graph reading alone.
    if (!host_auto.has_entry) {
        printf("FAIL host/auto: no cache entry for the weight after mul_mat\n");
        ok = false;
    } else if (host_auto.layout != host_coalesced.layout || host_auto.on_device != host_coalesced.on_device) {
        printf(
            "FAIL host/auto vs host/coalesced [override-inertness axis]: the COALESCED override changed the host path "
            "(auto: layout=%s on_device=%s; override: layout=%s on_device=%s) -- the override is NOT inert there, so "
            "the llama.cpp-0tkh adjudication needs re-opening\n",
            layout_name(host_auto.layout), host_auto.on_device ? "yes" : "no", layout_name(host_coalesced.layout),
            host_coalesced.on_device ? "yes" : "no");
        ok = false;
    }

    // Contract 3 -- LAYOUT.  A device-buft weight stays on the GPU, so a layout
    // choice is actually made and observable.
    if (!dev_coalesced.has_entry) {
        printf("FAIL dev/coalesced: no cache entry for the weight after mul_mat\n");
        ok = false;
    } else if (!dev_coalesced.on_device) {
        printf(
            "FAIL dev/coalesced [routing axis]: weight resolved off-device, so no layout choice was exercised -- the "
            "device fixture is routing host\n");
        ok = false;
    } else if (dev_coalesced.layout != GGML_LAYOUT_COALESCED) {
        printf("FAIL dev/coalesced [layout axis]: expected COALESCED on the device path, got %s\n",
               layout_name(dev_coalesced.layout));
        ok = false;
    }

    // Probe, never an assertion: does the override bind on the device path at
    // all?  dev/soa asks for a layout the default policy would not pick, so the
    // answer discriminates.  Both outcomes are informative, which is why
    // neither fails the test.
    if (dev_soa.has_entry && dev_soa.layout == GGML_LAYOUT_SOA) {
        printf(
            "[0tkh] NOTE: the override BINDS on the device path (dev/soa resolved SOA), so dev/coalesced is a genuine "
            "override test.\n");
    } else if (dev_auto.has_entry && dev_coalesced.has_entry && dev_auto.layout == dev_coalesced.layout) {
        printf(
            "[0tkh] NOTE: dev/auto already resolved %s with no override, and dev/soa resolved %s -- the override does "
            "NOT bind on the device path for a fresh weight, so dev/coalesced measures the DEFAULT policy, not the "
            "override (ggml-sycl.cpp:53014 refuses a non-AOS override without matching layout info).\n",
            layout_name(dev_auto.layout), dev_soa.has_entry ? layout_name(dev_soa.layout) : "(no entry)");
    }

    return ok;
}

static bool test_device_weight_layout_cache(int device_id) {
    reset_layout_choices();

    ggml_backend_t backend = ggml_backend_sycl_init(device_id);
    if (!backend) {
        fprintf(stderr, "Failed to init SYCL backend\n");
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_sycl_buffer_type(device_id);
    if (!buft) {
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to get SYCL buffer type\n");
        return false;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    const int64_t ncols = QK8_0 * MMVQ_COALESCED_TILE_BLOCKS;
    const int64_t nrows = 4;
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);
    ggml_set_name(weight, "attn_q.weight");

    const size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to allocate SYCL weight buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, ggml_backend_buffer_get_base(weight_buffer));

    const size_t weight_bytes = ggml_nbytes(weight);
    std::vector<uint8_t> host_data(weight_bytes);
    fill_pattern(host_data);
    ggml_backend_tensor_set(weight, host_data.data(), 0, host_data.size());

    ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, device_id);
    if (!key.valid) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to get cache key for device layout test\n");
        return false;
    }

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(device_id);
    if (!cache) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to get unified cache for device layout test\n");
        return false;
    }

    if (!expect_layout_cached("test_device_weight_layout_cache", cache, key, weight, device_id, GGML_LAYOUT_AOS)) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    if (!expect_layout_cached("test_device_weight_layout_cache", cache, key, weight, device_id,
                              GGML_LAYOUT_COALESCED)) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    if (cache->is_cached(key, GGML_LAYOUT_AOS)) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr,
                "test_device_weight_layout_cache: AOS cache entry not dropped after COALESCED cache for device "
                "weight\n");
        return false;
    }

    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

static bool test_layout_ptr_eviction_guard(int device_id) {
    reset_layout_choices();

    ggml_backend_t backend = ggml_backend_sycl_init(device_id);
    if (!backend) {
        fprintf(stderr, "Failed to init SYCL backend\n");
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_sycl_buffer_type(device_id);
    if (!buft) {
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to get SYCL buffer type\n");
        return false;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 2 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to init ggml context for eviction guard test\n");
        return false;
    }

    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4096);
    ggml_set_name(weight, "attn_q.bias");

    const size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to allocate SYCL weight buffer\n");
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, ggml_backend_buffer_get_base(weight_buffer));

    auto * extra = new ggml_tensor_extra_gpu();
    weight->extra = extra;

    ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, device_id);
    if (!key.valid) {
        delete extra;
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to get cache key for eviction guard test\n");
        return false;
    }

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(device_id);
    if (!cache) {
        delete extra;
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to get unified cache for eviction guard test\n");
        return false;
    }

    std::vector<float> host_data(ggml_nelements(weight), 1.0f);
    bool               needs_fill = false;
    void *             layout_ptr = cache->ensure_cached_alloc(
        key, host_data.data(), ggml_nbytes(weight), ggml_nbytes(weight),
        ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false, &needs_fill);
    if (!layout_ptr || !cache->is_cached(key, GGML_LAYOUT_AOS)) {
        delete extra;
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Failed to cache AOS layout for eviction guard test\n");
        return false;
    }

    extra->layout.mode       = GGML_LAYOUT_AOS;
    extra->layout.data_ptr   = layout_ptr;
    extra->layout.size       = ggml_nbytes(weight);
    extra->layout.owns_memory = false;
    extra->layout.device_id  = device_id;
    extra->layout.qtype      = weight->type;
    extra->layout.n_elements = ggml_nelements(weight);
    extra->layout.n_experts  = 1;

    cache->remove(key, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS);

    void * resolved_ptr = ggml_sycl_resolve_tensor_ptr(weight, device_id);
    if (resolved_ptr != weight->data) {
        delete extra;
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "Evicted layout pointer was reused unexpectedly\n");
        return false;
    }

    weight->extra = nullptr;
    delete extra;
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return true;
}

static bool test_model_load_host_buffer_avoids_pinned(int device_id) {
    GGML_UNUSED(device_id);

    ggml_sycl_load_txn load{};
    ggml_sycl_model_token model{};
    if (ggml_backend_sycl_model_load_begin(&load) != GGML_SYCL_LIFECYCLE_OK) return false;
    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();
    const size_t               size      = 128ULL * 1024ULL * 1024ULL;  // 128MB
    ggml_backend_buffer_t      buffer    = ggml_backend_buft_alloc_buffer(host_buft, size);
    if (ggml_backend_sycl_model_load_end(load, true, &model) != GGML_SYCL_LIFECYCLE_OK) return false;

    if (buffer == nullptr) {
        (void) ggml_backend_sycl_model_unloaded_token(model);
        fprintf(stderr, "test_model_load_host_buffer_avoids_pinned: allocation failed\n");
        return false;
    }

    void *             ptr   = ggml_backend_buffer_get_base(buffer);
    const sycl::usm::alloc typ = sycl::get_pointer_type(ptr, dpct::get_in_order_queue().get_context());
    ggml_backend_buffer_free(buffer);
    (void) ggml_backend_sycl_model_unloaded_token(model);

    if (typ == sycl::usm::alloc::host || typ == sycl::usm::alloc::shared) {
        fprintf(stderr,
                "test_model_load_host_buffer_avoids_pinned: model-load host buffer is USM %s (alloc type %d); "
                "expected unregistered host memory\n",
                typ == sycl::usm::alloc::host ? "host-pinned" : "shared", (int) typ);
        return false;
    }

    return true;
}

static bool test_model_load_preload_caches_weight(int device_id) {
    ggml_sycl::test_clear_host_weight_registry();
    ggml_sycl_load_txn load{};
    ggml_sycl_model_token model{};
    if (ggml_backend_sycl_model_load_begin(&load) != GGML_SYCL_LIFECYCLE_OK) return false;

    ggml_backend_t backend = ggml_backend_sycl_init(device_id);
    if (!backend) {
        fprintf(stderr, "test_model_load_preload_caches_weight: backend init failed\n");
        (void) ggml_backend_sycl_model_load_end(load, false, nullptr);
        return false;
    }

    ggml_init_params params{};
    params.mem_size   = 16 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        (void) ggml_backend_sycl_model_load_end(load, false, nullptr);
        ggml_backend_free(backend);
        fprintf(stderr, "test_model_load_preload_caches_weight: ctx init failed\n");
        return false;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 64);
    ggml_set_name(weight, "attn_q.weight");

    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();
    const size_t weight_size             = ggml_backend_buft_get_alloc_size(host_buft, weight);
    ggml_backend_buffer_t weight_buf     = ggml_backend_buft_alloc_buffer(host_buft, weight_size);
    if (!weight_buf) {
        (void) ggml_backend_sycl_model_load_end(load, false, nullptr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        fprintf(stderr, "test_model_load_preload_caches_weight: buffer alloc failed\n");
        return false;
    }

    ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buf, weight, ggml_backend_buffer_get_base(weight_buf));

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev) {
        ggml_backend_sycl_register_host_weight_tensor(dev, weight);
    }
    ggml_backend_sycl_register_weight_usage(ggml_get_name(weight), GGML_SYCL_TENSOR_USAGE_ATTENTION_WEIGHT);

    std::vector<uint8_t> weight_data(ggml_nbytes(weight), 0);
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size());

    if (ggml_backend_sycl_model_load_end(load, true, &model) != GGML_SYCL_LIFECYCLE_OK) return false;

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(device_id);
    ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, device_id);
    if (!cache || !key.valid || !cache->is_cached_any(key)) {
        fprintf(stderr, "test_model_load_preload_caches_weight: cache miss\n");
        ggml_backend_buffer_free(weight_buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        (void) ggml_backend_sycl_model_unloaded_token(model);
        return false;
    }

    ggml_backend_buffer_free(weight_buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    (void) ggml_backend_sycl_model_unloaded_token(model);
    return true;
}

int main() {
    // The first restored run's evidence (artifacts/b8uv/first-run.txt) had its
    // stdout FAIL lines flushed at exit, landing them tens of lines away from
    // the stderr context that explains them.  Unbuffered keeps the two streams
    // in causal order so one run can be read as a transcript.
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0", 1);
    }

    ggml_sycl::test_clear_layout_override();
    setenv("GGML_SYCL_WEIGHTS_EVICTABLE", "1", 1);
    setenv("GGML_SYCL_XMX_MOE", "1", 1);
    setenv("GGML_SYCL_XMX_MOE_TILED", "1", 1);
    setenv("GGML_SYCL_PINNED_CHUNK_MB", "256", 1);

    const auto & info = ggml_sycl_info();
    if (info.device_count <= 0) {
        // 77 (ctest SKIP_RETURN_CODE), not 0: nothing was verified, so this must not
        // report success. See llama.cpp-k208.
        fprintf(stderr, "SKIP: no SYCL devices available -- NO DEVICE WORK WAS PERFORMED.\n");
        return 77;
    }

    const int device_id = 0;
    const bool xmx_supported = info.devices[device_id].xmx_caps.supported &&
                               info.devices[device_id].xmx_caps.supports_int8;

    // Every sub-test runs even after an earlier one fails, and the summary below
    // reports all seven verdicts.  The first restored run failed FIVE of them
    // while the ticket recorded one, because the failures were scattered through
    // ~90 lines of interleaved backend logging with no roll-up.
    const bool selection = test_layout_selection(device_id, xmx_supported);
    const bool aos_drop  = test_aos_drop(device_id);
    const bool choice    = test_mul_mat_layout_choice(device_id);
    const bool dev_cache = test_device_weight_layout_cache(device_id);
    const bool evict     = test_layout_ptr_eviction_guard(device_id);
    const bool no_pinned = test_model_load_host_buffer_avoids_pinned(device_id);
    const bool preload   = test_model_load_preload_caches_weight(device_id);

    const struct {
        const char * name;
        bool         ok;
    } results[] = {
        { "test_layout_selection",                     selection },
        { "test_aos_drop",                             aos_drop  },
        { "test_mul_mat_layout_choice",                choice    },
        { "test_device_weight_layout_cache",           dev_cache },
        { "test_layout_ptr_eviction_guard",            evict     },
        { "test_model_load_host_buffer_avoids_pinned", no_pinned },
        { "test_model_load_preload_caches_weight",     preload   },
    };

    bool ok       = true;
    int  n_failed = 0;
    printf("\n=== test-layout-cache summary ===\n");
    for (size_t i = 0; i < sizeof(results) / sizeof(results[0]); ++i) {
        printf("  %-44s %s\n", results[i].name, results[i].ok ? "PASS" : "FAIL");
        if (!results[i].ok) {
            ok = false;
            n_failed++;
        }
    }
    printf("=== %d/%d sub-tests passed ===\n", (int) (sizeof(results) / sizeof(results[0])) - n_failed,
           (int) (sizeof(results) / sizeof(results[0])));

    return ok ? 0 : 1;
}
#endif
