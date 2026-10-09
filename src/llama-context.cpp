#include "llama-context.h"

#include "ggml-backend.h"
#include "ggml.h"
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
#    include "ggml-sycl.h"
#endif
#include "llama-arch.h"
#include "llama-auto-ubatch.h"
#include "llama-batch.h"
#include "llama-context-tenant.h"
#include "llama-ext.h"
#include "llama-fused-landing.h"
#include "llama-fused-resolution.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-load-measure.h"
#include "llama-measure-plan.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-residency-fixpoint.h"
#include "llama-sampler.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

//
// llama_context
//

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};


static bool llama_context_sycl_hooks_enabled() {
    return !ggml_backend_device_backends_disabled();
}

static ggml_backend_reg_t llama_context_sycl_reg_from_dev(ggml_backend_dev_t dev) {
    if (!llama_context_sycl_hooks_enabled() || dev == nullptr) {
        return nullptr;
    }
    auto * reg = ggml_backend_dev_backend_reg(dev);
    return reg != nullptr && std::strcmp(ggml_backend_reg_name(reg), "SYCL") == 0 ? reg : nullptr;
}

static bool llama_context_dev_is_sycl(ggml_backend_dev_t dev) {
    return llama_context_sycl_reg_from_dev(dev) != nullptr;
}

// llama.cpp-xojq (quality round 1 Q4): the null-dev and null-reg checks
// below were duplicated verbatim across six proc-address lookups (the two
// pre-existing ones, runtime_proc/recheck_proc, folded into this helper in
// the same edit that added the four new ones for Task 4b). Every lookup is
// now one reinterpret_cast wrapping this call with its own symbol name and
// return type -- no template, since each caller's decltype differs.
static void * llama_context_sycl_proc_addr(ggml_backend_dev_t dev, const char * name) {
    if (!dev) {
        return nullptr;
    }
    auto * reg = llama_context_sycl_reg_from_dev(dev);
    if (!reg) {
        return nullptr;
    }
    return ggml_backend_reg_get_proc_address(reg, name);
}

static bool llama_context_backend_is_sycl(ggml_backend_t backend) {
    return backend != nullptr && llama_context_dev_is_sycl(ggml_backend_get_device(backend));
}

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
static int llama_context_sycl_device_index(ggml_backend_dev_t dev, int fallback) {
    const char * name = ggml_backend_dev_name(dev);
    if (name != nullptr && std::strncmp(name, GGML_SYCL_NAME, std::strlen(GGML_SYCL_NAME)) == 0) {
        const char * p = name + std::strlen(GGML_SYCL_NAME);
        if (*p >= '0' && *p <= '9') {
            return std::atoi(p);
        }
    }
    return fallback;
}

static bool llama_context_has_sycl_backend(const std::vector<ggml_backend_ptr> & backends) {
    for (const auto & backend : backends) {
        if (llama_context_backend_is_sycl(backend.get())) {
            return true;
        }
    }
    return false;
}

#    ifdef GGML_USE_SYCL
static void llama_context_sycl_attach_sched_plan(ggml_backend_sched_t sched,
                                                 const std::vector<ggml_backend_ptr> & backends) {
    if (llama_context_sycl_hooks_enabled() && sched != nullptr && llama_context_has_sycl_backend(backends)) {
        ggml_backend_sycl_set_sched_placement_plan(sched);
    }
}
#    endif

struct llama_context_sycl_exec_hooks {
    decltype(&ggml_backend_sycl_execution_context_create)               create        = nullptr;
    decltype(&ggml_backend_sycl_execution_context_bind_backend)         bind          = nullptr;
    decltype(&ggml_backend_sycl_execution_context_close_if_idle)        close_if_idle = nullptr;
    // Owner-targeted teardown sequence. A SYCL DSO older than llama.cpp-o6jx
    // exports none of these, and teardown then falls back to close_if_idle.
    decltype(&ggml_backend_sycl_execution_context_drain_terminal_events)       drain_terminal = nullptr;
    decltype(&ggml_backend_sycl_execution_context_begin_drain)                 begin_drain    = nullptr;
    decltype(&ggml_backend_sycl_execution_context_extract_control_host_allocs) extract_allocs = nullptr;
    decltype(&ggml_backend_sycl_execution_context_release_control_host_allocs) release_allocs = nullptr;
    decltype(&ggml_backend_sycl_execution_context_finish_drain)                finish_drain   = nullptr;

    bool has_drain_sequence() const {
        return drain_terminal && begin_drain && extract_allocs && release_allocs && finish_drain;
    }
};

#    ifdef GGML_USE_SYCL
static llama_context_sycl_exec_hooks llama_context_sycl_exec_procs(ggml_backend_dev_t /*dev*/) {
    llama_context_sycl_exec_hooks hooks;
    hooks.create         = &ggml_backend_sycl_execution_context_create;
    hooks.bind           = &ggml_backend_sycl_execution_context_bind_backend;
    hooks.close_if_idle  = &ggml_backend_sycl_execution_context_close_if_idle;
    hooks.drain_terminal = &ggml_backend_sycl_execution_context_drain_terminal_events;
    hooks.begin_drain    = &ggml_backend_sycl_execution_context_begin_drain;
    hooks.extract_allocs = &ggml_backend_sycl_execution_context_extract_control_host_allocs;
    hooks.release_allocs = &ggml_backend_sycl_execution_context_release_control_host_allocs;
    hooks.finish_drain   = &ggml_backend_sycl_execution_context_finish_drain;
    return hooks;
}
#    endif

// Owner-targeted context teardown (canonical §12.4/§12.8, llama.cpp-o6jx).
//
// The call order is mandatory: quiesce this exact context, take its drain
// ticket, wait again for its terminal events with the ticket held, extract the
// control-host batch, destroy that batch with no lock held, then finish the
// drain. Calling close_if_idle() on a context that ran graphs reports
// GGML_SYCL_EXECUTION_DEVICE_BUSY (result=4) because its per-device execution
// token is still owned; that path stays only as the fallback for a SYCL DSO too
// old to export the sequence.
static void llama_context_sycl_exec_drain_and_close(const char *                          func,
                                                    const llama_context_sycl_exec_hooks & hooks,
                                                    ggml_sycl_exec_context_id             context) {
    if (context.value == 0) {
        return;
    }

    if (!hooks.has_drain_sequence()) {
        if (hooks.close_if_idle) {
            const auto close_rc = hooks.close_if_idle(context);
            if (close_rc != GGML_SYCL_EXECUTION_OK && close_rc != GGML_SYCL_EXECUTION_STALE) {
                LLAMA_LOG_ERROR("%s: failed to close SYCL execution context: result=%d\n", func, (int) close_rc);
            }
        }
        return;
    }

    // A context nobody else knows about is already gone: STALE here is the
    // idempotent repeat, not a failure, and it must not fall back to a sweep.
    const auto quiesce_rc = hooks.drain_terminal(context);
    if (quiesce_rc != GGML_SYCL_EXECUTION_OK) {
        if (quiesce_rc != GGML_SYCL_EXECUTION_STALE) {
            LLAMA_LOG_ERROR("%s: failed to quiesce SYCL execution context: result=%d\n", func, (int) quiesce_rc);
        }
        return;
    }

    ggml_sycl_exec_drain_ticket ticket   = {};
    const auto                  begin_rc = hooks.begin_drain(context, &ticket);
    if (begin_rc != GGML_SYCL_EXECUTION_OK) {
        if (begin_rc != GGML_SYCL_EXECUTION_STALE) {
            LLAMA_LOG_ERROR("%s: failed to begin SYCL execution drain: result=%d\n", func, (int) begin_rc);
        }
        return;
    }

    // With the ticket held no new session or graph epoch can start, so this is
    // the wait that actually proves quiescence. It runs outside every backend
    // lock.
    const auto wait_rc = hooks.drain_terminal(context);
    if (wait_rc != GGML_SYCL_EXECUTION_OK) {
        LLAMA_LOG_ERROR("%s: failed to drain SYCL execution context: result=%d\n", func, (int) wait_rc);
    }

    ggml_sycl_exec_control_host_alloc_batch batch      = {};
    const auto                              extract_rc = hooks.extract_allocs(&ticket, &batch);
    if (extract_rc != GGML_SYCL_EXECUTION_OK) {
        // Deliberate: returning here strands the registry entry in DRAINING for
        // the process lifetime, because finish_drain is its only exit and also
        // the only path that erases it. Accepted rather than papered over -- the
        // terminal drain above has already cleared this context's device owners
        // (quarantining when the terminal is unprovable), so nothing device-side
        // is held, and every mem_handle is released independently of registry
        // state. A rollback counterpart is registry work owned by 1q72:
        // llama.cpp-34hr.
        LLAMA_LOG_ERROR("%s: failed to extract SYCL control-host allocations: result=%d\n", func, (int) extract_rc);
        return;
    }

    // Final mem_handle destruction, outside every lock, before the drain ends.
    const auto release_rc = hooks.release_allocs(ticket, &batch);
    if (release_rc != GGML_SYCL_EXECUTION_OK) {
        LLAMA_LOG_ERROR("%s: failed to release SYCL control-host allocations: result=%d\n", func, (int) release_rc);
    }

    const auto finish_rc = hooks.finish_drain(ticket, &batch);
    if (finish_rc != GGML_SYCL_EXECUTION_OK) {
        // Same deliberate tradeoff as the extract failure above, and the reason
        // this one should be unreachable: finish_drain's only non-identity
        // refusal is DEVICE_BUSY, which requires a live invocation on one of
        // this context's devices -- exactly what the two terminal drains have
        // already released. Recovery needs a registry rollback (llama.cpp-34hr).
        LLAMA_LOG_ERROR("%s: failed to finish SYCL execution drain: result=%d\n", func, (int) finish_rc);
    }
}
#else
// No SYCL backend can be present in a build with neither macro, so the shared callers (the measure-scope
// requirement in sched_reserve_impl) stay unguarded and compile everywhere (llama.cpp-txho).
static bool llama_context_has_sycl_backend(const std::vector<ggml_backend_ptr> &) {
    return false;
}
#endif

#if defined(GGML_BACKEND_DL) && !defined(GGML_USE_SYCL)
struct llama_context_sycl_dl_compute_hooks {
    decltype(&ggml_backend_sycl_host_compute_buffer_type)         host_compute    = nullptr;
    decltype(&ggml_backend_sycl_cpu_offload_compute_buffer_type)  cpu_compute     = nullptr;
    decltype(&ggml_backend_sycl_cpu_offload_available)            cpu_available   = nullptr;
    decltype(&ggml_backend_sycl_has_active_placement_plan)        has_active_plan = nullptr;
    int                                                           device_index    = -1;
};

static llama_context_sycl_dl_compute_hooks llama_context_sycl_compute_procs(ggml_backend_dev_t dev) {
    llama_context_sycl_dl_compute_hooks hooks;
    if (!dev) {
        return hooks;
    }
    auto * reg = llama_context_sycl_reg_from_dev(dev);
    if (!reg) {
        return hooks;
    }
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
        if (ggml_backend_reg_dev_get(reg, i) == dev) {
            hooks.device_index = static_cast<int>(i);
            break;
        }
    }
    if (hooks.device_index < 0) {
        return {};
    }
    hooks.host_compute = reinterpret_cast<decltype(hooks.host_compute)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_host_compute_buffer_type"));
    hooks.cpu_compute = reinterpret_cast<decltype(hooks.cpu_compute)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_cpu_offload_compute_buffer_type"));
    hooks.cpu_available = reinterpret_cast<decltype(hooks.cpu_available)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_cpu_offload_available"));
    hooks.has_active_plan = reinterpret_cast<decltype(hooks.has_active_plan)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_has_active_placement_plan"));
    return hooks;
}

static llama_context_sycl_exec_hooks llama_context_sycl_exec_procs(ggml_backend_dev_t dev) {
    llama_context_sycl_exec_hooks hooks;
    if (!dev) {
        return hooks;
    }
    auto * reg = llama_context_sycl_reg_from_dev(dev);
    if (!reg) {
        return hooks;
    }
    hooks.create = reinterpret_cast<decltype(hooks.create)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_execution_context_create"));
    hooks.bind = reinterpret_cast<decltype(hooks.bind)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_execution_context_bind_backend"));
    hooks.close_if_idle = reinterpret_cast<decltype(hooks.close_if_idle)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_execution_context_close_if_idle"));
    hooks.drain_terminal = reinterpret_cast<decltype(hooks.drain_terminal)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_execution_context_drain_terminal_events"));
    hooks.begin_drain = reinterpret_cast<decltype(hooks.begin_drain)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_execution_context_begin_drain"));
    hooks.extract_allocs = reinterpret_cast<decltype(hooks.extract_allocs)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_execution_context_extract_control_host_allocs"));
    hooks.release_allocs = reinterpret_cast<decltype(hooks.release_allocs)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_execution_context_release_control_host_allocs"));
    hooks.finish_drain = reinterpret_cast<decltype(hooks.finish_drain)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_execution_context_finish_drain"));
    return hooks;
}

static decltype(&ggml_backend_sycl_set_runtime_context_for_model) llama_context_sycl_runtime_proc(
    ggml_backend_dev_t dev) {
    return reinterpret_cast<decltype(&ggml_backend_sycl_set_runtime_context_for_model)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_set_runtime_context_for_model"));
}

// llama.cpp-oyfl: lookup for the NARROW flash-attn-only re-check,
// mirroring llama_context_sycl_runtime_proc() above but for
// ggml_backend_sycl_recheck_runtime_context_flash_attn() (see that
// declaration's comment, ggml-sycl.h, for why it is a separate entry point
// rather than a second call into the full transaction).
static decltype(&ggml_backend_sycl_recheck_runtime_context_flash_attn) llama_context_sycl_recheck_proc(
    ggml_backend_dev_t dev) {
    return reinterpret_cast<decltype(&ggml_backend_sycl_recheck_runtime_context_flash_attn)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_recheck_runtime_context_flash_attn"));
}

// llama.cpp-xojq (nphx Task 4b): proc-address lookups for the auto
// micro-batch trial's four SYCL entry points, mirroring
// llama_context_sycl_runtime_proc()/llama_context_sycl_recheck_proc() above.
// A GGML_BACKEND_DL build whose SYCL DSO predates llama.cpp-tsfl (the probe
// and fallback counter) or this task (auto_ubatch_enabled's registration,
// moe_gpu_ubatch_max) exports none of these; sycl_select_auto_ubatch()
// treats a nullptr proc the same way every other caller in this file
// already does for llama_context_sycl_runtime_proc() -- as "unavailable",
// never dereferenced.
static decltype(&ggml_backend_sycl_probe_runtime_context_for_model) llama_context_sycl_probe_proc(
    ggml_backend_dev_t dev) {
    return reinterpret_cast<decltype(&ggml_backend_sycl_probe_runtime_context_for_model)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_probe_runtime_context_for_model"));
}

static decltype(&ggml_backend_sycl_compute_buffer_host_fallbacks) llama_context_sycl_fallbacks_proc(
    ggml_backend_dev_t dev) {
    return reinterpret_cast<decltype(&ggml_backend_sycl_compute_buffer_host_fallbacks)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_compute_buffer_host_fallbacks"));
}

// llama.cpp-kpjw: the per-rung hold-spill check (see ggml_backend_sycl_planned_hold_spill_fits in ggml-sycl.h). A SYCL
// DSO that predates it exports nothing; the trial then skips the check, never dereferences a null.
static decltype(&ggml_backend_sycl_planned_hold_spill_fits) llama_context_sycl_hold_spill_proc(ggml_backend_dev_t dev) {
    return reinterpret_cast<decltype(&ggml_backend_sycl_planned_hold_spill_fits)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_planned_hold_spill_fits"));
}

// llama.cpp-mmi1: the by-name text for a refused scheduler compute buffer (see ggml_backend_sycl_compute_refusal_advice
// in ggml-sycl.h). A SYCL DSO that predates it exports nothing and the refusal carries no extra text.
static decltype(&ggml_backend_sycl_compute_refusal_advice) llama_context_sycl_compute_refusal_proc(
    ggml_backend_dev_t dev) {
    return reinterpret_cast<decltype(&ggml_backend_sycl_compute_refusal_advice)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_compute_refusal_advice"));
}

// llama.cpp-kpjw: re-reads the KV room a pinned -ub's hold epoch is judged with (ggml_backend_sycl_planned_hold_epoch_refresh
// in ggml-sycl.h). A SYCL DSO that predates it exports nothing and is skipped.
static decltype(&ggml_backend_sycl_planned_hold_epoch_refresh) llama_context_sycl_hold_epoch_refresh_proc(
    ggml_backend_dev_t dev) {
    return reinterpret_cast<decltype(&ggml_backend_sycl_planned_hold_epoch_refresh)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_planned_hold_epoch_refresh"));
}

static decltype(&ggml_backend_sycl_auto_ubatch_enabled) llama_context_sycl_auto_ubatch_enabled_proc(
    ggml_backend_dev_t dev) {
    return reinterpret_cast<decltype(&ggml_backend_sycl_auto_ubatch_enabled)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_auto_ubatch_enabled"));
}

static decltype(&ggml_backend_sycl_moe_gpu_ubatch_max) llama_context_sycl_moe_gpu_ubatch_max_proc(
    ggml_backend_dev_t dev) {
    return reinterpret_cast<decltype(&ggml_backend_sycl_moe_gpu_ubatch_max)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_moe_gpu_ubatch_max"));
}

// llama.cpp-7n6n (wires nphx Task 5): proc-address lookups for the
// persisted auto n_ubatch tuning cache's four entry points, mirroring the
// four lookups just above. A GGML_BACKEND_DL build whose SYCL DSO predates
// this task exports none of these; sycl_select_auto_ubatch() treats a
// nullptr proc as "cache unavailable" -- it skips the lookup/store and
// runs the ladder exactly as it did before this task, never dereferencing
// a null function pointer.
static decltype(&ggml_backend_sycl_ubatch_cache_enabled) llama_context_sycl_ubatch_cache_enabled_proc(
    ggml_backend_dev_t dev) {
    return reinterpret_cast<decltype(&ggml_backend_sycl_ubatch_cache_enabled)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_ubatch_cache_enabled"));
}

static decltype(&ggml_backend_sycl_ubatch_cache_path) llama_context_sycl_ubatch_cache_path_proc(
    ggml_backend_dev_t dev) {
    return reinterpret_cast<decltype(&ggml_backend_sycl_ubatch_cache_path)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_ubatch_cache_path"));
}

static decltype(&ggml_backend_sycl_ubatch_cache_lookup_layout1) llama_context_sycl_ubatch_cache_lookup_proc(
    ggml_backend_dev_t dev) {
    return reinterpret_cast<decltype(&ggml_backend_sycl_ubatch_cache_lookup_layout1)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_ubatch_cache_lookup_layout1"));
}

static decltype(&ggml_backend_sycl_ubatch_cache_store_layout1) llama_context_sycl_ubatch_cache_store_proc(
    ggml_backend_dev_t dev) {
    return reinterpret_cast<decltype(&ggml_backend_sycl_ubatch_cache_store_layout1)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_ubatch_cache_store_layout1"));
}
#endif

// The SYCL backend's capability-only supports_op (see ggml_backend_sycl_supports_op_capability in ggml-sycl.h), or
// nullptr when the device is not SYCL or the loaded SYCL DSO predates it; resolve_fused_ops() then falls back to
// the operand heuristic in llama-fused-landing.h.
static llama_fused_capability_fn llama_context_sycl_capability_proc(ggml_backend_dev_t dev) {
#if defined(GGML_USE_SYCL)
    return llama_context_dev_is_sycl(dev) ? &ggml_backend_sycl_supports_op_capability : nullptr;
#elif defined(GGML_BACKEND_DL)
    return reinterpret_cast<llama_fused_capability_fn>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_supports_op_capability"));
#else
    GGML_UNUSED(dev);
    return nullptr;
#endif
}

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
// llama.cpp-kpjw (r7 I3): a pinned -ub publishes its plan ONCE, before the memory module (the KV cache, the recurrent
// state) exists, so the KV room its hold epoch began with predates both. This re-reads it once they exist, before the
// compute buffers are reserved, so the context-init check judges with the room the rung actually had. (What the rung
// does not own is not snapshotted: it is every raw row but the rung's scheduler compute rows, read when asked.)
static void llama_context_sycl_hold_epoch_refresh(const std::vector<ggml_backend_ptr> & backends) {
    for (auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (!llama_context_dev_is_sycl(dev)) {
            continue;
        }
#    ifdef GGML_USE_SYCL
        auto refresh_fn = &ggml_backend_sycl_planned_hold_epoch_refresh;
#    else
        auto refresh_fn = llama_context_sycl_hold_epoch_refresh_proc(dev);
        if (!refresh_fn) {
            continue;
        }
#    endif
        refresh_fn(backend.get());
    }
}

// llama.cpp-kpjw: the realized hold-spill check over every SYCL backend of a context, for a reserve that ran at
// `n_ubatch` (see ggml_backend_sycl_planned_hold_spill_fits in ggml-sycl.h). False when ANY backend's compute buffers
// the planned dense scratch kept out of the RUNTIME zone spilled outside the arena and left its card under the driver
// headroom; `*largest_ub` is then the smallest -ub the refusing backends say still fits (0: none known to). A SYCL
// DSO that predates the entry exports nothing and is skipped, never dereferenced.
static bool llama_context_sycl_hold_spill_fits(const std::vector<ggml_backend_ptr> & backends,
                                               uint32_t                              n_ubatch,
                                               uint32_t *                            largest_ub) {
    if (largest_ub) {
        *largest_ub = 0;
    }
    bool fits = true;
    for (auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (!llama_context_dev_is_sycl(dev)) {
            continue;
        }
#    ifdef GGML_USE_SYCL
        auto hold_spill_fn = &ggml_backend_sycl_planned_hold_spill_fits;
#    else
        auto hold_spill_fn = llama_context_sycl_hold_spill_proc(dev);
        if (!hold_spill_fn) {
            continue;
        }
#    endif
        uint32_t backend_largest = 0;
        if (!hold_spill_fn(backend.get(), n_ubatch, &backend_largest)) {
            if (largest_ub) {
                *largest_ub = fits ? backend_largest : std::min(*largest_ub, backend_largest);
            }
            fits = false;
        }
    }
    return fits;
}
#endif

// llama.cpp-mmi1: the text a "failed to allocate compute ... buffers" refusal appends when a SYCL backend could not
// place the scheduler compute buffer: what was asked, the room each tier had, the largest -ub that fits and the
// GGML_SYCL_VRAM_BUDGET_PCT that would free enough (never a smaller -c; KV is placed, not shrunk). Empty when no SYCL
// backend recorded such a refusal, so a failure of another cause says nothing wrong, and with no SYCL backend or a
// SYCL DSO that predates the entry. `n_ubatch` is the shape the reserve ran at.
static std::string llama_context_sycl_compute_refusal_text(const std::vector<ggml_backend_ptr> & backends,
                                                           uint32_t                              n_ubatch) {
    std::string text;
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    for (auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (!llama_context_dev_is_sycl(dev)) {
            continue;
        }
#    ifdef GGML_USE_SYCL
        auto advice_fn = &ggml_backend_sycl_compute_refusal_advice;
#    else
        auto advice_fn = llama_context_sycl_compute_refusal_proc(dev);
        if (!advice_fn) {
            continue;
        }
#    endif
        char buf[2048];
        if (advice_fn(backend.get(), n_ubatch, buf, sizeof(buf)) > 0) {
            text += text.empty() ? ": " : "; ";
            text += buf;
        }
    }
#else
    GGML_UNUSED(backends);
    GGML_UNUSED(n_ubatch);
#endif
    return text;
}

// llama.cpp-kpjw: opens the SYCL backend's scheduler-compute scope for its lifetime (see
// llama_context::sycl_compute_scope_fn). A null function (no SYCL backend, or a library without the export) opens
// nothing.
struct sycl_compute_scope_guard {
    void (*fn)(bool);

    explicit sycl_compute_scope_guard(void (*f)(bool)) : fn(f) {
        if (fn) {
            fn(true);
        }
    }

    ~sycl_compute_scope_guard() {
        if (fn) {
            fn(false);
        }
    }

    sycl_compute_scope_guard(const sycl_compute_scope_guard &)             = delete;
    sycl_compute_scope_guard & operator=(const sycl_compute_scope_guard &) = delete;
};

llama_context::sycl_compute_scope_fn_t llama_context::sycl_compute_scope_fn() {
    if (!sycl_compute_scope_resolved) {
        sycl_compute_scope_resolved = true;
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            if (!llama_context_dev_is_sycl(dev)) {
                continue;
            }
#    ifdef GGML_USE_SYCL
            sycl_compute_scope_cached = &ggml_backend_sycl_compute_alloc_scope;
#    else
            sycl_compute_scope_cached = reinterpret_cast<sycl_compute_scope_fn_t>(
                llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_compute_alloc_scope"));
#    endif
            break;
        }
#endif
    }
    return sycl_compute_scope_cached;
}

// llama.cpp-kpjw: the ONLY two places a graph is allocated on this context's scheduler. A buffer the SYCL backend
// places while the scope is open is positively a scheduler compute buffer (it asks for the KV-zone-first placement,
// and feeds the per-rung request record the hold-spill fit judges from); one it places outside the scope is an ordinary
// request of the RUNTIME zone that nothing plans for or counts. Every allocation on the scheduler is a compute
// allocation, whoever asks (the decode graph, the reserve, the K-shift graph llama_kv_cache::update builds), so none
// goes to ggml_backend_sched_alloc_graph / ggml_backend_sched_reserve bare: a source gate pins that.
bool llama_context::sched_alloc_graph(ggml_cgraph * gf) {
    sycl_compute_scope_guard sycl_scope(sycl_compute_scope_fn());
    return ggml_backend_sched_alloc_graph(sched.get(), gf);
}

bool llama_context::sched_reserve_graph(ggml_cgraph * gf) {
    sycl_compute_scope_guard sycl_scope(sycl_compute_scope_fn());
    return ggml_backend_sched_reserve(sched.get(), gf);
}

// llama.cpp-38af: the compute-buffer buft for the CPU backend when the first
// device is a SYCL device. It is the generic host buft's pinned memory under a
// distinct identity that the SYCL backend never reports as supported, so
// ggml-backend-sched copies every CPU-produced activation into the SYCL
// backend's device compute buffer instead of a SYCL op reading pinned host
// memory in place. Returns nullptr for a non-SYCL device, or a SYCL module that
// predates the export; the caller then keeps the device's generic host buft.
static ggml_backend_buffer_type_t llama_context_sycl_cpu_activation_buft(ggml_backend_dev_t dev) {
#if defined(GGML_USE_SYCL)
    return llama_context_dev_is_sycl(dev) ? ggml_backend_sycl_cpu_activation_buffer_type() : nullptr;
#elif defined(GGML_BACKEND_DL)
    auto proc = reinterpret_cast<decltype(&ggml_backend_sycl_cpu_activation_buffer_type)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_cpu_activation_buffer_type"));
    return proc ? proc() : nullptr;
#else
    GGML_UNUSED(dev);
    return nullptr;
#endif
}

// llama.cpp-38af: the backend's own answer to "will the CPU execute any part of
// this graph", read from its published placement plan (host-planned dense layer,
// host-planned KV, or a fully host-planned expert tensor). False for a non-SYCL
// device, no active plan, or a SYCL module that predates the export, in which
// case the caller keeps the generic host buft.
static bool llama_context_sycl_plan_has_cpu_work(ggml_backend_dev_t dev) {
#if defined(GGML_USE_SYCL)
    return llama_context_dev_is_sycl(dev) && ggml_backend_sycl_plan_has_cpu_work(dev);
#elif defined(GGML_BACKEND_DL)
    auto proc = reinterpret_cast<decltype(&ggml_backend_sycl_plan_has_cpu_work)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_plan_has_cpu_work"));
    return proc && proc(dev);
#else
    GGML_UNUSED(dev);
    return false;
#endif
}

// llama.cpp-38af: the compute-buffer buft for the CPU backend. `buft` is the
// backend's default. The first device's host buft is used for faster transfer of
// the intermediate state; when the CPU will actually produce activations a SYCL
// op consumes -- the backend's plan has CPU work, or this is a partial offload
// (n_gpu_layers does not cover every layer and the output) -- it gets the
// dedicated activation buft instead, so the scheduler copies those activations to
// the device. With no CPU work every input is read out of the host buft in place,
// and the dedicated identity would only add split-copy names to the graph (the
// dkw0 replay-futility detector disables command-graph replay on a '#' name).
//
// The answer depends on the placement plan, which the auto-ubatch resyncs
// re-publish after the constructor enumerates the backends, so sched_reserve()
// calls this again right before it builds each scheduler.
static ggml_backend_buffer_type_t llama_context_cpu_compute_buft(const llama_model &        model,
                                                                 ggml_backend_buffer_type_t buft,
                                                                 bool                       log) {
    if (model.devices.empty()) {
        return buft;
    }
    const auto & dev = model.devices[0];
    if (auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev)) {
        buft = host_buft;
    }
    const bool plan_cpu_work   = llama_context_sycl_plan_has_cpu_work(dev.dev);
    const bool partial_offload = model.n_gpu_layers() <= model.hparams.n_layer_all;
    if (plan_cpu_work || partial_offload) {
        if (auto * activation_buft = llama_context_sycl_cpu_activation_buft(dev.dev)) {
            buft = activation_buft;
        }
    }
    // One line per selection; the backend logs which plan clause fired just above it.
    if (!log) {
        return buft;
    }
    LLAMA_LOG_DEBUG("[SYCL-CPU-ACT] CPU compute buft '%s': plan_cpu_work=%d partial_offload=%d\n",
                    ggml_backend_buft_name(buft), plan_cpu_work, partial_offload);
    return buft;
}

// llama.cpp-38af: re-select the CPU backend's compute buft against the placement plan as it stands NOW.
// The auto-ubatch candidate and settle resyncs re-plan after the constructor chose it, and no plan
// mutation happens between a reserve's call here and its scheduler's construction (the narrow flash-attn
// recheck in resolve_fused_ops() runs with allow_replan=false), so this is the final answer for that
// scheduler; the pipeline-parallel retry reuses it. The MEASURE and the ALLOC of one reserve both call it
// so they plan against the same buft.
static void llama_context_cpu_compute_buft_reselect(const llama_model &                       model,
                                                    const std::vector<ggml_backend_t> &       backend_ptrs,
                                                    std::vector<ggml_backend_buffer_type_t> & backend_buft,
                                                    bool                                      log) {
    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        if (!model.devices.empty() &&
            ggml_backend_dev_type(ggml_backend_get_device(backend_ptrs[i])) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            backend_buft[i] =
                llama_context_cpu_compute_buft(model, ggml_backend_get_default_buffer_type(backend_ptrs[i]), log);
        }
    }
}

// llama.cpp-7n6n (wires nphx Task 5): a cheap, deterministic FNV-1a hash
// over every loaded tensor's (name, byte size) -- a proxy for "this exact
// set of quantized weights", used only to invalidate the persisted auto
// n_ubatch cache entry when the model file underneath changes (re-quantised,
// replaced) even though its GGUF general.name and total byte count
// (llama_model::size()) might coincidentally still match. Not a general
// model-identity primitive: it does not need to be collision-resistant or
// order-independent across different loaders, only reproducible for the
// SAME loader loading the SAME file (tensors_by_name's iteration order is
// the load order, which is deterministic for one GGUF).
static uint64_t llama_context_sycl_model_tensor_hash(const llama_model & model) {
    uint64_t h         = 0xcbf29ce484222325ULL;  // FNV-1a 64-bit offset basis
    auto     mix_bytes = [&h](const void * data, size_t n) {
        const uint8_t * p = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < n; ++i) {
            h ^= p[i];
            h *= 0x100000001b3ULL;  // FNV-1a 64-bit prime
        }
    };
    for (const auto & nt : model.tensors_by_name) {
        mix_bytes(nt.first.data(), nt.first.size());
        const size_t nbytes = ggml_nbytes(nt.second);
        mix_bytes(&nbytes, sizeof(nbytes));
    }
    return h;
}

// llama.cpp-oyfl: name the specific ggml_sycl_lifecycle_result the narrow
// re-check can return, so a thrown exception distinguishes the guard's own
// PLAN_REJECTED (whose arithmetic and remediation are already printed by
// ggml_sycl_check_nonfa_attn_scratch() to the [SYCL-PLAN] log) from every
// other result, which are argument-validation/identity failures this call
// should not normally see at all. Guarded by the SAME broad condition as
// the block at lines 87-233 above (GGML_USE_SYCL || GGML_BACKEND_DL) --
// NOT the narrower GGML_BACKEND_DL-without-GGML_USE_SYCL condition the
// proc-lookup helpers immediately above this comment need. This function
// only needs the enum ggml-sycl.h declares under that broader condition
// (see the #include near the top of this file), and its caller
// (sycl_recheck_runtime_context_flash_attn(), far below) is reachable in
// a direct GGML_USE_SYCL build too, where that narrower block never
// compiles at all. (Previously defined inside that narrower block by
// mistake, which left it undeclared in a direct GGML_USE_SYCL build --
// build-oyfl-6 caught this.)
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
static const char * sycl_recheck_lifecycle_result_name(ggml_sycl_lifecycle_result rc) {
    switch (rc) {
        case GGML_SYCL_LIFECYCLE_OK:
            return "OK";
        case GGML_SYCL_LIFECYCLE_NULL_OUTPUT:
            return "NULL_OUTPUT (invalid backend or context)";
        case GGML_SYCL_LIFECYCLE_FOREIGN_BACKEND:
            return "FOREIGN_BACKEND (not a SYCL device)";
        case GGML_SYCL_LIFECYCLE_STALE_IDENTITY:
            return "STALE_IDENTITY (no plan is currently published, the model token no longer matches "
                   "the published plan, or the plan snapshot changed under the lock)";
        case GGML_SYCL_LIFECYCLE_BUSY:
            return "BUSY (module admission refused -- shutdown in progress)";
        case GGML_SYCL_LIFECYCLE_PLAN_REJECTED:
            return "PLAN_REJECTED (the non-FA attention scratch guard refused this shape -- see the "
                   "[SYCL-PLAN] log lines above for the arithmetic and remediation)";
        default:
            return "unrecognized ggml_sycl_lifecycle_result";
    }
}
#endif

// The backend's plan-scope entry points. A SYCL DSO that does not export them leaves
// every proc null; a planned context cannot exist without them, because the copy that
// owns the scopes is acquired through the same table.
struct llama_context_sycl_plan_procs {
    decltype(&ggml_backend_sycl_plan_scope_open)              scope_open              = nullptr;
    decltype(&ggml_backend_sycl_plan_scope_open_load_measure) scope_open_load_measure = nullptr;
    decltype(&ggml_backend_sycl_plan_scope_failure)           scope_failure           = nullptr;
    decltype(&ggml_backend_sycl_plan_scope_close)             scope_close             = nullptr;
};

static llama_context_sycl_plan_procs llama_context_sycl_plan_procs_for(const std::vector<ggml_backend_ptr> & backends) {
    llama_context_sycl_plan_procs procs;
    for (const auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (!llama_context_dev_is_sycl(dev)) {
            continue;
        }
#ifdef GGML_USE_SYCL
        procs.scope_open              = &ggml_backend_sycl_plan_scope_open;
        procs.scope_open_load_measure = &ggml_backend_sycl_plan_scope_open_load_measure;
        procs.scope_failure           = &ggml_backend_sycl_plan_scope_failure;
        procs.scope_close             = &ggml_backend_sycl_plan_scope_close;
#elif defined(GGML_BACKEND_DL)
        procs.scope_open = reinterpret_cast<decltype(procs.scope_open)>(
            llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_plan_scope_open"));
        procs.scope_open_load_measure = reinterpret_cast<decltype(procs.scope_open_load_measure)>(
            llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_plan_scope_open_load_measure"));
        procs.scope_failure = reinterpret_cast<decltype(procs.scope_failure)>(
            llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_plan_scope_failure"));
        procs.scope_close = reinterpret_cast<decltype(procs.scope_close)>(
            llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_plan_scope_close"));
#endif
        break;
    }
    return procs;
}

// The L4 entry points (the tenant publish, coverage query, load-time late check and residency probe). The backend
// declares them in ggml-sycl.h; a backend that does not define them answers a null proc address
// and the readers in llama-context-tenant.h then fail closed. Every link mode resolves them the
// same way, through the SYCL reg's proc address by the names ggml-sycl-l4-procs.h pins, from the
// first SYCL backend of the context. No weak reference, no direct reference: one path.
[[maybe_unused]] static llama_sycl_l4_procs llama_context_sycl_l4_procs_for_dev(ggml_backend_dev_t dev) {
    llama_sycl_l4_procs procs;
    if (!llama_context_dev_is_sycl(dev)) {
        return procs;
    }
    procs.publish = reinterpret_cast<decltype(procs.publish)>(
        llama_context_sycl_proc_addr(dev, GGML_SYCL_PROC_SET_RUNTIME_CONTEXT_DESC));
    procs.coverage =
        reinterpret_cast<decltype(procs.coverage)>(llama_context_sycl_proc_addr(dev, GGML_SYCL_PROC_TENANT_COVERAGE));
    procs.late_check =
        reinterpret_cast<decltype(procs.late_check)>(llama_context_sycl_proc_addr(dev, GGML_SYCL_PROC_LOAD_LATE_CHECK));
    procs.probe_residency = reinterpret_cast<decltype(procs.probe_residency)>(
        llama_context_sycl_proc_addr(dev, GGML_SYCL_PROC_PROBE_RESIDENCY));
    return procs;
}

[[maybe_unused]] static llama_sycl_l4_procs llama_context_sycl_l4_procs_for(
    const std::vector<ggml_backend_ptr> & backends) {
    for (const auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (llama_context_dev_is_sycl(dev)) {
            return llama_context_sycl_l4_procs_for_dev(dev);
        }
    }
    return {};
}

// llama.cpp-7gno: what the constructor's planned-reserve decision reads. The chunk-cap copy's two procs come the way the
// scope's own do (the symbols in a direct build, the reg's proc address under GGML_BACKEND_DL), from the first SYCL backend
// of the context. `plan_active` is the process-global active-plan predicate at construction.
struct llama_context_sycl_plan_caps_procs {
    decltype(&ggml_backend_sycl_plan_caps_new)  caps_new    = nullptr;
    decltype(&ggml_backend_sycl_plan_caps_free) caps_free   = nullptr;
    bool                                        plan_active = false;
};

[[maybe_unused]] static llama_context_sycl_plan_caps_procs llama_context_sycl_plan_caps_procs_for(
    const std::vector<ggml_backend_ptr> & backends) {
    llama_context_sycl_plan_caps_procs procs;
    for (const auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (!llama_context_dev_is_sycl(dev)) {
            continue;
        }
#ifdef GGML_USE_SYCL
        procs.caps_new    = &ggml_backend_sycl_plan_caps_new;
        procs.caps_free   = &ggml_backend_sycl_plan_caps_free;
        procs.plan_active = llama_context_sycl_hooks_enabled() && ggml_backend_sycl_has_active_placement_plan();
#elif defined(GGML_BACKEND_DL)
        procs.caps_new = reinterpret_cast<decltype(procs.caps_new)>(
            llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_plan_caps_new"));
        procs.caps_free = reinterpret_cast<decltype(procs.caps_free)>(
            llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_plan_caps_free"));
        const auto hooks  = llama_context_sycl_compute_procs(dev);
        procs.plan_active = hooks.has_active_plan && hooks.has_active_plan();
#endif
        break;
    }
    return procs;
}

// llama.cpp-7gno: whether the L4 surface the planned reserve needs is all there. moua's three procs
// (llama_sycl_l4_procs::available()) are not enough by themselves: the residency fixpoint also needs the tenant-aware
// residency probe, which L4 does not define yet. Until that probe is looked up here, and this flag flips with it, no
// context is planned whatever the backend exports, so the three procs landing cannot turn the legacy publish path
// into a construction failure.
// This flag is an interim latch, not a second source of truth: the eventual single source of "the probe is wired" is
// the probe proc's presence in llama_sycl_l4_procs::available(). llama.cpp-hdpd names the proc, adds it to available()
// and deletes this constant in the same commit (llama_context_l4_ready then asks the proc, not the constant).
static constexpr bool llama_context_residency_probe_wired = false;

[[maybe_unused]] static bool llama_context_l4_ready(const std::vector<ggml_backend_ptr> & backends) {
    return llama_context_residency_probe_wired && llama_context_sycl_l4_procs_for(backends).available();
}

// A plan scope open on the calling thread for one MEASURE or ALLOC of one context,
// closed when it leaves scope. While it is open the compute bufts' get_max_size answer
// from the context's chunk-cap copy. It opens only for a context that owns that copy
// (`caps` non-null): for any other it stays closed, and the caller does not ask.
struct llama_plan_scope {
    llama_plan_scope(const llama_context_sycl_plan_procs & procs,
                     uint32_t                              context_id,
                     enum ggml_sycl_plan_scope_mode        mode,
                     ggml_backend_sycl_plan_caps_t         caps) :
        failure_fn(procs.scope_failure),
        close_fn(procs.scope_close) {
        if (caps != nullptr && procs.scope_open != nullptr && close_fn != nullptr) {
            scope = procs.scope_open(context_id, mode, caps);
        }
    }

    // The load-time measure's scope: it holds no copy and freezes nothing, and reads the plan override
    // the measure function installed.
    llama_plan_scope(const llama_context_sycl_plan_procs & procs, enum ggml_sycl_measure_stage stage) :
        failure_fn(procs.scope_failure),
        close_fn(procs.scope_close) {
        if (procs.scope_open_load_measure != nullptr && close_fn != nullptr) {
            scope = procs.scope_open_load_measure(stage);
        }
    }

    ~llama_plan_scope() {
        if (scope != nullptr) {
            close_fn(scope);
        }
    }

    llama_plan_scope(const llama_plan_scope &)             = delete;
    llama_plan_scope & operator=(const llama_plan_scope &) = delete;

    bool is_open() const { return scope != nullptr; }

    // The first failed read in the scope (a cap the copy could not answer), or null.
    const char * failure() const { return scope != nullptr && failure_fn != nullptr ? failure_fn(scope) : nullptr; }

  private:
    decltype(&ggml_backend_sycl_plan_scope_failure) failure_fn = nullptr;
    decltype(&ggml_backend_sycl_plan_scope_close)   close_fn   = nullptr;
    void *                                          scope      = nullptr;
};

// The sink the resolution report prints through, and the trampolines the unwind guard calls.
static void fused_resolution_sink(fused_resolution_level level, const char * text) {
    if (level == FUSED_RESOLUTION_LEVEL_WARN) {
        LLAMA_LOG_WARN("%s\n", text);
    } else {
        LLAMA_LOG_INFO("%s\n", text);
    }
}

struct fused_resolution_guard_arg {
    llama_context *          ctx;
    const fused_resolution * record;
};

static void fused_resolution_report_trampoline(void * arg) {
    auto * a = static_cast<fused_resolution_guard_arg *>(arg);
    a->ctx->fused_resolution_report(*a->record);
}

static void fused_resolution_lost_trampoline(void * arg) noexcept {
    static_cast<fused_resolution_guard_arg *>(arg)->ctx->fused_resolution_report_lost();
}

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params,
              llama_measure_context_args * measure) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    measure_only = measure != nullptr;
    if (measure_only) {
        measure_log_quiet.emplace();
    }

    // A model the measure cannot walk is refused by name, before anything is built. llama_load_measure
    // asks the same question first and returns it as `unsupported`; this is the constructor's own guard.
    if (measure_only) {
        const std::string unsupported = llama_measure_unsupported_reason(model);
        if (!unsupported.empty()) {
            throw llama_measure_unsupported(unsupported);
        }
    }

    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    if (!measure_only) {
        LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);
    }

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        if (!measure_only) {
            LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                            __func__, cparams.n_rs_seq);
        }
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    embd_layer_inp.resize(hparams.n_layer() + 1);

    cparams.ctx_type          = params.ctx_type;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.pooling_type      = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    if (llama_model_needs_ctx_other(model)) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        cparams.rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            if (!measure_only) {
                LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                        __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
            }
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = false;

    cparams.fused_lid = true;
    cparams.auto_flid = false;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    // (a measure-only context sizes no sampling graph)
    if (!measure_only && params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                if (!measure_only) {
                    LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
                }
            }
        }
    }

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;
    cparams.swa_full   = params.swa_full;

    // initialized later
    cparams.pipeline_parallel = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            if (!measure_only) {
                LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
            }
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            if (!measure_only) {
                LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
            }
        }
    }

    if (!measure_only) {
        LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
        LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
        LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
        LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
        LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
        LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
        LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
        LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
        LLAMA_LOG_INFO("%s: swa_full              = %s\n",   __func__, cparams.swa_full ? "true" : "false");
        LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
        LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
        LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
        LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
        LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);
    }

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        if (!measure_only) {
            LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                    __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
        }
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        if (!measure_only) {
            LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                    __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
        }
    }

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    auto sycl_exec_close_if_idle = [&](ggml_backend_dev_t dev) {
        if (!dev || sycl_exec_context.value == 0) {
            return;
        }
        // Construction unwind: no graph has run, so the context is genuinely
        // idle and close_if_idle is the exact operation. Teardown of a context
        // that ran graphs uses llama_context_sycl_exec_drain_and_close instead.
        auto exec_hooks = llama_context_sycl_exec_procs(dev);
        if (exec_hooks.close_if_idle) {
            (void) exec_hooks.close_if_idle(sycl_exec_context);
        }
        sycl_exec_context = {};
    };

    struct sycl_exec_context_scope {
        decltype(sycl_exec_close_if_idle) & close_fn;
        ggml_backend_dev_t                 dev = nullptr;
        bool                               armed = false;

        ~sycl_exec_context_scope() {
            if (armed) {
                close_fn(dev);
            }
        }
    } sycl_exec_scope{ sycl_exec_close_if_idle, nullptr, false };
#endif

    if (!hparams.vocab_only && measure_only) {
        // The measure backends are the caller's: no device init, no execution context, no threadpool
        // and no output buffer. The CPU backend is the last of them.
        measure_stage = measure->stage;
        backends      = std::move(measure->backends);
        if (backends.empty()) {
            throw std::runtime_error("a measure-only context needs its backends");
        }
        backend_cpu = backends.back().get();
        if (ggml_backend_dev_type(ggml_backend_get_device(backend_cpu)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            throw std::runtime_error("the last backend of a measure-only context must be the CPU backend");
        }
    }

    if (!hparams.vocab_only && !measure_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
        bool sycl_exec_created = false;
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            if (!llama_context_dev_is_sycl(dev)) {
                continue;
            }
            sycl_exec_scope.dev = dev;
            auto exec_hooks = llama_context_sycl_exec_procs(dev);
            if (!exec_hooks.create || !exec_hooks.bind || !exec_hooks.close_if_idle) {
                throw std::runtime_error(format("missing SYCL execution lifecycle procedures for %s backend", ggml_backend_dev_name(dev)));
            }
            auto create_exec = exec_hooks.create;
            auto bind_exec   = exec_hooks.bind;
            if (!create_exec || !bind_exec) {
                continue;
            }
            if (!sycl_exec_created) {
                const auto exec_rc = create_exec(&sycl_exec_context);
                if (exec_rc != GGML_SYCL_EXECUTION_OK) {
                    throw std::runtime_error(format("failed to allocate SYCL execution context: result=%d", (int) exec_rc));
                }
                sycl_exec_created = true;
                sycl_exec_scope.armed = true;
            }
            const auto bind_rc = bind_exec(backend.get(), sycl_exec_context);
            if (bind_rc != GGML_SYCL_EXECUTION_OK) {
                sycl_exec_close_if_idle(dev);
                throw std::runtime_error(format("failed to bind SYCL execution context: result=%d", (int) bind_rc));
            }
            sycl_exec_context_bound = true;
        }
        sycl_resync_runtime_context_flash_attn();
#endif

        // add ACCEL backends (such as BLAS)
        const size_t backend_dev_count = ggml_backend_dev_count();
        for (size_t i = 0; i < backend_dev_count; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            if (!measure_only) {
                LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buffer_name    (buf_output.get()),
                        ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
            }
        }
    }

    // init backends
    // The backend enumeration and the pipeline-parallel decision read the model, cparams and the backends, never
    // the memory module, so they run first: whatever measures the compute buffers during construction needs the
    // bufts and the final pipeline_parallel flag.
    if (!hparams.vocab_only) {
        if (!measure_only) {
            LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);
        }

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

#ifdef GGML_USE_SYCL
        int sycl_gpu_idx = 0;
#endif
        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto * dev = ggml_backend_get_device(backend.get());
            auto backend_type = ggml_backend_dev_type(dev);

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                // (llama.cpp-38af: or its activation twin when the CPU produces activations; re-selected in
                // sched_reserve() once the placement plan is final)
                buft = llama_context_cpu_compute_buft(model, buft, !measure_only);
            }
#ifdef GGML_USE_SYCL
            else if (backend_type == GGML_BACKEND_DEVICE_TYPE_GPU && llama_context_dev_is_sycl(dev)) {
                const int sycl_dev = llama_context_sycl_device_index(dev, sycl_gpu_idx);

                const bool plan_active =
                    llama_context_sycl_hooks_enabled() && ggml_backend_sycl_has_active_placement_plan();

                if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
                    if (plan_active) {
                        // Its alloc_buffer takes a direct host-pinned allocation before any scope
                        // check: an unscoped GPU compute buffer in host memory.
                        throw std::runtime_error(
                            "tensor-split host compute buffer under an active SYCL placement plan");
                    }
                    buft = ggml_backend_sycl_host_compute_buffer_type(sycl_dev);
                    if (!measure_only) {
                        LLAMA_LOG_DEBUG("%s: using SYCL host compute buffer for GPU %d in tensor split mode\n",
                                        __func__, sycl_dev);
                    }
                } else {
                    bool use_host_compute = false;
                    const char * env_host_compute = std::getenv("GGML_SYCL_HOST_COMPUTE");
                    if (env_host_compute != nullptr) {
                        use_host_compute = std::atoi(env_host_compute) != 0;
                    } else {
                        const char * env_cpu_offload = std::getenv("GGML_SYCL_CPU_OFFLOAD");
                        if (env_cpu_offload != nullptr && std::atoi(env_cpu_offload) != 0) {
                            if (!ggml_backend_sycl_cpu_offload_available()) {
                                static bool warned_cpu_offload_unavailable = false;
                                if (!warned_cpu_offload_unavailable) {
                                    if (!measure_only) {
                                        LLAMA_LOG_WARN("%s: GGML_SYCL_CPU_OFFLOAD=1 but no SYCL CPU device is available; "
                                                       "keeping GPU compute buffers device-local\n", __func__);
                                    }
                                    warned_cpu_offload_unavailable = true;
                                }
                            } else {
                                static bool warned_host_compute_opt_in = false;
                                if (!warned_host_compute_opt_in) {
                                    if (!measure_only) {
                                        LLAMA_LOG_INFO("%s: GGML_SYCL_CPU_OFFLOAD=1 active; host-pinned compute buffers "
                                                       "remain opt-in (set GGML_SYCL_HOST_COMPUTE=1 to force)\n", __func__);
                                    }
                                    warned_host_compute_opt_in = true;
                                }
                            }
                        }
                    }
                    if (use_host_compute && plan_active) {
                        // D12: a GPU op's compute buffer in host memory is the forbidden GPU
                        // zero-copy read of host memory, so the device buft stays.
                        static bool warned_host_compute_plan = false;
                        if (!warned_host_compute_plan) {
                            if (!measure_only) {
                                LLAMA_LOG_WARN(
                                    "%s: GGML_SYCL_HOST_COMPUTE=1 is not honoured under an active SYCL "
                                    "placement plan: a GPU op's compute buffer in host memory is a GPU "
                                    "zero-copy read of host memory; keeping the device compute buffer\n",
                                    __func__);
                            }
                            warned_host_compute_plan = true;
                        }
                        use_host_compute = false;
                    }
                    if (use_host_compute) {
                        buft = ggml_backend_sycl_cpu_offload_compute_buffer_type(sycl_dev);
                        if (!measure_only) {
                            LLAMA_LOG_INFO("%s: using SYCL host-pinned compute buffer for GPU %d\n", __func__, sycl_dev);
                        }
                    }
                }

                sycl_gpu_idx++;
            }
#elif defined(GGML_BACKEND_DL)
            else if (backend_type == GGML_BACKEND_DEVICE_TYPE_GPU) {
                const auto hooks = llama_context_sycl_compute_procs(dev);
                if (hooks.host_compute && hooks.cpu_compute && hooks.cpu_available) {
                    const int sycl_dev = hooks.device_index;
                    const bool plan_active = hooks.has_active_plan && hooks.has_active_plan();
                    if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
                        if (plan_active) {
                            throw std::runtime_error(
                                "tensor-split host compute buffer under an active SYCL placement plan");
                        }
                        buft = hooks.host_compute(sycl_dev);
                    } else if (const char * env = std::getenv("GGML_SYCL_HOST_COMPUTE"); env && std::atoi(env) != 0) {
                        if (plan_active) {
                            static bool warned_host_compute_plan = false;
                            if (!warned_host_compute_plan) {
                                if (!measure_only) {
                                    LLAMA_LOG_WARN(
                                        "%s: GGML_SYCL_HOST_COMPUTE=1 is not honoured under an active SYCL "
                                        "placement plan: a GPU op's compute buffer in host memory is a GPU "
                                        "zero-copy read of host memory; keeping the device compute buffer\n",
                                        __func__);
                                }
                                warned_host_compute_plan = true;
                            }
                        } else {
                            buft = hooks.cpu_compute(sycl_dev);
                        }
                    } else if (const char * env = std::getenv("GGML_SYCL_CPU_OFFLOAD"); env && std::atoi(env) != 0) {
                        (void) hooks.cpu_available();
                    }
                }
            }
#endif

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        if (!measure_only) {
            LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());
        }

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

#ifdef GGML_USE_SYCL
        if (pipeline_parallel && llama_context_sycl_hooks_enabled() && ggml_backend_sycl_has_active_placement_plan()) {
            pipeline_parallel = false;
        }
#elif defined(GGML_BACKEND_DL)
        if (pipeline_parallel) {
            for (const auto & backend : backends) {
                const auto hooks = llama_context_sycl_compute_procs(ggml_backend_get_device(backend.get()));
                if (hooks.has_active_plan && hooks.has_active_plan()) {
                    pipeline_parallel = false;
                    break;
                }
            }
        }
#endif

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            if (!measure_only) {
                LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
            }
        }
    }

    // llama.cpp-7gno: the auto n_ubatch trial's decision, and everything the planned ladder's rung set depends on,
    // are made here, before the memory module exists (sycl_auto_ubatch_prepare); the reserve below runs the trial
    // on them.
    bool sycl_auto_ubatch_trial = false;
    if (!hparams.vocab_only && !measure_only) {
        // llama.cpp-xojq (nphx Task 4b, c-wgxn): run the SYCL auto
        // micro-batch selection trial IN PLACE OF the unconditional
        // sched_reserve() below, when all four conditions hold: the caller
        // did not pin -ub explicitly (n_ubatch_auto), this context has a
        // SYCL backend, GGML_SYCL_AUTO_UBATCH allows it, and the model is
        // causal (comment c-dcct: a non-causal model's n_ubatch == n_batch
        // semantics must never be shrunk by the trial). Any condition false
        // falls through to today's single sched_reserve() call, unchanged.
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
        const bool sycl_backend_present     = llama_context_has_sycl_backend(backends);
        bool       sycl_auto_ubatch_enabled = false;
        // The accessor is consulted only once the cheap conditions hold, so an
        // older SYCL DSO's proc lookup does not run for a pinned -ub or a
        // non-causal model.
        if (params.n_ubatch_auto && cparams.causal_attn && sycl_backend_present) {
#    ifdef GGML_USE_SYCL
            sycl_auto_ubatch_enabled = ggml_backend_sycl_auto_ubatch_enabled();
#    else
            ggml_backend_dev_t sycl_dev = nullptr;
            for (auto & backend : backends) {
                ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
                if (llama_context_dev_is_sycl(dev)) {
                    sycl_dev = dev;
                    break;
                }
            }
            auto auto_ubatch_enabled_fn = llama_context_sycl_auto_ubatch_enabled_proc(sycl_dev);
            sycl_auto_ubatch_enabled    = auto_ubatch_enabled_fn && auto_ubatch_enabled_fn();
#    endif
        }
        sycl_auto_ubatch_trial = llama_auto_ubatch_trial_runs(params.n_ubatch_auto, cparams.causal_attn,
                                                              sycl_backend_present, sycl_auto_ubatch_enabled);
#endif
        if (sycl_auto_ubatch_trial) {
            sycl_auto_ubatch_prepare(params.type_k, params.type_v);
        }

        // llama.cpp-7gno: a context under an active SYCL placement plan owns a chunk-cap copy and runs the residency
        // fixpoint before its memory module exists. Production-unreachable until the residency probe is wired
        // (llama_context_l4_ready): every context stays unplanned and takes the legacy publish path. The copy is
        // acquired straight into the member, so a fixpoint refusal frees it while the constructor unwinds.
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
        // The proc lookups are made only once L4 is ready: decide() answers UNPLANNED before it reads them otherwise, so
        // until then a SYCL context pays for none of them.
        const bool                               plan_l4_ready = llama_context_l4_ready(backends);
        const llama_context_sycl_plan_caps_procs plan_procs =
            plan_l4_ready ? llama_context_sycl_plan_caps_procs_for(backends) : llama_context_sycl_plan_caps_procs{};
        const llama_plan_caps_decision plan_decision =
            llama_plan_caps_decide(llama_context_has_sycl_backend(backends), plan_procs.plan_active,
                                   plan_procs.caps_new != nullptr, plan_procs.caps_free != nullptr, plan_l4_ready);
        if (plan_decision == LLAMA_PLAN_CAPS_REFUSE_NO_PROCS) {
            throw std::runtime_error(llama_plan_caps_missing_procs_reason());
        }
        if (plan_decision == LLAMA_PLAN_CAPS_ACQUIRE) {
            plan_caps = llama_plan_caps_ptr(plan_procs.caps_new(), llama_plan_caps_deleter{ plan_procs.caps_free });
            if (!plan_caps) {
                throw std::runtime_error("ggml_backend_sycl_plan_caps_new failed to create the chunk-cap copy");
            }
            sched_residency_fixpoint();
        }
#endif
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };

        memory.reset(model.create_memory(params_mem, cparams, measure_only));
    }

    // the measure-only context's one MEASURE, on a scheduler of its own: no ALLOC, no ladder, no publish
    if (!hparams.vocab_only && measure_only) {
        if (cparams.pipeline_parallel) {
            measure_status = { sched_reserve_status::FAILED,
                               "pipeline parallelism is on for the measured placement (no plan override is active)" };
        } else {
            sched_measure_storage storage(cparams);
            sched_reserve_state   measure_state = storage.state();

            measure_status = sched_reserve_impl(sched_reserve_mode::MEASURE, measure_state);
            if (measure_status.status == sched_reserve_status::OK && !storage.cparams.flash_attn &&
                ggml_is_quantized(params.type_v)) {
                measure_status = { sched_reserve_status::REFUSED,
                                   "quantized V cache was requested, but this requires Flash Attention" };
            }
            measure_plan = std::move(storage.plan);
        }
    }

    // reserve the compute buffers
    if (!hparams.vocab_only && !measure_only) {
        // A pinned -ub publishes its plan once, before the memory module exists: re-read the KV room that epoch is
        // judged with now that the KV cache and the recurrent state do. (The ladder publishes per rung, after them, and
        // begins its own epochs; refreshing the construction-time one first changes nothing for it.)
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
        if (llama_context_has_sycl_backend(backends)) {
            llama_context_sycl_hold_epoch_refresh(backends);
        }
#endif
        if (sycl_auto_ubatch_trial) {
            sycl_select_auto_ubatch();
        } else {
            sched_reserve();
        }
        auto_ubatch_prep.reset();

        // llama.cpp-kpjw: the realized hold-spill check, for the reserve that is final whichever way it was made. The
        // ladder asks it per rung (try_candidate); a pinned -ub, or a ladder that never ran, reserves once with nobody
        // asking, and a compute buffer the planned dense scratch's hold kept out of the RUNTIME zone that then lives
        // outside the arena can leave the card under the driver headroom the arena expects (B50, Qwen, -ub 1024: a
        // 461 MB buffer, flash attention out of resources at the first graph and a hang). Refuse the context here,
        // by name, with the -ub that fits, instead.
        //
        // The gap, not covered here: a lazy re-reserve LATER (sched_need_reserve set by an adapter change, a toggled
        // embeddings or causal mode) makes new compute buffers this check never sees. The hold-induced spill of
        // those is checked by nobody until the next context-init; tracked separately, not handled here.
        //
        // Skipped when the trial just passed the same check for this very sched (sycl_hold_spill_validated_ub): the
        // two readings of the live free memory can differ at the margin, and the ladder's winner must not be
        // overturned by a re-read.
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
        if (llama_context_has_sycl_backend(backends) && sycl_hold_spill_validated_ub != cparams.n_ubatch) {
            uint32_t largest_ub = 0;
            if (!llama_context_sycl_hold_spill_fits(backends, cparams.n_ubatch, &largest_ub)) {
                throw std::runtime_error(format(
                    "compute buffers held out of the SYCL RUNTIME zone for the planned dense scratch spilled outside "
                    "the VRAM arena and left a card under the driver headroom the arena expects (n_ubatch=%u); %s",
                    cparams.n_ubatch,
                    largest_ub != 0 ?
                        format("the largest -ub that fits is about %u, a power of two, estimated by scaling the "
                               "measured compute buffers (or free VRAM on the card)",
                               largest_ub)
                            .c_str() :
                        "no -ub is known to fit: free VRAM on the card"));
            }
        }
#endif

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    if (!measure_only) {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    sycl_exec_scope.armed = false;
#endif
}

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    llama_context(model, params, nullptr) {
}

llama_context::~llama_context() {
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();

    // when training, ggml_opt allocates extra buffers through the scheduler, so the sizes no longer match the expectation
    // (a measure-only context ran no ALLOC and has nothing to compare)
    if (!measure_only && !model.hparams.no_alloc && !opt_ctx) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                if (!measure_only) {
                    LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                        __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
                }
            } else {
                if (!measure_only) {
                    LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                        __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
                }
            }
        }
    }
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    if (!measure_only && sycl_exec_context_bound && sycl_exec_context.value != 0) {
        try {
            synchronize();
        } catch (const std::exception & e) {
            if (!measure_only) {
                LLAMA_LOG_ERROR("%s: failed to synchronize before SYCL execution drain: %s\n", __func__, e.what());
            }
        } catch (...) {
            if (!measure_only) {
                LLAMA_LOG_ERROR("%s: failed to synchronize before SYCL execution drain\n", __func__);
            }
        }
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            if (!llama_context_dev_is_sycl(dev)) {
                continue;
            }
            llama_context_sycl_exec_drain_and_close(__func__, llama_context_sycl_exec_procs(dev), sycl_exec_context);
            break;
        }
    }
#endif
    ggml_opt_free(opt_ctx);
}

// The publish: the FULL runtime-context transaction (KV replan, MoE MMID
// reaccount/materialize, plan republish) on every SYCL backend this context
// has, since this is where n_ctx/n_ubatch are established for the context in
// the first place. It returns what happened instead of throwing, and it never
// waits: a result that is not OK is mapped by value, not retried.
//   - every non-OK result is a REFUSED naming the result code, BUSY included;
//     BUSY is also named a [CONTEXT-PLAN-BUG], since under the replan scope
//     the backend answers a non-ACTIVE module with PLAN_REJECTED instead.
// sycl_resync_runtime_context_flash_attn() below is the throwing form for the
// callers that still run before a reserve of their own.
sched_reserve_result llama_context::sycl_publish_runtime_context(bool flash_attn) {
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    for (auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
#    ifdef GGML_USE_SYCL
        auto runtime_context_fn =
            llama_context_dev_is_sycl(dev) ? &ggml_backend_sycl_set_runtime_context_for_model : nullptr;
#    else
        auto runtime_context_fn = llama_context_sycl_runtime_proc(dev);
#    endif
        if (runtime_context_fn) {
            const auto & owner = model.get_sycl_model_token();
            if (owner.model_id == 0 || owner.load_txn_id == 0) {
                continue;
            }
            const ggml_sycl_model_token token = { owner.model_id, owner.load_txn_id, owner.slot,
                                                  owner.slot_generation };
            // llama.cpp-3aos: cparams.kv_unified threaded the
            // same way cparams.n_seq_max already is -- SYCL's KV planner
            // needs it to pick the right SWA sizing mode (see
            // ggml_backend_sycl_set_runtime_context_for_model()'s
            // declaration comment, ggml-sycl.h). llama.cpp-uajm:
            // cparams.swa_full likewise -- with it set llama_kv_cache_iswa
            // allocates SWA layers at full n_ctx and the planner must too.
            const auto rc = runtime_context_fn(backend.get(), token, cparams.n_ctx, cparams.n_ubatch, cparams.n_seq_max,
                                               cparams.kv_unified, cparams.swa_full, flash_attn);
            // Under the process-global replan scope (L0) the backend's module
            // guard is the single emission site for "the module is not
            // ACTIVE": it logs, aborts under strict and answers PLAN_REJECTED,
            // never BUSY. A BUSY that still reaches this call means the guard
            // drifted, so say so; the REFUSED below carries the result code.
            if (rc == GGML_SYCL_LIFECYCLE_BUSY) {
                LLAMA_LOG_ERROR(
                    "[CONTEXT-PLAN-BUG] publish answered BUSY under the replan scope: module guard drifted\n");
            }
            // Not OK is mapped by value and never waited on. The caller (sched_reserve_nothrow) leaves
            // sched_need_reserve set, so the next decode or encode runs the whole transaction again,
            // this publish included; until one succeeds no graph is computed (decode and encode return -2).
            if (rc != GGML_SYCL_LIFECYCLE_OK) {
                // A plan the transaction refused is a fit verdict on the rung; BUSY, a stale identity or any other
                // lifecycle result is not, and the ladder must not lower -ub for it.
                return { sched_reserve_status::REFUSED,
                         format("failed to activate exact SYCL model plan: result=%d", (int) rc),
                         rc == GGML_SYCL_LIFECYCLE_PLAN_REJECTED };
            }
        }
    }
#endif
    return { sched_reserve_status::OK, "" };
}

// llama.cpp-oyfl: the constructor's own call, right after model activation,
// before any auto flash_attn_type is resolved. See the declaration in
// llama-context.h. resolve_fused_ops() does NOT call this: once an AUTO
// flash_attn_type resolves, only flash_attn_enabled has changed, so it calls
// the narrow sycl_recheck_runtime_context_flash_attn() below instead, rather
// than re-running this whole transaction for no reason.
void llama_context::sycl_resync_runtime_context_flash_attn() {
    const sched_reserve_result result = sycl_publish_runtime_context(cparams.flash_attn);
    if (result.status != sched_reserve_status::OK) {
        throw std::runtime_error(result.reason);
    }
}

// llama.cpp-oyfl: the narrow re-check, called ONLY by resolve_fused_ops()
// once an AUTO flash_attn_type resolves. Calls
// ggml_backend_sycl_recheck_runtime_context_flash_attn() (see its own
// comment for why this is a separate, minimal entry point rather than a
// second call into the full transaction above, and for why it is NOT
// read-only: it takes the same module-admission guard and tensor-inventory
// lock the full transaction does) -- no BUSY retry here: BUSY from this
// call means the module admission guard refused because a shutdown is in
// progress, which a second try would not change.
void llama_context::sycl_recheck_runtime_context_flash_attn() {
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    for (auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
#    ifdef GGML_USE_SYCL
        auto recheck_fn =
            llama_context_dev_is_sycl(dev) ? &ggml_backend_sycl_recheck_runtime_context_flash_attn : nullptr;
#    else
        auto recheck_fn = llama_context_sycl_recheck_proc(dev);
#    endif
        if (recheck_fn) {
            const auto & owner = model.get_sycl_model_token();
            if (owner.model_id == 0 || owner.load_txn_id == 0) {
                continue;
            }
            const ggml_sycl_model_token token = { owner.model_id, owner.load_txn_id, owner.slot,
                                                  owner.slot_generation };
            const auto                  rc    = recheck_fn(backend.get(), token, cparams.flash_attn);
            if (rc != GGML_SYCL_LIFECYCLE_OK) {
                const std::string what = format(
                    "non-FA attention scratch guard re-check failed for the resolved flash-attention "
                    "state: %s",
                    sycl_recheck_lifecycle_result_name(rc));
                // A plan the transaction refused is a fit verdict on the rung; BUSY, a stale identity or any other
                // lifecycle result is not, and the trial must not lower -ub for it.
                if (rc == GGML_SYCL_LIFECYCLE_PLAN_REJECTED) {
                    throw llama_auto_ubatch_fit_refusal(what);
                }
                throw std::runtime_error(what);
            }
        }
    }
#endif
}

void llama_context::resolve_fused_ops(sched_reserve_state &          state,
                                      const llama_memory_context_i * mctx,
                                      uint32_t                       n_seqs) {
    GGML_ASSERT(state.resolution != nullptr);
    fused_resolution & resolution = *state.resolution;

    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled, fused_resolution_entry id) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(state, n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        fused_resolution_entry_data entry;
        entry.present    = true;
        entry.probe_name = probe.name;

        ggml_backend_dev_t cpu_landing_dev = nullptr;
        // the device of the landings the layer's device could not have executed (see
        // llama_fused_cpu_landing_is_placement); their count and layer are entry.n_cpu_gaps / cpu_gap_layer
        ggml_backend_dev_t cpu_gap_dev     = nullptr;
        // the capability query of each layer device, looked up once (a reg-proc lookup in DL builds), not per node
        std::map<ggml_backend_dev_t, llama_fused_capability_fn> capability_procs;

        for (const auto & node : static_cast<llm_graph_result *>(state.gf_res_reserve.get())->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t     backend_fused = ggml_backend_sched_get_tensor_backend(state.sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                // A fused op landing on the CPU backend is not evidence the op is
                // unsupported: CPU is the reference implementation for every fused op.
                // For the SYCL backend's tiered-KV placement this CPU landing is the
                // DESIGNED outcome for a host-demoted layer's attention op (owner ruling:
                // placement decides the executor -- docs/backend/sycl-memory-design.md).
                // Disabling FA globally on that mismatch is strictly worse than leaving it
                // fused: it forces EVERY layer, not just the demoted one, onto the unfused
                // KQ path, whose materialized [n_kv, n_tokens, n_head] intermediate can
                // exceed available memory at a large n_ctx (llama.cpp-7nzm: a 17.3GB SYCL0
                // compute-buffer request at n_ctx=131072 on GPT-OSS 20B traced to exactly
                // this codepath).
                if (!device_fused || ggml_backend_dev_type(device_fused) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                    entry.mismatch           = true;
                    entry.mismatch_layer     = node.il;
                    entry.mismatch_layer_dev = device_layer ? ggml_backend_dev_name(device_layer) : "none";
                    entry.mismatch_probe_dev = device_fused ? ggml_backend_dev_name(device_fused) : "none";
                    break;
                }

                auto capability = capability_procs.find(device_layer);
                if (capability == capability_procs.end()) {
                    capability =
                        capability_procs.emplace(device_layer, llama_context_sycl_capability_proc(device_layer)).first;
                }

                if (llama_fused_cpu_landing_is_placement(device_layer, node.tensor, capability->second)) {
                    entry.n_cpu_landings++;
                    entry.cpu_landing_layer = node.il;
                    cpu_landing_dev         = device_fused;
                } else {
                    entry.n_cpu_gaps++;
                    entry.cpu_gap_layer = node.il;
                    cpu_gap_dev         = device_layer;
                }
            }
        }

        entry.enabled         = !entry.mismatch;
        entry.cpu_landing_dev = cpu_landing_dev ? ggml_backend_dev_name(cpu_landing_dev) : "CPU";
        entry.cpu_gap_dev     = cpu_gap_dev ? ggml_backend_dev_name(cpu_gap_dev) : "none";
        enabled               = entry.enabled;
        resolution.entry[id]  = entry;
    };

    // A group's header is recorded when its group starts, so a throw in the group's first probe
    // still has it.
    auto start_group = [&](fused_resolution_entry id, const char * header) {
        resolution.entry[id].present = true;
        resolution.entry[id].header  = header;
    };

    if (state.cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, state.cparams.flash_attn, FUSED_RESOLUTION_ENTRY_FA);
        state.cparams.auto_fa = false;

        // llama.cpp-oyfl: state.cparams.flash_attn just went from "not yet
        // resolved" (defaulted true for AUTO, see its init above) to its
        // real, hardware-resolved value. The SYCL non-FA attention scratch
        // guard threaded through the constructor's runtime-context call saw
        // the pre-resolution value and would have skipped the check for an
        // AUTO context that just resolved to OFF. Re-check it now that the
        // real value is known -- still inside sched_reserve(), called from
        // the constructor, so a refusal here is still a clean exception,
        // not a mid-prefill abort. Narrow re-check only: n_ctx/n_ubatch
        // have not changed, only flash_attn_enabled has, so this does not
        // need the full runtime-context transaction.
        // A MEASURE publishes nothing, so there is nothing for the re-check to check.
        if (state.measure == nullptr) {
            sycl_recheck_runtime_context_flash_attn();
        }
    }

    if (state.cparams.auto_fgdn) {
        start_group(FUSED_RESOLUTION_ENTRY_GDN_HEADER, "resolving fused Gated Delta Net support:");
        resolve(llm_fused_op_gdn_ar_probe, state.cparams.fused_gdn_ar, FUSED_RESOLUTION_ENTRY_GDN_AR);
        resolve(llm_fused_op_gdn_ch_probe, state.cparams.fused_gdn_ch, FUSED_RESOLUTION_ENTRY_GDN_CH);
        state.cparams.auto_fgdn = false;
    }

    if (state.cparams.auto_flid) {
        start_group(FUSED_RESOLUTION_ENTRY_LID_HEADER, "resolving fused Lightning Indexer support:");
        resolve(llm_fused_op_lid_probe, state.cparams.fused_lid, FUSED_RESOLUTION_ENTRY_LID);
        state.cparams.auto_flid = false;
    }

    if (state.cparams.auto_fhc) {
        start_group(FUSED_RESOLUTION_ENTRY_HC_HEADER, "resolving fused DeepSeek V4 HC support:");
        resolve(llm_fused_op_dsv4_hc_pre_probe, state.cparams.fused_dsv4_hc_pre, FUSED_RESOLUTION_ENTRY_HC_PRE);
        resolve(llm_fused_op_dsv4_hc_comb_probe, state.cparams.fused_dsv4_hc_comb, FUSED_RESOLUTION_ENTRY_HC_COMB);
        resolve(llm_fused_op_dsv4_hc_post_probe, state.cparams.fused_dsv4_hc_post, FUSED_RESOLUTION_ENTRY_HC_POST);
        state.cparams.auto_fhc = false;
    }
}

// llama.cpp-7gno: the trial's hoisted block, run by the constructor before the memory module exists. It makes every
// decision the planned ladder's rung set depends on (the SYCL backends and procs, the cap and its MoE ceiling, the
// tuning-cache lookup and the rung set) once, and stores what sycl_select_auto_ubatch() needs afterwards in
// `auto_ubatch_prep`; the ladder does not look any of it up again. A condition that makes the trial take its single
// reserve (no SYCL backend, a DSO without the trial's entry points, a cap under the first rung, a model with no token)
// leaves `auto_ubatch_prep` empty and does the lookup nowhere, as before. Nothing here prints: the
// `[SYCL-PLAN] tuning cache` WARN and the outcome WARN stay at their sites in sycl_select_auto_ubatch(), whose text
// this move does not touch. `tenant_rung_set` is written here, so the host hold's set exists before the memory
// module.
void llama_context::sycl_auto_ubatch_prepare(ggml_type type_k, ggml_type type_v) {
    auto_ubatch_prep.reset();
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    auto prep = std::make_unique<sycl_auto_ubatch_prep>();

    auto & sycl_backends = prep->backends;
    for (auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (llama_context_dev_is_sycl(dev)) {
            sycl_backends.push_back(
                { backend.get(), llama_context_sycl_device_index(dev, (int) sycl_backends.size()) });
        }
    }
    if (sycl_backends.empty()) {
        // The caller's own gate (llama_context_has_sycl_backend()) already
        // checked this before calling us; defensive only.
        return;
    }

#    ifdef GGML_USE_SYCL
    auto probe_fn      = &ggml_backend_sycl_probe_runtime_context_for_model;
    auto fallback_fn   = &ggml_backend_sycl_compute_buffer_host_fallbacks;
    auto hold_spill_fn = &ggml_backend_sycl_planned_hold_spill_fits;
#    else
    ggml_backend_dev_t first_dev   = ggml_backend_get_device(sycl_backends.front().backend);
    auto               probe_fn    = llama_context_sycl_probe_proc(first_dev);
    auto               fallback_fn = llama_context_sycl_fallbacks_proc(first_dev);
    auto               moe_cap_fn  = llama_context_sycl_moe_gpu_ubatch_max_proc(first_dev);
    auto               hold_spill_fn = llama_context_sycl_hold_spill_proc(first_dev);
    if (!probe_fn || !fallback_fn) {
        // A SYCL DSO too old to export the trial's own entry points --
        // ggml_backend_sycl_auto_ubatch_enabled() should already have kept
        // the caller from reaching here for the same reason; this is the
        // defensive mirror. Fall back to today's single reserve.
        return;
    }
#    endif

    const auto & ladder = llama_auto_ubatch_ladder;

    // A direct GGML_USE_SYCL build's accessor is a real, always-defined
    // function -- called unconditionally, no null check (there is nothing to
    // be null). The DL-without-SYCL lookup genuinely can return nullptr on an
    // older SYCL DSO, so it reports the ceiling unavailable and the cap gets
    // no MoE-specific narrowing. A dense model never consults the ceiling.
#    ifdef GGML_USE_SYCL
    const uint32_t moe_cap           = model.hparams.n_expert > 0 ? ggml_backend_sycl_moe_gpu_ubatch_max() : 0;
    const bool     moe_cap_available = true;
#    else
    const uint32_t moe_cap           = (model.hparams.n_expert > 0 && moe_cap_fn) ? moe_cap_fn() : 0;
    const bool     moe_cap_available = moe_cap_fn != nullptr;
#    endif
    // The MoE ceiling is the binding cap whenever it does not exceed the
    // batch/ctx cap, not only when it strictly narrows it -- a context whose
    // batch/ctx cap already equals moe_cap is bound by the ceiling exactly as
    // much as one where moe_cap is smaller, so "ladder exhausted" would
    // misreport why the ladder stopped. The helper gates on
    // moe_cap_available so a DSO without the accessor never reports a ceiling
    // that was never consulted.
    bool           moe_bound = false;
    const uint32_t cap       = llama_auto_ubatch_cap(cparams.n_batch, cparams.n_ctx, model.hparams.n_expert, moe_cap,
                                                     moe_cap_available, &moe_bound);

    // llama.cpp-xojq (quality round 1 Q2a): the smallest ladder rung does
    // not fit at all -- any -c below 512, or llama-bench's own pp128/tg128
    // rows -- so the trial cannot try a single candidate. Take exactly the
    // pre-trial path (one reserve at whatever n_ubatch the constructor's
    // own clamp already resolved) and log no WARN: a WARN here would report
    // an empty `tried` list against a ladder that never got a chance to
    // run, and the one WARN this function does log is reserved for "the
    // trial actually ran".
    if (cap < ladder[0]) {
        return;
    }

    const auto & owner = model.get_sycl_model_token();
    // llama.cpp-xojq (quality round 1 Q7): the same zero-token guard
    // sycl_resync_runtime_context_flash_attn()/sycl_recheck_runtime_context_
    // flash_attn() apply per backend before building a token -- a model
    // that never had a SYCL token published is not "not the published
    // model" (GGML_SYCL_LIFECYCLE_STALE_IDENTITY); it simply has nothing
    // for the probe to evaluate against. Same no-WARN pre-trial fallback
    // as the cap check above.
    if (owner.model_id == 0 || owner.load_txn_id == 0) {
        return;
    }
    // llama.cpp-7n6n (wires nphx Task 5): the persisted auto n_ubatch cache's
    // four entry points, mirroring probe_fn/fallback_fn/moe_cap_fn's own
    // direct-vs-DL resolution just above.
#    ifdef GGML_USE_SYCL
    auto cache_enabled_fn = &ggml_backend_sycl_ubatch_cache_enabled;
    auto cache_path_fn    = &ggml_backend_sycl_ubatch_cache_path;
    auto cache_lookup_fn  = &ggml_backend_sycl_ubatch_cache_lookup_layout1;
    auto cache_store_fn   = &ggml_backend_sycl_ubatch_cache_store_layout1;
#    else
    auto cache_enabled_fn = llama_context_sycl_ubatch_cache_enabled_proc(first_dev);
    auto cache_path_fn    = llama_context_sycl_ubatch_cache_path_proc(first_dev);
    auto cache_lookup_fn  = llama_context_sycl_ubatch_cache_lookup_proc(first_dev);
    auto cache_store_fn   = llama_context_sycl_ubatch_cache_store_proc(first_dev);
#    endif
    const bool have_cache_accessors = cache_enabled_fn && cache_path_fn && cache_lookup_fn && cache_store_fn;

    // The ORDERED dev_index of every SYCL backend this context has. This is
    // not the whole participating set: a collapsed multi-GPU run has one
    // backend here while the planner also uses a hidden GPU, so the backend
    // extends it (see ggml_sycl_ubatch_cache_key's comment, ggml-sycl.h).
    std::vector<int> & cache_devices = prep->cache_devices;
    cache_devices.reserve(sycl_backends.size());
    for (auto & sb : sycl_backends) {
        cache_devices.push_back(sb.dev_index);
    }

    ggml_sycl_ubatch_cache_key & cache_key = prep->cache_key;
    cache_key.devices    = cache_devices.data();
    cache_key.n_devices  = static_cast<uint32_t>(cache_devices.size());
    cache_key.model_name = model.name.c_str();
    cache_key.model_size = model.size();
    cache_key.model_hash = llama_context_sycl_model_tensor_hash(model);
    cache_key.n_ctx      = cparams.n_ctx;
    cache_key.n_batch    = cparams.n_batch;
    cache_key.flash_attn = cparams.flash_attn;
    // llama.cpp-3aos: two contexts differing only in kv_unified need
    // different auto n_ubatch candidates once KV sizing depends on it
    // (kv_layer_bytes_for_kind(), unified-cache.hpp) -- must not share a
    // cache entry (CACHE_VERSION 3, ggml-sycl.h's struct comment).
    cache_key.kv_unified = cparams.kv_unified;
    // llama.cpp-uajm: swa_full changes every SWA layer's KV bytes (sized as
    // FULL when set), so a CLI run (false) and a raw-API context (true)
    // must not share one cache entry either (CACHE_VERSION 4).
    cache_key.swa_full   = cparams.swa_full;
    cache_key.n_seq_max  = cparams.n_seq_max;
    cache_key.type_k     = static_cast<int32_t>(type_k);
    cache_key.type_v     = static_cast<int32_t>(type_v);

    // llama.cpp-7n6n: the sentinel below is the ONLY
    // arm that can reach the tuning-cache WARN with an empty parenthetical
    // -- ggml_backend_sycl_ubatch_cache_path() is never called when the
    // accessors are unavailable (an old GGML_BACKEND_DL SYCL DSO), and an
    // empty buffer used to print a bare "()" there. The
    // accessor's own return is now checked too -- a call that fails (an
    // out-of-range device, or a path too long for this buffer) leaves
    // cache_path_buf at its ORIGINAL sentinel value, not a partially-written
    // one, since the accessor itself never touches the buffer on failure.
    char (&cache_path_buf)[sizeof(prep->cache_path_buf)] = prep->cache_path_buf;
    std::strncpy(cache_path_buf, "(no cache accessor in this backend build)", sizeof(cache_path_buf) - 1);
    if (have_cache_accessors && !cache_path_fn(cache_devices.front(), cache_path_buf, sizeof(cache_path_buf))) {
        std::strncpy(cache_path_buf, "(cache path unavailable)", sizeof(cache_path_buf) - 1);
        cache_path_buf[sizeof(cache_path_buf) - 1] = '\0';
    }

    // Already clamped to n_batch by the constructor's own `cparams.n_ubatch = std::min(cparams.n_batch, ...)`
    // assignment, above: the floor of the rung set, and the rung the context runs at when nothing climbs.
    const uint32_t fallback_ubatch = cparams.n_ubatch;

    const bool cache_available = have_cache_accessors && cache_enabled_fn();

    // The one tuning-cache lookup. A cached value below fallback_ubatch must
    // not win either (the "never silently shrink" contract applies to a cached
    // hit exactly as much as to a fresh ladder rung), so the value counts only
    // when llama_auto_ubatch_cached_valid accepts it; that same value feeds
    // the rung set, so a cache candidate is always a member of it.
    uint32_t   cached_ubatch         = 0;
    char       cached_reason_buf[64] = { 0 };
    const bool cache_found =
        cache_available && cache_lookup_fn(&cache_key, &cached_ubatch, cached_reason_buf, sizeof(cached_reason_buf));
    const bool     cache_usable = cache_found && llama_auto_ubatch_cached_valid(ladder, llama_auto_ubatch_ladder_size,
                                                                                cached_ubatch, fallback_ubatch, cap);
    const uint32_t cache_set_value = cache_usable ? cached_ubatch : 0;

    // The candidates this trial may try: fallback_ubatch, the ladder rungs in
    // [fallback_ubatch, cap] and the usable cached value, ascending. The loop
    // below iterates the set's ladder members, so it carries no bound checks
    // of its own -- a rung outside [fallback_ubatch, cap] is not in the set.
    uint32_t     rung_set[llama_auto_ubatch_rung_set_capacity];
    uint32_t     rungs[llama_auto_ubatch_rung_set_capacity];
    const size_t n_rung_set =
        llama_auto_ubatch_rung_set(ladder, llama_auto_ubatch_ladder_size, fallback_ubatch, cap, cache_set_value,
                                   rung_set, llama_auto_ubatch_rung_set_capacity);
    const size_t n_rungs = llama_auto_ubatch_ladder_members(rung_set, n_rung_set, ladder, llama_auto_ubatch_ladder_size,
                                                            rungs, llama_auto_ubatch_rung_set_capacity);
    std::vector<uint32_t> rung_ladder(rungs, rungs + n_rungs);
    // The set is the host hold's: R_h is folded over every rung the trial may try, at its first transaction.
    tenant_rung_set.assign(rung_set, rung_set + n_rung_set);

    prep->probe_fn             = probe_fn;
    prep->fallback_fn          = fallback_fn;
    prep->hold_spill_fn        = hold_spill_fn;
    prep->cache_enabled_fn     = cache_enabled_fn;
    prep->cache_store_fn       = cache_store_fn;
    prep->have_cache_accessors = have_cache_accessors;
    prep->cache_available      = cache_available;
    prep->cap                  = cap;
    prep->fallback_ubatch      = fallback_ubatch;
    prep->moe_bound            = moe_bound;
    prep->cached_ubatch        = cached_ubatch;
    std::memcpy(prep->cached_reason_buf, cached_reason_buf, sizeof(prep->cached_reason_buf));
    prep->cache_usable = cache_usable;
    prep->rung_ladder  = rung_ladder;
    auto_ubatch_prep   = std::move(prep);
#else
    GGML_UNUSED(type_k);
    GGML_UNUSED(type_v);
#endif
}

// llama.cpp-xojq (nphx Task 4b, comment c-wgxn): the auto micro-batch
// selection trial. See its declaration in llama-context.h and the
// constructor's own gate right above its one call site for when this runs.
//
// Tries the ladder {512, 1024, 2048, 4096} ascending, each candidate capped
// by min(n_batch, n_ctx) and, for a MoE model (model.hparams.n_expert > 0),
// additionally by the GPU MoE routing ceiling
// (ggml_backend_sycl_moe_gpu_ubatch_max(), llama.cpp-ohkx) until that
// ceiling is lifted -- a candidate whose ring still fits but whose routing
// silently falls off the GPU path is not a win. Per candidate, on every
// SYCL backend this context has: Task 2's non-publishing probe
// (ggml_backend_sycl_probe_runtime_context_for_model) must accept it without
// demoting KV (a BUSY answer is not retried: the candidate loses with
// "transaction busy"); only then is it
// published (sycl_resync_runtime_context_flash_attn(), which every SYCL
// backend's probe already accepted) and given a full sched_reserve() cycle
// -- a fresh sched+galloc every call (sched_reserve()'s own
// sched.reset(ggml_backend_sched_new(...)) destroys the whole scheduler and
// its buffers on every call, so a losing candidate's oversized buffers are
// freed by that reset, not left behind). A candidate whose reserve lands any
// compute buffer host-pinned (ggml_backend_sycl_compute_buffer_host_
// fallbacks() > 0, read AFTER the publish that resets it) also loses. The
// last candidate that clears all three checks wins; if none do, the floor is
// today's pre-trial default (already clamped to n_batch by the
// constructor's own `cparams.n_ubatch = std::min(cparams.n_batch, ...)`
// assignment, which runs before this function is entered). The settle step
// re-publishes and re-reserves at the winner whenever the published
// plan/sched do not already describe it (a losing final candidate's demand
// must not be left live) -- see c-wgxn's own safety analysis for why a
// settle-down transaction cannot introduce a new KV demotion (its demand is
// never larger than one already accepted).
//
// At most FOUR GGML_LOG_WARN lines report the outcome (llama.cpp-7n6n,
// Task 5, wires the persisted auto n_ubatch cache into this trial). The
// first, `[SYCL-PLAN] tuning cache %s: n_ubatch=%u (%s)`, reports the CACHE
// lookup's own outcome (hit/miss/disabled) before the ladder decision below
// it. The second, `[SYCL-PLAN] tuning cache store failed: %s`, is emitted
// only when the cache store below actually runs and returns false --
// not on every start. The third, `[SYCL-PLAN] auto n_ubatch lowered from %u to
// %u: ...` (llama.cpp-kpjw), is emitted only when the result ended up under the
// rung that was asked for: the default lost and the downward continuation found
// a smaller rung that fits, or the settle's publish refused the rung that won
// the ladder and a smaller one fit. The fourth (the
// pre-existing one) is
// `[SYCL-PLAN] auto n_ubatch=...`, with one of these ten stop reasons:
// "ladder exhausted" (no candidate lost -- either the cap stopped the ladder or
// all four rungs were accepted), "MoE GPU routing ceiling" (the MoE cap bound,
// whether it narrowed a larger batch/ctx cap or merely matched it),
// "transaction refused", "transaction busy" (BUSY: the probe is not retried),
// "not the published model" (GGML_SYCL_LIFECYCLE_STALE_IDENTITY -- a second
// model published after this one loaded), "KV would be demoted", "compute
// buffer fell back to host", "hold spill left no headroom" (the rung's own
// compute buffers were kept out of the RUNTIME zone by the planned dense
// scratch's hold, spilled, and left a card under the driver headroom; checked
// after sched_reserve() returns, because only then do the buffers exist),
// "compute buffers did not fit" (the in-loop
// candidate's own sched_reserve() threw -- e.g. its host-pinned retry inside
// graph_reserve() also failed -- caught like the candidate publish;
// non-terminal, so a later start can still resume the ladder above the cached
// rung), or "cached" (a persisted value passed the same per-candidate
// validation a ladder rung uses, so the ladder never ran). Candidate refusals
// inside the probe itself log at GGML_LOG_INFO, not ERROR (Task 2), so a
// multi-candidate trial does not print one scary refusal per losing candidate.
void llama_context::sycl_select_auto_ubatch() {
    sycl_hold_spill_validated_ub = 0;
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    // llama.cpp-7gno: the decisions below the SYCL backend enumeration (the procs, the cap, the cache lookup and the
    // rung set) were made by the constructor's hoisted block (sycl_auto_ubatch_prepare); an empty prep is the
    // single-reserve exit every one of them used to take here.
    if (!auto_ubatch_prep) {
        sched_reserve();
        return;
    }
    const sycl_auto_ubatch_prep & prep = *auto_ubatch_prep;

    const auto & sycl_backends = prep.backends;
    auto         probe_fn      = prep.probe_fn;
    auto         fallback_fn   = prep.fallback_fn;
    auto         hold_spill_fn = prep.hold_spill_fn;

    const auto &   ladder    = llama_auto_ubatch_ladder;
    const uint32_t cap       = prep.cap;
    const bool     moe_bound = prep.moe_bound;
    const char *   stop      = moe_bound ? "MoE GPU routing ceiling" : "ladder exhausted";

    // The zero-token guard (a model that never had a SYCL token published has nothing for the probe to evaluate
    // against) is the hoisted block's: a model without one never reaches here with a prep.
    const auto & owner = model.get_sycl_model_token();
    const ggml_sycl_model_token token = { owner.model_id, owner.load_txn_id, owner.slot, owner.slot_generation };

    // The floor the hoisted block built the rung set over, read back: one source for it, not a second read of
    // cparams.n_ubatch after the residency fixpoint's measures have run.
    const uint32_t fallback_ubatch         = prep.fallback_ubatch;
    uint32_t       last_good               = 0;
    uint32_t       hold_spill_validated_ub = 0;  // the rung try_candidate last passed the realized hold-spill check for
    uint32_t       lowered_from            = 0;  // the default the downward continuation lowered from (0: it did not)
    bool           fallback_tried          = false;  // the default itself was a rung the ladder asked about
    bool           descent_ran             = false;  // the downward continuation ran (and may have found nothing)
    bool           rung_fit_refused        = false;  // the rung try_candidate last lost was a real fit refusal
    uint32_t       lowest_refused          = 0;      // the smallest rung this start asked and lost (0: none yet)
    const char *   last_stop            = nullptr;   // the stop reason of the last rung asked (named errors report it)
    uint32_t       cache_refused_ub     = 0;         // a cached rung that failed its revalidation by a fit refusal
    const char *   cache_refused_reason = nullptr;
    std::string    lowered_cause;                    // why the result is under the rung that was asked for
    bool           sched_matches_last_good = false;
    bool           published_any           = false;
    bool           publish_dirty           = false;  // a candidate publish threw, possibly after landing somewhere
    std::string    tried;

    // A rung that lost: the reason is the last one a named error reports, and the rung lowers the bound the -ub a
    // refusal names is capped under (the advice never names a rung this start already lost).
    auto note_loss = [&](uint32_t c, const char * reason) {
        last_stop      = reason;
        lowest_refused = lowest_refused == 0 ? c : std::min(lowest_refused, c);
    };

    // llama.cpp-kpjw: release the previous rung's compute buffers. They are still alive after a rung (sched_reserve()
    // only replaces them when the next one reserves), and the ones a rung placed in the arena's KV zone sit in exactly
    // the room a runtime-context transaction measures its KV headroom against (ggml_sycl_kv_capacity_live reads the
    // zone's free bytes), while the ones it placed outside the arena are raw bytes the ledger holds against the card.
    // A rung that already lost, or a smaller winner, would depress the next transaction's capacity and refuse it for
    // room it is about to be given back. So BOTH transactions run after this: every rung's (try_candidate) and the
    // settle's republish of the winner, whose F3 must judge the card the realized check judged (the winner's own
    // buffers, not a loser's) or the two verdicts differ on one rung. The sched is re-reserved afterwards whenever
    // this rung does not end up the winner (sched_matches_last_good is false from here until a reserve succeeds), which
    // costs one extra reserve when a rung loses at its probe and nothing otherwise: a rung that reserves replaces the
    // sched anyway, and the settle re-reserves unconditionally.
    auto release_rung_buffers = [&]() {
        synchronize();
        for (auto & res : gf_res_prev) {
            res.reset();
        }
        gf_res_reserve.reset();
        gf_res_prev_active = nullptr;
        sched.reset();
        sched_need_reserve      = true;
        sched_matches_last_good = false;
        hold_spill_validated_ub = 0;
    };

    // llama.cpp-7n6n: ONE per-candidate validator,
    // shared by the cache-hit revalidation below and the ladder loop -- the
    // two call sites used to carry independently-maintained copies of this
    // exact sequence (probe, a BUSY losing the candidate -> publish in a try/catch ->
    // sched_reserve -> host-fallback check), which is also where a bug used
    // to live: the cache copy wrote a losing reason into the SAME `stop`
    // variable the ladder's own vocabulary owns, so a cache miss could leave
    // a stale reason in `stop` for a ladder that went on to finish cleanly.
    // Returning the reason instead of assigning it anywhere fixes both: the
    // ladder call site is the only one that ever writes it into `stop`; the
    // cache call site puts it in its own separate WARN parenthetical
    // (below). Returns nullptr on a full pass (accepted, published,
    // reserved, and no host-pinned fallback); otherwise the stop-reason
    // string that lost. Mutates `cparams.n_ubatch`, `published_any`, and
    // `sched_matches_last_good` exactly as the two pre-refactor copies each
    // did inline, and additionally sets `publish_dirty` when the publish
    // throws -- callers must not assume `cparams.n_ubatch` is unchanged
    // after a losing call, since the probe stage can already have set it
    // via the publish before the host-fallback stage fails.
    auto try_candidate = [&](uint32_t c) -> const char * {
        rung_fit_refused = false;
        release_rung_buffers();

        for (auto & sb : sycl_backends) {
            ggml_sycl_runtime_context_probe probe{};
            // llama.cpp-3aos: cparams.kv_unified -- the probe
            // must be given the SAME kv_unified the candidate would
            // actually publish with, or its accept/reject answer is for the
            // wrong KV shape (ggml-sycl.h, this probe's declaration).
            // llama.cpp-uajm: and the same swa_full, for the same reason.
            const auto rc = probe_fn(sb.backend, token, cparams.n_ctx, c, cparams.n_seq_max, cparams.kv_unified,
                                     cparams.swa_full, cparams.flash_attn, &probe);

            // BUSY is the module-admission refusal, not lock contention, so
            // the probe is not retried: the candidate simply loses.
            if (rc == GGML_SYCL_LIFECYCLE_BUSY) {
                return "transaction busy";
            }
            if (rc == GGML_SYCL_LIFECYCLE_STALE_IDENTITY) {
                return "not the published model";
            }
            if (rc != GGML_SYCL_LIFECYCLE_OK) {
                return "transaction refused";
            }
            // The probe ran and said no: the candidate does not fit (a lifecycle failure above is no such verdict).
            if (!probe.accepted) {
                rung_fit_refused = true;
                return "transaction refused";
            }
            if (probe.would_demote_kv) {
                return "KV would be demoted";
            }
        }

        cparams.n_ubatch = c;
        // llama.cpp-xojq (nphx Task 4b, Task 2 final review addendum): an
        // accepted probe does NOT guarantee this publish succeeds -- the
        // probe's own exit branch returns before the publication-ID check,
        // MMID workspace materialization, and the CAS
        // (ggml_sycl_run_runtime_context_transaction), all of which still
        // run for a real publish and can still refuse (a race against
        // another live update). sycl_resync_runtime_context_flash_attn()
        // throws on that refusal; for a CANDIDATE publish (unlike the
        // settle publish below) that must be treated as an ordinary
        // "transaction refused" stop, not an escaped exception. This
        // publish also runs strictly BEFORE the sched_reserve() call right
        // below, so the narrow flash-attn re-check
        // (sycl_recheck_runtime_context_flash_attn(), invoked from
        // resolve_fused_ops() inside sched_reserve()) evaluates this
        // candidate's own just-published plan, not a stale one. On a
        // caught refusal below: sycl_resync_runtime_context_flash_attn()
        // walks every SYCL backend this context has in order and can
        // throw partway through (e.g. device 0 published this losing
        // candidate before device 1 refused), so the published plan can
        // no longer be trusted to describe last_good on ANY device --
        // marking sched_matches_last_good false forces the settle step
        // below to re-reserve last_good, and setting publish_dirty forces
        // it to re-publish last_good on every device: any publish attempt
        // that may have partially landed forces that republish, whatever
        // candidate was tried last. Nothing new is logged here at WARN or
        // above -- the transaction's own publish path already logged its
        // ERROR for the refusal.
        try {
            sycl_resync_runtime_context_flash_attn();
        } catch (const std::exception &) {
            sched_matches_last_good = false;
            publish_dirty           = true;
            return "transaction refused";
        }
        // llama.cpp-xojq (quality round 1 Q2b): this publish just took
        // effect on every SYCL backend (the try above did not throw), so
        // device state may now differ from fallback_ubatch even if this
        // candidate goes on to lose the host-fallback check below -- the
        // settle step's own publish gate reads this flag to know whether
        // it must correct that state back.
        published_any      = true;
        sched_need_reserve = true;
        // Guarded like the candidate publish right above it: sched_reserve()
        // throws "failed to allocate compute pp/tg buffers" when
        // graph_reserve() fails even after its own host-pinned-retry fallback
        // (ggml-sycl.cpp's ggml_backend_sycl_buffer_type_alloc_buffer path),
        // and can also throw from inside resolve_fused_ops()'s call to
        // sycl_recheck_runtime_context_flash_attn(). A candidate the probe
        // accepted but whose reserve throws must lose like any other candidate,
        // not abort context creation outright -- the pre-trial fallback_ubatch
        // path would otherwise have succeeded. The settle step below still
        // recovers to last_good; the settle's OWN reserve stays unguarded,
        // matching today's behaviour for a context that does not fit at all.
        //
        // sched_reserve()'s own pipeline-parallel fallback (its "retrying
        // without pipeline parallelism" branch below) sets
        // cparams.pipeline_parallel = false PERMANENTLY the moment a reserve
        // needs it, win or lose. Save it here, immediately before the call that
        // can flip it, and restore it on every path where THIS candidate goes
        // on to lose -- so a losing candidate's fallback never leaves pipeline
        // parallelism disabled for last_good, which may never have needed it.
        const bool pipeline_parallel_before_reserve = cparams.pipeline_parallel;
        try {
            sched_reserve();
        } catch (const std::exception & e) {
            // One stop reason covers every throw from this reserve
            // (graph_reserve failure, a flash-attn/non-FA scratch recheck
            // refusal, memory-module init), and the outcome WARN carries
            // only that reason, so name the actual cause here -- at INFO
            // because the loss is recoverable. Swallowing a recheck refusal
            // is safe only because published_any is already true, which
            // forces the settle's full publish.
            LLAMA_LOG_INFO("[SYCL-PLAN] auto n_ubatch candidate %u: compute buffer reserve failed: %s\n", c, e.what());
            cparams.pipeline_parallel = pipeline_parallel_before_reserve;
            sched_matches_last_good   = false;
            // Only a fit verdict (the compute buffers, or the plan the recheck refused) starts or continues the
            // descent; a lifecycle failure or a memory module that would not initialize is no verdict on the rung.
            rung_fit_refused          = dynamic_cast<const llama_auto_ubatch_fit_refusal *>(&e) != nullptr;
            return "compute buffers did not fit";
        }
        sched_matches_last_good = true;

        for (auto & sb : sycl_backends) {
            if (fallback_fn(sb.dev_index) > 0) {
                cparams.pipeline_parallel = pipeline_parallel_before_reserve;
                sched_matches_last_good   = false;
                return "compute buffer fell back to host";
            }
        }
        // llama.cpp-kpjw: every buffer of this rung exists now, so the compute buffers the planned dense scratch's
        // hold kept out of the RUNTIME zone are known exactly. A rung whose spill pushed a card under the driver
        // headroom the arena expects outside itself would exhaust it at its first graph (B50, Qwen PPL at
        // auto-ub1024): it loses here, for every rung, every flash-attention mode and the cached rung, and the
        // ladder lands lower. The recheck inside sched_reserve() cannot do this: it runs after a 1-token probe
        // reserve, before the worst-case reserves, and only for the first rung under auto_fa.
        for (auto & sb : sycl_backends) {
            uint32_t rung_largest_ub = 0;
            if (hold_spill_fn && !hold_spill_fn(sb.backend, c, &rung_largest_ub)) {
                LLAMA_LOG_INFO(
                    "[SYCL-PLAN] auto n_ubatch candidate %u: hold spill left no headroom (the largest -ub that fits is "
                    "about %u)\n",
                    c, rung_largest_ub);
                cparams.pipeline_parallel = pipeline_parallel_before_reserve;
                sched_matches_last_good   = false;
                rung_fit_refused          = true;
                return "hold spill left no headroom";
            }
        }
        // The check ran for every backend only when the hook exists (a backend-DL SYCL library that predates it
        // leaves nothing validated, and the constructor's check, which has no hook either, skips it the same way).
        hold_spill_validated_ub = hold_spill_fn ? c : 0;
        return nullptr;
    };

    // llama.cpp-7n6n (wires nphx Task 5): the persisted auto n_ubatch cache's entry points and key, resolved and
    // built by the hoisted block together with the lookup.
    auto cache_enabled_fn = prep.cache_enabled_fn;
    auto cache_store_fn   = prep.cache_store_fn;

    const bool                         have_cache_accessors = prep.have_cache_accessors;
    const ggml_sycl_ubatch_cache_key & cache_key            = prep.cache_key;
    const char *                       cache_path_buf       = prep.cache_path_buf;

    // Try the cache BEFORE running the ladder. A hit is revalidated through
    // try_candidate() -- the EXACT same per-candidate steps the ladder below
    // uses -- so this is not a shortcut around that validation, only around
    // re-discovering the value from scratch. `stop` is untouched by every
    // arm below: a failed revalidation's reason is reported only in this
    // cache block's own WARN, never left behind for the ladder's outcome
    // line to inherit.
    //
    // llama.cpp-7n6n: a hit's STORED REASON decides
    // what happens next, not just whether it validates. "ladder exhausted"
    // and "MoE GPU routing ceiling" are TERMINAL -- nothing above the
    // cached value was ever going to fit anyway (the ladder ran to the cap,
    // or the MoE ceiling was already the binding constraint), so a hit on
    // either still skips the ladder entirely, exactly as before this
    // finding. Any OTHER stored reason means the ladder previously stopped
    // SHORT of the cap for a reason that may no longer hold (a transient
    // probe/publish/host-fallback loss) -- accepting the cached rung and
    // then RESUMING the ladder from the next rung above it (the
    // `c <= cache_resume_above` skip in the loop below) gives a value that
    // pinned itself low on a bad day a chance to climb back up on a later
    // one, which is what this header's own "at worst one extra
    // revalidation, never a wrong choice" contract (docs/backend/
    // sycl-env-vars.md) actually promises -- a hit alone did not deliver
    // that promise before this fix.
    bool         ladder_needed       = true;
    const bool   cache_available     = prep.cache_available;
    const char * cache_state         = "disabled";
    uint32_t     cache_report_ubatch = 0;
    std::string  cache_paren         = cache_path_buf;
    uint32_t     cache_resume_above  = 0;  // ladder rungs at or below this are skipped (0 = skip none)
    bool         cache_resumed       = false;
    uint32_t     cache_resume_ubatch = 0;  // the validated value the resume started from
    std::string  cache_resume_reason;      // its stored reason, for the "unchanged outcome" compare at the store gate

    // The one tuning-cache lookup was made by the hoisted block; its answer and the ladder rungs of the rung set (the
    // candidates this trial may try, ascending) are read here.
    const uint32_t                cached_ubatch    = prep.cached_ubatch;
    const char *                  cached_reason_buf = prep.cached_reason_buf;
    const bool                    cache_usable     = prep.cache_usable;
    const std::vector<uint32_t> & rung_ladder      = prep.rung_ladder;

    if (cache_available) {
        if (!cache_usable) {
            cache_state = "miss";
        } else {
            // The cached candidate was actually validated here (whether it
            // passed or lost) -- name it in `tried` so the final outcome
            // WARN's "(tried %s; %s)" is never empty for a hit, and so a
            // lost cache candidate that falls through to the ladder is
            // still visible in the trail even though it is not itself a
            // ladder rung.
            tried += (tried.empty() ? "" : ",") + std::to_string(cached_ubatch);

            const char * cache_reason = try_candidate(cached_ubatch);
            if (cache_reason == nullptr) {
                last_good           = cached_ubatch;
                cache_report_ubatch = cached_ubatch;
                cache_state         = "hit";
                const bool terminal = std::strcmp(cached_reason_buf, "ladder exhausted") == 0 ||
                                      std::strcmp(cached_reason_buf, "MoE GPU routing ceiling") == 0;
                if (terminal) {
                    stop          = "cached";
                    ladder_needed = false;
                    cache_paren   = "cached, terminal";
                } else {
                    cache_paren         = "cached, resuming ladder above " + std::to_string(cached_ubatch);
                    cache_resume_above  = cached_ubatch;
                    cache_resumed       = true;
                    cache_resume_ubatch = cached_ubatch;
                    cache_resume_reason = cached_reason_buf;
                    // `stop` stays at its pre-cache value ("ladder
                    // exhausted" or "MoE GPU routing ceiling", set before
                    // this block ran) -- if the resumed ladder finds
                    // nothing better above the cached rung, that value is
                    // exactly the correct reason to report, matching a
                    // normal from-scratch run that reached the same result.
                }
            } else {
                cache_state = "miss";
                cache_paren = "cached " + std::to_string(cached_ubatch) + " refused: " + cache_reason;
                note_loss(cached_ubatch, cache_reason);
                // A fit refusal is a loss this start now knows: the ladder stops AT that rung instead of paying for
                // it again. (A lifecycle anomaly or a race is no verdict on the rung, and the ladder asks it afresh.)
                if (rung_fit_refused) {
                    cache_refused_ub     = cached_ubatch;
                    cache_refused_reason = cache_reason;
                    // The default was asked (as the cached rung), and lost.
                    fallback_tried       = fallback_tried || cached_ubatch == fallback_ubatch;
                }
            }
        }
    }

    // One format string, one call site -- hit/miss/disabled is an argument,
    // not three separately-literal WARN calls that could drift out of sync
    // with each other or with the format string above them.
    LLAMA_LOG_WARN("[SYCL-PLAN] tuning cache %s: n_ubatch=%u (%s)\n", cache_state, cache_report_ubatch,
                   cache_paren.c_str());

    // The general form of the cap < ladder[0] early exit above. The loop below
    // skips every rung under fallback_ubatch and stops at the first rung over
    // cap, so with no rung in [fallback_ubatch, cap] it tries nothing, and the
    // [SYCL-PLAN] auto n_ubatch= WARN would report an empty `tried` list and a
    // stop reason for a ladder that never ran, then persist that reason as a
    // terminal cache entry. Two ordinary shapes reach this with fallback_ubatch
    // well under the largest rung: n_batch=1000 with n_ubatch=600 (512 is under
    // the floor, 1024 over the cap), and a MoE model whose routing ceiling
    // narrows cap below an explicit n_ubatch (nothing clamps fallback_ubatch to
    // that ceiling). Placed AFTER the cache lookup so a persisted value at or
    // above the floor can still be revalidated and reported; gated on
    // tried.empty() so a cache attempt this trial (a hit, or a lost cache
    // candidate) still gets its normal outcome WARN and store logic. Skips only
    // the [SYCL-PLAN] auto n_ubatch= WARN and the cache store: the tuning cache
    // WARN above has already printed its miss/disabled line, unlike the earlier
    // pre-trial exits, which all return before the cache lookup.
    if (tried.empty() &&
        !llama_auto_ubatch_ladder_has_candidate(ladder, llama_auto_ubatch_ladder_size, fallback_ubatch, cap)) {
        sched_reserve();
        return;
    }

    // Never let the ladder pick something SMALLER than the caller's own
    // explicit n_ubatch (fallback_ubatch) -- a raw-API caller can set
    // llama_context_params.n_ubatch above the ladder's first rung together
    // with n_ubatch_auto=true, and the "never silently shrink" gotcha
    // (docs/plans/2026-09-10-auto-ubatch.md) applies to that value too, not
    // only to a rung the trial itself already accepted. rung_ladder holds no
    // rung under fallback_ubatch or over cap, so a rung the ladder never tries
    // can never become last_good; if nothing wins, last_good falls through to
    // fallback_ubatch itself below, which was always safe -- it is exactly
    // today's pre-trial default.
    for (uint32_t c : rung_ladder) {
        if (!ladder_needed) {
            break;
        }
        // llama.cpp-7n6n: a non-terminal cache hit
        // already validated `cache_resume_above` (0 when there was no such
        // hit) -- do not re-try rungs at or below it.
        if (c <= cache_resume_above) {
            continue;
        }
        // The cached rung that failed its revalidation by a fit refusal: the ascending ladder ends at the first loss,
        // and this one is already known.
        if (cache_refused_ub != 0 && c >= cache_refused_ub) {
            stop      = cache_refused_reason;
            last_stop = cache_refused_reason;
            break;
        }
        if (c == fallback_ubatch) {
            fallback_tried = true;
        }
        tried += (tried.empty() ? "" : ",") + std::to_string(c);

        const char * reason = try_candidate(c);
        if (reason != nullptr) {
            stop = reason;
            note_loss(c, reason);
            break;
        }
        last_good = c;
    }

    // A pure race is no shape limit; everything else that ends the ladder is one (this is read again below).
    const bool stop_is_pure_race =
        std::strcmp(stop, "transaction busy") == 0 || std::strcmp(stop, "not the published model") == 0;

    // llama.cpp-kpjw: the default rung itself lost, so nothing at or above it won. When that loss was a real fit
    // refusal (the probe's own verdict, compute buffers that did not fit, the hold spill) the trial must not turn a
    // loadable model into an init failure: a smaller -ub is not a smaller context (n_ctx and the KV placement are
    // unchanged). It continues DOWNWARD (256, 128, 64 from a default of 512) and settles on the first rung that fits,
    // announced loud. A loss that is no fit refusal (a lifecycle failure, a publish that threw or raced, a KV that
    // would be demoted, a host fallback) never lowers -ub, here or at any rung of the walk: the walk ends at the first
    // such loss. Only when the default was itself tried (a default that is not a rung was never asked, so lowering
    // past it would skip the one value that was always safe to reserve). A pinned -ub never reaches this function.
    if (last_good == 0 && ladder_needed && fallback_tried && rung_fit_refused) {
        descent_ran                    = true;
        // A rung below the default is about to be published, so the ring is sized for it whatever happens: the settle
        // republishes the rung it keeps.
        publish_dirty                  = true;
        bool           descent_aborted = false;
        const uint32_t ended           = llama_auto_ubatch_descend(fallback_ubatch, cap, [&](uint32_t c) {
            tried.append(tried.empty() ? "" : ",").append(std::to_string(c));
            const char * rung_reason = try_candidate(c);
            if (rung_reason == nullptr) {
                return true;
            }
            note_loss(c, rung_reason);
            descent_aborted = !rung_fit_refused;
            return descent_aborted;
        });
        if (ended != 0 && !descent_aborted) {
            last_good     = ended;
            lowered_from  = fallback_ubatch;
            lowered_cause = format("the default did not fit (%s)", stop);
        }
    }

    if (last_good == 0) {
        last_good               = fallback_ubatch;
        sched_matches_last_good = false;
    }

    // Settle: the RESERVE must always run here when this trial has not
    // already reserved a sched matching last_good -- a context that skips
    // this entirely (e.g. the very first candidate's probe refused) would
    // otherwise end up with NO compute buffers at all, since this function
    // replaces the constructor's own unconditional sched_reserve() call.
    // The PUBLISH inside it is narrower, and follows one invariant: any
    // candidate publish attempt that may have landed on any device -- one that
    // took effect (published_any) or one that threw partway through the
    // backends (publish_dirty) -- forces a republish of last_good on every
    // device, whatever candidate was tried last. It is skipped only when no
    // such attempt happened -- the plan the constructor's own earlier publish
    // already put in place is then still correct, and republishing the same
    // value would be a redundant runtime-context transaction with no state
    // change. The flags already cover a changed cparams.n_ubatch:
    // try_candidate() writes it only immediately before a publish attempt,
    // which always sets one of them. The helper's own n_ubatch !=
    // fallback_ubatch term is a defensive backstop for a future writer that
    // skips the publish, not a case this call site can reach today.
    //
    // The first rung tried is the smallest rung >= fallback_ubatch (the loop
    // skips rungs under the floor), so it equals fallback_ubatch only when
    // fallback_ubatch is itself a rung. In that case a first-rung loss on the
    // host-fallback check leaves last_good == fallback_ubatch == the rung that
    // just lost (the "last_good == 0" branch above), so this gate still fires
    // (sched_matches_last_good is false) and re-publishes/re-reserves the
    // identical value -- one wasted transaction+reserve cycle, not a wrong one.
    // When fallback_ubatch sits between rungs, that losing rung was published
    // above it, so the same cycle is needed to put fallback_ubatch back. This
    // is deliberately NOT narrowed to also skip whenever cparams.n_ubatch ==
    // last_good: sched_matches_last_good is also set false by a partial
    // multi-device publish failure (a losing candidate's publish can throw
    // after already succeeding on an earlier device -- see try_candidate()'s
    // own comment on that), and in that case cparams.n_ubatch can
    // coincidentally equal last_good while the sched genuinely does NOT
    // describe it consistently across every backend. The value alone cannot
    // distinguish those two cases without also carrying which reason set the
    // flag false, so this gate stays conservative and only skips when
    // sched_matches_last_good is actually true -- the settle-skipped condition
    // must still imply the ring and plan describe the winner.
    //
    // The ring's state after a probe whose rollback failed is ARITHMETIC, not
    // something this gate has to know about structurally. The ascending phase
    // tries only candidates >= fallback_ubatch (the cache lookup rejects a
    // smaller value and the loop skips rungs below it), so a rollback failure
    // there leaves the ring sized for a value >= fallback_ubatch. The downward
    // continuation is the one place a rung BELOW fallback_ubatch is published,
    // and it sets publish_dirty before its first rung, so whatever it leaves
    // (a rung that won, or every rung lost and the ring sized for the smallest
    // one it published) is republished at last_good: the rung it kept, or
    // fallback_ubatch when none did. need_publish is true whenever any candidate
    // publish took effect (published_any) or threw (publish_dirty -- it may
    // have landed on some devices before one refused), and then the settle
    // republishes and re-plans last_good on every device. Otherwise every
    // candidate lost at its probe, before cparams.n_ubatch was ever written, so
    // no candidate won, last_good is fallback_ubatch, cparams.n_ubatch still
    // equals it, and no device plan changed. Without a republish the settle only
    // re-reserves fallback_ubatch: the ring is exact when the failed candidate
    // was fallback_ubatch and oversized, never undersized, otherwise. The
    // settle's own publish gate never inspects the ring directly -- it does not
    // need to.
    // The sched the constructor is left with is the validated one only when the settle below does not re-reserve.
    sycl_hold_spill_validated_ub =
        (sched_matches_last_good && cparams.n_ubatch == last_good) ? hold_spill_validated_ub : 0;
    if (!sched_matches_last_good || cparams.n_ubatch != last_good) {
        const bool need_publish =
            llama_auto_ubatch_settle_needs_publish(published_any, publish_dirty, cparams.n_ubatch, fallback_ubatch);
        cparams.n_ubatch = last_good;
        // The previous rung's compute buffers are released BEFORE the settle's publish, exactly as before every rung's
        // transaction (release_rung_buffers): a loser's buffers still alive here are raw bytes the ledger holds
        // against the card and KV-zone room the transaction measures against, so the settle's F3 would judge a card
        // the realized check, which accepted this rung with them credited back, did not.
        release_rung_buffers();
        // The settle's publish is a fit check of its own (the transaction-time spill bound, against a card the winner's
        // reserve has since been released from). A refusal of it is a refusal of this rung, and the context fails by
        // name; there is no second walk down the ladder from here. The ONE hold-spill fit function (the entry the
        // realized check asks) decides what the refusal is: when it accepts the rung, the settle was refused for some
        // other reason (a race, a model that went away) and the refusal leaves as it came; when it refuses, the error
        // names the -ub that function accepts, capped under every rung this start already lost, and the stop reason of
        // the last rung that was asked.
        std::exception_ptr settle_error;
        std::string        settle_refusal;
        if (need_publish) {
            try {
                sycl_resync_runtime_context_flash_attn();
            } catch (const std::exception & e) {
                settle_error   = std::current_exception();
                settle_refusal = e.what();
            }
        }
        if (settle_error) {
            uint32_t largest_ub = 0;
            if (llama_context_sycl_hold_spill_fits(backends, last_good, &largest_ub)) {
                std::rethrow_exception(settle_error);
            }
            const uint32_t advice = llama_auto_ubatch_advice(largest_ub, lowest_refused);
            throw std::runtime_error(format(
                "auto n_ubatch: %s (tried %s; last stop: %s), and the settle at %u was refused (%s); %s",
                descent_ran ? format("no -ub from %u down to %u fits this context", fallback_ubatch,
                                     llama_auto_ubatch_descent_floor)
                                  .c_str() :
                              format("%u does not fit this context", last_good).c_str(),
                tried.c_str(), last_stop != nullptr ? last_stop : stop, last_good, settle_refusal.c_str(),
                advice != 0 ? format("the largest -ub that fits is about %u, a power of two, estimated by scaling the "
                                     "measured compute buffers (or free VRAM on the card)",
                                     advice)
                                  .c_str() :
                              "no -ub is known to fit: free VRAM on the card"));
        } else {
            sched_need_reserve = true;
            sched_reserve();
        }
    }

    // llama.cpp-7n6n: persist whatever the ladder
    // (fresh or resumed) just chose. NEVER after a cache hit that skipped
    // the ladder entirely (`!ladder_needed`, i.e. a TERMINAL hit) -- the
    // entry it validated is already the one on disk. NEVER for a pure race
    // ("transaction busy"/"not the published model") -- persisting a
    // transient contention outcome as if it were a real shape limit would
    // be exactly the sticky-hit bug this finding fixes, just moved one
    // level up. NEVER when a RESUMED hit's ladder run reproduced the exact
    // same (last_good, reason) it started from -- an identical outcome is
    // not worth a rewrite. Otherwise, store the REAL reason (`stop`), not a
    // fixed "ladder" literal -- this is what lets a future lookup on this
    // entry tell a terminal outcome from a transient one (see the cache
    // block above). The store's own return is now checked -- a false
    // (e.g. an unwritable cache directory) logs exactly one WARN; it is
    // still never fatal to this trial's own outcome either way.
    const bool resumed_outcome_unchanged =
        cache_resumed && last_good == cache_resume_ubatch && cache_resume_reason == stop;
    // A LOWERED result is not worth storing for itself: the lookup refuses any value under the ladder's first rung, so
    // it could only ever be a miss, and the next start runs the ladder again. It IS stored when a cached rung was just
    // refused (llama.cpp-kpjw): the entry that rung came from would otherwise sit there and be paid for on every
    // start, and the lowered value overwrites it as the miss it is. A walk that ran and found nothing stores nothing:
    // last_good is then the default, which did not fit.
    const bool store_outcome = lowered_from == 0 ? !descent_ran : cache_refused_ub != 0;
    if (ladder_needed && !stop_is_pure_race && store_outcome && !resumed_outcome_unchanged && have_cache_accessors &&
        cache_enabled_fn()) {
        if (!cache_store_fn(&cache_key, last_good, stop)) {
            LLAMA_LOG_WARN("[SYCL-PLAN] tuning cache store failed: %s\n", cache_path_buf);
        }
    }

    // One announcement for every way the result ended up under the rung that was asked for (the default lost, or the
    // settle refused a winner).
    if (lowered_from != 0) {
        LLAMA_LOG_WARN(
            "[SYCL-PLAN] auto n_ubatch lowered from %u to %u: %s; a smaller -ub is not a smaller context; "
            "pass -ub N to override\n",
            lowered_from, last_good, lowered_cause.c_str());
    }

    LLAMA_LOG_WARN("[SYCL-PLAN] auto n_ubatch=%u for n_ctx=%u n_batch=%u (tried %s; %s); pass -ub N to override\n",
                   cparams.n_ubatch, cparams.n_ctx, cparams.n_batch, tried.c_str(), stop);
    // llama.cpp-xojq (quality round 1 Q6): names the pre-trial value this
    // line supersedes (the constructor's own "n_ubatch = ..." INFO line
    // above, before this function's one call site), rather than leaving the
    // reader to infer it.
    LLAMA_LOG_INFO("%s: n_ubatch = %u (auto, was %u)\n", __func__, cparams.n_ubatch, fallback_ubatch);
#else
    sched_reserve();
#endif
}

// `log` is false for a measure-only context, which prints nothing
static int llama_graph_n_input_tensors(ggml_cgraph * gf, bool log) {
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            users[node].push_back(node);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = node->src[j];
            if (!src) {
                break;
            }
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                users[src].push_back(node);
            }
        }
    }

    if (!log) {
        return (int) users.size();
    }

    for (const auto & [tensor, nodes] : users) {
        if (tensor->op != GGML_OP_NONE) {
            LLAMA_LOG_WARN("%s: input tensor '%32s' has op %s, expected GGML_OP_NONE\n",
                    __func__, tensor->name, ggml_op_name(tensor->op));
        }
        for (const ggml_tensor * node : nodes) {
            LLAMA_LOG_DEBUG("%s: input tensor '%32s' [%s, ne = { %5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 " }] is used by node '%s' (%s)\n",
                    __func__, tensor->name, ggml_type_name(tensor->type),
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    node->name, ggml_op_name(node->op));
        }
    }

    return (int) users.size();
}

sched_reserve_state llama_context::member_reserve_state() {
    return { sched, gf_res_prev, gf_res_reserve, gf_res_prev_active, n_outputs, n_input_tensors, cparams };
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    const sched_reserve_result result = sched_reserve_transaction();
    if (result.status != sched_reserve_status::OK) {
        if (result.fit_refusal) {
            throw llama_auto_ubatch_fit_refusal(result.reason);
        }
        throw std::runtime_error(result.reason);
    }
}

bool llama_context::sched_reserve_nothrow() {
    if (!sched_need_reserve) {
        return true;
    }

    sched_need_reserve = false;

    try {
        const sched_reserve_result result = sched_reserve_transaction();
        if (result.status == sched_reserve_status::OK) {
            return true;
        }
        LLAMA_LOG_ERROR("%s: %s\n", __func__, result.reason.c_str());
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, err.what());
    } catch (...) {
        LLAMA_LOG_ERROR("%s: unknown exception\n", __func__);
    }

    // The scheduler is no longer reserved for the graphs decode builds, so the
    // next decode or encode reserves again from the start.
    sched_need_reserve = true;
    return false;
}

void llama_context::fused_resolution_report(const fused_resolution & record) {
    if (!measure_only) {
        fused_resolution_emit(record, fused_resolution_printed, "resolve_fused_ops", fused_resolution_sink);
    }
}

void llama_context::fused_resolution_report_lost() noexcept {
    if (!measure_only) {
        LLAMA_LOG_WARN("resolve_fused_ops: resolution lines lost: exception while unwinding\n");
    }
}

// llama.cpp-7gno: the constructor's residency fixpoint (design 2.7), reached only by a context that acquired its
// chunk-cap copy, which needs the residency probe (llama_context_l4_ready). The pure iteration is
// llama_residency_fixpoint (llama-residency-fixpoint.h); what is missing is the backend's tenant-aware probe, which L4
// does not define yet, so this refuses by name instead of publishing a plan nobody checked.
//
// The call that replaces this stub (llama.cpp-hdpd) owes two things the pure iteration cannot do for it. The probe it
// passes returns a llama_residency_probe_answer, and the adapter from ggml_backend_sycl_probe_residency maps OK to OK
// and every other status (GEOMETRY_NOT_WIRED, NOT_ANSWERED, any value it does not know) to a status other than OK,
// never to an all-zero residency. A null proc address (the backend does not export it) maps to NOT_ANSWERED, and
// host_resident is read only when the ggml status is OK. REFUSED and NOT_ANSWERED behave the same in the fixpoint and
// differ only in the reason. And the call throws llama_residency_fixpoint_refusal_text(result) for every result for
// which llama_residency_fixpoint_refused(result.status) holds, so a PROBE_FAILED fixpoint is a named construction
// failure and never an acquired plan (llama.cpp-71hq).
void llama_context::sched_residency_fixpoint() {
    throw std::runtime_error(
        "the SYCL residency fixpoint has no tenant-aware residency probe to run (llama.cpp-7gno, moua L4 step 3)");
}

// The reserve transaction. A planned context measures first, publishes what the measure
// resolved, and only then allocates; any other context allocates, as it always has. The
// measure's resolved fused-op flags reach the context's own cparams only after the publish
// has succeeded, so a refused publish leaves the context as it found it.
sched_reserve_result llama_context::sched_reserve_transaction() {
    if (plan_caps) {
        sched_measure_storage storage(cparams);
        sched_reserve_state   measure_state = storage.state();

        const sched_reserve_result measured = sched_reserve_impl(sched_reserve_mode::MEASURE, measure_state);
        if (measured.status != sched_reserve_status::OK) {
            return measured;
        }

        // the tenant section: the measured compute caps, per tier
        const std::vector<llama_tenant_buft_caps> tenant_caps = measure_tenant_caps(storage.plan);
        std::string                               tenant_reason;
        if (!llama_tenant_section_from_caps(tenant_caps, tenant_section, tenant_reason)) {
            // Not a fit verdict: the builder refuses only a plan inconsistency (a device compute buft with no device
            // index), never a capacity shortfall, so lowering -ub must not hide it. A capacity refusal arrives from the
            // publish below (PLAN_REJECTED) and carries the fit flag there.
            return { sched_reserve_status::REFUSED, tenant_reason };
        }
        // The host tier is held at R_h, the largest any rung of the ladder's set needs, so the section carries it
        // from the first publish on and a rung of the set never grows a host slot.
        if (!tenant_host_hold_ready) {
            tenant_host_hold_measure_and_fold(tenant_section);
        }
        llama_tenant_section_apply_host_hold(tenant_section, tenant_host_hold);

        tenant_key = llama_tenant_key_digest(tenant_section);
        tenant_plan_report(storage.plan, storage.cparams.n_ubatch);

        // the resolution is printed at the publish attempt, whether it commits or is refused
        fused_resolution_report(storage.resolution);

        const sched_reserve_result published = sycl_publish_runtime_context(storage.cparams.flash_attn);
        if (published.status != sched_reserve_status::OK) {
            return published;
        }

        cparams.flash_attn         = storage.cparams.flash_attn;
        cparams.auto_fa            = storage.cparams.auto_fa;
        cparams.fused_gdn_ar       = storage.cparams.fused_gdn_ar;
        cparams.fused_gdn_ch       = storage.cparams.fused_gdn_ch;
        cparams.auto_fgdn          = storage.cparams.auto_fgdn;
        cparams.fused_lid          = storage.cparams.fused_lid;
        cparams.auto_flid          = storage.cparams.auto_flid;
        cparams.fused_dsv4_hc_pre  = storage.cparams.fused_dsv4_hc_pre;
        cparams.fused_dsv4_hc_comb = storage.cparams.fused_dsv4_hc_comb;
        cparams.fused_dsv4_hc_post = storage.cparams.fused_dsv4_hc_post;
        cparams.auto_fhc           = storage.cparams.auto_fhc;
    }

    sched_reserve_state state = member_reserve_state();
    fused_resolution    resolution;
    state.resolution = &resolution;
    return sched_reserve_impl(sched_reserve_mode::ALLOC, state);
}

void llama_context::tenant_host_hold_measure_and_fold(const std::vector<ggml_sycl_context_tenant_desc> & current) {
    llama_tenant_host_hold hold;
    llama_tenant_host_hold_fold(hold, current);

    // the set is ascending and holds the rung the context runs at now; that rung is `current`.
    // A rung that is skipped here and measures fine when the ladder later tries it can carry a host slot above
    // R_h; the publish applies R_h as a maximum (llama_tenant_section_apply_host_hold), so the section it
    // publishes still holds the larger slot and stays correct.
    for (const uint32_t rung : tenant_rung_set) {
        if (rung == cparams.n_ubatch) {
            continue;
        }
        // A rung that cannot be measured is left out of the hold, whether its MEASURE refuses or throws: it must
        // not fail the transaction of the rung the context is running at.
        try {
            sched_measure_storage storage(cparams);
            storage.cparams.n_ubatch  = rung;
            sched_reserve_state state = storage.state();

            const sched_reserve_result measured = sched_reserve_impl(sched_reserve_mode::MEASURE, state);
            if (measured.status != sched_reserve_status::OK) {
                LLAMA_LOG_DEBUG("%s: rung %u left out of the host hold: %s\n", __func__, rung, measured.reason.c_str());
                continue;
            }

            std::vector<ggml_sycl_context_tenant_desc> rung_section;
            std::string                                reason;
            if (!llama_tenant_section_from_caps(measure_tenant_caps(storage.plan), rung_section, reason)) {
                LLAMA_LOG_DEBUG("%s: rung %u left out of the host hold: %s\n", __func__, rung, reason.c_str());
                continue;
            }
            llama_tenant_host_hold_fold(hold, rung_section);
        } catch (const std::exception & err) {
            LLAMA_LOG_WARN("%s: rung %u left out of the host hold: %s\n", __func__, rung, err.what());
        }
    }

    // Not recorded, and not ready, before the loop completes. What is recorded after it is the hold of the rungs
    // that could be measured: a hold missing the skipped rungs.
    tenant_host_hold       = hold;
    tenant_host_hold_ready = true;

    if (!measure_only) {
        LLAMA_LOG_INFO("%s\n", llama_tenant_host_hold_line(tenant_host_hold).c_str());
    }
}

std::vector<llama_tenant_buft_caps> llama_context::measure_tenant_caps(const sched_measure_plan & plan) const {
    std::vector<llama_tenant_buft_caps> out;
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    for (const auto & entry : plan.bufts) {
        for (size_t i = 0, sycl_ordinal = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend_ptrs[i]);
            if (llama_context_dev_is_sycl(dev)) {
                // the device index its name carries; the position among SYCL backends is the fallback
                if (entry.buft == backend_buft[i]) {
                    llama_tenant_buft_caps c;
                    c.device = llama_context_sycl_device_index(dev, (int) sycl_ordinal);
                    c.host   = false;
                    c.cap    = entry.cap;
                    c.max_chunk_size = entry.max_chunk_size;
                    llama_tenant_caps_set_peaks(c, entry.peaks);
                    out.push_back(c);
                    break;
                }
                sycl_ordinal++;
                continue;
            }
            // The CPU backend computes in the host buffer type of the first model device, or in that
            // buffer's activation twin when the CPU produces activations a SYCL op consumes
            // (llama_context_cpu_compute_buft): the same pinned host memory under two identities.
            if (entry.buft == backend_buft[i] && !model.devices.empty() &&
                llama_context_dev_is_sycl(model.devices[0].dev) &&
                (entry.buft == ggml_backend_dev_host_buffer_type(model.devices[0].dev) ||
                 entry.buft == llama_context_sycl_cpu_activation_buft(model.devices[0].dev))) {
                llama_tenant_buft_caps c;
                c.device = -1;
                c.host   = true;
                c.cap    = entry.cap;
                c.max_chunk_size = entry.max_chunk_size;
                llama_tenant_caps_set_peaks(c, entry.peaks);
                out.push_back(c);
                break;
            }
        }
    }
#else
    GGML_UNUSED(plan);
#endif
    return out;
}

void llama_context::tenant_plan_report(const sched_measure_plan & plan, uint32_t n_ubatch) {
    // the devices the section names a compute slot on, in section order
    std::vector<int32_t> devices;
    for (const auto & e : tenant_section) {
        if (e.cohort == GGML_SYCL_CONTEXT_COHORT_COMPUTE &&
            std::find(devices.begin(), devices.end(), e.device) == devices.end()) {
            devices.push_back(e.device);
        }
    }
    for (const int32_t dev : devices) {
        llama_tenant_plan_line_fields f;
        f.ctx_id     = (uint32_t) sycl_exec_context.value;
        f.device     = dev;
        f.n_ubatch   = n_ubatch;
        f.n_measured = plan.n_measured;
        f.measure_ms = plan.measure_ms;
        f.republish  = tenant_republish;
        f.covered    = tenant_covered;
        for (const auto & e : tenant_section) {
            if (e.cohort == GGML_SYCL_CONTEXT_COHORT_COMPUTE && e.device == dev) {
                f.compute_load += e.slot_bytes;
                if (e.slot_index == 0) {
                    f.cap0 = e.slot_bytes;
                }
            }
        }
        f.compute_delta          = (int64_t) f.compute_load - (int64_t) tenant_compute_load[dev];
        tenant_compute_load[dev] = f.compute_load;
        LLAMA_LOG_INFO("%s\n", llama_tenant_plan_line(f).c_str());
    }
}

// llama.cpp-p6i0 (R2 discriminator): the compute trace, one line per compute buffer type and graph, in one
// format for the load-time measure (side=measure) and the allocating reserve (side=reserve), so the measured
// term and the real compute buffer can be compared chunk by chunk.
static const char * llama_compute_trace_kind_name(llama_measure_kind kind) {
    switch (kind) {
        case LLAMA_MEASURE_KIND_PP:
            return "pp";
        case LLAMA_MEASURE_KIND_TG:
            return "tg";
        case LLAMA_MEASURE_KIND_PP_AGAIN:
            return "pp_again";
        case LLAMA_MEASURE_KIND_STREAM:
            return "stream";
        case LLAMA_MEASURE_KIND_SHIFT:
            return "shift";
    }
    return "unknown";
}

static std::string llama_compute_trace_mib_list(const std::vector<size_t> & bytes) {
    std::string out = "[";
    size_t      sum = 0;
    for (size_t c = 0; c < bytes.size(); ++c) {
        out += format("%s%.1f", c == 0 ? "" : ", ", bytes[c] / 1024.0 / 1024.0);
        sum += bytes[c];
    }
    return out + format("] MiB, total %.1f MiB", sum / 1024.0 / 1024.0);
}

static std::string llama_compute_trace_line(const char *                side,
                                            const char *                stage,
                                            const char *                buft,
                                            size_t                      gi,
                                            const llama_measure_graph & g,
                                            int                         n_splits,
                                            const std::vector<size_t> & peaks) {
    return format(
        "[LOAD-PLAN] compute trace side=%s stage=%s buft=%s graph=%zu kind=%s n_tokens=%u n_seqs=%u "
        "n_outputs=%u n_streams=%u splits=%d chunks=%s",
        side, stage, buft, gi, llama_compute_trace_kind_name(g.kind), g.n_tokens, g.n_seqs, g.n_outputs, g.n_streams,
        n_splits, llama_compute_trace_mib_list(peaks).c_str());
}

// The allocating reserve's side: the graph just reserved, read from the scheduler as the measure reads it.
static void llama_compute_trace_reserve(ggml_backend_sched_t                            sched,
                                        const std::vector<ggml_backend_t> &             backends,
                                        const std::vector<ggml_backend_buffer_type_t> & bufts,
                                        size_t                                          gi,
                                        const llama_measure_graph &                     g) {
    const int                               max_chunks = ggml_gallocr_max_chunks();
    std::vector<size_t>                     peak(max_chunks, 0);
    std::vector<ggml_backend_buffer_type_t> seen;
    const int                               n_splits = ggml_backend_sched_get_n_splits(sched);
    for (size_t i = 0; i < backends.size(); ++i) {
        if (std::find(seen.begin(), seen.end(), bufts[i]) != seen.end()) {
            continue;
        }
        seen.push_back(bufts[i]);
        size_t    max_chunk_size = 0;
        const int n_chunks =
            ggml_backend_sched_get_reserved_chunk_peaks(sched, backends[i], peak.data(), max_chunks, &max_chunk_size);
        const std::vector<size_t> peaks(peak.begin(), peak.begin() + std::max(0, std::min(n_chunks, max_chunks)));
        LLAMA_LOG_INFO(
            "%s: %s\n", "sched_reserve",
            llama_compute_trace_line("reserve", "alloc", ggml_backend_buft_name(bufts[i]), gi, g, n_splits, peaks)
                .c_str());
    }
}

sched_reserve_result llama_context::sched_reserve_impl(sched_reserve_mode mode, sched_reserve_state & state) {
    // An exception leaving this reserve (a throw inside resolve_fused_ops, say) still prints what the
    // record holds; a normal return prints nothing here.
    GGML_ASSERT(state.resolution != nullptr);
    fused_resolution_guard_arg    guard_arg = { this, state.resolution };
    fused_resolution_unwind_guard unwind_guard(fused_resolution_report_trampoline, fused_resolution_lost_trampoline,
                                               &guard_arg);

    if (mode == sched_reserve_mode::MEASURE) {
        return sched_measure_impl(state);
    }

    LLAMA_LOG_INFO("%s: reserving ...\n", "sched_reserve");

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs   = state.cparams.n_seq_max;
    const uint32_t n_tokens = std::min(state.cparams.n_ctx, state.cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", "sched_reserve", max_nodes);

    for (auto & res : state.gf_res_prev) {
        res.reset();
    }
    state.gf_res_reserve.reset(new llm_graph_result(max_nodes));
    state.gf_res_prev_active = nullptr;

    // llama.cpp-38af: the CPU compute buft as the plan stands now (see the helper).
    llama_context_cpu_compute_buft_reselect(model, backend_ptrs, backend_buft, true);

    // The scheduler is created inside the scope, so its allocator reads the context's frozen
    // chunk caps. Only a context that owns the copy opens one.
    const llama_context_sycl_plan_procs plan_procs = llama_context_sycl_plan_procs_for(backends);
    llama_plan_scope plan_scope(plan_procs, (uint32_t) sycl_exec_context.value, GGML_SYCL_PLAN_SCOPE_ALLOC,
                                plan_caps.get());
    if (plan_caps && !plan_scope.is_open()) {
        return { sched_reserve_status::FAILED, "the ALLOC plan scope did not open" };
    }

    state.sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes,
                                             state.cparams.pipeline_parallel, state.cparams.op_offload));
#ifdef GGML_USE_SYCL
    llama_context_sycl_attach_sched_plan(state.sched.get(), backends);
#endif

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", "sched_reserve");
        mctx = memory->init_full();
        if (!mctx) {
            return { sched_reserve_status::FAILED, "failed to initialize memory module" };
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", "sched_reserve", n_tokens, n_seqs,
                    n_outputs);

    resolve_fused_ops(state, mctx.get(), n_seqs);
    fused_resolution_report(*state.resolution);

    // reserve worst-case graph
    int n_splits_pp        = -1;
    int n_nodes_pp         = -1;
    int n_inputs_pp        = -1;
    int n_input_tensors_pp = -1;

    int n_splits_tg        = -1;
    int n_nodes_tg         = -1;
    int n_inputs_tg        = -1;
    int n_input_tensors_tg = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, state.cparams.n_outputs_max);
    // The -ub the pp reserves ran at, for a refusal's advice: a context smaller than -ub reserved at n_ctx tokens, and
    // naming a -ub for that shape would be wrong, so it is 0 (no -ub is named).
    const uint32_t refused_ub_pp = n_tokens == state.cparams.n_ubatch ? state.cparams.n_ubatch : 0;

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(state, n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc,
                                  model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (state.cparams.pipeline_parallel) {
                // A planned context never has pipeline parallelism (the constructor turns it
                // off under a plan), so this retry cannot rebuild a scheduler the scope has
                // already handed its caps to.
                GGML_ASSERT(!plan_caps);
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n",
                               "sched_reserve");
                state.cparams.pipeline_parallel = false;
                state.sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                                                         max_nodes, false, state.cparams.op_offload));
#ifdef GGML_USE_SYCL
                llama_context_sycl_attach_sched_plan(state.sched.get(), backends);
#endif
                gf = graph_reserve(state, n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                return { sched_reserve_status::FAILED,
                         "failed to allocate compute pp buffers" +
                             llama_context_sycl_compute_refusal_text(backends, refused_ub_pp),
                         true };
            }
        }

        n_splits_pp        = ggml_backend_sched_get_n_splits(state.sched.get());
        n_nodes_pp         = ggml_graph_n_nodes(gf);
        n_inputs_pp        = static_cast<llm_graph_result *>(state.gf_res_reserve.get())->inputs.size();
        n_input_tensors_pp = state.n_input_tensors;

        llama_measure_graph g;
        g.kind      = LLAMA_MEASURE_KIND_PP;
        g.n_tokens  = n_tokens;
        g.n_seqs    = n_seqs;
        g.n_outputs = n_outputs_pp;
        LLAMA_LOG_INFO("%s: [LOAD-PLAN] compute trace side=reserve stage=alloc n_ctx=%u n_ubatch=%u n_seq_max=%u\n",
                       "sched_reserve", state.cparams.n_ctx, state.cparams.n_ubatch, n_seqs);
        llama_compute_trace_reserve(state.sched.get(), backend_ptrs, backend_buft, 0, g);
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    {
        auto * gf = graph_reserve(state, n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            return { sched_reserve_status::FAILED,
                     "failed to allocate compute tg buffers" +
                         llama_context_sycl_compute_refusal_text(backends, 0),
                     true };
        }

        n_splits_tg        = ggml_backend_sched_get_n_splits(state.sched.get());
        n_nodes_tg         = ggml_graph_n_nodes(gf);
        n_inputs_tg        = static_cast<llm_graph_result *>(state.gf_res_reserve.get())->inputs.size();
        n_input_tensors_tg = state.n_input_tensors;

        llama_measure_graph g;
        g.kind      = LLAMA_MEASURE_KIND_TG;
        g.n_tokens  = n_seqs;
        g.n_seqs    = n_seqs;
        g.n_outputs = n_seqs;
        llama_compute_trace_reserve(state.sched.get(), backend_ptrs, backend_buft, 1, g);
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: the worst case graph is not always reached for `n_seqs > 1`
        //       need to implement a more robust mechanism that tries a few different inputs and analyzes the results
        ggml_cgraph * gf = nullptr;
        switch (model.arch) {
            case LLM_ARCH_KIMI_LINEAR:
            case LLM_ARCH_MINIMAX_01:
                // [TAG_RESERVE_DIAG_DECAY]
                // the `inp_diag_decay` tensor size scales with `n_seq_tokens^2` which
                // makes `n_seqs == 1` use more memory for the compute graph compared to `n_seqs > 1`
                gf = graph_reserve(state, n_tokens, 1, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
                break;
            default:
                gf = graph_reserve(state, n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
        };

        if (!gf) {
            return { sched_reserve_status::FAILED,
                     "failed to allocate compute pp buffers" +
                         llama_context_sycl_compute_refusal_text(backends, refused_ub_pp),
                     true };
        }

        llama_measure_graph g;
        g.kind      = LLAMA_MEASURE_KIND_PP_AGAIN;
        g.n_tokens  = n_tokens;
        g.n_seqs    = (model.arch == LLM_ARCH_KIMI_LINEAR || model.arch == LLM_ARCH_MINIMAX_01) ? 1 : n_seqs;
        g.n_outputs = n_outputs_pp;
        llama_compute_trace_reserve(state.sched.get(), backend_ptrs, backend_buft, 2, g);
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(state.sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", "sched_reserve", ggml_backend_buft_name(buft),
                           backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    {
        const bool diff = n_nodes_pp != n_nodes_tg || n_splits_pp != n_splits_tg ||
                          n_inputs_pp != n_inputs_tg || n_input_tensors_pp != n_input_tensors_tg;

        const auto val = [diff](int v_pp, int v_tg) -> std::string {
            return diff ? format("%d / %d", v_pp, v_tg) : format("%d", v_pp);
        };

        LLAMA_LOG_INFO("%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n",
                       "sched_reserve", diff ? format(" (pp bs=%d, tg bs=%d)", n_tokens, n_seqs).c_str() : "",
                       val(n_nodes_pp, n_nodes_tg).c_str(), val(n_splits_pp, n_splits_tg).c_str(),
                       val(n_inputs_pp, n_inputs_tg).c_str(), val(n_input_tensors_pp, n_input_tensors_tg).c_str());
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n", "sched_reserve", (t_end_us - t_start_us) / 1000.0,
                   ggml_backend_sched_get_n_copies(state.sched.get()));

    // A cap the copy could not answer during this reserve leaves chunks the plan did not size.
    if (const char * failure = plan_scope.failure()) {
        return { sched_reserve_status::REFUSED, failure };
    }

    return { sched_reserve_status::OK, "" };
}

sched_reserve_result llama_context::sched_measure_impl(sched_reserve_state & state) {
    GGML_ASSERT(state.measure != nullptr);
    GGML_ASSERT(!state.cparams.pipeline_parallel);

    if (!plan_caps && !measure_only) {
        return { sched_reserve_status::FAILED, "a measure needs the context's chunk-cap copy" };
    }

    const int64_t t_start_us = ggml_time_us();

    // The scheduler below is created inside the scope, so its allocator reads the frozen caps. A
    // load-time measure holds no copy: its LOAD_MEASURE scope reads the stage's caps, and a measure
    // over no SYCL backend (a CPU-only host) has no scope to open.
    llama_context_cpu_compute_buft_reselect(model, backend_ptrs, backend_buft, !measure_only);

    const llama_context_sycl_plan_procs plan_procs = llama_context_sycl_plan_procs_for(backends);
    llama_plan_scope plan_scope = measure_only ? llama_plan_scope(plan_procs, measure_stage) :
                                                 llama_plan_scope(plan_procs, (uint32_t) sycl_exec_context.value,
                                                                  GGML_SYCL_PLAN_SCOPE_MEASURE, plan_caps.get());
    const bool scope_required = plan_caps || (measure_only && llama_context_has_sycl_backend(backends));
    if (scope_required && !plan_scope.is_open()) {
        return { sched_reserve_status::FAILED, measure_only ? "the LOAD_MEASURE plan scope did not open" :
                                                              "the MEASURE plan scope did not open" };
    }

    const uint32_t n_seqs   = state.cparams.n_seq_max;
    const uint32_t n_tokens = std::min(state.cparams.n_ctx, state.cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    state.gf_res_reserve.reset(new llm_graph_result(max_nodes));
    state.gf_res_prev_active = nullptr;

    state.sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes,
                                             false, state.cparams.op_offload));
#ifdef GGML_USE_SYCL
    llama_context_sycl_attach_sched_plan(state.sched.get(), backends);
#endif

    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
        if (!mctx) {
            return { sched_reserve_status::FAILED, "failed to initialize memory module" };
        }
    }

    resolve_fused_ops(state, mctx.get(), n_seqs);

    llama_measure_set_params params;
    params.n_tokens            = n_tokens;
    params.n_seq_max           = n_seqs;
    params.n_outputs_max       = state.cparams.n_outputs_max;
    params.kv_unified          = state.cparams.kv_unified;
    params.n_layer_nextn       = model.hparams.n_layer_nextn;
    params.warmup              = state.cparams.warmup;
    params.pp_again_single_seq = model.arch == LLM_ARCH_KIMI_LINEAR || model.arch == LLM_ARCH_MINIMAX_01;

    const std::vector<llama_measure_graph> graphs = llama_measure_graph_set(params);

    const int           max_chunks = ggml_gallocr_max_chunks();
    std::vector<size_t> sizes(backend_ptrs.size(), 0);
    std::vector<size_t> peak(max_chunks, 0);

    sched_measure_plan & plan = *state.measure;
    plan.graphs.clear();
    plan.bufts.clear();
    plan.n_splits_max = 0;
    plan.n_splits.clear();

    // Two backends of one buffer type share an allocator and report the same layout, so each buffer type
    // is read once per graph.
    std::string layout_failure;

    auto read_layout = [&](size_t gi) -> bool {
        const int n_splits = ggml_backend_sched_get_n_splits(state.sched.get());
        plan.n_splits_max  = std::max(plan.n_splits_max, n_splits);
        plan.n_splits.push_back(n_splits);

        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            sched_measure_buft * entry = nullptr;
            for (auto & e : plan.bufts) {
                if (e.buft == backend_buft[i]) {
                    entry = &e;
                    break;
                }
            }
            if (entry == nullptr) {
                plan.bufts.emplace_back();
                entry       = &plan.bufts.back();
                entry->buft = backend_buft[i];
            }
            if (entry->peaks.size() > gi) {
                continue;
            }

            size_t    max_chunk_size = 0;
            const int n_chunks       = ggml_backend_sched_get_reserved_chunk_peaks(state.sched.get(), backend_ptrs[i],
                                                                                   peak.data(), max_chunks, &max_chunk_size);
            if (gi > 0 && max_chunk_size != entry->max_chunk_size) {
                layout_failure =
                    format("the chunk size of %s changed between measured graphs", ggml_backend_buft_name(entry->buft));
                return false;
            }
            entry->max_chunk_size = max_chunk_size;
            entry->peaks.emplace_back(peak.begin(), peak.begin() + std::max(0, std::min(n_chunks, max_chunks)));
        }
        return true;
    };

    size_t gi = 0;
    for (; gi < graphs.size(); ++gi) {
        const llama_measure_graph & g = graphs[gi];

        state.cparams.embeddings = g.embeddings;
        state.cparams.warmup     = g.warmup;
        if (model.hparams.n_layer_nextn > 0) {
            state.cparams.embeddings_nextn        = g.nextn;
            state.cparams.embeddings_nextn_masked = g.nextn_masked;
            state.cparams.nextn_layer_offset      = g.nextn_offset;
        }

        // A stream graph is built on a memory context that spans exactly its streams.
        llama_memory_context_ptr       stream_mctx;
        const llama_memory_context_i * graph_mctx = mctx.get();
        if (g.n_streams != 0 && memory) {
            stream_mctx = memory->init_reserve(g.n_streams);
            if (!stream_mctx) {
                return { sched_reserve_status::FAILED,
                         format("failed to initialize the %u-stream reserve memory", g.n_streams) };
            }
            graph_mctx = stream_mctx.get();
        }

        auto * gf = graph_reserve(state, g.n_tokens, g.n_seqs, g.n_outputs, graph_mctx, true, sizes.data());
        if (!gf) {
            return { sched_reserve_status::FAILED, format("failed to measure graph %zu of %zu", gi, graphs.size()) };
        }

        if (!read_layout(gi)) {
            return { sched_reserve_status::FAILED, layout_failure };
        }
    }

    // The K-shift graph of each sub-cache that can shift: update() allocates it on the context's scheduler,
    // so its compute buffer is part of the plan. It is built with the ALLOC reserve's own cparams.
    std::vector<const llama_kv_cache *> shift_caches;
    if (memory) {
        memory->get_shift_caches(shift_caches);
    }

    for (const llama_kv_cache * kv : shift_caches) {
        ggml_cgraph * gf = graph_reserve_shift(state, kv, sizes.data());
        if (!gf) {
            return { sched_reserve_status::FAILED, format("failed to measure the K-shift graph %zu of %zu",
                                                          gi - graphs.size(), shift_caches.size()) };
        }

        if (!read_layout(gi)) {
            return { sched_reserve_status::FAILED, layout_failure };
        }
        gi++;
    }

    // A cap the copy could not answer during this measure leaves peaks nobody sized.
    if (const char * failure = plan_scope.failure()) {
        return { sched_reserve_status::REFUSED, failure };
    }

    for (auto & entry : plan.bufts) {
        std::string reason;
        if (!llama_measure_chunk_plan(entry.peaks, entry.max_chunk_size, (size_t) max_chunks, entry.cap, reason)) {
            return { sched_reserve_status::REFUSED,
                     format("%s (compute buffer type %s)", reason.c_str(), ggml_backend_buft_name(entry.buft)) };
        }
    }

    plan.graphs = graphs;
    for (size_t k = 0; k < shift_caches.size(); ++k) {
        llama_measure_graph shift;
        shift.kind = LLAMA_MEASURE_KIND_SHIFT;
        plan.graphs.push_back(shift);
    }
    plan.n_measured = (uint32_t) (graphs.size() + shift_caches.size());
    plan.measure_ms = (ggml_time_us() - t_start_us) / 1000.0;

    if (!measure_only) {
        LLAMA_LOG_DEBUG("%s: measured %u graphs in %.2f ms\n", __func__, plan.n_measured, plan.measure_ms);
    }

    return { sched_reserve_status::OK, "" };
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

bool llama_context::is_measure_only() const {
    return measure_only;
}

const sched_reserve_result & llama_context::get_measure_status() const {
    return measure_status;
}

const sched_measure_plan & llama_context::get_measure_plan() const {
    return measure_plan;
}

bool llama_context::holds_exec_context() const {
    return sycl_exec_context_bound || sycl_exec_context.value != 0;
}

bool llama_context::holds_output_buffer() const {
    return buf_output != nullptr;
}

std::vector<llama_tenant_buft_caps> llama_context::get_measure_tenant_caps() const {
    return measure_tenant_caps(measure_plan);
}

static decltype(&ggml_backend_sycl_measure_plan_override_install) g_measure_install_override = nullptr;
static decltype(&ggml_backend_sycl_measure_plan_override_clear)   g_measure_clear_override   = nullptr;

llama_measure_override_procs llama_context_sycl_measure_override_procs(ggml_backend_dev_t dev) {
    llama_measure_override_procs procs;
#ifdef LLAMA_PRIVATE_TEST_OBJECTS
    if (g_measure_install_override != nullptr || g_measure_clear_override != nullptr) {
        procs.install = g_measure_install_override;
        procs.clear   = g_measure_clear_override;
        return procs;
    }
#endif
    if (!llama_context_dev_is_sycl(dev)) {
        return procs;
    }
#ifdef GGML_USE_SYCL
    procs.install = &ggml_backend_sycl_measure_plan_override_install;
    procs.clear   = &ggml_backend_sycl_measure_plan_override_clear;
#elif defined(GGML_BACKEND_DL)
    procs.install = reinterpret_cast<decltype(procs.install)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_measure_plan_override_install"));
    procs.clear = reinterpret_cast<decltype(procs.clear)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_measure_plan_override_clear"));
#endif
    return procs;
}

#ifdef LLAMA_PRIVATE_TEST_OBJECTS
void llama_context_sycl_measure_override_procs_override_for_testing(
    decltype(&ggml_backend_sycl_measure_plan_override_install) install_fn,
    decltype(&ggml_backend_sycl_measure_plan_override_clear)   clear_fn) {
    g_measure_install_override = install_fn;
    g_measure_clear_override   = clear_fn;
}
#endif

#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
// The measure backend of one SYCL device: non-owning, no SYCL context (ggml-sycl.h).
static ggml_backend_t llama_context_sycl_measure_backend_init(ggml_backend_dev_t dev, int device) {
#    ifdef GGML_USE_SYCL
    GGML_UNUSED(dev);
    return ggml_backend_sycl_measure_backend_init(device);
#    else
    auto fn = reinterpret_cast<decltype(&ggml_backend_sycl_measure_backend_init)>(
        llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_measure_backend_init"));
    return fn != nullptr ? fn(device) : nullptr;
#    endif
}
#endif

bool llama_model_needs_ctx_other(const llama_model & model) {
    return model.arch == LLM_ARCH_GEMMA4_ASSISTANT ||
           ((model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) &&
            (model.tok_embd == nullptr || model.output == nullptr));
}

std::string llama_measure_unsupported_reason(const llama_model & model) {
    if (llama_model_has_encoder(&model)) {
        return format("%s has an encoder graph the load-time measure does not walk", model.arch_name().c_str());
    }
    if (llama_model_needs_ctx_other(model)) {
        return format("%s needs ctx_other, which a load-time measure has none of", model.arch_name().c_str());
    }
    return "";
}

// The measure's side of the compute trace (llama.cpp-p6i0, R2 discriminator): its shape and KV residency, then
// one line per compute buffer type and graph, then each buffer type's per-chunk peak, which is the term.
static std::vector<std::string> llama_load_measure_trace(const sched_measure_plan &       plan,
                                                         enum ggml_sycl_measure_stage     stage,
                                                         uint32_t                         n_ctx,
                                                         uint32_t                         n_ubatch,
                                                         uint32_t                         n_seq_max,
                                                         const llama_kv_residency_tally & kv) {
    std::vector<std::string> out;
    const char *             stage_name = llama_load_measure_stage_name(stage);
    out.push_back(
        format("[LOAD-PLAN] compute trace side=measure stage=%s n_ctx=%u n_ubatch=%u n_seq_max=%u "
               "graphs=%u splits_max=%d measure_ms=%.2f kv_layers device=%u host=%u cpu=%u",
               stage_name, n_ctx, n_ubatch, n_seq_max, plan.n_measured, plan.n_splits_max, plan.measure_ms, kv.n_device,
               kv.n_host, kv.n_cpu));
    for (const auto & b : plan.bufts) {
        const char * buft_name = ggml_backend_buft_name(b.buft);
        for (size_t gi = 0; gi < b.peaks.size() && gi < plan.graphs.size(); ++gi) {
            const int n_splits = gi < plan.n_splits.size() ? plan.n_splits[gi] : -1;
            out.push_back(
                llama_compute_trace_line("measure", stage_name, buft_name, gi, plan.graphs[gi], n_splits, b.peaks[gi]));
        }
        out.push_back(format("[LOAD-PLAN] compute trace side=measure stage=%s buft=%s peak per chunk %s", stage_name,
                             buft_name, llama_compute_trace_mib_list(llama_measure_peak_per_chunk(b.peaks)).c_str()));
    }
    return out;
}

// Prints a measure's trace. Called by the measure's callers: the measure-only context is quiet while it lives,
// and it is gone once llama_load_measure has returned.
static void llama_load_measure_log_trace(const llama_load_measure_result & measured) {
    for (const std::string & line : measured.trace) {
        LLAMA_LOG_INFO("%s: %s\n", "llama_load_measure", line.c_str());
    }
}

llama_load_measure_result llama_load_measure_run(const llama_model &                  model,
                                                 llama_measure_context_args &         args,
                                                 const llama_measure_override_procs & procs,
                                                 uint32_t                             n_ctx,
                                                 uint64_t                             load_txn,
                                                 enum ggml_sycl_measure_stage         stage,
                                                 int                                  first_device) {
    llama_load_measure_result out;

    if (const std::string why = llama_measure_unsupported_reason(model); !why.empty()) {
        out.unsupported = true;
        out.refusal     = llama_load_measure_refusal_text(stage, first_device, why);
        return out;
    }

    // Declaration order is the unwind order: the override is declared after the holder, so on a normal
    // exit it clears first and the context (which owns the backends once its constructor has taken them
    // from `args`) goes after; on a throw from the constructor the context has already unwound, and the
    // override clears next.
    // The KV residency of the caches the measure context builds (the R2 discriminator's KV line); declared
    // before the holder so the count outlives the context.
    llama_kv_residency_tally       kv_tally;
    llama_kv_residency_tally_scope kv_tally_scope(kv_tally);
    std::unique_ptr<llama_context> holder;
    // The context's params come first: the override re-fits the plan's KV residency for their KV shape.
    const llama_context_params       params   = llama_load_measure_context_params(n_ctx, model.hparams.n_ctx_train);
    const ggml_sycl_measure_kv_shape kv_shape = llama_load_measure_kv_shape(params);
    llama_measure_plan_override      guard(procs, load_txn, stage, &kv_shape);
    if (!guard.installed()) {
        out.refusal = llama_load_measure_refusal_text(stage, first_device, guard.failure());
        return out;
    }

    args.stage = stage;

    try {
        holder.reset(new llama_context(model, params, &args));
    } catch (const llama_measure_unsupported & e) {
        out.unsupported = true;
        out.refusal     = llama_load_measure_refusal_text(stage, first_device, e.what());
        return out;
    } catch (const std::exception & e) {
        out.refusal = llama_load_measure_refusal_text(stage, first_device, e.what());
        return out;
    }

    const sched_reserve_result & status = holder->get_measure_status();
    if (status.status != sched_reserve_status::OK) {
        out.refusal = llama_load_measure_refusal_text(stage, first_device, status.reason);
        return out;
    }

    for (const auto & c : holder->get_measure_tenant_caps()) {
        llama_load_measure_device d;
        d.device      = c.device;
        d.host        = c.host;
        d.chunk_bytes = c.chunk_bytes;
        d.total       = c.total;
        d.cap         = c.max_chunk_size;
        out.devices.push_back(std::move(d));
    }
    out.n_splits = holder->get_measure_plan().n_splits_max;

    out.trace = llama_load_measure_trace(holder->get_measure_plan(), stage, holder->n_ctx(), holder->n_ubatch(),
                                         holder->n_seq_max(), kv_tally);

    out.ok = true;
    return out;
}

llama_load_measure_result llama_load_measure(const llama_model &          model,
                                             uint32_t                     n_ctx,
                                             uint64_t                     load_txn,
                                             enum ggml_sycl_measure_stage stage) {
    llama_load_measure_result out;
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    std::vector<ggml_backend_dev_t> sycl_devs;
    for (const auto & d : model.devices) {
        if (llama_context_dev_is_sycl(d.dev)) {
            sycl_devs.push_back(d.dev);
        }
    }
    if (sycl_devs.empty()) {
        out.ok = true;  // nothing on a SYCL device: no compute-slot term to measure
        return out;
    }
    const int first_device = llama_context_sycl_device_index(sycl_devs[0], 0);

    // a model the measure cannot walk is told so before any backend is created for it
    if (const std::string why = llama_measure_unsupported_reason(model); !why.empty()) {
        out.unsupported = true;
        out.refusal     = llama_load_measure_refusal_text(stage, first_device, why);
        return out;
    }

    llama_measure_context_args args;
    for (size_t i = 0; i < sycl_devs.size(); ++i) {
        const int      device  = llama_context_sycl_device_index(sycl_devs[i], (int) i);
        ggml_backend_t backend = llama_context_sycl_measure_backend_init(sycl_devs[i], device);
        if (backend == nullptr) {
            out.refusal = llama_load_measure_refusal_text(stage, device, "measure backend init failed");
            return out;
        }
        args.backends.emplace_back(backend);
    }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu == nullptr) {
        out.refusal = llama_load_measure_refusal_text(stage, first_device, "no CPU backend");
        return out;
    }
    args.backends.emplace_back(cpu);

    return llama_load_measure_run(model, args, llama_context_sycl_measure_override_procs(sycl_devs[0]), n_ctx, load_txn,
                                  stage, first_device);
#else
    GGML_UNUSED(model);
    GGML_UNUSED(n_ctx);
    GGML_UNUSED(load_txn);
    GGML_UNUSED(stage);
    out.ok = true;
    return out;
#endif
}

llama_load_probe_result llama_load_probe_bound(const llama_model &                            model,
                                               uint32_t                                       n_ctx,
                                               struct ggml_sycl_load_txn                      txn,
                                               const std::vector<llama_measure_dummy_entry> & weights) {
    llama_load_probe_result out;
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    llama_sycl_l4_procs procs;
    for (const auto & d : model.devices) {
        if (llama_context_dev_is_sycl(d.dev)) {
            procs = llama_context_sycl_l4_procs_for_dev(d.dev);
            break;
        }
    }
    if (!procs.available()) {
        return out;
    }

    llama_measure_dummy_scope dummies(weights);
    if (dummies.failed()) {
        out.refusal =
            llama_load_measure_refusal_text(GGML_SYCL_MEASURE_STAGE_PROBE, -1, "a weight stand-in buffer was refused");
        return out;
    }
    const llama_load_measure_result measured = llama_load_measure(model, n_ctx, txn.id, GGML_SYCL_MEASURE_STAGE_PROBE);
    llama_load_measure_log_trace(measured);
    if (!measured.ok) {
        (measured.unsupported ? out.unsupported : out.refusal) = measured.refusal;
        return out;
    }
    out.devices  = measured.devices;
    out.measured = true;
#else
    GGML_UNUSED(model);
    GGML_UNUSED(n_ctx);
    GGML_UNUSED(txn);
    GGML_UNUSED(weights);
#endif
    return out;
}

llama_late_check_result llama_load_late_check(const llama_model &                            model,
                                              uint32_t                                       n_ctx,
                                              struct ggml_sycl_load_txn                      txn,
                                              const std::vector<llama_measure_dummy_entry> & weights) {
    llama_late_check_result out;
#if defined(GGML_USE_SYCL) || defined(GGML_BACKEND_DL)
    llama_sycl_l4_procs procs;
    for (const auto & d : model.devices) {
        if (llama_context_dev_is_sycl(d.dev)) {
            procs = llama_context_sycl_l4_procs_for_dev(d.dev);
            break;
        }
    }
    if (!procs.available()) {
        return out;  // no c(P) can have been recorded without the L4 entry points
    }

    llama_measure_dummy_scope dummies(weights);
    if (dummies.failed()) {
        out.refusal = llama_load_measure_refusal_text(GGML_SYCL_MEASURE_STAGE_CANDIDATE_C, -1,
                                                      "a weight stand-in buffer was refused");
        return out;
    }
    const llama_load_measure_result measured =
        llama_load_measure(model, n_ctx, txn.id, GGML_SYCL_MEASURE_STAGE_CANDIDATE_C);
    llama_load_measure_log_trace(measured);
    if (!measured.ok) {
        (measured.unsupported ? out.unsupported : out.refusal) = measured.refusal;
        return out;
    }

    const uint32_t n_ubatch = llama_load_measure_context_params(n_ctx, model.hparams.n_ctx_train).n_ubatch;
    return llama_late_check_fold(procs, txn, measured.devices, n_ubatch);
#else
    GGML_UNUSED(model);
    GGML_UNUSED(n_ctx);
    GGML_UNUSED(txn);
    GGML_UNUSED(weights);
    return out;
#endif
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

llama_memory_update_result llama_context::memory_update(bool optimize) {
    if (!memory) {
        return LLAMA_MEMORY_UPDATE_NONE;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return LLAMA_MEMORY_UPDATE_NONE;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return LLAMA_MEMORY_UPDATE_NONE;
                }
        }

        // reset the previous graph results to make sure that they won't be reused
        // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_prev_active = nullptr;

        if (!mctx->apply()) {
            // a failed update stays pending (the K-shift is not marked done), and decode must not run over it.
            // The K-shift has already reset the scheduler, and a refused allocation lost its buffers, so the next
            // decode reserves again even if the pending update is dropped before it is retried
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
            sched_need_reserve = true;
            return LLAMA_MEMORY_UPDATE_FAILED;
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    // a failure here (a refusal, or a throw from the memory or the graph build) must not cross decode: the scheduler
    // is no longer reserved for this graph, so the next decode reserves again
    try {
        const auto mctx = memory->init_full();
        if (!mctx) {
            LLAMA_LOG_ERROR("%s: failed to initialize memory context\n", __func__);
            sched_need_reserve = true;
            return LLAMA_MEMORY_UPDATE_FAILED;
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
            sched_need_reserve = true;
            return LLAMA_MEMORY_UPDATE_FAILED;
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update: %s\n", __func__, err.what());
        sched_need_reserve = true;
        return LLAMA_MEMORY_UPDATE_FAILED;
    }

    return LLAMA_MEMORY_UPDATE_DONE;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // Both values are in every measured graph set (llama_measure_graph_set), so a toggle needs no
    // reserve; reserving per toggle would republish per batch on the server.
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;

    // Measured, like set_embeddings: with n_layer_nextn > 0 every variant is in the graph set.
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;

    // Measured: with n_layer_nextn > 0 every offset is in the graph set.
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

void llama_context::set_warmup(bool value) {
    // a measure-only context measures one fixed graph set and never reserves again
    if (measure_only) {
        return;
    }

    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // Fork-local, both directions, only on a change: a planned context measures the warmup graph
    // (n_expert_used := n_expert) while it is on, and the toggle back is reused in place.
    sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    auto * res = get_gf_res_prev();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    if (!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
    } else {
        gf_res_prev_active = nullptr;
        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (!sched_alloc_graph(gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        gf_res_prev_active = res;
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch_ext & batch_inp) {
    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != hparams.n_embd_inp_enc()) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the encoder input %u\n",
                __func__, batch_inp.n_embd, hparams.n_embd_inp_enc());
        return -1;
    }

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always output all tokens and skip position continuity checks (output_all=true)
    if (!balloc->init(batch_inp, model.vocab, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    if (!sched_reserve_nothrow()) {
        LLAMA_LOG_ERROR("%s: failed to reserve the compute buffers\n", __func__);
        return -2;
    }

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch_ext & batch_inp) {
    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != batch_inp.n_embd_inp) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n",
                __func__, batch_inp.n_embd, batch_inp.n_embd_inp);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    // embedding contexts output every token even when no token is explicitly marked as output
    if (has_samplers) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (const auto & tok : batch_inp.tokens) {
            if (!output_all && !tok.output) {
                continue;
            }

            for (auto seq_id : tok.seq_ids) {
                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    output_swaps.clear();

    if (!sched_reserve_nothrow()) {
        LLAMA_LOG_ERROR("%s: failed to reserve the compute buffers\n", __func__);
        return -2;
    }

    // counted only once the reserve succeeded: a -2 here processed nothing
    n_queued_tokens += n_tokens_all;

    bool did_optimize = false;

    // handle any pending shifts/copies
    if (memory_update(false) == LLAMA_MEMORY_UPDATE_FAILED) {
        LLAMA_LOG_ERROR("%s: failed to update the memory\n", __func__);

        return -2;
    }

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        const auto update_res = memory_update(true);
                        if (update_res == LLAMA_MEMORY_UPDATE_FAILED) {
                            LLAMA_LOG_ERROR("%s: failed to update the memory\n", __func__);

                            return -2;
                        }
                        if (update_res == LLAMA_MEMORY_UPDATE_DONE) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
            }
        }

        if (has_samplers) {
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
            copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched.get(), &sampling.logits_count);
            copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched.get(), &sampling.probs_count);
            copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = true;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

void llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
    }
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0*n_embd_out + k], embd.data[i1*n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        if (embd_layer_inp.size() > 0) {
            for (int lid = 0; lid < (int) embd_layer_inp.size(); ++lid) {
                if (embd_layer_inp[lid].size > 0) {
                    for (uint64_t k = 0; k < n_embd; ++k) {
                        std::swap(embd_layer_inp[lid].data[i0*n_embd + k], embd_layer_inp[lid].data[i1*n_embd + k]);
                    }
                }
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    uint32_t res;
    if (model.arch == LLM_ARCH_KIMI_K3) {
        // the n_tokens*40 budget below is exhausted at ubatch 3840
        res = std::max<uint32_t>(n_tokens * 160, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_HRM_TEXT) {
        // the 128-slot looped graph needs roughly one stack per token budget
        res = std::max<uint32_t>(n_tokens * 80, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        model.arch == LLM_ARCH_HY_V4) {
        res = std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*model.n_tensors());
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    uint32_t n_sampling_nodes = 0;
    uint32_t n_sampling_nodes_max = 0;
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        const uint32_t n_nodes = llama_sampler_backend_n_nodes(sampler);
        n_sampling_nodes += n_nodes;
        if (cparams.n_outputs_max_per_seq > 1) {
            n_sampling_nodes_max = std::max(n_sampling_nodes_max, n_nodes);
        }
    }

    const uint32_t n_sampling_outputs_max = std::min<uint64_t>(
            std::min(n_tokens, cparams.n_outputs_max),
            (uint64_t) cparams.n_seq_max * cparams.n_outputs_max_per_seq);

    res += n_sampling_nodes;
    if (n_sampling_outputs_max > 1) {
        res += (n_sampling_outputs_max - 1) * n_sampling_nodes_max;
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

llm_graph_result * llama_context::get_gf_res_prev() {
    auto & res = gf_res_prev[n_outputs > 0];
    if (!res) {
        res.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
    }
    return res.get();
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    sched_reserve_state state = member_reserve_state();
    return graph_reserve(state, n_tokens, n_seqs, n_outputs, mctx, split_only, sizes);
}

ggml_cgraph * llama_context::graph_reserve_shift(sched_reserve_state &  state,
                                                 const llama_kv_cache * kv,
                                                 size_t *               sizes) {
    GGML_ASSERT(kv != nullptr);

    ggml_backend_sched_reset(state.sched.get());

    // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
    for (auto & res : state.gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    state.gf_res_prev_active = nullptr;

    auto * res = state.gf_res_reserve.get();

    res->reset();

    auto * gf = kv->build_graph_shift(res, this);

    GGML_ASSERT(sizes != nullptr);
    ggml_backend_sched_reserve_size(state.sched.get(), gf, sizes);

    return gf;
}

ggml_cgraph * llama_context::graph_reserve(sched_reserve_state &          state,
                                           uint32_t                       n_tokens,
                                           uint32_t                       n_seqs,
                                           uint32_t                       n_outputs,
                                           const llama_memory_context_i * mctx,
                                           bool                           split_only,
                                           size_t *                       sizes) {
    if (!measure_only) {
        LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    }
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        if (!measure_only) {
            LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
        }
    }

    ggml_backend_sched_reset(state.sched.get());

    // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
    for (auto & res : state.gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    state.gf_res_prev_active = nullptr;

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = state.n_outputs;

    state.n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, state.cparams.n_outputs_max_per_seq);

    auto * res = state.gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(state.cparams.ctx_type),
                                      state.sched.get(), state.cparams, state.n_outputs);

    res->reset();

    auto * gf = model.build_graph(gparams);

    state.n_input_tensors = llama_graph_n_input_tensors(gf, !measure_only);
    state.n_outputs       = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(state.sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(state.sched.get(), gf);
        }
    } else {
        // A MEASURE reserves on its own scheduler, which no plan scope or hold record covers; the context's own
        // scheduler goes through sched_reserve_graph, so the compute scope is open while its buffers are placed.
        const bool reserved = state.measure ? ggml_backend_sched_reserve(state.sched.get(), gf) : sched_reserve_graph(gf);
        if (!reserved) {
            GGML_ASSERT(!sizes);
            if (!measure_only) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
            }
            return nullptr;
        }
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return graph_params(res, ubatch, mctx, gtype, sched.get(), cparams, n_outputs);
}

llm_graph_params llama_context::graph_params(llm_graph_result *             res,
                                             const llama_ubatch &           ubatch,
                                             const llama_memory_context_i * mctx,
                                             llm_graph_type                 gtype,
                                             ggml_backend_sched_t           sched_arg,
                                             const llama_cparams &          cparams_arg,
                                             uint32_t                       n_outputs_arg) const {
    return {
        /*.arch        =*/model.arch,
        /*.hparams     =*/model.hparams,
        /*.cparams     =*/cparams_arg,
        /*.ubatch      =*/ubatch,
        /*.gtype       =*/gtype,
        /*.sched       =*/sched_arg,
        /*.backend_cpu =*/backend_cpu,
        /*.cvec        =*/cvec.get(),
        /*.loras       =*/loras.get(),
        /*.mctx        =*/mctx,
        /*.cross       =*/&cross,
        /*.prec_policy =*/&model.prec_policy,
        /*.samplers    =*/sampling.samplers,
        /*.n_outputs   =*/n_outputs_arg,
        /*.cb          =*/graph_get_cb(sched_arg),
        /*.res         =*/res,
    };
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb(ggml_backend_sched_t sched_arg) const {
    return [this, sched_arg](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched_arg, cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void discard() override {
        rinfos.clear();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            const int64_t n = rinfo.size/ggml_element_size(rinfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            if (mbuf_cur.n_tensors == mbuf.n_tensors) {
                // an equal tensor count does not imply the same chunking, e.g. save ranges [2,1] vs restore runs [1,2]
                bool same_chunking = true;
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    if (ggml_nbytes(mbuf_cur.cpy[i]) != ggml_nbytes(mbuf.org[i])) {
                        same_chunking = false;
                        break;
                    }
                }

                if (same_chunking) {
                    // same chunking: copy 1:1 by index
                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
                    }
                    continue;
                }
            }

            // different chunking: copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org)
            // with a byte cursor. Write and read enumerate the same logical data in the same order but may chunk
            // it differently (even with an equal number of tensors), so copy across tensor boundaries rather than
            // 1:1 by index.
            const size_t total = mbuf_cur.total_size;

            ggml_init_params params_scratch = {
                /*.mem_size   =*/ 2*(mbuf_cur.cpy.size() + mbuf.org.size())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_scratch = ggml_init(params_scratch);

            size_t src_pos  = 0;
            size_t dst_pos  = 0;
            size_t src_j    = 0;
            size_t dst_i    = 0;
            size_t src_base = 0;
            size_t dst_base = 0;

            while (src_pos < total) {
                const auto & src_t = mbuf_cur.cpy[src_j];
                const auto & dst_t = mbuf.org[dst_i];

                const size_t src_size = ggml_nbytes(src_t);
                const size_t dst_size = ggml_nbytes(dst_t);

                const size_t src_off  = src_pos - src_base;
                const size_t dst_off  = dst_pos - dst_base;

                const size_t n_copy = std::min(src_size - src_off, dst_size - dst_off);

                const size_t   el   = ggml_element_size(src_t);
                const int64_t n_el = (int64_t) (n_copy / el);

                auto * src_v = ggml_view_1d(ctx_scratch, src_t, n_el, src_off);
                ggml_backend_view_init(src_v);
                auto * dst_v = ggml_view_1d(ctx_scratch, dst_t, n_el, dst_off);
                ggml_backend_view_init(dst_v);

                ggml_backend_tensor_copy(src_v, dst_v);

                src_pos += n_copy;
                dst_pos += n_copy;

                if (src_pos - src_base == src_size) {
                    src_base = src_pos;
                    ++src_j;
                }
                if (dst_pos - dst_base == dst_size) {
                    dst_base = dst_pos;
                    ++dst_i;
                }
            }

            GGML_ASSERT(src_pos == total && dst_pos == total);
            // any tensors left unvisited hold no data
            for (size_t i = src_j; i < mbuf_cur.cpy.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf_cur.cpy[i]) == 0);
            }
            for (size_t i = dst_i; i < mbuf.org.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf.org[i]) == 0);
            }

            ggml_free(ctx_scratch);
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    void discard() override {
        rinfos.clear();
        buf_size = 0;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io.discard();
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io->discard();
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (tokens_out == nullptr) {
            const size_t n_token_max = (file.size() - file.tell()) / sizeof(llama_token);
            if (n_token_count > n_token_max) {
                LLAMA_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__, n_token_count, n_token_max);
                return 0;
            }

            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    if (plan_caps) {
        GGML_ABORT(
            "training is not supported while a SYCL placement plan is active: ggml-opt would reallocate this "
            "context's planned scheduler outside its claim scope; llama.cpp-q64b");
    }

    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    if (cparams.flash_attn) {
        LLAMA_LOG_INFO("%s: disabling flash attention, FLASH_ATTN_EXT has no backward pass\n", __func__);
        cparams.flash_attn = false;

        // the graph changes without flash attention, need to reserve again
        sched_need_reserve = true;
        sched_reserve();
    }

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        // TODO: use llama_batch_ext here
        {
            llama_batch_compat compat(this, batch);
            if (!balloc->init(*compat.batch_ext, model.vocab, true)) {
                LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
                return;
            }
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = get_gf_res_prev();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            // the optimizer graph is allocated outside sched, so the next decode must rebuild
            gf_res_prev_active = nullptr;
            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.n_ubatch_auto               =*/ false, // fork-local (llama.cpp-nphx); see include/llama.h
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (model->get_split_state_ud.n_devices == 1) {
            LLAMA_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
        }
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        const auto & cparams = ctx->get_cparams();

        if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN && cparams.rope_freq_scale != model->hparams.rope_freq_scale_train) {
            LLAMA_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__, model->hparams.n_ctx_train);
            model->hparams.n_ctx_train = cparams.n_ctx_orig_yarn / cparams.rope_freq_scale;
            LLAMA_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, model->hparams.n_ctx_train);
        }

        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

// compat: llama_batch -> llama_batch_ext -> encode/decode

int llama_context::encode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp, model.hparams.n_embd_inp_enc());
    return encode(*compat.batch_ext);
}

int llama_context::decode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp);
    return decode(*compat.batch_ext);
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    switch (type) {
        case LLAMA_PROCESS_TYPE_ENCODE: return ctx->encode(*batch);
        case LLAMA_PROCESS_TYPE_DECODE: return ctx->decode(*batch);
    }
    return -1;
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
