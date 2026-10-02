// C7h-2 (zhcn): the measure-only llama_context. CPU only: tiny synthetic models on no device.
//
// A load-time measure builds a transient llama_context through the measure-only constructor, runs one
// MEASURE on a scheduler of its own and reads the plan it leaves. This test builds that context over a
// CPU backend and checks what the design promises of it:
//
//   - the MEASURE ran: status OK, graphs measured, a chunk plan per compute buffer type, a split count;
//   - the plan is the real context's: its worst-case chunk peak equals the compute buffer the real
//     context's own reserve allocated for the same parameters;
//   - the context leaves no buffer behind: ggml-backend's live-buffer count is the same before and after
//     (that its memory is built no_alloc is test-layer-shapes' check, and the constructor's call is pinned by
//     the log-silence source gate);
//   - it prints nothing the constructor, the destructor or the reserve print for a real context: a log
//     capture over its whole life holds no line of any of their prefixes, and the same capture over a real
//     context holds many (the positive control that the capture sees them);
//   - the real cparams are not written: a context's own flags are what the caller passed;
//   - a model the measure cannot walk is refused by name with llama_measure_unsupported, for every memory
//     kind that has no no_alloc form, for a model with an encoder graph and for a ctx_other arch; the load-time
//     measure turns that into `unsupported`, not a failure, and clears its plan override on that path too;
//   - the plan's K-shift graphs are the memory's shift sub-caches, the compute term is the per-chunk peak
//     over the measured graphs (the real compute buffer is its sum), set_warmup is a no-op on a measure-only
//     context, and the memory modules' own constructor logs are silent under no_alloc.

#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-context.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-layer-shapes.h"
#include "../src/llama-load-measure.h"
#include "../src/llama-measure-plan.h"
#include "../src/llama-model.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "test-tiny-model.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <set>
#include <string>
#include <vector>

static int n_failed       = 0;
static int n_peak_checked = 0;
static int n_shift_checked = 0;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            n_failed++;                                                       \
            fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
            fprintf(stderr, __VA_ARGS__);                                     \
            fprintf(stderr, "\n");                                            \
        }                                                                     \
    } while (0)

// the lines the measure-only context must not print: the prefixes of the shared constructor body,
// the destructor, the reserve and the graph build
static const char * const k_silent_prefixes[] = {
    "llama_context:",
    "~llama_context:",
    "sched_reserve:",
    "sched_reserve_impl:",
    "sched_measure_impl:",
    "resolve_fused_ops:",
    "graph_reserve:",
    "llama_graph_n_input_tensors:",
    // the memory modules' constructors, which log buffer and cache sizes
    "llama_kv_cache:",
    "llama_kv_cache_iswa:",
    "llama_memory_recurrent:",
};

// the lines the memory modules print for a real context: the positive control that the capture sees them
static const char * const k_memory_prefixes[] = { "llama_kv_cache:", "llama_kv_cache_iswa:",
                                                  "llama_memory_recurrent:" };

struct log_capture {
    std::vector<std::string> lines;
    ggml_log_callback        old_cb   = nullptr;
    void *                   old_data = nullptr;

    static void callback(ggml_log_level, const char * text, void * user_data) {
        static_cast<log_capture *>(user_data)->lines.emplace_back(text);
    }

    void start() {
        lines.clear();
        llama_log_get(&old_cb, &old_data);
        llama_log_set(callback, this);
    }

    void stop() { llama_log_set(old_cb, old_data); }

    size_t n_memory_lines() const {
        size_t n = 0;
        for (const auto & line : lines) {
            for (const char * prefix : k_memory_prefixes) {
                if (line.compare(0, strlen(prefix), prefix) == 0) {
                    n++;
                    break;
                }
            }
        }
        return n;
    }

    size_t n_silent_violations() const {
        size_t n = 0;
        for (const auto & line : lines) {
            for (const char * prefix : k_silent_prefixes) {
                if (line.compare(0, strlen(prefix), prefix) == 0) {
                    n++;
                    break;
                }
            }
        }
        return n;
    }
};

struct fixture {
    llama_model_ptr model;
};

static bool build_model(fixture & fx, llm_arch arch) {
    gguf_context_ptr gguf = get_gguf_ctx(arch, moe_mandatory(arch));

    llama_model_params mp                   = llama_model_default_params();
    mp.progress_callback                    = silent_model_load_progress;
    static ggml_backend_dev_t no_devices[1] = { nullptr };
    mp.devices                              = no_devices;
    mp.n_gpu_layers                         = 0;
    mp.load_mode                            = LLAMA_LOAD_MODE_NONE;
    mp.use_extra_bufts                      = false;

    tensor_data_params tp = { 1234, 0.1f };
    fx.model.reset(llama_model_init_from_user(gguf.get(), set_tensor_data, &tp, mp));
    return fx.model != nullptr;
}

static llama_context_params make_params(bool flash_attn, uint32_t n_outputs_max = 0) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = 512;
    cp.flash_attn_type      = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.n_batch              = 64;
    cp.n_ubatch             = 64;
    cp.n_seq_max            = 1;
    cp.n_outputs_max        = n_outputs_max;
    cp.kv_unified           = true;
    cp.n_threads            = 2;
    cp.n_threads_batch      = 2;
    return cp;
}

static llama_measure_context_args cpu_args() {
    llama_measure_context_args args;
    ggml_backend_t             cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu != nullptr) {
        args.backends.emplace_back(cpu);
    }
    return args;
}

// the compute term of a buft: the peak of each chunk over the measured graphs, summed -- the quantity
// llama-measure-plan.h defines once and the tenant caps, the chunk plan and the late check all read
static size_t chunk_peak_total(const sched_measure_buft & entry) {
    return llama_measure_peak_total(llama_measure_peak_per_chunk(entry.peaks));
}

static void check_arch(llm_arch arch, bool flash_attn, uint32_t n_outputs_max) {
    const char * name = llm_arch_name(arch);

    fixture fx;
    if (!build_model(fx, arch)) {
        CHECK(false, "%s: the model did not build", name);
        return;
    }

    const llama_context_params cp = make_params(flash_attn, n_outputs_max);

    // the real context, under the capture: the positive control
    log_capture real_log;
    real_log.start();
    llama_context_ptr real;
    try {
        real.reset(llama_init_from_model(fx.model.get(), cp));
    } catch (const std::exception & e) {
        real_log.stop();
        CHECK(false, "%s: the real context did not build: %s", name, e.what());
        return;
    }
    real_log.stop();
    CHECK(real != nullptr, "%s: no real context", name);
    if (!real) {
        return;
    }
    CHECK(real_log.n_silent_violations() > 10, "%s: the capture saw only %zu lines a real context prints", name,
          real_log.n_silent_violations());
    CHECK(real_log.n_memory_lines() > 0, "%s: the capture saw no line a real context's memory module prints", name);

    // the measure-only context
    const size_t live_before = ggml_backend_test_live_buffer_count();

    llama_measure_context_args args = cpu_args();
    CHECK(!args.backends.empty(), "%s: no CPU backend", name);
    if (args.backends.empty()) {
        return;
    }

    log_capture          measure_log;
    size_t               live_during = 0;
    sched_measure_plan   plan;
    sched_reserve_result status;
    bool                 flags_untouched = false;
    measure_log.start();
    try {
        {
            llama_context measure(*fx.model, cp, &args);
            live_during = ggml_backend_test_live_buffer_count();
            status      = measure.get_measure_status();
            plan        = measure.get_measure_plan();

            // the cparams the caller passed are the cparams it has: the resolution went to a copy
            flags_untouched = measure.get_cparams().auto_fa == (cp.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO);

            // a measure-only context holds no exec context, no output buffer and no compute buffer
            CHECK(!measure.holds_exec_context(), "%s: a measure-only context minted an exec context", name);
            CHECK(!measure.holds_output_buffer(), "%s: a measure-only context reserved an output buffer", name);
            CHECK(measure.get_sched() == nullptr, "%s: a measure-only context kept a scheduler", name);
            CHECK(measure.is_measure_only(), "%s: the flag is not set", name);

            // a measure-only context measures one fixed graph set: warmup never re-reserves it
            const bool warmup_before = measure.get_cparams().warmup;
            measure.set_warmup(!warmup_before);
            CHECK(measure.get_cparams().warmup == warmup_before, "%s: set_warmup changed a measure-only context", name);
        }
    } catch (const std::exception & e) {
        measure_log.stop();
        CHECK(false, "%s: the measure-only context threw: %s", name, e.what());
        return;
    }
    measure_log.stop();

    CHECK(status.status == sched_reserve_status::OK, "%s: the measure ended %d: %s", name, (int) status.status,
          status.reason.c_str());
    CHECK(plan.n_measured > 0, "%s: no graph measured", name);
    CHECK(plan.n_splits_max >= 1, "%s: split count %d", name, plan.n_splits_max);
    CHECK(!plan.bufts.empty(), "%s: no compute buffer type planned", name);
    for (const auto & entry : plan.bufts) {
        CHECK(!entry.cap.empty(), "%s: no chunk cap for %s", name, ggml_backend_buft_name(entry.buft));
        CHECK(entry.max_chunk_size > 0, "%s: no chunk size for %s", name, ggml_backend_buft_name(entry.buft));
    }
    CHECK(flags_untouched, "%s: the caller's cparams were written", name);

    // the K-shift graphs of the plan are the shift sub-caches of the memory, and come last
    {
        std::vector<const llama_kv_cache *> shift_caches;
        if (real->get_memory() != nullptr) {
            real->get_memory()->get_shift_caches(shift_caches);
        }
        size_t n_shift = 0;
        bool   tail    = true;
        for (size_t i = 0; i < plan.graphs.size(); ++i) {
            const bool is_shift = plan.graphs[i].kind == LLAMA_MEASURE_KIND_SHIFT;
            n_shift += is_shift ? 1 : 0;
            tail = tail && (!is_shift || i + shift_caches.size() >= plan.graphs.size());
        }
        CHECK(n_shift == shift_caches.size(), "%s: %zu K-shift graphs measured for %zu shift sub-caches", name, n_shift,
              shift_caches.size());
        CHECK(tail, "%s: a K-shift graph is not among the last", name);
        CHECK(plan.n_measured == plan.graphs.size(), "%s: n_measured %u for %zu graphs", name, plan.n_measured,
              plan.graphs.size());
        n_shift_checked += shift_caches.empty() ? 0 : 1;
    }

    // the memory it measured held size-0 dummies, and nothing is left after it
    CHECK(ggml_backend_test_live_buffer_count() == live_before, "%s: live buffers %zu before, %zu after", name,
          live_before, ggml_backend_test_live_buffer_count());
    (void) live_during;

    // it printed nothing a real context prints
    CHECK(measure_log.n_silent_violations() == 0, "%s: the measure-only context printed %zu silenced lines, first: %s",
          name, measure_log.n_silent_violations(), [&]() -> const char * {
              for (const auto & line : measure_log.lines) {
                  for (const char * prefix : k_silent_prefixes) {
                      if (line.compare(0, strlen(prefix), prefix) == 0) {
                          return line.c_str();
                      }
                  }
              }
              return "";
          }());

    // the plan is the real context's: the worst chunk peak is the buffer its own reserve allocated
    if (!real->get_model().hparams.no_alloc && ggml_backend_sched_get_n_backends(real->get_sched()) == 1) {
        const size_t real_size =
            ggml_backend_sched_get_buffer_size(real->get_sched(), ggml_backend_sched_get_backend(real->get_sched(), 0));
        size_t measured = 0;
        for (const auto & entry : plan.bufts) {
            measured = std::max(measured, chunk_peak_total(entry));
        }
        n_peak_checked++;
        CHECK(measured > 0, "%s: measured nothing", name);
        CHECK(measured <= real_size && real_size <= measured + 1024 * 1024,
              "%s: the measured per-chunk peak total %zu B is not the real context's compute buffer %zu B", name,
              measured, real_size);
    }
}

// --- a model the measure cannot walk (I5, I6) -------------------------------------------------------------

static int    g_run_installs  = 0;
static int    g_run_clears    = 0;
static size_t g_live_at_clear = 0;  // ggml-backend's live buffers at the moment of the last clear

static bool fake_install_ok(uint64_t, enum ggml_sycl_measure_stage) {
    g_run_installs++;
    return true;
}

static void fake_clear_counted() {
    g_run_clears++;
    g_live_at_clear = ggml_backend_test_live_buffer_count();
}

// the memory kinds (llama_memory_kind_unsupported names) the fixtures' refusals named
static std::set<std::string> g_unsupported_kinds_seen;

// An architecture the measure refuses by name: the constructor throws llama_measure_unsupported, the
// load-time measure returns `unsupported` (never a failure), and the plan override it installed is cleared
// exactly as often as it was installed. `by_memory_kind`: the refusal comes from create_memory, inside the
// constructor, after the override was installed; otherwise from the model's own shape, before anything is built.
static bool check_unsupported_arch(llm_arch arch, bool by_memory_kind) {
    const char * name = llm_arch_name(arch);

    fixture fx;
    if (!build_model(fx, arch)) {
        fprintf(stderr, "  SKIP %s: fixture did not build\n", name);
        return false;
    }
    const llama_model & model = *fx.model;

    // the constructor's own refusal, by type and by name
    {
        llama_measure_context_args args  = cpu_args();
        bool                       typed = false;
        std::string                what;
        try {
            llama_context measure(model, make_params(true), &args);
        } catch (const llama_measure_unsupported & e) {
            typed = true;
            what  = e.what();
        } catch (const std::exception & e) {
            what = std::string("untyped: ") + e.what();
        }
        CHECK(typed, "%s: the constructor did not throw llama_measure_unsupported (%s)", name, what.c_str());
        CHECK(what.find(name) != std::string::npos, "%s: the refusal does not name the architecture (%s)", name,
              what.c_str());
        if (by_memory_kind) {
            CHECK(what.find("no no_alloc form") != std::string::npos, "%s: the refusal does not name the reason (%s)",
                  name, what.c_str());
            static const llama_memory_kind kinds[] = { LLAMA_MEMORY_KIND_MSA, LLAMA_MEMORY_KIND_DSA,
                                                       LLAMA_MEMORY_KIND_DSA_ISWA, LLAMA_MEMORY_KIND_DSV4,
                                                       LLAMA_MEMORY_KIND_HYBRID_IDX };
            for (llama_memory_kind k : kinds) {
                if (what.find(llama_memory_kind_unsupported(k)) != std::string::npos) {
                    g_unsupported_kinds_seen.insert(llama_memory_kind_unsupported(k));
                }
            }
        }
    }

    // the load-time measure: `unsupported`, not ok, the named text, the override cleared as often as it was
    // installed, nothing left behind, and the caller's backends still its own
    {
        g_run_installs                                 = 0;
        g_run_clears                                   = 0;
        const size_t                       live_before = ggml_backend_test_live_buffer_count();
        llama_measure_context_args         args        = cpu_args();
        const llama_measure_override_procs procs       = { &fake_install_ok, &fake_clear_counted };
        const llama_load_measure_result    r =
            llama_load_measure_run(model, args, procs, 512, 7, GGML_SYCL_MEASURE_STAGE_PROBE, 0);
        CHECK(!r.ok && r.unsupported, "%s: the measure did not come back unsupported (ok=%d unsupported=%d)", name,
              (int) r.ok, (int) r.unsupported);
        CHECK(r.refusal.rfind("[LOAD-PLAN] compute-slot measure failed at probe on device 0: ", 0) == 0,
              "%s: refusal text: %s", name, r.refusal.c_str());
        CHECK(g_run_installs == g_run_clears, "%s: the override was installed %d times and cleared %d", name,
              g_run_installs, g_run_clears);
        if (by_memory_kind) {
            // the throw came out of the constructor, with the override installed: the unwind cleared it
            CHECK(g_run_installs == 1, "%s: the constructor's refusal ran with %d installs", name, g_run_installs);
        } else {
            CHECK(g_run_installs == 0, "%s: a model refused by shape installed the override %d times", name,
                  g_run_installs);
        }
        CHECK(ggml_backend_test_live_buffer_count() == live_before, "%s: the unsupported measure left buffers behind",
              name);
        // the constructor takes the backends out of `args` when it gets that far: a model refused by shape never
        // reaches it and the caller keeps them, a refusal from create_memory unwound a context that had them
        CHECK(args.backends.empty() == by_memory_kind, "%s: %zu backends left in args after a %s refusal", name,
              args.backends.size(), by_memory_kind ? "memory-kind" : "shape");
    }

    if (!by_memory_kind) {
        CHECK(!llama_measure_unsupported_reason(model).empty(), "%s: llama_measure_unsupported_reason is empty", name);
    } else {
        CHECK(llama_measure_unsupported_reason(model).empty(),
              "%s: a memory-kind refusal is reported before the memory is built", name);
    }
    return true;
}

// M1: a measure that fails after the override went in unwinds it. The model builds, the override is
// installed and cleared exactly once on a good run, a refused install is a named refusal with no clear, and a
// missing proc is a named refusal that never builds the context.
static void check_run_paths() {
    fixture fx;
    if (!build_model(fx, LLM_ARCH_LLAMA)) {
        CHECK(false, "the llama fixture did not build");
        return;
    }
    const llama_model & model = *fx.model;

    {
        g_run_installs                           = 0;
        g_run_clears                             = 0;
        g_live_at_clear                                = 0;
        const size_t                       live_before = ggml_backend_test_live_buffer_count();
        llama_measure_context_args         args  = cpu_args();
        const llama_measure_override_procs procs = { &fake_install_ok, &fake_clear_counted };
        const llama_load_measure_result    r =
            llama_load_measure_run(model, args, procs, 512, 7, GGML_SYCL_MEASURE_STAGE_CANDIDATE_B, 0);
        CHECK(r.ok && !r.unsupported && r.refusal.empty(), "a good measure refused: %s", r.refusal.c_str());
        CHECK(g_run_installs == 1 && g_run_clears == 1, "a good run: %d installs, %d clears", g_run_installs,
              g_run_clears);
        // the override is cleared while the measure context is still alive (its scheduler's buffers exist),
        // and the context is gone after: a clear that ran after the destruct would see the buffers released
        CHECK(g_live_at_clear > live_before,
              "the override was cleared with %zu live buffers (%zu before the run): "
              "after the context was destroyed",
              g_live_at_clear, live_before);
        CHECK(ggml_backend_test_live_buffer_count() == live_before, "the run left %zu buffers (%zu before)",
              ggml_backend_test_live_buffer_count(), live_before);
        // (a CPU-only measure has no SYCL device or host-tier term to return; the terms it does return add up)
        for (const auto & d : r.devices) {
            size_t sum = 0;
            for (size_t c : d.chunk_bytes) {
                sum += c;
            }
            CHECK(d.total == sum && d.total > 0, "the term %zu is not the sum %zu of its chunks", d.total, sum);
        }
        CHECK(r.n_splits >= 1, "split count %d", r.n_splits);
    }
    {
        g_run_installs                           = 0;
        g_run_clears                             = 0;
        llama_measure_context_args         args  = cpu_args();
        const llama_measure_override_procs procs = { [](uint64_t, enum ggml_sycl_measure_stage) {
                                                        g_run_installs++;
                                                        return false;
                                                    },
                                                     &fake_clear_counted };
        const llama_load_measure_result r =
            llama_load_measure_run(model, args, procs, 512, 7, GGML_SYCL_MEASURE_STAGE_CANDIDATE_C, 3);
        CHECK(!r.ok && !r.unsupported, "a refused install is not a plain refusal");
        CHECK(
            r.refusal == "[LOAD-PLAN] compute-slot measure failed at late on device 3: plan override nested (refused)",
            "refusal text: %s", r.refusal.c_str());
        CHECK(g_run_installs == 1 && g_run_clears == 0, "refused install: %d installs, %d clears", g_run_installs,
              g_run_clears);
    }
    {
        g_run_installs                       = 0;
        g_run_clears                         = 0;
        llama_measure_context_args      args = cpu_args();
        const llama_load_measure_result r = llama_load_measure_run(model, args, { nullptr, &fake_clear_counted }, 512,
                                                                   7, GGML_SYCL_MEASURE_STAGE_PROBE, 0);
        CHECK(!r.ok && !r.unsupported && r.refusal.find("plan override proc missing") != std::string::npos,
              "a missing proc: %s", r.refusal.c_str());
        CHECK(g_run_clears == 0, "a missing proc cleared %d times", g_run_clears);
    }
    {
        // a constructor that throws a plain error (an n_ctx the model cannot hold) is a failure, not unsupported
        g_run_installs                           = 0;
        g_run_clears                             = 0;
        llama_measure_context_args         args  = cpu_args();
        const llama_measure_override_procs procs = { &fake_install_ok, &fake_clear_counted };
        args.backends.clear();  // no backend at all: the constructor cannot build a scheduler
        const llama_load_measure_result r =
            llama_load_measure_run(model, args, procs, 512, 7, GGML_SYCL_MEASURE_STAGE_PROBE, 0);
        CHECK(!r.ok && !r.unsupported, "a backend-less measure was not a plain failure (unsupported=%d)",
              (int) r.unsupported);
        CHECK(g_run_installs == g_run_clears, "a failed construction: %d installs, %d clears", g_run_installs,
              g_run_clears);
    }
}

int main() {
    static const llm_arch archs[] = { LLM_ARCH_LLAMA, LLM_ARCH_GEMMA3, LLM_ARCH_MAMBA, LLM_ARCH_QWEN35 };
    for (llm_arch arch : archs) {
        for (bool fa : { false, true }) {
            // 0: every token yields an output. 1: the output gather leaves one row, so the last layer
            // runs on a single row -- the shape a ubatch with few outputs gives, which on SYCL is where a
            // zero-row node (a dispatch no-op, llama.cpp-479i) appears. The measure must still bound the
            // real compute buffer there.
            for (uint32_t n_outputs_max : { 0u, 1u }) {
                check_arch(arch, fa, n_outputs_max);
            }
        }
    }
    check_run_paths();

    // every memory kind without a no_alloc form: the architectures that reach one. A kind no fixture
    // reaches is covered by test-layer-shapes' by-kind table; this arm shows the measure's own mapping.
    int n_memory_refused = 0;
    for (llm_arch arch : { LLM_ARCH_DEEPSEEK32, LLM_ARCH_DEEPSEEK4, LLM_ARCH_MINIMAX_M3, LLM_ARCH_QWEN4EXP,
                           LLM_ARCH_DOTS3NOTE, LLM_ARCH_HY_V4, LLM_ARCH_GLM_DSA }) {
        n_memory_refused += check_unsupported_arch(arch, true) ? 1 : 0;
    }
    CHECK(n_memory_refused >= 1 && !g_unsupported_kinds_seen.empty(), "VOID: %d memory-kind refusals, %zu kinds named",
          n_memory_refused, g_unsupported_kinds_seen.size());
    for (const auto & k : g_unsupported_kinds_seen) {
        fprintf(stderr, "  unsupported kind refused by the measure: %s\n", k.c_str());
    }

    // an encoder graph and a ctx_other arch are refused by name before anything is built
    int n_shape_refused = 0;
    for (llm_arch arch : { LLM_ARCH_T5 }) {
        n_shape_refused += check_unsupported_arch(arch, false) ? 1 : 0;
    }
    CHECK(n_shape_refused >= 1, "VOID: no encoder architecture was refused by shape");

    // the ctx_other archs go through llama_model_needs_ctx_other, the one predicate the constructor and the
    // measure's refusal share: a fixture whose tensors make it true must be refused, and a model that is not
    // such an arch must not (a mutant that reads the predicate as false refuses nothing and dies on the count)
    int n_ctx_other_refused = 0;
    for (llm_arch arch : { LLM_ARCH_GEMMA4_ASSISTANT }) {
        fixture fx;
        if (!build_model(fx, arch)) {
            CHECK(false, "%s: the ctx_other fixture did not build", llm_arch_name(arch));
            continue;
        }
        CHECK(llama_model_needs_ctx_other(*fx.model), "%s: the predicate does not name it", llm_arch_name(arch));
        n_ctx_other_refused += check_unsupported_arch(arch, false) ? 1 : 0;
    }
    CHECK(n_ctx_other_refused >= 1, "VOID: no ctx_other architecture was refused by shape");
    {
        fixture fx;
        if (build_model(fx, LLM_ARCH_LLAMA)) {
            CHECK(!llama_model_needs_ctx_other(*fx.model), "a plain llama model is not a ctx_other arch");
        }
    }

    // a run that compared no plan with a real context proved nothing about the plan
    CHECK(n_shift_checked >= 1, "no plan was compared with a memory that has a K-shift sub-cache");
    CHECK(n_peak_checked >= 12, "only %d measured plans were compared with a real context", n_peak_checked);

    if (n_failed != 0) {
        fprintf(stderr, "%d check(s) failed\n", n_failed);
        return 1;
    }
    printf("test-measure-context: ok (%d plans compared with a real context)\n", n_peak_checked);
    return 0;
}
