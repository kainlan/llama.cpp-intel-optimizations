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
//   - a memory kind with no no_alloc form refuses by name, as a status-free throw from the constructor.

#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-context.h"
#include "../src/llama-kv-cache.h"
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
#include <string>
#include <vector>

static int n_failed       = 0;
static int n_peak_checked = 0;

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
    "llama_context:",      "~llama_context:",    "sched_reserve:", "sched_reserve_impl:",
    "sched_measure_impl:", "resolve_fused_ops:", "graph_reserve:", "llama_graph_n_input_tensors:",
};

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

// the largest sum over a graph's chunk peaks, across the measured graphs of a compute buffer type
static size_t worst_total_peak(const sched_measure_buft & entry) {
    size_t worst = 0;
    for (const auto & graph : entry.peaks) {
        size_t sum = 0;
        for (size_t p : graph) {
            sum += p;
        }
        worst = std::max(worst, sum);
    }
    return worst;
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
            measured = std::max(measured, worst_total_peak(entry));
        }
        n_peak_checked++;
        CHECK(measured > 0, "%s: measured nothing", name);
        CHECK(measured <= real_size && real_size <= measured + 1024 * 1024,
              "%s: the measured worst peak %zu B is not the real context's compute buffer %zu B", name, measured,
              real_size);
    }
}

static void check_refusal() {
    // a memory kind with no no_alloc form refuses by name; DEEPSEEK32 has an indexer key cache
    fixture fx;
    if (!build_model(fx, LLM_ARCH_DEEPSEEK32)) {
        return;  // the fixture is optional here
    }
    llama_measure_context_args args  = cpu_args();
    bool                       threw = false;
    std::string                what;
    try {
        llama_context measure(*fx.model, make_params(true), &args);
    } catch (const std::exception & e) {
        threw = true;
        what  = e.what();
    }
    CHECK(threw, "a memory kind with no no_alloc form built a measure-only context");
    CHECK(what.find("no no_alloc form") != std::string::npos, "the refusal does not name the reason: %s", what.c_str());
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
    check_refusal();

    // a run that compared no plan with a real context proved nothing about the plan
    CHECK(n_peak_checked >= 12, "only %d measured plans were compared with a real context", n_peak_checked);

    if (n_failed != 0) {
        fprintf(stderr, "%d check(s) failed\n", n_failed);
        return 1;
    }
    printf("test-measure-context: ok (%d plans compared with a real context)\n", n_peak_checked);
    return 0;
}
