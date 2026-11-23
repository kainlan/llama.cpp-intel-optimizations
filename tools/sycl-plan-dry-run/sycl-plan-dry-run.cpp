// llama-sycl-plan-dry-run: the host plan a SYCL load of a GGUF model would make, printed term by term, without loading
// the model (th32 §4.2 "One source for the number"). It reads the GGUF header only, builds the load's early inventory
// through the load's own builder, and hands it to the SYCL backend's plan dry run, which plans it through the planner
// the load's stage calls and stores nothing. The SYCL backend is initialised (the planner reads device info), but no
// model, context, cache or buffer is created. With no cache there is no active arena, and the planners subtract the
// arena's zone reservations only with one, so each plan's device share is an upper bound and its host terms a lower
// bound; the printout carries a line that says so.
//
// It plans the EARLY inventory: the one the load plans before create_tensor, every tensor of the file but the lazy
// rows the record defers to the storage tier. The dry run plans that inventory because it is the one the load's own
// early plan plans (th32 §4.2's "the host plan a load would make" is the early plan's), so it differs from the load's
// late inventory in exactly the ways the load's early plan does, three of them:
//   - tensors created off SYCL, on another backend's buffer: in the early inventory, not the late one;
//   - file tensors the load never creates, a TENSOR_SKIP tensor and one the architecture does not use: in the early
//     inventory, not the late one;
//   - the copies a TENSOR_DUPLICATED create makes, such as a tied output head created from token_embd: in the late
//     inventory, not the early one, which holds the file's one tensor.

#include "common.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "llama.h"
#include "sycl-plan-dry-run-args.h"

// The load's inventory builder over a header read (llama_model_sycl_header_inventory), an undecorated libllama
// internal kept out of the public API and reached the way tests/test-llama-lazy-range-prefill.cpp reaches the loader's.
#include "../src/llama-model.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

static_assert(SYCL_PLAN_DRY_RUN_ARGS_MAX_DEVICES == GGML_SYCL_PLAN_DRY_RUN_MAX_DEVICES,
              "--device-budget names the devices the export reads");

typedef enum ggml_sycl_lifecycle_result (*sycl_plan_dry_run_fn)(const struct ggml_sycl_tensor_inventory *,
                                                                const struct ggml_sycl_plan_dry_run_opts *,
                                                                struct ggml_sycl_plan_dry_run_result *);

static void print_usage(FILE * out, const char * argv0) {
    std::fprintf(out,
                 "usage: %s -m <file.gguf> [options]\n"
                 "\n"
                 "Prints the host plan a SYCL load of the model would make, term by term, and stores nothing.\n"
                 "Run it with no other process loading a model on these devices.\n"
                 "\n"
                 "  -m, --model FILE         the GGUF file (the first split of a split model)\n"
                 "  --mem-available GIB      plan as if MemAvailable were GIB (fractions allowed)\n"
                 "  --device-budget D=GIB    plan device D against GIB of VRAM instead of its own budget; repeatable\n"
                 "  -c, --ctx-size N         the context the load plans room for (0, the default: the model's)\n"
                 "  -lzm, --lazy-mode MODE   on, auto or off, as for llama-cli\n"
                 "  -h, --help               this text\n",
                 argv0);
}

// The dry run's own lines go to stdout, and so do the lazy-range record's, in the pre-fill's own words: a table the
// record defers to the storage tier is left out of the plan, and its row says so beside the plan. Warnings and errors
// go to stderr; the rest of the log is dropped.
static void log_callback(enum ggml_log_level level, const char * text, void * /*user*/) {
    if (std::strstr(text, "[PLAN-DRY-RUN") != nullptr || std::strstr(text, "lazy-range record:") != nullptr) {
        std::fputs(text, stdout);
    } else if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

static const char * result_name(enum ggml_sycl_lifecycle_result rc) {
    switch (rc) {
        case GGML_SYCL_LIFECYCLE_OK:
            return "OK";
        case GGML_SYCL_LIFECYCLE_NULL_OUTPUT:
            return "NULL_OUTPUT (a struct this backend does not read)";
        case GGML_SYCL_LIFECYCLE_LOAD_BUSY:
            return "LOAD_BUSY (a load is open or a model is live)";
        case GGML_SYCL_LIFECYCLE_BUSY:
            return "BUSY (the backend module is being changed)";
        case GGML_SYCL_LIFECYCLE_NOT_FOUND:
            return "NOT_FOUND (no SYCL device)";
        case GGML_SYCL_LIFECYCLE_EFFECT_FAILED:
            return "EFFECT_FAILED (planning threw)";
        default:
            return "another status";
    }
}

struct dry_run_call {
    sycl_plan_dry_run_fn                           fn = nullptr;
    ggml_sycl_plan_dry_run_opts                    opts{};
    std::unique_ptr<ggml_sycl_plan_dry_run_result> result;
    enum ggml_sycl_lifecycle_result                rc     = GGML_SYCL_LIFECYCLE_OK;
    bool                                           called = false;
};

static void on_inventory(const ggml_sycl_tensor_inventory *   inventory,
                         const ggml_sycl_placement_envelope * envelope,
                         void *                               user_data) {
    dry_run_call * call = static_cast<dry_run_call *>(user_data);
    call->opts.envelope = envelope;
    call->rc            = call->fn(inventory, &call->opts, call->result.get());
    call->called        = true;
    call->opts.envelope = nullptr;  // the envelope lives only for the callback
}

int main(int argc, char ** argv) {
    sycl_plan_dry_run_args args;
    std::string            err;
    if (!sycl_plan_dry_run_parse_args(argc, argv, args, err)) {
        std::fprintf(stderr, "error: %s\n\n", err.c_str());
        print_usage(stderr, argv[0]);
        return 1;
    }
    if (args.help) {
        print_usage(stdout, argv[0]);
        return 0;
    }

    llama_log_set(log_callback, nullptr);
    llama_backend_init();
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("SYCL");
    if (reg == nullptr) {
        std::fprintf(stderr, "error: no SYCL backend is registered in this build\n");
        llama_backend_free();
        return 1;
    }

    dry_run_call call;
    call.fn = reinterpret_cast<sycl_plan_dry_run_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_sycl_plan_dry_run"));
    if (call.fn == nullptr) {
        std::fprintf(stderr, "error: the SYCL backend has no ggml_backend_sycl_plan_dry_run\n");
        llama_backend_free();
        return 1;
    }
    call.opts.struct_size         = sizeof(call.opts);
    call.opts.version             = GGML_SYCL_PLAN_DRY_RUN_VERSION;
    call.opts.mem_available_set   = args.mem_available_set ? 1 : 0;
    call.opts.mem_available_bytes = args.mem_available_bytes;
    call.opts.n_device_budgets    = args.n_device_budgets;
    for (uint32_t d = 0; d < args.n_device_budgets; ++d) {
        call.opts.device_budget_bytes[d] = args.device_budget_bytes[d];
    }
    call.result              = std::make_unique<ggml_sycl_plan_dry_run_result>();
    *call.result             = ggml_sycl_plan_dry_run_result{};
    call.result->struct_size = sizeof(ggml_sycl_plan_dry_run_result);
    call.result->version     = GGML_SYCL_PLAN_DRY_RUN_VERSION;

    // The load's own parameters: a header read, the requested context as the load's hint (mapped as every tool that
    // takes -c maps it), no micro-batch named.
    llama_model_params params = llama_model_default_params();
    params.n_ctx_hint         = common_n_ctx_hint(args.n_ctx);
    params.n_ubatch_hint      = 0;
    if (args.lazy_mode_set) {
        params.lazy_mode = args.lazy_mode;
    }

    const bool read = llama_model_sycl_header_inventory(args.model.c_str(), params, on_inventory, &call);
    int        rc   = 0;
    if (!read || !call.called) {
        std::fprintf(stderr, "error: could not build the inventory from the GGUF header of %s\n", args.model.c_str());
        rc = 1;
    } else if (call.rc != GGML_SYCL_LIFECYCLE_OK) {
        std::fprintf(stderr, "error: the plan dry run returned %s\n", result_name(call.rc));
        rc = 1;
    } else {
        const ggml_sycl_plan_dry_run_result & res = *call.result;
        if (res.n_plans == 0) {
            std::printf("[PLAN-DRY-RUN] no plan: the load would plan nothing\n");
        }
        for (uint32_t i = 0; i < res.n_plans; ++i) {
            const ggml_sycl_plan_dry_run_plan & p  = res.plans[i];
            const double                        MB = 1024.0 * 1024.0;
            if (p.expected_host <= p.host_budget) {
                std::printf(
                    "[PLAN-DRY-RUN] plan=%u device=%d verdict: expected_host %.1f MB fits host_budget %.1f MB\n", i,
                    p.device, p.expected_host / MB, p.host_budget / MB);
            } else {
                std::printf(
                    "[PLAN-DRY-RUN] plan=%u device=%d verdict: expected_host %.1f MB exceeds host_budget %.1f MB by "
                    "%.1f MB\n",
                    i, p.device, p.expected_host / MB, p.host_budget / MB, (p.expected_host - p.host_budget) / MB);
            }
        }
    }
    llama_backend_free();
    return rc;
}
