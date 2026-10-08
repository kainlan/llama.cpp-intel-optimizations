#pragma once

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "llama-auto-ubatch.h"
#include "llama-context-tenant.h"
#include "llama.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// The load-time measure (zhcn design 2.10): one function that builds a transient measure-only
// llama_context over a placement, runs one MEASURE and returns, per device, the compute chunk sizes at
// the stage's cap, their total, the scheduler's split count and the cap it read. Its call sites are the
// load's three stages (ggml_sycl_measure_stage); there is never a second implementation.
//
// This header holds what a host test can run without a device: the plan-override guard and the dummy
// buffers a measure puts under the weights, both over plain function pointers and ggml-backend calls.
// The measure itself and the late check are in llama-context.cpp.

struct llama_model;
struct llama_measure_context_args;

// A model the load-time measure cannot measure, by name: a memory kind with no no_alloc form, an
// architecture that needs a second context, a model with an encoder graph. It is not a failure and not a
// refusal of the load: the measure returns it as `unsupported`, and the load goes on the unplanned path with
// one WARN that names it. Anything else a measure throws is a failure of the measure and refuses the load.
struct llama_measure_unsupported : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The two backend entry points the plan-override guard calls, taken together from one function so a
// missing one is a named refusal and an override can never be installed without its clear.
struct llama_measure_override_procs {
    decltype(&ggml_backend_sycl_measure_plan_override_install) install = nullptr;
    decltype(&ggml_backend_sycl_measure_plan_override_clear)   clear   = nullptr;
};

// In a static build the backend's two functions; under GGML_BACKEND_DL the entries found by name through
// the SYCL reg of `dev`. A backend that does not export them leaves both null.
llama_measure_override_procs llama_context_sycl_measure_override_procs(ggml_backend_dev_t dev);

#ifdef LLAMA_PRIVATE_TEST_OBJECTS
// The vehicle's build replaces the pair (null, null restores the backend's own).
void llama_context_sycl_measure_override_procs_override_for_testing(
    decltype(&ggml_backend_sycl_measure_plan_override_install) install_fn,
    decltype(&ggml_backend_sycl_measure_plan_override_clear)   clear_fn);
#endif

// The plan override of one measure, as a scope. Its constructor installs, its destructor clears, and
// nothing else does. It is declared after the measure backends and the holder of the measure context
// and before the context is constructed, so the constructor's pipeline-parallel read and create_memory's
// KV read see it, and on every exit, a throw included, it clears first, then the context goes, then the
// backends.
//
// A missing proc installs nothing and names itself; an install that answers false (a nest, or no plan
// staged for the load) installs nothing and names itself. The measure then refuses by that name and
// never runs without the override.
class llama_measure_plan_override {
  public:
    llama_measure_plan_override(const llama_measure_override_procs & procs,
                                uint64_t                             load_txn,
                                enum ggml_sycl_measure_stage         stage) :
        clear_fn(procs.clear) {
        if (procs.install == nullptr || procs.clear == nullptr) {
            failure_text = "plan override proc missing";
            return;
        }
        if (!procs.install(load_txn, stage)) {
            failure_text = "plan override nested";
            return;
        }
        installed_ = true;
    }

    ~llama_measure_plan_override() {
        if (installed_) {
            clear_fn();
        }
    }

    llama_measure_plan_override(const llama_measure_plan_override &)             = delete;
    llama_measure_plan_override & operator=(const llama_measure_plan_override &) = delete;

    bool installed() const { return installed_; }

    // null when the override is installed, else why it is not
    const char * failure() const { return failure_text; }

  private:
    decltype(&ggml_backend_sycl_measure_plan_override_clear) clear_fn     = nullptr;
    const char *                                             failure_text = nullptr;
    bool                                                     installed_   = false;
};

// One ggml context of the weights and the buffer type its tensors will live in.
struct llama_measure_dummy_entry {
    ggml_backend_buffer_type_t buft = nullptr;
    ggml_context *             ctx  = nullptr;
};

// The weights' stand-ins for one measure: a size-0 buffer of each entry's buffer type, marked as
// holding weights, set as the buffer of every tensor of that entry's context that has none. The
// scheduler then pins a weight-consuming op to the weight's backend and starts a new split for an
// incompatible weight, as it does for the real buffers.
//
// Its destructor, on every exit, nulls the buffer of each tensor it set, then frees the buffers: a
// tensor that kept a dummy would trip ggml_backend_tensor_alloc's `buffer == NULL` assertion at the
// real allocation. A tensor that already had a buffer is left alone and left set.
class llama_measure_dummy_scope {
  public:
    explicit llama_measure_dummy_scope(const std::vector<llama_measure_dummy_entry> & entries) {
        for (const auto & e : entries) {
            if (e.buft == nullptr || e.ctx == nullptr || ggml_get_first_tensor(e.ctx) == nullptr) {
                continue;
            }
            ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(e.buft, /*size =*/0);
            if (buf == nullptr) {
                failed_ = true;
                continue;
            }
            bufs_.push_back(buf);
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for (ggml_tensor * t = ggml_get_first_tensor(e.ctx); t != nullptr; t = ggml_get_next_tensor(e.ctx, t)) {
                if (t->buffer == nullptr) {
                    t->buffer = buf;
                    set_.push_back(t);
                }
            }
        }
    }

    ~llama_measure_dummy_scope() {
        for (ggml_tensor * t : set_) {
            t->buffer = nullptr;
        }
        for (ggml_backend_buffer_t buf : bufs_) {
            ggml_backend_buffer_free(buf);
        }
    }

    llama_measure_dummy_scope(const llama_measure_dummy_scope &)             = delete;
    llama_measure_dummy_scope & operator=(const llama_measure_dummy_scope &) = delete;

    // a buffer type refused its size-0 buffer: the measure would see a weight with no buffer
    bool failed() const { return failed_; }

    size_t n_buffers() const { return bufs_.size(); }

  private:
    std::vector<ggml_backend_buffer_t> bufs_;
    std::vector<ggml_tensor *>         set_;
    bool                               failed_ = false;
};

// The text of every refusal of a load-time measure, in one place:
//   [LOAD-PLAN] compute-slot measure failed at <probe|admitted|late> on device %d: <reason> (refused)
inline const char * llama_load_measure_stage_name(enum ggml_sycl_measure_stage stage) {
    switch (stage) {
        case GGML_SYCL_MEASURE_STAGE_PROBE:
            return "probe";
        case GGML_SYCL_MEASURE_STAGE_CANDIDATE_B:
            return "admitted";
        case GGML_SYCL_MEASURE_STAGE_CANDIDATE_C:
            return "late";
    }
    return "unknown";
}

inline std::string llama_load_measure_refusal_text(enum ggml_sycl_measure_stage stage,
                                                   int                          device,
                                                   const std::string &          reason) {
    return std::string("[LOAD-PLAN] compute-slot measure failed at ") + llama_load_measure_stage_name(stage) +
           " on device " + std::to_string(device) + ": " + reason + " (refused)";
}

// The shape a load-time measure uses: n_ubatch is the auto ladder's bottom rung (the smallest ubatch any
// context of this load can end up with, from the helper the real context's trial reads), and the caller's
// n_ctx, the training context for 0; every other parameter is the default.
inline llama_context_params llama_load_measure_context_params(uint32_t n_ctx, uint32_t n_ctx_train) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx                = n_ctx != 0 ? n_ctx : n_ctx_train;
    params.n_ubatch             = llama_auto_ubatch_ladder[0];
    params.n_batch              = std::max(params.n_batch, params.n_ubatch);
    return params;
}

// The measured compute term of one device buft (or of the host buft, `host`): the chunk cap the
// scope answered, each chunk's planned size, and their total.
struct llama_load_measure_device {
    int32_t             device = -1;  // the post-selector SYCL device index; -1 for the host buft
    bool                host   = false;
    std::vector<size_t> chunk_bytes;  // llama_measure_peak_per_chunk(): the peak of each gallocr chunk
    size_t              total = 0;    // llama_measure_peak_total(chunk_bytes)
    size_t              cap   = 0;    // the largest chunk the scope allowed
};

struct llama_load_measure_result {
    bool                                   ok          = false;
    bool                                   unsupported = false;  // !ok because the model cannot be measured
    std::string                            refusal;              // the named refusal when !ok
    std::vector<llama_load_measure_device> devices;
    int                                    n_splits = 0;         // the most splits any measured graph took
};

// Whether a context of this model cannot be built without a second context to read from: the gemma4 assistant
// shares its target's KV cache, and an EAGLE3 or DFLASH draft without its own embedding or output tensors
// borrows the target's. The context constructor and the measure's unsupported reason both ask this.
bool llama_model_needs_ctx_other(const llama_model & model);

// Why this model cannot be measured by a measure-only context, or "": an encoder graph is not walked, and
// an architecture that needs ctx_other has none to read. (A memory kind with no no_alloc form is named by
// create_memory, which throws llama_measure_unsupported for it.)
std::string llama_measure_unsupported_reason(const llama_model & model);

// The measure over a caller-built set of backends. The backends are in `args` until the context's
// constructor takes them (a model refused by shape never gets that far, and the caller keeps them); from
// then on the context owns them and frees them with itself. The plan override (installed before the context
// is constructed) is declared after the context's holder, so on a normal exit it clears first and the
// context, backends included, goes after; on a throw from the constructor the context has already unwound
// and the override clears next. A throw from the constructor, a status that is not OK and a missing proc
// come back as the one refusal text; a llama_measure_unsupported comes back as `unsupported`.
llama_load_measure_result llama_load_measure_run(const llama_model &                  model,
                                                 llama_measure_context_args &         args,
                                                 const llama_measure_override_procs & procs,
                                                 uint32_t                             n_ctx,
                                                 uint64_t                             load_txn,
                                                 enum ggml_sycl_measure_stage         stage,
                                                 int                                  first_device);

// The measure at `stage` for the load `load_txn`, over `model`'s placement (the plan override names the
// plan; the model's dev_layer and weight buffers are what the placement is). A model with no SYCL device
// measures nothing and is ok with no devices.
llama_load_measure_result llama_load_measure(const llama_model &          model,
                                             uint32_t                     n_ctx,
                                             uint64_t                     load_txn,
                                             enum ggml_sycl_measure_stage stage);

// What the backend said of each device's term at the late stage, and what the load does with it.
struct llama_late_check_result {
    std::string          refusal;       // non-empty: the load is refused, by this named text
    std::string          unsupported;   // non-empty: the model cannot be measured; the load goes on, with a WARN
    uint32_t             n_ubatch = 0;  // the measure's ubatch, for the text of a miss
    std::vector<int32_t> not_recorded;  // devices with no early term to compare: NOT a pass
    // the measured compute term c(P) of each not_recorded device, in the same order: what the late measure found,
    // printed so a load can be read against the real compute buffer even though nothing was compared
    std::vector<size_t>  not_recorded_bytes;
};

// The WARN the loader prints for a device the backend recorded nothing for: nothing was compared, which is
// not a pass. `measured_bytes` is the late measure's c(P) term for that device, in MiB with one decimal.
inline std::string llama_late_check_not_recorded_text(int32_t device, uint32_t n_ubatch, size_t measured_bytes) {
    char mib[32];
    std::snprintf(mib, sizeof(mib), "%.1f", measured_bytes / 1024.0 / 1024.0);
    return "[LOAD-PLAN] late check on device " + std::to_string(device) +
           ": no early compute term was recorded for this load, nothing was compared (ubatch " +
           std::to_string(n_ubatch) + "; measured compute term " + mib + " MiB on device " + std::to_string(device) +
           ")";
}

// Folds the measured devices through the backend's late check. A device the backend recorded nothing for
// stays in `not_recorded`; it is never read as EQUAL. The first REFUSED ends the fold with the named refusal
// (the backend has logged its own canonical line). The host tier is skipped: the backend's entry point takes a
// SYCL device index, and no host term is recorded at the early stage today. That skip is of the late
// COMPARISON only. A host-tier refusal can still occur at every stage, probe and admitted included, but
// inside the measure, not in a comparison: the CPU backend's host compute buft (or its activation twin) has
// no refusal source and its chunk cap is the per-process constant, while its chunk plan is refused like any
// buft's when the peaks need more chunks than allowed, and that comes back as the measure's own failure
// naming the stage. (SYCL_CpuOffloadCompute does refuse under a plan scope, by design.) The refusal text's
// device field is the first SYCL device's, so a host-tier refusal reads "on device N" while its reason names
// SYCL_Host or SYCL_CpuActivation, whichever is the compute buft. (llama has no probe or admitted call site
// yet; the backend planner's stages will read the host term from the same result.)
inline llama_late_check_result llama_late_check_fold(const llama_sycl_l4_procs &                    procs,
                                                     struct ggml_sycl_load_txn                      txn,
                                                     const std::vector<llama_load_measure_device> & devices,
                                                     uint32_t                                       n_ubatch) {
    llama_late_check_result out;
    out.n_ubatch = n_ubatch;
    for (const auto & d : devices) {
        if (d.host) {
            continue;
        }
        switch (llama_sycl_l4_late_check(procs, txn, d.device, d.total)) {
            case GGML_SYCL_LATE_CHECK_EQUAL:
            case GGML_SYCL_LATE_CHECK_SHRINK_ADMITTED:
                break;
            case GGML_SYCL_LATE_CHECK_REFUSED:
                out.refusal =
                    llama_load_measure_refusal_text(GGML_SYCL_MEASURE_STAGE_CANDIDATE_C, d.device,
                                                    "the final placement needs more compute than the admitted term");
                return out;
            case GGML_SYCL_LATE_CHECK_NOT_RECORDED:
                out.not_recorded.push_back(d.device);
                out.not_recorded_bytes.push_back(d.total);
                break;
        }
    }
    return out;
}

// The late check (stage (c)): after the dev_layer sync and before the mappings are initialised, measure
// the load's final placement over the real weights' dummies and hand each device's term to the backend.
// It measures only when the backend exports all three L4 entry points: without them no c(P) was recorded
// and there is nothing to compare, so the call is inert (an empty result).
llama_late_check_result llama_load_late_check(const llama_model &                            model,
                                              uint32_t                                       n_ctx,
                                              struct ggml_sycl_load_txn                      txn,
                                              const std::vector<llama_measure_dummy_entry> & weights);
