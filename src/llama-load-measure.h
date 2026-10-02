#pragma once

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
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

// The measured compute term of one device buft (or of the host buft, `host`): the chunk cap the
// scope answered, each chunk's planned size, and their total.
struct llama_load_measure_device {
    int32_t             device = -1;  // the post-selector SYCL device index; -1 for the host buft
    bool                host   = false;
    std::vector<size_t> chunk_bytes;  // one per gallocr chunk
    size_t              total = 0;
    size_t              cap   = 0;    // the largest chunk the scope allowed
};

struct llama_load_measure_result {
    bool                                   ok = false;
    std::string                            refusal;       // the named refusal when !ok
    std::vector<llama_load_measure_device> devices;
    int                                    n_splits = 0;  // the most splits any measured graph took
};

// The measure at `stage` for the load `load_txn`, over `model`'s placement (the plan override names the
// plan; the model's dev_layer and weight buffers are what the placement is). `n_ctx` is the envelope's
// (0: the model's training context), n_ubatch is the auto ladder's bottom rung and every other cparam is
// the default. A throw from the measure or a refusal of any kind comes back as
//   [LOAD-PLAN] compute-slot measure failed at <probe|admitted|late> on device %d: <reason> (refused)
// and never as a missing term. A model with no SYCL device measures nothing and is ok with no devices.
llama_load_measure_result llama_load_measure(const llama_model &          model,
                                             uint32_t                     n_ctx,
                                             uint64_t                     load_txn,
                                             enum ggml_sycl_measure_stage stage);

// The late check (stage (c)): after the dev_layer sync and before the mappings are initialised, measure
// the load's final placement over the real weights' dummies and hand each device's term to the backend.
// Returns "" when the load may go on and the named refusal when it may not. It measures only when the
// backend exports all three L4 entry points: without them no c(P) was recorded and there is nothing to
// compare, so the call is inert.
std::string llama_load_late_check(const llama_model &                            model,
                                  uint32_t                                       n_ctx,
                                  struct ggml_sycl_load_txn                      txn,
                                  const std::vector<llama_measure_dummy_entry> & weights);
