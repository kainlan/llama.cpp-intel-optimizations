#pragma once

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "llama-auto-ubatch.h"
#include "llama-context-tenant.h"
#include "llama-kv-cache.h"
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
    decltype(&ggml_backend_sycl_measure_plan_override_install_kv) install = nullptr;
    decltype(&ggml_backend_sycl_measure_plan_override_clear)   clear   = nullptr;
};

// In a static build the backend's two functions; under GGML_BACKEND_DL the entries found by name through
// the SYCL reg of `dev`. A backend that does not export them leaves both null.
llama_measure_override_procs llama_context_sycl_measure_override_procs(ggml_backend_dev_t dev);

#ifdef LLAMA_PRIVATE_TEST_OBJECTS
// The vehicle's build replaces the pair (null, null restores the backend's own).
void llama_context_sycl_measure_override_procs_override_for_testing(
    decltype(&ggml_backend_sycl_measure_plan_override_install_kv) install_fn,
    decltype(&ggml_backend_sycl_measure_plan_override_clear)      clear_fn);
#endif

// The plan override of one measure, as a scope. Its constructor installs, its destructor clears, and
// nothing else does. It is declared after the measure backends and the holder of the measure context
// and before the context is constructed, so the constructor's pipeline-parallel read and create_memory's
// KV read see it, and on every exit, a throw included, it clears first, then the context goes, then the
// backends.
//
// `kv_shape` is the measure context's KV shape (llama_load_measure_kv_shape): the backend re-fits the
// staged plan's KV residency for it, so the measure's KV sits where the context's will (llama.cpp-p6i0).
//
// A missing proc installs nothing and names itself; an install that answers false (a nest, no plan
// staged for the load, or a KV re-fit the backend refused) installs nothing and names itself. The
// measure then refuses by that name and never runs without the override.
class llama_measure_plan_override {
  public:
    llama_measure_plan_override(const llama_measure_override_procs &      procs,
                                uint64_t                                  load_txn,
                                enum ggml_sycl_measure_stage              stage,
                                const struct ggml_sycl_measure_kv_shape * kv_shape) :
        clear_fn(procs.clear) {
        if (procs.install == nullptr || procs.clear == nullptr) {
            failure_text = "plan override proc missing";
            return;
        }
        if (!procs.install(load_txn, stage, kv_shape)) {
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

// The KV shape of a measure context built with `params`, which the plan override's KV re-fit sizes for
// (llama.cpp-p6i0). Every field is the params' own, so the re-fit and the context agree.
inline struct ggml_sycl_measure_kv_shape llama_load_measure_kv_shape(const llama_context_params & params) {
    struct ggml_sycl_measure_kv_shape shape;
    shape.n_ctx      = params.n_ctx;
    shape.n_ubatch   = params.n_ubatch;
    shape.n_seq_max  = params.n_seq_max;
    shape.kv_unified = params.kv_unified;
    shape.swa_full   = params.swa_full;
    return shape;
}

// The n_ctx a load-time measure runs at: the caller's, or the training context for 0. The measure's params, the
// reservation and the record all read it, so the n_ctx a term is reserved and recorded at is the one it was measured
// at, and never 0 (llama.cpp-p6i0).
inline uint32_t llama_load_measure_n_ctx(uint32_t n_ctx, uint32_t n_ctx_train) {
    return n_ctx != 0 ? n_ctx : n_ctx_train;
}

// The shape a load-time measure uses: n_ubatch is the auto ladder's bottom rung (from the helper the real context's
// trial reads), and the caller's n_ctx, the training context for 0; every other parameter is the default. The bottom
// rung is the ubatch a MoE model's auto pick stays at (its cap is MOE_GPU_UBATCH_MAX) and the one a pinned -ub 512
// runs at. A dense model's auto pick climbs above it, to 2048 at the default n_batch, into whatever room the zones
// have left: that larger buffer is not reserved. The caller's -c and -ub do not reach the load (fkpg).
inline llama_context_params llama_load_measure_context_params(uint32_t n_ctx, uint32_t n_ctx_train) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx                = llama_load_measure_n_ctx(n_ctx, n_ctx_train);
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
    // A device only: the context memory the measure placed on this device's plain buffer type (a hybrid or recurrent
    // model's state). The real context allocates it in the RUNTIME zone before the compute buffer (llama.cpp-p6i0).
    size_t              state_bytes = 0;
};

// The reservation's door for one measured device (llama.cpp-p6i0): its per-chunk peaks, at the n_ctx it was
// measured at.
inline bool llama_sycl_l4_reserve_compute_term(const llama_sycl_l4_procs &       procs,
                                               struct ggml_sycl_load_txn         txn,
                                               const llama_load_measure_device & d,
                                               uint32_t                          n_ctx) {
    return llama_sycl_l4_reserve_compute_term(procs, txn, d.device, d.chunk_bytes, n_ctx);
}

struct llama_load_measure_result {
    bool                                   ok          = false;
    bool                                   unsupported = false;  // !ok because the model cannot be measured
    std::string                            refusal;              // the named refusal when !ok
    std::vector<llama_load_measure_device> devices;
    int                                    n_splits = 0;         // the most splits any measured graph took
    llama_kv_residency_tally               kv;                   // the KV residency of the caches the measure built
    // llama.cpp-p6i0 (the compute trace): the measure's per-buffer-type, per-graph chunk peaks and splits, its
    // shape and its KV residency, as INFO lines. The measure-only context is quiet while it lives, so
    // llama_load_measure prints them once the context is gone.
    std::vector<std::string>               trace;
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
    // devices whose late state had no state term to compare (llama.cpp-p6i0), and that state, in the same order
    std::vector<int32_t> state_not_recorded;
    std::vector<size_t>  state_not_recorded_bytes;
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

// The WARN the loader prints for a device whose late state had no state term to compare (llama.cpp-p6i0): that state
// is allocated in the RUNTIME zone with nothing reserved for it, the defect the state term exists to prevent.
inline std::string llama_late_check_state_not_recorded_text(int32_t device, size_t measured_bytes) {
    char mib[32];
    std::snprintf(mib, sizeof(mib), "%.1f", measured_bytes / 1024.0 / 1024.0);
    return "[LOAD-PLAN] late state check on device " + std::to_string(device) +
           ": no state term was recorded for this load, nothing was compared (measured state " + mib +
           " MiB on device " + std::to_string(device) + "; it is allocated in the RUNTIME zone unplanned)";
}

// The context-init comparison of one SYCL device's recurrent state with the load's planned state term
// (llama.cpp-p6i0). Every load-time measure runs at n_seq_max 1, the only shape a load sees, so a context with more
// sequences allocates more state than the term holds, RUNTIME-first and before the compute buffer, and the excess
// draws down the compute term. -np is a normal user option: the context is kept as asked and this is a WARN, never a
// refusal. The comparison is exact (one byte over warns). Empty when the state fits the term.
inline std::string llama_context_state_excess_text(int32_t  device,
                                                   uint32_t n_seq_max,
                                                   uint64_t planned_bytes,
                                                   uint64_t real_bytes) {
    if (real_bytes <= planned_bytes) {
        return {};
    }
    char real_mib[32];
    char planned_mib[32];
    char excess_mib[32];
    std::snprintf(real_mib, sizeof(real_mib), "%.1f", real_bytes / 1024.0 / 1024.0);
    std::snprintf(planned_mib, sizeof(planned_mib), "%.1f", planned_bytes / 1024.0 / 1024.0);
    std::snprintf(excess_mib, sizeof(excess_mib), "%.1f", (real_bytes - planned_bytes) / 1024.0 / 1024.0);
    return "[LOAD-PLAN] state term exceeded on device " + std::to_string(device) + ": this context holds " + real_mib +
           " MiB of recurrent state at n_seq_max " + std::to_string(n_seq_max) + ", the load planned " + planned_mib +
           " MiB (measured at n_seq_max 1, the only shape a load sees); the " + excess_mib +
           " MiB excess is allocated in the RUNTIME zone unplanned and draws down the compute term, so a compute "
           "buffer chunk can land outside RUNTIME (zone=raw). The context is kept as asked";
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
// SYCL_Host or SYCL_CpuActivation, whichever is the compute buft. The probe and admitted stages skip the host tier
// the same way (llama_load_probe_bound, llama_admitted_check_fold): no host term is reserved or recorded.
//
// A device whose late measure placed recurrent state on it (llama.cpp-p6i0) then has that state checked against the
// state term under its own name, never against c(P): REFUSED ends the fold by name, NOT_RECORDED lists the device in
// `state_not_recorded`. A device whose late state is zero is not asked, so a state that left a device entirely (its
// layers moved to the CPU at the dev_layer sync) gives no shrink WARN; its reservation stands, as a shrink's does.
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
        if (d.state_bytes == 0) {
            continue;
        }
        switch (llama_sycl_l4_late_check_state(procs, txn, d.device, d.state_bytes)) {
            case GGML_SYCL_LATE_CHECK_EQUAL:
            case GGML_SYCL_LATE_CHECK_SHRINK_ADMITTED:
                break;
            case GGML_SYCL_LATE_CHECK_REFUSED:
                out.refusal = llama_load_measure_refusal_text(
                    GGML_SYCL_MEASURE_STAGE_CANDIDATE_C, d.device,
                    "the final placement needs more recurrent state than the reserved state term");
                return out;
            case GGML_SYCL_LATE_CHECK_NOT_RECORDED:
                out.state_not_recorded.push_back(d.device);
                out.state_not_recorded_bytes.push_back(d.state_bytes);
                break;
        }
    }
    return out;
}

// The probe measure (stage (a), llama.cpp-p6i0): after create_tensor and before the late plan packs the
// weights, measure the probe placement's compute term C-hat over the weights' stand-ins and hand each SYCL
// device's chunks to the backend, which reserves them as a planned RUNTIME term so the pack leaves room for the
// compute buffer. A device the backend does not reserve for (it says why at WARN) keeps its compute buffer
// unplanned. Inert, like the late check, unless the backend exports the L4 entry points and both load terms.
struct llama_load_probe_result {
    std::string                            refusal;      // non-empty: the measure failed, by this named text
    std::string                            unsupported;  // non-empty: the model cannot be measured
    bool                                   measured = false;
    std::vector<llama_load_measure_device> devices;      // C-hat per device when measured
    std::vector<int32_t>                   not_reserved;  // SYCL devices the backend declined to reserve for
    llama_kv_residency_tally               kv;            // the probe measure's KV residency (the admitted fold's)
};

// The probe's reservations (llama.cpp-p6i0), per measured SYCL device and in order: its state term first, when the
// measure placed context memory on its plain buffer type, then its compute term. The state goes first because the real
// context allocates it in the RUNTIME zone before the compute buffer: a compute term reserved without it is drawn down
// by the state, and the buffer's last chunk lands outside the arena (Qwen3.8 on the B70). A device whose state the
// backend declines is declined whole and is not offered its compute term; one whose compute term is declined keeps
// its state term, which is real memory either way. The host tier reserves nothing. Returns the declined devices.
inline std::vector<int32_t> llama_load_probe_reserve(const llama_sycl_l4_procs &                    procs,
                                                     struct ggml_sycl_load_txn                      txn,
                                                     const std::vector<llama_load_measure_device> & devices,
                                                     uint32_t                                       n_ctx) {
    std::vector<int32_t> declined;
    for (const llama_load_measure_device & d : devices) {
        if (d.host) {
            continue;
        }
        if (d.state_bytes != 0 && !llama_sycl_l4_reserve_state_term(procs, txn, d.device, d.state_bytes)) {
            declined.push_back(d.device);
            continue;
        }
        // false: the backend said why at WARN, and this device's compute buffer stays unplanned
        if (!llama_sycl_l4_reserve_compute_term(procs, txn, d, n_ctx)) {
            declined.push_back(d.device);
        }
    }
    return declined;
}

llama_load_probe_result llama_load_probe_bound(const llama_model &                            model,
                                               uint32_t                                       n_ctx,
                                               struct ggml_sycl_load_txn                      txn,
                                               const std::vector<llama_measure_dummy_entry> & weights);

// The admitted check (stage (b), llama.cpp-p6i0): c(P), measured at the admitted placement after the late plan
// packed the weights, against the probe bound C-hat the pack reserved room for. Both are sized in the reservation's
// units (ggml_backend_sycl_load_compute_term_bytes: each chunk at the RUNTIME allocator's grain, summed), so the
// comparison is the one the reservation can honour, not one between raw sums. c(P) <= C-hat is admitted and c(P),
// never C-hat, is what the ledger records for the late check to compare; c(P) > C-hat, a device with no probe bound,
// or a term the backend cannot size refuses the load by name.
//
// The one exception is the KV-residency delta. The probe re-fits the KV residency for the room the zones leave
// before the term is carved; the admitted measure re-fits it after RUNTIME grew by the term and the late plan packed
// the weights. Neither room is guaranteed larger, so KV layers can move between device, host and CPU from one measure
// to the other, and the compute graph moves with them: C-hat does not bound c(P) then, and no monotone bound exists
// (a graph's peak is not monotone in placement; GPT-OSS on the B50 moved two KV layers to the host and c(P) stayed
// equal). So c(P) > C-hat with a residency that moved is admitted, its excess in the reservation's units named on the
// term (`kv_excess`) for the loader's WARN, and c(P) is recorded as usual. The excess is not reserved: the arena is
// already packed, and that part of the compute buffer can land outside the RUNTIME zone, as every byte of it did
// before the reservation existed. With an unmoved residency the growth is a planning defect and still refuses.
//
// A device the backend declined to reserve for
// (`not_reserved`) has no reservation to compare with: it is listed, not compared and not recorded, so its compute
// buffer stays unplanned and its late check answers NOT_RECORDED, as before the reservation existed. The host tier
// is skipped, as in the late fold.
struct llama_admitted_term {
    int32_t device         = -1;
    bool    reserved       = true;  // false: the backend declined the probe bound; neither compared nor recorded
    size_t  probe_term     = 0;     // C-hat in the reservation's units: the room the pack left
    size_t  admitted_term  = 0;     // c(P) in the reservation's units, compared with probe_term
    size_t  admitted_bytes = 0;     // c(P) as the measure's total: what the ledger records and the late check compares
    size_t  kv_excess      = 0;     // admitted_term - probe_term, admitted because the KV residency moved; unreserved
    size_t  state_bytes    = 0;     // the probe's state, the one the reservation holds: recorded under its own name
    bool    state_recorded = false;  // the backend recorded state_bytes as this device's state term
};

struct llama_admitted_check_result {
    std::string                      refusal;         // non-empty: the load is refused, by this named text
    std::string                      unsupported;     // non-empty: the model cannot be measured; the load goes on
    std::vector<llama_admitted_term> terms;           // one per SYCL device, when admitted
    uint32_t                         n_ctx      = 0;  // the n_ctx and ubatch the measure ran at
    uint32_t                         n_ubatch   = 0;
    size_t                           n_recorded = 0;  // the terms the backend recorded
    size_t                           n_state_recorded = 0;  // the state terms the backend recorded
    llama_kv_residency_tally         probe_kv;        // the KV residency each measure saw
    llama_kv_residency_tally         admitted_kv;
};

inline bool llama_kv_residency_same(const llama_kv_residency_tally & a, const llama_kv_residency_tally & b) {
    return a.n_device == b.n_device && a.n_host == b.n_host && a.n_cpu == b.n_cpu;
}

inline llama_admitted_check_result llama_admitted_check_fold(const llama_sycl_l4_procs &                    procs,
                                                             const std::vector<llama_load_measure_device> & probe,
                                                             const std::vector<int32_t> & not_reserved,
                                                             const std::vector<llama_load_measure_device> & admitted,
                                                             uint32_t                                       n_ctx,
                                                             uint32_t                                       n_ubatch,
                                                             const llama_kv_residency_tally &               probe_kv,
                                                             const llama_kv_residency_tally & admitted_kv) {
    llama_admitted_check_result out;
    out.n_ctx           = n_ctx;
    out.n_ubatch        = n_ubatch;
    out.probe_kv        = probe_kv;
    out.admitted_kv     = admitted_kv;
    const bool kv_moved = !llama_kv_residency_same(probe_kv, admitted_kv);
    const auto refuse = [&](int32_t device, const std::string & why) {
        out.refusal = "[LOAD-PLAN] compute-slot-exceeds-probe-bound on device " + std::to_string(device) + ": " + why +
                      " at n_ctx " + std::to_string(n_ctx) + " ubatch " + std::to_string(n_ubatch) + " (refused)";
        out.terms.clear();
    };
    for (const auto & d : admitted) {
        if (d.host) {
            continue;
        }
        const llama_load_measure_device * bound = nullptr;
        for (const auto & p : probe) {
            if (!p.host && p.device == d.device) {
                bound = &p;
                break;
            }
        }
        if (bound == nullptr) {
            refuse(d.device, "c(P) " + std::to_string(d.total) + " B and no probe bound");
            return out;
        }
        llama_admitted_term t;
        t.device         = d.device;
        t.admitted_bytes = d.total;
        t.reserved       = std::find(not_reserved.begin(), not_reserved.end(), d.device) == not_reserved.end();
        t.state_bytes    = bound->state_bytes;
        const bool sized = llama_sycl_l4_compute_term_bytes(procs, bound->chunk_bytes, &t.probe_term) &&
                           llama_sycl_l4_compute_term_bytes(procs, d.chunk_bytes, &t.admitted_term);
        if (t.reserved && !sized) {
            refuse(d.device, "the compute term cannot be sized in the reservation's units");
            return out;
        }
        if (t.reserved && t.admitted_term > t.probe_term) {
            if (!kv_moved) {
                refuse(d.device, "c(P) " + std::to_string(t.admitted_term) + " B > probe bound C-hat " +
                                     std::to_string(t.probe_term) + " B in the reservation's units");
                return out;
            }
            t.kv_excess = t.admitted_term - t.probe_term;
        }
        out.terms.push_back(t);
    }
    return out;
}

// Records each reserved admitted term, c(P), at the measure's n_ctx. Returns how many the backend recorded: a record
// it refuses is not counted, and the late check then answers NOT_RECORDED for that device. A device with no
// reservation is not recorded.
inline size_t llama_admitted_record(const llama_sycl_l4_procs &         procs,
                                    struct ggml_sycl_load_txn           txn,
                                    const llama_admitted_check_result & admitted,
                                    uint32_t                            n_ctx) {
    size_t n = 0;
    for (const auto & t : admitted.terms) {
        if (t.reserved && llama_sycl_l4_record_compute_term(procs, txn, t.device, t.admitted_bytes, n_ctx)) {
            ++n;
        }
    }
    return n;
}

// Records each reserved device's state, the probe's (the one the reservation holds), under the state term's own name
// at the measure's n_ctx (llama.cpp-p6i0), and marks each term whose state the backend recorded. A device with no
// state or no reservation records nothing. Returns how many were recorded. The state is never added to c(P): the
// late check compares a late state with this record and a late c(P) with c(P).
inline size_t llama_admitted_record_state(const llama_sycl_l4_procs &   procs,
                                          struct ggml_sycl_load_txn     txn,
                                          llama_admitted_check_result & admitted,
                                          uint32_t                      n_ctx) {
    size_t n = 0;
    for (auto & t : admitted.terms) {
        t.state_recorded = t.reserved && t.state_bytes != 0 &&
                           llama_sycl_l4_record_state_term(procs, txn, t.device, t.state_bytes, n_ctx);
        n += t.state_recorded ? 1 : 0;
    }
    return n;
}

llama_admitted_check_result llama_load_admitted_check(const llama_model &                            model,
                                                      uint32_t                                       n_ctx,
                                                      struct ggml_sycl_load_txn                      txn,
                                                      const std::vector<llama_measure_dummy_entry> & weights,
                                                      const llama_load_probe_result &                probe);

// The late check (stage (c)): after the dev_layer sync and before the mappings are initialised, measure
// the load's final placement over the real weights' dummies and hand each device's term to the backend.
// It measures only when the backend exports all three L4 entry points: without them no c(P) was recorded
// and there is nothing to compare, so the call is inert (an empty result).
llama_late_check_result llama_load_late_check(const llama_model &                            model,
                                              uint32_t                                       n_ctx,
                                              struct ggml_sycl_load_txn                      txn,
                                              const std::vector<llama_measure_dummy_entry> & weights);
