#pragma once

// The measured-tenant section of a context's published plan: the cohort table
// and the visitor contract that fill the section while the measure pass walks a
// graph. The element and the cohort ids are in context-tenant-desc.h, a header
// C can include, so the extern "C" backend header can name the element without
// a second definition.
//
// The tier, scope and lifetime of a cohort come from the table below; the zone
// is added with the allocator's zone vocabulary, not here.
//
// This header names only ggml types, so a host test builds it without a device.
// Its two functions are C++ and are not part of the backend's proc-address
// surface; libggml-sycl exports them only because nothing hides symbols, and a
// caller across the dlopen boundary needs a proc-address export first.

#include "context-tenant-desc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum ggml_sycl_context_cohort_tier {
    GGML_SYCL_CONTEXT_COHORT_TIER_DEVICE,       // a slot on the element's device
    GGML_SYCL_CONTEXT_COHORT_TIER_HOST_PINNED,  // a slot in the host arena; the element's device is -1
};

enum ggml_sycl_context_cohort_scope {
    GGML_SYCL_CONTEXT_COHORT_SCOPE_CONTEXT,  // held by the context's registry entry
};

// The lifetime class of the slots. Today every registered cohort is the
// context's own; a class that is not is added with its first cohort.
enum ggml_sycl_context_cohort_lifetime {
    GGML_SYCL_CONTEXT_COHORT_LIFETIME_CONTEXT,
};

struct ggml_sycl_context_cohort_info {
    uint32_t                          id;
    const char *                      name;
    ggml_sycl_context_cohort_tier     tier;
    ggml_sycl_context_cohort_scope    scope;
    ggml_sycl_context_cohort_lifetime lifetime;
};

// The row of `cohort`, or nullptr for an id the table does not know. A reader
// of a published section refuses an unknown id rather than guessing its tier.
const ggml_sycl_context_cohort_info * ggml_sycl_context_cohort_lookup(uint32_t cohort);

// The element's device for a slot of `cohort` measured on `device`: -1 for the
// host tier, `device` otherwise. The one place the tier decides the device.
int32_t ggml_sycl_context_cohort_element_device(uint32_t cohort, int32_t device);

// Contract (b): a runtime site claims `bytes` from a slot of `slot_bytes`, and
// a claim over the slot is a [CONTEXT-PLAN-BUG], never a silent growth.
inline bool ggml_sycl_context_claim_fits(uint64_t bytes, uint64_t slot_bytes) {
    return bytes <= slot_bytes;
}

namespace ggml_sycl {

enum class context_measure_phase : uint8_t {
    PP,
    TG,
    SHIFT,
};

// The graph split a visited node is in: the split's inputs and the copies the
// scheduler inserted for them. Both arrays are borrowed from the measure
// scheduler for the duration of one visit.
struct context_measure_split {
    int                         index    = 0;
    const ggml_tensor * const * inputs   = nullptr;
    size_t                      n_inputs = 0;
    const ggml_tensor * const * copies   = nullptr;
    size_t                      n_copies = 0;
};

// What a visitor may read about the measure pass. The scheduler lookups are
// callbacks so the header names no scheduler type. `sched_backend(t)` answers
// for any tensor, leaf INPUT tensors included, and returns nullptr for a tensor
// the scheduler assigned no backend; a visitor treats that as an error.
struct context_measure_view {
    int32_t                       device                                                 = 0;
    context_measure_phase         phase                                                  = context_measure_phase::PP;
    uint32_t                      n_streams                                              = 1;
    const context_measure_split * split                                                  = nullptr;
    void *                        sched_ctx                                              = nullptr;
    ggml_backend_t (*sched_backend)(void * sched_ctx, const ggml_tensor * tensor)        = nullptr;
    ggml_backend_buffer_type_t (*backend_buft)(void * sched_ctx, ggml_backend_t backend) = nullptr;
};

// What the visitors fill. Each slot is the maximum over every measured graph
// of the demand recorded for it, so a visitor adds on every visit and never
// reads the slot back. The first error wins and is named; a result with an
// error carries no tenants.
class context_demand_accum {
  public:
    // Records `bytes` for slot `slot_index` of `cohort`. The element's device
    // comes from the cohort's tier, not from the caller. An unknown cohort is
    // a named error.
    void demand(const context_measure_view & view, uint32_t cohort, uint32_t slot_index, uint64_t bytes);

    // Records a named error: a demand that cannot be sized is not a zero.
    void fail(const std::string & what);

    bool ok() const { return error_.empty(); }

    const std::string & error() const { return error_; }

    // The tenant section, ordered by (device, cohort, slot_index). Empty when
    // an error was recorded.
    std::vector<ggml_sycl_context_tenant_desc> tenants() const;

  private:
    struct slot {
        int32_t  device;
        uint32_t cohort;
        uint32_t slot_index;
        uint64_t bytes;
    };

    std::vector<slot> slots_;
    std::string       error_;
};

// A visitor reads one node of one measured graph. It calls the runtime site's
// own size function (contract (a)), so the measure and the site cannot size
// the same demand two ways.
using context_measure_visitor_fn = void (*)(const ggml_tensor *          node,
                                            const context_measure_view & view,
                                            context_demand_accum &       acc);

struct context_measure_visitor {
    const char *               name;
    context_measure_visitor_fn fn;
};

// The static visitor table, terminated by an entry with a null fn. Nothing
// registers itself: a producer adds its row to the table in the .cpp.
const context_measure_visitor * context_measure_visitors();

// Runs every visitor of `visitors` on `node`. The measure walker calls this
// with the table above; a test passes its own.
void context_measure_visit(const context_measure_visitor * visitors,
                           const ggml_tensor *             node,
                           const context_measure_view &    view,
                           context_demand_accum &          acc);

}  // namespace ggml_sycl
