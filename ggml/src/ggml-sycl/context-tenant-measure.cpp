#include "context-tenant-measure.hpp"

#include <algorithm>
#include <cstdio>

namespace {

// One row per cohort id, in id order: the lookup indexes by id, and the host
// test checks the order. Every registered cohort is the context's own.
const ggml_sycl_context_cohort_info k_cohorts[GGML_SYCL_CONTEXT_COHORT_COUNT] = {
    { GGML_SYCL_CONTEXT_COHORT_COMPUTE,           "context-compute",           GGML_SYCL_CONTEXT_COHORT_TIER_DEVICE,
     GGML_SYCL_CONTEXT_COHORT_SCOPE_CONTEXT, GGML_SYCL_CONTEXT_COHORT_LIFETIME_CONTEXT },
    { GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST,      "context-compute-host",      GGML_SYCL_CONTEXT_COHORT_TIER_HOST_PINNED,
     GGML_SYCL_CONTEXT_COHORT_SCOPE_CONTEXT, GGML_SYCL_CONTEXT_COHORT_LIFETIME_CONTEXT },
    { GGML_SYCL_CONTEXT_COHORT_FATTN_MATERIALIZE, "context-fattn-materialize", GGML_SYCL_CONTEXT_COHORT_TIER_DEVICE,
     GGML_SYCL_CONTEXT_COHORT_SCOPE_CONTEXT, GGML_SYCL_CONTEXT_COHORT_LIFETIME_CONTEXT },
    { GGML_SYCL_CONTEXT_COHORT_NONFA_STAGE,       "context-nonfa-stage",       GGML_SYCL_CONTEXT_COHORT_TIER_DEVICE,
     GGML_SYCL_CONTEXT_COHORT_SCOPE_CONTEXT, GGML_SYCL_CONTEXT_COHORT_LIFETIME_CONTEXT },
    { GGML_SYCL_CONTEXT_COHORT_GRAPH_STAGE,       "context-graph-stage",       GGML_SYCL_CONTEXT_COHORT_TIER_DEVICE,
     GGML_SYCL_CONTEXT_COHORT_SCOPE_CONTEXT, GGML_SYCL_CONTEXT_COHORT_LIFETIME_CONTEXT },
};

// No visitor is registered yet; the first producer adds its row before the
// terminator.
const ggml_sycl::context_measure_visitor k_visitors[] = {
    { nullptr, nullptr },
};

}  // namespace

const ggml_sycl_context_cohort_info * ggml_sycl_context_cohort_lookup(uint32_t cohort) {
    return cohort < GGML_SYCL_CONTEXT_COHORT_COUNT ? &k_cohorts[cohort] : nullptr;
}

int32_t ggml_sycl_context_cohort_element_device(uint32_t cohort, int32_t device) {
    const ggml_sycl_context_cohort_info * info = ggml_sycl_context_cohort_lookup(cohort);
    return info && info->tier == GGML_SYCL_CONTEXT_COHORT_TIER_HOST_PINNED ? -1 : device;
}

namespace ggml_sycl {

void context_demand_accum::demand(const context_measure_view & view,
                                  uint32_t                     cohort,
                                  uint32_t                     slot_index,
                                  uint64_t                     bytes) {
    const ggml_sycl_context_cohort_info * info = ggml_sycl_context_cohort_lookup(cohort);
    if (info == nullptr) {
        char what[96];
        std::snprintf(what, sizeof(what), "demand for unknown cohort id %u", (unsigned) cohort);
        fail(what);
        return;
    }
    // A device-tier slot on device -1 would read as a host-tier slot.
    if (info->tier == GGML_SYCL_CONTEXT_COHORT_TIER_DEVICE && view.device < 0) {
        fail(std::string("device-tier cohort ") + info->name + " demanded with a negative device");
        return;
    }
    // A zero-byte demand (the fattn slot is 0 by default) records no element:
    // an element is a slot to carve and to claim, and there is nothing to claim.
    if (bytes == 0) {
        return;
    }
    const int32_t device = ggml_sycl_context_cohort_element_device(cohort, view.device);
    for (slot & s : slots_) {
        if (s.device == device && s.cohort == cohort && s.slot_index == slot_index) {
            s.bytes = std::max(s.bytes, bytes);
            return;
        }
    }
    slots_.push_back({ device, cohort, slot_index, bytes });
}

void context_demand_accum::fail(const std::string & what) {
    if (error_.empty()) {
        error_ = what;
    }
}

std::vector<ggml_sycl_context_tenant_desc> context_demand_accum::tenants() const {
    std::vector<ggml_sycl_context_tenant_desc> out;
    if (!error_.empty()) {
        return out;
    }
    std::vector<slot> sorted = slots_;
    std::sort(sorted.begin(), sorted.end(), [](const slot & a, const slot & b) {
        if (a.device != b.device) {
            return a.device < b.device;
        }
        if (a.cohort != b.cohort) {
            return a.cohort < b.cohort;
        }
        return a.slot_index < b.slot_index;
    });
    out.reserve(sorted.size());
    for (const slot & s : sorted) {
        ggml_sycl_context_tenant_desc desc = {};
        desc.struct_size                   = sizeof(desc);
        desc.cohort                        = s.cohort;
        desc.slot_index                    = s.slot_index;
        desc.device                        = s.device;
        desc.slot_bytes                    = s.bytes;
        out.push_back(desc);
    }
    return out;
}

const context_measure_visitor * context_measure_visitors() {
    return k_visitors;
}

void context_measure_visit(const context_measure_visitor * visitors,
                           const ggml_tensor *             node,
                           const context_measure_view &    view,
                           context_demand_accum &          acc) {
    for (const context_measure_visitor * v = visitors; v && v->fn; ++v) {
        v->fn(node, view, acc);
    }
}

}  // namespace ggml_sycl
