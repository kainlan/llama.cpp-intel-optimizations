#pragma once

#include "ggml-sycl-cohort.h"
#include "ggml-sycl-l4-procs.h"
#include "ggml-sycl.h"
#include "llama-measure-plan.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// The llama side of the measured-tenant publish (zhcn design 2.6, 2.7, 3.3). Pure helpers over
// plain integers and the backend's wire structs: tests/test-context-tenant-section.cpp executes
// them on the host with no device.
//
//   llama_sycl_l4_procs        the four L4 entry points and the fail-closed way to call each;
//   llama_tenant_section_*     the tenant section from the measured compute caps, and its merge
//                              with the backend visitors' demands;
//   llama_tenant_key_digest()  the tenant key;
//   llama_tenant_plan_line()   the line the scorer matches.

// The L4 entry points: the tenant publish, the coverage query, the load-time late check and the
// residency probe (ggml-sycl.h, "The measured-tenant publish"). A backend that predates them leaves a
// proc null, and each reader below then answers the value that makes the caller do the safe thing:
// no publish happened (UNSUPPORTED), the section needs a transaction (GROWTH), nothing was
// compared (NOT_RECORDED), and the probe gave no answer (NOT_ANSWERED). The table is filled by
// llama-context.cpp through the SYCL reg's proc address in every link mode, by the names
// ggml-sycl-l4-procs.h pins.
struct llama_sycl_l4_procs {
    decltype(&ggml_backend_sycl_set_runtime_context_desc) publish         = nullptr;
    decltype(&ggml_backend_sycl_tenant_coverage)          coverage        = nullptr;
    decltype(&ggml_backend_sycl_load_late_check)          late_check      = nullptr;
    decltype(&ggml_backend_sycl_probe_residency)          probe_residency = nullptr;

    // A planned context needs all four: a publish that cannot be covered-checked, a load that
    // cannot be late-checked, or a plan whose residency cannot be probed, is half a plan.
    bool available() const {
        return publish != nullptr && coverage != nullptr && late_check != nullptr && probe_residency != nullptr;
    }
};

inline ggml_sycl_lifecycle_result llama_sycl_l4_publish(const llama_sycl_l4_procs &            procs,
                                                        ggml_backend_t                         backend,
                                                        struct ggml_sycl_model_token           model,
                                                        uint32_t                               n_ctx,
                                                        uint32_t                               n_ubatch,
                                                        uint32_t                               n_seq_max,
                                                        bool                                   kv_unified,
                                                        bool                                   swa_full,
                                                        bool                                   flash_attn_enabled,
                                                        const ggml_sycl_runtime_context_desc * desc) {
    if (procs.publish == nullptr) {
        return GGML_SYCL_LIFECYCLE_UNSUPPORTED;
    }
    return procs.publish(backend, model, n_ctx, n_ubatch, n_seq_max, kv_unified, swa_full, flash_attn_enabled, desc);
}

// GROWTH is the answer to anything this reader cannot vouch for: a null proc, and a value
// outside the enum (a newer backend's answer is not "covered" to an older reader).
inline ggml_sycl_tenant_coverage llama_sycl_l4_coverage(const llama_sycl_l4_procs &            procs,
                                                        ggml_backend_t                         backend,
                                                        uint32_t                               n_ctx,
                                                        uint32_t                               n_ubatch,
                                                        uint32_t                               n_seq_max,
                                                        bool                                   kv_unified,
                                                        bool                                   swa_full,
                                                        bool                                   flash_attn_enabled,
                                                        const ggml_sycl_runtime_context_desc * candidate) {
    if (procs.coverage == nullptr) {
        return GGML_SYCL_TENANT_COVERAGE_GROWTH;
    }
    const ggml_sycl_tenant_coverage r =
        procs.coverage(backend, n_ctx, n_ubatch, n_seq_max, kv_unified, swa_full, flash_attn_enabled, candidate);
    switch (r) {
        case GGML_SYCL_TENANT_COVERAGE_EQUAL:
        case GGML_SYCL_TENANT_COVERAGE_COVERED:
            return r;
        default:
            return GGML_SYCL_TENANT_COVERAGE_GROWTH;
    }
}

// NOT_RECORDED is "nothing was checked", which no caller reads as a pass. A null proc and a
// value outside the enum both read as it.
inline ggml_sycl_late_check_result llama_sycl_l4_late_check(const llama_sycl_l4_procs & procs,
                                                            struct ggml_sycl_load_txn   txn,
                                                            int32_t                     device,
                                                            uint64_t                    compute_bytes) {
    if (procs.late_check == nullptr) {
        return GGML_SYCL_LATE_CHECK_NOT_RECORDED;
    }
    const ggml_sycl_late_check_result r = procs.late_check(txn, device, compute_bytes);
    switch (r) {
        case GGML_SYCL_LATE_CHECK_EQUAL:
        case GGML_SYCL_LATE_CHECK_SHRINK_ADMITTED:
        case GGML_SYCL_LATE_CHECK_REFUSED:
            return r;
        default:
            return GGML_SYCL_LATE_CHECK_NOT_RECORDED;
    }
}

// The residency probe's one door: no other code calls the proc pointer or the symbol (scripts/check-sycl-l4-proc-registration.py
// pins it). NOT_ANSWERED is the answer to anything this reader cannot vouch for: a null proc, and a value outside the
// enum (a newer backend's status is not "OK" to an older reader). `out->n_layer` is cleared to 0 for those, so a stale
// count left in the caller's struct is never read as an answer. Only OK carries a vector, and this door never writes
// out->host_resident: the backend does, on OK alone. GEOMETRY_NOT_WIRED and every other refusal pass through as
// themselves, and a caller must treat every status but OK as "no answer", never as "no host layers".
inline ggml_sycl_residency_probe_status llama_sycl_l4_probe_residency(const llama_sycl_l4_procs &  procs,
                                                                      ggml_backend_t               backend,
                                                                      struct ggml_sycl_model_token model,
                                                                      uint32_t                     n_ctx,
                                                                      uint32_t                     n_ubatch,
                                                                      uint32_t                     n_seq_max,
                                                                      bool                         kv_unified,
                                                                      bool                         swa_full,
                                                                      bool                         flash_attn_enabled,
                                                                      const ggml_sycl_runtime_context_desc * desc,
                                                                      struct ggml_sycl_residency_probe *     out) {
    if (procs.probe_residency == nullptr) {
        if (out != nullptr) {
            out->n_layer = 0;
        }
        return GGML_SYCL_RESIDENCY_PROBE_NOT_ANSWERED;
    }
    const ggml_sycl_residency_probe_status r = procs.probe_residency(
        backend, model, n_ctx, n_ubatch, n_seq_max, kv_unified, swa_full, flash_attn_enabled, desc, out);
    switch (r) {
        case GGML_SYCL_RESIDENCY_PROBE_NOT_ANSWERED:
        case GGML_SYCL_RESIDENCY_PROBE_OK:
        case GGML_SYCL_RESIDENCY_PROBE_GEOMETRY_NOT_WIRED:
        case GGML_SYCL_RESIDENCY_PROBE_INVALID:
        case GGML_SYCL_RESIDENCY_PROBE_HEAD_SLOT_REFUSED:
        case GGML_SYCL_RESIDENCY_PROBE_NO_PROMOTION_VIOLATED:
        case GGML_SYCL_RESIDENCY_PROBE_N_LAYER_CAP_TOO_SMALL:
        case GGML_SYCL_RESIDENCY_PROBE_FOREIGN_BACKEND:
            return r;
        default:
            if (out != nullptr) {
                out->n_layer = 0;
            }
            return GGML_SYCL_RESIDENCY_PROBE_NOT_ANSWERED;
    }
}

// The measured chunk caps of one compute buffer type: `device` is the post-selector SYCL
// device index of a device buft and the device that measured a host buft, and `cap[c]` is
// llama_measure_chunk_plan's cap for chunk c.
struct llama_tenant_buft_caps {
    int32_t             device = -1;
    bool                host   = false;
    std::vector<size_t> cap;
    size_t              max_chunk_size = 0;  // the largest chunk the buft's allocator allowed
    std::vector<size_t> chunk_bytes;         // the peak of each chunk over the measured graphs
    size_t              total = 0;           // their sum
};

// A buft's compute term from the peaks each measured graph left in each chunk (`peaks[g][c]`): the peak of
// every chunk over the graphs, and their sum. Both come from llama-measure-plan.h's one definition, so the
// caps the context hands the backend, the chunk plan and the late check read the same quantity.
inline void llama_tenant_caps_set_peaks(llama_tenant_buft_caps & c, const std::vector<std::vector<size_t>> & peaks) {
    c.chunk_bytes = llama_measure_peak_per_chunk(peaks);
    c.total       = llama_measure_peak_total(c.chunk_bytes);
}

inline bool llama_tenant_element_less(const ggml_sycl_context_tenant_desc & a,
                                      const ggml_sycl_context_tenant_desc & b) {
    if (a.device != b.device) {
        return a.device < b.device;
    }
    if (a.cohort != b.cohort) {
        return a.cohort < b.cohort;
    }
    return a.slot_index < b.slot_index;
}

// Folds `e` into `section` (kept ordered by (device, cohort, slot_index)): an element that is
// already there keeps the larger of the two slot sizes, a new one is inserted in order.
inline void llama_tenant_section_add(std::vector<ggml_sycl_context_tenant_desc> & section,
                                     const ggml_sycl_context_tenant_desc &        e) {
    auto it = std::lower_bound(section.begin(), section.end(), e, llama_tenant_element_less);
    if (it != section.end() && !llama_tenant_element_less(e, *it)) {
        it->slot_bytes = std::max(it->slot_bytes, e.slot_bytes);
        return;
    }
    section.insert(it, e);
}

// Merges the backend visitors' demands into the section by maximum.
inline void llama_tenant_section_merge(std::vector<ggml_sycl_context_tenant_desc> &       section,
                                       const std::vector<ggml_sycl_context_tenant_desc> & extra) {
    for (const auto & e : extra) {
        llama_tenant_section_add(section, e);
    }
}

// The COMPUTE and COMPUTE_HOST elements of the measured caps. A device buft's chunk c is slot
// c of COMPUTE on its device; a host buft's chunk c is slot c of COMPUTE_HOST on device -1, and
// the demands several devices measured on it merge by maximum (the host buft is one buft). A
// zero cap makes no element: there is nothing to claim. A device buft with no device index is
// refused by name, with `out` empty, rather than recorded as a host slot.
inline bool llama_tenant_section_from_caps(const std::vector<llama_tenant_buft_caps> &  bufts,
                                           std::vector<ggml_sycl_context_tenant_desc> & out,
                                           std::string &                                reason) {
    out.clear();
    reason.clear();
    for (const auto & b : bufts) {
        if (!b.host && b.device < 0) {
            char what[96];
            std::snprintf(what, sizeof(what), "a device compute buft has no device index (%d)", (int) b.device);
            reason = what;
            out.clear();
            return false;
        }
        for (size_t c = 0; c < b.cap.size(); ++c) {
            if (b.cap[c] == 0) {
                continue;
            }
            ggml_sycl_context_tenant_desc e = {};
            e.struct_size                   = sizeof(e);
            e.cohort     = b.host ? GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST : GGML_SYCL_CONTEXT_COHORT_COMPUTE;
            e.slot_index = (uint32_t) c;
            e.device     = b.host ? -1 : b.device;
            e.slot_bytes = (uint64_t) b.cap[c];
            llama_tenant_section_add(out, e);
        }
    }
    return true;
}

// The tenant key (design 2.2): a digest of each element's (device, cohort, slot_index,
// slot_bytes), in that order and in section order. Tier, zone, lifetime and scope are functions
// of the cohort and struct_size is the publisher's layout, so none of them enter it. FNV-1a, 64
// bit, over the fields as little-endian bytes so the key does not depend on the host's layout.
inline uint64_t llama_tenant_key_digest(const std::vector<ggml_sycl_context_tenant_desc> & section) {
    uint64_t h   = 1469598103934665603ull;
    auto     mix = [&h](uint64_t v, int n_bytes) {
        for (int i = 0; i < n_bytes; ++i) {
            h ^= (v >> (8 * i)) & 0xff;
            h *= 1099511628211ull;
        }
    };
    mix((uint64_t) section.size(), 8);
    for (const auto & e : section) {
        mix((uint64_t) (uint32_t) e.device, 4);
        mix((uint64_t) e.cohort, 4);
        mix((uint64_t) e.slot_index, 4);
        mix(e.slot_bytes, 8);
    }
    return h;
}

// The fields of one device's plan line (design 2.5: "The plan line's text and fields").
struct llama_tenant_plan_line_fields {
    uint32_t ctx_id        = 0;
    int32_t  device        = 0;
    uint32_t n_ubatch      = 0;
    uint64_t compute_load  = 0;
    int64_t  compute_delta = 0;
    uint64_t cap0          = 0;
    uint32_t n_measured    = 0;
    double   measure_ms    = 0.0;
    uint32_t republish     = 0;
    uint32_t covered       = 0;
};

// The head of the line a scorer matches: the tag, then `tenant plan: ctx=%u dev=%d `, then
// compute_load=. A line with the tag alone is never a plan line. The per-element, ring and
// host-tier fields follow in the caller.
inline std::string llama_tenant_plan_line(const llama_tenant_plan_line_fields & f) {
    char buf[320];
    std::snprintf(
        buf, sizeof(buf),
        "[CONTEXT-PLAN] tenant plan: ctx=%u dev=%d n_ubatch=%u compute_load=%llu compute_delta=%lld cap0=%llu "
        "n_measured=%u measure_ms=%.3f republish=%u covered=%u",
        (unsigned) f.ctx_id, (int) f.device, (unsigned) f.n_ubatch, (unsigned long long) f.compute_load,
        (long long) f.compute_delta, (unsigned long long) f.cap0, (unsigned) f.n_measured, f.measure_ms,
        (unsigned) f.republish, (unsigned) f.covered);
    return std::string(buf);
}
