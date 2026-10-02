// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// The published section of a context and the load's compute-term ledger
// (llama.cpp-moua L4 step 3).  Host-testable: it names only the public ABI structs
// (ggml-sycl.h), the cohort table (context-tenant-measure.hpp) and the standard
// library, so a test builds it with no device.
//
//   parse_runtime_context_desc   reads a ggml_sycl_runtime_context_desc once, under its
//                                struct_size and version gates, into an owning section,
//                                or refuses it by name;
//   runtime_context_tenant_key   the tenant key, the same digest llama computes;
//   classify_tenant_coverage     EQUAL / COVERED / GROWTH of a candidate against the
//                                published section, fail-closed;
//   load_compute_ledger          the term the early stage admitted per (load, device),
//                                and the late-check rule that compares a late measure
//                                with it.
//
// Nothing here is synchronized: the registry entry that holds a section is read and
// written under the registry's leaf mutex.  The ledger is a plain value keyed by (load
// transaction, device); it takes no lock, so a process-wide instance needs its owner's lock
// (a load's record and check run on the loading thread and its clear can run from load_end on
// another).

#ifndef GGML_SYCL_RUNTIME_CONTEXT_SECTION_HPP
#define GGML_SYCL_RUNTIME_CONTEXT_SECTION_HPP

#include "context-tenant-measure.hpp"
#include "ggml-sycl.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace ggml_sycl {

// The arguments of a publish or a coverage query that are not in the descriptor: the
// shape the demands depend on.  KV coverage depends on n_ctx, n_seq_max, kv_unified
// and swa_full, and the compute and tenant demand on n_ubatch and flash_attn.
struct runtime_context_geometry {
    uint32_t n_ctx      = 0;
    uint32_t n_ubatch   = 0;
    uint32_t n_seq_max  = 0;
    bool     kv_unified = false;
    bool     swa_full   = false;
    bool     flash_attn = false;

    bool operator==(const runtime_context_geometry & o) const {
        return n_ctx == o.n_ctx && n_ubatch == o.n_ubatch && n_seq_max == o.n_seq_max && kv_unified == o.kv_unified &&
               swa_full == o.swa_full && flash_attn == o.flash_attn;
    }
};

// One measured slot, by value.  The element's struct_size is the publisher's layout and
// is not kept.
struct runtime_context_tenant {
    int32_t  device     = 0;
    uint32_t cohort     = 0;
    uint32_t slot_index = 0;
    uint64_t slot_bytes = 0;
};

inline bool runtime_context_tenant_less(const runtime_context_tenant & a, const runtime_context_tenant & b) {
    if (a.device != b.device) {
        return a.device < b.device;
    }
    if (a.cohort != b.cohort) {
        return a.cohort < b.cohort;
    }
    return a.slot_index < b.slot_index;
}

// The KV and recurrent-state sections as published, by value.
struct runtime_context_kv_shape {
    int32_t                              type_k   = 0;
    int32_t                              type_v   = 0;
    bool                                 v_trans  = false;
    bool                                 no_alloc = false;
    bool                                 sidecar  = false;
    uint32_t                             n_stream = 0;
    std::vector<ggml_sycl_kv_layer_desc> layers;
    std::vector<ggml_sycl_rs_layer_desc> rs_layers;
};

// What one publish carries, validated and owned: the descriptor is not retained after the
// call, so this is the only copy.  tenants is in (device, cohort, slot_index) order.
// tenants_planned is derived at parse time from the tenant section, here and nowhere
// else: the descriptor carries zhcn's section only for a context its MEASURE planned, so
// a non-empty section is what "planned" means.
struct runtime_context_section {
    runtime_context_geometry            geometry;
    runtime_context_kv_shape            kv;
    std::vector<runtime_context_tenant> tenants;
    uint64_t                            tenant_key      = 0;
    bool                                tenants_planned = false;
};

enum class runtime_context_desc_status : uint8_t {
    OK = 0,
    NULL_DESC,
    SHORT_STRUCT,     // struct_size below the version-1 layout
    UNKNOWN_VERSION,  // a version this reader does not know
    BAD_PAD,          // pad0 is not 0
    BAD_FLAG,         // a flag byte other than 0 or 1
    BAD_ARRAY,        // a count with no array, a stride below the element, or a count above the cap
    BAD_ELEMENT,      // an element whose own struct_size is below the element layout
    UNKNOWN_COHORT,   // a cohort id the table does not know
    BAD_DEVICE,       // a device-tier element off the device range, or a host-tier one not on -1
    ZERO_SLOT,        // an element with no bytes: there is nothing to carve or to claim
    DUPLICATE_SLOT,   // two elements at one (device, cohort, slot_index)
};

inline const char * runtime_context_desc_status_text(runtime_context_desc_status s) {
    switch (s) {
        case runtime_context_desc_status::OK:
            return "ok";
        case runtime_context_desc_status::NULL_DESC:
            return "null descriptor";
        case runtime_context_desc_status::SHORT_STRUCT:
            return "descriptor struct_size is below the version-1 layout";
        case runtime_context_desc_status::UNKNOWN_VERSION:
            return "descriptor version is not known to this backend";
        case runtime_context_desc_status::BAD_PAD:
            return "descriptor pad0 is not 0";
        case runtime_context_desc_status::BAD_FLAG:
            return "descriptor flag byte is not 0 or 1";
        case runtime_context_desc_status::BAD_ARRAY:
            return "descriptor array is missing, strided below its element, or over the cap";
        case runtime_context_desc_status::BAD_ELEMENT:
            return "descriptor element struct_size is below its layout";
        case runtime_context_desc_status::UNKNOWN_COHORT:
            return "tenant element names a cohort the table does not know";
        case runtime_context_desc_status::BAD_DEVICE:
            return "tenant element's device does not match its cohort's tier";
        case runtime_context_desc_status::ZERO_SLOT:
            return "tenant element has no bytes";
        case runtime_context_desc_status::DUPLICATE_SLOT:
            return "two tenant elements share one (device, cohort, slot_index)";
    }
    return "unknown";
}

// An array count above this is refused before any element is read: no real model has more
// layers, and the cap keeps a corrupt count from turning into a long read.
constexpr uint32_t RUNTIME_CONTEXT_DESC_MAX_ELEMENTS = 1u << 16;

// The tenant key (zhcn design 2.2): a digest of each element's (device, cohort, slot_index,
// slot_bytes), in that order and in section order.  FNV-1a, 64 bit, over the fields as
// little-endian bytes.  llama computes the same digest for its matched-key path
// (llama_tenant_key_digest); tests/test-runtime-context-section.cpp pins the two equal, since
// they are one fact with two readers.
inline uint64_t runtime_context_tenant_key(const std::vector<runtime_context_tenant> & section) {
    uint64_t h   = 1469598103934665603ull;
    auto     mix = [&h](uint64_t v, int n_bytes) {
        for (int i = 0; i < n_bytes; ++i) {
            h ^= (v >> (8 * i)) & 0xff;
            h *= 1099511628211ull;
        }
    };
    mix((uint64_t) section.size(), 8);
    for (const runtime_context_tenant & e : section) {
        mix((uint64_t) (uint32_t) e.device, 4);
        mix((uint64_t) e.cohort, 4);
        mix((uint64_t) e.slot_index, 4);
        mix(e.slot_bytes, 8);
    }
    return h;
}

namespace runtime_context_detail {

// The element at index i of an array the publisher laid out at its own stride.
inline const unsigned char * element_at(const void * base, uint32_t stride, uint32_t i) {
    return static_cast<const unsigned char *>(base) + (size_t) stride * i;
}

inline bool array_ok(const void * base, uint32_t count, uint32_t stride, size_t element_size) {
    if (count == 0) {
        return true;
    }
    return base != nullptr && count <= RUNTIME_CONTEXT_DESC_MAX_ELEMENTS && stride >= element_size;
}

}  // namespace runtime_context_detail

// Reads `desc` once into `out`.  `n_devices` is the number of SYCL devices a device-tier
// element may name.  On any refusal `out` is left empty and the status says which gate.
//
// The gates, in order: null; struct_size below the version-1 layout (a reader treats a field
// beyond the publisher's struct_size as absent, and version 1 has no absent field); an unknown
// version; pad0; the flag bytes; each array's count, stride and base; then every element
// under its own struct_size and the cohort table.  A struct_size above this build's is read
// as far as this build's layout goes.
inline runtime_context_desc_status parse_runtime_context_desc(const ggml_sycl_runtime_context_desc * desc,
                                                              const runtime_context_geometry &       geometry,
                                                              int                                    n_devices,
                                                              runtime_context_section &              out) {
    using status = runtime_context_desc_status;
    out          = runtime_context_section{};
    if (desc == nullptr) {
        return status::NULL_DESC;
    }
    if (desc->struct_size < sizeof(ggml_sycl_runtime_context_desc)) {
        return status::SHORT_STRUCT;
    }
    if (desc->version != GGML_SYCL_RUNTIME_CONTEXT_DESC_VERSION) {
        return status::UNKNOWN_VERSION;
    }
    if (desc->pad0 != 0) {
        return status::BAD_PAD;
    }
    if (desc->v_trans > 1 || desc->no_alloc > 1 || desc->sidecar > 1) {
        return status::BAD_FLAG;
    }
    if (!runtime_context_detail::array_ok(desc->layers, desc->n_layer, desc->layer_desc_size,
                                          sizeof(ggml_sycl_kv_layer_desc)) ||
        !runtime_context_detail::array_ok(desc->tenants, desc->n_tenants, desc->tenant_desc_size,
                                          sizeof(ggml_sycl_context_tenant_desc)) ||
        !runtime_context_detail::array_ok(desc->rs_layers, desc->n_rs_layer, desc->rs_layer_desc_size,
                                          sizeof(ggml_sycl_rs_layer_desc))) {
        return status::BAD_ARRAY;
    }

    runtime_context_section s;
    s.geometry    = geometry;
    s.kv.type_k   = desc->type_k;
    s.kv.type_v   = desc->type_v;
    s.kv.v_trans  = desc->v_trans != 0;
    s.kv.no_alloc = desc->no_alloc != 0;
    s.kv.sidecar  = desc->sidecar != 0;
    s.kv.n_stream = desc->n_stream;
    for (uint32_t i = 0; i < desc->n_layer; ++i) {
        ggml_sycl_kv_layer_desc l;
        std::memcpy(&l, runtime_context_detail::element_at(desc->layers, desc->layer_desc_size, i), sizeof(l));
        s.kv.layers.push_back(l);
    }
    for (uint32_t i = 0; i < desc->n_rs_layer; ++i) {
        ggml_sycl_rs_layer_desc r;
        std::memcpy(&r, runtime_context_detail::element_at(desc->rs_layers, desc->rs_layer_desc_size, i), sizeof(r));
        s.kv.rs_layers.push_back(r);
    }
    for (uint32_t i = 0; i < desc->n_tenants; ++i) {
        ggml_sycl_context_tenant_desc e;
        std::memcpy(&e, runtime_context_detail::element_at(desc->tenants, desc->tenant_desc_size, i), sizeof(e));
        if (e.struct_size < sizeof(ggml_sycl_context_tenant_desc)) {
            return status::BAD_ELEMENT;
        }
        const ggml_sycl_context_cohort_info * info = ggml_sycl_context_cohort_lookup(e.cohort);
        if (info == nullptr) {
            return status::UNKNOWN_COHORT;
        }
        if (info->tier == GGML_SYCL_CONTEXT_COHORT_TIER_HOST_PINNED ? e.device != -1 :
                                                                      (e.device < 0 || e.device >= n_devices)) {
            return status::BAD_DEVICE;
        }
        if (e.slot_bytes == 0) {
            return status::ZERO_SLOT;
        }
        s.tenants.push_back({ e.device, e.cohort, e.slot_index, e.slot_bytes });
    }
    std::sort(s.tenants.begin(), s.tenants.end(), runtime_context_tenant_less);
    for (size_t i = 1; i < s.tenants.size(); ++i) {
        if (!runtime_context_tenant_less(s.tenants[i - 1], s.tenants[i])) {
            return status::DUPLICATE_SLOT;
        }
    }
    s.tenant_key      = runtime_context_tenant_key(s.tenants);
    s.tenants_planned = !s.tenants.empty();
    out               = std::move(s);
    return status::OK;
}

// The elements are compared member by member: a publisher copies a whole element, padding
// included (ggml_sycl_kv_layer_desc ends in two padding bytes), and equal values with different
// padding bytes are the same shape.
inline bool runtime_context_layer_equal(const ggml_sycl_kv_layer_desc & a, const ggml_sycl_kv_layer_desc & b) {
    return a.n_embd_k_gqa == b.n_embd_k_gqa && a.n_embd_v_gqa == b.n_embd_v_gqa && a.n_head_kv == b.n_head_kv &&
           a.n_embd_head_k == b.n_embd_head_k && a.has_kv == b.has_kv && a.is_swa == b.is_swa;
}

inline bool runtime_context_layer_equal(const ggml_sycl_rs_layer_desc & a, const ggml_sycl_rs_layer_desc & b) {
    return a.il == b.il && a.type_r == b.type_r && a.type_s == b.type_s && a.n_embd_r == b.n_embd_r &&
           a.n_embd_s == b.n_embd_s && a.n_rows == b.n_rows;
}

// The KV and recurrent-state sections of two publishes, value for value.
inline bool runtime_context_kv_shape_equal(const runtime_context_kv_shape & a, const runtime_context_kv_shape & b) {
    if (a.type_k != b.type_k || a.type_v != b.type_v || a.v_trans != b.v_trans || a.no_alloc != b.no_alloc ||
        a.sidecar != b.sidecar || a.n_stream != b.n_stream || a.layers.size() != b.layers.size() ||
        a.rs_layers.size() != b.rs_layers.size()) {
        return false;
    }
    for (size_t i = 0; i < a.layers.size(); ++i) {
        if (!runtime_context_layer_equal(a.layers[i], b.layers[i])) {
            return false;
        }
    }
    for (size_t i = 0; i < a.rs_layers.size(); ++i) {
        if (!runtime_context_layer_equal(a.rs_layers[i], b.rs_layers[i])) {
            return false;
        }
    }
    return true;
}

// Would publishing `candidate` need a transaction, given what `published` already holds?
// Fail-closed: GROWTH is the answer to everything not proved otherwise, a null published
// section included.
//
// COVERED only when the shape demands no more than the published one and every candidate slot
// has a published slot at the same (device, cohort, slot_index) with at least its bytes.  The
// shape demands no more when the KV and recurrent-state sections are the same, kv_unified,
// swa_full and flash_attn are the same (a flip changes the cell count or adds the non-FA staging
// in a direction this reader does not model), the plan state (tenants_planned) is the same,
// and each of n_ctx, n_ubatch and n_seq_max is at most the published value.  A candidate with
// a zero n_ctx, n_ubatch or n_seq_max has no shape to compare (the ledger refuses a zero n_ctx
// for the same reason), so it is GROWTH, never coverage.
// EQUAL only when the geometry, the shape and every slot are byte-equal.
inline ggml_sycl_tenant_coverage classify_tenant_coverage(const runtime_context_section * published,
                                                          const runtime_context_section & candidate) {
    if (published == nullptr) {
        return GGML_SYCL_TENANT_COVERAGE_GROWTH;
    }
    const runtime_context_geometry & p = published->geometry;
    const runtime_context_geometry & c = candidate.geometry;
    if (c.n_ctx == 0 || c.n_ubatch == 0 || c.n_seq_max == 0 ||
        !runtime_context_kv_shape_equal(published->kv, candidate.kv) ||
        published->tenants_planned != candidate.tenants_planned || c.kv_unified != p.kv_unified ||
        c.swa_full != p.swa_full || c.flash_attn != p.flash_attn || c.n_ctx > p.n_ctx || c.n_ubatch > p.n_ubatch ||
        c.n_seq_max > p.n_seq_max) {
        return GGML_SYCL_TENANT_COVERAGE_GROWTH;
    }
    for (const runtime_context_tenant & e : candidate.tenants) {
        auto it =
            std::lower_bound(published->tenants.begin(), published->tenants.end(), e, runtime_context_tenant_less);
        if (it == published->tenants.end() || runtime_context_tenant_less(e, *it) || it->slot_bytes < e.slot_bytes) {
            return GGML_SYCL_TENANT_COVERAGE_GROWTH;
        }
    }
    // Both tenant lists are sorted and duplicate-free, and every candidate slot found a published
    // slot at its own (device, cohort, slot_index) above, so with equal counts the i-th slots
    // share a key: only the bytes can still differ.
    bool equal = c == p && candidate.tenants.size() == published->tenants.size();
    for (size_t i = 0; equal && i < candidate.tenants.size(); ++i) {
        equal = candidate.tenants[i].slot_bytes == published->tenants[i].slot_bytes;
    }
    return equal ? GGML_SYCL_TENANT_COVERAGE_EQUAL : GGML_SYCL_TENANT_COVERAGE_COVERED;
}

// The level a ledger line is for.  INFO is dropped at default verbosity in every tool, so a line a caller
// must see in a normal run is WARN or ERROR, and the ledger picks the level so a host test can pin it.
enum load_log_level {
    LOAD_LOG_LEVEL_NONE  = 0,
    LOAD_LOG_LEVEL_INFO  = 1,
    LOAD_LOG_LEVEL_WARN  = 2,
    LOAD_LOG_LEVEL_ERROR = 3,
};

// The compute term the early stage admitted, per (load transaction, device): c(P) of zhcn's
// measure (zhcn design 2.10, call site (b)).  The late check (call site (c)) compares the
// late measure with it under the one late-check rule moua and 23mk share.
//
// The term is recorded only when the load carries an n_ctx.  An envelope with n_ctx == 0 has
// no candidate shape to measure, so a value recorded for it would be a guess; the record is
// refused and the late check answers NOT_RECORDED, which no caller reads as a pass.
//
// The strings are canonical (moua design 2.4.2 (b), step 3; ruling Z13.1); zhcn and
// 23mk mirror them by citation.
class load_compute_ledger {
  public:
    static constexpr const char * TERM = "compute";

    // The zone the compute term names in the refusal line: the cohort the term is measured for, from the
    // one cohort table (context-tenant-measure.cpp, "context-compute").  vram_zone_name() in unified-cache.cpp
    // has no zone for it (KV, WEIGHT, ONEDNN, RUNTIME, SCRATCH) and the design (sycl-kv-region-segregation.md,
    // late-check, term `compute`) names the term only, so the cohort name is the one existing authority.  The
    // lookup of an enumerator below GGML_SYCL_CONTEXT_COHORT_COUNT cannot fail, and the test pins the name.
    static const char * zone() { return ggml_sycl_context_cohort_lookup(GGML_SYCL_CONTEXT_COHORT_COMPUTE)->name; }

    struct check_result {
        ggml_sycl_late_check_result result = GGML_SYCL_LATE_CHECK_NOT_RECORDED;
        std::string                 line;                         // the line to log, empty for none
        load_log_level              level = LOAD_LOG_LEVEL_NONE;  // the level of `line`; NONE exactly when it is empty
        bool shrink_counted               = false;  // an admitted shrink: the line is the WARN and the counter takes +1
    };

    // Records c(P) = `bytes` for (txn, device).  False, recording nothing, when `n_ctx` is 0, `txn` is 0, or
    // `txn_is_open` is false: `txn_is_open` is whether `txn` is the open load transaction, which the caller reads
    // under the lock that guards this ledger, so a record cannot land after the clear of a load that ended.  A
    // second record of the same key replaces the first: the early stage can stage a candidate again, and the late
    // check reads the admitted one.
    bool record(uint64_t txn, int32_t device, uint64_t bytes, uint32_t n_ctx, bool txn_is_open) {
        if (n_ctx == 0 || txn == 0 || device < 0 || !txn_is_open) {
            return false;
        }
        terms_[key{ txn, device }].admitted = bytes;
        return true;
    }

    // The late check of (txn, device) against `late_bytes`.  `txn_is_open` is whether `txn` is the
    // open load transaction, which the caller reads from the lifecycle registry under the lock that guards
    // this ledger.
    //   not open                                    NOT_RECORDED, WARN on every call (a transaction
    //                                               that is no load is a caller defect)
    //   open, nothing recorded for the key          NOT_RECORDED, INFO ONCE per (load, device): this is the
    //                                               expected answer until L6 records anything, a load that
    //                                               records nothing asks every device once, and the line
    //                                               would repeat per call
    //   late == admitted                            EQUAL, no line
    //   late  > admitted                            REFUSED, the late string, ERROR
    //   late  < admitted                            SHRINK_ADMITTED, the shrink WARN once per
    //                                               (load, device, term); the admitted term stands
    check_result check(uint64_t txn, int32_t device, uint64_t late_bytes, bool txn_is_open) {
        check_result r;
        char         line[320];
        if (!txn_is_open) {
            std::snprintf(
                line, sizeof(line),
                "[LOAD-PLAN] late check on device %d: transaction %llu is not the open load transaction, nothing was "
                "compared",
                (int) device, (unsigned long long) txn);
            r.line  = line;
            r.level = LOAD_LOG_LEVEL_WARN;
            return r;
        }
        auto it = terms_.find(key{ txn, device });
        if (it == terms_.end()) {
            if (!unrecorded_logged_.insert(key{ txn, device }).second) {
                return r;  // already said once for this (load, device)
            }
            std::snprintf(
                line, sizeof(line),
                "[LOAD-PLAN] late check on device %d: no early term was recorded for transaction %llu, nothing was "
                "compared",
                (int) device, (unsigned long long) txn);
            r.line  = line;
            r.level = LOAD_LOG_LEVEL_INFO;
            return r;
        }
        entry & e = it->second;
        if (late_bytes > e.admitted) {
            std::snprintf(
                line, sizeof(line),
                "[LOAD-PLAN] the late inventory changes the zones admitted at the early stage: term %s in zone %s on "
                "device %d, early %zu B, late %zu B (refused)",
                TERM, zone(), (int) device, (size_t) e.admitted, (size_t) late_bytes);
            r.result = GGML_SYCL_LATE_CHECK_REFUSED;
            r.line   = line;
            r.level  = LOAD_LOG_LEVEL_ERROR;
            return r;
        }
        if (late_bytes < e.admitted) {
            r.result = GGML_SYCL_LATE_CHECK_SHRINK_ADMITTED;
            if (!e.shrink_logged) {
                e.shrink_logged = true;
                std::snprintf(
                    line, sizeof(line),
                    "[ZONE-PLAN-BUG] the late inventory shrinks term %s on device %d: early %zu B, late %zu B "
                    "(admitted; the early reservation stands)",
                    TERM, (int) device, (size_t) e.admitted, (size_t) late_bytes);
                r.line           = line;
                r.level          = LOAD_LOG_LEVEL_WARN;
                r.shrink_counted = true;
            }
            return r;
        }
        r.result = GGML_SYCL_LATE_CHECK_EQUAL;
        return r;
    }

    // The load's commit or rollback drops its terms.  Returns how many it dropped.
    size_t clear(uint64_t txn) {
        size_t dropped = 0;
        for (auto it = unrecorded_logged_.begin(); it != unrecorded_logged_.end();) {
            it = it->txn == txn ? unrecorded_logged_.erase(it) : std::next(it);
        }
        for (auto it = terms_.begin(); it != terms_.end();) {
            if (it->first.txn == txn) {
                it = terms_.erase(it);
                ++dropped;
            } else {
                ++it;
            }
        }
        return dropped;
    }

    size_t size() const { return terms_.size(); }

  private:
    struct key {
        uint64_t txn;
        int32_t  device;

        bool operator<(const key & o) const { return txn != o.txn ? txn < o.txn : device < o.device; }
    };

    struct entry {
        uint64_t admitted      = 0;
        bool     shrink_logged = false;
    };

    std::map<key, entry> terms_;
    std::set<key>        unrecorded_logged_;  // the (load, device) pairs whose NOT_RECORDED line was given
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_RUNTIME_CONTEXT_SECTION_HPP
