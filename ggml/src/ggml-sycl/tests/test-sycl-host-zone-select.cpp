//
// test-sycl-host-zone-select: select_zone over role x category x kv_spill_to_host, and the lifetime checks over a fake
// registry (docs/design/sycl-mmap-host-tier.md §4.4 "Host-side non-weight requests: routed by role" and "Lifetimes",
// §M57 (b), §M62 (D)). Host-only: no unified_cache, no SYCL.
//
// Positive controls. Each is planted by one compile-time macro and has its own ctest target, which scores the failure
// text this file prints (ggml/src/ggml-sycl/CMakeLists.txt):
//   HOST_ZONE_PLAN_MUTANT_OWNER_BY_LIFETIME        (host-zone-plan.cpp) the owner check keyed by lifetime alone
//   HOST_ZONE_PLAN_MUTANT_DEVICE_CHECK_EVERY_FREE  (host-zone-plan.cpp) the DEVICE check at every backend free
//   HOST_ZONE_SELECT_MUTANT_BUFFER_CHECK_AT_BACKEND (this file) the buffer-owned check moved to backend destruction
//   HOST_ZONE_SELECT_MUTANT_MODEL_CHECK_AT_BUFFER_FREE (this file) the MODEL check at the first weight buffer's free
// The two in this file move a check in the fake registry's order, so they score the fixture's model of the order, not
// production. Where production runs each owner check is pinned by tests/test-sycl-host-zone-boundary-owner-source.py
// (O1-O6).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "host-zone-plan-test-util.hpp"
#include "host-zone-plan.hpp"

#include <vector>

using namespace ggml_sycl;
using namespace host_zone_plan_test;

namespace {

const char * zone_name(host_zone_id z) {
    switch (z) {
        case host_zone_id::WEIGHT:
            return "WEIGHT";
        case host_zone_id::KV:
            return "KV";
        case host_zone_id::STAGING:
            return "STAGING";
        case host_zone_id::SCRATCH:
            return "SCRATCH";
        default:
            return "?";
    }
}

struct route_row {
    const char *     what;
    alloc_role       role;
    runtime_category category;
    alloc_lifetime   lifetime;
    bool             persistent;
    bool             spill;
    host_zone_id     want;
};

void test_routing_rows() {
    // One row per rule of §4.4's routing, in rule order, plus the two buft intents the design names. R, C, L and Z are
    // alloc_role, runtime_category, alloc_lifetime and host_zone_id.
    using R                = alloc_role;
    using C                = runtime_category;
    using L                = alloc_lifetime;
    using Z                = host_zone_id;
    const route_row rows[] = {
        { "role KV",                        R::KV,             C::HOST_COMPUTE, L::CONTEXT, false, false, Z::KV      },
        { "device-KV spill to host",        R::OTHER,          C::KV_CACHE,     L::GRAPH,   false, true,  Z::KV      },
        { "role WEIGHT",                    R::WEIGHT,         C::OTHER,        L::MODEL,   false, false, Z::WEIGHT  },
        { "role EXPERT_STAGING",            R::EXPERT_STAGING, C::HOST_COMPUTE, L::GRAPH,   false, false, Z::SCRATCH },
        { "lifetime UNOWNED",               R::COMPUTE,        C::HOST_COMPUTE, L::UNOWNED, false, false, Z::WEIGHT  },
        { "declared persistent owner",      R::STAGING,        C::OTHER,        L::CONTEXT, true,  false, Z::WEIGHT  },
        { "category HOST_COMPUTE",          R::COMPUTE,        C::HOST_COMPUTE, L::CONTEXT, false, false, Z::SCRATCH },
        { "category EXPERT_CACHE",          R::COMPUTE,        C::EXPERT_CACHE, L::MODEL,   false, false, Z::SCRATCH },
        { "role STAGING, category STAGING", R::STAGING,        C::STAGING,      L::GRAPH,   false, false, Z::STAGING },
        { "role COMPUTE, category OTHER",   R::COMPUTE,        C::OTHER,        L::GRAPH,   false, false, Z::STAGING },
        // The SYCL_KV_Host buft's intent: role KV, HOST_COMPUTE, CONTEXT.
        { "SYCL_KV_Host intent",            R::KV,             C::HOST_COMPUTE, L::CONTEXT, false, false, Z::KV      },
        // The post-load SYCL_Host buft's intent: role COMPUTE (WEIGHT only under in_model_load), HOST_COMPUTE, CONTEXT.
        { "post-load SYCL_Host intent",     R::COMPUTE,        C::HOST_COMPUTE, L::CONTEXT, false, false, Z::SCRATCH },
    };
    for (const route_row & r : rows) {
        const host_zone_id got = select_zone(r.role, r.category, r.lifetime, r.persistent, r.spill);
        HZP_CHECK(got == r.want, "%s selects zone %s, expected %s", r.what, zone_name(got), zone_name(r.want));
    }
}

void test_no_category_selects_weight() {
    // No category clause selects WEIGHT: for a non-WEIGHT role with a GRAPH, CONTEXT, MODEL or DEVICE lifetime and no
    // persistent owner, no category reaches WEIGHT.
    const alloc_role     roles[]     = { alloc_role::COMPUTE, alloc_role::STAGING, alloc_role::GRAPH_TMP,
                                         alloc_role::TP_TMP,  alloc_role::CONTROL, alloc_role::OTHER };
    const alloc_lifetime lifetimes[] = { alloc_lifetime::GRAPH, alloc_lifetime::CONTEXT, alloc_lifetime::MODEL,
                                         alloc_lifetime::DEVICE };
    for (alloc_role role : roles) {
        for (alloc_lifetime lt : lifetimes) {
            for (int c = 0; c < (int) runtime_category::COUNT; ++c) {
                const host_zone_id got = select_zone(role, (runtime_category) c, lt, false, false);
                HZP_CHECK(got != host_zone_id::WEIGHT, "role %d category %d lifetime %s selects WEIGHT by category",
                          (int) role, c, alloc_lifetime_name(lt));
            }
        }
    }
}

void test_graph_boundary() {
    using Z                                 = host_zone_id;
    using L                                 = alloc_lifetime;
    using K                                 = alloc_owner_kind;
    const host_zone_record_view by_design[] = {
        { Z::SCRATCH, L::CONTEXT, { K::BUFFER, 1 }, "sycl_host_buffer",        64 },
        { Z::SCRATCH, L::MODEL,   { K::MODEL, 2 },  "moe_hybrid_decode_table", 64 },
        { Z::SCRATCH, L::DEVICE,  { K::DEVICE, 0 }, "pinned_buffer_pool",      64 },
        { Z::STAGING, L::UNOWNED, {},               "tp_staging",              64 },
    };
    HZP_CHECK(host_zone_graph_boundary_live(by_design, 4, host_zone_id::SCRATCH) == 0,
              "a CONTEXT, MODEL or DEVICE record trips the SCRATCH graph-boundary check (%zu counted)",
              host_zone_graph_boundary_live(by_design, 4, host_zone_id::SCRATCH));
    HZP_CHECK(host_zone_graph_boundary_live(by_design, 4, host_zone_id::STAGING) == 0,
              "an UNOWNED record trips the STAGING graph-boundary check");
    const host_zone_record_view graph_row[] = {
        { host_zone_id::SCRATCH, alloc_lifetime::GRAPH, {}, "expert_staging", 64 },
    };
    HZP_CHECK(host_zone_graph_boundary_live(graph_row, 1, host_zone_id::SCRATCH) == 1,
              "a GRAPH record does not trip the SCRATCH graph-boundary check");
}

// ---------------------------------------------------------------------------------------------------------------
// Owner boundaries.
// ---------------------------------------------------------------------------------------------------------------

// A fake registry: the rows still live, and the owner-boundary check the backend runs at each boundary.
struct fake_registry {
    std::vector<host_zone_record_view> rows;

    void add(host_zone_id zone, alloc_lifetime lt, alloc_owner_ref owner, const char * cohort, size_t bytes) {
        rows.push_back({ zone, lt, owner, cohort, bytes });
    }

    void release(alloc_owner_ref owner) {
        for (size_t i = 0; i < rows.size();) {
            if (rows[i].owner.kind == owner.kind && rows[i].owner.id == owner.id) {
                rows.erase(rows.begin() + (long) i);
            } else {
                ++i;
            }
        }
    }

    // Releases the one row of `owner` that is `bytes` long: one extra's allocation, where several share an owner.
    void release_row(alloc_owner_ref owner, size_t bytes) {
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].owner.kind == owner.kind && rows[i].owner.id == owner.id && rows[i].bytes == bytes) {
                rows.erase(rows.begin() + (long) i);
                return;
            }
        }
    }

    size_t check(alloc_lifetime lt, alloc_owner_ref owner, bool strict = false) const {
        return host_zone_owner_boundary_check(rows.data(), rows.size(), lt, owner, strict);
    }
};

const alloc_owner_ref buffer_7{ alloc_owner_kind::BUFFER, 7 };
const alloc_owner_ref buffer_8{ alloc_owner_kind::BUFFER, 8 };
const alloc_owner_ref backend_9{ alloc_owner_kind::BACKEND, 9 };
const alloc_owner_ref model_3{ alloc_owner_kind::MODEL, 3 };
const alloc_owner_ref model_4{ alloc_owner_kind::MODEL, 4 };
const alloc_owner_ref device_0{ alloc_owner_kind::DEVICE, 0 };

void test_owner_leak_warns() {
    {
        // A buffer-owned CONTEXT record still live after its buffer's free_buffer.
        fake_registry reg;
        reg.add(host_zone_id::SCRATCH, alloc_lifetime::CONTEXT, buffer_7, "sycl_host_buffer", 4096);
        capture_begin();
        const size_t n = reg.check(alloc_lifetime::CONTEXT, buffer_7);
        HZP_CHECK(n == 1 && log_has("[HOST-ZONE] leaked CONTEXT record: sycl_host_buffer 4096 owner=buffer/7"),
                  "a buffer-owned CONTEXT record live after free_buffer printed no leaked-record WARN (%zu)", n);
    }
    {
        // A backend-member record live after `delete sycl_ctx`.
        fake_registry reg;
        reg.add(host_zone_id::STAGING, alloc_lifetime::CONTEXT, backend_9, "context_readback_staging", 8192);
        capture_begin();
        const size_t n = reg.check(alloc_lifetime::CONTEXT, backend_9);
        HZP_CHECK(n == 1 && log_has("[HOST-ZONE] leaked CONTEXT record: context_readback_staging 8192 owner=backend/9"),
                  "a backend-member record live after delete sycl_ctx printed no leaked-record WARN (%zu)", n);
    }
    {
        // A MODEL record live at its model's free.
        fake_registry reg;
        reg.add(host_zone_id::SCRATCH, alloc_lifetime::MODEL, model_3, "moe_hybrid_decode_table", 1024);
        capture_begin();
        const size_t n = reg.check(alloc_lifetime::MODEL, model_3);
        HZP_CHECK(n == 1 && log_has("[HOST-ZONE] leaked MODEL record: moe_hybrid_decode_table 1024 owner=model/3"),
                  "a MODEL record live at its model's free printed no leaked-record WARN (%zu)", n);
    }
}

void test_device_last_backend() {
    // Two live backends on device 0 share a DEVICE holder. The first free is not the last: no check. The second is.
    fake_registry reg;
    reg.add(host_zone_id::SCRATCH, alloc_lifetime::DEVICE, device_0, "pinned_buffer_pool", 2048);
    int live = 2;
    capture_begin();
    --live;  // the first backend's `delete sycl_ctx`
    size_t fired_early = 0;
    if (host_zone_device_check_due(live)) {
        fired_early = reg.check(alloc_lifetime::DEVICE, device_0);
    }
    HZP_CHECK(fired_early == 0,
              "the DEVICE check fired at a non-last backend's free while a second backend is live (%zu "
              "leaked)",
              fired_early);
    --live;  // the last backend's
    size_t fired_last = 0;
    if (host_zone_device_check_due(live)) {
        fired_last = reg.check(alloc_lifetime::DEVICE, device_0);
    }
    HZP_CHECK(fired_last == 1 && log_has("[HOST-ZONE] leaked DEVICE record: pinned_buffer_pool 2048 owner=device/0"),
              "a DEVICE record live after the last backend's free printed no leaked-record WARN (%zu)", fired_last);
}

void test_destruction_order() {
    // A fake llama_context: buf_output, a backend member, the scheduler's compute buffer and the memory's KV buffer.
    // It frees them in llama_context's destructor order: buf_output, then the backends, then the compute buffers, then
    // the KV buffers. Each owner's record is released by its owner before that owner's boundary check runs.
    const alloc_owner_ref buf_output{ alloc_owner_kind::BUFFER, 101 };
    const alloc_owner_ref compute{ alloc_owner_kind::BUFFER, 102 };
    const alloc_owner_ref kv{ alloc_owner_kind::BUFFER, 103 };
    const alloc_owner_ref backend{ alloc_owner_kind::BACKEND, 201 };
    fake_registry         reg;
    reg.add(host_zone_id::SCRATCH, alloc_lifetime::CONTEXT, buf_output, "sycl_host_buffer", 512);
    reg.add(host_zone_id::STAGING, alloc_lifetime::CONTEXT, backend, "context_readback_staging", 512);
    reg.add(host_zone_id::SCRATCH, alloc_lifetime::CONTEXT, compute, "sycl_host_buffer", 4096);
    reg.add(host_zone_id::KV, alloc_lifetime::CONTEXT, kv, "kv_host_buffer", 8192);

    capture_begin();
    size_t leaked = 0;
    reg.release(buf_output);  // buf_output's free_buffer
    leaked += reg.check(alloc_lifetime::CONTEXT, buf_output);
    reg.release(backend);     // `delete sycl_ctx`
    leaked += reg.check(alloc_lifetime::CONTEXT, backend);
#if defined(HOST_ZONE_SELECT_MUTANT_BUFFER_CHECK_AT_BACKEND)
    // Positive control: the buffer-owned check taken at backend-context destruction, while the compute and KV buffers
    // are still live. It moves the check in this fixture's order only; O6 of the boundary-owner pin scores production.
    leaked += reg.check(alloc_lifetime::CONTEXT, compute);
    leaked += reg.check(alloc_lifetime::CONTEXT, kv);
#endif
    reg.release(compute);  // the scheduler's compute buffers
    leaked += reg.check(alloc_lifetime::CONTEXT, compute);
    reg.release(kv);       // the memory's KV buffers
    leaked += reg.check(alloc_lifetime::CONTEXT, kv);
    HZP_CHECK(leaked == 0 && !log_has("[HOST-ZONE] leaked"),
              "llama_context's destruction order printed %zu leaked-record WARN(s)", leaked);
}

void test_other_owner_does_not_fire() {
    // The multi-model case: a live record of a different context or model is not this owner's.
    fake_registry reg;
    reg.add(host_zone_id::SCRATCH, alloc_lifetime::CONTEXT, buffer_8, "sycl_host_buffer", 4096);
    reg.add(host_zone_id::SCRATCH, alloc_lifetime::MODEL, model_4, "moe_hybrid_decode_table", 1024);
    capture_begin();
    const size_t ctx_n   = reg.check(alloc_lifetime::CONTEXT, buffer_7);
    const size_t model_n = reg.check(alloc_lifetime::MODEL, model_3);
    HZP_CHECK(ctx_n == 0 && model_n == 0,
              "the owner check fired on another context's or model's live record (%zu CONTEXT, %zu MODEL)", ctx_n,
              model_n);
}

// The MODEL boundary (§4.4 "Lifetimes"): a decode table is owned by its weight's model id, and its extra is released
// in one of two places: a device weight buffer's free_buffer releases its own tensors' extras, and the model's slot
// release releases the host-registered weight extras. A model's weights span several weight buffers, so the check
// runs once, at the slot release, after both.
void test_model_boundary_at_slot_release() {
    const alloc_owner_ref model_5{ alloc_owner_kind::MODEL, 5 };
    const alloc_owner_ref model_6{ alloc_owner_kind::MODEL, 6 };
    {
        fake_registry reg;
        reg.add(host_zone_id::SCRATCH, alloc_lifetime::MODEL, model_5, "moe_hybrid_decode_table", 1024);  // buffer 1
        reg.add(host_zone_id::SCRATCH, alloc_lifetime::MODEL, model_5, "moe_hybrid_decode_table", 2048);  // buffer 2
        reg.add(host_zone_id::SCRATCH, alloc_lifetime::MODEL, model_5, "moe_hybrid_decode_table", 3072);  // host extra
        reg.add(host_zone_id::SCRATCH, alloc_lifetime::MODEL, model_6, "moe_hybrid_decode_table", 4096);  // live model
        capture_begin();
        size_t leaked = 0;
        reg.release_row(model_5, 1024);  // weight buffer 1's free_buffer releases its extras
#if defined(HOST_ZONE_SELECT_MUTANT_MODEL_CHECK_AT_BUFFER_FREE)
        // Positive control: the check at the first weight buffer's free, keyed by the model id its siblings share.
        // It moves the check in this fixture's order only; O6 of the boundary-owner pin scores production.
        leaked += reg.check(alloc_lifetime::MODEL, model_5);
#endif
        reg.release_row(model_5, 2048);                       // weight buffer 2's
        reg.release_row(model_5, 3072);                       // the slot release's host-registered weight extras
        leaked += reg.check(alloc_lifetime::MODEL, model_5);  // the boundary
        HZP_CHECK(leaked == 0 && !log_has("[HOST-ZONE] leaked"),
                  "the MODEL boundary printed %zu leaked-record WARN(s) for a model whose rows all went with its "
                  "extras",
                  leaked);
    }
    {
        // A decode table whose extra no release reached: the boundary names it, and not the other model's.
        fake_registry reg;
        reg.add(host_zone_id::SCRATCH, alloc_lifetime::MODEL, model_5, "moe_hybrid_decode_table", 1024);
        reg.add(host_zone_id::SCRATCH, alloc_lifetime::MODEL, model_6, "moe_hybrid_decode_table", 4096);
        capture_begin();
        const size_t n = reg.check(alloc_lifetime::MODEL, model_5);
        HZP_CHECK(n == 1 && log_has("[HOST-ZONE] leaked MODEL record: moe_hybrid_decode_table 1024 owner=model/5") &&
                      !log_has("owner=model/6"),
                  "a leaked decode table at its model's slot release printed %zu leaked-record WARN(s), expected 1 "
                  "naming model/5 only",
                  n);
    }
    // Strict leases: the same leak aborts.
    const child_result r = run_in_child([&]() {
        fake_registry reg;
        reg.add(host_zone_id::SCRATCH, alloc_lifetime::MODEL, model_5, "moe_hybrid_decode_table", 1024);
        reg.check(alloc_lifetime::MODEL, model_5, /*strict=*/true);
    });
    HZP_CHECK(
        r.rc == 134 && r.err.find("[HOST-ZONE] leaked MODEL record: moe_hybrid_decode_table 1024 owner=model/5") !=
                           std::string::npos,
        "a strict leaked decode table at its model's slot release: rc %d, expected 134; child stderr:\n%s", r.rc,
        r.err.c_str());
}

// One strict case: the leak aborts in its own process, scored on rc 134 plus the leaked-record text.
void check_strict_abort(const char *    name,
                        alloc_lifetime  lt,
                        alloc_owner_ref owner,
                        host_zone_id    zone,
                        const char *    cohort,
                        const char *    text) {
    const child_result r = run_in_child([&]() {
        fake_registry reg;
        reg.add(zone, lt, owner, cohort, 4096);
        if (lt != alloc_lifetime::DEVICE || host_zone_device_check_due(0)) {
            reg.check(lt, owner, /*strict=*/true);
        }
    });
    HZP_CHECK(r.rc == 134 && r.err.find(text) != std::string::npos &&
                  r.err.find("GGML_SYCL_STRICT_LEASES=1") != std::string::npos,
              "strict %s leak: rc %d, expected 134 with the leaked-record text; child stderr:\n%s", name, r.rc,
              r.err.c_str());
}

// The strict cases, one per lifetime the owner check covers.
void test_strict_aborts() {
    check_strict_abort("buffer-owned CONTEXT", alloc_lifetime::CONTEXT, buffer_7, host_zone_id::SCRATCH,
                       "sycl_host_buffer", "[HOST-ZONE] leaked CONTEXT record: sycl_host_buffer 4096 owner=buffer/7");
    check_strict_abort("backend-member CONTEXT", alloc_lifetime::CONTEXT, backend_9, host_zone_id::STAGING,
                       "context_readback_staging",
                       "[HOST-ZONE] leaked CONTEXT record: context_readback_staging 4096 owner=backend/9");
    check_strict_abort("MODEL", alloc_lifetime::MODEL, model_3, host_zone_id::SCRATCH, "moe_hybrid_decode_table",
                       "[HOST-ZONE] leaked MODEL record: moe_hybrid_decode_table 4096 owner=model/3");
    check_strict_abort("DEVICE at the last backend", alloc_lifetime::DEVICE, device_0, host_zone_id::SCRATCH,
                       "pinned_buffer_pool",
                       "[HOST-ZONE] leaked DEVICE record: pinned_buffer_pool 4096 owner=device/0");
}

}  // namespace

int main() {
    test_routing_rows();
    test_no_category_selects_weight();
    test_graph_boundary();
    test_owner_leak_warns();
    test_device_last_backend();
    test_destruction_order();
    test_other_owner_does_not_fire();
    test_model_boundary_at_slot_release();
    test_strict_aborts();
    return finish("test-sycl-host-zone-select");
}
