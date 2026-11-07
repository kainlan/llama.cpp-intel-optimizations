//
// register_host_expert and register_host_weight refuse a pointer no registry knows
// (docs/design/sycl-mmap-host-tier.md §4.8 "The tier-switch census", §7 C6 test-sycl-unregistered-host-refusal).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Cases, each in its own child: expert drives register_host_expert, weight drives register_host_weight. Each of those
// registers two pointers on device 0's cache under a planned-materialization token, as a model load does:
//   - control: a slice of a FILE_MAPPED reservation with its owner, the registration a SYCL_Mapped weight makes. It
//     must be accepted, and the cache must show its entry, so a refusal of everything and a cache the lookups cannot
//     see both read as a failure of the control and never as the refusal working;
//   - subject: a malloc'ed block, which neither registry knows (query_location answers UNREGISTERED), with no owner.
//     It must be refused: the call returns false, hands out no lease and leaves no entry. Today it is accepted as a
//     HOST_MMAP entry that owns nothing (non_owning_external_host), which is what C6 retires.
// A block the registry turned out to know, or a control the cache refused, makes the case VOID, not a pass. The refusal
// also counts: the process-wide refusal counter moves by one for the subject and by none for the control.
//
// A third case, interior, is the production shape of the same hazard. A host expert is registered over a slice inside
// a pinned allocation (a SYCL_Host buffer's expert), and query_location answers UNREGISTERED for such a slice because
// the runtime registry is exact-base. The slice must still be accepted, as a HOST_PINNED entry, by both calls and with
// the refusal counter unmoved: a refusal keyed on query_location alone would fire on a real load.
//
// A fourth case, direct, is the direct stage's host fallback (§4.8, §5.3). A destination no zone can hold sends
// direct_stage_expert to its host fallback; with the pinned copy refused as well (a source length no pinned pool can
// grant), the fallback publishes the source in place. A slice inside a pinned allocation is published in place as
// HOST_PINNED, and a malloc'ed source is refused, counted, and leaves no entry: it has no owner to publish under.

#include "file-mapped-test-common.hpp"
#include "pinned-host-block-test-common.hpp"

using namespace ggml_sycl;
using scratch_return_test::checks;

namespace {

// A malloc'ed block that stays live for the case, and the figure the registry gives for it.
struct host_block {
    void *       ptr   = nullptr;
    const size_t bytes = 4096;

    host_block() {
        ptr = std::aligned_alloc(4096, bytes);
        if (ptr != nullptr) {
            std::memset(ptr, 0x5a, bytes);
        }
    }

    ~host_block() { std::free(ptr); }

    host_block(const host_block &)             = delete;
    host_block & operator=(const host_block &) = delete;
};

const char * location_name(cache_location l) {
    switch (l) {
        case cache_location::DEVICE:
            return "DEVICE";
        case cache_location::HOST_PINNED:
            return "HOST_PINNED";
        case cache_location::HOST_MMAP:
            return "HOST_MMAP";
        case cache_location::UNKNOWN:
            return "UNKNOWN";
        case cache_location::STORAGE:
            return "STORAGE";
    }
    return "?";
}

const char * retire_status_name(expert_retire_status status) {
    switch (status) {
        case expert_retire_status::INVALID:
            return "INVALID";
        case expert_retire_status::NOT_FOUND:
            return "NOT_FOUND";
        case expert_retire_status::WITHDRAWN:
            return "WITHDRAWN";
        case expert_retire_status::DEFERRED:
            return "DEFERRED";
    }
    return "?";
}

ggml_sycl_cache_id key_of(uint64_t aux) {
    ggml_sycl_cache_id key{};
    key.valid    = true;
    key.model_id = 0x756e7265ull;
    key.aux_id   = aux;
    return key;
}

// The entry the cache holds for `key`, by the lookup the case's registration feeds.
const weight_entry * entry_of(unified_cache * cache, bool expert, ggml_sycl_cache_id key) {
    return expert ? cache->lookup_expert(key) : cache->lookup_weight(key);
}

int run_register(checks & c, bool expert) {
    const char * const api = expert ? "register_host_expert" : "register_host_weight";
    if (!file_mapped_test::entry_ok(c)) {
        return c.status();
    }
    if (ggml_backend_sycl_init(0) == nullptr) {
        c.void_case("ggml_backend_sycl_init(0) failed");
        return c.status();
    }
    unified_cache * cache = get_unified_cache_for_device(0);
    if (cache == nullptr) {
        c.void_case("device 0 has no unified cache");
        return c.status();
    }

    // The control: a mapped slice with the mapping's owner.
    file_mapped_test::fixture_file f(16);
    if (!f.ok) {
        c.void_case("the fixture file could not be written");
        return c.status();
    }
    mem_handle mapping = file_mapped_test::map_whole(c, f, 0);
    if (!mapping.valid()) {
        return c.status();
    }
    const resolved_ptr view = mapping.resolve();
    if (!view || view.on_device || !alloc_registry::instance().is_file_mapped(view.ptr)) {
        c.void_case("the mapping did not resolve to a host pointer the registry names FILE_MAPPED");
        return c.status();
    }
    const size_t       page        = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    uint8_t * const    slice       = static_cast<uint8_t *>(view.ptr) + page;
    const size_t       slice_bytes = 4 * page;
    ggml_sycl_cache_id control_key = key_of(1);
    ggml_sycl_cache_id subject_key = key_of(2);
    mem_handle         control_lease;
    bool               control_registered = false;
    {
        scoped_planned_materialization planned(cache, "test-sycl-unregistered-host-refusal/control");
        if (expert) {
            control_registered = cache->register_host_expert_owned(control_key, slice, slice_bytes, GGML_LAYOUT_AOS,
                                                                   &control_lease, mapping);
        } else {
            auto storage_owner = std::make_shared<mem_handle>(mapping);
            control_registered =
                cache->register_host_weight(control_key, slice, slice_bytes, GGML_LAYOUT_AOS, &control_lease,
                                            std::shared_ptr<void>(storage_owner, slice));
        }
    }
    const weight_entry * control_entry = entry_of(cache, expert, control_key);
    if (!control_registered || !control_lease.valid() || control_entry == nullptr) {
        c.void_case(std::string("the control, a FILE_MAPPED slice with its owner, was not registered by ") + api +
                    " (returned " + std::to_string(control_registered) + ", lease " +
                    std::to_string(control_lease.valid()) + ", entry " + std::to_string(control_entry != nullptr) +
                    "): a refusal of everything would read as the refusal working");
        return c.status();
    }

    if (unified_cache_unregistered_host_registration_refusals() != 0) {
        c.fail("the control, a FILE_MAPPED slice with its owner, moved the refusal counter to " +
               std::to_string(unified_cache_unregistered_host_registration_refusals()) + ", expected 0");
    }

    // The subject: a block neither registry knows.
    host_block block;
    if (block.ptr == nullptr) {
        c.void_case("aligned_alloc returned null");
        return c.status();
    }
    const memory_location known = query_location(block.ptr, 0);
    if (known.tier != alloc_tier::UNREGISTERED || alloc_registry::instance().lookup_copy(block.ptr)) {
        c.void_case("the malloc'ed block is known to a registry (tier " + std::to_string(static_cast<int>(known.tier)) +
                    "): it is not the UNREGISTERED subject");
        return c.status();
    }

    mem_handle     lease;
    bool           registered = false;
    const uint64_t refusals   = unified_cache_unregistered_host_registration_refusals();
    {
        scoped_planned_materialization planned(cache, "test-sycl-unregistered-host-refusal/subject");
        registered = expert ?
                         cache->register_host_expert(subject_key, block.ptr, block.bytes, GGML_LAYOUT_AOS, &lease) :
                         cache->register_host_weight(subject_key, block.ptr, block.bytes, GGML_LAYOUT_AOS, &lease);
    }
    const weight_entry * entry = entry_of(cache, expert, subject_key);
    if (registered || lease.valid() || entry != nullptr) {
        std::string seen = std::string(api) + " accepted a malloc'ed UNREGISTERED pointer: returned " +
                           std::to_string(registered) + ", lease valid=" + std::to_string(lease.valid());
        if (entry != nullptr) {
            seen += ", entry location=" + std::string(location_name(entry->location)) + "(" +
                    std::to_string(static_cast<int>(entry->location)) + ")";
        } else {
            seen += ", no entry";
        }
        c.fail(seen + "; expected a refusal: returned 0, no lease, no entry");
    }
    const uint64_t refused = unified_cache_unregistered_host_registration_refusals() - refusals;
    c.expect(refused == 1, std::string(api) + " counted " + std::to_string(refused) +
                               " unregistered-host refusal(s) for the malloc'ed pointer, expected 1");

    // The control is still the cache's: the refusal took nothing else down with it.
    c.expect(entry_of(cache, expert, control_key) != nullptr,
             std::string("the mapped control entry is gone after the subject's ") + api);

    lease         = {};
    control_lease = {};
    if (expert) {
        (void) cache->retire_expert_entry_exact(control_key, GGML_LAYOUT_AOS, "test-sycl-unregistered-host-refusal");
        cache->process_deferred_frees_public();
    }
    if (c.status() == 0) {
        fprintf(stderr, "[UNREGISTERED-HOST] %s refused the malloc'ed pointer and kept the mapped control\n", api);
    }
    return c.status();
}

int run_interior(checks & c) {
    if (!file_mapped_test::entry_ok(c)) {
        return c.status();
    }
    if (ggml_backend_sycl_init(0) == nullptr) {
        c.void_case("ggml_backend_sycl_init(0) failed");
        return c.status();
    }
    unified_cache * cache = get_unified_cache_for_device(0);
    if (cache == nullptr) {
        c.void_case("device 0 has no unified cache");
        return c.status();
    }
    constexpr size_t bytes = 4ull * 1024ull * 1024ull;
    if (!cache->host_zones_configured()) {
        cache->configure_host_zones(2 * bytes, 2ull * 1024ull * 1024ull, 4ull * 1024ull * 1024ull,
                                    2ull * 1024ull * 1024ull);
    }
    if (!cache->host_zones_configured()) {
        c.void_case("host zones could not be configured");
        return c.status();
    }
    // The request a host-planned expert's buffer makes (test-mem-ops.cpp, the host expert lease copy).
    alloc_request req{};
    req.device                              = 0;  // the cache under test is device 0's
    req.size                                = bytes;
    req.intent.role                         = alloc_role::WEIGHT;
    req.intent.constraints.must_host_pinned = true;
    req.intent.constraints.use_pinned_pool  = true;
    allocation_result allocation            = unified_allocate_owner(req);
    if (!allocation) {
        c.void_case("the pinned pool refused the " + std::to_string(bytes) + "-byte allocation");
        return c.status();
    }
    mem_handle         owner = mem_handle::from_owned_alloc(std::move(allocation.owner), GGML_LAYOUT_AOS);
    const resolved_ptr base  = owner.resolve();
    if (!base || base.on_device) {
        c.void_case("the pinned allocation did not resolve to host memory");
        return c.status();
    }
    // A slice off the base, as the preload's expert_aos is, registered with no owner of its own.
    constexpr size_t slice_bytes = 4096;
    void * const     slice       = static_cast<uint8_t *>(base.ptr) + 8192;
    const alloc_tier exact       = query_location(slice, 0).tier;
    if (query_location(base.ptr, 0).tier != alloc_tier::HOST_PINNED || exact != alloc_tier::UNREGISTERED) {
        c.void_case("the base is not a HOST_PINNED row, or query_location already resolves the slice (tier " +
                    std::to_string(static_cast<int>(exact)) + "): the containment lookup is not scored");
        return c.status();
    }

    // A dense weight's entry keeps a direct-table mirror that holds a lease, and nothing short of a model reset
    // removes it, so the weight is registered with the pinned allocation as its storage owner, as the preload does for
    // a dense weight (§4.8): the entry then keeps its memory alive and cannot outlive it. This case therefore does not
    // exercise an unowned dense weight over a pinned slice; the mirror's lease is the reason, and retiring it is beyond
    // what the cache offers a test. The expert is registered with no owner, as the preload's no-owner branch does,
    // and is retired below.
    const auto weight_storage = std::make_shared<mem_handle>(owner);

    const uint64_t refusals = unified_cache_unregistered_host_registration_refusals();
    for (int kind = 0; kind < 2; ++kind) {
        const bool         expert = kind == 0;
        const char * const api    = expert ? "register_host_expert" : "register_host_weight";
        ggml_sycl_cache_id key    = key_of(10 + kind);
        mem_handle         lease;
        bool               registered = false;
        {
            scoped_planned_materialization planned(cache, "test-sycl-unregistered-host-refusal/interior");
            registered = expert ? cache->register_host_expert(key, slice, slice_bytes, GGML_LAYOUT_AOS, &lease) :
                                  cache->register_host_weight(key, slice, slice_bytes, GGML_LAYOUT_AOS, &lease,
                                                              std::shared_ptr<void>(weight_storage, slice));
        }
        const weight_entry * entry = entry_of(cache, expert, key);
        c.expect(
            registered && lease.valid() && entry != nullptr && entry->location == cache_location::HOST_PINNED,
            std::string(api) + " refused or misplaced a slice inside a pinned allocation: returned " +
                std::to_string(registered) + ", lease valid=" + std::to_string(lease.valid()) +
                (entry != nullptr ? ", entry location=" + std::string(location_name(entry->location)) : ", no entry") +
                "; expected accepted as HOST_PINNED");

        // The unowned expert entry points into `owner`'s memory and the process-wide cache outlives this function, so
        // it is retired before the allocation is freed: an entry that names memory must not outlive it.
        if (expert) {
            lease = {};
            const expert_retire_status retired =
                cache->retire_expert_entry_exact(key, GGML_LAYOUT_AOS, "test-sycl-unregistered-host-refusal");
            cache->process_deferred_frees_public();
            // WITHDRAWN means the storage is reclaimed. DEFERRED leaves the canonical entry naming the block, and
            // entry_of reads only the direct-table mirror, which retirement erases in either case, so the status is
            // what is checked.
            c.expect(retired == expert_retire_status::WITHDRAWN,
                     std::string("retiring the entry ") + api + " registered over the pinned slice returned status " +
                         retire_status_name(retired) +
                         ", expected WITHDRAWN: the entry would name memory freed when "
                         "the allocation drops");
        }
    }
    const uint64_t refused = unified_cache_unregistered_host_registration_refusals() - refusals;
    c.expect(refused == 0, "the pinned slice moved the refusal counter by " + std::to_string(refused) + ", expected 0");
    if (c.status() == 0) {
        fprintf(stderr, "[UNREGISTERED-HOST] a slice inside a pinned allocation was accepted by both calls\n");
    }
    return c.status();
}

int run_direct(checks & c) {
    if (!file_mapped_test::entry_ok(c)) {
        return c.status();
    }
    if (ggml_backend_sycl_init(0) == nullptr) {
        c.void_case("ggml_backend_sycl_init(0) failed");
        return c.status();
    }
    unified_cache * cache = get_unified_cache_for_device(0);
    if (cache == nullptr) {
        c.void_case("device 0 has no unified cache");
        return c.status();
    }
    sycl::queue &    q             = cache->get_queue();
    constexpr size_t slice_bytes   = 4096;
    // No zone holds this destination and no pinned pool grants this source length, so both allocations are refused and
    // the fallback decides what to publish.
    constexpr size_t dst_bytes     = static_cast<size_t>(1) << 50;
    constexpr size_t lying_src_len = static_cast<size_t>(1) << 40;

    // The control: a slice inside a pinned allocation, which only the containment lookup resolves.
    pinned_host_block_test::block registered(q, 8192, 0x5a);
    uint8_t * const               slice = registered.data() + 256;
    if (query_location(slice, 0).tier != alloc_tier::UNREGISTERED) {
        c.void_case("query_location already resolves the slice: the containment lookup is not scored");
        return c.status();
    }
    // The subject: a malloc'ed source neither registry knows.
    host_block block;
    if (block.ptr == nullptr) {
        c.void_case("aligned_alloc returned null");
        return c.status();
    }
    if (query_location(block.ptr, 0).tier != alloc_tier::UNREGISTERED) {
        c.void_case("the malloc'ed block is known to a registry: it is not the UNREGISTERED subject");
        return c.status();
    }
    {
        // The subject's source length is a lie that only a refused pinned copy hides; a pool that grants it would copy
        // that many bytes from a 4 KiB block, so the precondition is checked by asking for it first.
        alloc_request req{};
        req.queue                               = &q;
        req.device                              = 0;
        req.size                                = lying_src_len;
        req.alignment                           = 256;
        req.intent.role                         = alloc_role::WEIGHT;
        req.intent.constraints.must_host_pinned = true;
        req.intent.constraints.use_pinned_pool  = true;
        allocation_result probe                 = unified_allocate_owner(req);
        if (probe) {
            c.void_case("the pinned pool granted a " + std::to_string(lying_src_len) + "-byte block");
            return c.status();
        }
    }

    const ggml_sycl_cache_id control_key = key_of(20);
    const ggml_sycl_cache_id subject_key = key_of(21);

    uint64_t            refusals = unified_cache_unregistered_host_registration_refusals();
    direct_stage_result control;
    control =
        cache->direct_stage_expert(control_key, slice, slice_bytes, dst_bytes, GGML_LAYOUT_SOA, nullptr, nullptr, &q);
    const size_t control_dropped = cache->drop_expert_entries_for_tensor_layout(
        std::vector<ggml_sycl_cache_id>{ control_key }, GGML_LAYOUT_AOS, "test-sycl-unregistered-host-refusal");
    if (!control.ok || control.ptr != slice || control_dropped != 1) {
        c.void_case("the control, a slice inside a pinned allocation, was not published in place by the fallback (ok " +
                    std::to_string(control.ok) + ", ptr==slice " + std::to_string(control.ptr == slice) + ", entries " +
                    std::to_string(control_dropped) + "): a refusal of everything would read as the refusal working");
        return c.status();
    }
    c.expect(unified_cache_unregistered_host_registration_refusals() == refusals,
             "the control moved the refusal counter by " +
                 std::to_string(unified_cache_unregistered_host_registration_refusals() - refusals) + ", expected 0");

    refusals = unified_cache_unregistered_host_registration_refusals();
    direct_stage_result subject;
    subject = cache->direct_stage_expert(subject_key, block.ptr, lying_src_len, dst_bytes, GGML_LAYOUT_SOA, nullptr,
                                         nullptr, &q);
    const size_t subject_entries = cache->drop_expert_entries_for_tensor_layout(
        std::vector<ggml_sycl_cache_id>{ subject_key }, GGML_LAYOUT_AOS, "test-sycl-unregistered-host-refusal");
    if (subject.ok || subject_entries != 0) {
        c.fail("direct_stage_expert published an UNREGISTERED source in place: ok=" + std::to_string(subject.ok) +
               ", ptr==source=" + std::to_string(subject.ptr == block.ptr) +
               ", entries=" + std::to_string(subject_entries) + "; expected a refusal: not ok, no entry");
    }
    const uint64_t refused = unified_cache_unregistered_host_registration_refusals() - refusals;
    c.expect(refused == 1, "direct_stage_expert counted " + std::to_string(refused) +
                               " unregistered-host refusal(s) for the malloc'ed source, expected 1");
    if (c.status() == 0) {
        fprintf(stderr,
                "[UNREGISTERED-HOST] direct_stage_expert refused the malloc'ed source and kept the pinned slice\n");
    }
    return c.status();
}

int run_case(const scratch_return_test::child_args & a) {
    if (!scratch_return_test::child_begin(a)) {
        return 2;
    }
    checks c{ a.case_name.c_str() };
    if (a.case_name == "expert") {
        return run_register(c, true);
    }
    if (a.case_name == "weight") {
        return run_register(c, false);
    }
    if (a.case_name == "interior") {
        return run_interior(c);
    }
    if (a.case_name == "direct") {
        return run_direct(c);
    }
    fprintf(stderr, "unknown case '%s'\n", a.case_name.c_str());
    return 2;
}

}  // namespace

int main(int argc, char ** argv) {
    static const std::vector<const char *> all = { "expert", "weight", "interior", "direct" };
    return scratch_return_test::cases_main(argc, argv, all, run_case, std::chrono::minutes(2));
}
