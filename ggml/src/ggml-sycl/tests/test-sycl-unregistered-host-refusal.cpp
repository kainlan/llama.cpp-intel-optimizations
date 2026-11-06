//
// register_host_expert and register_host_weight refuse a pointer no registry knows
// (docs/design/sycl-mmap-host-tier.md §4.8 "The tier-switch census", §7 C6 test-sycl-unregistered-host-refusal).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Two cases, each in its own child: expert drives register_host_expert, weight drives register_host_weight. Each case
// registers two pointers on device 0's cache under a planned-materialization token, as a model load does:
//   - control: a slice of a FILE_MAPPED reservation with its owner, the registration a SYCL_Mapped weight makes. It
//     must be accepted, and the cache must show its entry, so a refusal of everything and a cache the lookups cannot
//     see both read as a failure of the control and never as the refusal working;
//   - subject: a malloc'ed block, which neither registry knows (query_location answers UNREGISTERED), with no owner.
//     It must be refused: the call returns false, hands out no lease and leaves no entry. Today it is accepted as a
//     HOST_MMAP entry that owns nothing (non_owning_external_host), which is what C6 retires.
// A block the registry turned out to know, or a control the cache refused, makes the case VOID, not a pass.

#include "file-mapped-test-common.hpp"

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

    mem_handle lease;
    bool       registered = false;
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
    fprintf(stderr, "unknown case '%s'\n", a.case_name.c_str());
    return 2;
}

}  // namespace

int main(int argc, char ** argv) {
    static const std::vector<const char *> all = { "expert", "weight" };
    return scratch_return_test::cases_main(argc, argv, all, run_case, std::chrono::minutes(2));
}
