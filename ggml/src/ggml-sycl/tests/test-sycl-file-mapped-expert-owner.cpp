//
// register_host_expert over a mapped slice stores the mapping's owner, and the mapping outlives the buffer while an
// entry holds it (docs/design/sycl-mmap-host-tier.md §4.3-4.4, §7 C4 test-sycl-file-mapped-expert-owner).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// One case, "owner". It maps a fixture file through the FILE_MAPPED front door on device 0's cache, as a SYCL_Mapped
// buffer does, and registers a slice inside it as a host expert with the whole mapping's handle as the owner, as the
// preload does for a served-in-place expert. The test's handle stands in for the buffer's: dropping it is the buffer's
// free. Then, with only the entry and its lease holding the mapping, the file is still mapped, the lease reports the
// FILE_MAPPED owner and resolves to the slice, and the slice reads the file's bytes. Dropping the lease and retiring
// the entry unmaps it, so the entry held the last reference and nothing else leaked one.

#include "file-mapped-test-common.hpp"

using namespace ggml_sycl;
using scratch_return_test::checks;

namespace {

int run_owner(checks & c) {
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
    if (!view || view.on_device || !alloc_registry::instance().is_file_mapped(view.ptr) ||
        !file_mapped_test::maps_file(f.path)) {
        c.void_case("the mapping did not resolve to a host pointer the registry names FILE_MAPPED over " + f.path);
        return c.status();
    }

    // A slice inside the mapping, off its base: one page in, four pages long.
    const size_t       page         = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t       slice_offset = page;
    const size_t       slice_bytes  = 4 * page;
    uint8_t * const    slice        = static_cast<uint8_t *>(view.ptr) + slice_offset;
    static const char  key_tag      = 0;
    ggml_sycl_cache_id key{};
    key.valid    = true;
    key.model_id = 0x66696c65ull;
    key.aux_id   = reinterpret_cast<uintptr_t>(&key_tag);

    mem_handle lease;
    bool       registered = false;
    {
        scoped_planned_materialization planned(cache, "test-sycl-file-mapped-expert-owner/registration");
        registered = cache->register_host_expert_owned(key, slice, slice_bytes, GGML_LAYOUT_AOS, &lease, mapping);
    }
    if (!registered || !lease.valid()) {
        c.void_case("register_host_expert_owned refused the mapped slice");
        return c.status();
    }
    mapping = {};  // The buffer's free: from here only the entry and the lease hold the mapping.

    if (!file_mapped_test::maps_file(f.path)) {
        c.fail(
            "the mapping was unmapped when the buffer's reference dropped while an entry holds the slice: the "
            "entry did not store the owner");
        return c.status();
    }
    alloc_metadata owner{};
    c.expect(
        lease.owner_metadata(&owner) && owner.tier == alloc_tier::FILE_MAPPED,
        "the lease does not report a FILE_MAPPED owner (tier=" + std::to_string(static_cast<int>(owner.tier)) + ")");
    const resolved_ptr held = lease.resolve();
    c.expect(held.ptr == slice, "the lease does not resolve to the registered slice");
    bool same = held.ptr == slice;
    for (size_t i = 0; same && i < slice_bytes; ++i) {
        same = slice[i] == file_mapped_test::pattern(slice_offset + i);
    }
    c.expect(same, "the slice does not read the file's bytes while the entry holds the mapping");
    if (c.status() != 0) {
        return c.status();
    }

    lease = {};
    c.expect(cache->retire_expert_entry_exact(key, GGML_LAYOUT_AOS, "test-sycl-file-mapped-expert-owner") ==
                 expert_retire_status::WITHDRAWN,
             "the registered expert was not withdrawn");
    cache->process_deferred_frees_public();
    c.expect(!file_mapped_test::maps_file(f.path),
             "the file is still mapped after the last entry released it: another reference leaked");
    fprintf(stderr,
            "[FILE-MAPPED-OWNER] the mapping outlived the buffer while the entry held it, and unmapped after\n");
    return c.status();
}

int run_case(const scratch_return_test::child_args & a) {
    if (!scratch_return_test::child_begin(a)) {
        return 2;
    }
    checks c{ a.case_name.c_str() };
    if (a.case_name == "owner") {
        return run_owner(c);
    }
    fprintf(stderr, "unknown case '%s'\n", a.case_name.c_str());
    return 2;
}

}  // namespace

int main(int argc, char ** argv) {
    static const std::vector<const char *> all = { "owner" };
    return file_mapped_test::gated_main(argc, argv, all, run_case, std::chrono::minutes(2));
}
