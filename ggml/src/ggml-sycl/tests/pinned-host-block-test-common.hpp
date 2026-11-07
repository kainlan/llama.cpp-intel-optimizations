//
// A host block for the tests that register host memory with a unified_cache directly
// (docs/design/sycl-mmap-host-tier.md §4.8, §7 C6). register_host_expert and register_host_weight refuse a pointer no
// registry knows, so a std::vector's bytes no longer qualify; this block is a pinned allocation minted owner-first
// through unified_allocate_owner, which is a registered HOST_PINNED row. The block owns its memory, and a registration
// over it stays the unowned external kind the test meant, because the call still names no owner.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// require_host_usm_base makes the allocation a standalone host USM base on the queue given, so the block needs neither
// the global cache nor configured host zones: the local unified_cache objects these tests build can register over it.
//
// A cache entry registered over a block points at the block's memory, so the block must outlive the cache that holds
// the entry: declare it BEFORE the cache (locals are destroyed in reverse order). Against the process-wide cache, which
// outlives any function, retire the entry before the block is freed.

#pragma once

#include "../common.hpp"
#include "../unified-cache.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace pinned_host_block_test {

class block {
  public:
    block(sycl::queue & q, size_t bytes, uint8_t fill) : bytes_(bytes) {
        ggml_sycl::alloc_request req{};
        req.queue                                    = &q;
        req.device                                   = ggml_sycl_get_device_id_from_queue(q);
        req.size                                     = bytes;
        req.intent.role                              = ggml_sycl::alloc_role::WEIGHT;
        req.intent.constraints.must_host_pinned      = true;
        req.intent.constraints.require_host_usm_base = true;
        ggml_sycl::allocation_result allocation      = ggml_sycl::unified_allocate_owner(req);
        if (allocation) {
            owner_ = ggml_sycl::mem_handle::from_owned_alloc(std::move(allocation.owner), GGML_LAYOUT_AOS);
            ptr_   = static_cast<uint8_t *>(owner_.resolve().ptr);
        }
        if (ptr_ == nullptr) {
            std::fprintf(stderr, "FAIL: unified_allocate_owner refused a %zu-byte pinned host block\n", bytes);
            std::exit(1);
        }
        std::memset(ptr_, fill, bytes);
    }

    block(const block &)             = delete;
    block & operator=(const block &) = delete;

    uint8_t * data() const { return ptr_; }

    size_t size() const { return bytes_; }

    uint8_t & operator[](size_t i) const { return ptr_[i]; }

  private:
    ggml_sycl::mem_handle owner_;
    uint8_t *             ptr_   = nullptr;
    size_t                bytes_ = 0;
};

// Whole pages of a temporary file mapped through the FILE_MAPPED front door on `device`'s cache, as a SYCL_Mapped
// buffer holds them: a registered FILE_MAPPED row, with the owner that keeps the mapping alive. A registration over it
// names that owner (register_host_expert_owned).
class mapped_block {
  public:
    mapped_block(int device, size_t pages) {
        char      path[] = "/tmp/unregistered-host-test-XXXXXX";
        const int tmp    = mkstemp(path);
        if (tmp < 0) {
            fail("the fixture file could not be created");
        }
        bytes_ = pages * static_cast<size_t>(sysconf(_SC_PAGESIZE));
        std::vector<char> content(bytes_, 0x5c);
        const bool        written = write(tmp, content.data(), bytes_) == static_cast<ssize_t>(bytes_);
        close(tmp);
        const int fd = open(path, O_RDONLY);
        unlink(path);
        if (!written || fd < 0) {
            fail("the fixture file could not be written");
        }
        ggml_sycl::file_mapping_request req;
        req.device      = device;
        req.fd          = fd;
        req.file_size   = bytes_;
        req.reservation = { 0, bytes_ };
        req.segments    = {
            { 0, bytes_ }
        };
        ggml_sycl::allocation_result mapped = ggml_sycl::unified_allocate_file_mapping(req);
        close(fd);
        if (!mapped) {
            fail("unified_allocate_file_mapping refused the request");
        }
        owner_ = ggml_sycl::mem_handle::from_owned_alloc(std::move(mapped.owner));
        ptr_   = static_cast<uint8_t *>(owner_.resolve().ptr);
        if (ptr_ == nullptr) {
            fail("the mapping did not resolve to a host pointer");
        }
    }

    mapped_block(const mapped_block &)             = delete;
    mapped_block & operator=(const mapped_block &) = delete;

    uint8_t * data() const { return ptr_; }

    size_t size() const { return bytes_; }

    const ggml_sycl::mem_handle & owner() const { return owner_; }

  private:
    static void fail(const char * why) {
        std::fprintf(stderr, "FAIL: mapped host block: %s\n", why);
        std::exit(1);
    }

    ggml_sycl::mem_handle owner_;
    uint8_t *             ptr_   = nullptr;
    size_t                bytes_ = 0;
};

}  // namespace pinned_host_block_test
