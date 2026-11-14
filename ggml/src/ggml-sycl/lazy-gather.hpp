#pragma once

// A STORAGE table's ubatch gather (owner ruling 2026-11-14, gather in set_input): the rows a ubatch reads, copied from
// the row cache on a hit and read from the file through the IO engine on a miss, waited for before it returns.
// SYCL-free like lazy-io-pool.hpp, so the engine's host test links it.

#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

class lazy_row_cache;

struct lazy_gather_source {
    int      fd          = -1;  // the table's O_DIRECT descriptor
    uint64_t file_id     = 0;   // the file's identity from its device and inode; the in-flight table dedups on it
    uint64_t file_offset = 0;   // the table's first byte in the file
    size_t   row_size    = 0;
    int64_t  n_rows      = 0;
};

struct lazy_gather_stats {
    uint64_t rows  = 0;  // rows looked up
    uint64_t hits  = 0;  // rows the cache served
    uint64_t reads = 0;  // waiters submitted for misses: not disk reads, since rows of one block share a read
};

// Copies row rows[i] of the table to dst + i * row_size for i in [0, n). `context` names the caller's queue, `epoch`
// this call (distinct from the caller's previous call), `deadline` the consumer layer. Returns 0 when every row landed;
// EINVAL for a bad row or a short dst; the first failed read's errno; EIO when the engine refused a read (no live ring,
// or a span wider than a slot); ENOMEM when a lookup or a submit threw std::bad_alloc, EIO when one threw anything
// else. `stats`, when not null, is reset and then counts this call.
//
// It never returns with a waiter of its own undelivered: every path that submitted anything waits until each waiter
// the engine counted has been delivered, the error paths included, so the request, ticket and landing slot it keeps on
// its own frame, and dst, are never touched after it returns. That wait has no timeout and ends only when the IO
// engine's pool workers have delivered, and a worker takes the engine's lock, the pool registry's, the landing slot's
// and the row cache's shard lock between its reads. So call it holding none of those locks, and never from a pool
// worker; the gather itself holds no lock while it waits.
int lazy_gather_rows(lazy_row_cache *           cache,
                     const lazy_gather_source & src,
                     uint64_t                   context,
                     uint32_t                   epoch,
                     int32_t                    deadline,
                     const int32_t *            rows,
                     size_t                     n,
                     uint8_t *                  dst,
                     size_t                     dst_bytes,
                     lazy_gather_stats *        stats) noexcept;

}  // namespace ggml_sycl
