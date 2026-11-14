#include "lazy-gather.hpp"

#include "lazy-io-pool.hpp"
#include "lazy-row-cache.hpp"

#include <cerrno>
#include <new>

namespace ggml_sycl {

int lazy_gather_rows(lazy_row_cache *           cache,
                     const lazy_gather_source & src,
                     uint64_t                   context,
                     uint32_t                   epoch,
                     int32_t                    deadline,
                     const int32_t *            rows,
                     size_t                     n,
                     uint8_t *                  dst,
                     size_t                     dst_bytes,
                     lazy_gather_stats *        stats) noexcept {
    lazy_gather_stats   local;
    lazy_gather_stats & st = stats != nullptr ? *stats : local;
    st                     = {};
    if (n == 0) {
        return 0;
    }
    if (rows == nullptr || dst == nullptr || src.row_size == 0 || dst_bytes / src.row_size < n) {
        return EINVAL;
    }
    // The request, ticket and landing slot live on this frame, and no pointer to them leaves it but the waiters': the
    // wait below is what makes that safe. The slot's epoch is set once, here, before the first submit.
    lazy_gather_request req;
    lazy_io_ticket      ticket;
    lazy_landing_slot   slot;
    req.epoch      = epoch;
    req.ticket     = &ticket;
    ticket.request = &req;
    slot.epoch     = epoch;
    slot.data      = dst;
    slot.bytes     = n * src.row_size;

    const size_t      span_bytes = lazy_block_ring_span(src.row_size);
    const lazy_io_key key{ lazy_io_class::PREFETCH, epoch, deadline, lazy_io_kind::TABLE_ROWS };
    int               stopped = 0;  // why submitting stopped early; the reads already counted are still waited for
    for (size_t i = 0; i < n && stopped == 0; ++i) {
        const int64_t row = rows[i];
        if (row < 0 || row >= src.n_rows) {
            stopped = EINVAL;
            break;
        }
        ++st.rows;
        uint8_t * const out = dst + i * src.row_size;
        // A lookup takes the cache's shard lock and a submit allocates: either may throw, and this function is
        // noexcept with reads perhaps already counted on its frame, so a throw stops the submitting and still waits.
        try {
            if (cache != nullptr && cache->lookup_copy(row, out)) {
                ++st.hits;
                continue;
            }
            const uint64_t byte  = src.file_offset + static_cast<uint64_t>(row) * src.row_size;
            const uint64_t start = byte / LAZY_IO_BLOCK_BYTES * LAZY_IO_BLOCK_BYTES;
            lazy_io_waiter w;
            w.ticket         = &ticket;
            w.epoch          = epoch;
            w.landing        = &slot;
            w.landing_offset = i * src.row_size;
            w.src_offset     = byte - start;
            w.bytes          = src.row_size;
            w.cache          = cache;
            w.row            = row;
            // 0 is a refusal the engine did not count (no live ring, or a span the ring cannot hold); it has logged
            // which. The rows after it are not submitted.
            if (lazy_io_submit(context, src.fd, lazy_io_span{ src.file_id, start, span_bytes }, key, w) == 0) {
                stopped = EIO;
            } else {
                ++st.reads;
            }
        } catch (const std::bad_alloc &) {
            stopped = ENOMEM;
        } catch (...) {
            stopped = EIO;
        }
    }
    // Every path after the first submit reaches this wait. Its 0 proves every counted row was copied into dst, not
    // merely delivered, because this frame alone owns the request and the landing slot: nothing else can cancel or
    // drain the request or move the slot's epoch, so no delivery skips its copy, and each landing range,
    // [i * row_size, (i + 1) * row_size) with i < n, fits the slot's n * row_size bytes, so none is refused as outside
    // it. A refused submit counted nothing, and `stopped` reports it.
    const int read_error = lazy_io_ticket_wait(ticket);
    return stopped != 0 ? stopped : read_error;
}

}  // namespace ggml_sycl
