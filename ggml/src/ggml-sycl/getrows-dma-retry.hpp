#pragma once
// get_rows' DMA slice loop (llama.cpp-5efe C4 part b), host-linkable: it includes no SYCL header, so a host test can
// drive the real loop over a fake stream_dma (tests/test-sycl-getrows-refusal-terminal.cpp).
//
// get_rows streams a host-resident table through the DMA pool. A failed attempt retries at half the slice (down to a
// floor, and with a single buffer from the second attempt on), up to a retry limit. One failure is not ordinary: a
// refusal of a file-mapped source (unified_cache::stream_dma, design 5efe §5.3), which is deterministic and terminal:
// the slice size changes nothing, so the loop ends on it before its retry test, with one attempt counted and no retry
// line, and the caller routes it to the CPU executor.

#include <cstddef>

namespace ggml_sycl {

// One attempt at streaming the rows, as the loop sees it. The caller keeps the unified_cache::dma_stream_result itself
// (the loop needs only these two facts). An attempt that threw a device out-of-memory a smaller slice may cure reports
// neither (it did not stream); anything else that throws propagates out of the loop.
struct get_rows_dma_attempt {
    bool ok                  = false;  // the rows streamed
    bool refused_file_mapped = false;  // stream_dma refused a file-mapped source
};

struct get_rows_dma_outcome {
    bool   streamed_ok         = false;
    bool   refused_file_mapped = false;
    size_t attempts            = 0;
    size_t final_slice_bytes   = 0;  // the slice the last attempt ran at
};

// `attempt(slice_bytes, buffer_count)` runs one attempt, `clamp(bytes)` is the caller's slice quantization, and
// `on_retry(slice_bytes, next_slice_bytes, buffer_count)` is the "[GET_ROWS] DMA retry" log line.
template <typename Attempt, typename Clamp, typename OnRetry>
inline get_rows_dma_outcome get_rows_dma_retry(size_t  slice_bytes,
                                               size_t  buffer_count,
                                               size_t  min_slice_bytes,
                                               size_t  retry_limit,
                                               Attempt attempt,
                                               Clamp   clamp,
                                               OnRetry on_retry) {
    get_rows_dma_outcome out;
    for (size_t retry_count = 0; retry_count <= retry_limit; ++retry_count) {
        const get_rows_dma_attempt result = attempt(slice_bytes, buffer_count);
        out.attempts++;
        out.final_slice_bytes = slice_bytes;
        // A refusal is terminal: it ends the loop before the retry test, whatever the slice size.
        if (result.refused_file_mapped) {
            out.refused_file_mapped = true;
            break;
        }
        if (result.ok) {
            out.streamed_ok = true;
            break;
        }
        if (slice_bytes <= min_slice_bytes) {
            break;
        }
        const size_t next_slice = clamp(slice_bytes / 2);
        if (next_slice == slice_bytes) {
            break;
        }
        on_retry(slice_bytes, next_slice, buffer_count);
        slice_bytes  = next_slice;
        buffer_count = 1;
    }
    return out;
}

}  // namespace ggml_sycl
