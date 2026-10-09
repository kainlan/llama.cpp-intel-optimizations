// The compact host-expert MoE result scatter (llama.cpp-cre6): one copy of the compact result block into a planned
// device scratch, then one kernel placing each scratch row at its destination, must leave every destination byte
// exactly where the per-run copies it replaces left it. ggml-sycl/moe-host-scatter.hpp is SYCL-free, so this runs
// the copies and the kernel's row placement on host buffers, without a device.
#include "ggml-sycl/moe-host-scatter.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

// Counts every heap allocation, so a test can measure what one flush allocates. GCC inlines these into the standard
// containers and then pairs the malloc with the free as if they were new and free; they are not mismatched.
#if defined(__GNUC__) && !defined(__clang__)
#    pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
static size_t g_n_allocs = 0;

void * operator new(size_t n) {
    ++g_n_allocs;
    if (void * p = std::malloc(n ? n : 1)) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete(void * p) noexcept {
    std::free(p);
}

void operator delete(void * p, size_t) noexcept {
    std::free(p);
}

#define CHECK(cond, msg)                      \
    do {                                      \
        if (!(cond)) {                        \
            std::printf("FAILED: %s\n", msg); \
            return 1;                         \
        }                                     \
    } while (0)

using ggml_sycl::moe_scatter_chunk;
using ggml_sycl::moe_scatter_copy;
using ggml_sycl::moe_scatter_plan;
using ggml_sycl::moe_scatter_row;
using ggml_sycl::moe_scatter_run;

// One host-expert dispatch as the MUL_MAT_ID producers build it: entry ci's result row sits at (first + ci) * N floats
// of the output staging, and belongs at slot i1 of token i2 of dst (nb1 = N floats, nb2 = n_used rows).
struct host_entry {
    int64_t slot;   // i1
    int64_t token;  // i2
};

struct dispatch {
    int                     src    = 0;  // output staging buffer
    int                     dst    = 0;  // destination tensor buffer
    size_t                  first  = 0;  // first pool entry of the dispatch
    size_t                  view   = 0;  // dst view offset in its buffer, bytes
    int64_t                 N      = 0;
    int64_t                 n_used = 0;
    std::vector<host_entry> entries;
};

static void append_rows(const dispatch & d, std::vector<moe_scatter_row> & rows) {
    const size_t row_bytes = static_cast<size_t>(d.N) * sizeof(float);
    for (size_t ci = 0; ci < d.entries.size(); ++ci) {
        moe_scatter_row row;
        row.src        = d.src;
        row.src_offset = (d.first + ci) * row_bytes;
        row.dst        = d.dst;
        row.dst_offset = d.view + static_cast<size_t>(d.entries[ci].slot) * row_bytes +
                         static_cast<size_t>(d.entries[ci].token) * static_cast<size_t>(d.n_used) * row_bytes;
        row.bytes = row_bytes;
        rows.push_back(row);
    }
}

struct buffers {
    std::vector<std::vector<float>> src;
    std::vector<std::vector<float>> dst;
};

// Sources hold a value unique to (buffer, element); destinations start at a sentinel no source holds.
static buffers make_buffers(size_t n_src, size_t src_floats, size_t n_dst, size_t dst_floats) {
    buffers b;
    b.src.assign(n_src, std::vector<float>(src_floats));
    for (size_t s = 0; s < n_src; ++s) {
        for (size_t i = 0; i < src_floats; ++i) {
            b.src[s][i] = static_cast<float>(1000000 * (s + 1) + i);
        }
    }
    b.dst.assign(n_dst, std::vector<float>(dst_floats, -1.0f));
    return b;
}

static void copy_bytes(std::vector<float> &       to,
                       size_t                     to_off,
                       const std::vector<float> & from,
                       size_t                     from_off,
                       size_t                     bytes) {
    std::memcpy(reinterpret_cast<char *>(to.data()) + to_off, reinterpret_cast<const char *>(from.data()) + from_off,
                bytes);
}

// The per-run scatter this replaces.
static void apply_runs(const std::vector<moe_scatter_row> & rows, buffers & b, size_t * n_copies) {
    std::vector<moe_scatter_run> runs;
    ggml_sycl::moe_scatter_runs_build(rows, runs);
    for (const moe_scatter_run & run : runs) {
        copy_bytes(b.dst[run.dst], run.dst_offset, b.src[run.src], run.src_offset, run.bytes);
    }
    *n_copies = runs.size();
}

// The compact scatter: each chunk's copies into the scratch, then the kernel's per-row placement. False when it
// touches a scratch byte outside the planned scratch, reads a scratch row no copy filled, or names a destination
// beyond what the kernel takes.
static bool apply_plan(const moe_scatter_plan &             plan,
                       const std::vector<moe_scatter_row> & rows,
                       size_t                               scratch_bytes,
                       buffers &                            b) {
    std::vector<float> scratch(scratch_bytes / sizeof(float) + 1, -2.0f);
    for (const moe_scatter_chunk & chunk : plan.chunks) {
        if (chunk.n_rows == 0 || chunk.n_rows > ggml_sycl::MOE_SCATTER_MAX_ROWS ||
            chunk.first_row + chunk.n_rows > rows.size() || chunk.first_copy + chunk.n_copies > plan.copies.size()) {
            return false;
        }
        std::vector<bool> filled(chunk.n_rows * plan.row_bytes, false);
        for (size_t c = chunk.first_copy; c < chunk.first_copy + chunk.n_copies; ++c) {
            const moe_scatter_copy & copy = plan.copies[c];
            if (copy.scratch_offset + copy.bytes > scratch_bytes || copy.scratch_offset + copy.bytes > filled.size()) {
                return false;
            }
            copy_bytes(scratch, copy.scratch_offset, b.src[copy.src], copy.src_offset, copy.bytes);
            for (size_t i = 0; i < copy.bytes; ++i) {
                filled[copy.scratch_offset + i] = true;
            }
        }
        for (size_t r = 0; r < chunk.n_rows; ++r) {
            const moe_scatter_row & row = rows[chunk.first_row + r];
            if (row.dst < 0 || row.dst >= ggml_sycl::MOE_SCATTER_MAX_DSTS) {
                return false;
            }
            for (size_t i = 0; i < plan.row_bytes; ++i) {
                if (!filled[r * plan.row_bytes + i]) {
                    return false;
                }
            }
            copy_bytes(b.dst[row.dst], row.dst_offset, scratch, r * plan.row_bytes, plan.row_bytes);
        }
    }
    return true;
}

// Runs both forms on the same routing and compares every destination byte. Also checks the row -> slot -> dst map
// directly: each host entry's row lands at its own slot and token, and nothing else is written.
static int check_equivalent(const char *                  name,
                            const std::vector<dispatch> & ds,
                            size_t                        scratch_bytes,
                            size_t                        want_copies,
                            size_t                        want_chunks,
                            size_t *                      runs_out = nullptr) {
    std::vector<moe_scatter_row> rows;
    size_t                       n_src = 0, n_dst = 0, src_floats = 0, dst_floats = 0;
    for (const dispatch & d : ds) {
        append_rows(d, rows);
        n_src             = std::max(n_src, static_cast<size_t>(d.src) + 1);
        n_dst             = std::max(n_dst, static_cast<size_t>(d.dst) + 1);
        src_floats        = std::max(src_floats, (d.first + d.entries.size()) * static_cast<size_t>(d.N));
        int64_t max_token = 0;
        for (const host_entry & e : d.entries) {
            max_token = std::max(max_token, e.token);
        }
        dst_floats =
            std::max(dst_floats, d.view / sizeof(float) + static_cast<size_t>((max_token + 1) * d.n_used * d.N));
    }

    moe_scatter_plan plan;
    if (!ggml_sycl::moe_scatter_plan_build(rows, scratch_bytes, &plan)) {
        std::printf("FAILED: %s: the compact plan was refused\n", name);
        return 1;
    }
    buffers legacy  = make_buffers(n_src, src_floats, n_dst, dst_floats);
    buffers compact = make_buffers(n_src, src_floats, n_dst, dst_floats);
    size_t  n_runs  = 0;
    apply_runs(rows, legacy, &n_runs);
    if (!apply_plan(plan, rows, scratch_bytes, compact)) {
        std::printf("FAILED: %s: the plan left the scratch, read an unfilled row or named a bad destination\n", name);
        return 1;
    }
    if (legacy.dst != compact.dst) {
        std::printf("FAILED: %s: the compact scatter wrote different destination bytes than the per-run copies\n",
                    name);
        return 1;
    }
    // The map itself, independently of either implementation.
    std::vector<std::vector<bool>> written(n_dst, std::vector<bool>(dst_floats, false));
    for (const dispatch & d : ds) {
        for (size_t ci = 0; ci < d.entries.size(); ++ci) {
            const size_t dst_at =
                d.view / sizeof(float) +
                static_cast<size_t>(d.entries[ci].token * d.n_used + d.entries[ci].slot) * static_cast<size_t>(d.N);
            const size_t src_at = (d.first + ci) * static_cast<size_t>(d.N);
            for (int64_t k = 0; k < d.N; ++k) {
                if (compact.dst[d.dst][dst_at + k] != compact.src[d.src][src_at + k]) {
                    std::printf("FAILED: %s: entry %zu (slot %lld token %lld) is not at its destination row\n", name,
                                ci, (long long) d.entries[ci].slot, (long long) d.entries[ci].token);
                    return 1;
                }
                written[d.dst][dst_at + k] = true;
            }
        }
    }
    for (size_t t = 0; t < n_dst; ++t) {
        for (size_t i = 0; i < dst_floats; ++i) {
            if (!written[t][i] && compact.dst[t][i] != -1.0f) {
                std::printf("FAILED: %s: a destination float no host entry owns was written\n", name);
                return 1;
            }
        }
    }
    if (plan.copies.size() != want_copies || plan.chunks.size() != want_chunks) {
        std::printf("FAILED: %s: %zu copies in %zu chunks, want %zu in %zu (the per-run form takes %zu)\n", name,
                    plan.copies.size(), plan.chunks.size(), want_copies, want_chunks, n_runs);
        return 1;
    }
    if (plan.copies.size() > n_runs) {
        std::printf("FAILED: %s: the compact form issues more copies (%zu) than the per-run form (%zu)\n", name,
                    plan.copies.size(), n_runs);
        return 1;
    }
    if (runs_out) {
        *runs_out = n_runs;
    }
    return 0;
}

// Qwen3.8 decode shapes: top-k 10, gate/up N = 640, down N = 2560.
static constexpr int64_t k_n_used = 10;
static constexpr int64_t k_n_gate = 640;
static constexpr int64_t k_n_down = 2560;

static size_t scratch_for(int64_t rows, int64_t N) {
    return static_cast<size_t>(rows * N) * sizeof(float);
}

static dispatch decode(int src, int dst, size_t first, int64_t N, std::vector<int64_t> slots) {
    dispatch d;
    d.src    = src;
    d.dst    = dst;
    d.first  = first;
    d.N      = N;
    d.n_used = k_n_used;
    for (int64_t s : slots) {
        d.entries.push_back({ s, 0 });
    }
    return d;
}

static int test_decode_edges() {
    size_t runs = 0;
    // Single host expert.
    if (check_equivalent("single host expert", { decode(0, 0, 0, k_n_down, { 7 }) }, scratch_for(k_n_used, k_n_down), 1,
                         1, &runs) != 0) {
        return 1;
    }
    CHECK(runs == 1, "a single host expert is one per-run copy too");
    // All experts on the host, routed in slot order: the per-run form already merges them all.
    if (check_equivalent("all host, slot order", { decode(0, 0, 0, k_n_down, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }) },
                         scratch_for(k_n_used, k_n_down), 1, 1, &runs) != 0) {
        return 1;
    }
    CHECK(runs == 1, "all host experts in slot order are one per-run copy");
    // All on the host in routing (probability) order: one copy instead of ten.
    if (check_equivalent("all host, routing order", { decode(0, 0, 3, k_n_down, { 4, 0, 9, 2, 7, 1, 8, 3, 6, 5 }) },
                         scratch_for(k_n_used, k_n_down), 1, 1, &runs) != 0) {
        return 1;
    }
    CHECK(runs == 10, "all host experts in routing order are ten per-run copies");
    // The measured case: about four of ten on the host, scattered slots.
    if (check_equivalent("four host experts", { decode(0, 0, 0, k_n_down, { 1, 4, 5, 8 }) },
                         scratch_for(k_n_used, k_n_down), 1, 1, &runs) != 0) {
        return 1;
    }
    CHECK(runs == 3, "slots 1, 4-5, 8 are three per-run copies");
    return 0;
}

static int test_gate_up_merge() {
    size_t runs = 0;
    // Gate's job in pool entries [0, 4) and up's in [4, 7) of the same pool: one copy, one kernel, two destinations.
    if (check_equivalent("gate+up, shared pool",
                         { decode(0, 0, 0, k_n_gate, { 2, 3, 6, 9 }), decode(0, 1, 4, k_n_gate, { 0, 5, 8 }) },
                         scratch_for(2 * k_n_used, k_n_gate), 1, 1, &runs) != 0) {
        return 1;
    }
    CHECK(runs == 6, "gate {2,3},{6},{9} and up {0},{5},{8} are six per-run copies");
    // Both halves in the same destination buffer (two views of one compute buffer).
    dispatch gate = decode(0, 0, 0, k_n_gate, { 2, 3, 6, 9 });
    dispatch up   = decode(0, 0, 4, k_n_gate, { 0, 5, 8 });
    up.view       = static_cast<size_t>(k_n_used * k_n_gate) * sizeof(float);
    if (check_equivalent("gate+up, one destination buffer", { gate, up }, scratch_for(2 * k_n_used, k_n_gate), 1, 1) !=
        0) {
        return 1;
    }
    // Separate staging buffers (the managed over-capacity path): one copy per source, still one kernel. Up's rows
    // start where gate's end, offset for offset, so only the buffer tells the two runs apart.
    if (check_equivalent("gate+up, separate staging",
                         { decode(0, 0, 0, k_n_gate, { 2, 3, 6, 9 }), decode(1, 1, 4, k_n_gate, { 0, 5, 8 }) },
                         scratch_for(2 * k_n_used, k_n_gate), 2, 1) != 0) {
        return 1;
    }
    // Two destination buffers that meet offset for offset: gate's last row ends at byte 10 * N of buffer 0, and up's
    // row starts at byte 10 * N of buffer 1, with their sources contiguous. Only the destination buffer keeps the
    // per-run form from merging them into one copy that writes up's row into gate's buffer.
    dispatch adjacent_up = decode(0, 1, 2, k_n_gate, { 0 });
    adjacent_up.view     = static_cast<size_t>(k_n_used * k_n_gate) * sizeof(float);
    if (check_equivalent("gate+up, adjacent offsets in two buffers",
                         { decode(0, 0, 0, k_n_gate, { 8, 9 }), adjacent_up }, scratch_for(2 * k_n_used, k_n_gate), 1,
                         1, &runs) != 0) {
        return 1;
    }
    CHECK(runs == 2, "rows in two destination buffers are two per-run copies even when their offsets meet");
    // A wrapped pool ring (up reserved back at entry 0, gate further on): two copies, one kernel.
    if (check_equivalent("gate+up, wrapped ring",
                         { decode(0, 0, 16, k_n_gate, { 2, 3, 6, 9 }), decode(0, 1, 0, k_n_gate, { 0, 5, 8 }) },
                         scratch_for(2 * k_n_used, k_n_gate), 2, 1) != 0) {
        return 1;
    }
    return 0;
}

static int test_chunking() {
    // A prompt batch: 32 tokens, each host expert's tokens in expert order. 96 rows through a decode-sized
    // scratch (20 rows) take five chunks of one copy each, against about one per-run copy per row.
    dispatch prompt;
    prompt.N      = k_n_gate;
    prompt.n_used = k_n_used;
    for (int64_t e = 0; e < 3; ++e) {
        for (int64_t t = 0; t < 32; ++t) {
            prompt.entries.push_back({ (t + e * 3) % k_n_used, t });
        }
    }
    size_t runs = 0;
    if (check_equivalent("prompt batch, decode-sized scratch", { prompt }, scratch_for(2 * k_n_used, k_n_gate), 5, 5,
                         &runs) != 0) {
        return 1;
    }
    CHECK(runs == 87,
          "the prompt rows are one per-run copy each, except slot 9 of a token running into slot 0 of the next");
    // A scratch large enough for everything is still cut at one kernel launch's rows.
    dispatch big = prompt;
    for (int64_t t = 0; t < 4; ++t) {
        big.entries.push_back({ 0, 32 + t });
    }
    if (check_equivalent("100 rows, ample scratch", { big }, scratch_for(1000, k_n_gate), 2, 2) != 0) {
        return 1;
    }
    // A scratch that is not a whole number of rows holds the whole rows it can.
    if (check_equivalent("ragged scratch", { decode(0, 0, 0, k_n_down, { 1, 4, 5, 8 }) },
                         scratch_for(3, k_n_down) + 100, 2, 2) != 0) {
        return 1;
    }
    return 0;
}

static int test_refusals() {
    moe_scatter_plan             plan;
    std::vector<moe_scatter_row> rows;
    CHECK(!ggml_sycl::moe_scatter_plan_build(rows, scratch_for(k_n_used, k_n_down), &plan), "no rows, no plan");

    append_rows(decode(0, 0, 0, k_n_down, { 1, 4 }), rows);
    CHECK(!ggml_sycl::moe_scatter_plan_build(rows, scratch_for(1, k_n_down) - 4, &plan),
          "a scratch smaller than one row takes the per-run copies");
    CHECK(plan.copies.empty() && plan.chunks.empty(), "a refused plan is left empty");
    CHECK(!ggml_sycl::moe_scatter_plan_build(rows, 0, &plan), "no scratch, no plan");

    std::vector<moe_scatter_row> three_dsts = rows;
    three_dsts[1].dst                       = ggml_sycl::MOE_SCATTER_MAX_DSTS;
    CHECK(!ggml_sycl::moe_scatter_plan_build(three_dsts, scratch_for(k_n_used, k_n_down), &plan),
          "a destination beyond the kernel's two is refused");
    std::vector<moe_scatter_row> negative = rows;
    negative[0].dst                       = -1;
    CHECK(!ggml_sycl::moe_scatter_plan_build(negative, scratch_for(k_n_used, k_n_down), &plan),
          "a negative destination index is refused");

    std::vector<moe_scatter_row> ragged = rows;
    ragged[1].bytes -= sizeof(float);
    CHECK(!ggml_sycl::moe_scatter_plan_build(ragged, scratch_for(k_n_used, k_n_down), &plan),
          "rows of different lengths take the per-run copies");
    std::vector<moe_scatter_row> odd = rows;
    for (moe_scatter_row & r : odd) {
        r.bytes = 6;
    }
    CHECK(!ggml_sycl::moe_scatter_plan_build(odd, scratch_for(k_n_used, k_n_down), &plan),
          "a row that is not whole floats is refused (the kernel moves floats)");
    std::vector<moe_scatter_row> unaligned = rows;
    unaligned[1].dst_offset += 2;
    CHECK(!ggml_sycl::moe_scatter_plan_build(unaligned, scratch_for(k_n_used, k_n_down), &plan),
          "a destination that is not float-aligned is refused");

    // Two rows writing overlapping bytes: the per-run copies order them, one kernel launch does not.
    std::vector<moe_scatter_row> overlap = rows;
    overlap[1].dst_offset                = overlap[0].dst_offset + sizeof(float);
    CHECK(!ggml_sycl::moe_scatter_plan_build(overlap, scratch_for(k_n_used, k_n_down), &plan),
          "overlapping destinations are refused");
    std::vector<moe_scatter_row> same_row = rows;
    same_row[1].dst_offset                = same_row[0].dst_offset;
    CHECK(!ggml_sycl::moe_scatter_plan_build(same_row, scratch_for(k_n_used, k_n_down), &plan),
          "two rows bound for one destination row are refused");
    // The same offsets in different destination buffers do not overlap.
    std::vector<moe_scatter_row> two_bufs = same_row;
    two_bufs[1].dst                       = 1;
    CHECK(ggml_sycl::moe_scatter_plan_build(two_bufs, scratch_for(k_n_used, k_n_down), &plan),
          "equal offsets in different destination buffers are not an overlap");
    return 0;
}

static int test_scratch_bytes() {
    size_t bytes = 0;
    CHECK(ggml_sycl::moe_host_scatter_scratch_bytes(10, true, 640, &bytes) && bytes == 20 * 640 * sizeof(float),
          "split gate/up: both halves of a decode pair");
    CHECK(ggml_sycl::moe_host_scatter_scratch_bytes(10, false, 2560, &bytes) && bytes == 10 * 2560 * sizeof(float),
          "down (or fused gate_up): one op's rows");
    CHECK(ggml_sycl::moe_host_scatter_scratch_bytes(40, true, 640, &bytes) &&
              bytes == ggml_sycl::MOE_SCATTER_MAX_ROWS * 640 * sizeof(float),
          "capped at one kernel launch's rows");
    CHECK(ggml_sycl::moe_host_scatter_scratch_bytes(0, true, 640, &bytes) && bytes == 0, "a dense model plans nothing");
    CHECK(!ggml_sycl::moe_host_scatter_scratch_bytes(10, true, SIZE_MAX / 8, &bytes),
          "an overflowing figure is refused, not wrapped");
    return 0;
}

// Each reason a flush declines the compact form is reported once, and an earlier reason does not hide a later one.
static int test_decline_reported_once_per_reason() {
    std::atomic<uint32_t> seen{ 0 };
    CHECK(ggml_sycl::moe_scatter_decline_first(seen, ggml_sycl::MOE_SCATTER_DECLINE_GRAPH_RECORDING),
          "the first decline of a reason is reported");
    CHECK(!ggml_sycl::moe_scatter_decline_first(seen, ggml_sycl::MOE_SCATTER_DECLINE_GRAPH_RECORDING),
          "a repeated reason is not reported again");
    CHECK(ggml_sycl::moe_scatter_decline_first(seen, ggml_sycl::MOE_SCATTER_DECLINE_NO_SCRATCH),
          "a different reason after the first is still reported");
    CHECK(!ggml_sycl::moe_scatter_decline_first(seen, ggml_sycl::MOE_SCATTER_DECLINE_NO_SCRATCH),
          "the second reason is reported once too");
    for (uint32_t r = 0; r < ggml_sycl::MOE_SCATTER_DECLINE_COUNT; ++r) {
        CHECK(ggml_sycl::moe_scatter_decline_name(static_cast<ggml_sycl::moe_scatter_decline>(r))[0] != '\0',
              "every reason has a name for the WARN");
    }
    return 0;
}

// The scratch is planned only for a device whose layers keep experts on the host: an all-VRAM placement plans
// nothing, so nothing is claimed and nothing warns.
static int test_scratch_bytes_for_device() {
    using ggml_sycl::moe_host_scatter_tensor;
    moe_host_scatter_tensor gate;
    gate.split_gate_up = true;
    gate.row_elems     = 640;
    moe_host_scatter_tensor down;
    down.row_elems = 2560;
    size_t bytes   = 1;

    gate.device = 0;
    down.device = 0;
    CHECK(ggml_sycl::moe_host_scatter_scratch_bytes_for_device({ gate, down }, 0, 10, &bytes) && bytes == 0,
          "every expert on the device: no scratch is planned");

    down.has_host_experts = true;
    CHECK(ggml_sycl::moe_host_scatter_scratch_bytes_for_device({ gate, down }, 0, 10, &bytes) &&
              bytes == 10 * 2560 * sizeof(float),
          "host experts on device 0: the scratch covers their rows");

    down.device = 1;
    CHECK(ggml_sycl::moe_host_scatter_scratch_bytes_for_device({ gate, down }, 0, 10, &bytes) && bytes == 0,
          "host experts scattered by device 1 plan nothing on device 0");
    CHECK(ggml_sycl::moe_host_scatter_scratch_bytes_for_device({ gate, down }, 1, 10, &bytes) &&
              bytes == 10 * 2560 * sizeof(float),
          "they plan on device 1");

    gate.has_host_experts = true;
    gate.device           = -1;
    CHECK(ggml_sycl::moe_host_scatter_scratch_bytes_for_device({ gate }, 0, 10, &bytes) &&
              bytes == 20 * 640 * sizeof(float),
          "a tensor whose layer has no planned device plans on every device (device 0)");
    CHECK(ggml_sycl::moe_host_scatter_scratch_bytes_for_device({ gate }, 1, 10, &bytes) &&
              bytes == 20 * 640 * sizeof(float),
          "a tensor whose layer has no planned device plans on every device (device 1)");

    down.device                 = 0;
    moe_host_scatter_tensor bad = down;
    bad.row_elems               = SIZE_MAX / 8;
    CHECK(!ggml_sycl::moe_host_scatter_scratch_bytes_for_device({ down, bad }, 0, 10, &bytes),
          "an overflowing figure is refused, not wrapped");
    return 0;
}

// What one decode flush allocates to build its rows and plan. The flush used to build both afresh every call; it now
// reuses one workspace, so after the first flush a decode token allocates nothing here, including for a routing whose
// overlap check has to sort.
static int test_flush_allocations() {
    const std::vector<dispatch> in_order  = { decode(0, 0, 0, k_n_gate, { 2, 3, 6, 9 }),
                                              decode(0, 1, 4, k_n_gate, { 0, 5, 8 }) };
    const std::vector<dispatch> unordered = { decode(0, 0, 0, k_n_gate, { 9, 2, 6, 3 }),
                                              decode(0, 1, 4, k_n_gate, { 8, 0, 5 }) };
    const size_t                scratch   = scratch_for(2 * k_n_used, k_n_gate);
    const int                   n_flushes = 64;
    for (const std::vector<dispatch> * flush : { &in_order, &unordered }) {
        const size_t fresh_before = g_n_allocs;
        for (int i = 0; i < n_flushes; ++i) {
            std::vector<moe_scatter_row> rows;
            moe_scatter_plan             plan;
            for (const dispatch & d : *flush) {
                append_rows(d, rows);
            }
            CHECK(ggml_sycl::moe_scatter_plan_build(rows, scratch, &plan), "a decode flush takes the compact form");
        }
        const size_t fresh = (g_n_allocs - fresh_before) / n_flushes;

        std::vector<moe_scatter_row> rows;
        moe_scatter_plan             plan;
        for (int i = 0; i < 2; ++i) {  // the first flush sizes the workspace
            rows.clear();
            for (const dispatch & d : *flush) {
                append_rows(d, rows);
            }
            CHECK(ggml_sycl::moe_scatter_plan_build(rows, scratch, &plan), "a decode flush takes the compact form");
        }
        const size_t reused_before = g_n_allocs;
        for (int i = 0; i < n_flushes; ++i) {
            rows.clear();
            for (const dispatch & d : *flush) {
                append_rows(d, rows);
            }
            CHECK(ggml_sycl::moe_scatter_plan_build(rows, scratch, &plan), "a decode flush takes the compact form");
        }
        const size_t reused = g_n_allocs - reused_before;
        std::printf("allocations per decode flush building rows and plan (%s): fresh %zu, reused workspace %zu\n",
                    flush == &in_order ? "slot order" : "needs the overlap sort", fresh,
                    reused / static_cast<size_t>(n_flushes));
        CHECK(reused == 0, "a reused workspace allocates nothing per decode flush");
    }
    return 0;
}

int main() {
    if (test_decode_edges() != 0 || test_gate_up_merge() != 0 || test_chunking() != 0 || test_refusals() != 0 ||
        test_scratch_bytes() != 0 || test_decline_reported_once_per_reason() != 0 ||
        test_scratch_bytes_for_device() != 0 || test_flush_allocations() != 0) {
        return 1;
    }
    std::printf("OK: moe host scatter compaction\n");
    return 0;
}
