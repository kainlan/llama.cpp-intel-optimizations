// How often each expert of each MoE block is routed, counted per (block, expert) with no SYCL and no ggml in sight.
//
// Nothing in the backend otherwise knows which experts a run actually routes to. The hybrid decode route
// (ggml_sycl_moe_hybrid_decode_event_chained) sees the routed ids on its CPU pool worker, and the prompt route reads
// them from its ids snapshot; both feed one table of this type. The summaries answer "what hit rate would a
// use-ranked resident set of N slots per block have had on this run".
//
// A block is the model's block number (blk.N), the number the placement planner keys by, so a block's gate, up and
// down tensors share it. Only one of the three tensors of a block counts, so a routed (token, expert) pair counts
// once, not three times. The table is sized from the model's dimensions and never clamped: Qwen3.8 Flash Next is
// 48 blocks of 512 experts, 24576 cells, and a cell the table does not hold is refused and counted in out_of_range(),
// never written elsewhere.
//
// Threads. Every counter is a relaxed atomic: the pool worker counting decode ids and the thread counting a prompt
// never wait for each other, and a reader that summarises while a worker still counts sees a total that is at most a
// few increments behind, never a torn one. The table's size is fixed at construction and its arrays are never
// reallocated, so no reader can hold a pointer into storage that moved. Nothing here allocates after construction
// except the summary functions' own result strings and scratch vectors.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace ggml_sycl {

class moe_route_counts {
  public:
    // n_blocks x n_experts cells, all zero. A non-positive dimension makes an empty table that refuses every count.
    moe_route_counts(int n_blocks, int n_experts) :
        n_blocks_(n_blocks > 0 ? n_blocks : 0),
        n_experts_(n_experts > 0 ? n_experts : 0) {
        if (n_blocks_ == 0 || n_experts_ == 0) {
            n_blocks_  = 0;
            n_experts_ = 0;
        }
        const size_t cells  = cells_total();
        const size_t blocks = static_cast<size_t>(n_blocks_);
        decode_.reset(new std::atomic<uint64_t>[cells]);
        prompt_.reset(new std::atomic<uint64_t>[cells]);
        last_.reset(new std::atomic<uint8_t>[cells]);
        hits_.reset(new std::atomic<uint64_t>[blocks]);
        misses_.reset(new std::atomic<uint64_t>[blocks]);
        resident_.reset(new std::atomic<int32_t>[blocks]);
        reset();
    }

    moe_route_counts(const moe_route_counts &)             = delete;
    moe_route_counts & operator=(const moe_route_counts &) = delete;

    int n_blocks() const { return n_blocks_; }

    int n_experts() const { return n_experts_; }

    // One routed (token, expert) pair of a decode op. `on_device` says whether the expert was on the device (a hit)
    // or on the host (a miss) when it was routed. Returns false, and counts nothing but out_of_range(), when the
    // cell is not in the table.
    bool add_decode(int block, int expert, bool on_device) {
        if (!in_range(block, expert)) {
            out_of_range_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        const size_t cell = index(block, expert);
        decode_[cell].fetch_add(1, std::memory_order_relaxed);
        (on_device ? hits_ : misses_)[static_cast<size_t>(block)].fetch_add(1, std::memory_order_relaxed);
        // The last residency seen. A load first, so a steady placement never writes the cell's cache line twice.
        const uint8_t seen = on_device ? RESIDENT_DEVICE : RESIDENT_HOST;
        if (last_[cell].load(std::memory_order_relaxed) != seen) {
            last_[cell].store(seen, std::memory_order_relaxed);
        }
        return true;
    }

    // One routed (token, expert) pair of a prompt op. A prompt count says nothing about where the expert was.
    bool add_prompt(int block, int expert) {
        if (!in_range(block, expert)) {
            out_of_range_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        prompt_[index(block, expert)].fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // How many of the block's experts the plan holds on the device right now (the hybrid route's record knows it).
    // Out of range is ignored.
    void set_block_resident(int block, int n_resident) {
        if (block >= 0 && block < n_blocks_) {
            resident_[static_cast<size_t>(block)].store(n_resident, std::memory_order_relaxed);
        }
    }

    // Zero every counter, forget every residency, and re-arm claim_report().
    void reset() {
        const size_t cells = cells_total();
        for (size_t i = 0; i < cells; ++i) {
            decode_[i].store(0, std::memory_order_relaxed);
            prompt_[i].store(0, std::memory_order_relaxed);
            last_[i].store(RESIDENT_UNSEEN, std::memory_order_relaxed);
        }
        for (size_t b = 0; b < static_cast<size_t>(n_blocks_); ++b) {
            hits_[b].store(0, std::memory_order_relaxed);
            misses_[b].store(0, std::memory_order_relaxed);
            resident_[b].store(-1, std::memory_order_relaxed);
        }
        out_of_range_.store(0, std::memory_order_relaxed);
        reported_.store(false, std::memory_order_relaxed);
    }

    // True for the first caller since construction or reset(): the one printer of the final report.
    bool claim_report() { return !reported_.exchange(true, std::memory_order_acq_rel); }

    uint64_t decode_count(int block, int expert) const {
        return in_range(block, expert) ? decode_[index(block, expert)].load(std::memory_order_relaxed) : 0;
    }

    uint64_t prompt_count(int block, int expert) const {
        return in_range(block, expert) ? prompt_[index(block, expert)].load(std::memory_order_relaxed) : 0;
    }

    uint64_t block_hits(int block) const {
        return block >= 0 && block < n_blocks_ ? hits_[static_cast<size_t>(block)].load(std::memory_order_relaxed) : 0;
    }

    uint64_t block_misses(int block) const {
        return block >= 0 && block < n_blocks_ ? misses_[static_cast<size_t>(block)].load(std::memory_order_relaxed) :
                                                 0;
    }

    // -1 when the expert was never routed in decode, 0 when it was on the host the last time, 1 when on the device.
    int resident_at_last_use(int block, int expert) const {
        return in_range(block, expert) ?
                   static_cast<int>(last_[index(block, expert)].load(std::memory_order_relaxed)) - 1 :
                   -1;
    }

    // The experts of the block the plan holds on the device, -1 when it was never recorded.
    int block_resident(int block) const {
        return block >= 0 && block < n_blocks_ ? resident_[static_cast<size_t>(block)].load(std::memory_order_relaxed) :
                                                 -1;
    }

    // Counts refused for a cell outside the table (a stale table, or ids beyond the model's experts).
    uint64_t out_of_range() const { return out_of_range_.load(std::memory_order_relaxed); }

  private:
    static constexpr uint8_t RESIDENT_UNSEEN = 0;
    static constexpr uint8_t RESIDENT_HOST   = 1;
    static constexpr uint8_t RESIDENT_DEVICE = 2;

    size_t cells_total() const { return static_cast<size_t>(n_blocks_) * static_cast<size_t>(n_experts_); }

    bool in_range(int block, int expert) const {
        return block >= 0 && block < n_blocks_ && expert >= 0 && expert < n_experts_;
    }

    size_t index(int block, int expert) const {
        return static_cast<size_t>(block) * static_cast<size_t>(n_experts_) + static_cast<size_t>(expert);
    }

    int                                      n_blocks_;
    int                                      n_experts_;
    std::unique_ptr<std::atomic<uint64_t>[]> decode_;
    std::unique_ptr<std::atomic<uint64_t>[]> prompt_;
    std::unique_ptr<std::atomic<uint8_t>[]>  last_;
    std::unique_ptr<std::atomic<uint64_t>[]> hits_;
    std::unique_ptr<std::atomic<uint64_t>[]> misses_;
    std::unique_ptr<std::atomic<int32_t>[]>  resident_;
    std::atomic<uint64_t>                    out_of_range_{ 0 };
    std::atomic<bool>                        reported_{ false };
};

// ---------------------------------------------------------------------------------------------------------------
// Summaries. Pure reads of a table; a worker still counting makes them a few increments stale, nothing more.
// ---------------------------------------------------------------------------------------------------------------

// Routed pairs of the block in decode: its hits plus its misses.
inline uint64_t moe_route_block_pairs(const moe_route_counts & c, int block) {
    return c.block_hits(block) + c.block_misses(block);
}

// Share of the block's decode pairs that were routed to a device expert; 0 for a block nothing was routed in.
inline double moe_route_block_hit_rate(const moe_route_counts & c, int block) {
    const uint64_t pairs = moe_route_block_pairs(c, block);
    return pairs > 0 ? static_cast<double>(c.block_hits(block)) / static_cast<double>(pairs) : 0.0;
}

// Blocks with at least one decode pair. A block the direct route runs (all experts resident) reports no ids to the
// host and is never counted.
inline int moe_route_blocks_counted(const moe_route_counts & c) {
    int n = 0;
    for (int b = 0; b < c.n_blocks(); ++b) {
        n += moe_route_block_pairs(c, b) > 0 ? 1 : 0;
    }
    return n;
}

inline uint64_t moe_route_decode_hits(const moe_route_counts & c) {
    uint64_t n = 0;
    for (int b = 0; b < c.n_blocks(); ++b) {
        n += c.block_hits(b);
    }
    return n;
}

inline uint64_t moe_route_decode_misses(const moe_route_counts & c) {
    uint64_t n = 0;
    for (int b = 0; b < c.n_blocks(); ++b) {
        n += c.block_misses(b);
    }
    return n;
}

inline uint64_t moe_route_decode_pairs(const moe_route_counts & c) {
    return moe_route_decode_hits(c) + moe_route_decode_misses(c);
}

inline uint64_t moe_route_prompt_pairs(const moe_route_counts & c) {
    uint64_t n = 0;
    for (int b = 0; b < c.n_blocks(); ++b) {
        for (int e = 0; e < c.n_experts(); ++e) {
            n += c.prompt_count(b, e);
        }
    }
    return n;
}

// Hit rate over every counted block, each weighed by its routed pairs.
inline double moe_route_overall_hit_rate(const moe_route_counts & c) {
    const uint64_t pairs = moe_route_decode_pairs(c);
    return pairs > 0 ? static_cast<double>(moe_route_decode_hits(c)) / static_cast<double>(pairs) : 0.0;
}

// The hit rate of the whole model, derived: a block that was not counted took the direct route, so every pair it
// routed was a device hit and none was seen. Every block routes the same number of pairs per token (top-k experts for
// each token), so a block not counted routed as many pairs as the counted blocks do on average. With P the decode
// pairs, H the device hits, C the counted blocks and U the blocks not counted:
//     model hit rate = (H + U * P / C) / (P + U * P / C), and 0 when C is 0 (nothing to take the average from).
// This is an estimate from that assumption, not a measurement; moe_route_overall_hit_rate is the measured figure over
// the counted blocks alone.
inline double moe_route_model_hit_rate(const moe_route_counts & c) {
    const int counted = moe_route_blocks_counted(c);
    if (counted <= 0) {
        return 0.0;
    }
    const double pairs   = static_cast<double>(moe_route_decode_pairs(c));
    const double hits    = static_cast<double>(moe_route_decode_hits(c));
    const double per_not = pairs / static_cast<double>(counted);  // pairs a block that was not counted routed
    const double not_all = static_cast<double>(c.n_blocks() - counted) * per_not;
    return (hits + not_all) / (pairs + not_all);
}

namespace moe_route_detail {

// The decode counts of the block's `n` most-routed experts, and the block's total.
inline void top_n_sum(const moe_route_counts & c, int block, int n, uint64_t * top, uint64_t * total) {
    std::vector<uint64_t> counts(static_cast<size_t>(c.n_experts()));
    uint64_t              sum = 0;
    for (int e = 0; e < c.n_experts(); ++e) {
        counts[static_cast<size_t>(e)] = c.decode_count(block, e);
        sum += counts[static_cast<size_t>(e)];
    }
    const size_t keep = n <= 0 ? 0 : std::min(static_cast<size_t>(n), counts.size());
    if (keep > 0 && keep < counts.size()) {
        std::nth_element(counts.begin(), counts.begin() + static_cast<std::ptrdiff_t>(keep) - 1, counts.end(),
                         [](uint64_t a, uint64_t b) { return a > b; });
    }
    uint64_t top_sum = 0;
    for (size_t i = 0; i < keep; ++i) {
        top_sum += counts[i];
    }
    *top   = top_sum;
    *total = sum;
}

}  // namespace moe_route_detail

// The share of the block's routed pairs that its `n` most-routed experts account for: the hit rate a use-ranked
// resident set of n slots would have had on this run. 0 for a block nothing was routed in.
inline double moe_route_top_n_coverage(const moe_route_counts & c, int block, int n) {
    uint64_t top = 0;
    uint64_t sum = 0;
    moe_route_detail::top_n_sum(c, block, n, &top, &sum);
    return sum > 0 ? static_cast<double>(top) / static_cast<double>(sum) : 0.0;
}

// The same over every block, weighing each by its routed pairs: n slots in every block.
inline double moe_route_top_n_coverage_overall(const moe_route_counts & c, int n) {
    uint64_t top_all = 0;
    uint64_t sum_all = 0;
    for (int b = 0; b < c.n_blocks(); ++b) {
        uint64_t top = 0;
        uint64_t sum = 0;
        moe_route_detail::top_n_sum(c, b, n, &top, &sum);
        top_all += top;
        sum_all += sum;
    }
    return sum_all > 0 ? static_cast<double>(top_all) / static_cast<double>(sum_all) : 0.0;
}

// The mean number of resident experts per block. A counted block contributes what the plan holds on the device for
// it (a counted block whose residency was never recorded is left out of the mean), a block that was not counted is
// all resident (the direct route reports nothing), so a run that never decoded reads as fully resident.
inline double moe_route_mean_resident_experts(const moe_route_counts & c) {
    double sum   = 0.0;
    int    count = 0;
    for (int b = 0; b < c.n_blocks(); ++b) {
        if (moe_route_block_pairs(c, b) > 0) {
            if (c.block_resident(b) >= 0) {
                sum += c.block_resident(b);
                ++count;
            }
        } else {
            sum += c.n_experts();
            ++count;
        }
    }
    return count > 0 ? sum / count : 0.0;
}

// One CSV row per (block, expert), block-major. resident_at_last_use is -1 for an expert never routed in decode, 0
// for host, 1 for device.
inline std::string moe_route_csv(const moe_route_counts & c) {
    std::string out = "block,expert,decode_count,prompt_count,resident_at_last_use\n";
    out.reserve(out.size() + static_cast<size_t>(c.n_blocks()) * static_cast<size_t>(c.n_experts()) * 16);
    char row[96];
    for (int b = 0; b < c.n_blocks(); ++b) {
        for (int e = 0; e < c.n_experts(); ++e) {
            const int n = std::snprintf(
                row, sizeof(row), "%d,%d,%llu,%llu,%d\n", b, e, static_cast<unsigned long long>(c.decode_count(b, e)),
                static_cast<unsigned long long>(c.prompt_count(b, e)), c.resident_at_last_use(b, e));
            if (n > 0) {
                out.append(row, static_cast<size_t>(n) < sizeof(row) ? static_cast<size_t>(n) : sizeof(row) - 1);
            }
        }
    }
    return out;
}

// One line per block. A block with no decode pair is named as all resident and not counted: the direct route reports
// no ids to the host, so it is a gap in the figures, not a zero.
inline std::string moe_route_block_table(const moe_route_counts & c) {
    std::string out;
    char        line[256];
    for (int b = 0; b < c.n_blocks(); ++b) {
        uint64_t prompt = 0;
        for (int e = 0; e < c.n_experts(); ++e) {
            prompt += c.prompt_count(b, e);
        }
        int n = 0;
        if (moe_route_block_pairs(c, b) > 0) {
            n = std::snprintf(line, sizeof(line),
                              "[MOE-ROUTE-COUNTS] block %d routed pairs=%llu hit rate=%.1f%% (device %llu, host %llu) "
                              "resident experts=%d prompt pairs=%llu\n",
                              b, static_cast<unsigned long long>(moe_route_block_pairs(c, b)),
                              100.0 * moe_route_block_hit_rate(c, b), static_cast<unsigned long long>(c.block_hits(b)),
                              static_cast<unsigned long long>(c.block_misses(b)), c.block_resident(b),
                              static_cast<unsigned long long>(prompt));
        } else {
            n = std::snprintf(line, sizeof(line),
                              "[MOE-ROUTE-COUNTS] block %d all resident, not counted; prompt pairs=%llu\n", b,
                              static_cast<unsigned long long>(prompt));
        }
        if (n > 0) {
            out.append(line, static_cast<size_t>(n) < sizeof(line) ? static_cast<size_t>(n) : sizeof(line) - 1);
        }
    }
    return out;
}

// The one-line final summary: tokens, blocks counted and not counted, routed pairs, the decode hit rate, and the top-N
// coverage at N = the model's mean number of resident experts per block, rounded: the hit rate a use-ranked set with
// the same slots in every block would have had.
inline std::string moe_route_summary_line(const moe_route_counts & c, int64_t tokens) {
    const int    counted = moe_route_blocks_counted(c);
    const double mean    = moe_route_mean_resident_experts(c);
    const int    top_n   = static_cast<int>(std::lround(mean));
    char         line[768];
    int          n = std::snprintf(
        line, sizeof(line),
        "[MOE-ROUTE-COUNTS] tokens=%lld blocks counted=%d not counted=%d (all resident, not counted) of %d; "
                 "decode routed pairs=%llu hit rate=%.1f%% (device %llu, host %llu) "
                 "model hit rate=%.1f%% (blocks not counted taken as all device); top-%d coverage=%.1f%% "
                 "(a use-ranked set of the mean %.1f resident experts per block); prompt routed pairs=%llu",
        static_cast<long long>(tokens), counted, c.n_blocks() - counted, c.n_blocks(),
        static_cast<unsigned long long>(moe_route_decode_pairs(c)), 100.0 * moe_route_overall_hit_rate(c),
        static_cast<unsigned long long>(moe_route_decode_hits(c)),
        static_cast<unsigned long long>(moe_route_decode_misses(c)), 100.0 * moe_route_model_hit_rate(c), top_n,
        100.0 * moe_route_top_n_coverage_overall(c, top_n), mean,
        static_cast<unsigned long long>(moe_route_prompt_pairs(c)));
    std::string out(line,
                    n > 0 ? (static_cast<size_t>(n) < sizeof(line) ? static_cast<size_t>(n) : sizeof(line) - 1) : 0);
    if (c.out_of_range() > 0) {
        out += "; out of range=" + std::to_string(c.out_of_range());
    }
    return out;
}

}  // namespace ggml_sycl
