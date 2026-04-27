#include "llama-tensor-fitter.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

// Format a byte count as MB to one decimal place into a fixed-size buffer.
// Avoids std::format / iostream overhead in the err_msg path.
// SIZE_MAX is the unbounded-tier sentinel (caller passes it for an mmap
// spill tier); rendering it as a 17-digit MB count is technically correct
// but useless to a human chasing a diagnostic — emit "unbounded" instead.
std::string format_mb(size_t bytes) {
    if (bytes == SIZE_MAX) {
        return "unbounded";
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    return buf;
}

// Total-order comparator for non-P0 tensors (P0 is pre-placed at the call
// site; this orders the remaining P1/P2/P3 deterministically). Within a
// priority class, earlier layers go first (MoE-Infinity hot-half
// convention), larger tensors first as a fragmentation-reducing
// tie-breaker, then name ASC for stable determinism on ties.
bool sort_non_p0(const llama_tensor_placement_input & a, const llama_tensor_placement_input & b) {
    if (a.priority != b.priority) {
        return static_cast<int>(a.priority) < static_cast<int>(b.priority);
    }
    if (a.layer_idx != b.layer_idx) {
        return a.layer_idx < b.layer_idx;
    }
    if (a.bytes != b.bytes) {
        return a.bytes > b.bytes;
    }
    return a.name < b.name;
}

}  // namespace

std::optional<llama_tensor_placement_summary> llama_tensor_fit(const std::vector<llama_tensor_placement_input> & inputs,
                                                               const std::vector<llama_placement_tier> &         tiers,
                                                               std::string * err_msg) {
    auto fail = [&](const std::string & msg) -> std::optional<llama_tensor_placement_summary> {
        if (err_msg) {
            *err_msg = msg;
        }
        return std::nullopt;
    };

    if (tiers.empty()) {
        return fail("llama_tensor_fit: tier list is empty (no places to put any tensor)");
    }

    // Initialize per-tier accounting from the caller-declared budgets.
    std::vector<llama_placement_tier_stats> stats;
    stats.reserve(tiers.size());
    for (const auto & t : tiers) {
        stats.push_back({ t.tier_name, t.budget_bytes, 0, t.budget_bytes, 0 });
    }

    std::vector<llama_tensor_placement> placements;
    placements.reserve(inputs.size());

    // Step 1: P0 tensors all land on the fastest tier (tiers[0]). Walk the
    // input list in order so the diagnostic on overflow can list the exact
    // tensors that caused the breach.
    size_t                                                                  p0_total = 0;
    std::vector<std::reference_wrapper<const llama_tensor_placement_input>> p0_refs;
    for (const auto & in : inputs) {
        if (in.priority == LLAMA_TENSOR_PRIORITY_P0) {
            p0_total += in.bytes;
            p0_refs.emplace_back(in);
        }
    }
    if (p0_total > tiers[0].budget_bytes) {
        return fail("llama_tensor_fit: P0 tensors total " + format_mb(p0_total) + " exceeds tier '" +
                    tiers[0].tier_name + "' budget " + format_mb(tiers[0].budget_bytes) +
                    " (model is too large for the fastest tier regardless of policy)");
    }
    for (const auto & ref : p0_refs) {
        const auto & in = ref.get();
        placements.push_back({ in.name, tiers[0].tier_name, in.bytes });
        stats[0].used_bytes += in.bytes;
        stats[0].residual_bytes -= in.bytes;
        stats[0].n_tensors += 1;
    }

    // Step 2: Sort the non-P0 tensors and greedily pack into the tier list.
    // Bucket P3 last because P3 always lands on the final tier per spec; the
    // sort_non_p0 comparator already orders P1 < P2 < P3, but we drain P3
    // through a dedicated branch below to keep the spill semantics explicit.
    std::vector<llama_tensor_placement_input> rest;
    rest.reserve(inputs.size() - p0_refs.size());
    for (const auto & in : inputs) {
        if (in.priority != LLAMA_TENSOR_PRIORITY_P0) {
            rest.push_back(in);
        }
    }
    std::sort(rest.begin(), rest.end(), sort_non_p0);

    for (const auto & in : rest) {
        if (in.priority == LLAMA_TENSOR_PRIORITY_P3) {
            // P3 always goes on the last tier (host-spill convention).
            const size_t last = tiers.size() - 1;
            if (in.bytes > stats[last].residual_bytes) {
                return fail("llama_tensor_fit: P3 tensor '" + in.name + "' (" + format_mb(in.bytes) +
                            ") doesn't fit in last tier '" + tiers[last].tier_name + "' (residual " +
                            format_mb(stats[last].residual_bytes) +
                            "); append an unbounded mmap tier or increase the host budget");
            }
            placements.push_back({ in.name, tiers[last].tier_name, in.bytes });
            stats[last].used_bytes += in.bytes;
            stats[last].residual_bytes -= in.bytes;
            stats[last].n_tensors += 1;
            continue;
        }

        // P1, P2: take the lowest-index tier with enough residual.
        bool placed = false;
        for (size_t t = 0; t < tiers.size(); ++t) {
            if (in.bytes <= stats[t].residual_bytes) {
                placements.push_back({ in.name, tiers[t].tier_name, in.bytes });
                stats[t].used_bytes += in.bytes;
                stats[t].residual_bytes -= in.bytes;
                stats[t].n_tensors += 1;
                placed = true;
                break;
            }
        }
        if (!placed) {
            return fail("llama_tensor_fit: tensor '" + in.name + "' (" + format_mb(in.bytes) + ", priority " +
                        llama_tensor_priority_name(in.priority) + ") doesn't fit in any provided tier (last tier '" +
                        tiers.back().tier_name + "' had " + format_mb(stats.back().residual_bytes) +
                        " residual); provide more capacity or append an unbounded mmap tier");
        }
    }

    return llama_tensor_placement_summary{ std::move(placements), std::move(stats) };
}
