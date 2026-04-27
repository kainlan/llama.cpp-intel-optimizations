#pragma once

#include "llama-tensor-class.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

// PLACE-3 — Greedy budget fitter for weight placement across an ordered
// hierarchy of memory tiers (fastest first). Pure function: no GGUF I/O,
// no ggml types, no SYCL coupling. The caller (PLACE-4 / model load)
// reads tensor sizes from GGUF metadata and populates `bytes`; the
// caller (PLACE-12 / device discovery) decides which tiers exist and
// what their budgets are. PLACE-3 only decides which tier each tensor
// lands on.

// One tensor's input to the fitter. `priority` comes from PLACE-2;
// `bytes` is the GGUF tensor size; `cls`/`layer_idx` are echoed for
// the caller's convenience and are not consulted by the fit algorithm.
struct llama_tensor_placement_input {
    std::string           name;
    llama_tensor_class    cls;
    int                   layer_idx;
    llama_tensor_priority priority;
    size_t                bytes;
};

// One tier of the memory hierarchy, ordered fastest to slowest.
// Single-tier (e.g., one GPU) is the degenerate case. Caller appends a
// `{"host-mmap", SIZE_MAX}` final tier when an unbounded spill path is
// desired; PLACE-3 itself enforces every declared budget.
struct llama_placement_tier {
    std::string tier_name;
    size_t      budget_bytes;
};

// Per-tensor placement decision. `tier_name` mirrors the input tier the
// fitter chose; `bytes` echoes the input for callers building per-tier
// summaries from this struct alone.
struct llama_tensor_placement {
    std::string name;
    std::string tier_name;
    size_t      bytes;
};

// Per-tier residency stats, computed alongside the placement decisions.
// Useful for the GGML_SYCL_DEBUG planner-summary log line and for
// PLACE-4's "device buffer size > 0" acceptance check.
struct llama_placement_tier_stats {
    std::string tier_name;
    size_t      budget_bytes;
    size_t      used_bytes;
    size_t      residual_bytes;
    size_t      n_tensors;
};

// Bundled return: per-tensor placements + per-tier stats. Returned in a
// single struct because the caller almost always wants both, and the
// stats are a single linear pass over the placements anyway.
struct llama_tensor_placement_summary {
    std::vector<llama_tensor_placement>     placements;
    std::vector<llama_placement_tier_stats> per_tier;
};

// Greedy multi-tier fitter.
//
// Algorithm:
//   1. All P0 tensors land on tiers[0]. If their sum exceeds tiers[0]
//      budget, return nullopt with a diagnostic listing the overflow —
//      the model is too big for the fastest tier regardless of policy.
//   2. P1 then P2 tensors are sorted by (priority ASC, layer_idx ASC,
//      bytes DESC, name ASC) and greedily packed: each tensor lands on
//      the lowest-index tier that has enough residual budget.
//   3. P3 tensors land on the last tier (host-spill convention).
//   4. If any non-P0 tensor cannot fit in any tier — caller did not
//      provide enough total capacity and did not append an unbounded
//      spill tier — return nullopt with a diagnostic naming the
//      offending tensor and its size.
//
// Determinism: same inputs always produce the same output. Sort uses
// (priority, layer_idx, bytes DESC, name ASC) as a stable tie-breaker
// chain so the algorithm is independent of the input vector's order.
//
// On error, *err_msg (if non-null) receives a one-line human-readable
// diagnostic; the return value is std::nullopt.
//
// Empty `tiers` is an error (no places to put anything).
std::optional<llama_tensor_placement_summary> llama_tensor_fit(const std::vector<llama_tensor_placement_input> & inputs,
                                                               const std::vector<llama_placement_tier> &         tiers,
                                                               std::string * err_msg = nullptr);
