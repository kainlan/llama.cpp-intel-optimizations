#include "kv-runtime-demotion.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace ggml_sycl {

kv_demotion_result plan_runtime_kv_demotion(const kv_demotion_input & in) {
    kv_demotion_result r;
    r.vram_bytes_after = in.vram_bytes;

    if (in.vram_bytes <= in.vram_budget) {
        r.fits = true;
        return r;
    }

    // One pass over the device-resident layers of one kind, latest first.
    const int n_layers    = (int) in.kv_device.size();
    auto      demote_pass = [&](bool swa_pass) {
        for (int l = n_layers - 1; l >= 0; --l) {
            if (in.kv_device[l] < 0) {
                continue;  // already host
            }
            const bool is_swa = l < (int) in.swa_layer_mask.size() && in.swa_layer_mask[l] != 0;
            if (is_swa != swa_pass) {
                continue;
            }
            // llama.cpp-3aos: THIS layer's own recorded bytes, not a
            // uniform figure applied to every full-attention layer -- 0 means
            // nothing is recorded for it (untracked, or a SHARED layer with no
            // independent KV to move), so there is nothing to demote.
            const size_t layer_bytes = l < (int) in.layer_kv_bytes.size() ? in.layer_kv_bytes[l] : 0;
            if (layer_bytes == 0) {
                continue;
            }
            if (layer_bytes > r.vram_bytes_after) {
                return;  // demoting would underflow vram_bytes_after; refuse rather than wrap
            }
            r.demoted_layers.push_back(l);
            r.vram_bytes_after -= layer_bytes;
            r.host_kv_bytes_added += layer_bytes;
            if (r.vram_bytes_after <= in.vram_budget) {
                return;
            }
        }
    };
    demote_pass(false);
    // SWA KV is small (~1.5 MB/layer), so demoting it buys little and costs a
    // split: only as a last resort, and only when the caller asks.
    if (r.vram_bytes_after > in.vram_budget && in.demote_swa) {
        demote_pass(true);
    }

    r.fits = r.vram_bytes_after <= in.vram_budget;
    return r;
}

size_t kv_weight_capacity(size_t vram_budget, size_t shared_zone_capacity) {
    return shared_zone_capacity > 0 ? std::min(vram_budget, shared_zone_capacity) : vram_budget;
}

kv_demotion_result plan_device_kv_fit(const kv_device_fit_input & in) {
    const size_t       n_layers = in.kv_device.size();
    kv_demotion_input  one;
    size_t             kv_bytes = 0;
    kv_demotion_result overflow;
    one.vram_budget = in.capacity;
    one.kv_device.assign(n_layers, -1);
    one.layer_kv_bytes.assign(n_layers, 0);
    one.swa_layer_mask = in.swa_layer_mask;
    one.demote_swa     = in.demote_swa;
    for (size_t l = 0; l < n_layers; ++l) {
        if (in.kv_device[l] != in.device) {
            continue;
        }
        const size_t bytes = l < in.layer_kv_bytes.size() ? in.layer_kv_bytes[l] : 0;
        if (bytes > SIZE_MAX - kv_bytes) {
            return overflow;  // fits == false: an unrepresentable total is not a fit
        }
        one.kv_device[l]      = in.device;
        one.layer_kv_bytes[l] = bytes;
        kv_bytes += bytes;
    }
    one.vram_bytes = kv_bytes;
    return plan_runtime_kv_demotion(one);
}

bool kv_shape_changed(const kv_shape & published, const kv_shape & next) {
    return !published.runtime || published.n_ctx != next.n_ctx || published.n_seq_max != next.n_seq_max ||
           published.kv_unified != next.kv_unified || published.swa_full != next.swa_full;
}

bool kv_residency_needs_refit(const kv_shape & published, const kv_shape & next, bool context_admitted) {
    return kv_shape_changed(published, next) || !context_admitted;
}

bool kv_device_residency_changed(const std::unordered_map<int, int> & load_kv_device,
                                 const std::unordered_map<int, int> & published_kv_device,
                                 const std::unordered_map<int, int> & next_kv_device,
                                 int                                  device) {
    auto owner = [](const std::unordered_map<int, int> & kv_device, int layer) {
        const auto it = kv_device.find(layer);
        return it == kv_device.end() ? -1 : it->second;
    };
    for (const auto & entry : load_kv_device) {
        if (entry.second == device && owner(published_kv_device, entry.first) != owner(next_kv_device, entry.first)) {
            return true;
        }
    }
    return false;
}

kv_residency_result plan_runtime_kv_residency(const kv_residency_input & in) {
    kv_residency_result r;
    r.kv_device = in.load_kv_device;
    kv_device_fit_input fit;
    fit.layer_kv_bytes = in.layer_kv_bytes;
    fit.swa_layer_mask = in.swa_layer_mask;
    fit.demote_swa     = true;  // never shrink context: place all of its KV rather than refuse
    for (size_t i = 0; i < in.devices.size(); ++i) {
        fit.device    = in.devices[i];
        fit.kv_device = r.kv_device;

        size_t n_resident = 0;
        size_t kv_bytes   = 0;
        for (size_t l = 0; l < r.kv_device.size(); ++l) {
            if (r.kv_device[l] == fit.device && l < in.layer_kv_bytes.size() && in.layer_kv_bytes[l] > 0) {
                ++n_resident;
                kv_bytes += std::min(in.layer_kv_bytes[l], SIZE_MAX - kv_bytes);
            }
        }
        const bool   overflow  = in.per_layer_slack > 0 && n_resident > SIZE_MAX / in.per_layer_slack;
        const size_t slack     = overflow ? SIZE_MAX : n_resident * in.per_layer_slack;
        const size_t demand    = kv_bytes + std::min(slack, SIZE_MAX - kv_bytes);
        const size_t available = i < in.available.size() ? in.available[i] : 0;
        size_t       yieldable = 0;
        if (i < in.optional_layouts.size()) {
            for (size_t bytes : in.optional_layouts[i].bytes) {
                yieldable += std::min(bytes, SIZE_MAX - yieldable);
            }
        }
        const size_t yield    = demand > available ? std::min(demand - available, yieldable) : 0;
        const size_t headroom = available + std::min(yield, SIZE_MAX - available);
        fit.capacity          = headroom > slack ? headroom - slack : 0;
        if (i < in.fit_capacity.size()) {
            fit.capacity = std::min(fit.capacity, in.fit_capacity[i]);
        }
        kv_optional_layout_yield picks;
        if (yield > 0) {
            // The copies' bytes are headroom only where a layer lands in them.
            picks = plan_optional_layout_yield(in.optional_layouts[i], kv_device_layer_alloc_bytes(in, fit.device));
            fit.capacity = std::min(fit.capacity, kv_device_leading_layer_bytes(in, fit.device, picks.kv_layers));
        }
        r.yield_bytes.push_back(yield);
        r.yields.push_back(std::move(picks));

        const kv_demotion_result demotion = plan_device_kv_fit(fit);
        for (int l : demotion.demoted_layers) {
            r.kv_device[static_cast<size_t>(l)] = -1;
        }
        r.per_device.push_back(demotion);
        if (!demotion.fits) {
            r.fits           = false;
            r.refused_device = fit.device;
            return r;
        }
    }
    return r;
}

size_t kv_layers_allocatable(kv_zone_model zone, const std::vector<size_t> & layer_bytes) {
    size_t placed = 0;
    for (size_t bytes : layer_bytes) {
        bool fits = false;
        for (tlsf_allocator & allocator : zone) {
            if (allocator.allocate(bytes) != SIZE_MAX) {
                fits = true;
                break;
            }
        }
        if (!fits) {
            break;
        }
        ++placed;
    }
    return placed;
}

kv_optional_layout_yield plan_optional_layout_yield(const kv_optional_layouts & copies,
                                                    const std::vector<size_t> & layer_bytes) {
    const kv_zone_model & zone = copies.zone;
    if (zone.empty()) {
        kv_optional_layout_yield none;
        none.kv_layers = layer_bytes.size();
        return none;
    }
    std::vector<size_t> order;
    for (size_t i = 0; i < copies.copies.size(); ++i) {
        if (copies.copies[i].allocator < zone.size()) {
            order.push_back(i);
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const kv_zone_block & x = copies.copies[a];
        const kv_zone_block & y = copies.copies[b];
        return x.allocator != y.allocator ? x.allocator > y.allocator : x.offset > y.offset;
    });
    auto freed_model = [&](const std::vector<size_t> & picks) {
        kv_zone_model model = zone;
        for (size_t i : picks) {
            model[copies.copies[i].allocator].free(copies.copies[i].offset);
        }
        return model;
    };

    kv_optional_layout_yield r;
    std::vector<size_t>      picked;
    std::vector<size_t>      pending;
    r.kv_layers = kv_layers_allocatable(zone, layer_bytes);
    for (size_t i : order) {
        if (r.kv_layers >= layer_bytes.size()) {
            break;
        }
        pending.push_back(i);
        std::vector<size_t> trial = picked;
        trial.insert(trial.end(), pending.begin(), pending.end());
        const size_t trial_placed = kv_layers_allocatable(freed_model(trial), layer_bytes);
        if (trial_placed <= r.kv_layers) {
            continue;
        }
        // Keep only the pending copies this gain needs: one that no longer
        // coalesces into the block a layer took is dropped again.
        for (size_t p = 0; p + 1 < pending.size();) {
            std::vector<size_t> without = picked;
            for (size_t q = 0; q < pending.size(); ++q) {
                if (q != p) {
                    without.push_back(pending[q]);
                }
            }
            if (kv_layers_allocatable(freed_model(without), layer_bytes) >= trial_placed) {
                pending.erase(pending.begin() + (std::ptrdiff_t) p);
            } else {
                ++p;
            }
        }
        picked.insert(picked.end(), pending.begin(), pending.end());
        r.groups.push_back(pending);
        pending.clear();
        r.kv_layers = trial_placed;
    }
    return r;
}

std::vector<size_t> kv_device_layer_alloc_bytes(const kv_residency_input & in, int device) {
    std::vector<size_t> bytes;
    for (size_t l = 0; l < in.load_kv_device.size() && l < in.layer_kv_bytes.size(); ++l) {
        if (in.load_kv_device[l] == device && in.layer_kv_bytes[l] > 0) {
            bytes.push_back(kv_layer_alloc_bytes(in.layer_kv_bytes[l]));
        }
    }
    return bytes;
}

size_t kv_device_leading_layer_bytes(const kv_residency_input & in, int device, size_t n_layers) {
    size_t bytes = 0;
    for (size_t l = 0, n = 0; l < in.load_kv_device.size() && l < in.layer_kv_bytes.size() && n < n_layers; ++l) {
        if (in.load_kv_device[l] == device && in.layer_kv_bytes[l] > 0) {
            bytes += std::min(in.layer_kv_bytes[l], SIZE_MAX - bytes);
            ++n;
        }
    }
    return bytes;
}

int layer_block_kv_device(const std::vector<int> &    kv_device,
                          const std::vector<size_t> & layer_kv_bytes,
                          int                         start_layer,
                          int                         end_layer) {
    const int n_layers = (int) std::min(kv_device.size(), layer_kv_bytes.size());
    int       owner    = -1;
    bool      seen     = false;
    for (int l = std::max(start_layer, 0); l <= end_layer && l < n_layers; ++l) {
        if (layer_kv_bytes[l] == 0) {
            continue;
        }
        if (!seen) {
            owner = kv_device[l];
            seen  = true;
        } else if (owner != kv_device[l]) {
            return -2;
        }
    }
    return owner;
}

size_t rs_buffer_bytes(const std::vector<rs_layer_desc> & layers, size_t pad_to, kv_row_size_fn row_size) {
    if (row_size == nullptr) {
        return 0;
    }
    size_t bytes = 0;
    for (const rs_layer_desc & layer : layers) {
        const size_t r_bytes = layer.n_embd_r > 0 ? row_size(layer.type_r, layer.n_embd_r) * layer.n_rows : 0;
        const size_t s_bytes = layer.n_embd_s > 0 ? row_size(layer.type_s, layer.n_embd_s) * layer.n_rows : 0;
        bytes += kv_pad_bytes(r_bytes, pad_to) + kv_pad_bytes(s_bytes, pad_to);
    }
    return bytes;
}

bool demand_slot_key_matches(const held_slot & held, const context_side_demand & demand, uint32_t index) {
    return held.device == demand.device && held.scope == demand.scope && held.owner == demand.owner &&
           held.index == index &&
           (held.cohort == demand.cohort ||
            (held.cohort != nullptr && demand.cohort != nullptr && std::strcmp(held.cohort, demand.cohort) == 0));
}

size_t demand_find_reusable_slot(const std::vector<held_slot> & held,
                                 const context_side_demand &    demand,
                                 uint32_t                       index,
                                 size_t                         need) {
    for (size_t h = 0; h < held.size(); ++h) {
        if (demand_slot_key_matches(held[h], demand, index) && held[h].cap >= need) {
            return h;
        }
    }
    return SIZE_MAX;
}

demand_reconciliation context_demand_reconcile(const std::vector<context_side_demand> & demands,
                                               const std::vector<held_slot> &           held) {
    demand_reconciliation r;
    std::vector<char>     named(held.size(), 0);
    for (size_t d = 0; d < demands.size(); ++d) {
        const context_side_demand & demand = demands[d];
        for (size_t i = 0; i < demand.slots.size(); ++i) {
            const uint32_t  index = static_cast<uint32_t>(i);
            const size_t    need  = demand.slots[i];
            demand_slot_ref ref;
            ref.record = d;
            ref.index  = index;
            ref.size   = need;
            for (size_t h = 0; h < held.size(); ++h) {
                if (demand_slot_key_matches(held[h], demand, index)) {
                    named[h] = 1;
                }
            }
            const size_t reuse = demand_find_reusable_slot(held, demand, index, need);
            if (reuse != SIZE_MAX) {
                ref.held = reuse;
                r.reused.push_back(ref);
                continue;
            }
            if (need == 0) {
                continue;  // nothing to carve for an index that needs no bytes
            }
            for (size_t h = 0; h < held.size(); ++h) {
                if (demand_slot_key_matches(held[h], demand, index)) {
                    ref.held = h;  // held, but too small: the growth supersedes it
                    break;
                }
            }
            (ref.held == SIZE_MAX ? r.carved : r.superseded).push_back(ref);
        }
    }
    for (size_t h = 0; h < held.size(); ++h) {
        if (named[h]) {
            continue;
        }
        for (const context_side_demand & demand : demands) {
            if (demand.device == held[h].device && demand.scope == held[h].scope && demand.owner == held[h].owner) {
                r.unused.push_back(h);
                break;
            }
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// kv_region_fit (llama.cpp-moua §2.4.1)
//
// The working model is a list of pieces per run.  A run is a low-to-high
// stretch of one TLSF made of FREE pieces (unallocated blocks), OPT pieces (a
// yieldable optional tenant) and BARRIER pieces (anything the fit may not
// touch).  A placement reserves bytes at the top of a FREE piece (`used`, with
// the assignments in carve order); a yield turns OPT pieces FREE and merges the
// FREE neighbours, together with what they had reserved.  The commit yields
// first and carves after, so the layout is a chain of top carves, in
// assignment order, on the blocks the yields leave: what allocate_below does.
// ---------------------------------------------------------------------------
namespace {

// The allocator's block grain: every block offset is a multiple of it, and a top
// carve needs the free block's top on it (carve_gap).
constexpr size_t fit_grain() {
    return tlsf_allocator::block_grain;
}

enum : uint8_t { PK_FREE, PK_OPT, PK_BARRIER };

enum : uint8_t { AS_HEAD, AS_KV, AS_AFTER };

struct fit_piece {
    size_t  lo   = 0;
    size_t  hi   = 0;
    uint8_t kind = PK_FREE;
    bool    soft = false;  // BARRIER only: free in the allocator (a pending or non-owned range), not an allocated block
    size_t  used = 0;      // FREE: bytes reserved at the top by assignments
    std::vector<int> assigns;  // FREE: assignment ids in carve order

    size_t size() const { return hi - lo; }

    size_t room() const { return hi - lo - used; }
};

struct fit_run {
    std::vector<fit_piece> pieces;  // low to high
    bool                   frontier = false;
};

struct fit_retained {
    size_t           lo   = 0;
    size_t           hi   = 0;
    size_t           used = 0;  // packed upward from lo
    std::vector<int> assigns;
};

struct fit_tlsf {
    bool                      takes_kv           = true;
    bool                      takes_weight_named = true;
    std::vector<fit_run>      runs;
    std::vector<fit_retained> retained;
    size_t                    yield_prefix = 0;
    std::vector<size_t>       buried;  // offsets of buried tenants released
};

struct fit_assign {
    uint8_t kind = AS_HEAD;
    size_t  id   = 0;  // head index, slot-table index or after-KV term
    size_t  size = 0;
    size_t  tlsf = SIZE_MAX;
};

struct fit_state {
    std::vector<fit_tlsf>   t;
    std::vector<fit_assign> a;
};

// One KV layer in slot-table order.
struct fit_slot {
    uint32_t      layer       = 0;
    kv_slot_group group       = KV_SLOT_FULL;
    size_t        kv_alloc    = 0;  // kv_layer_alloc_bytes(kv_bytes)
    size_t        total       = 0;  // the slot's size, sidecar included
    bool          sidecar     = false;
    bool          forced_host = false;
    bool          self_placed = false;
};

bool cohort_equal(const char * a, const char * b) {
    return a == b || (a != nullptr && b != nullptr && std::strcmp(a, b) == 0);
}

// Split `p` around [a, b): the overlap becomes a BARRIER, the rest keeps its
// kind.  An OPT piece the range touches is a BARRIER whole: a tenant that is
// partly under a pending range cannot be released (it is still an allocated
// block, so a piece under it stays carvable).
void cut_piece(std::vector<fit_piece> & out, const fit_piece & p, size_t a, size_t b) {
    if (b <= p.lo || a >= p.hi || p.kind == PK_BARRIER) {
        out.push_back(p);
        return;
    }
    if (p.kind == PK_OPT) {
        fit_piece q = p;
        q.kind      = PK_BARRIER;
        out.push_back(q);
        return;
    }
    const size_t cut_lo = std::max(a, p.lo);
    const size_t cut_hi = std::min(b, p.hi);
    if (cut_lo > p.lo) {
        fit_piece q = p;
        q.hi        = cut_lo;
        out.push_back(q);
    }
    fit_piece bar = p;
    bar.lo        = cut_lo;
    bar.hi        = cut_hi;
    bar.kind      = PK_BARRIER;
    bar.soft      = true;  // a range is not an allocated block: nothing carves under it
    out.push_back(bar);
    if (cut_hi < p.hi) {
        fit_piece q = p;
        q.lo        = cut_hi;
        out.push_back(q);
    }
}

// The pieces of a low-to-high list of blocks, with every range in `blocked`
// treated as allocated.
std::vector<fit_piece> pieces_from_blocks(const std::vector<zone_block> & low_to_high,
                                          const std::vector<zone_range> & blocked) {
    std::vector<fit_piece> pieces;
    for (const zone_block & b : low_to_high) {
        if (b.size == 0) {
            continue;
        }
        fit_piece p;
        p.lo   = b.offset;
        p.hi   = b.offset + b.size;
        p.kind = b.free ? PK_FREE : (b.optional_tenant && b.yieldable ? PK_OPT : PK_BARRIER);
        pieces.push_back(p);
    }
    for (const zone_range & r : blocked) {
        if (r.size == 0) {
            continue;
        }
        std::vector<fit_piece> next;
        for (const fit_piece & p : pieces) {
            cut_piece(next, p, r.offset, r.offset + r.size);
        }
        pieces.swap(next);
    }
    return pieces;
}

// The complement of `own` over the whole region: a commit re-fit may place only
// inside its own ranges, so everything else is allocated to it.
std::vector<zone_range> complement_ranges(std::vector<zone_range> own) {
    std::sort(own.begin(), own.end(), [](const zone_range & a, const zone_range & b) { return a.offset < b.offset; });
    std::vector<zone_range> out;
    size_t                  at = 0;
    for (const zone_range & r : own) {
        if (r.offset > at) {
            out.push_back({ at, r.offset - at });
        }
        at = std::max(at, r.offset + r.size);
    }
    out.push_back({ at, SIZE_MAX - at });
    return out;
}

// `r` minus every range in `blocked`: the fragments no blocked range covers.
std::vector<zone_range> clip_range(const zone_range & r, const std::vector<zone_range> & blocked) {
    std::vector<zone_range> frags;
    if (r.size != 0) {
        frags.push_back(r);
    }
    for (const zone_range & b : blocked) {
        if (b.size == 0) {
            continue;
        }
        std::vector<zone_range> next;
        for (const zone_range & f : frags) {
            const size_t f_hi = f.offset + f.size;
            const size_t b_hi = b.offset + b.size;
            if (b.offset >= f_hi || b_hi <= f.offset) {
                next.push_back(f);
                continue;
            }
            if (b.offset > f.offset) {
                next.push_back({ f.offset, b.offset - f.offset });
            }
            if (b_hi < f_hi) {
                next.push_back({ b_hi, f_hi - b_hi });
            }
        }
        frags.swap(next);
    }
    return frags;
}

void coalesce_free(fit_run & run) {
    std::vector<fit_piece> out;
    for (fit_piece & p : run.pieces) {
        if (p.size() == 0) {
            continue;
        }
        if (!out.empty() && out.back().kind == PK_FREE && p.kind == PK_FREE && out.back().hi == p.lo) {
            fit_piece & lower = out.back();
            lower.hi          = p.hi;
            lower.used += p.used;
            lower.assigns.insert(lower.assigns.end(), p.assigns.begin(), p.assigns.end());
            std::sort(lower.assigns.begin(), lower.assigns.end());
        } else {
            out.push_back(std::move(p));
        }
    }
    run.pieces.swap(out);
}

fit_state build_state(const shared_zone_geometry & g, const kv_region_request & r) {
    fit_state S;
    S.t.resize(g.tlsfs.size());
    for (size_t t = 0; t < g.tlsfs.size(); ++t) {
        const tlsf_geometry & in  = g.tlsfs[t];
        fit_tlsf &            out = S.t[t];
        out.takes_kv              = in.takes_kv;
        out.takes_weight_named    = in.takes_weight_named;

        std::vector<zone_range> blocked;
        for (const zone_pending_range & p : in.pending_ranges) {
            if (!(p.first_context && r.first_context)) {
                blocked.push_back({ p.offset, p.size });
            }
        }
        if (r.commit_plan != nullptr) {
            const std::vector<zone_range> outside = complement_ranges(in.own_ranges);
            blocked.insert(blocked.end(), outside.begin(), outside.end());
        }

        if (!in.frontier.empty()) {
            std::vector<zone_block> low_to_high(in.frontier.rbegin(), in.frontier.rend());
            fit_run                 run;
            run.frontier = true;
            run.pieces   = pieces_from_blocks(low_to_high, blocked);
            out.runs.push_back(std::move(run));
        }
        for (const std::vector<zone_block> & blocks : in.side_runs) {
            fit_run run;
            run.pieces = pieces_from_blocks(blocks, blocked);
            out.runs.push_back(std::move(run));
        }
        // The ladder ends at the first block that is not yieldable: releasing a
        // tenant below it would leave that block as a hole between free ones.
        for (fit_run & run : out.runs) {
            if (!run.frontier) {
                continue;
            }
            bool ended = false;
            for (size_t i = run.pieces.size(); i-- > 0;) {
                if (run.pieces[i].kind == PK_BARRIER) {
                    ended = true;
                } else if (ended && run.pieces[i].kind == PK_OPT) {
                    run.pieces[i].kind = PK_BARRIER;
                }
            }
        }
        if (r.commit_plan != nullptr) {
            // No yield on a re-fit: the ranges are the whole room.
            for (fit_run & run : out.runs) {
                for (fit_piece & p : run.pieces) {
                    if (p.kind == PK_OPT) {
                        p.kind = PK_BARRIER;
                    }
                }
            }
        }
        for (fit_run & run : out.runs) {
            coalesce_free(run);
        }
        // A retained run is claimed whole, with no carve, so it obeys the same ranges
        // as everything else: what another transaction's pending range covers, and
        // what a commit re-fit does not own, is allocated to this call.
        for (const zone_range & rr : in.retained_runs) {
            for (const zone_range & frag : clip_range(rr, blocked)) {
                fit_retained ret;
                ret.lo = frag.offset;
                ret.hi = frag.offset + frag.size;
                out.retained.push_back(ret);
            }
        }
    }
    return S;
}

// Add FREE piece `p` to `t`, at the top of the run it continues, under the run it
// is the top neighbour of, or as a run of its own, and coalesce.
void add_free_piece(fit_tlsf & t, const fit_piece & p) {
    for (fit_run & run : t.runs) {
        if (!run.pieces.empty() && run.pieces.back().hi == p.lo) {
            run.pieces.push_back(p);
            coalesce_free(run);
            return;
        }
        if (!run.pieces.empty() && run.pieces.front().lo == p.hi) {
            run.pieces.insert(run.pieces.begin(), p);
            coalesce_free(run);
            return;
        }
    }
    fit_run run;
    run.pieces.push_back(p);
    t.runs.push_back(std::move(run));
}

bool eligible(const fit_tlsf & t, int route) {
    return route == 0 ? t.takes_kv : t.takes_weight_named;
}

// The gap of a frontier run: its topmost piece, when that is a FREE one.
bool gap_index(const fit_run & run, size_t & idx) {
    if (!run.frontier || run.pieces.empty() || run.pieces.back().kind != PK_FREE) {
        return false;
    }
    idx = run.pieces.size() - 1;
    return true;
}

// Whether the commit can carve at the top of FREE piece `idx`.  allocate_below
// takes the top of the free block directly under an ALLOCATED block, or the region
// end: the piece above must be an optional tenant or a hard barrier, or the piece
// must top the run.  A pending or non-owned range above is not an allocated block
// (the allocator does not know it), so a carve under it would land where the
// allocator puts it, not where the fit says; a top off the allocator's grain
// cannot be top-carved at all.
bool piece_carvable(const fit_run & run, size_t idx) {
    if (run.pieces[idx].hi % fit_grain() != 0) {
        return false;
    }
    if (idx + 1 >= run.pieces.size()) {
        return true;
    }
    const fit_piece & above = run.pieces[idx + 1];
    return !(above.kind == PK_BARRIER && above.soft);
}

// Whether the optional tenant at `idx` can contribute room: yielding it merges the
// FREE/OPT sequence it sits in into one block topped by the sequence's highest
// piece, and the commit carves in that block only when that top is carvable.
// pick_yield asks piece_carvable of the same top piece.
bool opt_reachable(const fit_run & run, size_t idx) {
    size_t top = idx;
    while (top + 1 < run.pieces.size() && run.pieces[top + 1].kind != PK_BARRIER) {
        ++top;
    }
    return piece_carvable(run, top);
}

struct fit_room {
    size_t tlsf     = SIZE_MAX;
    bool   retained = false;
    size_t run      = 0;  // retained: the room index
    size_t idx      = 0;  // the piece index
};

// One room a slot could take, in the tier order of §2.4.1.
struct fit_candidate_room {
    fit_room room;
    int      tier = 0;
    size_t   free = 0;
};

// Every tier 1-3 room that can hold `size` for `route`, best first: the lowest tier,
// then the smallest room (the lowest TLSF, run and piece at a tie).  A slot is never
// split, so a room that cannot hold it whole is not a candidate.
void candidate_rooms(const fit_state & S, size_t size, int route, std::vector<fit_candidate_room> & out) {
    out.clear();
    for (size_t t = 0; t < S.t.size(); ++t) {
        if (!eligible(S.t[t], route)) {
            continue;
        }
        for (size_t i = 0; i < S.t[t].retained.size(); ++i) {
            const fit_retained & r = S.t[t].retained[i];
            const size_t         f = r.hi - r.lo - r.used;
            if (f >= size) {
                fit_candidate_room c;
                c.room.tlsf     = t;
                c.room.retained = true;
                c.room.run      = i;
                c.tier          = 1;
                c.free          = f;
                out.push_back(c);
            }
        }
        for (size_t ri = 0; ri < S.t[t].runs.size(); ++ri) {
            const fit_run & run = S.t[t].runs[ri];
            size_t          gap = SIZE_MAX;
            size_t          gi  = 0;
            if (gap_index(run, gi)) {
                gap = gi;
            }
            for (size_t pi = 0; pi < run.pieces.size(); ++pi) {
                const fit_piece & p = run.pieces[pi];
                if (p.kind != PK_FREE || p.room() < size || !piece_carvable(run, pi)) {
                    continue;
                }
                fit_candidate_room c;
                c.room.tlsf = t;
                c.room.run  = ri;
                c.room.idx  = pi;
                c.tier      = pi == gap ? 2 : 3;
                c.free      = p.room();
                out.push_back(c);
            }
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const fit_candidate_room & a, const fit_candidate_room & b) {
        return a.tier != b.tier ? a.tier < b.tier : a.free < b.free;
    });
}

// The cheapest tier 1-3 room for `size`: the first candidate (best fit within a tier).
bool find_room(const fit_state & S, size_t size, int route, fit_room & out) {
    std::vector<fit_candidate_room> rooms;
    candidate_rooms(S, size, route, rooms);
    if (rooms.empty()) {
        return false;
    }
    out = rooms.front().room;
    return true;
}

void carve_room(fit_state & S, const fit_room & room, int assign_id, size_t size) {
    fit_tlsf & t        = S.t[room.tlsf];
    S.a[assign_id].tlsf = room.tlsf;
    if (room.retained) {
        fit_retained & r = t.retained[room.run];
        r.used += size;
        r.assigns.push_back(assign_id);
        return;
    }
    fit_piece & p = t.runs[room.run].pieces[room.idx];
    p.used += size;
    p.assigns.push_back(assign_id);
}

struct fit_candidate {
    size_t              tlsf = 0;
    size_t              run  = 0;
    std::vector<size_t> release;  // piece indices of the optional tenants
    size_t              lost     = 0;
    size_t              gained   = 0;
    bool                frontier = false;
};

size_t slots_gained(size_t span, const std::vector<size_t> & remaining, size_t from) {
    size_t acc = 0;
    size_t n   = 0;
    for (size_t i = from; i < remaining.size(); ++i) {
        if (acc + remaining[i] > span) {
            break;
        }
        acc += remaining[i];
        ++n;
    }
    return n > 0 ? n : 1;
}

// Tier 4: the cheapest yield that makes room for `size`, ranked by layout bytes
// lost per slot gained, the frontier before a buried run at a tie.  Each run's
// candidate is the smallest set of its releasable tenants, from the top down,
// whose release forms a free span for the slot.
bool pick_yield(const fit_state &           S,
                size_t                      size,
                int                         route,
                const std::vector<size_t> & remaining,
                size_t                      from,
                fit_candidate &             best) {
    bool have = false;
    for (size_t t = 0; t < S.t.size(); ++t) {
        if (!eligible(S.t[t], route)) {
            continue;
        }
        for (size_t ri = 0; ri < S.t[t].runs.size(); ++ri) {
            const fit_run & run            = S.t[t].runs[ri];
            bool            first_sequence = run.frontier;
            size_t          i              = run.pieces.size();
            while (i > 0) {
                const size_t  top = i - 1;
                const uint8_t k   = run.pieces[top].kind;
                if (k == PK_BARRIER) {
                    first_sequence = false;  // the ladder ends at a block it cannot release
                    --i;
                    continue;
                }
                size_t end = top;
                while (end > 0 && (run.pieces[end - 1].kind == PK_FREE || run.pieces[end - 1].kind == PK_OPT)) {
                    --end;
                }
                // The freed span merges into one block topped by piece `top`; if what
                // lies above that is a pending or non-owned range, nothing carves in it.
                if (!piece_carvable(run, top)) {
                    first_sequence = false;
                    i              = end;
                    continue;
                }
                std::vector<size_t> released;
                size_t              span = 0;
                size_t              lost = 0;
                for (size_t j = top + 1; j-- > end;) {
                    span += run.pieces[j].kind == PK_OPT ? run.pieces[j].size() : run.pieces[j].room();
                    if (run.pieces[j].kind != PK_OPT) {
                        continue;
                    }
                    released.push_back(j);
                    lost += run.pieces[j].size();
                    const size_t extra = (j > end && run.pieces[j - 1].kind == PK_FREE) ? run.pieces[j - 1].room() : 0;
                    if (span + extra >= size) {
                        fit_candidate c;
                        c.tlsf      = t;
                        c.run       = ri;
                        c.release   = released;
                        c.lost      = lost;
                        c.gained    = slots_gained(span + extra, remaining, from);
                        c.frontier  = first_sequence;
                        bool better = !have;
                        if (have) {
                            const size_t l = c.lost * best.gained;
                            const size_t r = best.lost * c.gained;
                            better         = l < r || (l == r && c.frontier && !best.frontier);
                        }
                        if (better) {
                            best = c;
                            have = true;
                        }
                        break;
                    }
                }
                first_sequence = false;
                i              = end;
            }
        }
    }
    return have;
}

void apply_yield(fit_state & S, const fit_candidate & c) {
    fit_tlsf & t   = S.t[c.tlsf];
    fit_run &  run = t.runs[c.run];
    for (size_t j : c.release) {
        run.pieces[j].kind = PK_FREE;
        if (!c.frontier) {
            t.buried.push_back(run.pieces[j].lo);
        }
    }
    if (c.frontier) {
        t.yield_prefix += c.release.size();
    }
    coalesce_free(run);
}

// Place one assignment.  Head slots use tiers 1-3 only; KV may yield.  Returns
// false when nothing fits.
bool place(fit_state &                 S,
           int                         assign_id,
           int                         route,
           bool                        allow_yield,
           const std::vector<size_t> & remaining,
           size_t                      from) {
    const size_t size = S.a[assign_id].size;
    fit_room     room;
    if (find_room(S, size, route, room)) {
        carve_room(S, room, assign_id, size);
        return true;
    }
    if (!allow_yield) {
        return false;
    }
    fit_candidate c;
    if (!pick_yield(S, size, route, remaining, from, c)) {
        return false;
    }
    apply_yield(S, c);
    if (!find_room(S, size, route, room)) {
        return false;
    }
    carve_room(S, room, assign_id, size);
    return true;
}

size_t room_total(const fit_state & S, size_t t) {
    size_t total = 0;
    for (const fit_retained & r : S.t[t].retained) {
        total += r.hi - r.lo - r.used;
    }
    for (const fit_run & run : S.t[t].runs) {
        for (size_t pi = 0; pi < run.pieces.size(); ++pi) {
            const fit_piece & p = run.pieces[pi];
            if (p.kind == PK_OPT) {
                total += opt_reachable(run, pi) ? p.size() : 0;
            } else if (p.kind == PK_FREE && piece_carvable(run, pi)) {
                total += p.room();
            }
        }
    }
    return total;
}

size_t tier13_free(const fit_state & S, size_t t) {
    size_t total = 0;
    for (const fit_retained & r : S.t[t].retained) {
        total += r.hi - r.lo - r.used;
    }
    for (const fit_run & run : S.t[t].runs) {
        for (size_t pi = 0; pi < run.pieces.size(); ++pi) {
            if (run.pieces[pi].kind == PK_FREE && piece_carvable(run, pi)) {
                total += run.pieces[pi].room();
            }
        }
    }
    return total;
}

// The demotion order of §2.4.1 over slot indices: the highest-index full-attention
// layer first, then the SWA layers the same way.
std::vector<size_t> demotion_order(const std::vector<fit_slot> & slots) {
    std::vector<size_t> order;
    for (int group = KV_SLOT_FULL; group <= KV_SLOT_SWA; ++group) {
        for (size_t i = slots.size(); i-- > 0;) {
            if (slots[i].group == group && !slots[i].forced_host && !slots[i].self_placed) {
                order.push_back(i);
            }
        }
    }
    return order;
}

struct kv_solve {
    bool              ok = false;
    std::vector<char> device;  // per slot
    fit_state         state;
    size_t            demoted = 0;
};

// The demotion loop of §2.4.1: ask the fit; on a miss demote the highest-index
// full-attention layer (K and V and sidecar together), then the SWA layers, and
// ask again.  `S0` already holds the head slots.
kv_solve solve_kv(const fit_state & S0, const std::vector<fit_slot> & slots) {
    kv_solve out;
    out.device.assign(slots.size(), 1);
    const std::vector<size_t> order = demotion_order(slots);
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].forced_host) {
            out.device[i] = 0;
        }
    }
    size_t next = 0;
    for (;;) {
        fit_state           S  = S0;
        bool                ok = true;
        std::vector<size_t> remaining;
        std::vector<size_t> slot_of;
        for (size_t i = 0; i < slots.size(); ++i) {
            if (out.device[i] && !slots[i].self_placed) {
                remaining.push_back(slots[i].total);
                slot_of.push_back(i);
            }
        }
        for (size_t n = 0; n < slot_of.size() && ok; ++n) {
            S.a.push_back({ AS_KV, slot_of[n], slots[slot_of[n]].total, SIZE_MAX });
            ok = place(S, (int) S.a.size() - 1, 0, true, remaining, n);
        }
        if (ok) {
            out.ok    = true;
            out.state = std::move(S);
            return out;
        }
        if (next >= order.size()) {
            return out;  // unreachable: with no device layer left nothing can miss
        }
        out.device[order[next++]] = 0;
        ++out.demoted;
    }
}

// The slot table: full attention first, then SWA, each in layer order.
std::vector<fit_slot> build_slots(const kv_region_request & r) {
    std::vector<fit_slot> slots;
    for (const kv_layer_slot_request & l : r.layers) {
        fit_slot s;
        s.layer    = l.layer;
        s.group    = l.group;
        s.kv_alloc = kv_layer_alloc_bytes(l.kv_bytes);
        s.sidecar  = l.sidecar_bytes != 0;
        s.total    = s.kv_alloc + (s.sidecar ? kv_layer_alloc_bytes(l.sidecar_bytes) : 0);
        for (uint32_t f : r.forced_host) {
            s.forced_host = s.forced_host || f == l.layer;
        }
        for (const kv_self_extent & e : r.self_extents) {
            for (const kv_self_slot & ss : e.slots) {
                s.self_placed = s.self_placed || ss.layer == l.layer;
            }
        }
        slots.push_back(s);
    }
    std::stable_sort(slots.begin(), slots.end(), [](const fit_slot & a, const fit_slot & b) {
        return a.group != b.group ? a.group < b.group : a.layer < b.layer;
    });
    return slots;
}

// Best fit over every TLSF a head may take lets an unconstrained head use up the
// tight room the one TLSF a constrained head needs offers, and the constrained
// head is then refused where an index-order placement would have fitted both.  So
// the heads with the fewest admissible TLSFs go first (request order at a tie),
// and best fit decides only inside each head's own admissible set.
std::vector<size_t> head_placement_order(const shared_zone_geometry & g, const kv_region_request & r) {
    size_t admissible[2] = { 0, 0 };
    for (const tlsf_geometry & t : g.tlsfs) {
        admissible[0] += t.takes_kv ? 1 : 0;
        admissible[1] += t.takes_weight_named ? 1 : 0;
    }
    std::vector<size_t> order(r.head_slots.size());
    for (size_t h = 0; h < order.size(); ++h) {
        order[h] = h;
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return admissible[r.head_slots[a].names_weight ? 1 : 0] < admissible[r.head_slots[b].names_weight ? 1 : 0];
    });
    return order;
}

struct verify_span {
    size_t lo = 0;
    size_t hi = 0;
};

// Merge touching spans so a range that crosses two of them is one.
void merge_spans(std::vector<verify_span> & v) {
    std::sort(v.begin(), v.end(), [](const verify_span & a, const verify_span & b) { return a.lo < b.lo; });
    std::vector<verify_span> out;
    for (const verify_span & s : v) {
        if (s.hi <= s.lo) {
            continue;
        }
        if (!out.empty() && out.back().hi >= s.lo) {
            out.back().hi = std::max(out.back().hi, s.hi);
        } else {
            out.push_back(s);
        }
    }
    v.swap(out);
}

bool spans_hold(const std::vector<verify_span> & v, size_t lo, size_t hi) {
    for (const verify_span & s : v) {
        if (s.lo <= lo && hi <= s.hi) {
            return true;
        }
    }
    return false;
}

// `v` less `taken`.
std::vector<verify_span> spans_minus(const std::vector<verify_span> & v, const std::vector<verify_span> & taken) {
    std::vector<verify_span> out;
    for (const verify_span & s : v) {
        size_t at = s.lo;
        for (const verify_span & t : taken) {
            if (t.hi <= at || t.lo >= s.hi) {
                continue;
            }
            if (t.lo > at) {
                out.push_back({ at, t.lo });
            }
            at = std::max(at, t.hi);
        }
        if (at < s.hi) {
            out.push_back({ at, s.hi });
        }
    }
    return out;
}

// §2.4.2 step 6, the commit re-fit.  Inside this call's own ranges nothing can have
// changed since the plan but a shortfall: another transaction's range or a weight
// that took part of a room the plan reserved.  So the re-fit does not solve again, it
// verifies the plan.  Solving again from the shrunk rooms is a second source for a
// fact the plan already fixed: best fit over rooms the size of the plan's usage picks
// differently from the plan's, and can place what the plan declined.
//
// Every planned head slot, KV layer slot and admitted after-KV term is a
// (TLSF, offset, bytes) extent in the plan.  One that is still wholly free in the
// re-snapshot stays exactly where it was.  A layer slot that is not free demotes
// that layer, and every survivor keeps its planned offset, which the commit's
// offset-fixed carve honours.  A head slot that is not free is the one thing
// re-placed: by best fit in the room the survivors leave, demoting layers in the
// demotion order until it fits.  Nothing is added: no layer the plan demoted, no
// after-KV term it declined.
kv_region_fit_result commit_verify(const shared_zone_geometry &  g,
                                   const kv_region_request &     r,
                                   const std::vector<fit_slot> & slots) {
    const kv_region_fit_result & plan   = *r.commit_plan;
    const size_t                 n_tlsf = g.tlsfs.size();
    kv_region_fit_result         res    = plan;
    res.yield_prefix.assign(n_tlsf, 0);
    res.buried_released.clear();
    res.tlsf_free.assign(n_tlsf, 0);
    res.refused_heads.clear();
    if (!plan.fits || plan.layers.size() != slots.size()) {
        res.fits          = false;
        res.refused_heads = plan.refused_heads;
        return res;
    }

    // The room this call owns, as the re-snapshot sees it: free blocks, and retained
    // runs a placement claims whole.
    const fit_state                       S = build_state(g, r);
    std::vector<std::vector<verify_span>> room_free(n_tlsf);
    std::vector<std::vector<verify_span>> room_retained(n_tlsf);
    for (size_t t = 0; t < n_tlsf; ++t) {
        for (const fit_run & run : S.t[t].runs) {
            for (const fit_piece & p : run.pieces) {
                if (p.kind == PK_FREE) {
                    room_free[t].push_back({ p.lo, p.hi });
                }
            }
        }
        for (const fit_retained & rr : S.t[t].retained) {
            room_retained[t].push_back({ rr.lo, rr.hi });
        }
        merge_spans(room_free[t]);
        merge_spans(room_retained[t]);
    }
    auto still_free = [&](size_t t, size_t lo, size_t hi) {
        return spans_hold(room_free[t], lo, hi) || spans_hold(room_retained[t], lo, hi);
    };

    // Layer slots.  A self-placed slot is already carved and is not the re-fit's.
    std::vector<char>   device(slots.size(), 0);
    std::vector<size_t> slot_lo(slots.size(), 0);
    for (size_t i = 0; i < slots.size(); ++i) {
        const kv_layer_placement & lp = plan.layers[i];
        if (!lp.device) {
            continue;
        }
        const kv_region_extent & e = plan.extents[lp.extent];
        slot_lo[i]                 = e.offset + lp.slot_offset;
        device[i]                  = e.kind == KV_EXTENT_SELF || still_free(e.tlsf, slot_lo[i], slot_lo[i] + lp.size);
    }
    std::vector<kv_demotion_cause> cause(slots.size(), KV_DEMOTE_NONE);
    for (size_t i = 0; i < slots.size(); ++i) {
        if (plan.layers[i].device && !device[i]) {
            cause[i] = KV_DEMOTE_CAPACITY;
        }
    }

    // Head slots and after-KV terms, the plan's own.  A head served by a reservation
    // or settled earlier carries no carve op and is not the re-fit's.
    struct head_state {
        bool   planned  = false;  // the plan carved it
        bool   placed   = false;
        bool   retained = false;
        size_t tlsf     = 0;
        size_t offset   = 0;
        size_t size     = 0;
    };

    std::vector<head_state> heads(plan.heads.size());
    for (const kv_carve_op & op : plan.carve_order) {
        if (op.kind != KV_CARVE_HEAD) {
            continue;
        }
        head_state & hs = heads[op.index];
        hs.planned      = true;
        hs.retained     = !op.carve;
        hs.tlsf         = op.tlsf;
        hs.offset       = op.offset;
        hs.size         = op.size;
        hs.placed       = still_free(op.tlsf, op.offset, op.offset + op.size);
    }
    std::vector<char> term_ok(plan.after_kv.size(), 0);
    for (size_t k = 0; k < plan.after_kv.size(); ++k) {
        const kv_after_kv_placement & ap        = plan.after_kv[k];
        bool                          on_device = false;
        for (size_t i = 0; i < slots.size(); ++i) {
            on_device = on_device || (slots[i].layer == r.after_kv[k].layer && device[i]);
        }
        term_ok[k] = ap.admitted && on_device && still_free(ap.tlsf, ap.offset, ap.offset + ap.size);
    }

    // What the survivors, the placed heads and the admitted terms occupy, per TLSF.
    auto occupied = [&](size_t t) {
        std::vector<verify_span> v;
        for (size_t i = 0; i < slots.size(); ++i) {
            if (device[i] && plan.extents[plan.layers[i].extent].tlsf == t) {
                v.push_back({ slot_lo[i], slot_lo[i] + plan.layers[i].size });
            }
        }
        for (const head_state & hs : heads) {
            if (hs.planned && hs.placed && hs.tlsf == t) {
                v.push_back({ hs.offset, hs.offset + hs.size });
            }
        }
        for (size_t k = 0; k < plan.after_kv.size(); ++k) {
            if (term_ok[k] && plan.after_kv[k].tlsf == t) {
                v.push_back({ plan.after_kv[k].offset, plan.after_kv[k].offset + plan.after_kv[k].size });
            }
        }
        // Not merged: spans_minus walks `taken` in offset order only.
        std::sort(v.begin(), v.end(), [](const verify_span & a, const verify_span & b) { return a.lo < b.lo; });
        return v;
    };

    // The displaced heads, constrained first like the plan, each by best fit.
    const std::vector<size_t> demote = demotion_order(slots);
    size_t                    next   = 0;
    std::vector<size_t>       refused;
    for (const size_t h : head_placement_order(g, r)) {
        head_state & hs = heads[h];
        if (!hs.planned || hs.placed) {
            continue;
        }
        const bool names_weight = r.head_slots[h].names_weight;
        for (;;) {
            // Retained room first, then the smallest free room that holds it: the
            // plan's tier order, over what the survivors leave.
            bool   found     = false;
            bool   retained  = false;
            size_t best_t    = 0;
            size_t best_lo   = 0;
            size_t best_room = SIZE_MAX;
            for (size_t t = 0; t < n_tlsf; ++t) {
                if (!(names_weight ? g.tlsfs[t].takes_weight_named : g.tlsfs[t].takes_kv)) {
                    continue;
                }
                const std::vector<verify_span> taken = occupied(t);
                for (int pass = 0; pass < 2; ++pass) {
                    const bool                     from_retained = pass == 0;
                    const std::vector<verify_span> spans =
                        spans_minus(from_retained ? room_retained[t] : room_free[t], taken);
                    for (const verify_span & s : spans) {
                        // A free room is carved at its top, on the allocator's grain;
                        // a retained run packs upward from its bottom.
                        size_t at = s.lo;
                        if (!from_retained) {
                            if (s.hi - s.lo < hs.size) {
                                continue;
                            }
                            at = (s.hi - hs.size) / tlsf_allocator::block_grain * tlsf_allocator::block_grain;
                        }
                        if (at < s.lo || at + hs.size > s.hi) {
                            continue;
                        }
                        const bool better = !found || (from_retained && !retained) ||
                                            (from_retained == retained && s.hi - s.lo < best_room);
                        if (better) {
                            found     = true;
                            retained  = from_retained;
                            best_t    = t;
                            best_lo   = at;
                            best_room = s.hi - s.lo;
                        }
                    }
                }
            }
            if (found) {
                hs.placed   = true;
                hs.retained = retained;
                hs.tlsf     = best_t;
                hs.offset   = best_lo;
                break;
            }
            // Nothing holds it: demote the next surviving layer, which frees its slot.
            while (next < demote.size() && !device[demote[next]]) {
                ++next;
            }
            if (next >= demote.size()) {
                refused.push_back(h);
                break;
            }
            device[demote[next]] = 0;
            cause[demote[next]]  = KV_DEMOTE_HEAD_SLOT;
            ++next;
            for (size_t k = 0; k < plan.after_kv.size(); ++k) {
                if (r.after_kv[k].layer == slots[demote[next - 1]].layer) {
                    term_ok[k] = 0;
                }
            }
        }
    }

    // Rebuild the result from the survivors.  Each plan extent becomes the maximal
    // runs of surviving slots in it, at the planned offsets.
    res.extents.clear();
    std::vector<std::vector<size_t>> runs_of(plan.extents.size());
    std::vector<size_t>              self_index(plan.extents.size(), SIZE_MAX);
    for (size_t ei = 0; ei < plan.extents.size(); ++ei) {
        const kv_region_extent & e = plan.extents[ei];
        if (e.kind == KV_EXTENT_SELF) {
            self_index[ei] = res.extents.size();
            res.extents.push_back(e);
            continue;
        }
        std::vector<size_t> members;
        for (size_t i = 0; i < slots.size(); ++i) {
            if (plan.layers[i].device && plan.layers[i].extent == ei) {
                members.push_back(i);
            }
        }
        std::sort(members.begin(), members.end(),
                  [&](size_t a, size_t b) { return plan.layers[a].slot_offset < plan.layers[b].slot_offset; });
        size_t at   = 0;
        bool   open = false;
        for (size_t i : members) {
            if (!device[i]) {
                open = false;
                continue;
            }
            if (!open) {
                kv_region_extent run;
                run.tlsf   = e.tlsf;
                run.offset = slot_lo[i];
                run.size   = 0;
                run.kind   = e.kind;
                res.extents.push_back(run);
                runs_of[ei].push_back(res.extents.size() - 1);
                at   = 0;
                open = true;
            }
            kv_layer_placement & lp = res.layers[i];
            lp.extent               = res.extents.size() - 1;
            lp.slot_offset          = at;
            lp.sidecar_offset       = slots[i].sidecar ? at + slots[i].kv_alloc : SIZE_MAX;
            at += slots[i].total;
            res.extents.back().size = at;
        }
    }
    for (size_t i = 0; i < slots.size(); ++i) {
        kv_layer_placement & lp = res.layers[i];
        if (plan.layers[i].device && plan.extents[plan.layers[i].extent].kind == KV_EXTENT_SELF) {
            lp.extent = self_index[plan.layers[i].extent];
        }
        if (plan.layers[i].device && !device[i]) {
            lp.device         = false;
            lp.cause          = cause[i];
            lp.extent         = SIZE_MAX;
            lp.slot_offset    = 0;
            lp.sidecar_offset = SIZE_MAX;
        }
    }
    for (size_t h = 0; h < heads.size(); ++h) {
        if (heads[h].planned && heads[h].placed) {
            res.heads[h].tlsf   = heads[h].tlsf;
            res.heads[h].offset = heads[h].offset;
        }
    }
    for (size_t k = 0; k < plan.after_kv.size(); ++k) {
        res.after_kv[k].admitted = term_ok[k] != 0;
    }
    res.carve_order.clear();
    for (const kv_carve_op & op : plan.carve_order) {
        if (op.kind == KV_CARVE_EXTENT) {
            std::vector<size_t> runs = runs_of[op.index];
            if (op.carve) {
                std::reverse(runs.begin(), runs.end());  // a carve chain runs from the top down
            }
            for (size_t ni : runs) {
                kv_carve_op o = op;
                o.index       = ni;
                o.offset      = res.extents[ni].offset;
                o.size        = res.extents[ni].size;
                o.demand      = res.extents[ni].size;
                res.carve_order.push_back(o);
            }
        } else if (op.kind == KV_CARVE_HEAD) {
            const head_state & hs = heads[op.index];
            if (hs.placed) {
                kv_carve_op o = op;
                o.tlsf        = hs.tlsf;
                o.offset      = hs.offset;
                o.size        = hs.size;
                o.demand      = hs.size;
                o.carve       = !hs.retained;
                res.carve_order.push_back(o);
            }
        } else if (term_ok[op.index]) {
            res.carve_order.push_back(op);
        }
    }

    res.refused_heads = refused;
    std::sort(res.refused_heads.begin(), res.refused_heads.end());
    res.fits = refused.empty();
    if (!res.fits) {
        for (size_t t = 0; t < n_tlsf; ++t) {
            const std::vector<verify_span> taken = occupied(t);
            for (const verify_span & s : spans_minus(room_free[t], taken)) {
                res.tlsf_free[t] += s.hi - s.lo;
            }
            for (const verify_span & s : spans_minus(room_retained[t], taken)) {
                res.tlsf_free[t] += s.hi - s.lo;
            }
        }
    }
    return res;
}

}  // namespace

kv_region_fit_result kv_region_fit(const shared_zone_geometry & g, const kv_region_request & r) {
    const std::vector<fit_slot> slots = build_slots(r);
    if (r.commit_plan != nullptr) {
        return commit_verify(g, r, slots);
    }
    kv_region_fit_result res;
    const size_t         n_tlsf = g.tlsfs.size();
    res.yield_prefix.assign(n_tlsf, 0);
    res.free_after_full_kv.assign(n_tlsf, 0);
    res.tlsf_free.assign(n_tlsf, 0);

    fit_state base = build_state(g, r);

    // Head slots, in request order, before any KV.  A head slot is served in
    // place by a reserved slot of the same key with cap >= the planned size.
    res.heads.resize(r.head_slots.size());
    std::vector<std::vector<char>> reservation_used(n_tlsf);
    for (size_t t = 0; t < n_tlsf; ++t) {
        reservation_used[t].assign(g.tlsfs[t].reservations.size(), 0);
    }
    std::vector<size_t>       head_assign(r.head_slots.size(), SIZE_MAX);
    const std::vector<size_t> no_remaining;

    const std::vector<size_t> head_order = head_placement_order(g, r);
    for (const size_t h : head_order) {
        const kv_head_slot_request & hs   = r.head_slots[h];
        const size_t                 need = kv_layer_alloc_bytes(hs.size);
        kv_head_placement &          out  = res.heads[h];
        out.head                          = h;
        out.size                          = need;
        bool settled                      = false;
        for (const kv_self_head & sh : r.self_heads) {
            if (sh.head == h) {
                out.tlsf   = sh.tlsf;
                out.offset = sh.offset;
                out.size   = sh.size;
                out.reused = false;
                settled    = true;
            }
        }
        if (settled) {
            continue;
        }
        size_t super_t = SIZE_MAX;
        size_t super_i = SIZE_MAX;
        for (size_t t = 0; t < n_tlsf && !settled; ++t) {
            for (size_t i = 0; i < g.tlsfs[t].reservations.size(); ++i) {
                const zone_reservation & rv = g.tlsfs[t].reservations[i];
                if (reservation_used[t][i] || rv.scope != hs.scope || rv.owner != hs.owner || rv.index != hs.index ||
                    !cohort_equal(rv.cohort, hs.cohort)) {
                    continue;
                }
                if (rv.size >= need) {
                    reservation_used[t][i] = 1;
                    out.reused             = true;
                    out.reservation        = i;
                    out.tlsf               = t;
                    out.offset             = rv.offset;
                    out.size               = rv.size;
                    settled                = true;
                    break;
                }
                if (super_t == SIZE_MAX) {
                    super_t = t;
                    super_i = i;
                }
            }
        }
        if (settled) {
            continue;
        }
        base.a.push_back({ AS_HEAD, h, need, SIZE_MAX });
        const int id   = (int) base.a.size() - 1;
        head_assign[h] = (size_t) id;
        if (!place(base, id, hs.names_weight ? 1 : 0, false, no_remaining, 0)) {
            res.refused_heads.push_back(h);
            continue;
        }
        if (super_t != SIZE_MAX) {
            reservation_used[super_t][super_i] = 1;
            res.superseded.push_back({ super_t, super_i });
        }
    }
    std::sort(res.refused_heads.begin(), res.refused_heads.end());
    if (!res.refused_heads.empty()) {
        for (size_t t = 0; t < n_tlsf; ++t) {
            res.tlsf_free[t] = tier13_free(base, t);
        }
        return res;
    }

    // The demotion loop, and the same loop with each cause's room added.
    const fit_state & S0     = base;
    kv_solve          solved = solve_kv(S0, slots);

    std::vector<char> ring_growth_device;
    if (!res.superseded.empty()) {
        fit_state S = S0;
        for (const kv_superseded_slot & sup : res.superseded) {
            // The old slot as FREE room, merged with whatever free block it touches:
            // freeing it would do exactly that.
            const zone_reservation & rv = g.tlsfs[sup.tlsf].reservations[sup.reservation];
            fit_piece                p;
            p.lo   = rv.offset;
            p.hi   = rv.offset + rv.size;
            p.kind = PK_FREE;
            add_free_piece(S.t[sup.tlsf], p);
        }
        ring_growth_device = solve_kv(S, slots).device;
    }
    std::vector<char> no_heads_device;
    if (solved.demoted != 0 && !r.head_slots.empty()) {
        fit_state S     = build_state(g, r);
        no_heads_device = solve_kv(S, slots).device;
    }

    // Placements.
    res.layers.resize(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        kv_layer_placement & lp = res.layers[i];
        lp.layer                = slots[i].layer;
        lp.device               = solved.device[i] != 0;
        lp.size                 = slots[i].total;
        if (!lp.device) {
            if (slots[i].forced_host) {
                lp.cause = KV_DEMOTE_FORCED_HOST;
            } else if (!ring_growth_device.empty() && ring_growth_device[i]) {
                lp.cause = KV_DEMOTE_RING_GROWTH;
            } else if (!no_heads_device.empty() && no_heads_device[i]) {
                lp.cause = KV_DEMOTE_HEAD_SLOT;
            } else {
                lp.cause = KV_DEMOTE_CAPACITY;
            }
        }
    }

    // The after-KV charge: device-resident layers only, ascending size, tiers
    // 1-3, and a miss is a decline.
    fit_state & F = solved.state;
    res.after_kv.resize(r.after_kv.size());
    std::vector<size_t> terms;
    for (size_t k = 0; k < r.after_kv.size(); ++k) {
        res.after_kv[k].term = k;
        res.after_kv[k].size = kv_layer_alloc_bytes(r.after_kv[k].size);
        bool on_device       = false;
        for (size_t i = 0; i < slots.size(); ++i) {
            on_device = on_device || (slots[i].layer == r.after_kv[k].layer && solved.device[i]);
        }
        if (on_device) {
            terms.push_back(k);
        }
    }
    std::stable_sort(terms.begin(), terms.end(),
                     [&](size_t a, size_t b) { return res.after_kv[a].size < res.after_kv[b].size; });
    // An after-KV charge only records a range and carves nothing, yet it goes through
    // find_room like a slot, so it too needs a carvable piece.  That is deliberate: it
    // keeps the charge inside room the later carves can really reach, and it must not
    // be relaxed into a placement the carve chain could not honour.
    for (size_t k : terms) {
        F.a.push_back({ AS_AFTER, k, res.after_kv[k].size, SIZE_MAX });
        const int id             = (int) F.a.size() - 1;
        res.after_kv[k].admitted = place(F, id, 0, false, no_remaining, 0);
    }

    // Layout: replay each TLSF's carves in placement order.
    auto emit_extent = [&](size_t t, size_t base_off, size_t size, size_t demand, kv_extent_kind kind, bool carve,
                           const std::vector<int> & kv_ids) {
        kv_region_extent e;
        e.tlsf   = t;
        e.offset = base_off;
        e.size   = size;
        e.kind   = kind;
        res.extents.push_back(e);
        const size_t ei = res.extents.size() - 1;
        size_t       at = 0;
        for (int id : kv_ids) {
            const size_t         si = F.a[id].id;
            kv_layer_placement & lp = res.layers[si];
            lp.extent               = ei;
            lp.slot_offset          = at;
            if (slots[si].sidecar) {
                lp.sidecar_offset = at + slots[si].kv_alloc;
            }
            at += slots[si].total;
        }
        kv_carve_op op;
        op.kind   = KV_CARVE_EXTENT;
        op.index  = ei;
        op.tlsf   = t;
        op.offset = base_off;
        op.size   = size;
        op.demand = demand;
        op.carve  = carve;
        res.carve_order.push_back(op);
    };
    auto emit_single = [&](const fit_assign & a, size_t t, size_t off, size_t size, bool carve) {
        kv_carve_op op;
        op.tlsf   = t;
        op.offset = off;
        op.size   = size;
        op.demand = a.size;
        op.carve  = carve;
        if (a.kind == AS_HEAD) {
            kv_head_placement & hp = res.heads[a.id];
            hp.tlsf                = t;
            hp.offset              = off;
            hp.size                = size;
            op.kind                = KV_CARVE_HEAD;
            op.index               = a.id;
        } else {
            kv_after_kv_placement & ap = res.after_kv[a.id];
            ap.tlsf                    = t;
            ap.offset                  = off;
            ap.size                    = size;
            op.kind                    = KV_CARVE_AFTER_KV;
            op.index                   = a.id;
        }
        res.carve_order.push_back(op);
    };
    // One chain of assignments in carve order.  A FREE piece's chain top-carves
    // downward from the piece's top edge; a retained run's packs upward from its
    // bottom and carves nothing.  Consecutive KV assigns are one extent.
    auto emit_chain = [&](size_t t, const std::vector<int> & assigns, size_t edge, bool downward,
                          kv_extent_kind kv_kind, bool carve) {
        size_t cur = edge;
        size_t k   = 0;
        while (k < assigns.size()) {
            const fit_assign & a = F.a[assigns[k]];
            if (a.kind == AS_KV) {
                size_t           k2    = k;
                size_t           total = 0;
                std::vector<int> ids;
                while (k2 < assigns.size() && F.a[assigns[k2]].kind == AS_KV) {
                    ids.push_back(assigns[k2]);
                    total += F.a[assigns[k2]].size;
                    ++k2;
                }
                const size_t b = downward ? cur - total : cur;
                emit_extent(t, b, total, total, kv_kind, carve, ids);
                cur = downward ? b : cur + total;
                k   = k2;
            } else {
                const size_t b = downward ? cur - a.size : cur;
                emit_single(a, t, b, a.size, carve && a.kind == AS_HEAD);
                cur = downward ? b : cur + a.size;
                ++k;
            }
        }
    };
    for (size_t t = 0; t < n_tlsf; ++t) {
        const fit_tlsf & ft = F.t[t];
        for (const fit_run & run : ft.runs) {
            for (size_t pi = 0; pi < run.pieces.size(); ++pi) {
                const fit_piece & p = run.pieces[pi];
                if (p.kind != PK_FREE || p.assigns.empty()) {
                    continue;
                }
                // Every offset and size is on the allocator's grain, so a piece's
                // carves tile its top exactly; a piece off the grain was never
                // offered a placement (piece_carvable).
                emit_chain(t, p.assigns, p.hi, /*downward=*/true,
                           run.frontier && pi + 1 == run.pieces.size() ? KV_EXTENT_FRONTIER : KV_EXTENT_HOLE,
                           /*carve=*/true);
            }
        }
        for (const fit_retained & rr : ft.retained) {
            emit_chain(t, rr.assigns, rr.lo, /*downward=*/false, KV_EXTENT_RETAINED, /*carve=*/false);
        }
        res.yield_prefix[t] = ft.yield_prefix;
        for (size_t off : ft.buried) {
            res.buried_released.push_back({ t, off });
        }
    }

    // Self extents are already carved: copy them in.
    for (const kv_self_extent & e : r.self_extents) {
        kv_region_extent re;
        re.tlsf   = e.tlsf;
        re.offset = e.offset;
        re.size   = e.size;
        re.kind   = KV_EXTENT_SELF;
        res.extents.push_back(re);
        for (const kv_self_slot & ss : e.slots) {
            for (size_t i = 0; i < slots.size(); ++i) {
                if (slots[i].layer == ss.layer) {
                    res.layers[i].device      = true;
                    res.layers[i].extent      = res.extents.size() - 1;
                    res.layers[i].slot_offset = ss.slot_offset;
                    res.layers[i].size        = ss.size;
                }
            }
        }
    }

    // free_after_full_kv: the room after the head slots minus every KV slot.
    {
        std::vector<long long> kv_charge(n_tlsf, 0);
        size_t                 last_kv = SIZE_MAX;
        for (size_t t = 0; t < n_tlsf; ++t) {
            if (S0.t[t].takes_kv) {
                last_kv = t;
            }
        }
        fit_state           T = S0;
        std::vector<size_t> remaining;
        for (const fit_slot & s : slots) {
            if (!s.self_placed) {
                remaining.push_back(s.total);
            }
        }
        size_t n = 0;
        for (size_t i = 0; i < slots.size(); ++i) {
            if (slots[i].self_placed) {
                continue;
            }
            T.a.push_back({ AS_KV, i, slots[i].total, SIZE_MAX });
            const int id = (int) T.a.size() - 1;
            if (place(T, id, 0, true, remaining, n)) {
                kv_charge[T.a[id].tlsf] += (long long) slots[i].total;
            } else if (last_kv != SIZE_MAX) {
                kv_charge[last_kv] += (long long) slots[i].total;
            }
            ++n;
        }
        for (size_t t = 0; t < n_tlsf; ++t) {
            res.free_after_full_kv[t] = (long long) room_total(S0, t) - kv_charge[t];
        }
    }

    // Weight-side holes too small for a slot the loop demoted.
    if (solved.demoted != 0) {
        // Smaller than the smallest slot the loop demoted: such a hole is room on
        // paper that no demoted layer could have used.
        size_t smallest = SIZE_MAX;
        for (size_t i = 0; i < slots.size(); ++i) {
            if (!solved.device[i] && !slots[i].forced_host) {
                smallest = std::min(smallest, slots[i].total);
            }
        }
        for (size_t t = 0; t < n_tlsf; ++t) {
            for (const fit_run & run : S0.t[t].runs) {
                size_t gi  = 0;
                size_t gap = gap_index(run, gi) ? gi : SIZE_MAX;
                for (size_t pi = 0; pi < run.pieces.size(); ++pi) {
                    const fit_piece & p = run.pieces[pi];
                    // The room a hole has left (a head slot may already fill it), as a
                    // slot would see it.
                    if (p.kind == PK_FREE && pi != gap && p.room() != 0 && p.room() < smallest &&
                        piece_carvable(run, pi)) {
                        res.sub_slot_holes.push_back({ t, p.lo });
                    }
                }
            }
        }
    }

    res.fits = solved.ok;
    return res;
}

}  // namespace ggml_sycl
