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

// tlsf_allocator::MIN_BLOCK_SIZE, which is private to the allocator: the gap
// smaller than size + this that a carve takes whole.  The fit == carve cases of
// test-kv-runtime-demotion run the real allocator and fail if the two differ.
constexpr size_t FIT_MIN_BLOCK = 256;

enum : uint8_t { PK_FREE, PK_OPT, PK_BARRIER };

enum : uint8_t { AS_HEAD, AS_KV, AS_AFTER };

struct fit_piece {
    size_t           lo   = 0;
    size_t           hi   = 0;
    uint8_t          kind = PK_FREE;
    size_t           used = 0;  // FREE: bytes reserved at the top by assignments
    std::vector<int> assigns;   // FREE: assignment ids in carve order

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
    uint32_t layer       = 0;
    uint8_t  group       = KV_SLOT_FULL;
    size_t   kv_alloc    = 0;  // kv_layer_alloc_bytes(kv_bytes)
    size_t   total       = 0;  // the slot's size, sidecar included
    bool     sidecar     = false;
    bool     forced_host = false;
    bool     self_placed = false;
};

bool cohort_equal(const char * a, const char * b) {
    return a == b || (a != nullptr && b != nullptr && std::strcmp(a, b) == 0);
}

// Split `p` around [a, b): the overlap becomes a BARRIER, the rest keeps its
// kind.  An OPT piece the range touches is a BARRIER whole: a tenant that is
// partly under a pending range cannot be released.
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
        if (r.commit_refit) {
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
        if (r.commit_refit) {
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
        for (const zone_range & rr : in.retained_runs) {
            if (rr.size != 0) {
                fit_retained ret;
                ret.lo = rr.offset;
                ret.hi = rr.offset + rr.size;
                out.retained.push_back(ret);
            }
        }
    }
    return S;
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

struct fit_room {
    size_t tlsf     = SIZE_MAX;
    bool   retained = false;
    size_t run      = 0;  // retained: the room index
    size_t idx      = 0;  // the piece index
};

// The cheapest tier 1-3 room for `size`: the first TLSF with room, best fit
// within it.  A slot is never split.
bool find_room(const fit_state & S, size_t size, int route, fit_room & out) {
    // Tier 1: retained runs.
    for (int tier = 1; tier <= 3; ++tier) {
        for (size_t t = 0; t < S.t.size(); ++t) {
            if (!eligible(S.t[t], route)) {
                continue;
            }
            size_t   best      = SIZE_MAX;
            size_t   best_free = SIZE_MAX;
            fit_room cand;
            cand.tlsf = t;
            if (tier == 1) {
                for (size_t i = 0; i < S.t[t].retained.size(); ++i) {
                    const fit_retained & r = S.t[t].retained[i];
                    const size_t         f = r.hi - r.lo - r.used;
                    if (f >= size && f < best_free) {
                        best      = i;
                        best_free = f;
                    }
                }
                if (best != SIZE_MAX) {
                    cand.retained = true;
                    cand.run      = best;
                    out           = cand;
                    return true;
                }
                continue;
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
                    if (p.kind != PK_FREE || p.room() < size) {
                        continue;
                    }
                    const bool is_gap = pi == gap;
                    if (is_gap != (tier == 2)) {
                        continue;
                    }
                    if (p.room() < best_free) {
                        best_free = p.room();
                        cand.run  = ri;
                        cand.idx  = pi;
                        best      = pi;
                    }
                }
            }
            if (best != SIZE_MAX) {
                out = cand;
                return true;
            }
        }
    }
    return false;
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
        for (const fit_piece & p : run.pieces) {
            if (p.kind == PK_FREE || p.kind == PK_OPT) {
                total += p.kind == PK_OPT ? p.size() : p.room();
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
        for (const fit_piece & p : run.pieces) {
            if (p.kind == PK_FREE) {
                total += p.room();
            }
        }
    }
    return total;
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
    std::vector<size_t> order;  // slot indices in demotion order
    for (int group = KV_SLOT_FULL; group <= KV_SLOT_SWA; ++group) {
        for (size_t i = slots.size(); i-- > 0;) {
            if (slots[i].group == group && !slots[i].forced_host && !slots[i].self_placed) {
                order.push_back(i);
            }
        }
    }
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

}  // namespace

kv_region_fit_result kv_region_fit(const shared_zone_geometry & g, const kv_region_request & r) {
    kv_region_fit_result res;
    const size_t         n_tlsf = g.tlsfs.size();
    res.yield_prefix.assign(n_tlsf, 0);
    res.free_after_full_kv.assign(n_tlsf, 0);
    res.tlsf_free.assign(n_tlsf, 0);

    // The slot table: full attention first, then SWA, each in layer order.
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
    for (size_t h = 0; h < r.head_slots.size(); ++h) {
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
            const zone_reservation & rv = g.tlsfs[sup.tlsf].reservations[sup.reservation];
            fit_run                  run;
            fit_piece                p;
            p.lo   = rv.offset;
            p.hi   = rv.offset + rv.size;
            p.kind = PK_FREE;
            run.pieces.push_back(p);
            S.t[sup.tlsf].runs.push_back(std::move(run));
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
    for (size_t k : terms) {
        F.a.push_back({ AS_AFTER, k, res.after_kv[k].size, SIZE_MAX });
        const int id             = (int) F.a.size() - 1;
        res.after_kv[k].admitted = place(F, id, 0, false, no_remaining, 0);
    }

    // Layout: replay each TLSF's carves in placement order.
    auto emit_extent = [&](size_t t, size_t base_off, size_t size, size_t demand, uint8_t kind, bool carve,
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
    for (size_t t = 0; t < n_tlsf; ++t) {
        const fit_tlsf & ft = F.t[t];
        for (const fit_run & run : ft.runs) {
            for (size_t pi = 0; pi < run.pieces.size(); ++pi) {
                const fit_piece & p = run.pieces[pi];
                if (p.kind != PK_FREE || p.assigns.empty()) {
                    continue;
                }
                // What is left under the carved bytes is part of the last carve
                // when it is smaller than MIN_BLOCK_SIZE (carve_gap's whole take).
                const size_t left   = p.room();
                const size_t absorb = left < FIT_MIN_BLOCK ? left : 0;
                size_t       cur    = p.hi;
                size_t       k      = 0;
                while (k < p.assigns.size()) {
                    const fit_assign & a = F.a[p.assigns[k]];
                    if (a.kind == AS_KV) {
                        size_t           k2    = k;
                        size_t           total = 0;
                        std::vector<int> ids;
                        while (k2 < p.assigns.size() && F.a[p.assigns[k2]].kind == AS_KV) {
                            ids.push_back(p.assigns[k2]);
                            total += F.a[p.assigns[k2]].size;
                            ++k2;
                        }
                        const size_t extra = k2 == p.assigns.size() ? absorb : 0;
                        const size_t b     = cur - total - extra;
                        emit_extent(t, b, total + extra, total,
                                    run.frontier && pi + 1 == run.pieces.size() ? KV_EXTENT_FRONTIER : KV_EXTENT_HOLE,
                                    true, ids);
                        cur = b;
                        k   = k2;
                    } else {
                        const size_t extra = k + 1 == p.assigns.size() ? absorb : 0;
                        const size_t b     = cur - a.size - extra;
                        emit_single(a, t, b, a.size + extra, a.kind == AS_HEAD);
                        cur = b;
                        ++k;
                    }
                }
            }
        }
        for (const fit_retained & rr : ft.retained) {
            size_t cur = rr.lo;
            size_t k   = 0;
            while (k < rr.assigns.size()) {
                const fit_assign & a = F.a[rr.assigns[k]];
                if (a.kind == AS_KV) {
                    size_t           k2    = k;
                    size_t           total = 0;
                    std::vector<int> ids;
                    while (k2 < rr.assigns.size() && F.a[rr.assigns[k2]].kind == AS_KV) {
                        ids.push_back(rr.assigns[k2]);
                        total += F.a[rr.assigns[k2]].size;
                        ++k2;
                    }
                    emit_extent(t, cur, total, total, KV_EXTENT_RETAINED, false, ids);
                    cur += total;
                    k = k2;
                } else {
                    emit_single(a, t, cur, a.size, false);
                    cur += a.size;
                    ++k;
                }
            }
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
                    if (p.kind == PK_FREE && pi != gap && p.size() != 0 && p.size() < smallest) {
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
