// llama.cpp-5tdy: the placement plan's per-(layer, role) residency summary must
// give the same answers as the per-expert sweep it replaced.
//
// has_host_experts(), all_experts_on_device(), count_experts_on_device(),
// count_planned_experts() and expert_on_device() are checked against an oracle
// that re-implements the original per-expert semantics straight from
// plan.entries: the semantic (layer, expert, role) index keeps the FIRST
// classified entry, the name-keyed "name:eN" fallback keeps the LAST entry, and
// a by-name query reads the semantic entry when there is one and the
// name-keyed entry otherwise. The oracle shares no code with the summary.
//
// The plan is checked after build_index(), after update_expert_placement()
// flips entries both ways (host <-> device, device <-> device, an entry that
// is counted in BOTH a semantic group and a fallback group), after a copy, and
// after a direct entry mutation followed by a build_index() rebuild.
//
// Host-only: synthetic plans, no queue, no device, no model.

#include "ggml-sycl.h"
#include "ggml-sycl/unified-cache.hpp"

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace {

using ggml_sycl::expert_residency_counts;
using ggml_sycl::expert_tensor_role;
using ggml_sycl::placement_entry;
using ggml_sycl::placement_plan;
using ggml_sycl::placement_priority;

int failures = 0;
int checks   = 0;

constexpr size_t kBytes   = 4096;
constexpr int    kExperts = 8;

// The original semantics, rebuilt from the entries on every query.
struct oracle {
    const placement_plan &                      plan;
    std::map<std::tuple<int, int, int>, size_t> semantic;  // first classified entry wins
    std::map<std::string, size_t>               by_name;   // last entry wins

    explicit oracle(const placement_plan & p) : plan(p) {
        for (size_t i = 0; i < plan.entries.size(); ++i) {
            const placement_entry & e = plan.entries[i];
            if (e.expert_id < 0) {
                continue;
            }
            by_name[e.name + ":e" + std::to_string(e.expert_id)] = i;
            if (e.expert_role == expert_tensor_role::UNKNOWN || e.layer_id < 0) {
                continue;
            }
            semantic.emplace(std::make_tuple(e.layer_id, e.expert_id, static_cast<int>(e.expert_role)), i);
        }
    }

    const placement_entry * semantic_entry(const std::string & name, int expert) const {
        const int  layer = ggml_sycl::expert_layer_from_tensor_name(name.c_str());
        const int  role  = static_cast<int>(ggml_sycl::expert_tensor_role_from_tensor_name(name.c_str()));
        const auto it    = semantic.find(std::make_tuple(layer, expert, role));
        return it == semantic.end() ? nullptr : &plan.entries[it->second];
    }

    const placement_entry * name_entry(const std::string & name, int expert) const {
        const auto it = by_name.find(name + ":e" + std::to_string(expert));
        return it == by_name.end() ? nullptr : &plan.entries[it->second];
    }

    bool has_host(const std::string & name, int64_t n) const {
        for (int64_t e = 0; e < n; ++e) {
            if (const placement_entry * p = semantic_entry(name, static_cast<int>(e))) {
                if (!p->on_device) {
                    return true;
                }
                continue;
            }
            const placement_entry * q = name_entry(name, static_cast<int>(e));
            if (q && !q->on_device) {
                return true;
            }
        }
        return false;
    }

    bool all_local(const std::string & name, int64_t n, int device) const {
        for (int64_t e = 0; e < n; ++e) {
            const placement_entry * p = semantic_entry(name, static_cast<int>(e));
            if (!p || !p->on_device || p->target_device != device) {
                return false;
            }
        }
        return true;
    }

    bool on_device(const std::string & name, int expert, int device) const {
        const placement_entry * p = semantic_entry(name, expert);
        if (!p) {
            p = name_entry(name, expert);
        }
        return p && p->on_device && (device < 0 || p->target_device == device);
    }

    int64_t count_on_device(const std::string & name, int64_t n, int device) const {
        int64_t count = 0;
        for (int64_t e = 0; e < n; ++e) {
            count += on_device(name, static_cast<int>(e), device) ? 1 : 0;
        }
        return count;
    }

    expert_residency_counts planned(const std::string & name, int64_t n, int device) const {
        expert_residency_counts c;
        for (int64_t e = 0; e < n; ++e) {
            const placement_entry * p = semantic_entry(name, static_cast<int>(e));
            if (!p) {
                continue;
            }
            c.found++;
            if (!p->on_device) {
                c.host++;
                continue;
            }
            c.on_device++;
            c.on_target += p->target_device == device ? 1 : 0;
        }
        return c;
    }
};

void check(bool ok, const std::string & stage, const std::string & what) {
    ++checks;
    if (!ok) {
        std::printf("  FAIL [%s] %s\n", stage.c_str(), what.c_str());
        ++failures;
    }
}

// The tensor names every stage queries: the split trio of three layers, a
// tensor whose entries are unclassified (fallback-only), a name that carries no
// layer, a name whose entries are indexed under ANOTHER layer, and a name the
// plan does not have at all.
const std::vector<std::string> & query_names() {
    static const std::vector<std::string> names = {
        "blk.0.ffn_gate_exps.weight",   "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight",
        "blk.1.ffn_gate_exps.weight",   "blk.1.ffn_up_exps.weight", "blk.1.ffn_down_exps.weight",
        "blk.2.ffn_gate_exps.weight",   "blk.2.ffn_up_exps.weight", "blk.2.ffn_down_exps.weight",
        "blk.5.ffn_custom_exps.weight", "ffn_gate_exps.weight",     "blk.7.ffn_up_exps.weight",
        "blk.9.ffn_down_exps.weight",
    };
    return names;
}

void compare(const placement_plan & plan, const std::string & stage) {
    const oracle  o(plan);
    const int64_t ns[]      = { 0, 1, 3, kExperts - 1, kExperts, kExperts + 1 };
    const int     devices[] = { -1, 0, 1, 2 };
    for (const std::string & name : query_names()) {
        const char * cname = name.c_str();
        for (int64_t n : ns) {
            const std::string at = name + " n=" + std::to_string(n);
            check(plan.has_host_experts(name, n) == o.has_host(name, n), stage, "has_host_experts " + at);
            check(plan.has_host_experts(cname, n, 0) == o.has_host(name, n), stage, "has_host_experts(char*) " + at);
            for (int dev : devices) {
                const std::string atd = at + " dev=" + std::to_string(dev);
                check(plan.all_experts_on_device(cname, n, dev) == (n <= 0 || o.all_local(name, n, dev)), stage,
                      "all_experts_on_device " + atd);
                check(plan.count_experts_on_device(cname, n, dev) == o.count_on_device(name, n, dev), stage,
                      "count_experts_on_device " + atd);
                const expert_residency_counts got  = plan.count_planned_experts(cname, n, dev);
                const expert_residency_counts want = o.planned(name, n, dev);
                check(got.found == want.found && got.host == want.host && got.on_device == want.on_device &&
                          got.on_target == want.on_target,
                      stage, "count_planned_experts " + atd);
            }
        }
        for (int e = 0; e < kExperts + 1; ++e) {
            for (int dev : devices) {
                check(plan.expert_on_device(name, e, dev) == o.on_device(name, e, dev), stage,
                      "expert_on_device " + name + " e=" + std::to_string(e) + " dev=" + std::to_string(dev));
            }
            const placement_entry * view = plan.find_expert_entry(cname, e);
            check(view == o.semantic_entry(name, e), stage, "find_expert_entry " + name + " e=" + std::to_string(e));
        }
    }
}

void add_expert(placement_plan &    plan,
                const std::string & name,
                int                 layer,
                int                 expert,
                expert_tensor_role  role,
                bool                on_device,
                int                 target) {
    plan.entries.push_back(
        { name, kBytes, kBytes, 0, placement_priority::MOE_UP, layer, expert, role, on_device, target });
}

// Residency patterns per (layer, role) so the groups differ: all on device 0,
// split across two devices, part on host, one entry on_device with target -1.
placement_plan synthetic_plan() {
    placement_plan plan{};
    plan.multi_device                   = true;
    const char *             roles[]    = { "gate", "up", "down" };
    const expert_tensor_role role_ids[] = { expert_tensor_role::GATE, expert_tensor_role::UP,
                                            expert_tensor_role::DOWN };
    for (int layer = 0; layer < 3; ++layer) {
        for (int r = 0; r < 3; ++r) {
            const std::string name = "blk." + std::to_string(layer) + ".ffn_" + roles[r] + "_exps.weight";
            for (int e = 0; e < kExperts; ++e) {
                bool on     = true;
                int  target = 0;
                if (layer == 1) {
                    target = e % 2;  // split across devices 0 and 1
                } else if (layer == 2) {
                    on     = e < 5;  // experts 5..7 on host
                    target = on ? 0 : -1;
                }
                if (layer == 2 && r == 2 && e == 1) {
                    target = -1;  // on_device with no target
                }
                add_expert(plan, name, layer, e, role_ids[r], on, target);
            }
        }
    }
    // Unclassified role: indexed by name only. Experts 0..3 on device 1, 4..7 on host.
    for (int e = 0; e < kExperts; ++e) {
        add_expert(plan, "blk.5.ffn_custom_exps.weight", 5, e, expert_tensor_role::UNKNOWN, e < 4, e < 4 ? 1 : -1);
    }
    // No layer in the name or the entry: indexed by name only.
    for (int e = 0; e < 4; ++e) {
        add_expert(plan, "ffn_gate_exps.weight", -1, e, expert_tensor_role::GATE, e != 2, e != 2 ? 0 : -1);
    }
    // The name says layer 7, the entry says layer 8: semantic key (8, e, UP),
    // and a by-name query for blk.7 misses it and falls back to the name key.
    // Such an entry is counted in both a semantic group and a fallback group.
    for (int e = 0; e < kExperts; ++e) {
        add_expert(plan, "blk.7.ffn_up_exps.weight", 8, e, expert_tensor_role::UP, true, 1);
    }
    // A duplicate of (0, 3, GATE) on host: the semantic index keeps the first.
    add_expert(plan, "blk.0.ffn_gate_exps.weight", 0, 3, expert_tensor_role::GATE, false, -1);
    // A dense weight, which no expert query may see.
    plan.entries.push_back({ "blk.0.attn_q.weight", kBytes, kBytes, 0, placement_priority::ATTENTION, 0, -1,
                             expert_tensor_role::UNKNOWN, true, 0 });
    return plan;
}

void flip(placement_plan &   plan,
          int                layer,
          int                expert,
          expert_tensor_role role,
          bool               on_device,
          int                target,
          const char *       what) {
    if (!plan.update_expert_placement(layer, expert, role, on_device, target)) {
        std::printf("  FAIL [flip] update_expert_placement refused: %s\n", what);
        ++failures;
    }
    compare(plan, what);
}

}  // namespace

int main() {
    placement_plan plan = synthetic_plan();
    plan.build_index();

    // Preconditions: the cases the comparisons rely on actually occur.
    {
        const oracle o(plan);
        check(o.all_local("blk.0.ffn_up_exps.weight", kExperts, 0), "pre", "layer 0 up is all on device 0");
        check(!o.all_local("blk.1.ffn_up_exps.weight", kExperts, 0), "pre", "layer 1 up is split");
        check(o.has_host("blk.2.ffn_up_exps.weight", kExperts), "pre", "layer 2 up has host experts");
        check(!o.has_host("blk.2.ffn_up_exps.weight", 3), "pre", "layer 2 up prefix [0,3) has none");
        check(o.has_host("blk.5.ffn_custom_exps.weight", kExperts), "pre", "fallback tensor has host experts");
        check(!o.has_host("blk.5.ffn_custom_exps.weight", 4), "pre", "fallback prefix [0,4) has none");
        check(o.on_device("blk.7.ffn_up_exps.weight", 0, 1), "pre", "blk.7 resolves through the name key");
        check(!o.has_host("blk.0.ffn_gate_exps.weight", kExperts), "pre", "the host duplicate is shadowed");
    }

    compare(plan, "build_index");

    flip(plan, 0, 4, expert_tensor_role::UP, false, -1, "device0 -> host");
    flip(plan, 0, 4, expert_tensor_role::UP, true, 0, "host -> device0");
    flip(plan, 1, 1, expert_tensor_role::DOWN, true, 0, "device1 -> device0");
    flip(plan, 1, 3, expert_tensor_role::DOWN, true, 0, "device1 -> device0 again");
    flip(plan, 2, 5, expert_tensor_role::GATE, true, 1, "host -> device1");
    flip(plan, 2, 2, expert_tensor_role::DOWN, true, 0, "unchanged update");
    flip(plan, 2, 1, expert_tensor_role::DOWN, true, 0, "target -1 -> device0");
    for (int e = 0; e < kExperts; ++e) {
        plan.update_expert_placement(1, e, expert_tensor_role::GATE, true, 1);
    }
    compare(plan, "whole layer 1 gate -> device1");
    flip(plan, 8, 2, expert_tensor_role::UP, false, -1, "entry in both groups -> host");
    flip(plan, 8, 2, expert_tensor_role::UP, true, 0, "entry in both groups -> device0");
    if (plan.update_expert_placement(5, 0, expert_tensor_role::UNKNOWN, false, -1)) {
        std::printf("  FAIL [flip] update_expert_placement accepted an unclassified key\n");
        ++failures;
    }

    const placement_plan copy = plan;
    compare(copy, "copy");

    // A direct mutation is invisible to the summary until build_index() runs.
    for (placement_entry & e : plan.entries) {
        if (e.name == "blk.0.ffn_down_exps.weight" && e.expert_id == 6) {
            e.on_device     = false;
            e.target_device = -1;
        }
        if (e.name == "blk.5.ffn_custom_exps.weight" && e.expert_id == 7) {
            e.on_device     = true;
            e.target_device = 1;
        }
    }
    plan.build_index();
    compare(plan, "rebuild after direct mutation");

    if (failures != 0) {
        std::printf("%d of %d check(s) failed\n", failures, checks);
        return 1;
    }
    std::printf("all %d residency-summary checks pass\n", checks);
    return 0;
}
