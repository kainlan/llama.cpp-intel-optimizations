// The host WEIGHT zone is sized from exact bytes plus named terms, never from a ratio.
//
// The planner sizes host_zone_weight_bytes in populate_host_zone_sizing(), which both planner builders call. The
// size is a fact the plan already knows: every host-resident weight at the bytes the cache materializes for it, plus
// one named term for each real extra. Slack nobody can name is a second source for that fact, and for a DeepSeek-V4.1
// Q2_K load a 20 % ratio over ~149 GiB of host experts is ~30 GiB the host budget cannot hold.
//
// What the cache materializes for a host-resident weight (S1-PRELOAD, ggml-sycl.cpp): a host-planned expert is one
// pinned WEIGHT allocation of tensor->nb[2] bytes, a host-planned dense weight one of ggml_nbytes(tensor). Both are
// the entry's AOS src_size: the inventory's size is ggml_nbytes, and an expert's is that over ne[2]. The terms:
//
//   host_weights     the sum of those bytes over the host-resident entries, one pinned block per entry
//   runtime_reserve  64 MiB for the runtime HOST_COMPUTE requests that also land in WEIGHT (the context's logits and
//                    embeddings host buffer, the server's prompt-checkpoint reads)
//   layout_copies    a second, converted copy of a host weight the plan keeps beside the first: none today
//   tlsf_bound       the TLSF rounding of each block to its grain, so the block count times the grain
//
// Arm 1 (no model, no device: the real planner over a synthetic Q2_K-shaped inventory, under a mocked device table)
// asserts the WEIGHT size is exactly the sum of those terms and is not the 1.2x ratio, and that the plan prints
// each term by name on a [HOST-PLAN] line.
//
// Arm 2 (the census) constructs a cache, configures the host zones from that plan, and makes every planned WEIGHT-zone
// allocation through the request shapes the materialization paths use. Each one that lands in host_zone_id::WEIGHT is
// attributed to a named term by its role and category, each term's bytes stay within the term, and the zone never
// grows. It needs a device, so a host without one exits 77 after arm 1 has passed.

#include "../common.hpp"
#include "../ggml-sycl-test.hpp"
#include "../unified-cache.hpp"
#include "sycl-test-skip.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

using namespace ggml_sycl;

namespace {

constexpr size_t MiB = 1024ull * 1024ull;

// The terms the test derives on its own, from the plan's entries, independent of the code under test.
constexpr size_t k_runtime_reserve_bytes = 64ull * MiB;
constexpr size_t k_tlsf_grain            = 256;

int g_checks = 0;
int g_fail   = 0;

void check(bool cond, const char * what) {
    g_checks++;
    if (!cond) {
        g_fail++;
        std::printf("  FAIL: %s\n", what);
    }
}

void check_eq(const char * what, size_t got, size_t want) {
    g_checks++;
    if (got != want) {
        g_fail++;
        std::printf("  FAIL: %s: got %zu, want %zu\n", what, got, want);
    }
}

// --- the mocked device table: discrete cards, enough to plan, nothing touched ---

ggml_sycl_device_info make_mock_sycl_info(int device_count) {
    ggml_sycl_device_info info{};
    info.device_count        = device_count;
    info.total_gpu_count     = device_count;
    info.host_max_alloc_size = 128ull * 1024ull * MiB;

    for (int d = 0; d < device_count; ++d) {
        const size_t       total_vram   = 16ull * 1024ull * MiB;
        sycl_device_info & dev          = info.devices[d];
        dev.cc                          = 1200;
        dev.nsm                         = 128;
        dev.smpbo                       = 64ull * 1024ull;
        dev.vmm                         = true;
        dev.total_vram                  = total_vram;
        dev.free_vram_at_init           = total_vram;
        dev.max_alloc_size              = total_vram - 512ull * MiB;
        dev.safe_max_alloc_size         = dev.max_alloc_size;
        dev.supports_soa_reorder        = true;
        dev.xmx_caps.compute_units      = dev.nsm;
        dev.xmx_caps.global_mem_size    = total_vram;
        dev.xmx_caps.max_mem_alloc_size = dev.max_alloc_size;
        std::snprintf(dev.device_name, sizeof(dev.device_name), "mock-weight-zone-terms-%d", d);
        info.max_work_group_sizes[d] = 256;
        info.default_tensor_split[d] = static_cast<float>(d + 1) / static_cast<float>(device_count);
        info.gpu_dpct_ids[d]         = d;
    }
    for (int src = 0; src < device_count; ++src) {
        for (int dst = 0; dst < device_count; ++dst) {
            sycl_peer_link_info & link = info.peer_links[src][dst];
            link.valid                 = true;
            link.src_device            = src;
            link.dst_device            = dst;
            link.same_device           = src == dst;
            link.same_backend          = true;
            link.same_platform         = true;
            link.same_sycl_context     = src == dst;
        }
    }
    return info;
}

// --- a Q2_K-shaped inventory: DeepSeek's quant type, at a size a test can hold ---

constexpr int     k_layers  = 2;
constexpr int     k_experts = 8;
constexpr int64_t k_embd    = 1024;
constexpr int64_t k_ffn     = 512;
constexpr int64_t k_vocab   = 4096;

placement_tensor_info tensor(const std::string & name, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2 = 1) {
    placement_tensor_info t(name, ggml_row_size(type, ne0) * static_cast<size_t>(ne1) * static_cast<size_t>(ne2));
    t.type  = type;
    t.ne[0] = ne0;
    t.ne[1] = ne1;
    t.ne[2] = ne2;
    t.ne[3] = 1;
    return t;
}

std::vector<placement_tensor_info> q2k_inventory() {
    std::vector<placement_tensor_info> inv;
    inv.push_back(tensor("token_embd.weight", GGML_TYPE_Q2_K, k_embd, k_vocab));
    for (int l = 0; l < k_layers; ++l) {
        const std::string blk = "blk." + std::to_string(l) + ".";
        inv.push_back(tensor(blk + "attn_norm.weight", GGML_TYPE_F32, k_embd, 1));
        inv.push_back(tensor(blk + "attn_q.weight", GGML_TYPE_Q2_K, k_embd, k_embd));
        inv.push_back(tensor(blk + "attn_k.weight", GGML_TYPE_Q2_K, k_embd, k_embd));
        inv.push_back(tensor(blk + "attn_v.weight", GGML_TYPE_Q2_K, k_embd, k_embd));
        inv.push_back(tensor(blk + "attn_output.weight", GGML_TYPE_Q2_K, k_embd, k_embd));
        inv.push_back(tensor(blk + "ffn_norm.weight", GGML_TYPE_F32, k_embd, 1));
        inv.push_back(tensor(blk + "ffn_gate_inp.weight", GGML_TYPE_F32, k_embd, k_experts));
        inv.push_back(tensor(blk + "ffn_gate_exps.weight", GGML_TYPE_Q2_K, k_embd, k_ffn, k_experts));
        inv.push_back(tensor(blk + "ffn_up_exps.weight", GGML_TYPE_Q2_K, k_embd, k_ffn, k_experts));
        inv.push_back(tensor(blk + "ffn_down_exps.weight", GGML_TYPE_Q2_K, k_ffn, k_embd, k_experts));
    }
    inv.push_back(tensor("output_norm.weight", GGML_TYPE_F32, k_embd, 1));
    inv.push_back(tensor("output.weight", GGML_TYPE_Q2_K, k_embd, k_vocab));
    return inv;
}

// --- the [HOST-PLAN] line, captured from the log callback ---

std::string g_host_plan_line;

void capture_log(ggml_log_level level, const char * text, void * user) {
    (void) level;
    (void) user;
    if (text && std::strstr(text, "[HOST-PLAN]") && std::strstr(text, "WEIGHT zone")) {
        g_host_plan_line = text;
    }
}

// The value printed after "<name>=" on the line, or SIZE_MAX when the name is absent.
size_t printed_term(const std::string & line, const char * name) {
    const std::string key = std::string(" ") + name + "=";
    const size_t      at  = line.find(key);
    if (at == std::string::npos) {
        return SIZE_MAX;
    }
    return static_cast<size_t>(std::strtoull(line.c_str() + at + key.size(), nullptr, 10));
}

struct derived_terms {
    size_t host_weights = 0;
    size_t blocks       = 0;

    size_t total() const { return host_weights + k_runtime_reserve_bytes + blocks * k_tlsf_grain; }
};

derived_terms derive_terms(const placement_plan & plan) {
    derived_terms t;
    for (const placement_entry & e : plan.entries) {
        if (e.on_device) {
            continue;
        }
        t.host_weights += e.src_size;
        t.blocks++;
    }
    return t;
}

placement_plan plan_q2k(const std::vector<placement_tensor_info> & inventory) {
    const ggml_sycl_device_info   info = make_mock_sycl_info(1);
    test_sycl_info_override_guard info_guard(info);
    placement_kv_info             kv{};
    // A budget far below the experts: every expert, and with them the dense weights, is host-resident.
    return compute_placement_plan(inventory, /*vram_budget=*/1 * MiB, /*device_id=*/0, kv, nullptr, k_experts);
}

// The other planner builder, over two cards with the same starved budget.
placement_plan plan_q2k_two_devices(const std::vector<placement_tensor_info> & inventory) {
    const ggml_sycl_device_info      info = make_mock_sycl_info(2);
    test_sycl_info_override_guard    info_guard(info);
    placement_kv_info                kv{};
    const std::vector<device_budget> budgets = {
        { 0, 1 * MiB, 16ull * 1024ull * MiB, 1.0, true },
        { 1, 1 * MiB, 16ull * 1024ull * MiB, 1.0, true },
    };
    return compute_multi_device_plan(budgets, inventory, k_layers, multi_gpu_mode::HYBRID, kv, nullptr, k_experts);
}

// The checks one plan's WEIGHT zone must pass, whichever builder made it. `line` is that plan's [HOST-PLAN] line.
//
// `counts_host_at_src_size` says whether the builder's weight_host_bytes counts a host expert at its AOS src_size,
// the bytes the cache materializes. The single-device builder does, and resets a host expert to AOS. The
// multi-device builder counts a host expert at the device layout's dst_size and leaves that layout on the entry, so
// its weight_host_bytes is not the materialized fact; the WEIGHT zone is sized from the entries in both, and the
// comparison is printed rather than asserted there.
void check_weight_zone(const char *           builder,
                       const placement_plan & plan,
                       const std::string &    line,
                       bool                   counts_host_at_src_size) {
    std::printf("  [%s]\n", builder);
    const derived_terms t                = derive_terms(plan);
    const size_t        n_expert_entries = static_cast<size_t>(k_layers) * 3 * k_experts;
    check(t.blocks >= n_expert_entries, "every routed expert is host-resident under the 1 MiB budget");
    check(t.host_weights > 0, "the plan placed weights on the host");
    if (counts_host_at_src_size) {
        // The plan's own host-weight accounting is the same fact: the term sums what the cache materializes.
        check_eq("host_weights == the plan's weight_host_bytes", t.host_weights, plan.weight_host_bytes);
    } else {
        std::printf("  note: host_weights=%zu, the builder's weight_host_bytes=%zu\n", t.host_weights,
                    plan.weight_host_bytes);
    }

    const size_t ratio_size = static_cast<size_t>(static_cast<double>(t.host_weights) * 1.2) + k_runtime_reserve_bytes;
    // Vacuity guard: the two answers must differ on this inventory, or "not 1.2x" would prove nothing.
    check(ratio_size != t.total(), "the fixture separates exact-plus-terms from the 1.2x ratio");
    check_eq("host_zone_weight_bytes == host_weights + runtime_reserve + layout_copies + tlsf_bound",
             plan.host_zone_weight_bytes, t.total());
    check(plan.host_zone_weight_bytes != ratio_size, "host_zone_weight_bytes is not the 1.2x ratio");

    // The builder sized the zone through the one terms function.
    const host_weight_zone_terms terms = placement_host_weight_zone_terms(plan);
    check_eq("placement_host_weight_zone_terms().total() is the planned size", terms.total(),
             plan.host_zone_weight_bytes);
    check_eq("terms.host_weights", terms.host_weights, t.host_weights);
    check_eq("terms.blocks", terms.blocks, t.blocks);

    check(!line.empty(), "a [HOST-PLAN] WEIGHT zone line was logged");
    if (!line.empty()) {
        std::printf("  logged: %s", line.c_str());
        check_eq("[HOST-PLAN] host_weights", printed_term(line, "host_weights"), t.host_weights);
        check_eq("[HOST-PLAN] blocks", printed_term(line, "blocks"), t.blocks);
        check_eq("[HOST-PLAN] runtime_reserve", printed_term(line, "runtime_reserve"), k_runtime_reserve_bytes);
        check_eq("[HOST-PLAN] layout_copies", printed_term(line, "layout_copies"), 0);
        check_eq("[HOST-PLAN] tlsf_bound", printed_term(line, "tlsf_bound"), t.blocks * k_tlsf_grain);
        check_eq("[HOST-PLAN] total", printed_term(line, "total"), t.total());
    }
}

// Arm 1: the WEIGHT zone is the exact bytes plus the named terms, and not 1.2x, from both planner builders.
void arm1_weight_zone_is_exact_bytes_plus_named_terms(placement_plan & out_plan) {
    std::printf("(1) a Q2_K-shaped inventory's WEIGHT zone is exact bytes plus named terms, not 1.2x\n");
    const std::vector<placement_tensor_info> inventory = q2k_inventory();

    ggml_log_callback prev_cb   = nullptr;
    void *            prev_data = nullptr;
    ggml_log_get(&prev_cb, &prev_data);
    ggml_log_set(capture_log, nullptr);
    g_host_plan_line.clear();
    const placement_plan single_line_plan = plan_q2k(inventory);
    const std::string    single_line      = g_host_plan_line;
    g_host_plan_line.clear();
    const placement_plan multi_plan = plan_q2k_two_devices(inventory);
    const std::string    multi_line = g_host_plan_line;
    ggml_log_set(prev_cb, prev_data);

    check_weight_zone("compute_placement_plan", single_line_plan, single_line, true);
    check_weight_zone("compute_multi_device_plan", multi_plan, multi_line, false);
    out_plan = single_line_plan;
}

// --- arm 2: the census ---

enum class weight_term { HOST_WEIGHTS, RUNTIME_RESERVE, UNATTRIBUTED };

// Which named term an allocation that landed in WEIGHT belongs to, from its owner's role and category alone.
weight_term attribute(const alloc_metadata & md) {
    if (md.role == alloc_role::WEIGHT && md.category != runtime_category::HOST_COMPUTE) {
        return weight_term::HOST_WEIGHTS;
    }
    if (md.category == runtime_category::HOST_COMPUTE) {
        return weight_term::RUNTIME_RESERVE;
    }
    return weight_term::UNATTRIBUTED;
}

// The request S1-PRELOAD makes for a host-planned weight (ggml_sycl_host_expert_alloc for an expert, the dense copy's
// unified_alloc for a dense weight): a pinned WEIGHT allocation of the entry's bytes.
alloc_request host_weight_request(sycl::queue & q, size_t bytes) {
    alloc_request req{};
    req.queue                               = &q;
    req.device                              = 0;
    req.size                                = bytes;
    req.intent.role                         = alloc_role::WEIGHT;
    req.intent.constraints.must_host_pinned = true;
    req.intent.constraints.use_pinned_pool  = true;
    return req;
}

// The request a SYCL_Host buffer makes after the model load (the context's output buffer): role STAGING, category
// HOST_COMPUTE, which routes to WEIGHT.
alloc_request host_compute_request(sycl::queue & q, size_t bytes) {
    alloc_request req{};
    req.queue                               = &q;
    req.device                              = 0;
    req.size                                = bytes;
    req.intent.role                         = alloc_role::STAGING;
    req.intent.category                     = runtime_category::HOST_COMPUTE;
    req.intent.cohort_id                    = "backend_host_buffer";
    req.intent.constraints.must_host_pinned = true;
    req.intent.constraints.use_pinned_pool  = true;
    return req;
}

bool arm2_every_weight_zone_allocation_has_a_term(const placement_plan & plan) {
    std::printf("(2) census: every WEIGHT-zone allocation is attributed to a named term, and the zone never grows\n");
    std::optional<sycl::device> dev_opt =
        sycl_test_require_gpu("the WEIGHT-zone census (arm 1, the sizing, DID run and its result stands)");
    if (!dev_opt) {
        return false;
    }
    sycl::queue     q(*dev_opt, sycl::property::queue::in_order{});
    unified_cache * cache = get_unified_cache(q);
    check(cache != nullptr, "the unified cache is available");
    if (!cache) {
        return true;
    }
    if (!cache->host_zones_configured()) {
        cache->configure_host_zones(plan.host_zone_weight_bytes, 2 * MiB, 2 * MiB, 2 * MiB);
    }
    check(cache->host_zones_configured(), "host zones configured from the plan");
    if (!cache->host_zones_configured()) {
        return true;
    }

    const size_t capacity_before = cache->host_zone_capacity(host_zone_id::WEIGHT);
    const size_t used_before     = cache->host_zone_used(host_zone_id::WEIGHT);
    check(capacity_before >= plan.host_zone_weight_bytes, "the WEIGHT zone holds at least the plan's size");

    const derived_terms     t = derive_terms(plan);
    std::vector<mem_handle> held;
    size_t                  attributed_weights = 0;
    size_t                  attributed_runtime = 0;
    size_t                  unattributed       = 0;
    size_t                  not_in_weight      = 0;

    auto take = [&](const alloc_request & req) -> size_t {
        const size_t before = cache->host_zone_used(host_zone_id::WEIGHT);
        mem_handle   h      = unified_allocate(req);
        if (!h.valid()) {
            return SIZE_MAX;
        }
        alloc_metadata md{};
        if (!h.owner_metadata(&md) || md.host_zone != host_zone_id::WEIGHT) {
            not_in_weight++;
        } else {
            const size_t landed = cache->host_zone_used(host_zone_id::WEIGHT) - before;
            switch (attribute(md)) {
                case weight_term::HOST_WEIGHTS:
                    attributed_weights += landed;
                    break;
                case weight_term::RUNTIME_RESERVE:
                    attributed_runtime += landed;
                    break;
                case weight_term::UNATTRIBUTED:
                    unattributed++;
                    break;
            }
        }
        held.push_back(std::move(h));
        return 0;
    };

    size_t failed = 0;
    for (const placement_entry & e : plan.entries) {
        if (!e.on_device && take(host_weight_request(q, e.src_size)) == SIZE_MAX) {
            failed++;
        }
    }
    // The runtime HOST_COMPUTE requests the reserve covers: a logits-sized output buffer.
    if (take(host_compute_request(q, 8 * MiB)) == SIZE_MAX) {
        failed++;
    }

    check_eq("planned WEIGHT allocations that failed", failed, 0);
    check_eq("planned WEIGHT allocations that landed outside WEIGHT", not_in_weight, 0);
    check_eq("WEIGHT allocations with no named term", unattributed, 0);
    check(attributed_weights >= t.host_weights, "the host_weights term's allocations hold at least its exact bytes");
    check(attributed_weights <= t.host_weights + t.blocks * k_tlsf_grain,
          "the host_weights allocations stay within host_weights + tlsf_bound");
    check(attributed_runtime <= k_runtime_reserve_bytes, "the HOST_COMPUTE allocations stay within runtime_reserve");
    check_eq("the WEIGHT zone did not grow", cache->host_zone_capacity(host_zone_id::WEIGHT), capacity_before);
    check(cache->host_zone_used(host_zone_id::WEIGHT) - used_before <= plan.host_zone_weight_bytes,
          "the census fits the planned WEIGHT zone");
    std::printf("  census: host_weights=%zu (term %zu + tlsf %zu) runtime=%zu (term %zu) blocks=%zu\n",
                attributed_weights, t.host_weights, t.blocks * k_tlsf_grain, attributed_runtime,
                k_runtime_reserve_bytes, t.blocks);

    held.clear();
    check_eq("the census released every WEIGHT allocation", cache->host_zone_used(host_zone_id::WEIGHT), used_before);
    return true;
}

}  // namespace

int main() {
    std::printf("=== the host WEIGHT zone is exact bytes plus named terms ===\n");
    // One chunk holds every zone, so the census measures TLSF rounding and not where a chunk ends.
    if (std::getenv("GGML_SYCL_PINNED_CHUNK_MB") == nullptr) {
        setenv("GGML_SYCL_PINNED_CHUNK_MB", "512", 0);
    }

    placement_plan plan{};
    arm1_weight_zone_is_exact_bytes_plus_named_terms(plan);

    const bool ran_census = arm2_every_weight_zone_allocation_has_a_term(plan);

    std::printf("=== %d checks, %d failures ===\n", g_checks, g_fail);
    if (g_fail > 0) {
        std::printf("FAILED\n");
        return 1;
    }
    if (!ran_census) {
        std::printf("SKIP: arm 1 passed; the census needs a device and proved nothing\n");
        return SYCL_TEST_SKIP;
    }
    std::printf("PASSED\n");
    return 0;
}
