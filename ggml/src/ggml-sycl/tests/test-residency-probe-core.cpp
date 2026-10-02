// Host tests of the residency probe's pure core (llama.cpp-moua L4 step 3d, llama.cpp-5cim):
// residency-probe.hpp.  The core answers, from one input value, which layers a context's plan
// leaves in host memory: the KV slots are fitted on each device's shared zone with the context's
// tenant head slots in front, and a layer the fit demotes (or the caller forced) is host-resident.
//
//   (1) determinism: the same input answers byte-identically, and the order the tenants arrive in
//       cannot matter (the core canonicalizes them);
//   (2) forced_host: a forced layer stays on the host even with room for it, and an index that names
//       a layer with no KV, a layer past the model, or the same layer twice is INVALID;
//   (3) no_promotion: a demotion of a layer the caller did not force is NO_PROMOTION_VIOLATED and
//       carries no vector, the same input without the flag is OK with that layer on the host, and a
//       demotion the caller forced is not a violation;
//   (4) layers with no KV answer 0 and never cause a demotion; layers the placement plan has on the
//       host answer 1 and take no room from the zone;
//   (5) a head slot that cannot fit is HEAD_SLOT_REFUSED naming its cohort, and a refusal carries no
//       vector;
//   (6) tenants only ever add host layers (probe(none) is a subset of probe(tenants), and a larger
//       tenant never un-demotes a layer), checked over a sweep, with a positive control where the tenant
//       head slot is the sole cause of a demotion so a core that ignored tenants fails;
//   (7) everything the core cannot vouch for is INVALID by name: a tenant of an unknown cohort, a
//       tenant or a layer on a device the input has no geometry for, layers that are not numbered 0..n-1.
//       A host-tier tenant names no zone and changes nothing.
//
// The build is -DNDEBUG, so CHECK is explicit and always runs.

#include "../residency-probe.hpp"
#include "kv-region-test-model.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

namespace krt = kv_region_test;
using ggml_sycl::residency_probe_core;
using ggml_sycl::residency_probe_input;
using ggml_sycl::residency_probe_layer;
using ggml_sycl::residency_probe_result;
using ggml_sycl::runtime_context_tenant;
using krt::MiB;

namespace {

// A zone of one gap: a context block at the top, weights at the bottom, `gap` between them
// (the shape test-kv-runtime-demotion.cpp uses for its head-slot cases).
krt::device_model gap_zone(size_t gap) {
    krt::device_model dev;
    dev.tlsfs.emplace_back(MiB + 16 * MiB + gap);
    dev.tlsfs[0].context(MiB);
    dev.tlsfs[0].weight(16 * MiB);
    return dev;
}

residency_probe_layer kv_layer(size_t kv_bytes, int32_t device = 0) {
    residency_probe_layer l;
    l.has_kv   = true;
    l.group    = ggml_sycl::KV_SLOT_FULL;
    l.kv_bytes = kv_bytes;
    l.device   = device;
    return l;
}

residency_probe_layer no_kv_layer() {
    residency_probe_layer l;
    l.has_kv   = false;
    l.kv_bytes = 999 * 1024 * MiB;  // never read: there is no KV to place
    l.device   = 0;
    return l;
}

runtime_context_tenant compute_tenant(int32_t device, uint32_t index, size_t bytes) {
    runtime_context_tenant t;
    t.device     = device;
    t.cohort     = GGML_SYCL_CONTEXT_COHORT_COMPUTE;
    t.slot_index = index;
    t.slot_bytes = bytes;
    return t;
}

// n_layer layers of 100 MiB of KV each on device 0, whose zone has `gap` bytes of room.  The 900 MiB case
// is the ring case of test-kv-runtime-demotion.cpp: eight layers fit, and a 150 MiB head slot costs one.
residency_probe_input single_device(size_t gap, uint32_t n_layer = 8) {
    residency_probe_input in;
    in.devices.push_back({ 0, gap_zone(gap).snapshot() });
    for (uint32_t l = 0; l < n_layer; ++l) {
        in.layers.push_back(kv_layer(100 * MiB));
    }
    return in;
}

std::vector<uint32_t> hosts(const residency_probe_result & r) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < r.host_resident.size(); ++i) {
        if (r.host_resident[i]) {
            out.push_back((uint32_t) i);
        }
    }
    return out;
}

bool is_subset(const std::vector<uint32_t> & a, const std::vector<uint32_t> & b) {
    return std::includes(b.begin(), b.end(), a.begin(), a.end());
}

constexpr auto OK = GGML_SYCL_RESIDENCY_PROBE_OK;

void case_deterministic() {
    residency_probe_input in = single_device(900 * MiB);
    in.tenants.push_back(compute_tenant(0, 0, 150 * MiB));
    const residency_probe_result a = residency_probe_core(in);
    const residency_probe_result b = residency_probe_core(in);
    CHECK(a.status == OK, "the ring case answers OK");
    CHECK(a.status == b.status && a.host_resident == b.host_resident && a.reason == b.reason,
          "the same input answers byte-identically");
    CHECK(a.host_resident.size() == 8, "one byte per layer");
}

void case_tenant_order_cannot_matter() {
    // Two devices, four tenants: every permutation of the tenant list answers the same bytes.
    residency_probe_input in;
    in.devices.push_back({ 0, gap_zone(900 * MiB).snapshot() });
    in.devices.push_back({ 1, gap_zone(700 * MiB).snapshot() });
    for (uint32_t l = 0; l < 8; ++l) {
        in.layers.push_back(kv_layer(150 * MiB, (int32_t) (l % 2)));
    }
    std::vector<runtime_context_tenant> tenants = {
        compute_tenant(0, 0, 30 * MiB),
        compute_tenant(0, 1, 20 * MiB),
        compute_tenant(1, 0, 200 * MiB),
        compute_tenant(1, 1, 120 * MiB),
    };
    std::sort(tenants.begin(), tenants.end(), ggml_sycl::runtime_context_tenant_less);
    in.tenants                        = tenants;
    const residency_probe_result want = residency_probe_core(in);
    CHECK(want.status == OK, "the two-device case answers OK");
    CHECK(!hosts(want).empty(), "the case demotes something, or the permutations prove nothing");
    size_t n_perm = 0;
    do {
        in.tenants                       = tenants;
        const residency_probe_result got = residency_probe_core(in);
        CHECK(got.status == want.status && got.host_resident == want.host_resident,
              "a permutation of the tenants changed the answer");
        ++n_perm;
    } while (std::next_permutation(tenants.begin(), tenants.end(),
                                   [](const runtime_context_tenant & a, const runtime_context_tenant & b) {
                                       return ggml_sycl::runtime_context_tenant_less(a, b);
                                   }));
    CHECK(n_perm == 24, "all 24 orders were tried");
}

void case_forced_host() {
    residency_probe_input in = single_device(2000 * MiB);
    CHECK(hosts(residency_probe_core(in)).empty(), "with room for everything nothing is on the host");
    in.forced_host           = { 2 };
    residency_probe_result r = residency_probe_core(in);
    CHECK(r.status == OK && hosts(r) == std::vector<uint32_t>({ 2 }),
          "a forced layer stays on the host with room for it");

    in.forced_host = { 3, 3 };
    CHECK(residency_probe_core(in).status == GGML_SYCL_RESIDENCY_PROBE_INVALID,
          "the same layer forced twice is INVALID");
    in.forced_host = { 8 };
    CHECK(residency_probe_core(in).status == GGML_SYCL_RESIDENCY_PROBE_INVALID,
          "a forced index past the model is INVALID");
    in.layers[5]   = no_kv_layer();
    in.forced_host = { 5 };
    r              = residency_probe_core(in);
    CHECK(r.status == GGML_SYCL_RESIDENCY_PROBE_INVALID && r.host_resident.empty() && !r.reason.empty(),
          "a forced layer with no KV is INVALID, with a reason and no vector");
}

void case_no_promotion() {
    residency_probe_input in = single_device(900 * MiB);
    in.tenants.push_back(compute_tenant(0, 0, 150 * MiB));
    residency_probe_result r = residency_probe_core(in);
    CHECK(r.status == OK && hosts(r) == std::vector<uint32_t>({ 7 }), "without the flag the head slot demotes layer 7");

    in.no_promotion = true;
    r               = residency_probe_core(in);
    CHECK(r.status == GGML_SYCL_RESIDENCY_PROBE_NO_PROMOTION_VIOLATED,
          "the same demotion under no_promotion is a violation");
    CHECK(r.host_resident.empty() && !r.reason.empty(), "a violation carries a reason and no vector");

    // The demotion the caller already forced is inherited, not a violation.
    in.forced_host = { 7 };
    r              = residency_probe_core(in);
    CHECK(r.status == OK && hosts(r) == std::vector<uint32_t>({ 7 }), "a forced demotion is not a violation");
}

void case_layers_without_a_zone_claim() {
    residency_probe_input base = single_device(900 * MiB);
    base.tenants.push_back(compute_tenant(0, 0, 150 * MiB));
    const residency_probe_result want = residency_probe_core(base);

    // A layer with no KV answers 0 and is no cause of a demotion, whatever bytes it carries.
    residency_probe_input in = base;
    in.layers.push_back(no_kv_layer());
    residency_probe_result r = residency_probe_core(in);
    CHECK(r.status == OK && r.host_resident.size() == 9 && r.host_resident[8] == 0, "a layer with no KV answers 0");
    CHECK(hosts(r) == hosts(want), "a layer with no KV changed the others' residency");

    // A layer the placement plan has on the host answers 1 and takes no room.
    in = base;
    in.layers.push_back(kv_layer(5000 * MiB, -1));
    r = residency_probe_core(in);
    CHECK(r.status == OK && r.host_resident.size() == 9 && r.host_resident[8] == 1, "a plan-host layer answers 1");
    r.host_resident.pop_back();
    CHECK(r.host_resident == want.host_resident, "a plan-host layer took room from the zone");
}

void case_head_slot_refused() {
    residency_probe_input in = single_device(100 * MiB);
    in.tenants.push_back(compute_tenant(0, 0, 150 * MiB));
    const residency_probe_result r = residency_probe_core(in);
    CHECK(r.status == GGML_SYCL_RESIDENCY_PROBE_HEAD_SLOT_REFUSED, "a head slot with no room is refused");
    CHECK(r.host_resident.empty(), "a refusal carries no vector");
    CHECK(r.reason.find("context-compute") != std::string::npos, "the refusal names the cohort");
}

void case_tenants_only_add_host_layers() {
    // The sweep: probe(none), then growing tenants, over several zone sizes and two forced sets.  The host set
    // never shrinks as a tenant is added or grows.
    const size_t gaps[] = { 600 * MiB, 850 * MiB, 900 * MiB, 1100 * MiB };
    for (const size_t gap : gaps) {
        for (int forced = 0; forced < 2; ++forced) {
            residency_probe_input none = single_device(gap);
            if (forced) {
                none.forced_host = { 1, 6 };
            }
            const residency_probe_result r0 = residency_probe_core(none);
            CHECK(r0.status == OK, "probe(none) answers OK");
            std::vector<uint32_t> prev = hosts(r0);
            for (size_t bytes = 25 * MiB; bytes <= 300 * MiB; bytes += 25 * MiB) {
                residency_probe_input in = none;
                in.tenants.push_back(compute_tenant(0, 0, bytes));
                const residency_probe_result r = residency_probe_core(in);
                if (r.status != OK) {
                    CHECK(r.status == GGML_SYCL_RESIDENCY_PROBE_HEAD_SLOT_REFUSED, "only a refusal can end the sweep");
                    break;
                }
                const std::vector<uint32_t> now = hosts(r);
                CHECK(is_subset(hosts(r0), now), "a tenant un-demoted a layer probe(none) had on the host");
                CHECK(is_subset(prev, now), "a larger tenant un-demoted a layer");
                prev = now;
            }
        }
    }

    // Positive control: the tenant head slot is the SOLE cause of the demotion, so a core that ignored tenants
    // answers the same for both inputs and fails here.
    residency_probe_input none = single_device(900 * MiB);
    residency_probe_input with = none;
    with.tenants.push_back(compute_tenant(0, 0, 150 * MiB));
    CHECK(hosts(residency_probe_core(none)).empty(), "control: nothing is demoted without the tenant");
    CHECK(hosts(residency_probe_core(with)) == std::vector<uint32_t>({ 7 }),
          "control: the tenant alone demotes layer 7");
}

void case_invalid_input() {
    residency_probe_input base = single_device(900 * MiB);

    residency_probe_input  in = base;
    runtime_context_tenant t  = compute_tenant(0, 0, 10 * MiB);
    t.cohort                  = 9999;
    in.tenants.push_back(t);
    CHECK(residency_probe_core(in).status == GGML_SYCL_RESIDENCY_PROBE_INVALID,
          "a tenant of an unknown cohort is INVALID");

    in = base;
    in.tenants.push_back(compute_tenant(3, 0, 10 * MiB));
    CHECK(residency_probe_core(in).status == GGML_SYCL_RESIDENCY_PROBE_INVALID,
          "a tenant on a device with no geometry is INVALID");

    in                  = base;
    in.layers[4].device = 5;
    CHECK(residency_probe_core(in).status == GGML_SYCL_RESIDENCY_PROBE_INVALID,
          "a layer planned on a device with no geometry is INVALID");

    // A host-tier tenant names no zone: the answer is the one without it.
    in = base;
    runtime_context_tenant h;
    h.device     = -1;
    h.cohort     = GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST;
    h.slot_index = 0;
    h.slot_bytes = 4096 * MiB;
    in.tenants.push_back(h);
    const residency_probe_result r = residency_probe_core(in);
    const residency_probe_result w = residency_probe_core(base);
    CHECK(r.status == OK && r.host_resident == w.host_resident, "a host-tier tenant changed the device answer");
}

// The caller's result struct is gated on what the caller declared, before the proc writes a byte of it: a pointer, a
// struct_size at least the one this module was built with and the version it knows.  An older or smaller struct (a
// caller built against a layout with fewer fields) is refused, never written past its declared size; a larger one
// (a newer caller) is read as the layout this module knows.
void case_out_struct_gate() {
    ggml_sycl_residency_probe out{};
    out.struct_size = sizeof(out);
    out.version     = GGML_SYCL_RESIDENCY_PROBE_VERSION;
    CHECK(ggml_sycl::residency_probe_out_declared(&out), "control: a correctly declared struct is accepted");
    CHECK(!ggml_sycl::residency_probe_out_declared(nullptr), "a null result is refused");

    for (uint32_t size = 0; size < sizeof(out); ++size) {
        out.struct_size = size;
        CHECK(!ggml_sycl::residency_probe_out_declared(&out),
              "a struct_size below the layout this module writes is refused");
    }
    out.struct_size = sizeof(out) - 1;
    CHECK(!ggml_sycl::residency_probe_out_declared(&out), "one byte short is refused");

    out.struct_size = sizeof(out) + 8;
    CHECK(ggml_sycl::residency_probe_out_declared(&out),
          "a larger (newer) struct is read as the layout this module knows");

    out.struct_size = sizeof(out);
    for (uint32_t version : { 0u, (uint32_t) GGML_SYCL_RESIDENCY_PROBE_VERSION + 1u, 0xFFFFFFFFu }) {
        out.version = version;
        CHECK(!ggml_sycl::residency_probe_out_declared(&out), "a version this module does not know is refused");
    }
}

}  // namespace

int main() {
    case_deterministic();
    case_tenant_order_cannot_matter();
    case_forced_host();
    case_no_promotion();
    case_layers_without_a_zone_claim();
    case_head_slot_refused();
    case_tenants_only_add_host_layers();
    case_invalid_input();
    case_out_struct_gate();
    std::printf("test-residency-probe-core: all cases passed\n");
    return 0;
}
