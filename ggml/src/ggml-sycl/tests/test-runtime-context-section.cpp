// Host tests of the published section and the load's compute-term ledger (llama.cpp-moua
// L4 step 3): runtime-context-section.hpp.
//
//   (1) parse: a descriptor is read once, at the publisher's own strides, into a sorted
//       owning section; tenants_planned is derived from the tenant section; the tenant key is
//       the digest llama computes;
//   (2) every refusal gate refuses by its own name and leaves nothing behind;
//   (3) coverage: EQUAL / COVERED / GROWTH, fail-closed, with one arm per way a candidate can
//       need more than the published section;
//   (4) the ledger: an early term is recorded only for a load that carries an n_ctx, and the late
//       check answers EQUAL / SHRINK_ADMITTED / REFUSED / NOT_RECORDED under the one rule;
//   (5) the registry entry that holds a published section: read, replace, drop, and never a final
//       drop under the registry's leaf lock.
//
// The build is -DNDEBUG, so CHECK is explicit and always runs.

#include "../kv-region-registry.hpp"
#include "../runtime-context-section.hpp"
#include "llama-context-tenant.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

using namespace ggml_sycl;

namespace {

constexpr int N_DEVICES = 2;

// A descriptor and the buffers it points at, laid out at a stride wider than the element so a
// reader that assumed sizeof(element) reads the wrong bytes.
struct built_desc {
    std::vector<unsigned char>     tenants_raw;
    std::vector<unsigned char>     layers_raw;
    std::vector<unsigned char>     rs_raw;
    ggml_sycl_runtime_context_desc desc{};
    uint32_t                       PAD = 8;

    template <typename T> void put(std::vector<unsigned char> & raw, size_t i, const T & v) {
        std::memcpy(raw.data() + i * (sizeof(T) + PAD), &v, sizeof(T));
    }

    built_desc(const std::vector<ggml_sycl_context_tenant_desc> & tenants,
               const std::vector<ggml_sycl_kv_layer_desc> &       layers,
               const std::vector<ggml_sycl_rs_layer_desc> &       rs,
               uint32_t                                           pad = 8) :
        PAD(pad) {
        tenants_raw.assign(tenants.size() * (sizeof(ggml_sycl_context_tenant_desc) + PAD), 0xEE);
        layers_raw.assign(layers.size() * (sizeof(ggml_sycl_kv_layer_desc) + PAD), 0xEE);
        rs_raw.assign(rs.size() * (sizeof(ggml_sycl_rs_layer_desc) + PAD), 0xEE);
        for (size_t i = 0; i < tenants.size(); ++i) {
            put(tenants_raw, i, tenants[i]);
        }
        for (size_t i = 0; i < layers.size(); ++i) {
            put(layers_raw, i, layers[i]);
        }
        for (size_t i = 0; i < rs.size(); ++i) {
            put(rs_raw, i, rs[i]);
        }
        desc.struct_size      = sizeof(desc);
        desc.version          = GGML_SYCL_RUNTIME_CONTEXT_DESC_VERSION;
        desc.type_k           = 1;
        desc.type_v           = 1;
        desc.v_trans          = 0;
        desc.no_alloc         = 0;
        desc.sidecar          = 0;
        desc.pad0             = 0;
        desc.n_stream         = 1;
        desc.n_layer          = (uint32_t) layers.size();
        desc.layer_desc_size  = (uint32_t) (sizeof(ggml_sycl_kv_layer_desc) + PAD);
        desc.layers           = layers.empty() ? nullptr : (const ggml_sycl_kv_layer_desc *) layers_raw.data();
        desc.n_tenants        = (uint32_t) tenants.size();
        desc.tenant_desc_size = (uint32_t) (sizeof(ggml_sycl_context_tenant_desc) + PAD);
        desc.tenants          = tenants.empty() ? nullptr : (const ggml_sycl_context_tenant_desc *) tenants_raw.data();
        desc.n_rs_layer       = (uint32_t) rs.size();
        desc.rs_layer_desc_size = (uint32_t) (sizeof(ggml_sycl_rs_layer_desc) + PAD);
        desc.rs_layers          = rs.empty() ? nullptr : (const ggml_sycl_rs_layer_desc *) rs_raw.data();
    }
};

ggml_sycl_context_tenant_desc tenant(int32_t device, uint32_t cohort, uint32_t index, uint64_t bytes) {
    ggml_sycl_context_tenant_desc e{};
    e.struct_size = sizeof(e);
    e.cohort      = cohort;
    e.slot_index  = index;
    e.device      = device;
    e.slot_bytes  = bytes;
    return e;
}

ggml_sycl_kv_layer_desc layer(uint32_t k, uint32_t v) {
    ggml_sycl_kv_layer_desc l{};
    l.n_embd_k_gqa  = k;
    l.n_embd_v_gqa  = v;
    l.n_head_kv     = 8;
    l.n_embd_head_k = 128;
    l.has_kv        = 1;
    return l;
}

ggml_sycl_rs_layer_desc rs_layer(uint32_t il) {
    ggml_sycl_rs_layer_desc r{};
    r.il       = il;
    r.type_r   = 0;
    r.type_s   = 0;
    r.n_embd_r = 64;
    r.n_embd_s = 128;
    r.n_rows   = 4;
    return r;
}

runtime_context_geometry geometry() {
    runtime_context_geometry g;
    g.n_ctx      = 8192;
    g.n_ubatch   = 512;
    g.n_seq_max  = 4;
    g.kv_unified = false;
    g.swa_full   = false;
    g.flash_attn = true;
    return g;
}

const std::vector<ggml_sycl_context_tenant_desc> base_tenants() {
    return { tenant(1, GGML_SYCL_CONTEXT_COHORT_COMPUTE, 1, 3000),
             tenant(-1, GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST, 0, 5000),
             tenant(0, GGML_SYCL_CONTEXT_COHORT_COMPUTE, 0, 2000),
             tenant(0, GGML_SYCL_CONTEXT_COHORT_COMPUTE, 1, 1000) };
}

runtime_context_section parse_ok(const ggml_sycl_runtime_context_desc * d, const runtime_context_geometry & g) {
    runtime_context_section s;
    CHECK(parse_runtime_context_desc(d, g, N_DEVICES, s) == runtime_context_desc_status::OK, "the descriptor parses");
    return s;
}

runtime_context_section base_section(const runtime_context_geometry & g = geometry()) {
    built_desc b(base_tenants(), { layer(1024, 1024), layer(1024, 1024) }, { rs_layer(2) });
    return parse_ok(&b.desc, g);
}

// ---- (1) parse -------------------------------------------------------------------------------

void case_parse_reads_strides_and_sorts() {
    built_desc                    b(base_tenants(), { layer(1024, 512), layer(2048, 0) }, { rs_layer(3) });
    const runtime_context_section s = parse_ok(&b.desc, geometry());
    CHECK(s.tenants.size() == 4, "all four elements are read at the publisher's stride");
    CHECK(s.tenants[0].device == -1 && s.tenants[0].cohort == GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST &&
              s.tenants[0].slot_bytes == 5000,
          "the host-tier element sorts first (device -1)");
    CHECK(s.tenants[1].device == 0 && s.tenants[1].slot_index == 0 && s.tenants[1].slot_bytes == 2000,
          "then device 0 slot 0");
    CHECK(s.tenants[2].device == 0 && s.tenants[2].slot_index == 1 && s.tenants[2].slot_bytes == 1000,
          "device 0 slot 1");
    CHECK(s.tenants[3].device == 1 && s.tenants[3].slot_bytes == 3000, "device 1");
    CHECK(s.kv.layers.size() == 2 && s.kv.layers[0].n_embd_k_gqa == 1024 && s.kv.layers[0].n_embd_v_gqa == 512 &&
              s.kv.layers[1].n_embd_k_gqa == 2048 && s.kv.layers[1].n_embd_v_gqa == 0,
          "the layers are read at their stride");
    CHECK(s.kv.rs_layers.size() == 1 && s.kv.rs_layers[0].il == 3, "the recurrent-state section is read");
    CHECK(s.kv.type_k == 1 && s.kv.n_stream == 1 && !s.kv.v_trans && !s.kv.no_alloc && !s.kv.sidecar,
          "the KV scalars are copied");
    CHECK(s.geometry == geometry(), "the geometry is the call's own");
    CHECK(s.tenants_planned, "a tenant section makes the context planned");
    CHECK(s.tenant_key == runtime_context_tenant_key(s.tenants), "the stored key is the digest of the sorted section");

    // A stride of exactly the element's size is a valid stride, not a refusal.
    built_desc                    tight(base_tenants(), { layer(1024, 512), layer(2048, 0) }, { rs_layer(3) }, 0);
    const runtime_context_section t = parse_ok(&tight.desc, geometry());
    CHECK(t.tenants.size() == 4 && t.tenants[3].slot_bytes == 3000 && t.kv.layers[1].n_embd_k_gqa == 2048 &&
              t.kv.rs_layers[0].il == 3,
          "a tight stride reads every element");

    // A struct_size above this build's layout is read as far as this build's layout goes.
    b.desc.struct_size = sizeof(b.desc) + 16;
    runtime_context_section wider;
    CHECK(parse_runtime_context_desc(&b.desc, geometry(), N_DEVICES, wider) == runtime_context_desc_status::OK,
          "a wider publisher parses");
}

void case_planned_follows_the_tenant_section() {
    built_desc                    none({}, { layer(1, 1) }, {});
    const runtime_context_section s = parse_ok(&none.desc, geometry());
    CHECK(!s.tenants_planned && s.tenants.empty(), "no tenant section: not planned");
    built_desc one({ tenant(0, GGML_SYCL_CONTEXT_COHORT_COMPUTE, 0, 1) }, {}, {});
    CHECK(parse_ok(&one.desc, geometry()).tenants_planned, "one tenant element: planned");
}

// The backend's digest and llama's are one fact with two readers.
void case_key_is_llamas_digest() {
    const runtime_context_section              s = base_section();
    std::vector<ggml_sycl_context_tenant_desc> as_wire;
    for (const runtime_context_tenant & e : s.tenants) {
        as_wire.push_back(tenant(e.device, e.cohort, e.slot_index, e.slot_bytes));
    }
    CHECK(s.tenant_key == llama_tenant_key_digest(as_wire),
          "the backend's tenant key is llama's digest of the same section");
    CHECK(runtime_context_tenant_key({}) == llama_tenant_key_digest({}), "and for the empty section");
}

// ---- (2) refusals ----------------------------------------------------------------------------

void expect_refused(built_desc & b, runtime_context_desc_status want, const char * msg) {
    runtime_context_section s;
    s.tenant_key                          = 99;
    const runtime_context_desc_status got = parse_runtime_context_desc(&b.desc, geometry(), N_DEVICES, s);
    CHECK(got == want, msg);
    CHECK(s.tenants.empty() && s.kv.layers.empty() && !s.tenants_planned && s.tenant_key == 0,
          "a refusal leaves nothing behind");
}

void case_refusals() {
    using st = runtime_context_desc_status;
    runtime_context_section s;
    CHECK(parse_runtime_context_desc(nullptr, geometry(), N_DEVICES, s) == st::NULL_DESC, "null descriptor");
    {
        built_desc b(base_tenants(), { layer(1, 1) }, {});
        b.desc.struct_size = sizeof(b.desc) - 1;
        expect_refused(b, st::SHORT_STRUCT, "struct_size below the version-1 layout");
    }
    {
        built_desc b(base_tenants(), { layer(1, 1) }, {});
        b.desc.version = GGML_SYCL_RUNTIME_CONTEXT_DESC_VERSION + 1;
        expect_refused(b, st::UNKNOWN_VERSION, "an unknown version");
        b.desc.version = 0;
        expect_refused(b, st::UNKNOWN_VERSION, "version 0");
    }
    {
        built_desc b(base_tenants(), { layer(1, 1) }, {});
        b.desc.pad0 = 1;
        expect_refused(b, st::BAD_PAD, "pad0 not 0");
    }
    {
        built_desc b(base_tenants(), { layer(1, 1) }, {});
        b.desc.v_trans = 2;
        expect_refused(b, st::BAD_FLAG, "a flag byte of 2");
    }
    {
        built_desc b(base_tenants(), { layer(1, 1) }, {});
        b.desc.sidecar = 2;
        expect_refused(b, st::BAD_FLAG, "a sidecar flag byte of 2");
    }
    {
        built_desc b(base_tenants(), { layer(1, 1) }, {});
        b.desc.no_alloc = 2;
        expect_refused(b, st::BAD_FLAG, "a no_alloc flag byte of 2");
    }
    {
        built_desc b(base_tenants(), { layer(1, 1) }, {});
        b.desc.tenants = nullptr;
        expect_refused(b, st::BAD_ARRAY, "tenants counted but absent");
    }
    {
        built_desc b(base_tenants(), { layer(1, 1) }, {});
        b.desc.tenant_desc_size = sizeof(ggml_sycl_context_tenant_desc) - 1;
        expect_refused(b, st::BAD_ARRAY, "a tenant stride below the element");
    }
    {
        built_desc b(base_tenants(), { layer(1, 1) }, {});
        b.desc.layer_desc_size = sizeof(ggml_sycl_kv_layer_desc) - 1;
        expect_refused(b, st::BAD_ARRAY, "a layer stride below the element");
    }
    {
        built_desc b(base_tenants(), { layer(1, 1) }, { rs_layer(0) });
        b.desc.rs_layers = nullptr;
        expect_refused(b, st::BAD_ARRAY, "rs layers counted but absent");
    }
    {
        built_desc b(base_tenants(), { layer(1, 1) }, {});
        b.desc.n_tenants = RUNTIME_CONTEXT_DESC_MAX_ELEMENTS + 1;
        expect_refused(b, st::BAD_ARRAY, "a count over the cap");
    }
    {
        auto t           = base_tenants();
        t[2].struct_size = sizeof(ggml_sycl_context_tenant_desc) - 1;
        built_desc b(t, { layer(1, 1) }, {});
        expect_refused(b, st::BAD_ELEMENT, "an element under its own struct_size");
    }
    {
        auto t      = base_tenants();
        t[1].cohort = GGML_SYCL_CONTEXT_COHORT_COUNT;
        built_desc b(t, { layer(1, 1) }, {});
        expect_refused(b, st::UNKNOWN_COHORT, "a cohort id the table does not know");
    }
    {
        auto t = base_tenants();
        t[1]   = tenant(0, GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST, 0, 5000);
        built_desc b(t, { layer(1, 1) }, {});
        expect_refused(b, st::BAD_DEVICE, "a host-tier element on a device index");
    }
    {
        auto t = base_tenants();
        t[0]   = tenant(-1, GGML_SYCL_CONTEXT_COHORT_COMPUTE, 1, 3000);
        built_desc b(t, { layer(1, 1) }, {});
        expect_refused(b, st::BAD_DEVICE, "a device-tier element on -1");
    }
    {
        auto t = base_tenants();
        t[0]   = tenant(N_DEVICES, GGML_SYCL_CONTEXT_COHORT_COMPUTE, 1, 3000);
        built_desc b(t, { layer(1, 1) }, {});
        expect_refused(b, st::BAD_DEVICE, "a device-tier element past the device count");
    }
    {
        auto t          = base_tenants();
        t[3].slot_bytes = 0;
        built_desc b(t, { layer(1, 1) }, {});
        expect_refused(b, st::ZERO_SLOT, "an element with no bytes");
    }
    {
        auto t = base_tenants();
        t.push_back(tenant(0, GGML_SYCL_CONTEXT_COHORT_COMPUTE, 0, 7));
        built_desc b(t, { layer(1, 1) }, {});
        expect_refused(b, st::DUPLICATE_SLOT, "two elements at one slot");
    }
}

// ---- (3) coverage ----------------------------------------------------------------------------

ggml_sycl_tenant_coverage cover(const runtime_context_section & published, const runtime_context_section & candidate) {
    return classify_tenant_coverage(&published, candidate);
}

void case_coverage_geometry() {
    const runtime_context_section pub = base_section();
    CHECK(classify_tenant_coverage(nullptr, pub) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "no published entry reads GROWTH");
    CHECK(cover(pub, base_section()) == GGML_SYCL_TENANT_COVERAGE_EQUAL, "byte-equal reads EQUAL");

    auto with = [&](auto edit) {
        runtime_context_geometry g = geometry();
        edit(g);
        return base_section(g);
    };
    CHECK(cover(pub, with([](runtime_context_geometry & g) { g.n_ctx = 4096; })) == GGML_SYCL_TENANT_COVERAGE_COVERED,
          "a smaller n_ctx is covered");
    CHECK(cover(pub, with([](runtime_context_geometry & g) { g.n_ubatch = 256; })) == GGML_SYCL_TENANT_COVERAGE_COVERED,
          "a smaller n_ubatch is covered");
    CHECK(cover(pub, with([](runtime_context_geometry & g) { g.n_seq_max = 1; })) == GGML_SYCL_TENANT_COVERAGE_COVERED,
          "a smaller n_seq_max is covered");
    CHECK(cover(pub, with([](runtime_context_geometry & g) { g.n_ctx = 8193; })) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "a larger n_ctx grows");
    CHECK(cover(pub, with([](runtime_context_geometry & g) { g.n_ubatch = 513; })) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "a larger n_ubatch grows");
    CHECK(cover(pub, with([](runtime_context_geometry & g) { g.n_seq_max = 5; })) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "a larger n_seq_max grows");
    CHECK(
        cover(pub, with([](runtime_context_geometry & g) { g.kv_unified = true; })) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
        "a kv_unified flip grows");
    CHECK(cover(pub, with([](runtime_context_geometry & g) { g.swa_full = true; })) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "a swa_full flip grows");
    CHECK(cover(pub, with([](runtime_context_geometry & g) { g.flash_attn = false; })) ==
              GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "a flash_attn flip grows");
    // the flips are growth from either side
    runtime_context_geometry unified          = geometry();
    unified.kv_unified                        = true;
    const runtime_context_section pub_unified = base_section(unified);
    CHECK(cover(pub_unified, base_section()) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "and back: a kv_unified flip is growth either way");
    // two fields moving opposite ways: the smaller one does not pay for the larger
    CHECK(cover(pub, with([](runtime_context_geometry & g) {
                    g.n_ctx    = 4096;
                    g.n_ubatch = 1024;
                })) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "a smaller n_ctx does not cover a larger n_ubatch");
}

void case_coverage_shape_and_plan_state() {
    const runtime_context_section pub    = base_section();
    auto                          edited = [&](auto edit) {
        runtime_context_section c = base_section();
        edit(c);
        return c;
    };
    CHECK(cover(pub, edited([](runtime_context_section & c) { c.kv.layers[1].n_embd_k_gqa += 1; })) ==
              GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "a layer width differs");
    CHECK(cover(pub, edited([](runtime_context_section & c) { c.kv.layers.pop_back(); })) ==
              GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "a layer is missing");
    CHECK(cover(pub, edited([](runtime_context_section & c) { c.kv.rs_layers[0].n_rows += 1; })) ==
              GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "a recurrent-state layer differs");
    CHECK(cover(pub, edited([](runtime_context_section & c) { c.kv.rs_layers.push_back(c.kv.rs_layers[0]); })) ==
              GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "the candidate has an extra recurrent-state layer");
    CHECK(cover(pub, edited([](runtime_context_section & c) { c.kv.rs_layers.clear(); })) ==
              GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "the candidate lacks the published recurrent-state layer");
    {
        runtime_context_section no_rs = base_section();
        no_rs.kv.rs_layers.clear();
        CHECK(cover(no_rs, base_section()) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
              "a recurrent-state layer against a published section with none");
    }
    CHECK(cover(pub, edited([](runtime_context_section & c) { c.kv.type_k += 1; })) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "type_k differs");
    CHECK(cover(pub, edited([](runtime_context_section & c) { c.kv.type_v += 1; })) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "type_v differs");
    CHECK(cover(pub, edited([](runtime_context_section & c) { c.kv.v_trans = true; })) ==
              GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "v_trans differs");
    CHECK(cover(pub, edited([](runtime_context_section & c) { c.kv.sidecar = true; })) ==
              GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "sidecar differs");
    CHECK(cover(pub, edited([](runtime_context_section & c) { c.kv.no_alloc = true; })) ==
              GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "no_alloc differs");
    CHECK(
        cover(pub, edited([](runtime_context_section & c) { c.kv.n_stream += 1; })) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
        "n_stream differs");
    // a candidate with no tenant section against a planned context needs the plan state back
    CHECK(cover(pub, edited([](runtime_context_section & c) {
                    c.tenants.clear();
                    c.tenants_planned = false;
                })) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "an unplanned candidate against a planned entry grows");
    runtime_context_section unplanned = base_section();
    unplanned.tenants.clear();
    unplanned.tenants_planned = false;
    CHECK(cover(unplanned, base_section()) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "a planned candidate against an unplanned entry grows");
    CHECK(cover(unplanned, unplanned) == GGML_SYCL_TENANT_COVERAGE_EQUAL, "two unplanned equal sections are EQUAL");
}

// A publisher copies whole elements, padding included: equal values with different padding bytes
// are one shape.
void case_coverage_ignores_element_padding() {
    const runtime_context_section pub   = base_section();
    runtime_context_section       dirty = base_section();
    unsigned char *               pad   = reinterpret_cast<unsigned char *>(&dirty.kv.layers[0]) +
                          offsetof(ggml_sycl_kv_layer_desc, is_swa) + sizeof(uint8_t);
    pad[0] = 0xAB;
    pad[1] = 0xCD;
    CHECK(std::memcmp(&pub.kv.layers[0], &dirty.kv.layers[0], sizeof(ggml_sycl_kv_layer_desc)) != 0,
          "the two layer elements differ in their padding bytes");
    CHECK(cover(pub, dirty) == GGML_SYCL_TENANT_COVERAGE_EQUAL, "padding bytes do not make a shape GROWTH");
    dirty.kv.layers[0].n_head_kv += 1;
    CHECK(cover(pub, dirty) == GGML_SYCL_TENANT_COVERAGE_GROWTH, "a changed member still does");
    // every member is compared, one at a time
    auto kv_edit = [&](auto edit) {
        runtime_context_section c = base_section();
        edit(c.kv.layers[1]);
        return cover(pub, c);
    };
    CHECK(kv_edit([](ggml_sycl_kv_layer_desc & l) { l.n_embd_k_gqa += 1; }) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "n_embd_k_gqa is compared");
    CHECK(kv_edit([](ggml_sycl_kv_layer_desc & l) { l.n_embd_v_gqa += 1; }) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "n_embd_v_gqa is compared");
    CHECK(kv_edit([](ggml_sycl_kv_layer_desc & l) { l.n_head_kv += 1; }) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "n_head_kv is compared");
    CHECK(kv_edit([](ggml_sycl_kv_layer_desc & l) { l.n_embd_head_k += 1; }) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "n_embd_head_k is compared");
    CHECK(kv_edit([](ggml_sycl_kv_layer_desc & l) { l.has_kv ^= 1; }) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "has_kv is compared");
    CHECK(kv_edit([](ggml_sycl_kv_layer_desc & l) { l.is_swa ^= 1; }) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "is_swa is compared");
    auto rs_edit = [&](auto edit) {
        runtime_context_section c = base_section();
        edit(c.kv.rs_layers[0]);
        return cover(pub, c);
    };
    CHECK(rs_edit([](ggml_sycl_rs_layer_desc & r) { r.il += 1; }) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "the recurrent-state il is compared");
    CHECK(rs_edit([](ggml_sycl_rs_layer_desc & r) { r.type_r += 1; }) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "type_r is compared");
    CHECK(rs_edit([](ggml_sycl_rs_layer_desc & r) { r.type_s += 1; }) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "type_s is compared");
    CHECK(rs_edit([](ggml_sycl_rs_layer_desc & r) { r.n_embd_r += 1; }) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "n_embd_r is compared");
    CHECK(rs_edit([](ggml_sycl_rs_layer_desc & r) { r.n_embd_s += 1; }) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "n_embd_s is compared");
    CHECK(rs_edit([](ggml_sycl_rs_layer_desc & r) { r.n_rows += 1; }) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "n_rows is compared");
}

// A candidate with no geometry has nothing to compare: never coverage, as the ledger refuses a zero n_ctx.
void case_coverage_refuses_zero_geometry() {
    const runtime_context_section pub  = base_section();
    auto                          with = [&](auto edit) {
        runtime_context_geometry g = geometry();
        edit(g);
        return base_section(g);
    };
    CHECK(cover(pub, with([](runtime_context_geometry & g) { g.n_ctx = 0; })) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "a zero n_ctx is not covered");
    CHECK(cover(pub, with([](runtime_context_geometry & g) { g.n_ubatch = 0; })) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "a zero n_ubatch is not covered");
    CHECK(cover(pub, with([](runtime_context_geometry & g) { g.n_seq_max = 0; })) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "a zero n_seq_max is not covered");
    const runtime_context_section zero = with([](runtime_context_geometry & g) { g.n_ctx = 0; });
    CHECK(cover(zero, zero) == GGML_SYCL_TENANT_COVERAGE_GROWTH, "a zero geometry is not even EQUAL to itself");
}

void case_coverage_slots() {
    const runtime_context_section pub    = base_section();
    auto                          edited = [&](auto edit) {
        runtime_context_section c = base_section();
        edit(c);
        return c;
    };
    CHECK(cover(pub, edited([](runtime_context_section & c) { c.tenants[1].slot_bytes -= 1; })) ==
              GGML_SYCL_TENANT_COVERAGE_COVERED,
          "a smaller slot is covered");
    CHECK(cover(pub, edited([](runtime_context_section & c) { c.tenants[1].slot_bytes += 1; })) ==
              GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "a larger slot grows");
    CHECK(cover(pub, edited([](runtime_context_section & c) {
                    c.tenants.push_back({ 1, GGML_SYCL_CONTEXT_COHORT_COMPUTE, 5, 1 });
                })) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "a slot the published section does not have grows");
    CHECK(cover(pub, edited([](runtime_context_section & c) { c.tenants[1].slot_index = 7; })) ==
              GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "the same bytes at another index are not covered");
    CHECK(cover(pub, edited([](runtime_context_section & c) {
                    c.tenants[2].device     = 1;
                    c.tenants[2].slot_index = 9;
                })) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "the same cohort on another device is not covered");
    CHECK(cover(pub, edited([](runtime_context_section & c) {
                    c.tenants[0].cohort = GGML_SYCL_CONTEXT_COHORT_NONFA_STAGE;
                    c.tenants[0].device = 0;
                })) == GGML_SYCL_TENANT_COVERAGE_GROWTH,
          "another cohort at the same index is not covered");
    // a candidate that is a subset of the published slots is covered, not equal
    CHECK(cover(pub, edited([](runtime_context_section & c) { c.tenants.pop_back(); })) ==
              GGML_SYCL_TENANT_COVERAGE_COVERED,
          "fewer slots than published are covered");
    // the published section may hold more than the candidate asks, and EQUAL needs every slot
    CHECK(cover(pub, edited([](runtime_context_section & c) { c.tenants.erase(c.tenants.begin()); })) ==
              GGML_SYCL_TENANT_COVERAGE_COVERED,
          "dropping the host slot is covered");
}

// ---- (4) the ledger --------------------------------------------------------------------------

void case_ledger_records_only_with_an_n_ctx() {
    load_compute_ledger l;
    CHECK(!l.record(7, 0, 1000, 0), "an envelope with n_ctx 0 records nothing");
    CHECK(l.size() == 0, "and the ledger is empty");
    const load_compute_ledger::check_result miss = l.check(7, 0, 1000, true);
    CHECK(miss.result == GGML_SYCL_LATE_CHECK_NOT_RECORDED,
          "the late check of an unrecorded load answers NOT_RECORDED");
    CHECK(miss.line.find("no early term was recorded") != std::string::npos, "and says nothing was compared");
    const load_compute_ledger::check_result again = l.check(7, 0, 1000, true);
    CHECK(again.result == GGML_SYCL_LATE_CHECK_NOT_RECORDED && again.line.empty(),
          "the second call for the same (load, device) answers NOT_RECORDED without a second line");
    CHECK(!l.check(7, 1, 1000, true).line.empty(), "another device of the same load says it once itself");
    CHECK(!l.check(8, 0, 1000, true).line.empty(), "another load says it once itself");
    CHECK(l.check(8, 0, 1000, true).line.empty(), "and not twice");
    l.clear(8);
    CHECK(!l.check(8, 0, 1000, true).line.empty(), "a load's clear forgets that it spoke");
    CHECK(l.check(7, 0, 1000, true).line.empty(), "and only its own: another load's once-only line stays given");
    CHECK(!l.check(7, 0, 1000, false).line.empty() && !l.check(7, 0, 1000, false).line.empty(),
          "a transaction that is not open logs on every call");
    CHECK(l.record(7, 0, 1000, 8192), "an envelope with an n_ctx records");
    CHECK(l.check(7, 0, 1000, true).result == GGML_SYCL_LATE_CHECK_EQUAL, "and the late check compares");
    CHECK(!l.record(0, 0, 1000, 8192), "transaction 0 records nothing");
    CHECK(!l.record(7, -1, 1000, 8192), "a host-tier device records nothing");
    CHECK(l.size() == 1, "refused records leave the ledger as it was");
}

void case_ledger_rule() {
    load_compute_ledger l;
    CHECK(l.record(7, 0, 1000, 8192) && l.record(7, 1, 2000, 8192), "two devices recorded");
    const load_compute_ledger::check_result eq = l.check(7, 0, 1000, true);
    CHECK(eq.result == GGML_SYCL_LATE_CHECK_EQUAL && eq.line.empty() && !eq.shrink_counted, "equal: EQUAL, no line");

    const load_compute_ledger::check_result big = l.check(7, 0, 1001, true);
    CHECK(big.result == GGML_SYCL_LATE_CHECK_REFUSED, "larger: REFUSED");
    CHECK(big.line ==
              "[LOAD-PLAN] the late inventory changes the zones admitted at the early stage: term compute in zone "
              "context-compute on "
              "device 0, early 1000 B, late 1001 B (refused)",
          "the canonical late string");
    CHECK(!big.shrink_counted, "a refusal counts no shrink");

    const load_compute_ledger::check_result small = l.check(7, 1, 1999, true);
    CHECK(small.result == GGML_SYCL_LATE_CHECK_SHRINK_ADMITTED, "smaller: SHRINK_ADMITTED");
    CHECK(
        small.line ==
            "[ZONE-PLAN-BUG] the late inventory shrinks term compute on device 1: early 2000 B, late 1999 B (admitted; "
            "the early reservation stands)",
        "the canonical shrink WARN");
    CHECK(small.shrink_counted, "the first shrink logs at WARN and counts once");
    const load_compute_ledger::check_result again = l.check(7, 1, 1500, true);
    CHECK(again.result == GGML_SYCL_LATE_CHECK_SHRINK_ADMITTED && again.line.empty() && !again.shrink_counted,
          "a second shrink on the same (load, device, term) is admitted without a second line or count");
    // the early reservation stands: a later larger value is compared with the ADMITTED term, not the shrunk one
    CHECK(l.check(7, 1, 2001, true).result == GGML_SYCL_LATE_CHECK_REFUSED,
          "the admitted term is not lowered by a shrink");
    CHECK(l.check(7, 1, 2000, true).result == GGML_SYCL_LATE_CHECK_EQUAL, "and still equals itself");
    // the other device's shrink is its own once-only
    CHECK(l.check(7, 0, 5, true).shrink_counted, "another device's first shrink counts");
}

void case_ledger_fails_closed() {
    load_compute_ledger l;
    CHECK(l.record(7, 0, 1000, 8192), "recorded");
    const load_compute_ledger::check_result closed = l.check(7, 0, 1000, false);
    CHECK(closed.result == GGML_SYCL_LATE_CHECK_NOT_RECORDED,
          "a transaction that is not open answers NOT_RECORDED, never EQUAL");
    CHECK(closed.line.find("not the open load transaction") != std::string::npos, "and logs why");
    CHECK(closed.line.find("EQUAL") == std::string::npos, "it never reads as a comparison");
    CHECK(l.check(7, 1, 1000, true).result == GGML_SYCL_LATE_CHECK_NOT_RECORDED,
          "a device with no record answers NOT_RECORDED");
    CHECK(l.check(8, 0, 1000, true).result == GGML_SYCL_LATE_CHECK_NOT_RECORDED,
          "another load's transaction answers NOT_RECORDED");
    // a later record of the same key replaces the admitted term
    CHECK(l.record(7, 0, 400, 8192), "re-recorded, smaller");
    CHECK(l.check(7, 0, 400, true).result == GGML_SYCL_LATE_CHECK_EQUAL, "the replacing record is the admitted term");
    CHECK(l.record(7, 0, 4000, 8192), "re-recorded, larger");
    CHECK(l.check(7, 0, 4000, true).result == GGML_SYCL_LATE_CHECK_EQUAL, "and the larger one replaces too");
}

void case_ledger_clear() {
    load_compute_ledger l;
    CHECK(l.record(7, 0, 1, 1) && l.record(7, 1, 2, 1) && l.record(8, 0, 3, 1), "three records");
    CHECK(l.clear(7) == 2, "a load's clear drops its own terms");
    CHECK(l.size() == 1, "and not another load's");
    CHECK(l.check(7, 0, 1, true).result == GGML_SYCL_LATE_CHECK_NOT_RECORDED, "a cleared load answers NOT_RECORDED");
    CHECK(l.check(8, 0, 3, true).result == GGML_SYCL_LATE_CHECK_EQUAL, "the other load is intact");
    CHECK(l.clear(7) == 0, "a second clear is a no-op");
}

// ---- (5) the registry entry's published section -------------------------------------------------

// A section whose last drop records whether the registry's leaf lock was held at that instant.
std::shared_ptr<const runtime_context_section> watched_section(bool * dropped, bool * dropped_under_lock) {
    return std::shared_ptr<const runtime_context_section>(
        new runtime_context_section(base_section()), [dropped, dropped_under_lock](const runtime_context_section * p) {
            *dropped            = true;
            *dropped_under_lock = kv_lock_witness::holds(KV_LOCK_L3_REGISTRY);
            delete p;
        });
}

void case_registry_published_section() {
    kv_region_registry reg;
    CHECK(reg.published_section(5) == nullptr, "a context with no entry has no section");
    CHECK(reg.drop_published_section(5) == nullptr && reg.size() == 0,
          "dropping a section the context never had creates no entry");

    auto first = std::make_shared<const runtime_context_section>(base_section());
    CHECK(reg.set_published_section(5, first) == nullptr, "the first publish replaces nothing");
    CHECK(reg.published_section(5) == first && reg.size() == 1, "the entry holds the section just published");
    CHECK(reg.published_section(6) == nullptr, "another context's entry is separate");
    kv_region_entry seen;
    CHECK(reg.lookup(5, seen) && seen.tenant_key == first->tenant_key && seen.published == first,
          "the entry carries the section's tenant key");

    runtime_context_section changed = base_section();
    changed.tenants[0].slot_bytes += 1;
    changed.tenant_key = runtime_context_tenant_key(changed.tenants);
    auto second        = std::make_shared<const runtime_context_section>(changed);
    auto previous      = reg.set_published_section(5, second);
    CHECK(previous == first, "a republish returns the section it replaced, moved out");
    CHECK(reg.published_section(5) == second && reg.lookup(5, seen) && seen.tenant_key == second->tenant_key,
          "and the entry now holds the new section and its key");
    previous.reset();

    // The entry's other fields survive a section change; the key follows the section only when no slot table owns it.
    kv_region_entry with_extent;
    with_extent.extents.push_back(std::make_shared<int>(1));
    with_extent.published  = first;
    with_extent.tenant_key = first->tenant_key;
    (void) reg.publish(7, std::move(with_extent));
    (void) reg.drop_published_section(7);
    CHECK(reg.lookup(7, seen) && seen.extents.size() == 1 && seen.published == nullptr && seen.tenant_key == 0,
          "dropping the section keeps the extents and clears the key it described");
    kv_region_entry with_slots;
    with_slots.tenants    = std::make_shared<kv_tenant_slots>();
    with_slots.published  = first;
    with_slots.tenant_key = 99;
    (void) reg.publish(8, std::move(with_slots));
    (void) reg.drop_published_section(8);
    CHECK(reg.lookup(8, seen) && seen.tenants != nullptr && seen.tenant_key == 99,
          "a slot table keeps its own key when the section goes");
    // ... and when the section is replaced: the key is the one the held reservation was built for.
    kv_region_entry replaced_slots;
    replaced_slots.tenants    = std::make_shared<kv_tenant_slots>();
    replaced_slots.published  = first;
    replaced_slots.tenant_key = 99;
    (void) reg.publish(11, std::move(replaced_slots));
    CHECK(reg.set_published_section(11, second) == first, "a replace over a slot table returns the old section");
    CHECK(reg.lookup(11, seen) && seen.published == second && seen.tenants != nullptr && seen.tenant_key == 99 &&
              seen.tenant_key != second->tenant_key,
          "a slot table keeps its own key when the section is replaced");
    (void) reg.drop_published_section(11);
    // With no slot table the key is the section's own, whatever the entry held before, and there is no
    // argument to disagree with it.
    kv_region_entry stale_key;
    stale_key.extents.push_back(std::make_shared<int>(1));
    stale_key.tenant_key = 12345;
    (void) reg.publish(12, std::move(stale_key));
    (void) reg.set_published_section(12, second);
    CHECK(reg.lookup(12, seen) && seen.tenant_key == second->tenant_key, "the entry's key is the section's");
    (void) reg.drop_published_section(12);
    CHECK(reg.lookup(12, seen) && seen.tenant_key == 0 && seen.extents.size() == 1,
          "and the drop clears it, keeping the extents");
    // A typed null through set_published_section is the same drop.
    const std::shared_ptr<const runtime_context_section> none;
    CHECK(reg.set_published_section(13, none) == nullptr && !reg.lookup(13, seen),
          "a typed null on a context with no entry creates nothing");
    (void) reg.set_published_section(13, second);
    CHECK(reg.set_published_section(13, none) == second && !reg.lookup(13, seen),
          "a typed null drops the section and erases the entry it leaves empty");
    (void) reg.drop_published_section(5);
    CHECK(reg.published_section(5) == nullptr && !reg.lookup(5, seen) && reg.size() == 4,
          "dropping the section of an otherwise empty entry erases the entry");

    // A section's last drop is the caller's, with the leaf lock released.
    bool dropped = false, under_lock = true;
    CHECK(reg.set_published_section(9, watched_section(&dropped, &under_lock)) == nullptr, "a watched section in");
    CHECK(!dropped, "the registry keeps it alive");
    (void) reg.drop_published_section(9);
    CHECK(dropped && !under_lock, "its last drop ran after the leaf lock was released");
    dropped    = false;
    under_lock = true;
    (void) reg.set_published_section(10, watched_section(&dropped, &under_lock));
    (void) reg.set_published_section(10, std::make_shared<const runtime_context_section>(base_section()));
    CHECK(dropped && !under_lock, "so did the drop of a replaced section");
    CHECK(kv_lock_witness::violations() == 0, "the witness saw no lock-order violation");
}

}  // namespace

int main() {
    case_parse_reads_strides_and_sorts();
    case_planned_follows_the_tenant_section();
    case_key_is_llamas_digest();
    case_refusals();
    case_coverage_geometry();
    case_coverage_shape_and_plan_state();
    case_coverage_ignores_element_padding();
    case_coverage_refuses_zero_geometry();
    case_coverage_slots();
    case_ledger_records_only_with_an_n_ctx();
    case_ledger_rule();
    case_ledger_fails_closed();
    case_ledger_clear();
    case_registry_published_section();
    std::printf("test-runtime-context-section: all cases passed\n");
    return 0;
}
