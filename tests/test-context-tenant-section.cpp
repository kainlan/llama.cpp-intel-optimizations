// zhcn C7g-1: the llama side of the measured-tenant publish. Pure host: it links only
// src/llama-context-tenant.h and the public backend header, so no model, no device and no
// scheduler are involved.
//
// Cases:
//   (1) one header: the public backend header and the cohort ids compile in one translation
//       unit, and the tenant element is the 24-byte wire layout both headers agree on;
//   (2) the L4 procs fail closed: a null publish proc reads UNSUPPORTED, a null coverage proc
//       reads GROWTH, a null late-check proc reads NOT_RECORDED, and an answer outside the
//       enum's range reads as the same closed value instead of being passed on;
//   (3) a non-null proc is called with exactly the caller's arguments and its answer is
//       returned unchanged;
//   (4) the section builder: a device buft's chunk caps become COMPUTE elements indexed by
//       chunk, a host buft's become COMPUTE_HOST elements on device -1 merged by their maximum
//       across devices, a zero cap makes no element, and the result is ordered by
//       (device, cohort, slot_index);
//   (5) the backend's own visitor demands merge into the same section by maximum;
//   (6) a device-tier buft with no device index is a named refusal, not a guess;
//   (7) the tenant key is a digest of (device, cohort, slot_index, slot_bytes) in section
//       order: equal sections give equal keys and any one field changes it;
//   (8) the plan line's text begins with the fields the scorer matches;
//   (9) the compute term over several chunks is the per-chunk peak, summed;
//   (10) the host-tier HOLD (design 3.3): R_h[i] is the maximum over the ladder's rungs of the rung
//       section's COMPUTE_HOST slot i, taken from the section the builder produced (no second
//       derivation), and the section the publish carries holds R_h[i] at every index. Perturbing one
//       rung's measurement changes exactly the slots it dominates. This case proves the fold
//       arithmetic only. That the transaction folds the sections its own MEASUREs produced (and
//       skips a rung whose MEASURE refuses or throws, and marks the hold ready only after the
//       loop) is pinned by tests/test-sycl-tenant-section-source.py, which a host test cannot
//       reach because the transaction lives in llama-context.cpp.
//   (11) the residency probe's one door (llama.cpp-moua L4 step 3d): a null proc and an answer outside the enum
//       read NOT_ANSWERED, only OK is OK, GEOMETRY_NOT_WIRED is not OK and is passed on as itself, the door
//       never touches the caller's host_resident bytes, and a table without the probe proc is not L4.

#include "../src/llama-context-tenant.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int n_failed = 0;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            n_failed++;                                                       \
            fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
            fprintf(stderr, __VA_ARGS__);                                     \
            fprintf(stderr, "\n");                                            \
        }                                                                     \
    } while (0)

// What the fake procs saw.
struct call_record {
    int                                    n_publish   = 0;
    int                                    n_coverage  = 0;
    int                                    n_late      = 0;
    int                                    n_probe     = 0;
    uint32_t                               probe_n_ctx = 0;
    uint32_t                               n_ubatch    = 0;
    uint64_t                               late_bytes  = 0;
    int32_t                                late_dev    = -2;
    uint64_t                               late_id     = 0;
    const ggml_sycl_runtime_context_desc * desc        = nullptr;
};

static call_record g_calls;

static ggml_sycl_lifecycle_result fake_publish(ggml_backend_t,
                                               struct ggml_sycl_model_token,
                                               uint32_t,
                                               uint32_t n_ubatch,
                                               uint32_t,
                                               bool,
                                               bool,
                                               bool,
                                               const ggml_sycl_runtime_context_desc * desc) {
    g_calls.n_publish++;
    g_calls.n_ubatch = n_ubatch;
    g_calls.desc     = desc;
    return GGML_SYCL_LIFECYCLE_PLAN_REJECTED;
}

static ggml_sycl_tenant_coverage fake_coverage(ggml_backend_t,
                                               uint32_t,
                                               uint32_t n_ubatch,
                                               uint32_t,
                                               bool,
                                               bool,
                                               bool,
                                               const ggml_sycl_runtime_context_desc *) {
    g_calls.n_coverage++;
    g_calls.n_ubatch = n_ubatch;
    return GGML_SYCL_TENANT_COVERAGE_COVERED;
}

static ggml_sycl_tenant_coverage
wild_coverage(ggml_backend_t, uint32_t, uint32_t, uint32_t, bool, bool, bool, const ggml_sycl_runtime_context_desc *) {
    return (ggml_sycl_tenant_coverage) 77;
}

static ggml_sycl_late_check_result fake_late(struct ggml_sycl_load_txn txn, int32_t device, uint64_t bytes) {
    g_calls.n_late++;
    g_calls.late_id    = txn.id;
    g_calls.late_dev   = device;
    g_calls.late_bytes = bytes;
    return GGML_SYCL_LATE_CHECK_REFUSED;
}

static ggml_sycl_late_check_result wild_late(struct ggml_sycl_load_txn, int32_t, uint64_t) {
    return (ggml_sycl_late_check_result) 99;
}

// The probe procs: one that answers a status the test picks and writes n_layer the way the backend's proc does
// (never host_resident), and one that answers a value outside the enum.
static ggml_sycl_residency_probe_status g_probe_answer = GGML_SYCL_RESIDENCY_PROBE_NOT_ANSWERED;

static ggml_sycl_residency_probe_status fake_probe(ggml_backend_t,
                                                   struct ggml_sycl_model_token,
                                                   uint32_t n_ctx,
                                                   uint32_t,
                                                   uint32_t,
                                                   bool,
                                                   bool,
                                                   bool,
                                                   const ggml_sycl_runtime_context_desc *,
                                                   struct ggml_sycl_residency_probe * out) {
    g_calls.n_probe++;
    g_calls.probe_n_ctx = n_ctx;
    out->n_layer        = 12;
    return g_probe_answer;
}

static ggml_sycl_residency_probe_status wild_probe(ggml_backend_t,
                                                   struct ggml_sycl_model_token,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint32_t,
                                                   bool,
                                                   bool,
                                                   bool,
                                                   const ggml_sycl_runtime_context_desc *,
                                                   struct ggml_sycl_residency_probe * out) {
    out->n_layer = 12;
    return (ggml_sycl_residency_probe_status) 99;
}

static ggml_sycl_context_tenant_desc make_element(int32_t device, uint32_t cohort, uint32_t index, uint64_t bytes) {
    ggml_sycl_context_tenant_desc e = {};
    e.struct_size                   = sizeof(e);
    e.cohort                        = cohort;
    e.slot_index                    = index;
    e.device                        = device;
    e.slot_bytes                    = bytes;
    return e;
}

int main() {
    // (1) one header
    CHECK(sizeof(ggml_sycl_context_tenant_desc) == 24, "tenant element is %zu bytes",
          sizeof(ggml_sycl_context_tenant_desc));
    CHECK(GGML_SYCL_CONTEXT_COHORT_COMPUTE == 0 && GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST == 1,
          "the compute cohort ids are the published ones");

    // (2) null procs fail closed
    {
        llama_sycl_l4_procs            none;
        ggml_sycl_model_token          model = {};
        ggml_sycl_runtime_context_desc desc  = {};
        CHECK(!none.available(), "an empty table reports no L4");
        CHECK(llama_sycl_l4_publish(none, nullptr, model, 1, 1, 1, true, false, false, &desc) ==
                  GGML_SYCL_LIFECYCLE_UNSUPPORTED,
              "null publish proc must read UNSUPPORTED");
        CHECK(llama_sycl_l4_coverage(none, nullptr, 1, 1, 1, true, false, false, &desc) ==
                  GGML_SYCL_TENANT_COVERAGE_GROWTH,
              "null coverage proc must read GROWTH");
        CHECK(llama_sycl_l4_late_check(none, ggml_sycl_load_txn{ 5 }, 0, 123) == GGML_SYCL_LATE_CHECK_NOT_RECORDED,
              "null late-check proc must read NOT_RECORDED");

        llama_sycl_l4_procs wild;
        wild.coverage   = &wild_coverage;
        wild.late_check = &wild_late;
        CHECK(llama_sycl_l4_coverage(wild, nullptr, 1, 1, 1, true, false, false, &desc) ==
                  GGML_SYCL_TENANT_COVERAGE_GROWTH,
              "an out-of-range coverage answer must read GROWTH");
        CHECK(llama_sycl_l4_late_check(wild, ggml_sycl_load_txn{ 5 }, 0, 123) == GGML_SYCL_LATE_CHECK_NOT_RECORDED,
              "an out-of-range late-check answer must read NOT_RECORDED");
    }

    // (3) non-null procs are called through
    {
        llama_sycl_l4_procs procs;
        procs.publish    = &fake_publish;
        procs.coverage   = &fake_coverage;
        procs.late_check = &fake_late;
        CHECK(!procs.available(), "a table without the residency probe is not L4");
        procs.probe_residency = &fake_probe;
        CHECK(procs.available(), "a full table reports L4");
        ggml_sycl_model_token          model = {};
        ggml_sycl_runtime_context_desc desc  = {};
        g_calls                              = {};
        CHECK(llama_sycl_l4_publish(procs, nullptr, model, 4096, 512, 1, true, false, false, &desc) ==
                  GGML_SYCL_LIFECYCLE_PLAN_REJECTED,
              "the proc's own answer is returned");
        CHECK(g_calls.n_publish == 1 && g_calls.n_ubatch == 512 && g_calls.desc == &desc,
              "publish arguments forwarded");
        CHECK(llama_sycl_l4_coverage(procs, nullptr, 4096, 256, 1, true, false, false, &desc) ==
                  GGML_SYCL_TENANT_COVERAGE_COVERED,
              "the coverage answer is returned");
        CHECK(g_calls.n_coverage == 1 && g_calls.n_ubatch == 256, "coverage arguments forwarded");
        CHECK(llama_sycl_l4_late_check(procs, ggml_sycl_load_txn{ 41 }, 1, 987654) == GGML_SYCL_LATE_CHECK_REFUSED,
              "the late-check answer is returned");
        CHECK(g_calls.n_late == 1 && g_calls.late_id == 41 && g_calls.late_dev == 1 && g_calls.late_bytes == 987654,
              "late-check arguments forwarded");

        // The table is half-filled: availability needs all three.
        llama_sycl_l4_procs half;
        half.publish = &fake_publish;
        CHECK(!half.available(), "a half table is not L4");
    }

    // (11) the residency probe's door
    {
        using status = ggml_sycl_residency_probe_status;
        CHECK(GGML_SYCL_RESIDENCY_PROBE_NOT_ANSWERED == 0, "the zero value is the unanswered one");
        CHECK(GGML_SYCL_RESIDENCY_PROBE_OK == 1 && GGML_SYCL_RESIDENCY_PROBE_GEOMETRY_NOT_WIRED == 2 &&
                  GGML_SYCL_RESIDENCY_PROBE_INVALID == 3 && GGML_SYCL_RESIDENCY_PROBE_HEAD_SLOT_REFUSED == 4 &&
                  GGML_SYCL_RESIDENCY_PROBE_NO_PROMOTION_VIOLATED == 5 &&
                  GGML_SYCL_RESIDENCY_PROBE_N_LAYER_CAP_TOO_SMALL == 6 &&
                  GGML_SYCL_RESIDENCY_PROBE_FOREIGN_BACKEND == 7,
              "the status values are the published ones");
        CHECK((int) GGML_SYCL_RESIDENCY_PROBE_OK != (int) GGML_SYCL_LIFECYCLE_OK,
              "the probe's own enum: its OK is not the lifecycle's OK, which is 0");
        if (sizeof(void *) == 8) {
            CHECK(sizeof(ggml_sycl_residency_probe) == 24, "the probe out struct is %zu bytes",
                  sizeof(ggml_sycl_residency_probe));
        }

        ggml_sycl_model_token          model = {};
        ggml_sycl_runtime_context_desc desc  = {};
        uint8_t                        bytes[12];
        std::memset(bytes, 0xAB, sizeof(bytes));
        ggml_sycl_residency_probe out = {};
        out.struct_size               = sizeof(out);
        out.version                   = GGML_SYCL_RESIDENCY_PROBE_VERSION;
        out.n_layer_cap               = 12;
        out.host_resident             = bytes;
        auto untouched                = [&]() {
            for (uint8_t b : bytes) {
                if (b != 0xAB) {
                    return false;
                }
            }
            return true;
        };

        // a null proc reads NOT_ANSWERED, and leaves no stale n_layer to be read as an answer
        llama_sycl_l4_procs none;
        out.n_layer = 7;
        CHECK(llama_sycl_l4_probe_residency(none, nullptr, model, 4096, 512, 1, true, false, false, &desc, &out) ==
                  GGML_SYCL_RESIDENCY_PROBE_NOT_ANSWERED,
              "a null probe proc must read NOT_ANSWERED");
        CHECK(out.n_layer == 0 && untouched(), "a null proc leaves n_layer 0 and the bytes alone");

        // an answer outside the enum reads NOT_ANSWERED too
        llama_sycl_l4_procs wild;
        wild.probe_residency = &wild_probe;
        out.n_layer          = 7;
        CHECK(llama_sycl_l4_probe_residency(wild, nullptr, model, 4096, 512, 1, true, false, false, &desc, &out) ==
                  GGML_SYCL_RESIDENCY_PROBE_NOT_ANSWERED,
              "an out-of-range probe answer must read NOT_ANSWERED");
        CHECK(out.n_layer == 0 && untouched(), "an unknown answer is no answer: n_layer is cleared");

        // every published status is passed on as itself, with the caller's arguments; only OK is OK
        llama_sycl_l4_procs procs;
        procs.probe_residency = &fake_probe;
        for (int v = 0; v <= 7; ++v) {
            g_probe_answer = (status) v;
            g_calls        = {};
            out.n_layer    = 0;
            const status got =
                llama_sycl_l4_probe_residency(procs, nullptr, model, 4096, 512, 1, true, false, false, &desc, &out);
            CHECK((int) got == v, "status %d is passed on as itself, got %d", v, (int) got);
            CHECK(g_calls.n_probe == 1 && g_calls.probe_n_ctx == 4096, "the probe arguments are forwarded");
            CHECK((got == GGML_SYCL_RESIDENCY_PROBE_OK) == (v == 1), "only OK is OK (status %d)", v);
            CHECK(out.n_layer == 12, "the proc's n_layer is passed on");
        }
        CHECK(untouched(), "the door never writes the caller's host_resident bytes");

        // the table is L4 only with the probe: each of the four procs alone is not enough
        llama_sycl_l4_procs just_probe;
        just_probe.probe_residency = &fake_probe;
        CHECK(!just_probe.available(), "the probe alone is not L4");
        llama_sycl_l4_procs three;
        three.publish    = &fake_publish;
        three.coverage   = &fake_coverage;
        three.late_check = &fake_late;
        CHECK(!three.available(), "the three older procs without the probe are not L4");
    }

    // (4) the section builder
    {
        std::vector<llama_tenant_buft_caps> bufts;
        bufts.push_back({
  /*device*/ 1, /*host*/ false, { 300, 0, 50 }
        });
        bufts.push_back({ /*device*/ 0, /*host*/ false, { 700 } });
        bufts.push_back({
  /*device*/ 0, /*host*/ true, { 40, 10 }
        });
        bufts.push_back({ /*device*/ 1, /*host*/ true, { 60 } });
        std::vector<ggml_sycl_context_tenant_desc> out;
        std::string                                reason;
        CHECK(llama_tenant_section_from_caps(bufts, out, reason), "builder refused: %s", reason.c_str());
        // Expected order: (-1, HOST, 0)=60 (max of 40,60), (-1, HOST, 1)=10, (0, COMPUTE, 0)=700,
        // (1, COMPUTE, 0)=300, (1, COMPUTE, 2)=50. The zero cap at chunk 1 of device 1 is absent.
        CHECK(out.size() == 5, "section holds %zu elements", out.size());
        if (out.size() == 5) {
            CHECK(out[0].device == -1 && out[0].cohort == GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST &&
                      out[0].slot_index == 0 && out[0].slot_bytes == 60,
                  "element 0");
            CHECK(out[1].device == -1 && out[1].slot_index == 1 && out[1].slot_bytes == 10, "element 1");
            CHECK(out[2].device == 0 && out[2].cohort == GGML_SYCL_CONTEXT_COHORT_COMPUTE && out[2].slot_bytes == 700,
                  "element 2");
            CHECK(out[3].device == 1 && out[3].slot_index == 0 && out[3].slot_bytes == 300, "element 3");
            CHECK(out[4].device == 1 && out[4].slot_index == 2 && out[4].slot_bytes == 50, "element 4");
            for (const auto & e : out) {
                CHECK(e.struct_size == sizeof(e), "every element carries its own struct_size");
            }
        }

        // (5) the backend's demands merge by maximum
        std::vector<ggml_sycl_context_tenant_desc> extra;
        extra.push_back(make_element(0, GGML_SYCL_CONTEXT_COHORT_FATTN_MATERIALIZE, 0, 4096));
        extra.push_back(make_element(0, GGML_SYCL_CONTEXT_COHORT_COMPUTE, 0, 900));  // above the buft's 700
        extra.push_back(make_element(1, GGML_SYCL_CONTEXT_COHORT_COMPUTE, 0, 100));  // below the buft's 300
        llama_tenant_section_merge(out, extra);
        CHECK(out.size() == 6, "merged section holds %zu elements", out.size());
        if (out.size() == 6) {
            // (0, COMPUTE, 0)=900 stays at its place, (0, FATTN, 0) follows it.
            CHECK(out[2].device == 0 && out[2].cohort == GGML_SYCL_CONTEXT_COHORT_COMPUTE && out[2].slot_bytes == 900,
                  "the larger demand wins");
            CHECK(out[3].device == 0 && out[3].cohort == GGML_SYCL_CONTEXT_COHORT_FATTN_MATERIALIZE &&
                      out[3].slot_bytes == 4096,
                  "a new element is inserted in order");
            CHECK(out[4].device == 1 && out[4].slot_index == 0 && out[4].slot_bytes == 300, "the smaller demand loses");
        }
    }

    // (6) a device-tier buft with no device index refuses
    {
        std::vector<llama_tenant_buft_caps> bufts = {
            { -1, false, { 10 } }
        };
        std::vector<ggml_sycl_context_tenant_desc> out = { make_element(0, 0, 0, 1) };
        std::string                                reason;
        CHECK(!llama_tenant_section_from_caps(bufts, out, reason), "negative device on a device buft must refuse");
        CHECK(!reason.empty() && out.empty(), "a refusal names itself and leaves no elements");
    }

    // (7) the tenant key
    {
        std::vector<ggml_sycl_context_tenant_desc> a = { make_element(0, 0, 0, 100), make_element(1, 0, 0, 50) };
        std::vector<ggml_sycl_context_tenant_desc> b = a;
        CHECK(llama_tenant_key_digest(a) == llama_tenant_key_digest(b), "equal sections, equal keys");
        const uint64_t base = llama_tenant_key_digest(a);
        for (int field = 0; field < 4; ++field) {
            std::vector<ggml_sycl_context_tenant_desc> c = a;
            switch (field) {
                case 0:
                    c[1].device += 1;
                    break;
                case 1:
                    c[1].cohort += 1;
                    break;
                case 2:
                    c[1].slot_index += 1;
                    break;
                default:
                    c[1].slot_bytes += 1;
                    break;
            }
            CHECK(llama_tenant_key_digest(c) != base, "field %d must change the key", field);
        }
        std::vector<ggml_sycl_context_tenant_desc> shorter = { a[0] };
        CHECK(llama_tenant_key_digest(shorter) != base, "a shorter section must change the key");
        CHECK(llama_tenant_key_digest({}) != base, "an empty section has its own key");
        // struct_size is not a measured fact and does not enter the key.
        std::vector<ggml_sycl_context_tenant_desc> d = a;
        d[0].struct_size += 8;
        CHECK(llama_tenant_key_digest(d) == base, "struct_size is not part of the key");
    }

    // (8) the plan line
    {
        llama_tenant_plan_line_fields f;
        f.ctx_id               = 7;
        f.device               = 1;
        f.n_ubatch             = 512;
        f.compute_load         = 123456;
        f.compute_delta        = -4096;
        f.cap0                 = 99;
        f.n_measured           = 7;
        f.measure_ms           = 12.5;
        f.republish            = 0;
        f.covered              = 1;
        const std::string line = llama_tenant_plan_line(f);
        const std::string want =
            "[CONTEXT-PLAN] tenant plan: ctx=7 dev=1 n_ubatch=512 compute_load=123456 compute_delta=-4096 cap0=99 "
            "n_measured=7 measure_ms=12.500 republish=0 covered=1";
        CHECK(line == want, "plan line was '%s'", line.c_str());
    }

    // (9) the compute term over several chunks: the peak of each chunk over the measured graphs, summed. The
    // two graphs peak in different chunks, so the sum of the worst single graph (11) and the per-chunk sum (20)
    // differ, and only the second bounds both graphs run one after the other.
    {
        const std::vector<std::vector<size_t>> peaks = {
            { 10, 1  },
            { 1,  10 }
        };
        llama_tenant_buft_caps c;
        llama_tenant_caps_set_peaks(c, peaks);
        CHECK(c.chunk_bytes == std::vector<size_t>({ 10, 10 }), "per-chunk peaks hold %zu entries, first %zu",
              c.chunk_bytes.size(), c.chunk_bytes.empty() ? (size_t) 0 : c.chunk_bytes[0]);
        CHECK(c.total == 20, "the compute term is %zu, not the per-chunk sum 20", c.total);

        // a graph that touches fewer chunks leaves the others to the graphs that do
        const std::vector<std::vector<size_t>> ragged = {
            { 4 },
            { 1, 9, 2 }
        };
        llama_tenant_caps_set_peaks(c, ragged);
        CHECK(c.chunk_bytes == std::vector<size_t>({ 4, 9, 2 }) && c.total == 15, "ragged peaks give total %zu",
              c.total);

        // no measured graph, no term
        llama_tenant_caps_set_peaks(c, {});
        CHECK(c.chunk_bytes.empty() && c.total == 0, "no graphs gave total %zu", c.total);
    }

    // (10) the host-tier HOLD
    {
        // the host caps one rung measured: chunk c of the host buft is host slot c
        auto host_caps = [](std::vector<size_t> cap) {
            llama_tenant_buft_caps c;
            c.device = -1;
            c.host   = true;
            c.cap    = std::move(cap);
            return c;
        };
        auto dev_caps = [](int32_t device, std::vector<size_t> cap) {
            llama_tenant_buft_caps c;
            c.device = device;
            c.host   = false;
            c.cap    = std::move(cap);
            return c;
        };
        // a rung's section, built exactly as the transaction builds it
        auto rung_section = [&](const std::vector<llama_tenant_buft_caps> & caps) {
            std::vector<ggml_sycl_context_tenant_desc> s;
            std::string                                reason;
            CHECK(llama_tenant_section_from_caps(caps, s, reason), "the builder refused: %s", reason.c_str());
            return s;
        };
        auto host_bytes = [](const std::vector<ggml_sycl_context_tenant_desc> & s, uint32_t index) -> uint64_t {
            for (const auto & e : s) {
                if (e.cohort == GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST && e.slot_index == index) {
                    return e.slot_bytes;
                }
            }
            return 0;
        };
        // R_h over a rung set, the way the transaction folds it
        auto hold_of = [&](const std::vector<std::vector<llama_tenant_buft_caps>> & rungs) {
            llama_tenant_host_hold hold;
            for (const auto & caps : rungs) {
                llama_tenant_host_hold_fold(hold, rung_section(caps));
            }
            return hold;
        };

        // three rungs: 512 < 1024 < 2048. Slot 0 grows with the rung, slot 1 peaks at the middle one, and slot 2
        // exists only at the top rung.
        std::vector<std::vector<llama_tenant_buft_caps>> rungs = {
            { dev_caps(0, { 100 }), host_caps({ 152, 40 })     },
            { dev_caps(0, { 200 }), host_caps({ 304, 90 })     },
            { dev_caps(0, { 400 }), host_caps({ 608, 60, 25 }) },
        };
        llama_tenant_host_hold hold = hold_of(rungs);
        CHECK(hold.n_rungs == 3, "three rungs folded, got %u", hold.n_rungs);
        CHECK(hold.bytes == std::vector<uint64_t>({ 608, 90, 25 }), "R_h is the per-index maximum, got %zu entries",
              hold.bytes.size());

        // the section the publish carries: the first rung's own section raised to R_h at every host index; the
        // device slots stay the rung's own measurement
        std::vector<ggml_sycl_context_tenant_desc> section = rung_section(rungs[0]);
        llama_tenant_section_apply_host_hold(section, hold);
        CHECK(host_bytes(section, 0) == 608 && host_bytes(section, 1) == 90 && host_bytes(section, 2) == 25,
              "slots are %llu %llu %llu", (unsigned long long) host_bytes(section, 0),
              (unsigned long long) host_bytes(section, 1), (unsigned long long) host_bytes(section, 2));
        size_t n_dev = 0;
        for (const auto & e : section) {
            if (e.cohort == GGML_SYCL_CONTEXT_COHORT_COMPUTE) {
                CHECK(e.slot_bytes == 100, "a device slot is its own rung's measure, got %llu",
                      (unsigned long long) e.slot_bytes);
                n_dev++;
            }
        }
        CHECK(n_dev == 1, "the device slots are not the hold's to change");
        for (size_t i = 1; i < section.size(); ++i) {
            CHECK(!llama_tenant_element_less(section[i], section[i - 1]), "the section stays ordered");
        }

        // one source: the key a rung's section gets does not depend on which rung it was built at, once the hold is
        // applied to the host slots (the same host slots, the device slot the rung's own)
        std::vector<ggml_sycl_context_tenant_desc> top = rung_section(rungs[2]);
        llama_tenant_section_apply_host_hold(top, hold);
        std::vector<ggml_sycl_context_tenant_desc> mid = rung_section(rungs[1]);
        llama_tenant_section_apply_host_hold(mid, hold);
        for (uint32_t i = 0; i < 3; ++i) {
            CHECK(host_bytes(top, i) == host_bytes(mid, i) && host_bytes(mid, i) == host_bytes(section, i),
                  "host slot %u differs between rungs under the hold", i);
        }

        // perturb one rung's measurement: the slot it dominates changes, the others do not
        auto perturbed             = rungs;
        perturbed[1].back().cap[1] = 91;  // slot 1, dominated by the middle rung
        llama_tenant_host_hold p1  = hold_of(perturbed);
        CHECK(p1.bytes == std::vector<uint64_t>({ 608, 91, 25 }), "slot 1 follows the middle rung, got %llu",
              (unsigned long long) (p1.bytes.size() > 1 ? p1.bytes[1] : 0));
        std::vector<ggml_sycl_context_tenant_desc> s1 = rung_section(rungs[0]);
        llama_tenant_section_apply_host_hold(s1, p1);
        CHECK(host_bytes(s1, 1) == 91 && host_bytes(s1, 0) == 608 && host_bytes(s1, 2) == 25,
              "the carried slot changed with the measurement");

        perturbed                  = rungs;
        perturbed[0].back().cap[1] = 1000;  // a rung that now dominates slot 1
        CHECK(hold_of(perturbed).bytes[1] == 1000, "a rung's perturbation reaches the slot");
        perturbed                  = rungs;
        perturbed[2].back().cap[0] = 50;  // the top rung no longer dominates slot 0: the middle one does
        CHECK(hold_of(perturbed).bytes[0] == 304, "R_h is a maximum, not the last rung's value");

        // a live need above the hold at an index is the live measurement, left for the backend to refuse
        std::vector<ggml_sycl_context_tenant_desc> live = rung_section({ dev_caps(0, { 1 }), host_caps({ 700 }) });
        llama_tenant_section_apply_host_hold(live, hold);
        CHECK(host_bytes(live, 0) == 700, "a need above R_h is not clipped to it, got %llu",
              (unsigned long long) host_bytes(live, 0));
        CHECK(host_bytes(live, 1) == 90 && host_bytes(live, 2) == 25, "the other indices are held at R_h");

        // no host tier: nothing to hold, and an empty hold adds nothing
        std::vector<ggml_sycl_context_tenant_desc> devonly = rung_section({ dev_caps(0, { 100 }) });
        llama_tenant_host_hold                     none    = hold_of({ { dev_caps(0, { 100 }) } });
        CHECK(none.bytes.empty() && none.n_rungs == 1, "a device-only rung holds no host slot");
        llama_tenant_section_apply_host_hold(devonly, none);
        CHECK(devonly.size() == 1, "an empty hold adds no element");

        // a zero host slot is no element and no hold
        llama_tenant_host_hold z = hold_of({ { host_caps({ 0, 5 }) } });
        CHECK(z.bytes == std::vector<uint64_t>({ 0, 5 }), "a zero cap holds nothing");
        std::vector<ggml_sycl_context_tenant_desc> zs = rung_section({ host_caps({ 0, 5 }) });
        llama_tenant_section_apply_host_hold(zs, z);
        CHECK(zs.size() == 1 && zs[0].slot_index == 1, "a zero hold index makes no element");

        // the line the replay reads
        const std::string line = llama_tenant_host_hold_line(hold);
        CHECK(line == "[CONTEXT-PLAN] host hold: rungs=3 slots=3 bytes=608,90,25", "hold line was '%s'", line.c_str());
    }

    if (n_failed != 0) {
        fprintf(stderr, "%d check(s) failed\n", n_failed);
        return 1;
    }
    printf("ok\n");
    return 0;
}
