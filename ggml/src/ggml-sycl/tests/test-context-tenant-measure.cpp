#include "../context-tenant-measure.hpp"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>

// Release builds define NDEBUG, which would compile assert() away and let this
// test pass vacuously, so every check is an explicit one that always runs.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

using ggml_sycl::context_demand_accum;
using ggml_sycl::context_measure_phase;
using ggml_sycl::context_measure_view;
using ggml_sycl::context_measure_visitor;

static int g_visits = 0;

static void visitor_demand(const ggml_tensor *, const context_measure_view & view, context_demand_accum & acc) {
    g_visits++;
    acc.demand(view, GGML_SYCL_CONTEXT_COHORT_FATTN_MATERIALIZE, 0, 4096);
}

static void visitor_null_backend(const ggml_tensor *          node,
                                 const context_measure_view & view,
                                 context_demand_accum &       acc) {
    g_visits++;
    if (view.sched_backend(view.sched_ctx, node) == nullptr) {
        acc.fail("leaf has no backend assignment");
    }
}

static ggml_backend_t no_backend(void *, const ggml_tensor *) {
    return nullptr;
}

int main() {
    // The element layout is a cross-library ABI: a moved field is a silent
    // stride change for a reader built apart from the publisher.
    CHECK(sizeof(ggml_sycl_context_tenant_desc) == 24, "tenant element must be 24 bytes");
    CHECK(offsetof(ggml_sycl_context_tenant_desc, struct_size) == 0, "struct_size offset");
    CHECK(offsetof(ggml_sycl_context_tenant_desc, cohort) == 4, "cohort offset");
    CHECK(offsetof(ggml_sycl_context_tenant_desc, slot_index) == 8, "slot_index offset");
    CHECK(offsetof(ggml_sycl_context_tenant_desc, device) == 12, "device offset");
    CHECK(offsetof(ggml_sycl_context_tenant_desc, slot_bytes) == 16, "slot_bytes offset");

    // The table covers every cohort id, in id order, with unique names, and
    // answers nothing for an id it does not know.
    std::set<std::string> names;
    for (uint32_t id = 0; id < GGML_SYCL_CONTEXT_COHORT_COUNT; id++) {
        const ggml_sycl_context_cohort_info * info = ggml_sycl_context_cohort_lookup(id);
        CHECK(info != nullptr, "every cohort id has a table row");
        CHECK(info->id == id, "the table row of an id carries that id");
        CHECK(info->name != nullptr && info->name[0] != '\0', "every cohort has a name");
        CHECK(std::strncmp(info->name, "context-", 8) == 0, "zhcn's cohorts are context-* names");
        CHECK(names.insert(info->name).second, "cohort names are unique");
        CHECK(info->scope == GGML_SYCL_CONTEXT_COHORT_SCOPE_CONTEXT, "every registered cohort is CONTEXT scope");
    }
    CHECK(ggml_sycl_context_cohort_lookup(GGML_SYCL_CONTEXT_COHORT_COUNT) == nullptr, "an unknown id has no row");
    CHECK(ggml_sycl_context_cohort_lookup(0xffffffffu) == nullptr, "a wild id has no row");
    CHECK(std::strcmp(ggml_sycl_context_cohort_lookup(GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST)->name,
                      "context-compute-host") == 0,
          "the host cohort's name");

    // The tier decides the element's device, and nothing else does.
    CHECK(ggml_sycl_context_cohort_element_device(GGML_SYCL_CONTEXT_COHORT_COMPUTE, 1) == 1,
          "device tier keeps the device");
    CHECK(ggml_sycl_context_cohort_element_device(GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST, 1) == -1,
          "host tier is device -1");

    // Contract (b).
    CHECK(ggml_sycl_context_claim_fits(100, 100), "a claim equal to the slot fits");
    CHECK(!ggml_sycl_context_claim_fits(101, 100), "a claim over the slot does not fit");

    // The accumulator keeps the maximum per (device, cohort, index) across
    // graphs and orders the section.
    {
        context_demand_accum acc;
        context_measure_view view;
        view.device = 1;
        acc.demand(view, GGML_SYCL_CONTEXT_COHORT_COMPUTE, 1, 100);
        acc.demand(view, GGML_SYCL_CONTEXT_COHORT_COMPUTE, 0, 50);
        acc.demand(view, GGML_SYCL_CONTEXT_COHORT_COMPUTE, 1, 300);  // a later graph's larger demand
        acc.demand(view, GGML_SYCL_CONTEXT_COHORT_COMPUTE, 1, 70);   // a smaller one after it: no change
        acc.demand(view, GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST, 0, 9);
        view.device = 0;
        acc.demand(view, GGML_SYCL_CONTEXT_COHORT_COMPUTE, 0, 7);
        CHECK(acc.ok(), "valid demands record no error");

        const auto t = acc.tenants();
        CHECK(t.size() == 4, "four distinct slots");
        // order: (device, cohort, index): -1 host first, then device 0, then device 1
        CHECK(t[0].device == -1 && t[0].cohort == GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST && t[0].slot_bytes == 9,
              "the host slot sorts first and carries device -1");
        CHECK(t[1].device == 0 && t[1].slot_index == 0 && t[1].slot_bytes == 7, "device 0 slot");
        CHECK(t[2].device == 1 && t[2].slot_index == 0 && t[2].slot_bytes == 50, "device 1 index 0");
        CHECK(t[3].device == 1 && t[3].slot_index == 1 && t[3].slot_bytes == 300,
              "device 1 index 1 is the maximum over the visits");
        for (const auto & d : t) {
            CHECK(d.struct_size == sizeof(ggml_sycl_context_tenant_desc), "each element carries its stride");
        }
    }

    // An error is named, the first one wins, and the section is empty.
    {
        context_demand_accum acc;
        context_measure_view view;
        acc.demand(view, GGML_SYCL_CONTEXT_COHORT_COMPUTE, 0, 1);
        acc.demand(view, 999, 0, 1);
        CHECK(!acc.ok(), "an unknown cohort is an error");
        CHECK(acc.error().find("999") != std::string::npos, "the error names the id");
        acc.fail("second error");
        CHECK(acc.error().find("999") != std::string::npos, "the first error wins");
        CHECK(acc.tenants().empty(), "an error carries no tenants");
    }

    // The walker runs each visitor of the table it is given, and stops at the
    // null terminator.
    {
        const context_measure_visitor table[] = {
            { "demand",           visitor_demand       },
            { "backend",          visitor_null_backend },
            { nullptr,            nullptr              },
            { "after-terminator", visitor_demand       },
        };
        context_demand_accum acc;
        context_measure_view view;
        view.sched_backend = no_backend;
        g_visits           = 0;
        ggml_tensor node;
        std::memset(&node, 0, sizeof(node));
        ggml_sycl::context_measure_visit(table, &node, view, acc);
        CHECK(g_visits == 2, "both visitors before the terminator ran, and none after it");
        CHECK(!acc.ok() && acc.error() == "leaf has no backend assignment",
              "a null backend assignment is a named error");

        // The registered table is terminated.
        const context_measure_visitor * t = ggml_sycl::context_measure_visitors();
        CHECK(t != nullptr, "the visitor table exists");
        size_t n = 0;
        while (t[n].fn != nullptr && n < 64) {
            n++;
        }
        CHECK(n < 64, "the visitor table is terminated");
    }

    std::printf("PASS\n");
    return 0;
}
