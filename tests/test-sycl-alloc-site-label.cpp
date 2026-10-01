// H0: the construction-site label on alloc_request, alloc_intent and
// offload_buffer_request.
//
// Each type carries `site_file = __builtin_FILE(); site_line = __builtin_LINE();` as default member
// initialisers, so the raw-exit trace and the zone chokepoint can name the row that asked for an
// allocation. The labels are only as good as where the compiler evaluates those initialisers, and
// that is a property of the toolchain, so this test asserts it on the one that builds the backend
// (icpx) rather than assuming it:
//
//   * `T x{}`, a designated initialiser and a helper's defaulted arguments report the line of the
//     construction;
//   * a braceless `T x;` -- as a local, at namespace scope and as a class member -- is REPORTED, not
//     asserted: on icpx it names the CLASS DEFINITION, which is why every production declaration of
//     these types is written `T x{}`, but a toolchain that fixes that would not make anything wrong;
//   * a copy reports the source's site, so a copy that reaches an allocator for a different purpose
//     must carry its own cohort;
//   * an aggregate that holds one of the types reports the aggregate's construction when it is
//     brace-initialised.
//
// Each case puts its construction and its expected line on one physical line, so a failure names
// the line that is wrong. The allocator-side wrappers (unified_alloc reading req.site_*,
// acquire_offload_buffer copying its caller's site into areq) need a device and are not reached here.
//
// CPU-only by construction: nothing allocates, touches a device or a queue, or reaches the cache.

#include "common.hpp"

#include <cstdio>
#include <cstring>

namespace {

int g_failures = 0;

const char * base_name(const char * file) {
    const char * slash = std::strrchr(file, '/');
    return slash ? slash + 1 : file;
}

void expect_site(const char * what, const char * file, int line, const char * want_file, int want_line) {
    if (std::strcmp(base_name(file), want_file) != 0 || line != want_line) {
        std::fprintf(stderr, "FAIL [%s]: reports %s:%d, expected %s:%d\n", what, base_name(file), line, want_file,
                     want_line);
        g_failures++;
    } else {
        std::printf("ok   [%s]: %s:%d\n", what, base_name(file), line);
    }
}

// clang-format off
// Each construction shares a physical line with its expected `__LINE__`; a formatter that splits
// the line would move the construction off it, so the cases are not reformatted.

// A braceless declaration's label depends on the toolchain, so it is reported, never asserted.
void note_braceless(const char * what, const char * file, int line) {
    std::printf("note [%s]: reports %s:%d\n", what, base_name(file), line);
}

constexpr const char * k_this_file       = "test-sycl-alloc-site-label.cpp";

struct request_holder {
    ggml_sycl::alloc_request request;
};

// A braceless declaration at namespace scope, and one with braces.
ggml_sycl::alloc_request g_namespace_braceless;
ggml_sycl::alloc_request g_namespace_braced{};
const int                g_namespace_braced_line = __LINE__ - 1;

}  // namespace

int main() {
    using ggml_sycl::alloc_intent;
    using ggml_sycl::alloc_request;
    using ggml_sycl::offload_buffer_request;

    // T x{}: the construction line, for each type.
    alloc_request req_braced{}; const int req_braced_line = __LINE__;
    expect_site("alloc_request x{}", req_braced.site_file, req_braced.site_line, k_this_file, req_braced_line);

    alloc_intent intent_braced{}; const int intent_braced_line = __LINE__;
    expect_site("alloc_intent x{}", intent_braced.site_file, intent_braced.site_line, k_this_file,
                intent_braced_line);

    offload_buffer_request offload_braced{}; const int offload_braced_line = __LINE__;
    expect_site("offload_buffer_request x{}", offload_braced.site_file, offload_braced.site_line, k_this_file,
                offload_braced_line);

    // T x;: reported, not asserted. On the toolchain this was written against it names the class
    // definition, which is why production declarations are written `T x{}`; a toolchain that names
    // the declaration instead would make the braces unnecessary but not wrong, so this never fails.
    alloc_request req_braceless;
    note_braceless("alloc_request x;", req_braceless.site_file, req_braceless.site_line);
    alloc_intent intent_braceless;
    note_braceless("alloc_intent x;", intent_braceless.site_file, intent_braceless.site_line);
    offload_buffer_request offload_braceless;
    note_braceless("offload_buffer_request x;", offload_braceless.site_file, offload_braceless.site_line);
    note_braceless("namespace-scope alloc_request;", g_namespace_braceless.site_file, g_namespace_braceless.site_line);
    request_holder holder_braceless;
    note_braceless("class member, holder;", holder_braceless.request.site_file, holder_braceless.request.site_line);

    expect_site("namespace-scope alloc_request x{}", g_namespace_braced.site_file, g_namespace_braced.site_line,
                k_this_file, g_namespace_braced_line);

    // A class member: an aggregate that is brace-initialised reports its own construction.
    request_holder holder_braced{}; const int holder_braced_line = __LINE__;
    expect_site("class member, holder{}", holder_braced.request.site_file, holder_braced.request.site_line,
                k_this_file, holder_braced_line);

    // The nested member of a brace-initialised request is never what an allocator reads (it reads the
    // request's own site); where it points is reported for the record.
    note_braceless("alloc_request{}.intent (nested; never read by an allocator)", req_braced.intent.site_file,
                   req_braced.intent.site_line);

    // A designated initialiser reports its own line.
#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wc++20-designator"
#endif
    alloc_request req_designated{ .device = 1 }; const int req_designated_line = __LINE__;
#if defined(__clang__)
#    pragma clang diagnostic pop
#endif
    expect_site("alloc_request{ .device = 1 }", req_designated.site_file, req_designated.site_line, k_this_file,
                req_designated_line);

    // A copy carries the source's site.
    alloc_request req_copy = req_braced;
    expect_site("copy of alloc_request", req_copy.site_file, req_copy.site_line, k_this_file, req_braced_line);

    // A helper-built intent reports its caller, through the helper's defaulted arguments.
    const alloc_intent helper_device = ggml_sycl_transient_device_intent("h0_device"); const int helper_device_line = __LINE__;
    expect_site("ggml_sycl_transient_device_intent()", helper_device.site_file, helper_device.site_line, k_this_file,
                helper_device_line);
    const alloc_intent helper_host = ggml_sycl_transient_host_pinned_intent("h0_host"); const int helper_host_line = __LINE__;
    expect_site("ggml_sycl_transient_host_pinned_intent()", helper_host.site_file, helper_host.site_line, k_this_file,
                helper_host_line);

    // The two new classification fields exist and default to "not declared".
    if (req_braced.intent.constraints.cascade_step || req_braced.intent.constraints.unconverted_ticket != nullptr) {
        std::fprintf(stderr, "FAIL [constraints]: cascade_step / unconverted_ticket do not default off\n");
        g_failures++;
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "%d site-label case(s) failed\n", g_failures);
        return 1;
    }
    std::printf("PASS: site labels\n");
    return 0;
}

// clang-format on
