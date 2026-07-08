#include "ggml-sycl/ggml-sycl-test.hpp"

#include <cstdio>
#include <cstring>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static ggml_sycl::test_moe_glu_q8_artifact_input base_input() {
    ggml_sycl::test_moe_glu_q8_artifact_input in{};
    in.fused_candidate    = true;
    in.kernel_event_set   = true;
    in.fused_store_ok     = true;
    in.glu_row_contiguous = true;
    in.publish_ok         = true;
    in.handle_valid       = true;
    in.layer_match        = true;
    in.dims_match         = true;
    in.layout_soa         = true;
    in.ready_event_set    = true;
    return in;
}

static int test_fused_store_accepts_and_saves_publish_launch() {
    ggml_sycl::test_moe_glu_q8_counters_reset();
    auto out = ggml_sycl::test_moe_glu_q8_artifact_policy(base_input());
    CHECK(out.accepted, "fused GLU/Q8 store must accept");
    CHECK(std::strcmp(out.action, "fused-store") == 0, "action must be fused-store");
    CHECK(out.cached_down_allowed, "fresh fused artifact must allow cached down");
    CHECK(out.completion_event_set, "fused artifact must expose completion event");
    CHECK(out.expected_saved_launches == 1, "fused artifact must estimate one saved publish launch");

    auto counters = ggml_sycl::test_moe_glu_q8_counters_snapshot();
    CHECK(counters.candidates == 1, "candidate counter must increment");
    CHECK(counters.fused_store == 1, "fused-store counter must increment");
    CHECK(counters.publish == 0, "publish counter must stay zero");
    CHECK(counters.invalidated == 0, "invalidate counter must stay zero");
    CHECK(counters.cached_allow == 1, "cached-allow counter must increment");
    CHECK(counters.cached_reject == 0, "cached-reject counter must stay zero");
    return 0;
}

static int test_publish_fallback_accepts_only_when_row_contiguous() {
    auto in           = base_input();
    in.fused_store_ok = false;
    auto out          = ggml_sycl::test_moe_glu_q8_artifact_policy(in);
    CHECK(out.accepted, "publish fallback must accept when row-contiguous publish succeeds");
    CHECK(std::strcmp(out.action, "publish") == 0, "action must be publish");
    CHECK(out.expected_saved_launches == 0, "publish fallback saves no launch");

    in.glu_row_contiguous = false;
    auto bad              = ggml_sycl::test_moe_glu_q8_artifact_policy(in);
    CHECK(!bad.accepted, "non-contiguous GLU rows must reject publish fallback");
    CHECK(bad.invalidate_artifact, "non-contiguous publish failure must invalidate artifact");
    CHECK(std::strcmp(bad.reason, "shape") == 0, "non-contiguous reject reason must be shape");
    return 0;
}

static int test_failed_publish_invalidates_stale_artifact() {
    auto in           = base_input();
    in.fused_store_ok = false;
    in.publish_ok     = false;
    auto out          = ggml_sycl::test_moe_glu_q8_artifact_policy(in);
    CHECK(!out.accepted, "failed publish must reject");
    CHECK(out.invalidate_artifact, "failed publish must invalidate previous artifact");
    CHECK(!out.cached_down_allowed, "failed publish must not allow cached down");
    CHECK(std::strcmp(out.reason, "publish") == 0, "failed publish reject reason must be publish");
    return 0;
}

static int test_counters_cover_publish_invalidate_and_cached_results() {
    ggml_sycl::test_moe_glu_q8_counters_reset();

    auto fused = base_input();
    (void) ggml_sycl::test_moe_glu_q8_artifact_policy(fused);

    auto publish           = base_input();
    publish.fused_store_ok = false;
    (void) ggml_sycl::test_moe_glu_q8_artifact_policy(publish);

    auto stale           = base_input();
    stale.fused_store_ok = false;
    stale.publish_ok     = false;
    (void) ggml_sycl::test_moe_glu_q8_artifact_policy(stale);

    auto counters = ggml_sycl::test_moe_glu_q8_counters_snapshot();
    CHECK(counters.candidates == 3, "candidate counter must count every policy check");
    CHECK(counters.fused_store == 1, "fused-store counter must count accepted fused stores");
    CHECK(counters.publish == 1, "publish counter must count accepted publish fallbacks");
    CHECK(counters.invalidated == 1, "invalidated counter must count stale publish failures");
    CHECK(counters.cached_allow == 2, "cached-allow counter must count fused and publish acceptances");
    CHECK(counters.cached_reject == 1, "cached-reject counter must count rejected stale artifacts");
    return 0;
}

static int test_cached_down_requires_matching_artifact_identity() {
    auto in         = base_input();
    in.handle_valid = false;
    auto no_handle  = ggml_sycl::test_moe_glu_q8_artifact_policy(in);
    CHECK(!no_handle.cached_down_allowed, "invalid handle must block cached down");
    CHECK(std::strcmp(no_handle.reason, "artifact") == 0, "invalid handle reason must be artifact");

    in                 = base_input();
    in.ready_event_set = false;
    auto no_event      = ggml_sycl::test_moe_glu_q8_artifact_policy(in);
    CHECK(!no_event.cached_down_allowed, "missing ready event must block cached down");
    CHECK(std::strcmp(no_event.reason, "event") == 0, "missing event reason must be event");

    in              = base_input();
    in.layout_soa   = false;
    auto bad_layout = ggml_sycl::test_moe_glu_q8_artifact_policy(in);
    CHECK(!bad_layout.cached_down_allowed, "non-SOA artifact must block cached down");
    CHECK(std::strcmp(bad_layout.reason, "layout") == 0, "bad layout reason must be layout");
    return 0;
}

int main() {
    if (test_fused_store_accepts_and_saves_publish_launch() != 0) {
        return 1;
    }
    if (test_publish_fallback_accepts_only_when_row_contiguous() != 0) {
        return 1;
    }
    if (test_failed_publish_invalidates_stale_artifact() != 0) {
        return 1;
    }
    if (test_counters_cover_publish_invalidate_and_cached_results() != 0) {
        return 1;
    }
    if (test_cached_down_requires_matching_artifact_identity() != 0) {
        return 1;
    }
    std::puts("PASS: MoE GLU/Q8 artifact policy");
    return 0;
}
