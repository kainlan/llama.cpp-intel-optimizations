#include "sycl-timeline.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-sycl-timeline: %s\n", message);
        std::abort();
    }
}

int main() {
    using namespace ggml_sycl;

    require(!sycl_timeline_enabled_from_env(nullptr), "null env must disable timeline");
    require(!sycl_timeline_enabled_from_env(""), "empty env must disable timeline");
    require(!sycl_timeline_enabled_from_env("0"), "zero env must disable timeline");
    require(!sycl_timeline_enabled_from_env("off"), "off env must disable timeline");

    require(sycl_timeline_enabled_from_env("summary"), "summary env must enable timeline");
    require(sycl_timeline_enabled_from_env("timeline"), "timeline env must enable timeline");
    require(sycl_timeline_enabled_from_env("timeline+events"), "timeline+events env must enable timeline");

    const sycl_timeline_config cfg =
        sycl_timeline_config_from_values("timeline+events", "/tmp/unit-trace", "3", "4", "99");
    require(cfg.enabled, "timeline+events config must enable timeline");
    require(cfg.mode == sycl_timeline_mode::TIMELINE_EVENTS, "mode parse failed");
    require(cfg.output_path == "/tmp/unit-trace", "output path parse failed");
    require(cfg.token_start == 3, "token_start parse failed");
    require(cfg.token_count == 4, "token_count parse failed");
    require(cfg.max_events == 99, "max_events parse failed");

    sycl_timeline_reset_for_tests();
    sycl_timeline_set_config_for_tests(cfg);
    require(sycl_timeline_enabled(), "test override config must enable timeline");
    sycl_timeline_reset_for_tests();

    return 0;
}
