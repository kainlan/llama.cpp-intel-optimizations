#include "sycl-timeline.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static bool contains(const std::string & haystack, const char * needle) {
    return haystack.find(needle) != std::string::npos;
}

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

    {
        GGML_SYCL_TIMELINE_SCOPE("unit", "enabled-span", "case=enabled");
    }
    const std::string json = sycl_timeline_format_json_for_tests();
    require(contains(json, "\"traceEvents\""), "JSON must contain traceEvents");
    require(contains(json, "\"ph\":\"X\""), "JSON must contain complete events");
    require(contains(json, "\"cat\":\"unit\""), "JSON must contain category");
    require(contains(json, "\"name\":\"enabled-span\""), "JSON must contain span name");
    require(contains(json, "\"file\":\""), "JSON must contain file");
    require(contains(json, "\"line\":"), "JSON must contain line");
    require(contains(json, "\"function\":\"main\""), "JSON must contain function");
    require(contains(json, "\"metadata\":\"case=enabled\""), "JSON must contain metadata");

    sycl_timeline_config disabled_cfg = cfg;
    disabled_cfg.enabled              = false;
    sycl_timeline_reset_for_tests();
    sycl_timeline_set_config_for_tests(disabled_cfg);
    {
        GGML_SYCL_TIMELINE_SCOPE("unit", "disabled-span", "case=disabled");
    }
    require(sycl_timeline_format_json_for_tests() == "{\"traceEvents\":[]}",
            "disabled timeline must format as empty traceEvents");

    sycl_timeline_reset_for_tests();
    sycl_timeline_set_config_for_tests(cfg);
    {
        GGML_SYCL_TIMELINE_SCOPE("unit", "twin-a", "");
        GGML_SYCL_TIMELINE_SCOPE("unit", "twin-b", "");
    }
    const std::string twin_json = sycl_timeline_format_json_for_tests();
    require(contains(twin_json, "\"name\":\"twin-a\""), "first same-line-safe scope must be recorded");
    require(contains(twin_json, "\"name\":\"twin-b\""), "second same-line-safe scope must be recorded");

    sycl_timeline_reset_for_tests();

    return 0;
}
