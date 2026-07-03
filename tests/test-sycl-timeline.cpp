#include "sycl-timeline.hpp"

#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
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

static std::string make_temp_trace_dir() {
    std::array<char, sizeof("/tmp/test-sycl-timeline-XXXXXX")> tmpl = { "/tmp/test-sycl-timeline-XXXXXX" };
    char *                                                     dir  = mkdtemp(tmpl.data());
    require(dir != nullptr, "temporary trace directory must be created");
    return dir;
}

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    require(static_cast<bool>(in), "trace file must be readable");
    return { std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>() };
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

    unsetenv("GGML_SYCL_TIMELINE");
    unsetenv("GGML_SYCL_TIMELINE_OUTPUT");
    unsetenv("GGML_SYCL_TIMELINE_TOKEN_START");
    unsetenv("GGML_SYCL_TIMELINE_TOKEN_COUNT");
    unsetenv("GGML_SYCL_TIMELINE_MAX_EVENTS");
    sycl_timeline_config env_cfg = sycl_timeline_config_from_env();
    require(!env_cfg.enabled, "unset timeline env must disable timeline");
    require(env_cfg.output_path.empty(), "unset output env must leave output path empty");

    setenv("GGML_SYCL_TIMELINE", "timeline+events", 1);
    setenv("GGML_SYCL_TIMELINE_OUTPUT", "/tmp/env-trace.json", 1);
    setenv("GGML_SYCL_TIMELINE_TOKEN_START", "7", 1);
    setenv("GGML_SYCL_TIMELINE_TOKEN_COUNT", "8", 1);
    setenv("GGML_SYCL_TIMELINE_MAX_EVENTS", "123", 1);
    env_cfg = sycl_timeline_config_from_env();
    require(env_cfg.enabled, "env config must enable timeline");
    require(env_cfg.mode == sycl_timeline_mode::TIMELINE_EVENTS, "env mode parse failed");
    require(env_cfg.output_path == "/tmp/env-trace.json", "env output path parse failed");
    require(env_cfg.token_start == 7, "env token_start parse failed");
    require(env_cfg.token_count == 8, "env token_count parse failed");
    require(env_cfg.max_events == 123, "env max_events parse failed");

    unsetenv("GGML_SYCL_TIMELINE_OUTPUT");
    unsetenv("GGML_SYCL_TIMELINE_TOKEN_START");
    unsetenv("GGML_SYCL_TIMELINE_TOKEN_COUNT");
    unsetenv("GGML_SYCL_TIMELINE_MAX_EVENTS");
    env_cfg = sycl_timeline_config_from_env();
    require(env_cfg.output_path.empty(), "unset env output path must read as empty");
    require(env_cfg.token_start == 0, "unset env token_start must use default");
    require(env_cfg.token_count == 0, "unset env token_count must use default");
    require(env_cfg.max_events == 200000, "unset env max_events must use default");
    unsetenv("GGML_SYCL_TIMELINE");

    sycl_timeline_reset_for_tests();
    sycl_timeline_set_config_for_tests(cfg);
    require(sycl_timeline_enabled(), "test override config must enable timeline");

    {
        GGML_SYCL_TIMELINE_SCOPE("unit", "enabled-span", "case=enabled");
    }
    const std::string json = sycl_timeline_format_json_for_tests();
    require(contains(json, "\"traceEvents\""), "JSON must contain traceEvents");
    require(contains(json, "\"ph\":\"X\""), "JSON must contain complete events");
    require(contains(json, "\"pid\":"), "JSON must contain pid");
    require(contains(json, "\"tid\":"), "JSON must contain tid");
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
    // clang-format off
    { GGML_SYCL_TIMELINE_SCOPE("unit", "same-line-a", ""); GGML_SYCL_TIMELINE_SCOPE("unit", "same-line-b", ""); }
    // clang-format on
    const std::string same_line_json = sycl_timeline_format_json_for_tests();
    require(contains(same_line_json, "\"name\":\"same-line-a\""), "first same-line-safe scope must be recorded");
    require(contains(same_line_json, "\"name\":\"same-line-b\""), "second same-line-safe scope must be recorded");

    const std::string    trace_dir  = make_temp_trace_dir();
    const std::string    trace_path = trace_dir + "/trace.json";
    sycl_timeline_config file_cfg   = cfg;
    file_cfg.output_path            = trace_path;
    sycl_timeline_reset_for_tests();
    sycl_timeline_set_config_for_tests(file_cfg);
    {
        GGML_SYCL_TIMELINE_SCOPE("unit", "flush-span", "case=flush");
    }
    sycl_timeline_flush("unit-test");
    const std::string file_json = read_file(trace_path);
    require(contains(file_json, "\"traceEvents\""), "flushed trace file must contain traceEvents");
    require(contains(file_json, "\"name\":\"flush-span\""), "flushed trace file must contain span name");
    require(sycl_timeline_format_json_for_tests() == "{\"traceEvents\":[]}",
            "successful flush must consume buffered events");
    std::remove(trace_path.c_str());
    rmdir(trace_dir.c_str());

    sycl_timeline_config pathless_cfg = cfg;
    pathless_cfg.output_path.clear();
    sycl_timeline_reset_for_tests();
    sycl_timeline_set_config_for_tests(pathless_cfg);
    {
        GGML_SYCL_TIMELINE_SCOPE("unit", "pathless-flush-span", "case=pathless");
    }
    sycl_timeline_flush("unit-test-pathless");
    require(contains(sycl_timeline_format_json_for_tests(), "\"name\":\"pathless-flush-span\""),
            "pathless flush must be a no-op");

    sycl_timeline_reset_for_tests();

    return 0;
}
