// The queue-profiling resolver's pure parts (queue-profiling.hpp, common.cpp): ggml_sycl_queue_profiling_decide() and
// ggml_sycl_queue_profiling_requested_diagnostic(). No SYCL device is opened and no model is loaded: the test calls two
// functions of libggml-sycl that take plain values and an injected getenv, and the shared parse helpers of the header.
// The memoised wrapper (ggml_sycl_queue_profiling_enabled) is deliberately not called: it reads this process's
// environment once, at the first queue.
//
// The states under test: GGML_SYCL_QUEUE_PROFILING unset is off unless a profiling diagnostic is requested, then on;
// "1" is on; "0" is off whatever is requested; any other value counts as unset. A diagnostic counts as requested by its
// own parse of its own variable, and a diagnostic that considers itself enabled is always one the resolver names.

#include "../queue-profiling.hpp"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>

namespace {

int g_failures = 0;

#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                        \
            std::fprintf(stderr, "\n");                               \
            ++g_failures;                                             \
        }                                                             \
    } while (0)

using reason = ggml_sycl_queue_profiling_reason;

std::map<std::string, std::string> g_env;

const char * fake_getenv(const char * name) {
    const auto it = g_env.find(name);
    return it == g_env.end() ? nullptr : it->second.c_str();
}

void check_decision(const char * knob,
                    const char * diagnostic,
                    bool         want_enabled,
                    reason       want_reason,
                    const char * what) {
    const ggml_sycl_queue_profiling_decision d = ggml_sycl_queue_profiling_decide(knob, diagnostic);
    CHECK(d.enabled == want_enabled && d.reason == want_reason,
          "%s: knob=%s diagnostic=%s -> enabled=%d reason=%d, want %d/%d", what, knob ? knob : "(unset)",
          diagnostic ? diagnostic : "(none)", (int) d.enabled, (int) d.reason, (int) want_enabled, (int) want_reason);
}

// The requested diagnostic with exactly `name=value` in the environment; "" when none is requested.
std::string requested_with(const char * name, const char * value) {
    g_env.clear();
    g_env[name]    = value;
    const char * r = ggml_sycl_queue_profiling_requested_diagnostic(fake_getenv);
    return r ? r : "";
}

}  // namespace

int main() {
    // the list: the five diagnostics that read event profiling info, each once
    CHECK(ggml_sycl_queue_profiling_diagnostic_count == 5, "the list holds %zu diagnostics, expected 5",
          ggml_sycl_queue_profiling_diagnostic_count);
    for (size_t i = 0; i < ggml_sycl_queue_profiling_diagnostic_count; ++i) {
        for (size_t j = i + 1; j < ggml_sycl_queue_profiling_diagnostic_count; ++j) {
            CHECK(std::strcmp(ggml_sycl_queue_profiling_diagnostics[i], ggml_sycl_queue_profiling_diagnostics[j]) != 0,
                  "%s is in the list twice", ggml_sycl_queue_profiling_diagnostics[i]);
        }
    }

    // the three states of the knob, and the unknown value
    check_decision(nullptr, nullptr, false, reason::DEFAULT_OFF, "unset, nothing requested");
    for (size_t i = 0; i < ggml_sycl_queue_profiling_diagnostic_count; ++i) {
        const char * d = ggml_sycl_queue_profiling_diagnostics[i];
        check_decision(nullptr, d, true, reason::DIAGNOSTIC, d);
        check_decision("1", d, true, reason::KNOB_ON, d);
        check_decision("0", d, false, reason::KNOB_OFF, d);
        check_decision("2", d, true, reason::BAD_VALUE_DIAGNOSTIC, d);
    }
    check_decision("1", nullptr, true, reason::KNOB_ON, "1, nothing requested");
    check_decision("0", nullptr, false, reason::KNOB_OFF, "0, nothing requested");
    for (const char * bad : { "2", "true", "on", "", "01", "1 ", " 0", "-1", "yes" }) {
        check_decision(bad, nullptr, false, reason::BAD_VALUE_OFF, "unknown value, nothing requested");
    }

    // nothing set: nothing requested
    g_env.clear();
    CHECK(ggml_sycl_queue_profiling_requested_diagnostic(fake_getenv) == nullptr,
          "an empty environment requested a diagnostic");

    // GGML_SYCL_KERNEL_PROFILE: the profiler's own boolean parse
    for (const char * on : { "1", "true", "TRUE", "yes", "on", "2", "enabled", "garbage-longer-than-seven" }) {
        CHECK(requested_with("GGML_SYCL_KERNEL_PROFILE", on) == "GGML_SYCL_KERNEL_PROFILE", "KERNEL_PROFILE=%s is on",
              on);
    }
    for (const char * off : { "0", "false", "False", "no", "off", "OFF", "" }) {
        CHECK(requested_with("GGML_SYCL_KERNEL_PROFILE", off).empty(), "KERNEL_PROFILE=%s is off", off);
    }

    // GGML_SYCL_MXFP4_TG_PROFILE and _PP_PROFILE: a non-zero integer
    for (const char * name : { "GGML_SYCL_MXFP4_TG_PROFILE", "GGML_SYCL_MXFP4_PP_PROFILE" }) {
        for (const char * on : { "1", "2", "-1", " 3", "1x" }) {
            CHECK(requested_with(name, on) == name, "%s=%s is on", name, on);
        }
        for (const char * off : { "0", "", "abc", "x1", "0x1" }) {
            CHECK(requested_with(name, off).empty(), "%s=%s is off", name, off);
        }
    }

    // GGML_SYCL_ATTN_SYNC_PROFILE: set to anything, "0" included (its own parse is a presence test)
    for (const char * on : { "1", "0", "" }) {
        CHECK(requested_with("GGML_SYCL_ATTN_SYNC_PROFILE", on) == "GGML_SYCL_ATTN_SYNC_PROFILE",
              "ATTN_SYNC_PROFILE=%s is on", on);
    }

    // GGML_SYCL_MOE_EXEC_TRACE: only level 2 times the ring windows from device events
    for (const char * on : { "2", "3", "7" }) {
        CHECK(requested_with("GGML_SYCL_MOE_EXEC_TRACE", on) == "GGML_SYCL_MOE_EXEC_TRACE", "MOE_EXEC_TRACE=%s is on",
              on);
    }
    for (const char * off : { "0", "1", "", "junk", "-3" }) {
        CHECK(requested_with("GGML_SYCL_MOE_EXEC_TRACE", off).empty(), "MOE_EXEC_TRACE=%s is off", off);
    }

    // a variable that is not a profiling diagnostic requests nothing
    CHECK(requested_with("GGML_SYCL_MOE_EXEC", "ring").empty(), "GGML_SYCL_MOE_EXEC requested a diagnostic");
    CHECK(requested_with("GGML_SYCL_E2E_TG_PROFILE", "1").empty(), "a host-time profile requested event profiling");
    CHECK(requested_with("GGML_SYCL_QUEUE_PROFILING", "1").empty(), "the knob itself requested a diagnostic");

    // several requested: the first of the list names the WARN
    g_env.clear();
    g_env["GGML_SYCL_MXFP4_TG_PROFILE"] = "1";
    g_env["GGML_SYCL_KERNEL_PROFILE"]   = "1";
    const char * first                  = ggml_sycl_queue_profiling_requested_diagnostic(fake_getenv);
    CHECK(first != nullptr && std::strcmp(first, "GGML_SYCL_KERNEL_PROFILE") == 0,
          "the first diagnostic of the list names the WARN");

    // the shared parse helpers, which the diagnostics call themselves
    CHECK(!ggml_sycl_profile_env_nonzero(nullptr) && !ggml_sycl_profile_env_set(nullptr) &&
              ggml_sycl_profile_env_flag(nullptr, true) && !ggml_sycl_profile_env_flag(nullptr, false) &&
              ggml_sycl_profile_env_flag("", true),
          "an unset variable is the default of its parse");

    if (g_failures == 0) {
        std::printf("PASS test-sycl-queue-profiling-resolver\n");
        return 0;
    }
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
}
