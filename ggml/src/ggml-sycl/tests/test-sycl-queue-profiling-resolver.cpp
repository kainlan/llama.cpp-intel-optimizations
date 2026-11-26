// The queue-profiling resolver (queue-profiling.hpp, common.cpp). Two parts, neither opens a SYCL device or loads a
// model:
//
//  1. The pure parts, ggml_sycl_queue_profiling_decide() and ggml_sycl_queue_profiling_requested_diagnostic(), called
//     with plain values and an injected getenv, plus the shared parse helpers of the header.
//  2. The memoised wrapper, ggml_sycl_queue_profiling_enabled() and ggml_sycl_queue_profiling_resolved_reason(), which
//     read this process's environment once and print their WARN lines once. Each arm runs in a forked child, so each
//     resolves fresh: the child sets its environment, captures the log with ggml_log_set, asks twice, changes the
//     environment, asks again, and checks the reason, the answer and the WARN lines it saw.
//
// The states under test: GGML_SYCL_QUEUE_PROFILING unset is off unless a profiling diagnostic is requested, then on;
// "1" is on; "0" is off whatever is requested; any other value counts as unset. A diagnostic counts as requested by its
// own parse of its own variable, and a diagnostic that considers itself enabled is always one the resolver names.

#include "../queue-profiling.hpp"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#ifndef _WIN32
#    include <sys/wait.h>
#    include <unistd.h>
#endif

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

#ifndef _WIN32

// The WARN lines of the SYCL backend that one child process saw, in order.
std::vector<std::string> g_warns;

void capture_log(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level == GGML_LOG_LEVEL_WARN && std::strncmp(text, "[SYCL]", 6) == 0) {
        g_warns.emplace_back(text);
    }
}

const char * const KNOB = "GGML_SYCL_QUEUE_PROFILING";

using env_list = std::vector<std::pair<const char *, const char *>>;

struct resolver_arm {
    const char *              name;
    // set before the first question
    env_list                  env;
    bool                      want_enabled;
    reason                    want_reason;
    // a part of each WARN the resolution prints, in order
    std::vector<const char *> want_warns;
    // After the resolution the arm asks the refusing reader's WARN twice for one diagnostic and once for another: one
    // line per diagnostic, and each line holds every one of these parts (the cause, and the way out). Empty: the arm
    // does not ask.
    std::vector<const char *> want_unavailable;
};

// Runs one arm in the current process, which is a fresh child. Returns the number of failed checks.
int run_resolver_arm(const resolver_arm & arm) {
    g_failures = 0;
    for (size_t i = 0; i < ggml_sycl_queue_profiling_diagnostic_count; ++i) {
        unsetenv(ggml_sycl_queue_profiling_diagnostics[i]);
    }
    unsetenv(KNOB);
    for (const auto & kv : arm.env) {
        setenv(kv.first, kv.second, 1);
    }
    g_warns.clear();
    ggml_log_set(capture_log, nullptr);

    const bool   first        = ggml_sycl_queue_profiling_enabled();
    const bool   second       = ggml_sycl_queue_profiling_enabled();
    const reason first_reason = ggml_sycl_queue_profiling_resolved_reason();
    CHECK(first == arm.want_enabled, "%s: answers %d, want %d", arm.name, (int) first, (int) arm.want_enabled);
    CHECK(second == first, "%s: the second question answered %d after %d", arm.name, (int) second, (int) first);
    CHECK(first_reason == arm.want_reason, "%s: reason %d, want %d", arm.name, (int) first_reason,
          (int) arm.want_reason);
    CHECK(g_warns.size() == arm.want_warns.size(), "%s: %zu WARN lines, want %zu", arm.name, g_warns.size(),
          arm.want_warns.size());
    for (size_t i = 0; i < arm.want_warns.size() && i < g_warns.size(); ++i) {
        CHECK(g_warns[i].find(arm.want_warns[i]) != std::string::npos, "%s: WARN %zu is `%s`, want it to hold `%s`",
              arm.name, i, g_warns[i].c_str(), arm.want_warns[i]);
    }

    // the answer is fixed for the process: flip the knob and the diagnostic the other way, ask again, and nothing moves
    setenv(KNOB, arm.want_enabled ? "0" : "1", 1);
    setenv("GGML_SYCL_KERNEL_PROFILE", arm.want_enabled ? "0" : "1", 1);
    CHECK(ggml_sycl_queue_profiling_enabled() == first, "%s: the answer changed after setenv", arm.name);
    CHECK(ggml_sycl_queue_profiling_resolved_reason() == first_reason, "%s: the reason changed after setenv", arm.name);
    CHECK(g_warns.size() == arm.want_warns.size(), "%s: %zu WARN lines after setenv, want %zu", arm.name,
          g_warns.size(), arm.want_warns.size());

    if (!arm.want_unavailable.empty()) {
        const size_t before = g_warns.size();
        ggml_sycl_queue_profiling_warn_unavailable("GGML_SYCL_KERNEL_PROFILE");
        ggml_sycl_queue_profiling_warn_unavailable("GGML_SYCL_KERNEL_PROFILE");
        ggml_sycl_queue_profiling_warn_unavailable("GGML_SYCL_MXFP4_TG_PROFILE");
        CHECK(g_warns.size() == before + 2, "%s: %zu refusal WARN lines, want one per diagnostic (2)", arm.name,
              g_warns.size() - before);
        for (size_t i = before; i < g_warns.size(); ++i) {
            for (const char * part : arm.want_unavailable) {
                CHECK(g_warns[i].find(part) != std::string::npos, "%s: refusal WARN `%s` does not hold `%s`", arm.name,
                      g_warns[i].c_str(), part);
            }
        }
        if (g_warns.size() >= before + 2) {
            CHECK(g_warns[before].find("GGML_SYCL_KERNEL_PROFILE ") == 7 &&
                      g_warns[before + 1].find("GGML_SYCL_MXFP4_TG_PROFILE ") == 7,
                  "%s: each refusal WARN names its own diagnostic first", arm.name);
        }
    }
    return g_failures;
}

// Forks one child per arm, so each resolves the knob fresh in its own process. The parent never asks the memoised
// wrapper. Returns true: the arms ran.
bool check_resolver_arms() {
    const char * const bad_value_warn = "GGML_SYCL_QUEUE_PROFILING=yes is neither 0 nor 1; treating it as unset";
    const char * const kernel_warn = "GGML_SYCL_KERNEL_PROFILE is set, so the backend's queues carry event profiling";
    const char * const mxfp4_warn  = "GGML_SYCL_MXFP4_TG_PROFILE is set, so the backend's queues carry event profiling";

    std::vector<resolver_arm> arms;
    // Default off, then a diagnostic asked for afterwards: it gets the late refusal WARN (the knob did not resolve off
    // by its own value), not the knob-0 one.
    arms.push_back({
        "default off", {},
         false, reason::DEFAULT_OFF, {},
         { "resolved off before", "GGML_SYCL_QUEUE_PROFILING=1" }
    });
    arms.push_back({ "diagnostic turns it on",
                     { { "GGML_SYCL_KERNEL_PROFILE", "1" } },
                     true,
                     reason::DIAGNOSTIC,
                     { kernel_warn },
                     {} });
    // a requested diagnostic under knob 1 is not what turned profiling on: no "turned on" WARN
    arms.push_back({
        "knob 1", { { KNOB, "1" }, { "GGML_SYCL_KERNEL_PROFILE", "1" } },
         true, reason::KNOB_ON, {},
         {}
    });
    arms.push_back({
        "knob 0 refuses a requested diagnostic",
        { { KNOB, "0" }, { "GGML_SYCL_KERNEL_PROFILE", "1" } },
        false,
        reason::KNOB_OFF,
        {},
        { "GGML_SYCL_QUEUE_PROFILING=0", "GGML_SYCL_QUEUE_PROFILING=1" }
    });
    arms.push_back(
        { "bad value, nothing requested", { { KNOB, "yes" } }, false, reason::BAD_VALUE_OFF, { bad_value_warn }, {} });
    arms.push_back({
        "bad value, diagnostic requested",
        { { KNOB, "yes" }, { "GGML_SYCL_MXFP4_TG_PROFILE", "1" } },
        true,
        reason::BAD_VALUE_DIAGNOSTIC,
        { bad_value_warn, mxfp4_warn },
        {}
    });
    for (const resolver_arm & arm : arms) {
        std::fflush(stdout);
        std::fflush(stderr);
        const pid_t pid = fork();
        if (pid < 0) {
            CHECK(false, "%s: fork failed", arm.name);
            continue;
        }
        if (pid == 0) {
            const int failed = run_resolver_arm(arm);
            std::fflush(stdout);
            std::fflush(stderr);
            _exit(failed == 0 ? 0 : 1);  // no static destructors of the runtime in a child that never opened a device
        }
        int status = 0;
        CHECK(waitpid(pid, &status, 0) == pid, "%s: waitpid failed", arm.name);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "%s: the child failed (status %d)", arm.name, status);
    }
    return true;
}

#else

// No fork on this platform: the arms cannot run, and main reports the run as skipped, not passed.
bool check_resolver_arms() {
    return false;
}

#endif

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

    // the memoised wrapper, one forked child per state
    const bool arms_ran = check_resolver_arms();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    if (!arms_ran) {
        std::printf("SKIP test-sycl-queue-profiling-resolver: the forked arms need fork(); the pure checks passed\n");
        return 77;  // ctest's SKIP_RETURN_CODE: a skip that shows as a skip
    }
    std::printf("PASS test-sycl-queue-profiling-resolver\n");
    return 0;
}
