// Runtime gate for the SYCL backend's env-var self-report (llama.cpp-5bd8).
//
// ggml_check_sycl() reports which GGML_SYCL_* settings it parsed. That report
// used to be GGML_LOG_INFO only, and upstream 67b2b7f2f ("logs : reduce",
// #23021) put every backend INFO line above the default log threshold -- so the
// report stopped reaching users without anyone noticing, for months. A source
// assertion cannot catch that: the code was present and correct the whole time,
// it was the *output* that vanished. So this gate drives a real backend init
// and asserts the line arrives at the log callback.
//
// It deliberately checks both directions. A settings report that printed
// everything unconditionally would satisfy a "does the string appear" check
// while re-introducing the noise the WARN form exists to avoid, so the test
// also asserts that a variable it did NOT set is absent.
//
// Exits 77 (ctest SKIP_RETURN_CODE) when no SYCL device is present.

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            std::fprintf(stderr, "--- captured log ---\n%s\n", g_log.c_str());  \
            return 1;                                                           \
        }                                                                       \
    } while (0)

// The report line ggml_check_sycl() emits at WARN when any setting is
// non-default. Keep in sync with ggml/src/ggml-sycl/ggml-sycl.cpp.
#define REPORT_MARKER "non-default settings in effect"

static std::string g_log;

static void capture_log(enum ggml_log_level level, const char * text, void * user_data) {
    GGML_UNUSED(level);
    GGML_UNUSED(user_data);
    if (text != nullptr) {
        g_log += text;
    }
}

static bool contains(const std::string & haystack, const char * needle) {
    return haystack.find(needle) != std::string::npos;
}

static void set_env_var(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void clear_env_var(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

int main() {
    // GGML_SYCL_HANDLE_STRICT is diagnostic-only (it turns data_handle vs
    // data_device divergence into warnings and changes no behaviour), so it is
    // safe to force on for the duration of a device init.
    set_env_var("GGML_SYCL_HANDLE_STRICT", "1");

    // The negative half of the assertion below only means something if this is
    // genuinely unset, and ctest inherits the caller's environment.
    clear_env_var("GGML_SYCL_KQV_FORCE_SIMPLE");

    // Must be installed before the first backend init: ggml_check_sycl() runs
    // once per process behind a static guard, so there is no second chance.
    ggml_log_set(capture_log, nullptr);

    ggml_backend_reg_t reg = ggml_backend_reg_by_name("SYCL");
    if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
        std::fprintf(stderr, "SKIP: no SYCL device available\n");
        return 77;
    }

    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    CHECK(dev != nullptr, "ggml_backend_reg_dev_get returned null");

    // This is the path a real run takes: registry -> device iface init_backend
    // -> ggml_backend_sycl_init() -> ggml_check_sycl().
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    CHECK(backend != nullptr, "ggml_backend_dev_init failed");

    const size_t report_start = g_log.find(REPORT_MARKER);
    CHECK(report_start != std::string::npos, "SYCL settings report did not reach the log callback");

    // Assert against the report LINE, not the whole capture. The verbose INFO
    // table further down lists every variable including the ones left at their
    // defaults, so a whole-capture search would make the negative assertion
    // below fail against a perfectly correct report.
    const size_t      report_end = g_log.find('\n', report_start);
    const std::string report     = g_log.substr(report_start, report_end - report_start);

    CHECK(contains(report, "GGML_SYCL_HANDLE_STRICT=1"), "settings report omitted the variable that was set");
    CHECK(!contains(report, "GGML_SYCL_KQV_FORCE_SIMPLE"), "settings report listed a variable left at its default");

    ggml_backend_free(backend);

    std::fprintf(stderr, "PASS: SYCL settings report reached the log callback\n");
    return 0;
}
