// An exception that escapes a tool's entry function must end as an error line and status 1, not std::terminate.

#include "tool-entry.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include <unistd.h>

static int g_seen_argc = -1;
static char ** g_seen_argv = nullptr;

static int entry_throws_std(int, char **) {
    throw std::runtime_error("this custom template is not supported");
}

static int entry_throws_other(int, char **) {
    throw 7;
}

static int entry_returns_zero(int argc, char ** argv) {
    g_seen_argc = argc;
    g_seen_argv = argv;
    return 0;
}

static int entry_returns_three(int, char **) {
    return 3;
}

// Runs fn with stderr sent to a temporary file and returns what it wrote there.
static std::string captured_stderr(int (*fn)(), int & rc) {
    fflush(stderr);
    FILE * tmp = tmpfile();
    if (tmp == nullptr) {
        rc = -1;
        return "";
    }
    const int saved = dup(fileno(stderr));
    dup2(fileno(tmp), fileno(stderr));
    rc = fn();
    fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);

    std::string out;
    char        buf[256];
    rewind(tmp);
    size_t n = 0;
    while ((n = fread(buf, 1, sizeof(buf), tmp)) > 0) {
        out.append(buf, n);
    }
    fclose(tmp);
    return out;
}

static char   g_arg0[] = "tool";
static char   g_arg1[] = "-x";
static char * g_argv[] = { g_arg0, g_arg1, nullptr };

static int run_std() {
    return common_tool_entry("llama-test", entry_throws_std, 2, g_argv);
}

static int run_other() {
    return common_tool_entry("llama-test", entry_throws_other, 2, g_argv);
}

static int g_failures = 0;

#define CHECK(cond, what)                                 \
    do {                                                  \
        if (!(cond)) {                                    \
            fprintf(stdout, "FAIL: %s\n", what);          \
            g_failures++;                                 \
        } else {                                          \
            fprintf(stdout, "ok: %s\n", what);            \
        }                                                 \
    } while (0)

int main() {
    int               rc  = 0;
    const std::string err = captured_stderr(run_std, rc);
    CHECK(rc == 1, "a std::exception that escapes the entry gives status 1");
    CHECK(err == "llama-test: error: this custom template is not supported\n",
          "its message is printed as one error line");

    const std::string err_other = captured_stderr(run_other, rc);
    CHECK(rc == 1, "an exception that is not a std::exception gives status 1");
    CHECK(err_other == "llama-test: error: unknown exception\n", "and says so");

    CHECK(common_tool_entry("llama-test", entry_returns_zero, 2, g_argv) == 0, "a normal return of 0 passes through");
    CHECK(g_seen_argc == 2 && g_seen_argv == g_argv, "the entry receives argc and argv unchanged");
    CHECK(common_tool_entry("llama-test", entry_returns_three, 2, g_argv) == 3, "a non-zero return passes through");

    if (g_failures != 0) {
        fprintf(stdout, "%d check(s) failed\n", g_failures);
        return 1;
    }
    fprintf(stdout, "all tool-entry checks passed\n");
    return 0;
}
