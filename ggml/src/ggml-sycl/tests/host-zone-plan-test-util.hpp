//
// Shared scaffolding of the host-zone-plan host-only tests: a failure counter, a capture of ggml's log callback, and a
// child-process runner for the cases whose verdict is a process outcome (an abort under GGML_SYCL_STRICT_LEASES=1, or a
// helper that must throw rather than kill the process). A child's status is reported shell-style: 128 + the signal
// when it was killed, so SIGABRT reads as rc 134.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

namespace host_zone_plan_test {

inline int & failures() {
    static int n = 0;
    return n;
}

inline std::string & captured_log() {
    static std::string s;
    return s;
}

inline void capture_callback(ggml_log_level level, const char * text, void * user) {
    (void) level;
    (void) user;
    captured_log() += text;
    fputs(text, stderr);  // keep the production lines in the ctest output
}

// Starts a fresh capture of everything production logs through ggml's callback.
inline void capture_begin() {
    captured_log().clear();
    ggml_log_set(capture_callback, nullptr);
}

inline size_t count_of(const std::string & haystack, const std::string & needle) {
    size_t n   = 0;
    size_t pos = 0;
    while (!needle.empty() && (pos = haystack.find(needle, pos)) != std::string::npos) {
        ++n;
        pos += needle.size();
    }
    return n;
}

inline bool log_has(const std::string & needle) {
    return captured_log().find(needle) != std::string::npos;
}

#define HZP_CHECK(cond, ...)                            \
    do {                                                \
        if (!(cond)) {                                  \
            printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                        \
            printf("\n");                               \
            fflush(stdout);                             \
            ++host_zone_plan_test::failures();          \
        }                                               \
    } while (0)

struct child_result {
    int         rc = -1;  // exit status, or 128 + signal
    std::string err;      // the child's stderr
};

// Runs `body` in a forked child whose stderr is captured, with ggml's default log callback restored there (so every
// production line reaches that stderr in the order it was printed). The child exits 0 when `body` returns.
inline child_result run_in_child(const std::function<void()> & body) {
    child_result r;
    int          fds[2];
    if (pipe(fds) != 0) {
        return r;
    }
    fflush(stdout);
    fflush(stderr);
    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return r;
    }
    if (pid == 0) {
        close(fds[0]);
        dup2(fds[1], STDERR_FILENO);
        close(fds[1]);
        ggml_log_set(nullptr, nullptr);
        body();
        fflush(stderr);
        _exit(0);
    }
    close(fds[1]);
    char    buf[4096];
    ssize_t n;
    while ((n = read(fds[0], buf, sizeof(buf))) > 0) {
        r.err.append(buf, (size_t) n);
    }
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) {
        r.rc = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        r.rc = 128 + WTERMSIG(status);
    }
    return r;
}

inline int finish(const char * name) {
    if (failures() == 0) {
        printf("%s: PASS\n", name);
        return 0;
    }
    printf("%s: %d failure(s)\n", name, failures());
    return 1;
}

}  // namespace host_zone_plan_test
