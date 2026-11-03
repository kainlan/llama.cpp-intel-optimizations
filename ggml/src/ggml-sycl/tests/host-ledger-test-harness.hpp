//
// The harness both host ledger tests share: an arm's PASS, FAIL or VOID verdict, the captured log lines the arms read,
// and the child runner their abort legs use. Each test includes it once.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml.h"

#include <poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace host_ledger_test_harness {

enum class outcome { PASS, FAIL, VOID };

struct arm_result {
    outcome     result = outcome::PASS;
    std::string why;
};

// The first FAIL or VOID wins; a later check does not overwrite the reason.
inline void fail(arm_result & r, const std::string & why) {
    if (r.result == outcome::PASS) {
        r.result = outcome::FAIL;
        r.why    = why;
    }
}

inline void void_arm(arm_result & r, const std::string & why) {
    if (r.result == outcome::PASS) {
        r.result = outcome::VOID;
        r.why    = why;
    }
}

inline void expect(arm_result & r, bool condition, const std::string & why) {
    if (!condition) {
        fail(r, why);
    }
}

inline std::string num(uint64_t v) {
    return std::to_string(static_cast<unsigned long long>(v));
}

// --- the log lines the arms read ---

inline std::mutex               g_log_mutex;
inline std::vector<std::string> g_log_lines;

// The log callback: each line goes to stderr and into g_log_lines.
inline void capture_log(ggml_log_level level, const char * text, void * user) {
    (void) level;
    (void) user;
    if (text == nullptr) {
        return;
    }
    fputs(text, stderr);
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_lines.emplace_back(text);
}

inline void log_clear() {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_lines.clear();
}

// The captured lines that contain `a`, and `b` too when it is given.
inline size_t log_count_with(const char * a, const char * b = nullptr) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    size_t                      n = 0;
    for (const std::string & line : g_log_lines) {
        if (line.find(a) != std::string::npos && (b == nullptr || line.find(b) != std::string::npos)) {
            ++n;
        }
    }
    return n;
}

// The value printed after " <name>=" on the line, or SIZE_MAX when the name is absent.
inline size_t printed(const std::string & line, const char * name) {
    const std::string key = std::string(" ") + name + "=";
    const size_t      at  = line.find(key);
    if (at == std::string::npos) {
        return SIZE_MAX;
    }
    return static_cast<size_t>(std::strtoull(line.c_str() + at + key.size(), nullptr, 10));
}

// --- the child runner ---

struct child_run {
    bool        ran    = false;  // started, and exited or was signalled within the bound
    int         status = 0;
    std::string err;             // everything the child wrote to stderr
};

// Runs this binary with `argv` and `envp` and collects its stderr, killing it once `bound` has passed. The caller
// builds both before the fork, so the child calls only dup2, close and execve (the caller may have threads, and only
// async-signal-safe calls are legal between fork and exec).
inline child_run run_child(char * const * argv, char * const * envp, std::chrono::milliseconds bound) {
    child_run r;
    int       fds[2];
    if (pipe(fds) != 0) {
        return r;
    }
    const pid_t pid = fork();
    if (pid == 0) {
        dup2(fds[1], STDERR_FILENO);
        close(fds[0]);
        close(fds[1]);
        execve("/proc/self/exe", argv, envp);
        _exit(127);
    }
    close(fds[1]);
    if (pid < 0) {
        close(fds[0]);
        return r;
    }
    const auto deadline  = std::chrono::steady_clock::now() + bound;
    bool       timed_out = false;
    char       buf[512];
    for (;;) {
        const auto left =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (left <= 0) {
            timed_out = true;
            kill(pid, SIGKILL);
            break;
        }
        pollfd    p     = { fds[0], POLLIN, 0 };
        const int ready = poll(&p, 1, static_cast<int>(left));
        if (ready <= 0) {
            continue;  // EINTR, or the bound passed: the loop's top decides
        }
        const ssize_t got = read(fds[0], buf, sizeof(buf));
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            break;  // end of file: the child has exited
        }
        r.err.append(buf, static_cast<size_t>(got));
    }
    close(fds[0]);
    if (waitpid(pid, &r.status, 0) != pid) {
        return r;
    }
    r.ran = !timed_out;
    return r;
}

}  // namespace host_ledger_test_harness
