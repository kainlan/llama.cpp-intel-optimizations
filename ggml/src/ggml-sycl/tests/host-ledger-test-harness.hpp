//
// The harness both host ledger tests share: an arm's PASS, FAIL or VOID verdict, the loop that runs the arms, the
// captured log lines the arms read, and the child runner their abort legs use. Each test includes it once.
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
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

extern char ** environ;

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

// --- the arm loop ---

// In a namespace of its own, so a test that names the harness with a using-directive and keeps a loop of its own
// sees no second outcome_name; a test that runs its arms through this loop names it.
namespace arm_loop {

struct named_arm {
    const char * name;
    arm_result (*run)();
};

inline const char * outcome_name(outcome o) {
    return o == outcome::PASS ? "PASS" : o == outcome::FAIL ? "FAIL" : "VOID";
}

// Runs every arm, so one failure does not hide the rest, printing each verdict, then `label` with the counts. Returns
// the binary's status: 0 only when every arm passed, since a VOID arm fails the binary like a FAIL.
inline int run_arms(const std::vector<named_arm> & arms, const std::string & label) {
    int passed = 0, failed = 0, voided = 0;
    for (const named_arm & arm : arms) {
        const arm_result result = arm.run();
        std::printf("%s %s%s%s\n", outcome_name(result.result), arm.name, result.why.empty() ? "" : ": ",
                    result.why.c_str());
        std::fflush(stdout);
        passed += result.result == outcome::PASS;
        failed += result.result == outcome::FAIL;
        voided += result.result == outcome::VOID;
    }
    std::printf("%s: %d passed, %d failed, %d void\n", label.c_str(), passed, failed, voided);
    return failed == 0 && voided == 0 ? 0 : 1;
}

}  // namespace arm_loop

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
inline std::vector<std::string> log_lines_containing(const char * a, const char * b = nullptr) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::vector<std::string>    out;
    for (const std::string & line : g_log_lines) {
        if (line.find(a) != std::string::npos && (b == nullptr || line.find(b) != std::string::npos)) {
            out.push_back(line);
        }
    }
    return out;
}

// How many captured lines contain `a`, and `b` too when it is given.
inline size_t log_count_with(const char * a, const char * b = nullptr) {
    return log_lines_containing(a, b).size();
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

// This process's environment for a child, each "NAME=value" of `sets` replacing any inherited NAME, so the child's
// getenv reads the value given here and never the first of two entries.
inline std::vector<std::string> child_environment(const std::vector<std::string> & sets) {
    std::vector<std::string> env;
    for (char ** e = environ; e != nullptr && *e != nullptr; ++e) {
        bool replaced = false;
        for (const std::string & set : sets) {
            const size_t eq = set.find('=');
            if (eq != std::string::npos && std::strncmp(*e, set.c_str(), eq + 1) == 0) {
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            env.emplace_back(*e);
        }
    }
    env.insert(env.end(), sets.begin(), sets.end());
    return env;
}

// run_child over strings: builds the argv and envp arrays before the fork, as run_child requires, and keeps the
// strings alive until the child is reaped.
inline child_run run_child_strings(std::vector<std::string>  args,
                                   std::vector<std::string>  env,
                                   std::chrono::milliseconds bound) {
    std::vector<char *> argv;
    std::vector<char *> envp;
    for (std::string & a : args) {
        argv.push_back(a.data());
    }
    argv.push_back(nullptr);
    for (std::string & e : env) {
        envp.push_back(e.data());
    }
    envp.push_back(nullptr);
    return run_child(argv.data(), envp.data(), bound);
}

// How many lines of `text` contain `a`, and `b` too when it is given: log_count_with's match over a child's stderr
// rather than the captured log.
inline size_t text_count_with(const std::string & text, const char * a, const char * b = nullptr) {
    size_t count = 0;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        const std::string line = text.substr(start, end - start);
        if (line.find(a) != std::string::npos && (b == nullptr || line.find(b) != std::string::npos)) {
            count++;
        }
        start = end + 1;
    }
    return count;
}

}  // namespace host_ledger_test_harness
