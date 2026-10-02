//
// Diagnostic arms for llama.cpp-rb2h (B50 qwen35 perplexity nondeterminism with several sequences per ubatch).
//
// Each arm is an env toggle that is off by default and logs a one-time WARN when it is first consulted while
// set (INFO is dropped at default verbosity). They exist so one GPU round can tell which qhfp per-op change the
// variation follows; remove them with the fix.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstdlib>
#include <mutex>
#include <set>
#include <string>

inline bool ggml_sycl_rb2h_arm(const char * name) {
    const char * env = std::getenv(name);
    if (env == nullptr || std::atoi(env) == 0) {
        return false;
    }
    static std::mutex            mu;
    static std::set<std::string> announced;
    {
        std::lock_guard<std::mutex> lock(mu);
        if (announced.insert(name).second) {
            GGML_LOG_WARN("[RB2H] diagnostic arm %s is active\n", name);
        }
    }
    return true;
}
