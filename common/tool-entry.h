#pragma once

#include <cstdio>
#include <exception>

// Runs a tool's entry function and turns an exception that escapes it into an error line and status 1.
// Without a handler the exception reaches std::terminate: the process aborts and dumps core, and nothing that
// owns a model or a context is destroyed. With a handler here the stack unwinds first, so those owners release
// what they hold, and main returns normally.
static inline int common_tool_entry(const char * tool_name, int (*entry)(int, char **), int argc, char ** argv) {
    try {
        return entry(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "%s: error: %s\n", tool_name, e.what());
    } catch (...) {
        fprintf(stderr, "%s: error: unknown exception\n", tool_name);
    }
    return 1;
}
