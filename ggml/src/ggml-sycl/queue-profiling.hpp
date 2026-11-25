#pragma once

#include <cstdlib>
#include <cstring>

// Whether the backend builds its queues with sycl::property::queue::enable_profiling.
//
// The property is on by default: every queue the backend builds carries it, as it always has. On the Level Zero v2
// adapter it makes every submitted command stamp a device timestamp, one `urDeviceGetGlobalTimestamps` call a command,
// and only diagnostics read the result (event profiling info). GGML_SYCL_QUEUE_PROFILING=0 builds the queues without
// the property, so a run that reads no profiling info stops paying for the stamps. Unset, or any value but exactly
// "0", keeps the property.
//
// The answer comes from the environment, never from a macro. The queue builders (dpct/helper.hpp, common.hpp) and the
// readers of profiling info are header-only inlines or sit in many translation units, and each translation unit
// compiles its own copy of an inline; a macro read there would let the linker's pick of one copy decide for the whole
// process, and the queues and their readers could then disagree. An environment read gives every copy the same answer.
// The value is read once per copy of this inline function, and every copy reads the same process environment, so the
// copies agree.
//
// This header sits below dpct/helper.hpp so both it and common.hpp can use the predicate without an include cycle.
inline bool ggml_sycl_queue_profiling_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_SYCL_QUEUE_PROFILING");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

// Prints, once per process and per diagnostic, a WARN that `diagnostic` reports no figure because the queues carry no
// profiling (common.cpp). INFO would be dropped at the default verbosity.
void ggml_sycl_queue_profiling_warn_unavailable(const char * diagnostic);

// The guard of every reader of event profiling info: true when the queues carry profiling and the read may go ahead.
// A query on an event of a queue built without the property throws, so a reader that gets false must not call it, must
// report its "not available" value, and prints no figure. Naming the diagnostic is what gets the one WARN.
inline bool ggml_sycl_queue_profiling_readable(const char * diagnostic) {
    if (ggml_sycl_queue_profiling_enabled()) {
        return true;
    }
    ggml_sycl_queue_profiling_warn_unavailable(diagnostic);
    return false;
}
