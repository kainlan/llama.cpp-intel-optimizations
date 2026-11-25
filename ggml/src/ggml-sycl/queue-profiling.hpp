#pragma once

#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <cstring>

// Whether the backend builds its queues with sycl::property::queue::enable_profiling.
//
// The property is OFF by default. On the Level Zero v2 adapter a profiling queue reads a device timestamp for every
// command it submits (one `urDeviceGetGlobalTimestamps` call, an ioctl with a busy wait), and only diagnostics read the
// result (event profiling info). Taking the property off measured GPT-OSS 20B decode 12 to 26 % faster and a Qwen
// prefill 34 % faster, and nothing slower (driver 26.44, 2026-11-24).
//
// GGML_SYCL_QUEUE_PROFILING has three states:
//   unset  off, unless a profiling diagnostic is requested in the environment (the list below); then on, with one WARN
//          that names the variable that turned it on, because a profiling run is slower and its figures include the
//          timestamp cost;
//   "1"    on;
//   "0"    off; a requested diagnostic prints one WARN and reports no figure (warn_unless_readable below);
//   any other value: one WARN, then treated as unset.
// The answer is resolved once per process, at the first queue the backend builds, and never changes after that. A
// diagnostic variable set later than that is refused as if the knob were "0".
//
// The answer comes from the environment, never from a macro. The queue builders (dpct/helper.hpp, common.hpp) and the
// readers of profiling info sit in many translation units, and a macro read in a header-only inline would let the
// linker's pick of one copy decide for the whole process. The resolver below is therefore defined once, out of line,
// in libggml-sycl (common.cpp); every translation unit asks that one copy.
//
// This header sits below dpct/helper.hpp so both it and common.hpp can use the predicate without an include cycle.

// The variables that ask for event profiling info. One list: the resolver reads it, and every call of
// ggml_sycl_queue_profiling_warn_unless_readable() names exactly one entry. A new diagnostic that reads event
// profiling info is added here, in the resolver's dispatch (common.cpp), and in the parse helpers below.
inline constexpr const char * ggml_sycl_queue_profiling_diagnostics[] = {
    "GGML_SYCL_KERNEL_PROFILE",    "GGML_SYCL_MXFP4_TG_PROFILE", "GGML_SYCL_MXFP4_PP_PROFILE",
    "GGML_SYCL_ATTN_SYNC_PROFILE", "GGML_SYCL_MOE_EXEC_TRACE",
};

inline constexpr size_t ggml_sycl_queue_profiling_diagnostic_count =
    sizeof(ggml_sycl_queue_profiling_diagnostics) / sizeof(ggml_sycl_queue_profiling_diagnostics[0]);

// How each diagnostic reads its own variable. The diagnostic and the resolver both call these, so a diagnostic cannot
// consider itself enabled while the resolver does not count it as requested. A diagnostic whose own parse cannot be
// shared (GGML_SYCL_MOE_EXEC_TRACE, whose level comes from moe_exec_config_parse) is parsed by the resolver with that
// parser; a name in the list without any parse falls back to "set, non-empty and not exactly "0"".

// GGML_SYCL_MXFP4_TG_PROFILE, GGML_SYCL_MXFP4_PP_PROFILE: an integer that is not zero. Text that is no number is zero.
inline bool ggml_sycl_profile_env_nonzero(const char * value) {
    return value != nullptr && std::atoi(value) != 0;
}

// GGML_SYCL_ATTN_SYNC_PROFILE: set to anything, "0" and the empty string included.
inline bool ggml_sycl_profile_env_set(const char * value) {
    return value != nullptr;
}

// GGML_SYCL_KERNEL_PROFILE (and the profiler's other boolean variables): unset or empty is `default_value`; "0",
// "false", "no" and "off", in any case, are false; any other text is true.
inline bool ggml_sycl_profile_env_flag(const char * value, bool default_value) {
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    char         lowered[8];
    const size_t length = std::strlen(value);
    if (length < sizeof(lowered)) {
        for (size_t i = 0; i <= length; ++i) {
            lowered[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(value[i])));
        }
        if (std::strcmp(lowered, "0") == 0 || std::strcmp(lowered, "false") == 0 || std::strcmp(lowered, "no") == 0 ||
            std::strcmp(lowered, "off") == 0) {
            return false;
        }
    }
    return true;
}

// Why the knob resolved the way it did.
enum class ggml_sycl_queue_profiling_reason {
    DEFAULT_OFF,           // unset, nothing requested
    DIAGNOSTIC,            // unset, a diagnostic is requested: on
    KNOB_ON,               // "1"
    KNOB_OFF,              // "0"
    BAD_VALUE_OFF,         // any other value, nothing requested: treated as unset
    BAD_VALUE_DIAGNOSTIC,  // any other value, a diagnostic is requested: treated as unset, so on
};

struct ggml_sycl_queue_profiling_decision {
    bool                             enabled;
    ggml_sycl_queue_profiling_reason reason;
};

// The pure part of the resolver: no environment, no SYCL, no globals. `knob_value` is the text of
// GGML_SYCL_QUEUE_PROFILING or null; `requested_diagnostic` is the name of a requested diagnostic or null (common.cpp).
ggml_sycl_queue_profiling_decision ggml_sycl_queue_profiling_decide(const char * knob_value,
                                                                    const char * requested_diagnostic);

// The first name in ggml_sycl_queue_profiling_diagnostics whose variable, read through `getenv_fn`, asks for the
// diagnostic by that diagnostic's own parse; null when none does. No globals; a test passes its own getenv_fn.
const char * ggml_sycl_queue_profiling_requested_diagnostic(const char * (*getenv_fn)(const char * name));

// The memoised wrapper: reads GGML_SYCL_QUEUE_PROFILING and the diagnostics' variables once, prints its WARN lines
// once, and answers the same for the rest of the process (common.cpp).
bool                             ggml_sycl_queue_profiling_enabled();
ggml_sycl_queue_profiling_reason ggml_sycl_queue_profiling_resolved_reason();

// Prints, once per process and per diagnostic, a WARN that `diagnostic` reports no figure because the queues carry no
// profiling (common.cpp). INFO would be dropped at the default verbosity.
void ggml_sycl_queue_profiling_warn_unavailable(const char * diagnostic);

// The guard of every reader of event profiling info: true when the queues carry profiling and the read may go ahead.
// It is not a pure predicate: when the queues carry no profiling it prints the one WARN for `diagnostic`. A query on an
// event of a queue built without the property throws, so a reader that gets false must not call it, must report its
// "not available" value, and prints no figure. `diagnostic` is exactly one entry of
// ggml_sycl_queue_profiling_diagnostics, spelled as a string literal.
inline bool ggml_sycl_queue_profiling_warn_unless_readable(const char * diagnostic) {
    if (ggml_sycl_queue_profiling_enabled()) {
        return true;
    }
    ggml_sycl_queue_profiling_warn_unavailable(diagnostic);
    return false;
}
