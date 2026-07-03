#include "sycl-timeline.hpp"

#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string_view>

namespace ggml_sycl {
namespace {

struct sycl_timeline_state {
    std::mutex mutex;

    bool                 env_config_loaded   = false;
    sycl_timeline_config env_config          = {};
    bool                 test_config_enabled = false;
    sycl_timeline_config test_config         = {};
};

sycl_timeline_state & get_timeline_state() {
    static sycl_timeline_state * state = new sycl_timeline_state();
    return *state;
}

bool is_space(char ch) {
    return std::isspace(static_cast<unsigned char>(ch)) != 0;
}

std::string_view trim_ascii(const char * value) {
    if (value == nullptr) {
        return {};
    }

    const char * first = value;
    while (*first != '\0' && is_space(*first)) {
        ++first;
    }

    const char * last = first + std::strlen(first);
    while (last > first && is_space(*(last - 1))) {
        --last;
    }

    return { first, static_cast<size_t>(last - first) };
}

bool equals_ignore_case(std::string_view value, const char * literal) {
    const size_t literal_len = std::strlen(literal);
    if (value.size() != literal_len) {
        return false;
    }

    for (size_t i = 0; i < literal_len; ++i) {
        const unsigned char lhs = static_cast<unsigned char>(value[i]);
        const unsigned char rhs = static_cast<unsigned char>(literal[i]);
        if (std::tolower(lhs) != std::tolower(rhs)) {
            return false;
        }
    }

    return true;
}

sycl_timeline_mode parse_mode(const char * value) {
    const std::string_view mode = trim_ascii(value);
    if (mode.empty() || equals_ignore_case(mode, "0") || equals_ignore_case(mode, "off") ||
        equals_ignore_case(mode, "false") || equals_ignore_case(mode, "no") || equals_ignore_case(mode, "disabled")) {
        return sycl_timeline_mode::OFF;
    }
    if (equals_ignore_case(mode, "summary") || equals_ignore_case(mode, "1") || equals_ignore_case(mode, "on") ||
        equals_ignore_case(mode, "true") || equals_ignore_case(mode, "yes") || equals_ignore_case(mode, "enabled")) {
        return sycl_timeline_mode::SUMMARY;
    }
    if (equals_ignore_case(mode, "timeline")) {
        return sycl_timeline_mode::TIMELINE;
    }
    if (equals_ignore_case(mode, "timeline+events") || equals_ignore_case(mode, "events")) {
        return sycl_timeline_mode::TIMELINE_EVENTS;
    }
    return sycl_timeline_mode::OFF;
}

int parse_int_or_default(const char * value, int default_value, int min_value) {
    const char * first = value;
    if (first == nullptr) {
        return default_value;
    }
    while (*first != '\0' && is_space(*first)) {
        ++first;
    }
    if (*first == '\0') {
        return default_value;
    }

    char * end        = nullptr;
    errno             = 0;
    const long parsed = std::strtol(first, &end, 10);
    if (end == first || errno == ERANGE || parsed < min_value || parsed > INT_MAX) {
        return default_value;
    }
    while (*end != '\0' && is_space(*end)) {
        ++end;
    }
    if (*end != '\0') {
        return default_value;
    }
    return static_cast<int>(parsed);
}

sycl_timeline_config read_env_config() {
    return sycl_timeline_config_from_values(std::getenv("GGML_SYCL_TIMELINE"), std::getenv("GGML_SYCL_TIMELINE_OUTPUT"),
                                            std::getenv("GGML_SYCL_TIMELINE_TOKEN_START"),
                                            std::getenv("GGML_SYCL_TIMELINE_TOKEN_COUNT"),
                                            std::getenv("GGML_SYCL_TIMELINE_MAX_EVENTS"));
}

const sycl_timeline_config & current_config(sycl_timeline_state & state) {
    if (state.test_config_enabled) {
        return state.test_config;
    }
    if (!state.env_config_loaded) {
        state.env_config        = read_env_config();
        state.env_config_loaded = true;
    }
    return state.env_config;
}

}  // namespace

bool sycl_timeline_enabled_from_env(const char * value) {
    return parse_mode(value) != sycl_timeline_mode::OFF;
}

bool sycl_timeline_enabled() {
    sycl_timeline_state &       state = get_timeline_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    return current_config(state).enabled;
}

sycl_timeline_config sycl_timeline_config_from_env() {
    return read_env_config();
}

sycl_timeline_config sycl_timeline_config_from_values(const char * mode,
                                                      const char * output,
                                                      const char * token_start,
                                                      const char * token_count,
                                                      const char * max_events) {
    sycl_timeline_config cfg;

    cfg.mode        = parse_mode(mode);
    cfg.enabled     = cfg.mode != sycl_timeline_mode::OFF;
    cfg.output_path = output != nullptr ? output : "";
    cfg.token_start = parse_int_or_default(token_start, cfg.token_start, 0);
    cfg.token_count = parse_int_or_default(token_count, cfg.token_count, 0);
    cfg.max_events  = parse_int_or_default(max_events, cfg.max_events, 0);

    return cfg;
}

void sycl_timeline_reset_for_tests() {
    sycl_timeline_state &       state = get_timeline_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    state.env_config_loaded   = false;
    state.env_config          = {};
    state.test_config_enabled = false;
    state.test_config         = {};
}

void sycl_timeline_set_config_for_tests(const sycl_timeline_config & cfg) {
    sycl_timeline_state &       state = get_timeline_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    state.test_config_enabled = true;
    state.test_config         = cfg;
}

}  // namespace ggml_sycl
