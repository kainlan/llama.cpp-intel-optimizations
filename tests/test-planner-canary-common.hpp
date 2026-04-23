// Shared utilities for planner pre-flight canaries.
// Each canary writes a findings.md (human-readable) and a findings.json
// (machine-readable, appended to docs/plans/data/planner-canaries/summary.md).

#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace planner_canary {

// Result: PASS / FAIL / INCONCLUSIVE. INCONCLUSIVE is for "hardware couldn't
// run this canary" (e.g., only one GPU visible, D0.3 needs two).
enum class status { PASS, FAIL, INCONCLUSIVE };

inline const char * status_str(status s) {
    switch (s) {
        case status::PASS:         return "PASS";
        case status::FAIL:         return "FAIL";
        case status::INCONCLUSIVE: return "INCONCLUSIVE";
    }
    return "?";
}

struct findings {
    std::string canary_id;  // "D0.1", "D0.2", ...
    status      result = status::FAIL;
    std::string summary;    // one-line human-readable summary
    std::vector<std::pair<std::string, std::string>> kv;  // evidence
    std::string recommendation;  // one of: "A3a-approach-validated",
                                 // "switch-to-plan-B", "C2-keying-change-needed", etc.
};

// Minimal JSON string escape: handles the RFC 8259 mandatory escapes plus
// control chars below 0x20 as \u00XX. Does not attempt UTF-8 surrogate
// splitting — inputs are ASCII/UTF-8 and any multi-byte UTF-8 sequences are
// passed through verbatim, which is valid per RFC 8259.
inline std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char) c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char) c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// Write a human-readable findings document.
inline void write_markdown(const findings & f, const std::string & out_path) {
    std::ofstream out(out_path);
    assert(out.is_open() && "findings markdown file open failed");
    out << "# " << f.canary_id << " — " << status_str(f.result) << "\n\n";
    out << "**Summary**: " << f.summary << "\n\n";
    out << "**Recommendation**: " << f.recommendation << "\n\n";
    out << "## Evidence\n\n";
    for (const auto & p : f.kv) {
        out << "- **" << p.first << "**: " << p.second << "\n";
    }
    out << "\n";
}

// Write a machine-readable JSON document (no external deps — hand-rolled).
inline void write_json(const findings & f, const std::string & out_path) {
    std::ofstream out(out_path);
    assert(out.is_open() && "findings json file open failed");
    out << "{\n";
    out << "  \"canary_id\": \""      << json_escape(f.canary_id)      << "\",\n";
    out << "  \"result\": \""         << status_str(f.result)          << "\",\n";
    out << "  \"summary\": \""        << json_escape(f.summary)        << "\",\n";
    out << "  \"recommendation\": \"" << json_escape(f.recommendation) << "\",\n";
    out << "  \"evidence\": {\n";
    for (size_t i = 0; i < f.kv.size(); ++i) {
        out << "    \"" << json_escape(f.kv[i].first) << "\": \""
                        << json_escape(f.kv[i].second) << "\"";
        if (i + 1 < f.kv.size()) out << ",";
        out << "\n";
    }
    out << "  }\n";
    out << "}\n";
}

// Convenient kv setter.
inline void add(findings & f, const std::string & k, const std::string & v) {
    f.kv.emplace_back(k, v);
}

// Default model paths (override via env vars MISTRAL_PATH / GPTOSS_PATH if needed).
inline std::string mistral_path() {
    const char * p = std::getenv("MISTRAL_PATH");
    return p ? p : "/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf";
}
inline std::string gptoss_path() {
    const char * p = std::getenv("GPTOSS_PATH");
    return p ? p : "/Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf";
}

}  // namespace planner_canary
