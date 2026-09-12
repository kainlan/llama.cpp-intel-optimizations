//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_TUNING_CACHE_IO_HPP
#define GGML_SYCL_TUNING_CACHE_IO_HPP

// Tuning Cache I/O: Persistent storage for auto-tuning results
//
// This header provides:
// - JSON serialization/deserialization for TunedParams
// - Atomic file writes with versioning for crash safety
// - XDG-compliant cache directory management
// - Device-specific cache files for multi-GPU support

#include "tuning-engine.hpp"

#include <sys/stat.h>

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#    include <direct.h>
#endif

namespace ggml_sycl_tuning {

// Portable single-directory create (POSIX mkdir takes a mode; Windows _mkdir
// does not). Errors (e.g. already-exists) are ignored by the callers.
inline int sycl_tuning_mkdir(const char * path) {
#ifdef _WIN32
    return _mkdir(path);
#else
    return mkdir(path, 0755);
#endif
}

// =============================================================================
// Cache Version: Increment when format changes (for migration support)
// =============================================================================
// v2 (llama.cpp-7n6n): adds the "ubatch" entry kind below (UbatchCacheKey /
// UbatchCacheEntry / save_ubatch_cache / load_ubatch_cache), alongside the
// pre-existing matmul dispatch-tuning entry kind (TuningKey / TuningEntry /
// TunedParams / save_cache / load_cache) this header already implemented.
// The two kinds are otherwise independent -- different structs, different
// per-device files (get_cache_file() vs get_ubatch_cache_file(), the latter
// suffixed "-ubatch" so the two never collide) -- but share this constant,
// so a v1 file of EITHER kind is rejected by the version check already in
// load_cache()/load_ubatch_cache() below and gets rewritten wholesale on the
// next store; no separate migration code was needed for that.
constexpr int CACHE_VERSION = 2;

// =============================================================================
// Path Utilities
// =============================================================================

// Get cache directory path following XDG Base Directory spec
// Priority: $XDG_CACHE_HOME > $HOME/.cache > /tmp
inline std::string get_cache_dir() {
    const char* xdg = std::getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0]) {
        return std::string(xdg) + "/llama.cpp/sycl-tuning";
    }
    const char* home = std::getenv("HOME");
    if (home && home[0]) {
        return std::string(home) + "/.cache/llama.cpp/sycl-tuning";
    }
    return "/tmp/llama.cpp/sycl-tuning";
}

// Sanitize device name for use as filename
// Replaces unsafe characters with underscores
inline std::string sanitize_device_name(const std::string& device_name) {
    std::string safe_name;
    safe_name.reserve(device_name.size());
    for (char c : device_name) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') {
            safe_name += c;
        } else if (c == ' ') {
            safe_name += '_';
        }
        // Skip other characters (slashes, colons, etc.)
    }
    return safe_name;
}

// Get cache file path for a specific device
inline std::string get_cache_file(const std::string& device_name) {
    return get_cache_dir() + "/" + sanitize_device_name(device_name) + ".json";
}

// Create directory recursively (mkdir -p equivalent)
inline bool create_dir_recursive(const std::string& path) {
    std::string current;
    for (size_t i = 0; i < path.size(); i++) {
        current += path[i];
        if (path[i] == '/' && i > 0) {
            // Create intermediate directory
            sycl_tuning_mkdir(current.c_str());
        }
    }
    // Create final directory
    if (!current.empty() && current.back() != '/') {
        sycl_tuning_mkdir(current.c_str());
    }
    return true;
}

// =============================================================================
// TuningKey Serialization
// =============================================================================

// Convert TuningKey to string representation for JSON key
inline std::string key_to_string(const TuningKey& key) {
    std::ostringstream ss;
    ss << key.quant_type << "_"
       << static_cast<int>(key.batch_bucket) << "_"
       << key.K << "_"
       << key.N;
    return ss.str();
}

// Parse TuningKey from string representation
inline TuningKey key_from_string(const std::string& str) {
    TuningKey key;
    // Format: "quant_bucket_K_N"
    size_t pos1 = str.find('_');
    size_t pos2 = str.find('_', pos1 + 1);
    size_t pos3 = str.find('_', pos2 + 1);

    if (pos1 != std::string::npos && pos2 != std::string::npos && pos3 != std::string::npos) {
        key.quant_type = std::stoi(str.substr(0, pos1));
        key.batch_bucket = static_cast<BatchBucket>(std::stoi(str.substr(pos1 + 1, pos2 - pos1 - 1)));
        key.K = std::stoi(str.substr(pos2 + 1, pos3 - pos2 - 1));
        key.N = std::stoi(str.substr(pos3 + 1));
    }
    return key;
}

// =============================================================================
// TunedParams JSON Serialization
// =============================================================================

// Serialize TunedParams to JSON object string
inline std::string params_to_json(const TunedParams& p) {
    std::ostringstream ss;
    ss << "{"
       << "\"tile_m\":" << p.tile_m << ","
       << "\"tile_n\":" << p.tile_n << ","
       << "\"tile_k\":" << p.tile_k << ","
       << "\"workgroup_size\":" << p.workgroup_size << ","
       << "\"slm_kb\":" << static_cast<int>(p.slm_kb) << ","
       << "\"prefetch_depth\":" << static_cast<int>(p.prefetch_depth) << ","
       << "\"use_dpas\":" << (p.use_dpas ? "true" : "false") << ","
       << "\"layout_mode\":" << static_cast<int>(p.layout_mode)
       << "}";
    return ss.str();
}

// =============================================================================
// Simple JSON Parsing Helpers
// =============================================================================

// Skip whitespace in JSON string
inline size_t skip_whitespace(const std::string& json, size_t pos) {
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        pos++;
    }
    return pos;
}

// Parse integer value from JSON at key position
inline int parse_int(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return 0;

    pos += search.size();
    pos = skip_whitespace(json, pos);

    int val = 0;
    bool neg = false;
    if (pos < json.size() && json[pos] == '-') {
        neg = true;
        pos++;
    }
    while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) {
        val = val * 10 + (json[pos] - '0');
        pos++;
    }
    return neg ? -val : val;
}

// Parse boolean value from JSON at key position
inline bool parse_bool(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return false;

    pos += search.size();
    pos = skip_whitespace(json, pos);

    // Check for "true" (case-sensitive)
    if (pos + 4 <= json.size() && json.substr(pos, 4) == "true") {
        return true;
    }
    return false;
}

// Parse string value from JSON at key position
inline std::string parse_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos += search.size();
    pos = skip_whitespace(json, pos);

    if (pos >= json.size() || json[pos] != '"') return "";
    pos++;  // Skip opening quote

    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;  // Skip escape character
        }
        result += json[pos];
        pos++;
    }
    return result;
}

// Parse float value from JSON at key position
inline float parse_float(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return 0.0f;

    pos += search.size();
    pos = skip_whitespace(json, pos);

    size_t end_pos = pos;
    while (end_pos < json.size() &&
           (std::isdigit(static_cast<unsigned char>(json[end_pos])) ||
            json[end_pos] == '.' || json[end_pos] == '-' ||
            json[end_pos] == 'e' || json[end_pos] == 'E' || json[end_pos] == '+')) {
        end_pos++;
    }

    if (end_pos > pos) {
        return std::stof(json.substr(pos, end_pos - pos));
    }
    return 0.0f;
}

// Deserialize TunedParams from JSON object string
inline TunedParams params_from_json(const std::string& json) {
    TunedParams p;
    p.tile_m = static_cast<uint16_t>(parse_int(json, "tile_m"));
    p.tile_n = static_cast<uint16_t>(parse_int(json, "tile_n"));
    p.tile_k = static_cast<uint16_t>(parse_int(json, "tile_k"));
    p.workgroup_size = static_cast<uint16_t>(parse_int(json, "workgroup_size"));
    p.slm_kb = static_cast<uint8_t>(parse_int(json, "slm_kb"));
    p.prefetch_depth = static_cast<uint8_t>(parse_int(json, "prefetch_depth"));
    p.use_dpas = parse_bool(json, "use_dpas");
    p.layout_mode = static_cast<uint8_t>(parse_int(json, "layout_mode"));
    return p;
}

// =============================================================================
// TuningEntry Serialization
// =============================================================================

// Serialize TuningEntry to JSON string
inline std::string entry_to_json(const TuningEntry& entry) {
    std::ostringstream ss;
    ss << "{"
       << "\"key\":\"" << key_to_string(entry.key) << "\","
       << "\"params\":" << params_to_json(entry.params) << ","
       << "\"measured_tflops\":" << entry.measured_tflops << ","
       << "\"timestamp\":" << entry.timestamp
       << "}";
    return ss.str();
}

// Deserialize TuningEntry from JSON string
inline TuningEntry entry_from_json(const std::string& json) {
    TuningEntry entry;
    entry.key = key_from_string(parse_string(json, "key"));

    // Find params object
    size_t params_pos = json.find("\"params\":");
    if (params_pos != std::string::npos) {
        size_t start = json.find('{', params_pos);
        if (start != std::string::npos) {
            int brace_count = 1;
            size_t end = start + 1;
            while (end < json.size() && brace_count > 0) {
                if (json[end] == '{') brace_count++;
                if (json[end] == '}') brace_count--;
                end++;
            }
            entry.params = params_from_json(json.substr(start, end - start));
        }
    }

    entry.measured_tflops = parse_float(json, "measured_tflops");
    entry.timestamp = static_cast<int64_t>(parse_int(json, "timestamp"));

    return entry;
}

// =============================================================================
// Ubatch Cache Entry (v2): persists the SYCL auto micro-batch selection
// trial's chosen n_ubatch (llama.cpp-7n6n, wiring llama.cpp-nphx Task 5).
// Orthogonal to the matmul dispatch-tuning TuningKey/TuningEntry/TunedParams
// model above: this entry kind has its own key composition (device+driver+
// model identity+context shape, not quant_type/batch_bucket/K/N) and its
// own per-device file (get_ubatch_cache_file(), suffixed "-ubatch.json" so
// it never collides with the matmul cache's <device>.json for the SAME
// device), but shares CACHE_VERSION and the same atomic-write /
// version-mismatch-rejects-and-rewrites discipline as save_cache()/
// load_cache() above.
// =============================================================================

struct UbatchCacheKey {
    // sanitize_device_name(name) + "@" + driver_version -- both queried at
    // device init (ggml-sycl.cpp's ggml_sycl_info() builder) and carried in
    // sycl_device_info next to device_name (common.hpp). No PCI id: it
    // moves across boots on this host (see CLAUDE.md's device-topology
    // note), so it cannot be part of a stable key.
    std::string device_key;
    // GGUF general.name (llama_model::name) + llama_model::size()'s total
    // tensor bytes + a 64-bit FNV-1a hash over every tensor's (name, byte
    // size) -- the same model file copied elsewhere hits; a re-quantised
    // file misses (its tensor byte sizes change).
    std::string model_name;
    uint64_t    model_size = 0;
    uint64_t    model_hash = 0;
    uint32_t    n_ctx      = 0;
    uint32_t    n_batch    = 0;
    bool        flash_attn = false;

    bool operator==(const UbatchCacheKey & other) const {
        return device_key == other.device_key && model_name == other.model_name && model_size == other.model_size &&
               model_hash == other.model_hash && n_ctx == other.n_ctx && n_batch == other.n_batch &&
               flash_attn == other.flash_attn;
    }

    bool operator!=(const UbatchCacheKey & other) const { return !(*this == other); }
};

struct UbatchCacheEntry {
    UbatchCacheKey key;
    uint32_t       n_ubatch = 0;
    // Free-form diagnostic (see ggml_backend_sycl_ubatch_cache_store()'s own
    // comment, ggml-sycl.h): today's only writer always passes "ladder" --
    // a cache hit never re-stores itself, so "cached" never actually reaches
    // this field despite being a valid stop-reason string on the caller's
    // own vocabulary. Reserved, not currently written.
    std::string    reason;
    std::string    created;  // ISO 8601, UTC (e.g. "2026-09-11T12:34:56Z")
};

// Parse an unsigned 64-bit value from JSON at key position. parse_int()
// above is capped at a plain `int`; model_size and model_hash need the full
// 64-bit range (model_hash in particular is an arbitrary hash, not a count).
inline uint64_t parse_u64(const std::string & json, const std::string & key) {
    std::string search = "\"" + key + "\":";
    size_t      pos    = json.find(search);
    if (pos == std::string::npos) {
        return 0;
    }

    pos += search.size();
    pos = skip_whitespace(json, pos);

    uint64_t val = 0;
    while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) {
        val = val * 10 + static_cast<uint64_t>(json[pos] - '0');
        pos++;
    }
    return val;
}

// Serialize a UbatchCacheKey's fields into a JSON object body (no enclosing
// braces -- the caller wraps it alongside its own n_ubatch/reason/created
// fields, mirroring how entry_to_json() above nests params_to_json()).
inline std::string ubatch_key_to_json(const UbatchCacheKey & k) {
    std::ostringstream ss;
    ss << "\"device_key\":\"" << k.device_key << "\","
       << "\"model_name\":\"" << k.model_name << "\","
       << "\"model_size\":" << k.model_size << ","
       << "\"model_hash\":" << k.model_hash << ","
       << "\"n_ctx\":" << k.n_ctx << ","
       << "\"n_batch\":" << k.n_batch << ","
       << "\"flash_attn\":" << (k.flash_attn ? "true" : "false");
    return ss.str();
}

inline UbatchCacheKey ubatch_key_from_json(const std::string & json) {
    UbatchCacheKey k;
    k.device_key = parse_string(json, "device_key");
    k.model_name = parse_string(json, "model_name");
    k.model_size = parse_u64(json, "model_size");
    k.model_hash = parse_u64(json, "model_hash");
    k.n_ctx      = static_cast<uint32_t>(parse_int(json, "n_ctx"));
    k.n_batch    = static_cast<uint32_t>(parse_int(json, "n_batch"));
    k.flash_attn = parse_bool(json, "flash_attn");
    return k;
}

inline std::string ubatch_entry_to_json(const UbatchCacheEntry & e) {
    std::ostringstream ss;
    ss << "{" << ubatch_key_to_json(e.key) << ","
       << "\"n_ubatch\":" << e.n_ubatch << ","
       << "\"reason\":\"" << e.reason << "\","
       << "\"created\":\"" << e.created << "\""
       << "}";
    return ss.str();
}

inline UbatchCacheEntry ubatch_entry_from_json(const std::string & json) {
    UbatchCacheEntry e;
    e.key      = ubatch_key_from_json(json);
    e.n_ubatch = static_cast<uint32_t>(parse_int(json, "n_ubatch"));
    e.reason   = parse_string(json, "reason");
    e.created  = parse_string(json, "created");
    return e;
}

// Cache file path for one device's ubatch entries, under `cache_dir` (the
// caller resolves `cache_dir` itself -- get_cache_dir() for the default XDG
// location, or a GGML_SYCL_TUNING_CACHE_DIR override; unlike get_cache_file()
// above, this is not hardwired to get_cache_dir(), so a caller-supplied
// override never has to fight this header's own XDG default). Suffixed
// "-ubatch" so this store never collides with the matmul dispatch-tuning
// cache, which already claims "<sanitized device name>.json" for the SAME
// physical device.
inline std::string get_ubatch_cache_file(const std::string & cache_dir, const std::string & device_name) {
    return cache_dir + "/" + sanitize_device_name(device_name) + "-ubatch.json";
}

// Save every entry for one device, atomically (write to .tmp, then rename;
// same discipline as save_cache() above). Returns true on success, false on
// failure (e.g. an unwritable directory) -- never throws.
inline bool save_ubatch_cache(const std::string &                   cache_dir,
                              const std::string &                   device_name,
                              const std::vector<UbatchCacheEntry> & entries) {
    create_dir_recursive(cache_dir);

    std::string path      = get_ubatch_cache_file(cache_dir, device_name);
    std::string temp_path = path + ".tmp";

    std::ofstream f(temp_path);
    if (!f) {
        return false;
    }

    f << "{\n";
    f << "  \"version\": " << CACHE_VERSION << ",\n";
    f << "  \"device\": \"" << device_name << "\",\n";
    f << "  \"entries\": [\n";
    for (size_t i = 0; i < entries.size(); ++i) {
        f << "    " << ubatch_entry_to_json(entries[i]);
        if (i + 1 < entries.size()) {
            f << ",";
        }
        f << "\n";
    }
    f << "  ]\n";
    f << "}\n";

    f.close();
    if (!f) {
        std::remove(temp_path.c_str());
        return false;
    }

    if (std::rename(temp_path.c_str(), path.c_str()) != 0) {
        std::remove(temp_path.c_str());
        return false;
    }

    return true;
}

// Load every entry for one device. Returns false (entries left empty) on a
// missing file, an unreadable file, or a version mismatch (a v1 -- or any
// other-version -- file is rejected exactly like load_cache() above, so it
// reads as a plain miss and gets rewritten wholesale on the next store).
// Never throws.
inline bool load_ubatch_cache(const std::string &             cache_dir,
                              const std::string &             device_name,
                              std::vector<UbatchCacheEntry> & entries) {
    entries.clear();

    std::string   path = get_ubatch_cache_file(cache_dir, device_name);
    std::ifstream f(path);
    if (!f) {
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    int version = parse_int(content, "version");
    if (version != CACHE_VERSION) {
        return false;
    }

    size_t entries_pos = content.find("\"entries\":");
    if (entries_pos == std::string::npos) {
        return false;
    }
    size_t array_start = content.find('[', entries_pos);
    if (array_start == std::string::npos) {
        return false;
    }

    size_t pos = array_start + 1;
    while (pos < content.size()) {
        while (pos < content.size() &&
               (std::isspace(static_cast<unsigned char>(content[pos])) || content[pos] == ',')) {
            pos++;
        }
        if (pos >= content.size() || content[pos] == ']') {
            break;
        }
        if (content[pos] == '{') {
            int    brace_count = 1;
            size_t start       = pos;
            pos++;
            while (pos < content.size() && brace_count > 0) {
                if (content[pos] == '{') {
                    brace_count++;
                }
                if (content[pos] == '}') {
                    brace_count--;
                }
                pos++;
            }
            entries.push_back(ubatch_entry_from_json(content.substr(start, pos - start)));
        } else {
            pos++;  // Skip unexpected character
        }
    }

    return true;
}

// =============================================================================
// Cache File I/O
// =============================================================================

// Save cache to file with atomic write (write to .tmp, then rename)
// Returns true on success, false on failure
template<typename IterableCache>
inline bool save_cache(const IterableCache& cache, const std::string& device_name) {
    std::string path = get_cache_file(device_name);
    std::string temp_path = path + ".tmp";

    // Ensure directory exists
    create_dir_recursive(get_cache_dir());

    // Open temp file for writing
    std::ofstream f(temp_path);
    if (!f) {
        return false;
    }

    // Write JSON header
    f << "{\n";
    f << "  \"version\": " << CACHE_VERSION << ",\n";
    f << "  \"device\": \"" << device_name << "\",\n";
    f << "  \"entries\": [\n";

    // Write entries
    bool first = true;
    cache.for_each([&](const TuningKey& key, const TuningEntry& entry) {
        if (!first) {
            f << ",\n";
        }
        f << "    " << entry_to_json(entry);
        first = false;
    });

    // Close JSON structure
    f << "\n  ]\n";
    f << "}\n";

    f.close();

    if (!f) {
        // Write failed, remove temp file
        std::remove(temp_path.c_str());
        return false;
    }

    // Atomic rename (POSIX guarantees atomicity)
    if (std::rename(temp_path.c_str(), path.c_str()) != 0) {
        std::remove(temp_path.c_str());
        return false;
    }

    return true;
}

// Load cache from file
// Returns true on success (cache populated), false on failure (cache unchanged)
inline bool load_cache(TuningCache& cache, const std::string& device_name) {
    std::string path = get_cache_file(device_name);
    std::ifstream f(path);
    if (!f) {
        return false;  // File doesn't exist or can't be opened
    }

    // Read entire file
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    f.close();

    // Verify version compatibility
    int version = parse_int(content, "version");
    if (version != CACHE_VERSION) {
        // Incompatible version - could implement migration here
        return false;
    }

    // Verify device name matches (optional sanity check)
    std::string stored_device = parse_string(content, "device");
    // Note: We allow loading even if device name differs slightly
    // (e.g., driver version changes in device name string)

    // Find entries array
    size_t entries_pos = content.find("\"entries\":");
    if (entries_pos == std::string::npos) {
        return false;
    }

    size_t array_start = content.find('[', entries_pos);
    if (array_start == std::string::npos) {
        return false;
    }

    // Parse each entry object in the array
    size_t pos = array_start + 1;
    while (pos < content.size()) {
        // Skip whitespace and commas
        while (pos < content.size() &&
               (std::isspace(static_cast<unsigned char>(content[pos])) || content[pos] == ',')) {
            pos++;
        }

        // Check for end of array
        if (pos >= content.size() || content[pos] == ']') {
            break;
        }

        // Find entry object
        if (content[pos] == '{') {
            int brace_count = 1;
            size_t start = pos;
            pos++;
            while (pos < content.size() && brace_count > 0) {
                if (content[pos] == '{') brace_count++;
                if (content[pos] == '}') brace_count--;
                pos++;
            }

            // Parse and insert entry
            std::string entry_json = content.substr(start, pos - start);
            TuningEntry entry = entry_from_json(entry_json);
            cache.insert(entry);
        } else {
            pos++;  // Skip unexpected character
        }
    }

    return true;
}

// Check if cache file exists for device
inline bool cache_exists(const std::string& device_name) {
    std::string path = get_cache_file(device_name);
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// Delete cache file for device
inline bool delete_cache(const std::string& device_name) {
    std::string path = get_cache_file(device_name);
    return std::remove(path.c_str()) == 0;
}

}  // namespace ggml_sycl_tuning

#endif  // GGML_SYCL_TUNING_CACHE_IO_HPP
