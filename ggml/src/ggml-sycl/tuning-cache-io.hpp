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

#include <algorithm>
#include <cctype>
#include <climits>
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
#    include <process.h>
#else
#    include <unistd.h>
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

// Portable getpid (Windows: _getpid in <process.h>; POSIX: getpid in
// <unistd.h>) -- gives each process's atomic-write temp file a unique name
// (llama.cpp-7n6n): two processes writing the SAME device's cache
// concurrently used to interleave into one shared "<path>.tmp", so whichever
// renamed last could win with a partially-written file. A per-pid suffix
// makes that impossible; the pre-existing atomic rename-into-place still
// makes any ONE writer's own result crash-safe. A concurrent writer can
// still lose an entry the OTHER writer had (last rename wins) -- accepted,
// since that costs one extra ladder revalidation on the next start, never a
// wrong choice (see ubatch-tuning-cache.cpp's store for the same tradeoff).
inline long sycl_tuning_getpid() {
#ifdef _WIN32
    return static_cast<long>(_getpid());
#else
    return static_cast<long>(getpid());
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
// v3 (llama.cpp-3aos): adds UbatchCacheKey::kv_unified. A v2 entry parsed
// under v3 would have this field default to false via ubatch_key_from_json()
// (no "kv_unified" key present), which could falsely MATCH a real
// kv_unified=false query even though the entry was never validated against
// a kv_unified-aware KV-sizing formula -- the same silent-wrong-match shape
// the v1->v2 bump already fixed for n_seq_max/type_k/type_v/device_set_hash.
// Bumping the shared version rejects every v2 (and v1) file of either kind
// cleanly, same mechanism as above.
// v4 (llama.cpp-uajm): adds UbatchCacheKey::swa_full. Same shape again: a
// v3 entry has no "swa_full" key, would parse as swa_full=false, and could
// then match a real swa_full=false lookup (or be the only entry consulted
// for a swa_full=true raw-API context) although it was written before SWA
// layers were sized by that flag at all.
// v5 (llama.cpp-1oa3): UbatchCacheKey::device_key names every device whose
// budget decides the fit -- including a GPU the scheduler hides but the
// placement planner uses, marked "hidden:" -- with each one's budget
// percentage and external headroom, plus the multi-GPU placement knobs that
// are set, and device_set_hash (an index hash over the scheduler-visible
// backends only) is gone. A v4 entry's device_key names one card, so a v4
// B70-alone entry would still be read back for a collapsed level_zero:0,1
// split; bumping rejects every v4 file instead.
constexpr int CACHE_VERSION = 5;

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

// Parse integer value from JSON at key position. Accumulates the magnitude
// with an overflow check before each multiply-add (reject a digit whenever
// `val > (INT_MAX - digit) / 10`, the same test parse_u64() below uses
// against UINT64_MAX) rather than a fixed digit-count cap -- a fixed cap of
// N digits rejects some in-range N-digit values while accepting some
// out-of-range ones, since digit count alone doesn't determine whether a
// value overflows. Once a digit would overflow, accumulation stops for good
// (a later digit is still consumed, so the scan position stays correct for
// whatever comes after, but never accumulated again, even if a later digit
// alone would not itself have overflowed) -- this pins the result at
// whatever leading digits were seen before the overflow point, which is
// safe and reliably fails to match any real key a caller would ever look up
// with.
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
    bool overflowed = false;
    while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) {
        int digit = json[pos] - '0';
        if (!overflowed) {
            if (val > (INT_MAX - digit) / 10) {
                overflowed = true;
            } else {
                val = val * 10 + digit;
            }
        }
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

// Escape a string for embedding as a JSON string value: backslash, double
// quote, and control characters (< 0x20) as \uXXXX.
// parse_string() below already un-escapes on load; WITHOUT this, a
// model_name/device_key/reason containing a `"` or `\` (e.g. a Windows path
// "C:\models\foo.gguf", or a model name containing a quote) round-trips to
// a DIFFERENT string, so the stored key never matches the looked-up key on
// the next start -- no hit ever, and a new entry is appended on every run.
inline std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// Parse string value from JSON at key position. Handles the standard JSON
// escapes (\", \\, \/, \b, \f, \n, \r, \t) and \uXXXX --
// json_escape() above emits \uXXXX for every control character, so this
// must be able to read it back; a malformed \u sequence falls back to the
// pre-existing lenient behaviour of dropping the backslash and keeping the
// next character literally, so a hand-edited or partially-corrupted file
// still loads something rather than truncating the string at that point).
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
            char esc = json[pos + 1];
            if (esc == 'u' && pos + 5 < json.size()) {
                unsigned int code = 0;
                bool         ok   = true;
                for (int i = 0; i < 4 && ok; ++i) {
                    char h = json[pos + 2 + i];
                    code <<= 4;
                    if (h >= '0' && h <= '9') {
                        code |= static_cast<unsigned int>(h - '0');
                    } else if (h >= 'a' && h <= 'f') {
                        code |= static_cast<unsigned int>(h - 'a' + 10);
                    } else if (h >= 'A' && h <= 'F') {
                        code |= static_cast<unsigned int>(h - 'A' + 10);
                    } else {
                        ok = false;
                    }
                }
                if (ok) {
                    // UTF-8 encode; our own json_escape() only ever emits
                    // \u for control chars < 0x20 (single-byte), but encode
                    // the full BMP range correctly for a foreign/hand-edited
                    // file (no surrogate-pair handling -- not needed for
                    // anything this store writes itself).
                    if (code < 0x80) {
                        result += static_cast<char>(code);
                    } else if (code < 0x800) {
                        result += static_cast<char>(0xC0 | (code >> 6));
                        result += static_cast<char>(0x80 | (code & 0x3F));
                    } else {
                        result += static_cast<char>(0xE0 | (code >> 12));
                        result += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                        result += static_cast<char>(0x80 | (code & 0x3F));
                    }
                    pos += 6;
                    continue;
                }
                // Malformed \u -- fall through to the generic lenient case.
            } else {
                switch (esc) {
                    case '"':
                        result += '"';
                        pos += 2;
                        continue;
                    case '\\':
                        result += '\\';
                        pos += 2;
                        continue;
                    case '/':
                        result += '/';
                        pos += 2;
                        continue;
                    case 'b':
                        result += '\b';
                        pos += 2;
                        continue;
                    case 'f':
                        result += '\f';
                        pos += 2;
                        continue;
                    case 'n':
                        result += '\n';
                        pos += 2;
                        continue;
                    case 'r':
                        result += '\r';
                        pos += 2;
                        continue;
                    case 't':
                        result += '\t';
                        pos += 2;
                        continue;
                    default:
                        break;
                }
            }
            // Generic lenient fallback (pre-existing behaviour): drop the
            // backslash, keep the next character literally.
            pos++;
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

// One participating device's budget-relevant identity. The device name fixes
// the card's total VRAM, the driver version invalidates an entry across a
// driver upgrade, and budget_pct/external_headroom are the two stable inputs
// of that device's resolved VRAM budget (vram_budget_authority,
// unified-cache.hpp). The third input, free VRAM at init, is deliberately
// left out: it moves with every other tenant on the card, so keying on it
// would make nearly every start a miss.
struct UbatchDeviceIdentity {
    std::string device_name;  // raw; sanitized when composed into the key
    std::string driver_version;
    int         budget_pct        = 100;
    uint64_t    external_headroom = 0;  // bytes
};

// The device topology a context runs on, as indices into
// ggml_sycl_info().devices[].
struct UbatchDeviceTopology {
    // Every SYCL backend the context has, in order.
    std::vector<int> scheduler_devices;
    // ggml_sycl_info().device_count and total_gpu_count. A multi-GPU run
    // without GGML_SYCL_SPLIT_RATIO/TENSOR_SPLIT exposes only device 0 to the
    // scheduler, so devices [scheduler_visible_count, total_gpu_count) are
    // hidden from the context's backend list.
    int              scheduler_visible_count = 0;
    int              total_gpu_count         = 0;
    // Whether the placement planner may put layers, KV or experts on those
    // hidden GPUs (ggml_backend_sycl_moe_multi_gpu_requested(), the gate the
    // multi-device plan itself uses).
    bool             hidden_gpus_participate = false;
    // The multi-GPU placement knobs that are set (GGML_SYCL_MULTI_GPU_MODE,
    // GGML_SYCL_SPLIT_RATIO, GGML_SYCL_TENSOR_SPLIT), as "name=value" joined
    // by ';'; empty when none is. They change how work is divided across the
    // same devices, and so which n_ubatch fits.
    std::string      placement_config;
};

// The devices whose budgets decide which n_ubatch fits: the scheduler devices
// in order, then (when the planner may use them) each hidden physical GPU in
// index order. Keying on the scheduler devices alone made a collapsed
// level_zero:0,1 split and level_zero:0 alone the same set, [0].
inline std::vector<int> ubatch_participating_devices(const UbatchDeviceTopology & topo) {
    std::vector<int> devices = topo.scheduler_devices;
    if (!topo.hidden_gpus_participate) {
        return devices;
    }
    for (int d = topo.scheduler_visible_count; d < topo.total_gpu_count; ++d) {
        if (std::find(devices.begin(), devices.end(), d) == devices.end()) {
            devices.push_back(d);
        }
    }
    return devices;
}

// Compose the participating device set's key:
// "<sanitized name>@<driver>/pct=<p>/headroom=<bytes>" per device, in order,
// joined by ','. A device the scheduler hides is prefixed "hidden:": a
// collapsed level_zero:0,1 split (scheduler [0], device 1 hidden) and a
// SPLIT_RATIO/TENSOR_SPLIT split (scheduler [0,1]) have the same devices but
// different demand, since only the latter puts compute buffers on device 1.
// A non-empty placement_config is appended after '|'. `identities` is
// indexed by device index. Returns an empty string when a participating
// index has no identity, which the caller treats as "no key" rather than
// composing one that under-describes the set.
inline std::string ubatch_device_set_key(const UbatchDeviceTopology &              topo,
                                         const std::vector<UbatchDeviceIdentity> & identities) {
    const std::vector<int> devices = ubatch_participating_devices(topo);
    std::string            key;
    for (int d : devices) {
        if (d < 0 || static_cast<size_t>(d) >= identities.size()) {
            return std::string();
        }
        const UbatchDeviceIdentity & id = identities[static_cast<size_t>(d)];
        if (!key.empty()) {
            key += ',';
        }
        const bool hidden =
            std::find(topo.scheduler_devices.begin(), topo.scheduler_devices.end(), d) == topo.scheduler_devices.end();
        if (hidden) {
            key += "hidden:";
        }
        key += sanitize_device_name(id.device_name) + "@" + id.driver_version +
               "/pct=" + std::to_string(id.budget_pct) + "/headroom=" + std::to_string(id.external_headroom);
    }
    if (!key.empty() && !topo.placement_config.empty()) {
        key += "|" + topo.placement_config;
    }
    return key;
}

struct UbatchCacheKey {
    // ubatch_device_set_key() over every participating device, in order:
    // each one's sanitized name, driver version, budget percentage and
    // external headroom, whether the scheduler hides it, and the multi-GPU
    // placement knobs. The name and driver are queried at device init
    // (ggml-sycl.cpp's ggml_sycl_info() builder) and carried in
    // sycl_device_info next to device_name (common.hpp). No PCI id: it
    // moves across boots on this host (see CLAUDE.md's device-topology
    // note), so it cannot be part of a stable key.
    std::string device_key;
    // GGUF general.name (llama_model::name) + llama_model::size()'s total
    // tensor bytes + a 64-bit FNV-1a hash over every tensor's (name, byte
    // size) -- the same model file copied elsewhere hits; a re-quantised
    // file misses (its tensor byte sizes change).
    std::string model_name;
    uint64_t    model_size      = 0;
    uint64_t    model_hash      = 0;
    uint32_t    n_ctx           = 0;
    uint32_t    n_batch         = 0;
    bool        flash_attn      = false;
    // llama.cpp-7n6n: three key omissions that all land in the same
    // direction as the sticky cache-hit bug (a too-small cached value
    // pinned forever) -- a shape change in any of these can change which
    // candidates the ladder accepts without changing anything the key used
    // to track.
    uint32_t    n_seq_max       = 0;  // cparams.n_seq_max, passed to the probe on every candidate
    int32_t     type_k          = 0;  // params.type_k -- KV element type drives would_demote_kv
    int32_t     type_v          = 0;  // params.type_v -- ditto
    // llama.cpp-3aos: cparams.kv_unified -- once KV sizing depends on it
    // (kv_layer_bytes_for_kind(), unified-cache.hpp), two contexts
    // differing only in this flag need different auto n_ubatch candidates
    // and must not share one cache entry (CACHE_VERSION bumped to 3 above
    // so a pre-existing v2 entry, which has no opinion on this field,
    // cannot silently match either).
    bool        kv_unified      = false;
    // llama.cpp-uajm: llama_context_params::swa_full -- an SWA layer's KV
    // bytes are the window's with it false and a FULL layer's with it true
    // (kv_layer_bytes_for_kind(), unified-cache.hpp), so a CLI run
    // (common's default false) and a raw-API context
    // (llama_context_default_params()'s true) have different KV demand and
    // must not share one cache entry (CACHE_VERSION bumped to 4 above so a
    // v3 entry, which has no opinion on this field, cannot silently match).
    bool        swa_full        = false;

    bool operator==(const UbatchCacheKey & other) const {
        return device_key == other.device_key && model_name == other.model_name && model_size == other.model_size &&
               model_hash == other.model_hash && n_ctx == other.n_ctx && n_batch == other.n_batch &&
               flash_attn == other.flash_attn && n_seq_max == other.n_seq_max && type_k == other.type_k &&
               type_v == other.type_v && kv_unified == other.kv_unified && swa_full == other.swa_full;
    }

    bool operator!=(const UbatchCacheKey & other) const { return !(*this == other); }
};

struct UbatchCacheEntry {
    UbatchCacheKey key;
    uint32_t       n_ubatch = 0;
    // llama.cpp-7n6n: the REAL outcome the value was
    // stored with -- one of the eight sycl_select_auto_ubatch() stop
    // reasons, MINUS "cached" (a hit never re-stores an unchanged outcome;
    // see the store's own doc comment, ggml-sycl.h) and minus "transaction
    // busy"/"not the published model" (pure races the caller explicitly
    // does not persist). A future lookup uses this to tell a TERMINAL
    // outcome ("ladder exhausted", "MoE GPU routing ceiling" -- nothing
    // above this value was ever going to fit) from one that merely lost a
    // transient race, in which case the ladder resumes above the cached
    // value instead of trusting it forever.
    std::string    reason;
    std::string    created;  // ISO 8601, UTC (e.g. "2026-09-11T12:34:56Z")
};

// Parse an unsigned 64-bit value from JSON at key position. parse_int()
// above is capped at a plain `int`; model_size and model_hash need the full
// 64-bit range (model_hash in particular is an arbitrary FNV-1a hash, not a
// count, so roughly half of all possible values are >= 10^19, i.e. the full
// 20 digits UINT64_MAX itself needs). A FIXED 19-digit cap -- the largest
// digit count that can never overflow regardless of which digits they are
// -- used to reject every one of those legitimate 20-digit hashes: the
// parsed model_hash then differed from the one that was stored, so the key
// could never match its own entry again (every start missed). Digit COUNT
// alone cannot tell a valid 20-digit value from an overflowing one, so this
// checks for overflow before each multiply-add instead (reject a digit
// whenever `val > (UINT64_MAX - digit) / 10`); once a digit would overflow,
// accumulation stops for good (a later digit is still consumed, so the scan
// position stays correct for whatever comes after, but never accumulated
// again, even if a later digit alone would not itself have overflowed) --
// this pins the result at whatever leading digits were seen before the
// overflow point, which is safe and reliably fails to match any real key.
inline uint64_t parse_u64(const std::string & json, const std::string & key) {
    std::string search = "\"" + key + "\":";
    size_t      pos    = json.find(search);
    if (pos == std::string::npos) {
        return 0;
    }

    pos += search.size();
    pos = skip_whitespace(json, pos);

    uint64_t val        = 0;
    bool     overflowed = false;
    while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) {
        uint64_t digit = static_cast<uint64_t>(json[pos] - '0');
        if (!overflowed) {
            if (val > (UINT64_MAX - digit) / 10) {
                overflowed = true;
            } else {
                val = val * 10 + digit;
            }
        }
        pos++;
    }
    return val;
}

// Serialize a UbatchCacheKey's fields into a JSON object body (no enclosing
// braces -- the caller wraps it alongside its own n_ubatch/reason/created
// fields, mirroring how entry_to_json() above nests params_to_json()).
// device_key and model_name are free-form strings (a Windows path, or a
// model name containing a quote, are both real inputs) and so are routed
// through json_escape(); every other field here is
// numeric/boolean and needs no escaping.
inline std::string ubatch_key_to_json(const UbatchCacheKey & k) {
    std::ostringstream ss;
    ss << "\"device_key\":\"" << json_escape(k.device_key) << "\","
       << "\"model_name\":\"" << json_escape(k.model_name) << "\","
       << "\"model_size\":" << k.model_size << ","
       << "\"model_hash\":" << k.model_hash << ","
       << "\"n_ctx\":" << k.n_ctx << ","
       << "\"n_batch\":" << k.n_batch << ","
       << "\"flash_attn\":" << (k.flash_attn ? "true" : "false") << ","
       << "\"n_seq_max\":" << k.n_seq_max << ","
       << "\"type_k\":" << k.type_k << ","
       << "\"type_v\":" << k.type_v << ","
       << "\"kv_unified\":" << (k.kv_unified ? "true" : "false") << ","
       << "\"swa_full\":" << (k.swa_full ? "true" : "false");
    return ss.str();
}

inline UbatchCacheKey ubatch_key_from_json(const std::string & json) {
    UbatchCacheKey k;
    k.device_key      = parse_string(json, "device_key");
    k.model_name      = parse_string(json, "model_name");
    k.model_size      = parse_u64(json, "model_size");
    k.model_hash      = parse_u64(json, "model_hash");
    k.n_ctx           = static_cast<uint32_t>(parse_int(json, "n_ctx"));
    k.n_batch         = static_cast<uint32_t>(parse_int(json, "n_batch"));
    k.flash_attn      = parse_bool(json, "flash_attn");
    k.n_seq_max       = static_cast<uint32_t>(parse_int(json, "n_seq_max"));
    k.type_k          = static_cast<int32_t>(parse_int(json, "type_k"));
    k.type_v          = static_cast<int32_t>(parse_int(json, "type_v"));
    k.kv_unified      = parse_bool(json, "kv_unified");
    k.swa_full        = parse_bool(json, "swa_full");
    return k;
}

// `reason` is free-form (escaped for the same reason
// device_key/model_name are above).
inline std::string ubatch_entry_to_json(const UbatchCacheEntry & e) {
    std::ostringstream ss;
    ss << "{" << ubatch_key_to_json(e.key) << ","
       << "\"n_ubatch\":" << e.n_ubatch << ","
       << "\"reason\":\"" << json_escape(e.reason) << "\","
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

// Cap the entry count before a save (llama.cpp-7n6n): unbounded growth
// is plausible across many models/context shapes on one device, and
// `created` (an ISO-8601 timestamp, so plain string comparison sorts
// correctly) was written to every entry but never read until this. A no-op
// when `entries.size() <= max_entries`; otherwise keeps only the
// `max_entries` most recently created entries. Lives here (not in
// ubatch-tuning-cache.cpp, the only current caller) so it is testable from
// a host-only unit test with no SYCL device involved.
inline void cap_ubatch_cache_entries(std::vector<UbatchCacheEntry> & entries, size_t max_entries = 64) {
    if (entries.size() <= max_entries) {
        return;
    }
    std::sort(entries.begin(), entries.end(),
              [](const UbatchCacheEntry & a, const UbatchCacheEntry & b) { return a.created > b.created; });
    entries.resize(max_entries);
}

// Save every entry for one device, atomically (write to .tmp, then rename;
// same discipline as save_cache() above). Returns true on success, false on
// failure (e.g. an unwritable directory) -- never throws.
inline bool save_ubatch_cache(const std::string &                   cache_dir,
                              const std::string &                   device_name,
                              const std::vector<UbatchCacheEntry> & entries) {
    create_dir_recursive(cache_dir);

    std::string path      = get_ubatch_cache_file(cache_dir, device_name);
    // Per-pid temp name (llama.cpp-7n6n): see sycl_tuning_getpid()'s
    // own comment for why a shared "<path>.tmp" is unsafe under concurrent
    // writers.
    std::string temp_path = path + "." + std::to_string(sycl_tuning_getpid()) + ".tmp";

    std::ofstream f(temp_path);
    if (!f) {
        return false;
    }

    f << "{\n";
    f << "  \"version\": " << CACHE_VERSION << ",\n";
    f << "  \"device\": \"" << json_escape(device_name) << "\",\n";
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
    // Per-pid temp name (llama.cpp-7n6n): see sycl_tuning_getpid()'s
    // own comment for why a shared "<path>.tmp" is unsafe under concurrent
    // writers.
    std::string temp_path = path + "." + std::to_string(sycl_tuning_getpid()) + ".tmp";

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
    f << "  \"device\": \"" << json_escape(device_name) << "\",\n";
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
