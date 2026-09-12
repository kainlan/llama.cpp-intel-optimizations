//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// llama.cpp-7n6n (wires nphx Task 5): the persisted SYCL auto n_ubatch
// tuning cache's four public entry points
// (ggml_backend_sycl_ubatch_cache_enabled/_path/_lookup/_store, declared in
// ggml-sycl.h). Thin over tuning-cache-io.hpp's UbatchCacheKey/
// UbatchCacheEntry/save_ubatch_cache/load_ubatch_cache -- see that header's
// "Ubatch Cache Entry (v2)" section for the on-disk format this file reads
// and writes. This TU owns:
//   - the two env-var accessors (GGML_SYCL_TUNING_CACHE,
//     GGML_SYCL_TUNING_CACHE_DIR), memoized the same way
//     unified_cache_auto_ubatch_enabled() is (unified-cache.cpp);
//   - composing the on-disk device_key (sanitize_device_name(name) + "@" +
//     driver_version) from ggml_sycl_info().devices[device] -- callers pass
//     only a logical device index, never the raw strings, so this
//     composition happens in exactly one place;
//   - the ISO-8601 timestamp stamped on every store.
//
// GGML_SYCL_TUNING_CACHE=0 disables both lookup and store: every call below
// then returns false without touching the filesystem, which is exactly how
// callers already tolerate a cold cache (a miss), so no separate "disabled"
// code path is needed at the call sites in src/llama-context.cpp -- only
// ggml_backend_sycl_ubatch_cache_enabled() (queried once, for the trial's
// own diagnostic WARN) tells the two states apart.

#include "common.hpp"
#include "ggml-sycl.h"
#include "tuning-cache-io.hpp"

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

using namespace ggml_sycl_tuning;

namespace {

// Memoized, WARN-at-most-once-per-process accessor for
// GGML_SYCL_TUNING_CACHE -- mirrors unified_cache_auto_ubatch_enabled()'s
// own convention (unified-cache.cpp) exactly: unset/empty/"1" -> enabled;
// "0" -> disabled; any other non-empty value -> enabled, with one WARN.
bool ubatch_tuning_cache_env_enabled() {
    static const bool value = [] {
        const char * env = std::getenv("GGML_SYCL_TUNING_CACHE");
        if (env == nullptr || env[0] == '\0') {
            return true;
        }
        if (std::strcmp(env, "0") == 0) {
            return false;
        }
        if (std::strcmp(env, "1") != 0) {
            GGML_LOG_WARN(
                "[SYCL-PLAN] GGML_SYCL_TUNING_CACHE=\"%s\" is not \"0\" or \"1\" -- ignoring it and "
                "leaving the persisted auto n_ubatch cache enabled\n",
                env);
        }
        return true;
    }();
    return value;
}

// Memoized accessor for GGML_SYCL_TUNING_CACHE_DIR. Empty means "not set" --
// callers fall back to tuning-cache-io.hpp's get_cache_dir() (XDG).
const std::string & ubatch_tuning_cache_dir_override() {
    static const std::string value = [] {
        const char * env = std::getenv("GGML_SYCL_TUNING_CACHE_DIR");
        return (env && env[0]) ? std::string(env) : std::string();
    }();
    return value;
}

std::string ubatch_tuning_cache_dir() {
    const std::string & override_dir = ubatch_tuning_cache_dir_override();
    return override_dir.empty() ? get_cache_dir() : override_dir;
}

// Resolve `device`'s raw (unsanitized) name and driver version from
// ggml_sycl_info(), and compose the on-disk device_key
// (sanitize_device_name(name) + "@" + driver_version -- see
// UbatchCacheKey::device_key's comment for why no PCI id). Returns false
// (out untouched) for an out-of-range device -- the one thing every public
// entry point below must check before touching ggml_sycl_info().devices[].
bool resolve_device_key(int device, std::string & out_device_name, std::string & out_device_key) {
    const auto & info = ggml_sycl_info();
    if (device < 0 || device >= info.device_count) {
        return false;
    }
    out_device_name = info.devices[device].device_name;
    out_device_key  = sanitize_device_name(out_device_name) + "@" + info.devices[device].driver_version;
    return true;
}

UbatchCacheKey to_internal_key(const std::string & device_key, const ggml_sycl_ubatch_cache_key & c_key) {
    UbatchCacheKey k;
    k.device_key      = device_key;
    k.model_name      = c_key.model_name ? c_key.model_name : "";
    k.model_size      = c_key.model_size;
    k.model_hash      = c_key.model_hash;
    k.n_ctx           = c_key.n_ctx;
    k.n_batch         = c_key.n_batch;
    k.flash_attn      = c_key.flash_attn;
    k.n_seq_max       = c_key.n_seq_max;
    k.type_k          = c_key.type_k;
    k.type_v          = c_key.type_v;
    k.device_set_hash = c_key.device_set_hash;
    return k;
}

std::string iso8601_now_utc() {
    std::time_t t = std::time(nullptr);
    std::tm     tm_utc{};
#ifdef _WIN32
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    char buf[32] = { 0 };
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return std::string(buf);
}

}  // namespace

bool ggml_backend_sycl_ubatch_cache_enabled(void) {
    return ubatch_tuning_cache_env_enabled();
}

bool ggml_backend_sycl_ubatch_cache_path(int device, char * buf, size_t buf_size) {
    if (buf == nullptr || buf_size == 0) {
        return false;
    }
    std::string device_name, device_key;
    if (!resolve_device_key(device, device_name, device_key)) {
        return false;
    }
    // resolve_device_key() is shared with _lookup/_store below, which DO
    // need device_key; a path-only caller has no use for it.
    (void) device_key;
    std::string path = get_ubatch_cache_file(ubatch_tuning_cache_dir(), device_name);
    // `path.size() >= buf_size` would truncate --
    // refuse instead of silently handing back a wrong (truncated) path.
    if (path.size() >= buf_size) {
        return false;
    }
    std::strncpy(buf, path.c_str(), buf_size - 1);
    buf[buf_size - 1] = '\0';
    return true;
}

bool ggml_backend_sycl_ubatch_cache_lookup(const ggml_sycl_ubatch_cache_key * key,
                                           uint32_t *                         n_ubatch,
                                           char *                             reason_buf,
                                           size_t                             reason_buf_size) {
    if (key == nullptr || n_ubatch == nullptr || !ubatch_tuning_cache_env_enabled()) {
        return false;
    }
    std::string device_name, device_key;
    if (!resolve_device_key(key->device, device_name, device_key)) {
        return false;
    }
    const UbatchCacheKey lookup_key = to_internal_key(device_key, *key);

    std::vector<UbatchCacheEntry> entries;
    if (!load_ubatch_cache(ubatch_tuning_cache_dir(), device_name, entries)) {
        return false;  // missing, unreadable, or wrong-version file -- a plain miss
    }
    for (const auto & e : entries) {
        if (e.key == lookup_key) {
            *n_ubatch = e.n_ubatch;
            if (reason_buf != nullptr && reason_buf_size > 0) {
                std::strncpy(reason_buf, e.reason.c_str(), reason_buf_size - 1);
                reason_buf[reason_buf_size - 1] = '\0';
            }
            return true;
        }
    }
    return false;
}

bool ggml_backend_sycl_ubatch_cache_store(const ggml_sycl_ubatch_cache_key * key,
                                          uint32_t                           n_ubatch,
                                          const char *                       reason) {
    if (key == nullptr || !ubatch_tuning_cache_env_enabled()) {
        return false;
    }
    std::string device_name, device_key;
    if (!resolve_device_key(key->device, device_name, device_key)) {
        return false;
    }
    const UbatchCacheKey store_key = to_internal_key(device_key, *key);

    std::vector<UbatchCacheEntry> entries;
    // A missing, corrupt, or wrong-version file is a legitimate "start from
    // empty" case here (load_ubatch_cache() already treats all three
    // identically) -- never a reason to refuse the store; the whole point
    // of this call is to (re)write the file.
    //
    // This load-modify-save is NOT atomic across processes (llama.cpp-7n6n):
    // two starts storing for the SAME device concurrently can both
    // load the same snapshot, and whichever save_ubatch_cache() renames
    // last wins, silently dropping the other's entry. Per-pid temp names
    // (sycl_tuning_getpid()) only fix the WRITE race (two writers can no
    // longer corrupt one shared temp file); they do not make this
    // read-modify-write a transaction. Accepted: the dropped entry costs
    // its key one extra ladder run on its next start, never a wrong choice
    // -- the same advisory-cache tradeoff this whole store already makes.
    load_ubatch_cache(ubatch_tuning_cache_dir(), device_name, entries);

    bool replaced = false;
    for (auto & e : entries) {
        if (e.key == store_key) {
            e.n_ubatch = n_ubatch;
            e.reason   = reason ? reason : "";
            e.created  = iso8601_now_utc();
            replaced   = true;
            break;
        }
    }
    if (!replaced) {
        UbatchCacheEntry e;
        e.key      = store_key;
        e.n_ubatch = n_ubatch;
        e.reason   = reason ? reason : "";
        e.created  = iso8601_now_utc();
        entries.push_back(e);
    }

    // llama.cpp-7n6n: cap the per-device entry count.
    // The sort-and-truncate logic itself lives in tuning-cache-io.hpp's
    // cap_ubatch_cache_entries() so it is testable from a host-only unit
    // test with no SYCL device involved.
    cap_ubatch_cache_entries(entries);

    return save_ubatch_cache(ubatch_tuning_cache_dir(), device_name, entries);
}
