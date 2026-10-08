#pragma once

// Per-split segmented-graph slots (llama.cpp-7pm2 B2). Host-only: no SYCL, no ggml types, so the key and the
// slot state machine are unit-tested without a device (tests/test-sycl-graph-segment-cache.cpp).
//
// A decode token reaches graph_compute once per scheduler split. Splits from different layers can share a node
// count and a structural signature, so a slot is keyed on the structural signature, the tensor names, and the
// storage the split reads and writes (owner identity of each buffer plus the offset inside it, never a raw device
// address), together with the node count, the device and the phase. Each key owns its own recorded segments.
//
// Life of a key: the first call warms it up (runs direct, settles lazy allocations), the second records, later
// calls replay. A record that cannot produce a graph marks the key FAILED and it runs direct from then on. When a
// new key needs room the least recently used slot is evicted. A slot that held recorded graphs goes to the retired
// list: the caller must wait for the queue before it destroys them, because a replay may still be in flight.
// Keys that never repeat (more warmups than churn_limit with no replay in between) turn the cache off for good:
// every call then runs direct and every slot is retired.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ggml_sycl {
namespace graph_segment_cache {

// FNV-1a over bytes. Names are mixed with a terminator so ("ab","c") and ("a","bc") differ.
struct key_hasher {
    static constexpr uint64_t offset_basis = 1469598103934665603ULL;
    static constexpr uint64_t prime        = 1099511628211ULL;

    uint64_t h = offset_basis;

    void mix_byte(uint8_t b) {
        h ^= b;
        h *= prime;
    }

    void mix(uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            mix_byte(static_cast<uint8_t>(v >> (8 * i)));
        }
    }

    // A null name mixes a byte no UTF-8 name contains, so it differs from "".
    void mix_name(const char * s, size_t max_len) {
        if (s == nullptr) {
            mix_byte(0xff);
            mix_byte(0);
            return;
        }
        for (size_t i = 0; i < max_len && s[i] != '\0'; ++i) {
            mix_byte(static_cast<uint8_t>(s[i]));
        }
        mix_byte(0);
    }

    uint64_t value() const { return h; }
};

struct key {
    uint64_t signature = 0;  // ops, types, shapes, strides, view offsets, op params
    uint64_t names     = 0;  // node, src and leaf names
    uint64_t storage   = 0;  // per tensor: buffer owner identity + offset inside the buffer
    int32_t  n_nodes   = 0;
    int32_t  device    = -1;
    bool     is_decode = false;
};

inline bool operator==(const key & a, const key & b) {
    return a.signature == b.signature && a.names == b.names && a.storage == b.storage && a.n_nodes == b.n_nodes &&
           a.device == b.device && a.is_decode == b.is_decode;
}

inline bool operator!=(const key & a, const key & b) {
    return !(a == b);
}

enum class action {
    WARMUP,  // first sight of the key: run direct
    RECORD,  // second sight: record, then report record_succeeded or record_failed
    REPLAY,  // recorded: replay payload(key)
    DIRECT,  // the key failed to record, or the cache churned: run direct
};

enum class slot_state {
    WARMED,
    RECORDED,
    FAILED,
};

struct counters {
    uint64_t warmups   = 0;
    uint64_t records   = 0;
    uint64_t replays   = 0;
    uint64_t directs   = 0;
    uint64_t failures  = 0;
    uint64_t evictions = 0;
};

inline const char * action_name(action a) {
    switch (a) {
        case action::WARMUP:
            return "warmup";
        case action::RECORD:
            return "record";
        case action::REPLAY:
            return "replay";
        case action::DIRECT:
            return "direct";
    }
    return "?";
}

template <typename Payload> class slot_cache {
  public:
    static constexpr size_t default_capacity = 128;

    // churn_limit 0 means four times the capacity.
    explicit slot_cache(size_t capacity = default_capacity, size_t churn_limit = 0) :
        capacity_(capacity == 0 ? 1 : capacity),
        churn_limit_(churn_limit == 0 ? 4 * (capacity == 0 ? 1 : capacity) : churn_limit) {}

    action begin(const key & k) {
        if (churned_) {
            stats_.directs++;
            return action::DIRECT;
        }
        slot * s = find(k);
        if (s != nullptr) {
            s->last_use = ++clock_;
            switch (s->state) {
                case slot_state::WARMED:
                    stats_.records++;
                    return action::RECORD;
                case slot_state::RECORDED:
                    warmups_since_replay_ = 0;
                    stats_.replays++;
                    return action::REPLAY;
                case slot_state::FAILED:
                    stats_.directs++;
                    return action::DIRECT;
            }
        }
        if (++warmups_since_replay_ > churn_limit_) {
            churned_ = true;
            invalidate_all();
            stats_.directs++;
            return action::DIRECT;
        }
        if (slots_.size() >= capacity_) {
            evict_lru();
        }
        slot fresh;
        fresh.k        = k;
        fresh.state    = slot_state::WARMED;
        fresh.last_use = ++clock_;
        slots_.push_back(std::move(fresh));
        stats_.warmups++;
        return action::WARMUP;
    }

    // After RECORD. A key that was evicted or invalidated while recording is not re-inserted: its payload is
    // retired instead, since the caller may already have submitted it.
    void record_succeeded(const key & k, Payload && payload) {
        slot * s = find(k);
        if (s == nullptr) {
            retired_.push_back(std::move(payload));
            return;
        }
        if (s->has_payload) {
            retired_.push_back(std::move(s->payload));
        }
        s->payload     = std::move(payload);
        s->has_payload = true;
        s->state       = slot_state::RECORDED;
    }

    void record_failed(const key & k) {
        stats_.failures++;
        slot * s = find(k);
        if (s == nullptr) {
            return;
        }
        retire_payload(*s);
        s->state = slot_state::FAILED;
    }

    // The recorded payload for k, or null.
    Payload * payload(const key & k) {
        slot * s = find(k);
        return s != nullptr && s->has_payload && s->state == slot_state::RECORDED ? &s->payload : nullptr;
    }

    const slot_state * state(const key & k) const {
        const slot * s = find(k);
        return s != nullptr ? &s->state : nullptr;
    }

    // Retires every payload and forgets every key, FAILED ones included. Does not clear the churned flag.
    void invalidate_all() {
        for (slot & s : slots_) {
            retire_payload(s);
        }
        slots_.clear();
        warmups_since_replay_ = 0;
    }

    // Payloads that must be destroyed only after the queue that ran them has drained.
    std::vector<Payload> take_retired() {
        std::vector<Payload> out;
        out.swap(retired_);
        return out;
    }

    bool has_retired() const { return !retired_.empty(); }

    bool churned() const { return churned_; }

    size_t size() const { return slots_.size(); }

    size_t capacity() const { return capacity_; }

    bool any_payload() const {
        for (const slot & s : slots_) {
            if (s.has_payload) {
                return true;
            }
        }
        return !retired_.empty();
    }

    const counters & stats() const { return stats_; }

  private:
    struct slot {
        key        k{};
        slot_state state       = slot_state::WARMED;
        uint64_t   last_use    = 0;
        bool       has_payload = false;
        Payload    payload{};
    };

    slot * find(const key & k) {
        for (slot & s : slots_) {
            if (s.k == k) {
                return &s;
            }
        }
        return nullptr;
    }

    const slot * find(const key & k) const {
        for (const slot & s : slots_) {
            if (s.k == k) {
                return &s;
            }
        }
        return nullptr;
    }

    void retire_payload(slot & s) {
        if (s.has_payload) {
            retired_.push_back(std::move(s.payload));
            s.payload     = Payload{};
            s.has_payload = false;
        }
    }

    void evict_lru() {
        size_t victim = 0;
        for (size_t i = 1; i < slots_.size(); ++i) {
            if (slots_[i].last_use < slots_[victim].last_use) {
                victim = i;
            }
        }
        retire_payload(slots_[victim]);
        slots_.erase(slots_.begin() + static_cast<std::ptrdiff_t>(victim));
        stats_.evictions++;
    }

    size_t               capacity_;
    size_t               churn_limit_;
    std::vector<slot>    slots_;
    std::vector<Payload> retired_;
    uint64_t             clock_                = 0;
    size_t               warmups_since_replay_ = 0;
    bool                 churned_              = false;
    counters             stats_;
};

}  // namespace graph_segment_cache
}  // namespace ggml_sycl
