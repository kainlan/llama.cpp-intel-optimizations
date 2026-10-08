// Per-split segmented-graph slots (llama.cpp-7pm2). ggml-sycl/graph-segment-cache.hpp is SYCL-free, so the key
// and the slot state machine are checked here without a device. The payload stands in for a slot's recorded
// segments and retained handles; its id tells which recording a retired payload came from.
#include "ggml-sycl/graph-segment-cache.hpp"

#include <cstdio>
#include <cstring>

#define CHECK(cond, msg)                      \
    do {                                      \
        if (!(cond)) {                        \
            std::printf("FAILED: %s\n", msg); \
            return 1;                         \
        }                                     \
    } while (0)

namespace gsc = ggml_sycl::graph_segment_cache;

namespace {

struct fake_payload {
    int id = 0;
};

uint64_t hash_names(const char * const * names, int n) {
    gsc::key_hasher h;
    for (int i = 0; i < n; ++i) {
        h.mix_name(names[i], 64);
    }
    return h.value();
}

// A split as the scheduler hands it over: same structure and node count, layer told apart only by its names.
gsc::key layer_key(int layer) {
    char attn[32];
    char ffn[32];
    std::snprintf(attn, sizeof(attn), "attn_norm-%d", layer);
    std::snprintf(ffn, sizeof(ffn), "ffn_moe_out-%d", layer);
    const char * names[] = { attn, ffn };
    gsc::key     k;
    k.signature = 0x1234;
    k.names     = hash_names(names, 2);
    k.storage   = 0x5678;
    k.n_nodes   = 41;
    k.device    = 0;
    k.is_decode = true;
    return k;
}

int test_hasher() {
    const char * ab_c[] = { "ab", "c" };
    const char * a_bc[] = { "a", "bc" };
    CHECK(hash_names(ab_c, 2) != hash_names(a_bc, 2), "name terminator: (ab,c) and (a,bc) hash the same");

    const char * x_y[] = { "x", "y" };
    const char * y_x[] = { "y", "x" };
    CHECK(hash_names(x_y, 2) != hash_names(y_x, 2), "name order does not change the hash");

    const char * null_name[]  = { nullptr };
    const char * empty_name[] = { "" };
    CHECK(hash_names(null_name, 1) != hash_names(empty_name, 1), "a null name hashes like an empty one");

    gsc::key_hasher a;
    gsc::key_hasher b;
    a.mix(1);
    b.mix(1ULL << 8);
    CHECK(a.value() != b.value(), "mix() ignores the byte position");

    // max_len bounds the read: a name without a terminator inside the bound hashes its first max_len bytes.
    char            bounded[4] = { 'a', 'b', 'c', 'd' };
    const char *    abc[]      = { "abc" };
    gsc::key_hasher c;
    c.mix_name(bounded, 3);
    CHECK(c.value() == hash_names(abc, 1), "mix_name reads past max_len");
    return 0;
}

int test_key_fields() {
    const gsc::key base = layer_key(3);
    gsc::key       k    = base;
    CHECK(k == base, "a key differs from its copy");
    k.signature++;
    CHECK(k != base, "signature is not part of the key");
    k = base;
    k.names++;
    CHECK(k != base, "names are not part of the key");
    k = base;
    k.storage++;
    CHECK(k != base, "storage is not part of the key");
    k = base;
    k.n_nodes++;
    CHECK(k != base, "n_nodes is not part of the key");
    k        = base;
    k.device = 1;
    CHECK(k != base, "device is not part of the key");
    k           = base;
    k.is_decode = false;
    CHECK(k != base, "phase is not part of the key");
    return 0;
}

int test_lifecycle() {
    gsc::slot_cache<fake_payload> cache(8);
    const gsc::key                k = layer_key(0);
    CHECK(cache.begin(k) == gsc::action::WARMUP, "first sight is not a warmup");
    CHECK(cache.payload(k) == nullptr, "a warmed key has a payload");
    CHECK(cache.begin(k) == gsc::action::RECORD, "second sight does not record");
    cache.record_succeeded(k, fake_payload{ 7 });
    CHECK(cache.begin(k) == gsc::action::REPLAY, "a recorded key does not replay");
    CHECK(cache.payload(k) != nullptr && cache.payload(k)->id == 7, "replay does not see the recorded payload");
    CHECK(cache.begin(k) == gsc::action::REPLAY, "a second replay does not replay");
    CHECK(!cache.has_retired(), "replaying retired something");
    return 0;
}

// The defect the per-split slots replace: two splits with the same node count from different layers matched one
// slot, so one layer replayed the other's graph.
int test_same_n_nodes_layers_do_not_share() {
    gsc::slot_cache<fake_payload> cache(8);
    const gsc::key                a = layer_key(1);
    const gsc::key                b = layer_key(2);
    CHECK(a.n_nodes == b.n_nodes && a.signature == b.signature, "fixture: layers must share shape");
    cache.begin(a);
    cache.begin(b);
    CHECK(cache.begin(a) == gsc::action::RECORD, "layer 1 does not record");
    cache.record_succeeded(a, fake_payload{ 1 });
    CHECK(cache.begin(b) == gsc::action::RECORD, "layer 2 replays layer 1's recording");
    cache.record_succeeded(b, fake_payload{ 2 });
    CHECK(cache.begin(a) == gsc::action::REPLAY && cache.payload(a)->id == 1, "layer 1 lost its own recording");
    CHECK(cache.begin(b) == gsc::action::REPLAY && cache.payload(b)->id == 2, "layer 2 lost its own recording");
    CHECK(cache.size() == 2, "two layers do not hold two slots");
    return 0;
}

// Alternating splits, as one decode token interleaves them, all replay from the third token on.
int test_interleaved_splits_all_replay() {
    const int                     n_splits = 48;
    gsc::slot_cache<fake_payload> cache;
    for (int token = 0; token < 4; ++token) {
        for (int s = 0; s < n_splits; ++s) {
            const gsc::key    k = layer_key(s);
            const gsc::action a = cache.begin(k);
            if (token == 0) {
                CHECK(a == gsc::action::WARMUP, "token 0 split is not a warmup");
            } else if (token == 1) {
                CHECK(a == gsc::action::RECORD, "token 1 split does not record");
                cache.record_succeeded(k, fake_payload{ s });
            } else {
                CHECK(a == gsc::action::REPLAY, "a split does not replay after its record");
                CHECK(cache.payload(k)->id == s, "a split replays another split's recording");
            }
        }
    }
    CHECK(cache.stats().replays == 2 * n_splits, "replay count is not two tokens of splits");
    CHECK(!cache.churned(), "a steady interleave churned the cache");
    return 0;
}

int test_failed_runs_direct() {
    gsc::slot_cache<fake_payload> cache(8);
    const gsc::key                k = layer_key(0);
    cache.begin(k);
    CHECK(cache.begin(k) == gsc::action::RECORD, "second sight does not record");
    cache.record_failed(k);
    CHECK(cache.begin(k) == gsc::action::DIRECT, "a failed key does not run direct");
    CHECK(cache.begin(k) == gsc::action::DIRECT, "a failed key retries");
    CHECK(cache.payload(k) == nullptr, "a failed key has a payload");
    CHECK(cache.stats().failures == 1, "failure not counted");
    return 0;
}

int test_lru_eviction_retires_payload() {
    gsc::slot_cache<fake_payload> cache(2);
    const gsc::key                a = layer_key(1);
    const gsc::key                b = layer_key(2);
    const gsc::key                c = layer_key(3);
    cache.begin(a);
    cache.begin(a);
    cache.record_succeeded(a, fake_payload{ 1 });
    cache.begin(b);
    cache.begin(b);
    cache.record_succeeded(b, fake_payload{ 2 });
    CHECK(cache.begin(a) == gsc::action::REPLAY, "a does not replay");  // a is now the most recent
    CHECK(cache.begin(c) == gsc::action::WARMUP, "a new key is not a warmup");
    CHECK(cache.size() == 2, "eviction did not hold the capacity");
    CHECK(cache.payload(a) != nullptr, "the most recently used slot was evicted");
    CHECK(cache.state(b) == nullptr, "the least recently used slot survived");
    CHECK(cache.has_retired(), "the evicted recording was dropped instead of retired");
    std::vector<fake_payload> retired = cache.take_retired();
    CHECK(retired.size() == 1 && retired[0].id == 2, "the wrong recording was retired");
    CHECK(!cache.has_retired(), "take_retired left payloads behind");
    CHECK(cache.stats().evictions == 1, "eviction not counted");
    return 0;
}

int test_evicting_a_warmed_slot_retires_nothing() {
    gsc::slot_cache<fake_payload> cache(1);
    cache.begin(layer_key(1));
    cache.begin(layer_key(2));
    CHECK(!cache.has_retired(), "evicting a slot with no recording retired a payload");
    return 0;
}

int test_invalidate_all() {
    gsc::slot_cache<fake_payload> cache(8);
    const gsc::key                a = layer_key(1);
    const gsc::key                b = layer_key(2);
    cache.begin(a);
    cache.begin(a);
    cache.record_succeeded(a, fake_payload{ 1 });
    cache.begin(b);
    cache.begin(b);
    cache.record_failed(b);
    CHECK(cache.any_payload(), "a recorded slot reports no payload");
    cache.invalidate_all();
    CHECK(cache.size() == 0, "invalidate_all kept slots");
    std::vector<fake_payload> retired = cache.take_retired();
    CHECK(retired.size() == 1 && retired[0].id == 1, "invalidate_all did not retire the recording");
    CHECK(!cache.any_payload(), "payloads survive invalidate_all and take_retired");
    CHECK(cache.begin(a) == gsc::action::WARMUP, "an invalidated key replays");
    CHECK(cache.begin(b) == gsc::action::WARMUP, "an invalidated FAILED key stays failed");
    return 0;
}

// A record that finishes after its slot was invalidated must not resurrect the slot, but its payload was
// possibly submitted, so it is retired rather than dropped.
int test_record_after_invalidate_is_retired() {
    gsc::slot_cache<fake_payload> cache(8);
    const gsc::key                k = layer_key(0);
    cache.begin(k);
    CHECK(cache.begin(k) == gsc::action::RECORD, "second sight does not record");
    cache.invalidate_all();
    cache.record_succeeded(k, fake_payload{ 9 });
    CHECK(cache.payload(k) == nullptr, "a record after invalidation resurrected the slot");
    std::vector<fake_payload> retired = cache.take_retired();
    CHECK(retired.size() == 1 && retired[0].id == 9, "the orphaned recording was dropped instead of retired");
    return 0;
}

int test_churn_turns_cache_off() {
    gsc::slot_cache<fake_payload> cache(16, 6);
    const gsc::key                stable = layer_key(100);
    cache.begin(stable);
    cache.begin(stable);
    cache.record_succeeded(stable, fake_payload{ 100 });
    CHECK(cache.begin(stable) == gsc::action::REPLAY, "the stable key does not replay");
    // Keys that never repeat: six warmups are allowed, the seventh trips the guard.
    for (int i = 0; i < 6; ++i) {
        CHECK(cache.begin(layer_key(i)) == gsc::action::WARMUP, "a warmup under the churn limit ran direct");
    }
    CHECK(!cache.churned(), "the cache churned at the limit");
    CHECK(cache.begin(layer_key(6)) == gsc::action::DIRECT, "the warmup past the limit was not direct");
    CHECK(cache.churned(), "the churn guard did not latch");
    CHECK(cache.size() == 0, "a churned cache kept slots");
    CHECK(cache.begin(stable) == gsc::action::DIRECT, "a churned cache still replays");
    std::vector<fake_payload> retired = cache.take_retired();
    CHECK(retired.size() == 1 && retired[0].id == 100, "churn did not retire the recorded slot");
    return 0;
}

int test_replay_resets_churn_count() {
    gsc::slot_cache<fake_payload> cache(64, 6);
    const gsc::key                stable = layer_key(100);
    cache.begin(stable);
    cache.begin(stable);
    cache.record_succeeded(stable, fake_payload{ 100 });
    // Five new keys, a replay, five more: never six warmups in a row.
    for (int round = 0; round < 2; ++round) {
        for (int i = 0; i < 5; ++i) {
            cache.begin(layer_key(round * 10 + i));
        }
        CHECK(cache.begin(stable) == gsc::action::REPLAY, "the stable key stopped replaying");
    }
    CHECK(!cache.churned(), "warmups separated by replays churned the cache");
    return 0;
}

// A slot whose recording no longer matches what its split reads (an input's staging copy changed) is forgotten
// alone: its payload is retired for the drain, the key warms up again, and other keys keep replaying.
int test_forget_retires_one_key() {
    gsc::slot_cache<fake_payload> cache(8);
    const gsc::key                a = layer_key(1);
    const gsc::key                b = layer_key(2);
    for (const gsc::key & k : { a, b }) {
        cache.begin(k);
        cache.begin(k);
        cache.record_succeeded(k, fake_payload{ k == a ? 1 : 2 });
    }
    cache.forget(a);
    CHECK(cache.size() == 1, "forget kept the slot or dropped another one");
    CHECK(cache.payload(a) == nullptr, "a forgotten key still has a payload");
    std::vector<fake_payload> retired = cache.take_retired();
    CHECK(retired.size() == 1 && retired[0].id == 1, "forget did not retire the forgotten recording");
    CHECK(cache.begin(b) == gsc::action::REPLAY, "forget disturbed another key");
    CHECK(cache.begin(a) == gsc::action::WARMUP, "a forgotten key does not warm up again");
    cache.forget(layer_key(3));
    CHECK(cache.size() == 2 && !cache.has_retired(), "forgetting an unknown key changed the cache");
    return 0;
}

}  // namespace

int main() {
    if (test_hasher() || test_key_fields() || test_lifecycle() || test_same_n_nodes_layers_do_not_share() ||
        test_interleaved_splits_all_replay() || test_failed_runs_direct() || test_lru_eviction_retires_payload() ||
        test_evicting_a_warmed_slot_retires_nothing() || test_invalidate_all() ||
        test_record_after_invalidate_is_retired() || test_churn_turns_cache_off() || test_replay_resets_churn_count() ||
        test_forget_retires_one_key()) {
        return 1;
    }
    std::printf("test-sycl-graph-segment-cache: all checks passed\n");
    return 0;
}
