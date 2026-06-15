#include <unordered_set>

struct PrefetchRequest {
    const void * tensor_data;
};

struct BadPrefetchScheduler {
    std::unordered_set<const void *> active_prefetches_;

    void start(PrefetchRequest req) { active_prefetches_.insert(req.tensor_data); }

    void complete(const void * tensor_data) { active_prefetches_.erase(tensor_data); }

    bool active(const void * tensor_data) const { return active_prefetches_.count(tensor_data) > 0; }
};
