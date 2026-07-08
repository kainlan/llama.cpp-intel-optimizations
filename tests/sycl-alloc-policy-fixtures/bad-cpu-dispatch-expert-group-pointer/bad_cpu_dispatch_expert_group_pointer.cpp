#include <vector>

namespace std {
template <typename K, typename V>
struct unordered_map {
    V & operator[](const K &) {
        static V value;
        return value;
    }
};
}  // namespace std

struct task {
    const void * weight_host;
};

void bad_cpu_dispatch_expert_group_pointer(task * tasks, int n_tasks) {
    std::unordered_map<const void *, std::vector<int>> expert_groups;
    for (int i = 0; i < n_tasks; ++i) {
        expert_groups[tasks[i].weight_host].push_back(i);
    }
}
