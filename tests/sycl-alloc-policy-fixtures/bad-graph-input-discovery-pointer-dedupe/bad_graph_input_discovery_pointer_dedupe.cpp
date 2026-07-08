struct ggml_tensor {
    void * data;
};

namespace std {
template <typename T>
struct hash;

template <typename T>
struct unordered_set {
    struct insert_result {
        bool second;
    };

    insert_result insert(T) { return { true }; }
};
}  // namespace std

void bad_graph_input_discovery_pointer_dedupe(ggml_tensor * tensor) {
    std::unordered_set<const void *> seen_ptrs;
    if (!seen_ptrs.insert(tensor->data).second) {
        return;
    }
}
