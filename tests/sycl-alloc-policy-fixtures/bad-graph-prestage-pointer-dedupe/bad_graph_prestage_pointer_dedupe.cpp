#include <unordered_set>

struct ggml_tensor {
    void * data;
};

void bad_graph_prestage_dedupe(ggml_tensor * tensor) {
    std::unordered_set<void *> staged_pointers;
    if (staged_pointers.count(tensor->data)) {
        return;
    }
    staged_pointers.insert(tensor->data);
}
