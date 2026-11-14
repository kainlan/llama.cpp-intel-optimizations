// A graph input that cannot produce its data reports it, and the result says so.
#include "llama-graph.h"

#include <cstdio>
#include <memory>

struct reporting_input : llm_graph_input_i {
    bool fail;
    explicit reporting_input(bool f) : fail(f) {}
    void set_input(const llama_ubatch *) override {}
    bool set_input_failed() const override { return fail; }
};

int main() {
    llm_graph_result res(64);
    res.add_input(std::make_unique<reporting_input>(false));
    if (res.inputs_failed()) {
        std::printf("FAIL: a clean input reported a failure\n");
        return 1;
    }
    res.add_input(std::make_unique<reporting_input>(true));
    if (!res.inputs_failed()) {
        std::printf("FAIL: a failed input was not reported\n");
        return 1;
    }
    return 0;
}
