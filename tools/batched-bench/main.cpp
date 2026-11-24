#include "tool-entry.h"

int llama_batched_bench(int argc, char ** argv);

int main(int argc, char ** argv) {
    return common_tool_entry("llama-batched-bench", llama_batched_bench, argc, argv);
}
