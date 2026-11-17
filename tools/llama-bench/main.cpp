#include "tool-entry.h"

int llama_bench(int argc, char ** argv);

int main(int argc, char ** argv) {
    return common_tool_entry("llama-bench", llama_bench, argc, argv);
}
