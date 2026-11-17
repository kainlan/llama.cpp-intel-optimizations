#include "tool-entry.h"

int llama_quantize(int argc, char ** argv);

int main(int argc, char ** argv) {
    return common_tool_entry("llama-quantize", llama_quantize, argc, argv);
}
