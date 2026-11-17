#include "tool-entry.h"

int llama_completion(int argc, char ** argv);

int main(int argc, char ** argv) {
    return common_tool_entry("llama-completion", llama_completion, argc, argv);
}
