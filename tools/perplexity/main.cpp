#include "tool-entry.h"

int llama_perplexity(int argc, char ** argv);

int main(int argc, char ** argv) {
    return common_tool_entry("llama-perplexity", llama_perplexity, argc, argv);
}
