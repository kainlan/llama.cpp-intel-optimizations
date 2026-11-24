#include "tool-entry.h"

int llama_cli(int argc, char ** argv);

int main(int argc, char ** argv) {
    return common_tool_entry("llama-cli", llama_cli, argc, argv);
}
