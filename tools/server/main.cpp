#include "tool-entry.h"

int llama_server(int argc, char ** argv);

int main(int argc, char ** argv) {
    return common_tool_entry("llama-server", llama_server, argc, argv);
}
