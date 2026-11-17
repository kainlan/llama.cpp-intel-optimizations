#include "tool-entry.h"

int llama_fit_params(int argc, char ** argv);

int main(int argc, char ** argv) {
    return common_tool_entry("llama-fit-params", llama_fit_params, argc, argv);
}
