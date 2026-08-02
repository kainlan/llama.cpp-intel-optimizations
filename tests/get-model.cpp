#include "get-model.h"

#include "test-skip.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

char * get_model_or_exit(int argc, char *argv[]) {
    char * model_path;
    if (argc > 1) {
        model_path = argv[1];

    } else {
        model_path = getenv("LLAMACPP_TEST_MODELFILE");
        if (!model_path || strlen(model_path) == 0) {
            test_skip_no_model();
        }
    }

    return model_path;
}
