#pragma once

struct ggml_tensor;

enum llama_tensor_class {
    LLAMA_TENSOR_CLASS_MISC = 0,

    LLAMA_TENSOR_CLASS_ATTN_Q,
    LLAMA_TENSOR_CLASS_ATTN_K,
    LLAMA_TENSOR_CLASS_ATTN_V,
    LLAMA_TENSOR_CLASS_ATTN_O,
    LLAMA_TENSOR_CLASS_ATTN_QKV,
    LLAMA_TENSOR_CLASS_ATTN_NORM,

    LLAMA_TENSOR_CLASS_FFN_GATE_INP,
    LLAMA_TENSOR_CLASS_FFN_DENSE_GATE,
    LLAMA_TENSOR_CLASS_FFN_DENSE_UP,
    LLAMA_TENSOR_CLASS_FFN_DENSE_DOWN,
    LLAMA_TENSOR_CLASS_FFN_MOE_GATE_EXPS,
    LLAMA_TENSOR_CLASS_FFN_MOE_UP_EXPS,
    LLAMA_TENSOR_CLASS_FFN_MOE_DOWN_EXPS,
    LLAMA_TENSOR_CLASS_FFN_SHARED_GATE,
    LLAMA_TENSOR_CLASS_FFN_SHARED_UP,
    LLAMA_TENSOR_CLASS_FFN_SHARED_DOWN,
    LLAMA_TENSOR_CLASS_FFN_NORM,

    LLAMA_TENSOR_CLASS_EMBD_TOKEN,
    LLAMA_TENSOR_CLASS_EMBD_OUTPUT,
    LLAMA_TENSOR_CLASS_OUTPUT_NORM,
    LLAMA_TENSOR_CLASS_POSITION_EMBD,

    LLAMA_TENSOR_CLASS_COUNT,
};

struct llama_tensor_classification {
    llama_tensor_class cls;
    int                layer_idx;  // -1 when the tensor is not per-layer
};

llama_tensor_classification llama_tensor_classify(const char * name);
llama_tensor_classification llama_tensor_classify(const ggml_tensor * tensor);

const char * llama_tensor_class_name(llama_tensor_class cls);
