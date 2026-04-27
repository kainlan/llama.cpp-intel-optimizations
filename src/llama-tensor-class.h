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

// Placement priority tier — input to PLACE-3's greedy budget fitter.
//   P0: pin to device, fail loudly if it does not fit (every-token hot path).
//   P1: hot-half MoE experts (layer < n_layers / 2). Compete for VRAM.
//   P2: cold-half MoE experts (layer >= n_layers / 2). Compete for VRAM.
//   P3: spill to host (overflow from P1/P2 fitting).
enum llama_tensor_priority {
    LLAMA_TENSOR_PRIORITY_P0 = 0,
    LLAMA_TENSOR_PRIORITY_P1,
    LLAMA_TENSOR_PRIORITY_P2,
    LLAMA_TENSOR_PRIORITY_P3,

    LLAMA_TENSOR_PRIORITY_COUNT,
};

// Map a tensor's class + layer index + total layer count to a priority tier.
// Pure function: same inputs always produce the same tier. Caller passes
// n_layers from hparams (or scans the tensor list once); PLACE-2 itself never
// touches GGUF metadata or model state.
//
// For non per-layer tensors (layer_idx == -1), n_layers is unused; pass 0.
llama_tensor_priority llama_tensor_priority_for(llama_tensor_class cls, int layer_idx, int n_layers);

const char * llama_tensor_priority_name(llama_tensor_priority prio);
