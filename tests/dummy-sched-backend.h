#pragma once

// A backend with no compute, for tests of the scheduler's and the graph
// allocator's buffer handling. Its buffer type has a configurable max size,
// logs every alloc_buffer request in order, and can refuse a request. It
// lives wholly on the heap, so the pointers a scheduler keeps stay valid
// when the handle is moved.
//
// tests/test-alloc.cpp carries a similar dummy backend of its own. This one is
// separate on purpose: that file is upstream's, and editing it would make every
// merge from upstream a conflict. Do not fold the two together.

#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <memory>
#include <vector>

struct dummy_sched_backend {
    size_t max_buffer_size = 64;
    size_t alignment       = 8;
    bool   refuse_alloc    = false;

    // every alloc_buffer request, in call order, with its outcome
    std::vector<size_t> requests;
    int                 n_refused = 0;

    ggml_backend_buffer_i              buffer_interface = {};
    ggml_backend_device                device           = {};
    ggml_backend                       backend          = {};
    ggml_backend_buffer_type           buffer_type      = {};
    std::vector<ggml_backend_buffer_t> buffers;

    static dummy_sched_backend * of(void * ctx) { return (dummy_sched_backend *) ctx; }

    static const char * buft_name(ggml_backend_buffer_type_t) { return "dummy_sched_buffer_type"; }

    static ggml_backend_buffer_t buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
        dummy_sched_backend * b = of(buft->context);
        b->requests.push_back(size);
        if (b->refuse_alloc) {
            b->n_refused++;
            return nullptr;
        }
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, b->buffer_interface, b, size);
        b->buffers.push_back(buffer);
        return buffer;
    }

    static size_t buft_alignment(ggml_backend_buffer_type_t buft) { return of(buft->context)->alignment; }

    static size_t buft_max_size(ggml_backend_buffer_type_t buft) { return of(buft->context)->max_buffer_size; }

    static bool buft_is_host(ggml_backend_buffer_type_t) { return true; }

    static void buffer_free(ggml_backend_buffer_t buffer) {
        dummy_sched_backend * b = of(buffer->context);
        for (size_t i = 0; i < b->buffers.size(); i++) {
            if (b->buffers[i] == buffer) {
                b->buffers.erase(b->buffers.begin() + i);
                return;
            }
        }
        GGML_ABORT("freeing a buffer the dummy backend does not own");
    }

    static void * buffer_base(ggml_backend_buffer_t) { return (void *) 16; }

    static ggml_status buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor *) { return GGML_STATUS_SUCCESS; }

    static void buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor *, uint8_t, size_t, size_t) {}

    static void buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor *, const void *, size_t, size_t) {}

    static void buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor *, void *, size_t, size_t) {}

    static void buffer_clear(ggml_backend_buffer_t, uint8_t) {}

    static enum ggml_backend_dev_type dev_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_CPU; }

    static bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }

    static bool dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
        return dev->context == buft->context;
    }

    static const char * backend_name(ggml_backend_t) { return "dummy_sched_backend"; }

    static std::unique_ptr<dummy_sched_backend> make(size_t max_buffer_size, size_t alignment = 8) {
        auto b             = std::make_unique<dummy_sched_backend>();
        b->max_buffer_size = max_buffer_size;
        b->alignment       = alignment;

        b->buffer_interface.free_buffer   = buffer_free;
        b->buffer_interface.get_base      = buffer_base;
        b->buffer_interface.init_tensor   = buffer_init_tensor;
        b->buffer_interface.memset_tensor = buffer_memset_tensor;
        b->buffer_interface.set_tensor    = buffer_set_tensor;
        b->buffer_interface.get_tensor    = buffer_get_tensor;
        b->buffer_interface.clear         = buffer_clear;

        b->device.context             = b.get();
        b->device.iface.get_type      = dev_type;
        b->device.iface.supports_op   = dev_supports_op;
        b->device.iface.supports_buft = dev_supports_buft;

        b->backend.context        = b.get();
        b->backend.device         = &b->device;
        b->backend.iface.get_name = backend_name;

        b->buffer_type.device              = &b->device;
        b->buffer_type.context             = b.get();
        b->buffer_type.iface.get_name      = buft_name;
        b->buffer_type.iface.alloc_buffer  = buft_alloc;
        b->buffer_type.iface.get_alignment = buft_alignment;
        b->buffer_type.iface.get_max_size  = buft_max_size;
        b->buffer_type.iface.is_host       = buft_is_host;
        return b;
    }
};
