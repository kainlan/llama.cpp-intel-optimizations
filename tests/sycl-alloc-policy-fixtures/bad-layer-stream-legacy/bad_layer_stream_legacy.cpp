struct bad_layer_stream_state {
    void * buffer_allocs_[2];
    void * buffers_[2];
};

void bad_layer_stream_legacy_fixture(bad_layer_stream_state * state, void * ptr) {
    state->buffer_allocs_[0] = ptr;
    state->buffers_[0]       = ptr;
}
