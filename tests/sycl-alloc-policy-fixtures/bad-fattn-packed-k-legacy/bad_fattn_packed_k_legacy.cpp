struct bad_packed_k_state {
    struct alloc_view {
        void *        ptr;
        unsigned long size;
    } alloc;
};

bool bad_packed_k_legacy_fixture(bad_packed_k_state * out, bad_packed_k_state & packed) {
    return out->alloc.ptr != nullptr && packed.alloc.size > 0;
}
