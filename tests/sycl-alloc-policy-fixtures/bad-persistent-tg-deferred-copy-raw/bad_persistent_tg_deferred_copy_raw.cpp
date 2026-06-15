struct Kernel {
    void add_deferred_copy(int, void *, void *, unsigned long);
};

void bad_persistent_tg_deferred_copy_raw(Kernel & kernel, int source_op_idx, void * src_ptr, void * dst_ptr) {
    unsigned long nbytes = 4096;
    kernel.add_deferred_copy(source_op_idx, src_ptr, dst_ptr, nbytes);
}
