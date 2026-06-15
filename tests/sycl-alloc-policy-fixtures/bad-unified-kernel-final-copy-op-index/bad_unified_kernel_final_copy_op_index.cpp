void bad_final_copy_registration(int final_op_idx, void * copy_back_dst, size_t final_op_bytes) {
    add_deferred_copy(final_op_idx, nullptr, copy_back_dst, final_op_bytes);
}
