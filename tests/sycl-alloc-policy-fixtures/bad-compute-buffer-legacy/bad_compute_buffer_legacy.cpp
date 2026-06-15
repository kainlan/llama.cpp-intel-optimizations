struct alloc_handle {
    void * ptr;
};

struct bad_compute_buffer {
    void *       ptr;
    alloc_handle handle;
};

struct bad_compute_buffer_manager {
    alloc_handle scratch_handle_;
};
