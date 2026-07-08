struct bad_tp_input_storage {
    struct alloc_view {
        void * ptr;
    } alloc;

    void * data;

    void * data_ptr() const { return alloc.ptr ? alloc.ptr : data; }
};
