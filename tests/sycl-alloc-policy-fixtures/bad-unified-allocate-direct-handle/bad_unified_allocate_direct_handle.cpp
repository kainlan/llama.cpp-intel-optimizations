struct mem_handle {
    static mem_handle from_direct(void *, int, bool, int);
};

struct alloc_handle {
    void * ptr;
};

mem_handle unified_allocate_legacy(alloc_handle handle) {
    return mem_handle::from_direct(handle.ptr, 0, true, 0);
}
