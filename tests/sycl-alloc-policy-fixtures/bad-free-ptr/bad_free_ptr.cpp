bool unified_free_ptr(void * ptr, int device);

void bad_free_ptr_fixture(void * ptr) {
    (void) unified_free_ptr(ptr, 0);
}
