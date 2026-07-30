// Fixture: a checked dereference. The checker MUST accept this.
struct fake_extra {
    void * data_device_ptr(int dev) const;
    void * data_device_ptr_checked(int dev, const char * caller) const;
};

void use_checked_accessor(const fake_extra * extra, int id) {
    char * src_ptr = (char *) extra->data_device_ptr_checked(id, __func__);
    (void) (src_ptr + 16);
}

void use_explicit_null_check(const fake_extra * extra, int id) {
    char * src_ptr = (char *) extra->data_device_ptr(id);
    if (src_ptr == nullptr) {
        return;
    }
    (void) (src_ptr + 16);
}
