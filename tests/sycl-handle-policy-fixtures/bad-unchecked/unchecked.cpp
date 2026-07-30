// Fixture: an unchecked data_device_ptr() dereference. The checker MUST reject this.
struct fake_extra {
    void * data_device_ptr(int dev) const;
};

void use(const fake_extra * extra, int id) {
    char * src_ptr = (char *) extra->data_device_ptr(id);
    (void) (src_ptr + 16);
}
