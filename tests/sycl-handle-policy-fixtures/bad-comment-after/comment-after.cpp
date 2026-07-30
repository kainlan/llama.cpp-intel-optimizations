// Fixture: a genuinely unchecked dereference whose FOLLOWING line is a stale
// comment describing a check that no longer exists. The checker MUST still
// reject this. Stale comments like this are ordinary in code being refactored,
// which is exactly the code this gate polices.
struct fake_extra {
    void * data_device_ptr(int dev) const;
};

void use(const fake_extra * extra, int id) {
    char * src_ptr = (char *) extra->data_device_ptr(id);
    // if (src_ptr) was removed in a refactor
    (void) (src_ptr + 16);
}
