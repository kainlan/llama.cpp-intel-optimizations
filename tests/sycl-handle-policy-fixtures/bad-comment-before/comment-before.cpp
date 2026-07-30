// Fixture: a genuinely unchecked dereference whose PRECEDING line is a comment
// that merely mentions a guard. The checker MUST still reject this -- reading
// the comment as if it were the guard is a false NEGATIVE, the dangerous
// direction: it turns "nobody checked" into "the check passed".
struct fake_extra {
    void * data_device_ptr(int dev) const;
};

void use(const fake_extra * extra, int id) {
    // historically we did: if (extra->data_device_ptr(id)) { ... }
    char * src_ptr = (char *) extra->data_device_ptr(id);
    (void) (src_ptr + 16);
}
