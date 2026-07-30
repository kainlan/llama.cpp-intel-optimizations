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

// Regression: the idiomatic truthiness check. The checker MUST accept this --
// `if (ptr)` is the most common null check in the SYCL backend, and flagging it
// forces needless edits to code that is already safe (ggml-sycl.cpp:5034).
void use_truthiness_check(const fake_extra * extra, int id) {
    void * dev_ptr = extra->data_device_ptr(id);
    if (dev_ptr) {
        (void) dev_ptr;
    }
}

// Regression: the guard sits on the PRECEDING line, wrapping the assignment, so
// a forward-only scan cannot see it. The checker MUST accept this
// (ggml-sycl.cpp:6964-6965).
void use_preceding_guard(const fake_extra * extra, int id) {
    void * gate_ptr = nullptr;
    if (extra && extra->data_device_ptr(id)) {
        gate_ptr = extra->data_device_ptr(id);
    }
    (void) gate_ptr;
}

// Regression: a violation quoted inside an ordinary explanatory comment is NOT
// a dereference. This file and the backend both comment heavily, so a checker
// that reads comments breaks the gate on unrelated changes.
// Example: char * src_ptr = (char *) extra->data_device_ptr(id);
