// MIT license
// SPDX-License-Identifier: MIT
//
// The host-weight create-set record, its wire converter, the thread-local pending
// slot and the stage-scoped holder (llama.cpp-moua L4, th32pre D-R8).  Pure and
// SYCL-free: it includes only ggml-sycl.h for the wire struct, so a host test
// drives every arm.
//
// The flow it carries.  llama calls ggml_backend_sycl_publish_host_create_record
// immediately before each stage_inventory_plan call; the export's body is
// host_create_pending_set, which converts the wire struct with the ONE converter
// below and parks the copy in a thread_local slot tagged with the bound
// candidate.  The stage takes the slot once, at entry, into a
// host_create_stage_holder; every per-device populate in the stage reads the
// holder.  Consuming per populate instead would give device 0 the record and
// device 1 absent.
//
// A record is ABSENT, never garbage, when the export is missing, the struct is
// shorter than v1, present is 0, or the record is malformed (the last also
// carries a reason).  A malformed or absent record is STORED as absent so the
// configure step fails closed; a null pointer and a missing bound candidate are
// WARN-only no-ops that never overwrite a pending record.
//
// host_create_record is declared here, not in host-weight-pack.hpp: that file
// (rnh3-1p) includes this one for the type.
#pragma once

#include "ggml-sycl.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ggml_sycl {

// The create set as the planners see it: a COPY of the wire arrays, never the
// caller's pointers.  present = false is "absent", NOT "empty"; device[i] and
// alloc_bytes[i] are parallel, in create order.
struct host_create_record {
    bool                present = false;
    std::vector<int>    device;
    std::vector<size_t> alloc_bytes;
};

// An opaque candidate identity (the bound candidate's token); 0 = none.
using host_create_tag = uintptr_t;

// v1 is the only version: 32 bytes, as ggml_sycl_host_create_record declares it on LP64.
static constexpr uint32_t HOST_CREATE_RECORD_V1_SIZE = 32;
static_assert(sizeof(ggml_sycl_host_create_record) == HOST_CREATE_RECORD_V1_SIZE,
              "the v1 wire layout is 32 bytes; a layout change is a new version, not an edit of v1");

// More entries than any real load creates; a larger count is a garbage struct.
static constexpr uint64_t HOST_CREATE_RECORD_MAX_COUNT = 1ull << 22;

enum class host_create_wire_status {
    PRESENT,    // a well-formed present record
    ABSENT,     // null, a short struct, or present == 0
    MALFORMED,  // present but unusable (present not 0 or 1, a bad count or device); `why` names it
};

// The only parser of the wire struct: the export body and the 7v5c-2 dry-run both
// call it, so the dry-run cannot accept what the export refuses.  `out` is always
// set (to absent unless PRESENT); `why` (optional) is filled for every non-PRESENT
// status.
inline host_create_wire_status host_create_record_from_wire(const ggml_sycl_host_create_record * wire,
                                                            host_create_record *                 out,
                                                            std::string *                        why) {
    auto refuse = [&](host_create_wire_status st, const std::string & reason) {
        *out = host_create_record();
        if (why) {
            *why = reason;
        }
        return st;
    };
    if (wire == nullptr) {
        return refuse(host_create_wire_status::ABSENT, "null record");
    }
    // struct_size is read first: it is the only field an older, shorter struct is
    // guaranteed to hold.
    if (wire->struct_size < HOST_CREATE_RECORD_V1_SIZE) {
        return refuse(host_create_wire_status::ABSENT, "struct_size " + std::to_string(wire->struct_size) +
                                                           " is below the v1 size " +
                                                           std::to_string(HOST_CREATE_RECORD_V1_SIZE));
    }
    if (wire->present == 0) {
        return refuse(host_create_wire_status::ABSENT, "record not present");
    }
    // present is 0 or 1; any other value is a garbage struct, not a record.
    if (wire->present != 1) {
        return refuse(host_create_wire_status::MALFORMED,
                      "present is " + std::to_string(wire->present) + ", expected 0 or 1");
    }
    if (wire->count > HOST_CREATE_RECORD_MAX_COUNT) {
        return refuse(host_create_wire_status::MALFORMED, "count " + std::to_string(wire->count) +
                                                              " is above the cap " +
                                                              std::to_string(HOST_CREATE_RECORD_MAX_COUNT));
    }
    if (wire->count > 0 && (wire->device == nullptr || wire->alloc_bytes == nullptr)) {
        return refuse(host_create_wire_status::MALFORMED, "count > 0 with a null array");
    }
    host_create_record rec;
    rec.present = true;
    for (uint64_t i = 0; i < wire->count; ++i) {
        const int32_t d = wire->device[i];
        if (d < 0 || d >= GGML_SYCL_MAX_DEVICES) {
            return refuse(host_create_wire_status::MALFORMED, "device[" + std::to_string(i) +
                                                                  "] = " + std::to_string(d) + " is outside [0, " +
                                                                  std::to_string(GGML_SYCL_MAX_DEVICES) + ")");
        }
        const uint64_t b = wire->alloc_bytes[i];
        if (b > (uint64_t) SIZE_MAX) {
            return refuse(host_create_wire_status::MALFORMED,
                          "alloc_bytes[" + std::to_string(i) + "] overflows size_t");
        }
        rec.device.push_back((int) d);
        rec.alloc_bytes.push_back((size_t) b);
    }
    *out = std::move(rec);
    if (why) {
        why->clear();
    }
    return host_create_wire_status::PRESENT;
}

namespace host_create_detail {

struct pending_slot {
    bool               occupied = false;
    host_create_tag    tag      = 0;
    host_create_record record;
    std::string        why;  // why the stored record is absent, when it is
};

inline pending_slot & pending() {
    static thread_local pending_slot slot;
    return slot;
}

}  // namespace host_create_detail

// The export's body.  Returns true when it stored something.  A missing bound
// candidate (tag 0) and a null record are no-ops that leave a pending record
// alone; every other input, including an absent or malformed record, is stored
// (as absent where it is not PRESENT) so the stage fails closed.  `why` is filled
// with the reason a non-PRESENT record was stored or a call was a no-op.
inline bool host_create_pending_set(const ggml_sycl_host_create_record * wire,
                                    host_create_tag                      tag,
                                    std::string *                        why = nullptr) {
    if (tag == 0) {
        if (why) {
            *why = "no bound candidate";
        }
        return false;
    }
    if (wire == nullptr) {
        if (why) {
            *why = "null record";
        }
        return false;
    }
    host_create_detail::pending_slot & slot = host_create_detail::pending();
    std::string                        reason;
    host_create_record                 rec;
    host_create_record_from_wire(wire, &rec, &reason);
    slot.occupied = true;
    slot.tag      = tag;
    slot.record   = std::move(rec);
    slot.why      = reason;
    if (why) {
        *why = reason;
    }
    return true;
}

// Move the pending record out, ONCE.  The slot is empty afterwards.  Nothing
// pending, or a record published under another candidate, reads as absent with
// `why` naming which.
inline host_create_record host_create_pending_take(host_create_tag tag, std::string * why = nullptr) {
    host_create_detail::pending_slot & slot = host_create_detail::pending();
    host_create_record                 out;
    std::string                        reason;
    if (!slot.occupied) {
        reason = "no record published";
    } else if (slot.tag != tag) {
        reason = "record published under another candidate (" + std::to_string((unsigned long long) slot.tag) +
                 ", the stage is " + std::to_string((unsigned long long) tag) + ")";
    } else {
        out    = std::move(slot.record);
        reason = slot.why;
    }
    slot = host_create_detail::pending_slot();
    if (why) {
        *why = reason;
    }
    return out;
}

// The stage-scoped holder: takes the pending record once at construction, serves
// it to every read inside the scope, restores the enclosing holder on exit.  The
// per-device populates call current() and share one record.
//
// Holders are strictly LIFO and must be destroyed on the thread that constructed
// them: the current-holder pointer is thread_local and a holder restores its
// predecessor, so destroying out of order, or on another thread, leaves a dangling
// pointer behind.  Scope them as automatic variables; the destructor also runs when
// an exception unwinds the stage, so a thrown stage leaves no holder behind.
class host_create_stage_holder {
  public:
    explicit host_create_stage_holder(host_create_tag tag) : previous_(current_slot()) {
        record_        = host_create_pending_take(tag, &why_);
        current_slot() = this;
    }

    ~host_create_stage_holder() { current_slot() = previous_; }

    host_create_stage_holder(const host_create_stage_holder &)             = delete;
    host_create_stage_holder & operator=(const host_create_stage_holder &) = delete;

    const host_create_record & get() const { return record_; }

    const std::string & why() const { return why_; }

    // The innermost live holder on this thread, or nullptr outside any stage.
    static const host_create_stage_holder * current() { return current_slot(); }

  private:
    static host_create_stage_holder *& current_slot() {
        static thread_local host_create_stage_holder * cur = nullptr;
        return cur;
    }

    host_create_stage_holder * previous_;
    host_create_record         record_;
    std::string                why_;
};

}  // namespace ggml_sycl
