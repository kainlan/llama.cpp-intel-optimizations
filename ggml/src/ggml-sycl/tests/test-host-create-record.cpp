// Host tests of the host-weight create-set record (llama.cpp-moua L4 1c; th32pre
// test 1r): the wire ABI, the one converter, the thread-local pending slot and the
// stage-scoped holder.  The wire struct is built by hand here, so the oracle is the
// declared layout and not the converter.
//
// Arms:
//   (a) a short struct_size reads as absent, discriminating: a fully valid v1 record
//       laid out with struct_size 16 must be absent, the same bytes with 32 present,
//       a longer caller struct read to the v1 size only;
//   (b) malformed records are absent with a reason: a null array under count > 0, a
//       count above the cap;
//   (b2) a present value other than 0 or 1 is malformed;
//   (c) device out of range, with the boundary accepted;
//   (d) the arrays are copied, not pointed at;
//   (e) consume once; a stale record under another tag; tag 0 and a null record are
//       no-ops that leave a pending record alone; the slot is per thread;
//   (f) the holder serves one record to every read in its scope, is empty after, and
//       restores the enclosing holder, and an exception unwinds through it cleanly.
// Arm (g), that the export body and the dry-run are each one call of the converter,
// is a source-token check and lives in the F8 gate with the export (L4 step 3).
//
// The build is -DNDEBUG, so CHECK is explicit and always runs.

#include "../host-create-record.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

using namespace ggml_sycl;

namespace {

// A valid two-entry v1 record over caller-owned arrays.
struct wire_fixture {
    std::vector<int32_t>         device      = { 0, 1 };
    std::vector<uint64_t>        alloc_bytes = { 4096, 8192 };
    ggml_sycl_host_create_record wire;

    wire_fixture() {
        std::memset(&wire, 0, sizeof(wire));
        wire.struct_size = (uint32_t) sizeof(wire);
        wire.present     = 1;
        wire.count       = 2;
        wire.device      = device.data();
        wire.alloc_bytes = alloc_bytes.data();
    }
};

bool is_two_entry(const host_create_record & r) {
    return r.present && r.device == std::vector<int>({ 0, 1 }) && r.alloc_bytes == std::vector<size_t>({ 4096, 8192 });
}

void arm_a_short_struct_size() {
    wire_fixture f;
    CHECK(sizeof(f.wire) == 32, "a: the v1 layout is 32 bytes on this ABI");

    host_create_record out;
    std::string        why;
    f.wire.struct_size = 16;
    CHECK(host_create_record_from_wire(&f.wire, &out, &why) == host_create_wire_status::ABSENT && !out.present,
          "a: a valid record with struct_size 16 reads as ABSENT");
    CHECK(why.find("struct_size") != std::string::npos, "a: the reason names struct_size");

    f.wire.struct_size = 32;
    CHECK(host_create_record_from_wire(&f.wire, &out, &why) == host_create_wire_status::PRESENT && is_two_entry(out),
          "a: the same bytes with struct_size 32 read present with both entries");

    // A longer caller struct: v1 prefix, then bytes this reader has never heard of.
    struct longer {
        ggml_sycl_host_create_record v1;
        uint64_t                     future_field;
    } big;

    std::memset(&big, 0xAB, sizeof(big));
    big.v1             = f.wire;
    big.v1.struct_size = (uint32_t) sizeof(big);
    CHECK(host_create_record_from_wire(&big.v1, &out, &why) == host_create_wire_status::PRESENT && is_two_entry(out),
          "a: a longer caller struct is read to the v1 size");
}

void arm_a2_other_absents() {
    host_create_record out;
    std::string        why;
    CHECK(host_create_record_from_wire(nullptr, &out, &why) == host_create_wire_status::ABSENT && !out.present,
          "a2: a null pointer is absent");
    wire_fixture f;
    f.wire.present = 0;
    CHECK(host_create_record_from_wire(&f.wire, &out, &why) == host_create_wire_status::ABSENT && !out.present,
          "a2: present = 0 is absent even over valid arrays");
    wire_fixture g;
    g.wire.count       = 0;
    g.wire.device      = nullptr;
    g.wire.alloc_bytes = nullptr;
    CHECK(host_create_record_from_wire(&g.wire, &out, &why) == host_create_wire_status::PRESENT && out.present &&
              out.device.empty(),
          "a2: present with count 0 is a present EMPTY record, not absent");
}

void arm_b_malformed() {
    host_create_record out;
    std::string        why;
    {
        wire_fixture f;
        f.wire.device = nullptr;
        CHECK(host_create_record_from_wire(&f.wire, &out, &why) == host_create_wire_status::MALFORMED && !out.present &&
                  !why.empty(),
              "b: count 1+ with a null device array is malformed, with a reason");
    }
    {
        wire_fixture f;
        f.wire.alloc_bytes = nullptr;
        CHECK(host_create_record_from_wire(&f.wire, &out, &why) == host_create_wire_status::MALFORMED && !out.present,
              "b: count 1+ with a null alloc_bytes array is malformed");
    }
    {
        wire_fixture f;
        f.wire.count = HOST_CREATE_RECORD_MAX_COUNT + 1;
        CHECK(host_create_record_from_wire(&f.wire, &out, &why) == host_create_wire_status::MALFORMED && !out.present &&
                  why.find("cap") != std::string::npos,
              "b: a count above the cap is malformed and never walks the arrays");
        f.wire.count = 2;
        CHECK(host_create_record_from_wire(&f.wire, &out, &why) == host_create_wire_status::PRESENT,
              "b: control: the same struct at count 2 is fine");
    }
}

void arm_b2_present_values() {
    host_create_record out;
    std::string        why;
    wire_fixture       f;
    for (uint32_t v : { 2u, 3u, 0x100u, 0xFFFFFFFFu }) {
        f.wire.present = v;
        CHECK(host_create_record_from_wire(&f.wire, &out, &why) == host_create_wire_status::MALFORMED && !out.present &&
                  why.find("present") != std::string::npos,
              "b2: a present value other than 0 or 1 is malformed, never read as present");
    }
    f.wire.present = 1;
    CHECK(host_create_record_from_wire(&f.wire, &out, &why) == host_create_wire_status::PRESENT,
          "b2: control: present = 1 is fine");
}

void arm_c_device_range() {
    host_create_record out;
    std::string        why;
    wire_fixture       f;
    f.device[0] = GGML_SYCL_MAX_DEVICES;
    CHECK(host_create_record_from_wire(&f.wire, &out, &why) == host_create_wire_status::MALFORMED && !out.present &&
              why.find("device[0]") != std::string::npos,
          "c: device == GGML_SYCL_MAX_DEVICES is out of range, and the reason names the entry");
    f.device[0] = -1;
    CHECK(host_create_record_from_wire(&f.wire, &out, &why) == host_create_wire_status::MALFORMED && !out.present,
          "c: device -1 is out of range");
    f.device[0] = 0;
    f.device[1] = GGML_SYCL_MAX_DEVICES;
    CHECK(host_create_record_from_wire(&f.wire, &out, &why) == host_create_wire_status::MALFORMED,
          "c: a bad LATER entry refuses the whole record");
    f.device[1] = GGML_SYCL_MAX_DEVICES - 1;
    CHECK(host_create_record_from_wire(&f.wire, &out, &why) == host_create_wire_status::PRESENT &&
              out.device[1] == GGML_SYCL_MAX_DEVICES - 1,
          "c: the last valid ordinal is accepted");
}

void arm_d_arrays_are_copied() {
    std::string        why;
    host_create_record snapshot;
    {
        wire_fixture f;
        CHECK(host_create_pending_set(&f.wire, 7, &why), "d: the publish stores");
        // The caller scribbles over and releases its arrays.
        f.device[0]      = 99;
        f.alloc_bytes[1] = 0xDEAD;
        f.device.clear();
        f.device.shrink_to_fit();
        f.alloc_bytes.clear();
        f.alloc_bytes.shrink_to_fit();
    }
    snapshot = host_create_pending_take(7, &why);
    CHECK(is_two_entry(snapshot), "d: the stored record still equals the original values");
}

void arm_e_consume_once() {
    std::string why;
    // Drain anything a previous arm left.
    (void) host_create_pending_take(0, nullptr);
    {
        wire_fixture f;
        CHECK(host_create_pending_set(&f.wire, 11, &why), "e: publish under tag 11");
        const host_create_record first = host_create_pending_take(11, &why);
        CHECK(is_two_entry(first), "e: the first take is present");
        const host_create_record second = host_create_pending_take(11, &why);
        CHECK(!second.present && !why.empty(), "e: the second take, with no publish between, is absent with a reason");
    }
    {
        wire_fixture f;
        CHECK(host_create_pending_set(&f.wire, 21, &why), "e: publish under tag 21");
        const host_create_record other = host_create_pending_take(22, &why);
        CHECK(!other.present && why.find("another candidate") != std::string::npos,
              "e: a record published under tag 21 is absent under tag 22, with the mismatch named");
        CHECK(!host_create_pending_take(21, &why).present, "e: the mismatched take discards the stale record");
    }
    {
        // tag 0 and a null record never clobber a pending record.
        wire_fixture f;
        CHECK(host_create_pending_set(&f.wire, 31, &why), "e: publish under tag 31");
        wire_fixture other;
        other.device[0] = 5;
        CHECK(!host_create_pending_set(&other.wire, 0, &why) && why == "no bound candidate",
              "e: a publish with no bound candidate is a no-op");
        CHECK(!host_create_pending_set(nullptr, 31, &why) && why == "null record", "e: a null record is a no-op");
        CHECK(is_two_entry(host_create_pending_take(31, &why)),
              "e: the valid record published earlier is still there, not overwritten with absent");
    }
    {
        // a malformed publish IS stored, as absent with its reason (fails closed).
        wire_fixture f;
        f.wire.device = nullptr;
        CHECK(host_create_pending_set(&f.wire, 41, &why) && !why.empty(), "e: a malformed record is stored");
        const host_create_record r = host_create_pending_take(41, &why);
        CHECK(!r.present && why.find("null array") != std::string::npos,
              "e: it reads back absent, carrying the malformed reason");
    }
    {
        // the slot is per thread.
        wire_fixture f;
        CHECK(host_create_pending_set(&f.wire, 51, &why), "e: publish on this thread");
        bool other_present = true;
        std::thread([&] {
            other_present = host_create_pending_take(51, nullptr).present;
            wire_fixture g;
            std::string  w;
            host_create_pending_set(&g.wire, 51, &w);
        }).join();
        CHECK(!other_present, "e: another thread does not see this thread's record");
        CHECK(is_two_entry(host_create_pending_take(51, &why)),
              "e: and its own publish did not touch this thread's slot");
    }
}

void arm_f_holder() {
    std::string why;
    (void) host_create_pending_take(0, nullptr);
    CHECK(host_create_stage_holder::current() == nullptr, "f: no holder outside a stage");
    wire_fixture f;
    CHECK(host_create_pending_set(&f.wire, 61, &why), "f: publish");
    {
        host_create_stage_holder stage(61);
        CHECK(host_create_stage_holder::current() == &stage, "f: the stage is the current holder");
        for (int device = 0; device < 4; ++device) {
            CHECK(is_two_entry(host_create_stage_holder::current()->get()),
                  "f: every per-device read in the stage sees the same present record");
        }
        CHECK(!host_create_pending_take(61, nullptr).present, "f: the stage consumed the pending slot at entry");

        // A nested stage restores the outer holder on exit.
        wire_fixture inner;
        inner.device[0] = 2;
        CHECK(host_create_pending_set(&inner.wire, 62, &why), "f: publish for the nested stage");
        {
            host_create_stage_holder nested(62);
            CHECK(host_create_stage_holder::current() == &nested && nested.get().device[0] == 2,
                  "f: the nested holder serves its own record");
        }
        CHECK(host_create_stage_holder::current() == &stage && is_two_entry(stage.get()),
              "f: the outer holder is restored, unchanged, after the nested scope");
    }
    CHECK(host_create_stage_holder::current() == nullptr, "f: the holder is gone after the stage");

    // An exception unwinding through nested stages leaves no holder behind: the outer is
    // restored by the inner's destructor and then removed by its own.
    {
        wire_fixture outer;
        CHECK(host_create_pending_set(&outer.wire, 71, &why), "f: publish for the outer stage");
        bool caught = false;
        try {
            host_create_stage_holder outer_stage(71);
            wire_fixture             inner;
            inner.device[0] = 3;
            CHECK(host_create_pending_set(&inner.wire, 72, &why), "f: publish for the inner stage");
            host_create_stage_holder inner_stage(72);
            CHECK(host_create_stage_holder::current() == &inner_stage, "f: the inner stage is current");
            throw 42;
        } catch (int) {
            caught = true;
        }
        CHECK(caught, "f: the exception reached the catch");
        CHECK(host_create_stage_holder::current() == nullptr, "f: an unwound stage leaves no holder behind");
    }

    // A stage with nothing published holds absent, with the reason.
    {
        host_create_stage_holder stage(63);
        CHECK(!stage.get().present && !stage.why().empty(), "f: an unpublished stage holds absent, with a reason");
    }
}

}  // namespace

int main() {
    arm_a_short_struct_size();
    arm_a2_other_absents();
    arm_b_malformed();
    arm_b2_present_values();
    arm_c_device_range();
    arm_d_arrays_are_copied();
    arm_e_consume_once();
    arm_f_holder();
    std::printf("test-host-create-record: PASSED\n");
    return 0;
}
