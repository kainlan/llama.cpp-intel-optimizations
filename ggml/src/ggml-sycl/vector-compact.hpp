#pragma once
// One-pass, in-place removal of the elements a visitor picks from a std::vector.
//
// Why: the unified cache's deferred-free walks dropped each ready entry with `it = v.erase(it)`. Each erase from the
// middle move-assigns every later element down one slot, so a walk that drops most of n entries costs O(n^2) moves.
// Those entries are heavy (a handle with lease-state locking, an allocation reference and a sycl::event), and one evict
// boundary can queue ten thousand of them that all become ready together: the walk then spent seconds on the thread
// that submits the next graph. This helper does the same walk in O(n) moves.
//
// Contract:
// - visit(T &) is called once per element, in order. It returns true to drop the element.
// - O(n) moves: a kept element moves at most once, to its compacted slot, and not at all when it is already there.
// - Survivors keep their relative order.
// - visit may change a kept element in place, and the change persists.
// - A dropped element is destroyed during its own visit, before the next element is visited, so what its destructor
//   releases is released when and under the locks that erase() released it. visit may move the element out itself;
//   the moved-from shell is then what is destroyed, so T's moved-from state must be safe to destroy and to assign over.
// - If visit throws at element i, v is valid when the exception propagates: the survivors of [0, i) in order, then
//   elements [i, n) as they were (element i as visit left it), with no moved-from element left behind.
// Requires T's move constructor and move assignment not to throw: the closing erase runs in a destructor, during the
// unwinding when visit throws.
#include <cstddef>
#include <utility>
#include <vector>

namespace ggml_sycl {

template <typename T, typename Alloc, typename Visit>
size_t vector_compact_visit(std::vector<T, Alloc> & v, Visit && visit) {
    // Elements [0, w) are the survivors so far; [w, i) are moved-from shells. The guard erases the shells on every
    // exit, normal or by an exception from visit, so the vector never holds one. A catch-and-rethrow would do the same
    // on the throwing path, but a destructor covers both paths with one statement.
    struct close_gap {
        std::vector<T, Alloc> & v;
        size_t &                w;
        size_t &                i;

        ~close_gap() { v.erase(v.begin() + w, v.begin() + i); }
    };

    const size_t n = v.size();
    size_t       w = 0;
    size_t       i = 0;
    close_gap    gap{ v, w, i };
    for (; i < n; ++i) {
        if (visit(v[i])) {
            T dead = std::move(v[i]);  // destroyed here, before the next visit
            (void) dead;
        } else {
            if (w != i) {
                v[w] = std::move(v[i]);
            }
            ++w;
        }
    }
    return n - w;
}

}  // namespace ggml_sycl
