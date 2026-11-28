#pragma once
// One-pass, in-place removal of the elements a visitor picks from a std::vector.
//
// Why: the unified cache's deferred-free walks dropped each ready entry with `it = v.erase(it)`. Each erase from the
// middle move-assigns every later element down one slot, so a walk that drops most of n entries costs O(n^2) moves.
// Those entries are heavy (a handle with lease-state locking, an allocation reference and a sycl::event), and one evict
// boundary can queue ten thousand of them that all become ready together: the walk then spent seconds on the thread
// that submits the next graph. This helper does the same walk in O(n) moves.
// std::remove_if / std::erase_if will not do: their predicate may not modify or move the element, and remove_if
// releases a dropped element only when a later survivor is assigned over it, which moves the release to another time.
//
// Contract:
// - visit(T &) is called once per element, in order. It returns true to drop the element.
// - visit must not insert into, erase from or resize v: the element reference it holds would dangle. The callers
//   enforce this by holding the list's mutex across the walk.
// - O(n) moves: a kept element moves at most once, to its compacted slot, and not at all when it is already there.
// - Survivors keep their relative order.
// - visit may change a kept element in place, and the change persists.
// - A dropped element is destroyed as soon as its visit returns true, before the next element is visited, so what its
//   destructor releases is released when and under the locks that erase() released it. visit may move the element out
//   itself; the moved-from shell is then what is destroyed, so T's moved-from state must be safe to destroy and to
//   assign over.
// - If visit throws at element i, v is valid when the exception propagates: the survivors of [0, i) in order, then
//   elements [i, n) as they were, with element i as visit left it. A visit that moves the element out must do so as its
//   last step, after everything that can throw: if it moves element i out and then throws, the moved-from shell stays
//   in v at slot i.
// Requires T's move constructor and move assignment not to throw (checked at compile time): the closing erase runs in a
// destructor, during the unwinding when visit throws, and a drop is move-constructed out of its slot.
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace ggml_sycl {

template <typename T, typename Visit> size_t vector_compact_visit(std::vector<T> & v, Visit && visit) {
    static_assert(std::is_nothrow_move_constructible<T>::value && std::is_nothrow_move_assignable<T>::value,
                  "vector_compact_visit needs noexcept moves: a dropped element is move-constructed out of its slot, "
                  "and a throwing construct would leave element i half moved; the gap-closing erase move-assigns in a "
                  "destructor, so a move that throws while an exception from visit unwinds would call std::terminate");

    // Elements [0, w) are the survivors so far; [w, i) are moved-from shells. The guard erases the shells on every
    // exit, normal or by an exception from visit, so the vector never holds one. A catch-and-rethrow would do the same
    // on the throwing path, but a destructor covers both paths with one statement.
    struct close_gap {
        std::vector<T> & v;
        size_t &         w;
        size_t &         i;

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
