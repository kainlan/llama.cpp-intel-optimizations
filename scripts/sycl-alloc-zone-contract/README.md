# sycl-alloc-zone-contract data

Read by `scripts/check-sycl-alloc-zone-contract.py` (ctest `test-sycl-alloc-zone-contract`,
llama.cpp-23mk S2). JSON has no comments, so each file also carries a `_doc` field.

- `allowlist.json`: permanent exemptions, each with a `reason`. Matched by `(code, file, function[, name])`
  and pinned to an exact `count`. An entry that matches nothing fails (a renamed exempt function protects
  nothing), and so does one whose count differs from what it covers. Hand-edited.
- `debt.json`: the tree's current violations, keyed by construction node. Shrink-only in both directions:
  a violation that is not listed fails, and a listed entry that no longer violates fails, naming it.

Every `E-RAW` debt entry also carries `fate` and `cite`. `fate` is `deleted-by-<step>`, `converted-by-<step>`
or `sanctioned-internal`. A `sanctioned-internal` entry is one no step will ever shrink: it is a candidate for
the allowlist, and moving it there is the lead's decision, not the implementer's.

## Key shape

`file::function::node-kind:variable:text-hash#ordinal`. The text hash is of the construction's normalized
text (comments dropped, whitespace collapsed). The ordinal counts only identical constructions within the
same function. No line or byte offset appears, so adding code above a construction does not move its key;
changing the construction does, which is when its entry should be revisited.

## Shrinking the debt

Fix the violation, then delete its entry in the same commit. To rewrite the file:

    python3 scripts/check-sycl-alloc-zone-contract.py --write-debt

`--write-debt` refuses to add entries. `--allow-growth` exists for a deliberate key migration only (a change to
the key shape); do not use it to admit a new violation. The ctest never passes either flag.
