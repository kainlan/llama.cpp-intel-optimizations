# sycl-alloc-zone-contract data

Read by `scripts/check-sycl-alloc-zone-contract.py` (ctest `test-sycl-alloc-zone-contract`,
llama.cpp-23mk S2). JSON has no comments, so each file also carries a `_doc` field.

- `allowlist.json`: permanent exemptions, each with a `reason`. An entry names its subject, either
  `file` + `function` or a node `key`, plus the raw `name` for an E-RAW entry, and pins an exact `count` (the
  count is required). `name` is what stops a different allocator swapped into an exempt function from riding
  the exemption. An entry that matches nothing fails (a renamed exempt function protects nothing), and so does
  one whose count differs from what it covers. Hand-edited.
- `debt.json`: the tree's current violations, keyed by construction node. Shrink-only in both directions:
  a violation that is not listed fails, and a listed entry that no longer violates fails, naming it.

Every `E-RAW` debt entry also carries `fate` and `cite`. `fate` is `deleted-by-<step>`, `converted-by-<step>`,
`sanctioned-internal` or `pending-disposition`. A `sanctioned-internal` entry is one no step will ever shrink:
it is a candidate for the allowlist, and moving it there is the lead's decision, not the implementer's.
`pending-disposition` marks an entry whose fate nobody has ruled on yet; its `cite` says what is known.

## Key shape

`file::function::node-kind:variable:text-hash#ordinal`. The text hash is of the construction's normalized
declaration text plus the text of every later assignment bound to that declaration (comments dropped,
whitespace collapsed), so a write added to, or removed from, a listed construction moves its key. The ordinal counts only identical constructions within the
same function. No line or byte offset appears, so adding code above a construction does not move its key;
changing the construction does, which is when its entry should be revisited.

## Shrinking the debt

Fix the violation, then delete its entry in the same commit. To rewrite the file:

    python3 scripts/check-sycl-alloc-zone-contract.py --write-debt

`--write-debt` refuses to add entries. `--allow-growth` exists for a deliberate key migration only (a change to
the key shape); do not use it to admit a new violation. The ctest never passes either flag.
