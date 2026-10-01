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

Allowlist entries added in S2b (clause e): the raw allocator chain's three links in `unified-cache.cpp`
(`E-CHAIN-ALIGNED`, `E-CHAIN-TRACKED`, `E-CHAIN-RAW`; canonical contract sections 3 and 9.1, 23mk census row
`:18789/:18830/:1433`) and the three raw calls in vendored `dpct/helper.hpp` (`E-DPCT-MALLOC`,
`E-DPCT-DEVMEM-DEVICE`, `E-DPCT-DEVMEM-SHARED`; rulings M247). Each pins its function and count, so a second call, a
renamed function or a swapped allocator fails. The canonical contract has no row for dpct. Outside `dpct/helper.hpp`
the names `dpct_malloc` (identifier) and `device_memory`, `global_memory`, `constant_memory`, `shared_memory` (type
names) are clause (e) hits; a variable or parameter that is merely spelled `device_memory` is not.

Finding codes added by clause (c): `C-COHORT` (a copy of a request that is handed on without its own cohort
literal) and `C-SITE` (a wrapper from one request type to another that does not copy the source's site fields).
`DEFER-C` now marks only what the gate still cannot follow (a helper with several returns, a lambda or function-pointer call).

Every `E-RAW` debt entry also carries `fate` and `cite`. `fate` is `deleted-by-<step>`, `converted-by-<step>`,
`sanctioned-internal`, `sanctioned-vendored` (upstream code we do not edit) or `pending-disposition`. A `sanctioned-internal` entry is one no step will ever shrink:
it is a candidate for the allowlist, and moving it there is the lead's decision, not the implementer's.
`pending-disposition` marks an entry whose fate nobody has ruled on yet; its `cite` says what is known. Every E-RAW entry needs a `cite`, a ticket id or a design/census row of at
least 12 characters.

## Key shape

`file::function::node-kind:variable:text-hash#ordinal`. The text hash is of the construction's normalized
declaration text plus the text of every later assignment bound to that declaration (comments dropped,
whitespace collapsed), so a write added to, or removed from, a listed construction moves its key. The ordinal counts only identical VIOLATING constructions within the
same function, so adding a compliant twin cannot move a listed key. No line or byte offset appears, so adding code above a construction does not move its key;
changing the construction does, which is when its entry should be revisited.

## Dependency

The gate parses C++ with tree-sitter: `pip install tree-sitter-language-pack` in the python3 that CMake finds.
A missing module is a FAIL naming it, never a skip.

## Shrinking the debt

Fix the violation, then delete its entry in the same commit. To rewrite the file:

    python3 scripts/check-sycl-alloc-zone-contract.py --write-debt

`--write-debt` refuses to add entries. `--allow-growth` exists for a deliberate key migration only (a change to
the key shape); do not use it to admit a new violation. The ctest never passes either flag.
