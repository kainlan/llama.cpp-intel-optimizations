---
type: source
title: Runtime-bound AST proof fingerprints
status: insight
category: testing
created: 2026-09-15
updated: 2026-09-15
slug: ast-proof-fingerprints-need-pinned-python
---

# Runtime-bound AST proof fingerprints

A source-control equivalence proof generated with Python3.13.5 used `ast.dump` output omitting empty fields (5181 characters). The same unchanged helper under pinned Python3.12.11 included empty fields (5785 characters), causing a SETUP assertion before any component/dummy invocation—not a production RED.

The correction compared BOTH the immutable tested original control and extracted helper under the same pinned3.12 runtime, proved structural equality (`a992993390651e7ad33c2d3b00c267679117058c2ffbe91cbab2c6e1bee287b1`), and recorded version/dump options. It did not merely replace the expected digest. The failed attempt and original3.13 proof remain preserved.

Shared helper: `/home/kainlan/glkg-build-adapter-prep-Sqc0l1iv/cleanup_floor.py`, SHA256 `761b78a78f54feb67f1815b40908ee93b51da31c1e8f43165179c40d5c296c87`. Correction commit5798344; final package1beb461. Three final caller controls passed; integrated SPEC and independent QUALITY accepted the bounded preparation scope. See [[authoritative-retirement-after-close]].

*Category: testing*

---
*Captured: 2026-09-15*

## Related

_Add links to related pages._
