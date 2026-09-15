---
type: source
title: PermissionError repr loses the denied path
status: insight
category: bugfix
created: 2026-09-15
updated: 2026-09-15
slug: permissionerror-repr-loses-denied-path
---

# PermissionError repr loses the denied path

`repr(PermissionError(...))` can omit `filename` and `filename2`, even when the exception carries them. The first native FA ON attempt retained only `PermissionError(13, 'Permission denied')`, so its exact denied path cannot be recovered from that record. An audit marker before a read does not establish that the read succeeded.

The private diagnostics revision in `/home/kainlan/3aos-native-attribution-fed9` preserves bounded exception type/errno/path/path2, attempted-operation context and source-only traceback frames without reading source lines, locals or environment. Same-fixture exact-boundary component RED→GREEN was independently reviewed under tasks oj8m/9wzm; the initial setup-only pin error remains separately counted. This is attribution improvement, not a permission fix or native qualification.

Operation context is the last attempted marker, not proof of the failing syscall. Corroborate with exception filename and source frames. Output clipping does not bound intermediate `repr`/decode allocations or guarantee arbitrary custom exceptions cannot raise. See [[obs-2026-09-15-first-native-fa-on-preflight-failure-and-retained-gpu-custod]].

*Category: bugfix*

---
*Captured: 2026-09-15*

## Related

_Add links to related pages._
