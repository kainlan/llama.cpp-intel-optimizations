---
type: source
title: Hook retirement without validator bypass
status: insight
category: workflow
created: 2026-09-16
updated: 2026-09-16
slug: retired-hook-backup-without-validator-bypass
---

# Hook retirement without validator bypass

An executable obsolete `bd` post-checkout shim blocked the existing guarded checkout validator. Hash approval would not satisfy its executable-hook refusal. With explicit user permission, retiring only that canonical hook by atomic no-overwrite rename preserved its bytes, inode and0755 permissions without weakening the validator or running the hook.

The backup is recorded in [[sources/obs-2026-09-16-user-approved-obsolete-post-checkout-hook-retirement]]. Other shared hooks remain untouched. Restoration requires separate authorization and must never overwrite a newly installed canonical hook. Clearing this operational blocker grants no configuration/build authority and does not resolve independent tool-dependency checks.

*Category: workflow*

---
*Captured: 2026-09-16*

## Related

_Add links to related pages._
