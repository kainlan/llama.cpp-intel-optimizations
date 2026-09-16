---
type: source
title: ESRCH invalidates only the current process snapshot row
status: insight
category: bugfix
created: 2026-09-16
updated: 2026-09-16
slug: pidfd-esrch-invalidates-only-current-snapshot-row
---

# ESRCH invalidates only the current process snapshot row

After a real pidfd_open returns ESRCH for a tentative child, reusing that same census row in a later process-group audit can create false, sticky ownership ambiguity. The narrow correction records the exact PID/birth/full-row tuple only for that ESRCH and excludes only its identical row in the same census; reset this record on every census. It must not admit a process, erase historical ambiguity, suppress live unknown processes, or weaken retained live-pidfd handling.

Private shared candidate `04e7ab13…ddfce4` implements four added lines over unchanged `9695abb2…436a23`. The unchanged real-kernel scenario produced baseline exit1/0.679s and fixed exit0/0.640s; all three fixed after-close receipts qualified. Five guards/35 predicates and one genuine live-metadata regression/6 checks also passed. Independent SPEC `lh04` and QUALITY `fveh` passed; task `pfp4` closed for this narrow correction, not native/ABC/application qualification or historical lease release. Evidence: `2bf2b5554ccc50523e6f7b8a1d1092ee4b6c4806` and `7022279f4c76159f0ef659fb4de0176db3be3654`.

See [[obs-2026-09-16-stale-snapshot-fix-passes-real-kernel-red-green-replay]].

*Category: bugfix*

---
*Captured: 2026-09-16*

## Related

_Add links to related pages._
