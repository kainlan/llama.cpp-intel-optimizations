---
type: source
title: "Observation: Configure blocked by obsolete checkout hook and CMake identities"
tags:
  - sycl
  - stage-b
  - hooks
  - beads
  - cmake
  - blockers
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-configure-blocked-by-obsolete-checkout-hook-and-cmake-identi
relevance: high
observed_at: 2026-09-16T17:19:20.156Z
source_context: Bounded existing configure-path readiness check
---

# ⭐ Observation: Configure blocked by obsolete checkout hook and CMake identities

After accepted Stage A, read-only B preparation under llama.cpp-uvgc found concrete blockers. Shared /Apps/llama.cpp/.git/hooks/post-checkout is executable0755,491bytes,regularnotlink,SHA dcd7dd313bec02d8ddda36043a98977bcca10eb5d33df26d4eae3aefbd7292b3. Its complete shim only checks command-v bd (silentexit0ifabsent), then exec bd hooks run post-checkout "$@". Neither hook nor bd executed. Existing B validator rejects any executable named hook; hash approval cannot bypass it. User permission is required before retiring only that shared hook by unique backup rename preserving bytes/mode; no chmod/validator weakening/otherhook changes. w4ol/c-c68k records scope. Separate gap: declared CMake wrapper202bytes SHA22487d... has shebang /home/kainlan/miniconda3/bin/python3.13 and imports cmake; interpreter/package/real ELF dependencies not in seven declared identities, and B requires exact A.tool_identities equality, so cannot blindly append or substitute pins. Nested postlink proof needs concrete path/hash selection. B does not automatically inherit Stage A whole-E budget accounting. uvgc reset open/blocked, no B G/A/reservations/checkout/configure/build performed. Source remains clean d7 and CPP8c8 uncompiled.

*Relevance: high*
*Context: Bounded existing configure-path readiness check*
*Tags: sycl stage-b hooks beads cmake blockers*

---
*Observed: 2026-09-16T17:19:20.156Z*
