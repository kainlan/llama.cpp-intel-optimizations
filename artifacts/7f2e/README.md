# Immutable audit manifest for `9a0670712`

## Identity and closure statement

This directory freezes the completed `llama.cpp-7f2e` source/history audit.

- audited commit: `9a06707120dfa4595b46f2241ec40bbbd8476959` (`sycl: checkpoint unified memory ownership work`)
- sole parent: `d892fa8c054c34528cb550881825b7082fa63d17`
- commit date: 2026-06-15 04:40:42 -0400
- census: **338 changed paths**, **4,087 textual zero-context hunks**
- audit outcome: every changed path and textual hunk is represented and classified; no hunk or path is unclassified; no additional 9a-specific runtime defect remained after the fixes and child outcomes below.

`paths.tsv` has exactly one row for every changed path, including blob identities and its textual-hunk count. `hunks.tsv` has exactly one row for every `git diff --unified=0` textual hunk, source coordinates, line counts, SHA-256 of the raw hunk, classification, and disposition. The binary `.beads/beads.db` path is intentionally present in the path census with zero textual hunks. `verify_manifest.py` regenerates both tables solely from the two immutable Git objects and compares bytes, so additions, omissions, duplicates, drift, blank classifications, or unexpected residual classes fail closed.

The classifications record the already completed lead read of all 338 files/4,087 hunks (tracker comment `llama.cpp-7f2e` `c-65dx`), rather than claiming that a keyword scan substitutes for reading. The deterministic rules preserve that review result at hunk granularity and distinguish non-semantic formatting, production semantic/control/lifetime changes, additions/deletions, docs/tests/scripts/tooling/metadata, selector provenance, and the known defect hunks.

## Classification summary

Path areas (338 total):

| Area | Paths |
|---|---:|
| production | 79 |
| test | 222 |
| documentation | 14 |
| script | 9 |
| tooling | 11 |
| tracker-metadata | 3 |

Textual hunk classifications (4,087 total):

| Classification | Hunks |
|---|---:|
| production:format-only | 999 |
| production:semantic-control-or-lifetime | 1,095 |
| production:semantic-replacement | 855 |
| production:additive | 513 |
| production:deletion | 181 |
| test | 197 |
| test:selector-default | 76 |
| documentation | 33 |
| documentation:selector-default | 13 |
| script | 28 |
| script:selector-default | 4 |
| tooling | 80 |
| tracker-metadata | 2 |
| tracker-metadata:selector-default | 7 |
| known-defect:expert-staging-role-order | 1 |
| known-defect:live-weight-abort | 1 |
| known-defect:live-zone-reset-abort | 2 |

The 100 `*:selector-default` rows count zero-context diff hunks containing selector literals, including repeated tracker prose; they are not the semantic selector-site census. The source/provenance audit `llama.cpp-966h` established the authoritative **82 sites across 54 files** (79 direct replacements plus 3 centralized defaults).

## Confirmed defects and outcomes

All SHAs below are descendants of the audited commit and ancestors of the manifest's creation HEAD.

| SHA / tracker | Outcome |
|---|---|
| `282069dd4702c820bb0ac53757572b516c342746` | Fixed the 9a-introduced `EXPERT_STAGING` shadowing defect by evaluating role before the widened `HOST_COMPUTE` / `EXPERT_CACHE` category fallback; transient staging again selects SCRATCH. |
| `06f8887a60724783c78a4012feeec17c32b95465` | Restored unified-cache prompt headroom/resource reservation behavior after the checkpoint lineage. |
| `acdb192d43bf6b36a1a8e227aceb12d38845ecf1` | Replaced 9a's abort for live model-weight leases with preserve-and-continue plus safe identity remapping. |
| `4afdb6d9f1c247dd62cbf6a9a9802447efb64df4` | Replaced 9a's host/device zone-reset aborts with refusal to reclaim another model's live allocations. |
| `llama.cpp-5cxw` / `4a3d190f0d31bbeb1630061a3d25aefae8320e76` | Closed: reviewed unified docs merged; permanent unified-cache authority, removed opt-out versus topology MODE, and ownership/reset wording reconciled; zero-finding final review. Do not restore the opt-out. |
| `llama.cpp-gza7` / `d27e309ce03a55c8e0222534c81d4d28f075c4b6` | Closed: tier publication now reflects current-model actual planner host placement, with always-active/load-order coverage. Lead build, pinned run, mutation RED, exact restoration, and restored GREEN passed. |
| `llama.cpp-966h` | Closed: exhaustive selector provenance census plus live device/PCI map; no source fix SHA was required by that audit. |

## Selector provenance and lead device map

The 9a selector audit found 82 semantic sites/54 files. Historical `AGENTS.md` provenance showed that the old ordinal mapping called `level_zero:1` the B50 and the new default `level_zero:0` the B580; ordinal defaults therefore do not preserve a physical-device identity. The audit requires caller-explicit selectors and device/PCI/capability/VRAM evidence for device-specific or performance claims.

Lead tracker comment `llama.cpp-966h` `c-696a` records the live capture under `GPU.lock` at main SHA `deb8e6eed` (`/tmp/966h-device-map.log`, 2026-08-04T08:37:50-04:00):

- `level_zero:0` = Intel Arc Pro B70 20.2.0, PCI `0000:03:00.0`, device `8086:e223`
- `level_zero:1` = Intel Arc Pro B50 20.1.0, PCI `0000:07:00.0`, device `8086:e212`
- `level_zero:2` = iGPU, PCI `0000:00:02.0`, device `8086:7d67`

This live map supersedes assumptions based on the historical ordinal labels while preserving both as provenance.

## EXPERT_STAGING role-first and ancestry proof

In 9a, `unified_alloc()` widened the WEIGHT branch to include `cat == HOST_COMPUTE || cat == EXPERT_CACHE` while leaving it before `role == EXPERT_STAGING`; the same hunk's comment says transient EXPERT_STAGING remains SCRATCH. Thus a combined `role=EXPERT_STAGING, category=HOST_COMPUTE` request silently selected WEIGHT.

`282069dd4` moves the `role == EXPERT_STAGING` check before that category-widened WEIGHT branch. Its commit message identifies `test_expert_staging_host_compute_zone_ownership()` as the exact RED contract. Mechanical ancestry checks return zero for both `git merge-base --is-ancestor 9a0670712 282069dd4` and `git merge-base --is-ancestor 282069dd4 HEAD`.

## Mechanical reproduction

Run from the repository root (no build or GPU required):

```sh
python3 artifacts/7f2e/verify_manifest.py
```

Expected invariant lines:

```text
changed_paths=338
textual_hunks=4087
classified_hunks=4087
unclassified_paths=0
unclassified_hunks=0
manifest_match=yes
```

Independent Git counts:

```sh
git diff --name-only --no-renames 9a0670712^ 9a0670712 -- | wc -l
# 338

git diff --no-ext-diff --no-renames --unified=0 9a0670712^ 9a0670712 -- \
  | awk '/^@@ / { n++ } END { print n+0 }'
# 4087
```

The verifier also checks commit/parent identity, sequential unique IDs, path-to-hunk referential integrity, per-path hunk sums, exact deterministic table bytes, non-empty classifications/dispositions, and absence of the residual `production:other` class.
