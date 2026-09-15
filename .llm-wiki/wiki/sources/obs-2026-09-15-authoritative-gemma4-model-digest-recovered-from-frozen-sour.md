---
type: source
title: "Observation: Authoritative Gemma4 model digest recovered from frozen sources"
tags:
  - gemma4
  - provenance
  - sha256
  - fa
  - supervisor
status: observation
created: 2026-09-15
updated: 2026-09-15
slug: obs-2026-09-15-authoritative-gemma4-model-digest-recovered-from-frozen-sour
relevance: high
observed_at: 2026-09-15T00:06:24.212Z
source_context: FA launch preparation after independent QUALITY67bbaaf9; no real model execution
---

# ⭐ Observation: Authoritative Gemma4 model digest recovered from frozen sources

Frozen /home/kainlan/3aos-fa-closure-companion-pgophgop/worker.py lines29-30 and retained /home/kainlan/3aos-ab4-no-debug-ZbodP3/run-1/result.json and /home/kainlan/3aos-ab4-fa-route-amesh_en/run-1/result.json agree on /models/stock-gemma-4-E4B-it.Q8_0.gguf SHA256 f8854aa4480df62585a279e7ca0a881554fc18a41c59c4f62642d16a2ae47012 (64hex). Earlier compacted summaries dropped/transposed a character. This is retained-source provenance, NOT a fresh scan of current model bytes. Before any ON run the lead must verify current bytes, archive/runtime/env/device/kernel/memory/portal state and qualified external GPU-lock/controller custody. Frozen beb8 FA controller itself does NOT own GPU.lock; its slot_released field certifies owned-subtree retirement only. Tracker c-wnzi/c-lsie on llama.cpp-36a3 holds the read-only launch packet; no execution grant issued.

*Relevance: high*
*Context: FA launch preparation after independent QUALITY67bbaaf9; no real model execution*
*Tags: gemma4 provenance sha256 fa supervisor*

---
*Observed: 2026-09-15T00:06:24.212Z*
