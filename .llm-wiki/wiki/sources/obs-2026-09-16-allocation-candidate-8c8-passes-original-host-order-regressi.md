---
type: source
title: "Observation: Allocation candidate 8c8 passes original host-order regression"
tags:
  - sycl
  - glkg
  - allocation
  - regression-green
  - application
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-allocation-candidate-8c8-passes-original-host-order-regressi
relevance: high
observed_at: 2026-09-16T21:15:38.042Z
source_context: Actual runtime validation after returning to normal application workflow
---

# ⭐ Observation: Allocation candidate 8c8 passes original host-order regression

Actual application GREEN on8c8a0afaea0865dbf8ca13ba505c32dea559564e: existing host_inventory_initializes_zones ran once on B50, exit0/1.656s,1run1pass. Evidence /Apps/llama.cpp-wt-glkg-red-transaction/build/candidate8c8-host-order-SKPav1. Original d7 fixture SHA698bf7e21e19fd950993c35d28636d47fbbad38e0444a4de891857b746bb4834 unchanged. Nonempty279entries/210device/69host; late staging configures zones before993280-byte publicWEIGHT allocation; accounting restores exactly; rollbackresult10/candidate_absent1/active0/models0/baseline0;326MiBdevicearena+40pinnedchunksreleased, Shmemunchanged. Three host-only effect-drain selectors passed; tensor-placement43/43passed. Primitivebuild firsttimedout124/250.07s, continuation0/902.88s, test0/.30s; priorpassesnotrepeated. Logs candidate8c8-primitives-* and candidate8c8-tensor-continuation-*. Libccl.so.2/.so.1 linkerwarning retained. No sourcechanges or newinfrastructure. Broader staged-failure/empty-staging integration and integrated provisioning/load-end race remainunverified. UserapprovedoldGPU.lock backup at /Apps/llama.cpp/.git/GPU.lock.retired-20260916T203337Z-ffac8dcc9d574313969d5ac4ad212b37; newordinarytestlockreleased afterserialGPUtests. Do not redo completed checks or reimposeoldreservationblock.

*Relevance: high*
*Context: Actual runtime validation after returning to normal application workflow*
*Tags: sycl glkg allocation regression-green application*

---
*Observed: 2026-09-16T21:15:38.042Z*
