# HM Task 12: pointer-table campaign rerun — bc35296d5 (2026-08-15)

Bar: the 6 named campaign groups x 2 runs x 2 GPUs (24 cells), after the two
ytqx fixture repairs (93971a064 ready-event, bc35296d5 DIRECT extent).

Result: 4/4 ctest invocations rc=0 (B70 r1/r2, B50 r1/r2), plus one verbose
confirmation run on B70 showing ALL 18 suite cases executing and passing —
including the formerly 4/4-failing "MoE pointer-table cache stores IDs only"
and the identity-negative host ("storage-handle-first AoS route and fail-closed
variants"). 24/24 cells met with execution verified (passing ctest output is
suppressed; the -V run is the case-count proof per the lease-gate lesson).

Old f29cab5f3 evidence bundle rescued from /tmp before reboot loss:
sha256 d0d0fa28de2ea0196237e57f7b8bcab42f07d697da8b6c4aadaa7f66070f0b09 (matches
the recorded hash from llama.cpp-ytqx c-px0r).

Shmem flat (14.68 GB band), kernel log clean.
