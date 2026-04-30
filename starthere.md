Kick off PLACE epic via team-driven development
  Use the team-toolkit:team-driven-development skill to execute epic llama.cpp-p8ic3 (priority-driven weight placement) on
  branch feature/sycl-coalescing. Read CLAUDE.md first for build/test commands.
  Mission
  Make GPT-OSS 20B usable at full 131K context on Arc B580, and make every model split
intelligently when it doesn't fit.
  Today the placement planner makes one monolithic "all or none" buffer-type vote at load time — at 131K, 0 of 459 weight
  tensors land on the GPU, all 11.5 GB sit in pinned host memory while ~6 GB of VRAM weight zone goes unused next to KV. TG =
  4.83 tok/s. We're going to fix this with a research-backed priority policy.
  Design intent (the 'why' behind every bead in this epic)
  - The unified cache owns every byte — VRAM, pinned host, mmap. One allocator, one budget, one source of truth. No
● Here's a self-contained kickoff prompt — paste this into the new session.
  ---
  Kick off PLACE epic via team-driven development    m    b   d            placement. Raw tensor->data access in dis
● Here's a self-contained kickoff prompt — paste this into the new session.
  ---
  Kick off PLACE epic via team-driven development
  Use the team-toolkit:team-driven-development skill to execute epic llama.cpp-p8ic3 (priority-driven weight placeme
  branch feature/sycl-coalescing. Read CLAUDE.md first for build/test commands.
  Mission
  Make GPT-OSS 20B usable at full 131K context on Arc B580, and make every model split
intelligently when it doesn't
  Today the placement planner makes one monolithic "all or none" buffer-type vote at load time — at 131K, 0 of 459 w
  tensors land on the GPU, all 11.5 GB sit in pinned host memory while ~6 GB of VRAM weight zone goes unused next to
  4.83 tok/s. We're going to fix this with a research-backed priority policy.
  Design intent (the 'why' behind every bead in this epic)
  - The unified cache owns every byte — VRAM, pinned host, mmap. One allocator, one budget, one source of truth. No
  split-brain between weight planner, KV allocator, and compute-buffer scheduler.
  - Smart pointers (mem_handle) auto-resolve to the right memory — GPU dispatch code holds handles; the res
● Here's a self-contained kickoff prompt — paste this into the new session.
  ---
  Kick off PLACE epic via team-driven development
  Use the team-toolkit:team-driven-development skill to execute epic llama.cpp-p8ic3 (priority-driven weigh
  branch feature/sycl-coalescing. Read CLAUDE.md first for build/test commands.
  Mission
  Make GPT-OSS 20B usable at full 131K context on Arc B580, and make every model split
intelligently when i
  Today the placement planner makes one monolithic "all or none" buffer-type vote at load time — at 131K, 0
  tensors land on the GPU, all 11.5 GB sit in pinned host memory while ~6 GB of VRAM weight zone goes unuse
  4.83 tok/s. We're going to fix this with a research-backed priority policy.
  Design intent (the 'why' behind every bead in this epic)
  - The unified cache owns every byte — VRAM, pinned host, mmap. One allocator, one budget, one source of t
  split-brain between weight planner, KV allocator, and compute-buffer scheduler.
  - Smart pointers (mem_handle) auto-resolve to the right memory — GPU dispatch code holds handles; the res
   correct physical pointer (device / pinned host / mmap) based on current placement. Raw tensor->data acce
  the bug class we are eliminating (sibling epic llama.cpp-wjvse).
  - The planner is authoritative — at model load it decides where every weight goes; at context cre
●tHere's a self-contained kickoff prompt — paste this into the new session.
                                                 s a plan.ops[op_id] execution decision. If the plan ---s i
  Kick off PLACE epic via team-driven development
        r                             h         r     s   c          i         lways-hot path pinned,Use/the team-toolkit:team-driven-development skill to execute epic llama.cpp-p8ic3 (priority-driw
● Here's a self-contained kickoff prompt — paste this into the new session.
  Where this fits in the existing graph
  - Parent infrastructure: llama.cpp-3h5gm — Unified Memory Placement plan (adds n_ctx/n_ubatch/n_s
eq_max to
  llama_model_params, two-stage plan, real graph_reserve(no_alloc=true) in planner). Track A is in
flight;PLACE-3
  depends on llama.cpp-6l35a (A2: envelope consumed by planner).
  - Sibling: llama.cpp-wjvse — mem_handle enforcement, eliminating raw ->data from dispatch.
  - Sibling: llama.cpp-792vn.14 — heterogeneous KV mechanism (closed). PLACE-11 layers
priority pol
icy on top.
  The work
  Epic: llama.cpp-p8ic3 (12 children, descriptions complete with acceptance gates)
  Start here: bd ready — PLACE-1 (i7hhs, tensor classifier) is the only ready foundation task.experts ordered by layer index, P3 spills to host), multi-tier hardware (B580 →
B50 → hosp
,nand KV         3               a     e, 6 dense-validate, 7 env-overrides, 11 KV-tiering,
  experts ordered by layer index, P3 spills to host), multi-tier hardware (B580 → B50 → host
, and KV
  residency follows weight placement per-layer (the sk1xz colocation invariant: a layer's at
nd its
  KV stay on the same tier).
  Where this fits in the existing graph
  - Parent infrastructure: llama.cpp-3h5gm — Unified Memory Placement plan (adds n_ctx/n_uba
o
  llama_model_params, two-stage plan, real graph_reserve(no_alloc=true) in planner). Track A
LACE-3
  depends on llama.cpp-6l35a (A2: envelope consumed by planner).
  - Sibling: llama.cpp-wjvse — mem_handle enforcement, eliminating raw ->data from dispatch.
  - Sibling: llama.cpp-792vn.14 — heterogeneous KV mechanism (closed). PLACE-11 layers
prior
op.
eq_max
  - Sibling: llama.cpp-wjvse — mem_handle enforcement, eliminating raw ->data from
  dispatch.
  - Sibling: llama.cpp-792vn.14 — heterogeneous KV mechanism (closed). PLACE-11 layers
  priority policy on top.
  The work
  Epic: llama.cpp-p8ic3 (12 children, descriptions complete with acceptance gates)
  Start here: bd ready — PLACE-1 (i7hhs, tensor classifier) is the only ready
  foundation task.
  Chain: 1 → 2 → 3 → 4 → {5 MoE-validate, 6 dense-validate, 7 env-overrides, 11
  KV-tiering, 12 multi-GPU} → 8 docs. Stretch: 9 (profile-guided expert ordering), 10
  (token_embd tier A/B).
  PLACE-3 (the budget fitter) must be designed multi-tier-aware on day one — input is
  an ordered [(tier_name, budget_bytes), ...] list from fastest to slowest,
  single-device is the degenerate 1-tier case. See the bead's notes. Designing for
  "device or host" only would force a re-architecture at PLACE-12.
  Pre-reading (in this order)
  1. CLAUDE.md — build commands, env vars, conventions, machine-specific notes (patched   compute-runtime, device selection, perf targets)
  2. bd show llama.cpp-p8ic3 — epic + per-task acceptance gates
  3. docs/plans/2026-04-22-unified-memory-placement-plan.md — the parent infrastructure   design (~560 lines)
  4. docs/plans/2026-04-25-unified-memory-placement-readiness-review.md — current state   of Track A
  Success criteria (epic-level gate)
  - Mistral 7B at 100% budget: PP512 ≥ 1700, TG128 ≥ 80 (no regression vs current
  baseline)
  - Mistral 7B at 50% budget: planner auto-splits with documented PP/TG curve
  - GPT-OSS 20B at default ctx: no regression vs TG=16.47
  - GPT-OSS 20B at 131K + FA: TG > 6.0 tok/s (vs 4.83 baseline — the primary
  user-facing win)
  - Multi-GPU validation on B580+B50 with priority-driven distribution
  - Per-tensor priority dump on Mistral 7B + GPT-OSS 20B matches hand-curated expected
  hierarchy
  Operating rules
  - Activate Serena MCP for /Apps/llama.cpp before code work; use symbolic editing
  tools.
  - Beads for ALL task tracking — never TodoWrite. bd update <id> --status=in_progress
  to claim, bd close <id> --reason="..." when done. Run bd sync at session end.
  - Never --no-verify on commits, never push without explicit user OK.
  - Multi-GPU runs require ONEAPI_DEVICE_SELECTOR (single-GPU default is level_zero:0).  - Test gates per task are in each bead's description — implementers must hit them
  before reporting completion; spec-reviewer verifies against the bead text.