#!/usr/bin/env python3
# Host-fit calculator behind docs/design/sycl-storage-tier-lazy-tables.md §4.5
# (llama.cpp-th32, rev 2b). Hand arithmetic only: llama.cpp-7v5c (T27's
# host-only plan dry-run) replaces every number here once it exists.
#
#   python3 docs/design/sycl_storage_tier_host_fit.py
#
# No dependencies. GiB = 2^30. Sizing code cited at fork c69d5774d.

GIB = float(1 << 30)
MIB = 1.0 / 1024.0          # in GiB
CHUNK = 2.0                 # pinned-pool commit chunk, GiB (pinned-pool.hpp:115)
READINGS = [171.4, 181.8, 183.7, 190.4]   # MemAvailable at load, GiB, 2026-09-26
OS_RESERVE = 8.0            # max(8 GiB, MemTotal/32), unified-cache.cpp:4216
P3_MARGIN = 29.0            # 9g22 §4.5.2: MemAvailable_pre - expected_host - 5 >= 24
ENVELOPE = 2.5              # host_reserve 2 GiB + context envelope <= 0.5 (§4.3 step 3)
GATE_CACHES = 4.0           # 9g22 P1: 2 GiB per table, index included, two tables
DRAFT = 7.9                 # DSpark draft on the host (9g22 §10.3)
KV_MIN = 64 * MIB           # min_kv_zone, ggml-sycl.cpp:5626
N_CTX = 4096                # every gate pins -c


def align256(x):
    return (x + 255) // 256 * 256


def ring_gib(n_expert, n_embd, n_ff, n_ubatch=512):
    # src/llama-model.cpp:430-507 (c69d5774d). The loop has no type gate: every
    # MoE expert weight is sized as an MXFP4 WOQ slot. Expert tensors are
    # [n_embd, n_ff] (gate, up) and [n_ff, n_embd] (down), so max_k == max_n == n_embd.
    k, n = n_embd, n_ff
    slot = align256(n * (k // 2)) + align256((k // 32) * n)
    weight = slot * n_expert
    rows = n_ubatch * n_expert
    act = align256(rows * max(n_embd, n_ff) * 2)
    out = align256(rows * max(n_embd, n_ff) * 4)
    return weight / GIB, act / GIB, out / GIB, (weight + act + out) / GIB


def runtime_commit(x):
    n = int(x // CHUNK) + (1 if x - CHUNK * int(x // CHUNK) > 1e-9 else 0)
    return n * CHUNK


def zones(max_tensor, n_expert, ring, today):
    """Per-zone-set non-weight host terms, GiB.

    today=True: the sizing code as it is (uc:27685-27742, gs:5615-5653).
    today=False: after lever (s): ring type-gated to MXFP4 (0 here) and the
    VRAM-resident dma_staging_pool dropped from host SCRATCH and the RUNTIME pool.
    """
    mt = max_tensor
    routing_ids = n_expert * 4 * N_CTX / GIB
    # STAGING: s1_in_flight(1) x (max(2 x max_tensor, max_staging_pair) + 2 MiB)
    # + 16 MiB headroom + expert bias + routing ids + tp staging + MMID pool.
    # max_staging_pair == 2 x max_tensor here (SOA dst == src for these types),
    # tp staging 0 on one card, MMID pool and expert bias unknown (small).
    staging = 2 * mt + 2 * MIB + 16 * MIB + routing_ids
    dma = 2 * mt                                   # uc:27522-27541, unconditional
    onednn_reorder, moe_q8 = 0.078, 0.031          # 9g22 estimates, reused for Qwen
    scratch = mt + 32 * MIB + onednn_reorder + moe_q8 + mt   # + moe_cpu_expert_staging
    runtime = 0.091                                # oneDNN scratchpad + pp_pipeline
    if today:
        scratch += dma + ring
        runtime += dma + ring
    return {
        "staging": staging, "scratch": scratch, "runtime": runtime,
        "runtime_commit": runtime_commit(runtime), "kv": KV_MIN,
        "nonweight": staging + scratch + runtime + KV_MIN,
        "nonweight_committed": staging + scratch + runtime_commit(runtime) + KV_MIN,
    }


def verdict(need, reading):
    lo, hi = need
    if reading >= hi:
        return "fits (%.1f-%.1f left)" % (reading - hi, reading - lo)
    if reading >= lo:
        return "low corner only (%.1f left)" % (reading - lo)
    return "refused"


def show(label, req):
    need = (req[0] + OS_RESERVE, req[1] + OS_RESERVE)
    cells = " | ".join(verdict(need, r) for r in READINGS)
    print("  %-44s req %5.1f-%5.1f  needs %5.1f-%5.1f | %s" % (label, req[0], req[1], need[0], need[1], cells))


def main():
    # ---- the ring
    ds_ring = ring_gib(384, 5120, 2304)
    q_ring = ring_gib(512, 2560, 640)
    print("PP MoE oneDNN ring (weight / act / out / total GiB)")
    print("  DeepSeek Q2_K  %.3f / %.3f / %.3f / %.3f" % ds_ring)
    print("  Qwen3.8 Q8_0   %.3f / %.3f / %.3f / %.3f" % q_ring)

    ds_mt = 1946419200 / GIB      # blk.0.ffn_down_exps, 1856.25 MiB
    q_mt = 850 * MIB              # blk.0.ffn_down_exps
    ds = {t: zones(ds_mt, 384, ds_ring[3], t) for t in (True, False)}
    qw = {t: zones(q_mt, 512, q_ring[3], t) for t in (True, False)}
    print("\nNon-weight host terms per zone set (GiB): today | after (s)")
    for name, z in (("DeepSeek", ds), ("Qwen", qw)):
        for key in ("staging", "scratch", "runtime", "runtime_commit", "kv", "nonweight", "nonweight_committed"):
            print("  %-8s %-20s %6.2f | %6.2f" % (name, key, z[True][key], z[False][key]))

    # ---- Qwen, B70 alone (one card, one zone set)
    q_table = 54400261120 / GIB
    q_ex_s = (95.0, 98.0)                       # host experts after (s), B70 holds 21.5-24.5
    q_other = 0.629                             # token_embd on the host
    q_rnh3 = (0.1, 0.5)
    print("\nQwen3.8, B70 alone. Needed MemAvailable = before-table + table + 8")
    for today in (True, False):
        disp = q_ring[3] if today else 0.0      # the ring sets the VRAM RUNTIME floor today
        z = qw[today]
        before = tuple(q_ex_s[i] + disp + q_other + q_rnh3[i] + z["nonweight_committed"] + ENVELOPE for i in (0, 1))
        need = (before[0] + q_table + OS_RESERVE, before[1] + q_table + OS_RESERVE)
        tag = "today (ring displaces %.2f GiB of B70 experts)" % disp if today else "after (s)"
        print("  %s: before table %.1f-%.1f, needs %.1f-%.1f" % (tag, before[0], before[1], need[0], need[1]))
        for r in READINGS:
            left = (r - OS_RESERVE - before[1], r - OS_RESERVE - before[0])
            res = "HOST" if left[0] >= q_table else ("HOST at the low corner only" if left[1] >= q_table else "STORAGE")
            print("    @%.1f leftover for the table %.1f-%.1f -> %s" % (r, left[0], left[1], res))
        if today:
            nd = (need[0] - disp, need[1] - disp)
            print("    without the device displacement: needs %.1f-%.1f" % nd)

    # ---- DeepSeek Q2_K
    ds_experts = 196765286400 / GIB             # 183.25 GiB routed experts
    p1_s = (157.3, 160.3)                       # phase 1 host experts after (s): B70 holds 23-26
    two_s = (145.3, 152.3)                      # two cards after sgc5 + T25
    tok, rnh3 = 0.202, (0.1, 0.5)

    def step2(ex, z, disp):
        return tuple(ex[i] + disp + tok + rnh3[i] + z["nonweight_committed"] for i in (0, 1))

    print("\nDeepSeek Q2_K. Planner fit: needed MemAvailable = requirement + 8 (main-pool rounding, 0-2 GiB, excluded)")
    for today in (True, False):
        z = ds[today]
        disp = ds_ring[3] if today else 0.0
        s = step2(p1_s, z, disp)
        print(" %s: step 2 (as committed) %.1f-%.1f" % ("today's code" if today else "after (s)", s[0], s[1]))
        A = (s[0] + ENVELOPE, s[1] + ENVELOPE)
        show("A phase 1, no draft, floor caches", A)
        show("B A + gate caches (2 x 2 GiB)", (A[0] + GATE_CACHES, A[1] + GATE_CACHES))
        show("C A + DSpark draft", (A[0] + DRAFT, A[1] + DRAFT))
        if not today:
            s2 = step2(two_s, z, 0.0)
            D = (s2[0] + ENVELOPE, s2[1] + ENVELOPE)
            show("D two cards after sgc5 + T25, one set", D)
            show("E D + non-weight zones a second time", (D[0] + z["nonweight_committed"], D[1] + z["nonweight_committed"]))
        else:
            # every participating cache commits a full zone set, WEIGHT included (sgc5)
            s2 = step2(two_s, z, 2 * disp)
            show("F two cards, today (a whole second set)", (2 * s2[0] + ENVELOPE, 2 * s2[1] + ENVELOPE))
        # P3 (9g22 §4.5.2): expected_host = step 2 + gate caches; needs MemAvailable_pre >= expected_host + 29
        eh = (s[0] + GATE_CACHES, s[1] + GATE_CACHES)
        p3 = (eh[0] + P3_MARGIN, eh[1] + P3_MARGIN)
        print("  P3, gate config: expected_host %.1f-%.1f, needs MemAvailable_pre %.1f-%.1f" % (eh + p3))
        for r in READINGS:
            post = (r - eh[1] - 5, r - eh[0] - 5)
            print("    @%.1f predicted post-load MemAvailable %.1f to %.1f -> %s" %
                  (r, post[0], post[1], "passes" if post[0] >= 24 else "fails"))
        if not today:
            # cross-check of 9g22 rev 5 §4.5.2's cold set (the doc links that figure, it does not restate it)
            per_el = (2 * 1486356480 + 1946419200) / 384 / GIB
            for r in (READINGS[0], 189.6):
                d = (p3[0] - r, p3[1] - r)
                c = (max(0.0, d[0]) / 0.95, max(0.0, d[1]) / 0.95)   # slab 2C/40 counted
                print("    cold set cross-check @%.1f: deficit %.1f-%.1f -> C %.1f-%.1f GiB (+2 main-pool rounding), %d-%d expert-layers of %.4f GiB, %.0f-%.0f%% of %.2f GiB"
                      % (r, d[0], d[1], c[0], c[1], c[0] / per_el, c[1] / per_el, per_el,
                         100 * c[0] / ds_experts, 100 * c[1] / ds_experts, ds_experts))


if __name__ == "__main__":
    main()
