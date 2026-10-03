#!/usr/bin/env python3
"""RED stub: the API tests/test-moe-cache-sim.py pins, with no behaviour."""
from dataclasses import dataclass, field


@dataclass
class Step:
    index: int
    phase: int
    n_tokens: int
    layers: dict = field(default_factory=dict)


@dataclass
class Trace:
    header: dict
    steps: list


class Result:
    pass


def _todo(*a, **k):
    raise NotImplementedError


build_profile = resident_from_profile = simulate_static = _todo
simulate_first_touch = simulate_adaptive = slots_for_budget = _todo
sweep = sweep_loo = rows_to_csv = write_trace = read_trace = _todo
load_expert_sizes_gguf = _todo
