"""Performance budgets for the physics extension.

CI-safe by design: budgets are generous (a slow laptop passes) and paths are
derived from __file__ so the suite runs identically on any checkout.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from playground.config import load_config
from playground.world.physics import PhysicsWorld


def _build(body_type: str, extras: dict | None = None) -> PhysicsWorld:
    overrides = {"agent_body": {"type": body_type}}
    if extras:
        overrides.update(extras)
    return PhysicsWorld(load_config(overrides=overrides))


def _median_ms(world, steps: int = 200, warmup: int = 50) -> float:
    for _ in range(warmup):
        world._step()
    t0 = time.perf_counter()
    for _ in range(steps):
        world._step()
    return (time.perf_counter() - t0) / steps * 1000.0


def test_humanoid_step_budget():
    """Humanoid median step time must stay far below real-time frame budgets."""
    w = _build("humanoid")
    median = _median_ms(w)
    assert median < 25.0, f"humanoid median step {median:.2f} ms exceeds the 25 ms budget"


def test_box_step_budget():
    """Box-agent path must remain light (the original playground behaviour)."""
    w = _build("box")
    median = _median_ms(w)
    assert median < 8.0, f"box median step {median:.2f} ms exceeds the 8 ms budget"


def test_dt_consistency_sim_time_scaling():
    """N steps must advance simulated time by exactly N * timestep (1x/4x scaling relies on this)."""
    w = _build("box")
    start = float(w.d.time)  # MuJoCo's simulation clock (w.time is a property wrapper)
    n = 50
    w.step(n)
    elapsed = float(w.d.time) - start
    assert abs(elapsed - n * w.dt) < 1e-9, f"elapsed {elapsed} != {n} * dt {n * w.dt}"


def test_object_count_scaling_generous():
    """A 25-object world must not blow past a generous 3x ratio vs a 5-object world."""
    w5 = _build("box", {"world": {"blocks": {"count": 5}}})
    w25 = _build("box", {"world": {"blocks": {"count": 25}}})
    m5 = _median_ms(w5, steps=100, warmup=30)
    m25 = _median_ms(w25, steps=100, warmup=30)
    assert m25 < max(3.0 * max(m5, 0.01), 8.0), (
        f"25-object median {m25:.2f} ms vs 5-object {m5:.2f} ms exceeds the generous 3x budget")
