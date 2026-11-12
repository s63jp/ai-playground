"""Shared fixtures. Every test works in its own temporary workspace: nothing touches experiments/."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from playground.config import load_config  # noqa: E402
from playground.interface import Environment  # noqa: E402
from playground.memory import MemoryStore  # noqa: E402
from playground.storage import Database  # noqa: E402
from playground.world.physics import PhysicsWorld  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={"_root": str(tmp_path), "session": {"max_steps": 400}})


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    yield d
    d.close()


@pytest.fixture
def world(cfg):
    return PhysicsWorld(cfg)


@pytest.fixture
def make_env(cfg, db, tmp_path):
    counter = {"n": 0}

    def _make(agent_id: str = "agent_t", **kw) -> Environment:
        counter["n"] += 1
        mem = MemoryStore(tmp_path / "agents", agent_id, cfg, db=db)
        sid = db.create_session(agent_id=agent_id, mode=kw.pop("mode", "exploration"),
                                session_number=counter["n"], adapter="test")
        return Environment(cfg, db, mem, agent_id, sid, counter["n"], **kw)
    return _make


def goto(w: PhysicsWorld, x: float, y: float):
    p = w.agent_pos()
    d = np.array([x - p[0], y - p[1], 0.0])
    n = float(np.linalg.norm(d))
    if n > 0.01:
        return w.move(d / n, n)
    return None


@pytest.fixture
def walk():
    return goto
