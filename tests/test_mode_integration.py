"""Integration-point tests for the physics extension.

Validates the mode selector default and that every new subsystem exposes its
contract integration point. Cross-package imports use pytest.importorskip so a
parallel package under construction can never fail this suite.
"""
from __future__ import annotations

import pytest

from playground.config import load_config


def test_simulation_mode_default_is_playground():
    """config/default.yaml must ship a simulation.mode selector defaulting to 'playground'
    (Free Playground preserved per the spec's mode section)."""
    cfg = load_config()
    mode = cfg.get("simulation", {}).get("mode")
    assert mode == "playground", f"simulation.mode = {mode!r}, expected 'playground'"


MODES_DOC = {"playground", "training", "arena", "interactive"}


def test_simulation_mode_value_documented():
    """The four spec modes must be listed in the config comment block."""
    from pathlib import Path
    default_yaml = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    text = default_yaml.read_text(encoding="utf-8")
    assert "simulation" in text and "mode: playground" in text


def test_arena_integration_point():
    match = pytest.importorskip("playground.arena.match")
    assert hasattr(match, "ArenaMatch"), "playground.arena.match must expose ArenaMatch"


def test_rl_integration_point():
    # B2 migration: the broken pseudo-trainer (playground.rl.train) is removed;
    # the RL integration seam is the authoritative env + curriculum machinery.
    env_mod = pytest.importorskip("playground.rl.env")
    cur_mod = pytest.importorskip("playground.rl.curriculum")
    assert hasattr(env_mod, "LocoEnv"), "playground.rl.env must expose LocoEnv"
    assert callable(getattr(cur_mod, "Curriculum", None)), \
        "playground.rl.curriculum must expose Curriculum"


def test_player_integration_point():
    player = pytest.importorskip("playground.player")
    assert hasattr(player, "PlayerController"), "playground.player must expose PlayerController"


def test_locomotion_integration_point():
    loco = pytest.importorskip("playground.locomotion")
    assert hasattr(loco, "GaitController"), "playground.locomotion must expose GaitController"
