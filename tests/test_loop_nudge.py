"""The loop-breaker (2026-11-16).

A live 4B benchmark session cycled one out_of_reach pick_up for minutes; unbounded
repetition of an identical failing call wastes wall-clock while remaining legitimate
behaviour data. The runner now nudges the agent after N identical failing calls,
naming the exact repeat. Pins: streak detection (trailing run, success resets,
argument-sensitive), the nudge message, and the config kill-switch.
"""
import json

from playground.agents.prompts import loop_nudge
from playground.agents.adapters import ToolCall
from playground.session.runner import _failed_streak


def _res(ok=True, error=None):
    r = {"success": ok, "action": "x"}
    if error:
        r["error"] = {"code": error}
    return r


class _C(ToolCall):
    def __init__(self, name, args):
        super().__init__(id="t", name=name, args=args)


# --------------------------------------------------------------------------- happy
def test_detects_trailing_streak_of_identical_failures():
    results = [(_C("move", {"d": "N"}), _res(True)),
               (_C("pick_up", {"o": "B"}), _res(False, "out_of_reach")),
               (_C("pick_up", {"o": "B"}), _res(False, "out_of_reach")),
               (_C("pick_up", {"o": "B"}), _res(False, "out_of_reach")),
               (_C("pick_up", {"o": "B"}), _res(False, "out_of_reach")),
               (_C("pick_up", {"o": "B"}), _res(False, "out_of_reach"))]
    assert _failed_streak(results) == (5, "pick_up", json.dumps({"o": "B"}, sort_keys=True))


def test_success_resets_the_cycle():
    results = [(_C("pick_up", {"o": "B"}), _res(False, "out_of_reach")),
               (_C("pick_up", {"o": "B"}), _res(True)),
               (_C("pick_up", {"o": "B"}), _res(False, "out_of_reach"))]
    assert _failed_streak(results) == (1, "pick_up", json.dumps({"o": "B"}, sort_keys=True))


def test_different_arguments_do_not_extend_the_streak():
    results = [(_C("pick_up", {"o": "B"}), _res(False, "out_of_reach")),
               (_C("pick_up", {"o": "C"}), _res(False, "out_of_reach"))]
    assert _failed_streak(results) == (1, "pick_up", json.dumps({"o": "C"}, sort_keys=True))


def test_success_at_the_end_means_no_streak():
    results = [(_C("pick_up", {"o": "B"}), _res(False, "out_of_reach")) for _ in range(4)]
    results.append((_C("look_around", {}), _res(True)))
    assert _failed_streak(results) is None


# --------------------------------------------------------------------------- validate
def test_nudge_names_tool_and_args():
    msg = loop_nudge(4, "pick_up", '{"object_id": "block_09"}')
    assert "4 times" in msg and "pick_up" in msg and "block_09" in msg


def test_error_long_arguments_are_truncated_in_nudge():
    args = json.dumps({"position": [0.1] * 40})
    msg = loop_nudge(4, "place", args)
    assert len(msg) < 520 and "place(" in msg and args[:20] in msg
