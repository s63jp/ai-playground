"""Arena prompt wiring: separate solo/coop texts dispatched by session mode (M5).

Steered design 2026-11-17: arena sessions get their own prompt variants, split solo
vs coop - never merged with each other and never reusing exploration or benchmark
texts. Every arena session records the prompt id and the system prompt hash it ran
with, in the session_started note.
"""
from __future__ import annotations

import pytest

from playground.agents.adapters import PolicyAdapter
from playground.agents.prompts import (ARENA_COOP_SYSTEM, ARENA_SOLO_SYSTEM, ARENA_PROMPT_IDS,
                                       BENCHMARK_SYSTEM, arena_system, prompt_hash)
from playground.session.runner import memory_root, run_session


def _end_policy(system, opening, tools):
    yield ("call", "end_session", {})


def _run_mode(cfg, db, mode, agent_id="probe_agent"):
    return run_session(cfg, db, agent_id, PolicyAdapter(_end_policy), mode=mode)


def _system_message_text(db, sid) -> str:
    sys_msgs = [m for m in db.messages(sid) if m["role"] == "system"]
    assert sys_msgs, "the session must record its system prompt as a message"
    texts = {m["content"] for m in sys_msgs}
    assert len(texts) == 1, f"expected exactly one distinct system text, got {len(texts)}"
    return next(iter(texts))


def _started_note(db, sid) -> dict:
    evs = db.events(sid, types=["session_started"])
    assert len(evs) == 1, f"expected exactly one session_started event, got {len(evs)}"
    return evs[0]["data"]


class TestArenaDispatch:
    def test_solo_mode_uses_solo_text_with_matching_record(self, cfg, db):
        summary = _run_mode(cfg, db, "arena_solo")
        sid = summary["session_id"]
        assert _system_message_text(db, sid) == ARENA_SOLO_SYSTEM
        note = _started_note(db, sid)
        assert note["prompt_id"] == "arena_solo"
        assert note["system_prompt_hash"] == prompt_hash(ARENA_SOLO_SYSTEM)

    def test_coop_mode_uses_coop_text_with_matching_record(self, cfg, db):
        summary = _run_mode(cfg, db, "arena_coop")
        sid = summary["session_id"]
        assert _system_message_text(db, sid) == ARENA_COOP_SYSTEM
        note = _started_note(db, sid)
        assert note["prompt_id"] == "arena_coop"
        assert note["system_prompt_hash"] == prompt_hash(ARENA_COOP_SYSTEM)

    def test_solo_and_coop_texts_differ_for_solo_and_coop_sessions(self, cfg, db):
        solo = _run_mode(cfg, db, "arena_solo")
        coop = _run_mode(cfg, db, "arena_coop")
        solo_text = _system_message_text(db, solo["session_id"])
        coop_text = _system_message_text(db, coop["session_id"])
        assert solo_text != coop_text

    def test_benchmark_fallback_still_serves_benchmark_system(self, cfg, db):
        summary = _run_mode(cfg, db, "benchmark")
        assert _system_message_text(db, summary["session_id"]) == BENCHMARK_SYSTEM


class TestArenaPromptRegistry:
    def test_arena_ids_and_texts_are_separate(self):
        assert ARENA_SOLO_SYSTEM != ARENA_COOP_SYSTEM
        assert ARENA_SOLO_SYSTEM != BENCHMARK_SYSTEM
        assert ARENA_COOP_SYSTEM != BENCHMARK_SYSTEM
        assert len(ARENA_PROMPT_IDS) == 2 and set(ARENA_PROMPT_IDS) == {"arena_solo", "arena_coop"}

    def test_arena_system_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            arena_system("arena_freefight")
        # The known ids both select their own text.
        assert arena_system("arena_solo") == ARENA_SOLO_SYSTEM
        assert arena_system("arena_coop") == ARENA_COOP_SYSTEM
