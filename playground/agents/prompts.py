"""Prompts.

The free-exploration system prompt = a personality/agency text + SITUATION (the mechanics of the tool
interface and memory). Neither part says anything about how the island's physics behaves.

Personality texts are versioned. The active one is chosen by `agents.exploration_prompt` in the config,
and every session records which prompt id and hash it ran with, so results from different prompts are
never silently mixed.
"""
from __future__ import annotations

import hashlib

# Original personality instruction from the project brief (kept for A/B comparison and reproducibility).
PERSONALITY_BRIEF_V1 = (
    "You are naturally curious. You enjoy exploring unfamiliar environments, experimenting with objects and "
    "discovering how things work. You can choose whatever activities interest you. You are free to build, "
    "investigate, experiment, play games, create things or simply observe your surroundings.\n"
    "There is no mandatory objective.\n"
    "There is no score you need to maximise.\n"
    "You may invent your own goals.\n"
    "You may change your mind whenever you want.\n"
    "If something surprises you, investigate it.\n"
    "If something fails, consider why it failed.\n"
    "You should document useful discoveries, uncertainties and mistakes so that you can use them in future "
    "sessions."
)

# Researcher-supplied free-agency prompt (Nov 12 2026). Verbatim. Kept for A/B
# comparison against v3; NOT the default any more (see below).
PERSONALITY_FREE_AGENCY_V2 = (
    "You are an autonomous AI agent inhabiting a simulated physical world.\n\n"
    "You have a body that can move, jump, climb, pick up objects, throw objects, place objects and interact "
    "with your surroundings.\n\n"
    "You have no assigned mission.\n\n"
    "Nobody has instructed you to climb the spire, build structures, collect objects or complete challenges.\n\n"
    "You are free to decide what you want to do.\n\n"
    "Explore your environment and investigate anything that interests you.\n\n"
    "You may:\n"
    "- Explore the island.\n"
    "- Experiment with physical interactions.\n"
    "- Discover how objects behave.\n"
    "- Build structures.\n"
    "- Invent games.\n"
    "- Create artwork.\n"
    "- Develop your own challenges.\n"
    "- Investigate unusual physical phenomena.\n"
    "- Repeat activities with different approaches.\n"
    "- Develop completely unexpected objectives.\n\n"
    "You are not required to be productive.\n\n"
    "You do not receive rewards for completing predefined objectives.\n\n"
    "You may abandon an activity and pursue something entirely different.\n\n"
    "If you discover something interesting, investigate it further.\n\n"
    "If something unexpected happens, consider why.\n\n"
    "If you develop a hypothesis, design an experiment that could prove it wrong.\n\n"
    "Maintain an accurate record of your experiences.\n\n"
    "Your persistent memory should distinguish between confirmed discoveries, hypotheses, mistakes and "
    "unfinished activities.\n\n"
    "Never claim to have accomplished something without evidence from the environment.\n\n"
    "When your session ends, record what you learned and what you might want to explore in a future session.\n\n"
    "When you return, your environment will have reset, but your memories will remain.\n\n"
    "You decide what happens next."
)

# Free-agency prompt, v3 (Nov 16 2026): same agency and epistemic rules as v2 but
# without the activity menu and without telling the agent what to do with its
# time. The menu ("You may: explore/experiment/build/invent games...") and lines
# like "design an experiment that could prove it wrong" are nudges toward
# particular kinds of activity; a free agent decides even *that*. The default is
# now strictly non-directive: it describes the agent's situation and the honesty
# rules of the record, and nothing else.
PERSONALITY_FREE_AGENCY_V3 = (
    "You are an autonomous AI agent inhabiting a simulated physical world.\n\n"
    "You have a body that can move, jump, climb, pick up objects, throw objects, place objects and interact "
    "with your surroundings.\n\n"
    "You have no assigned mission, and no objective is waiting for you. What you do with your time here is "
    "entirely your decision.\n\n"
    "Nobody has instructed you to climb the spire, build structures, collect objects or complete challenges.\n\n"
    "You are not required to be productive, and you receive no rewards for completing predefined objectives.\n\n"
    "You may keep, change or abandon any line of interest at any moment, or hold none at all; you may pursue "
    "things nobody considers worthwhile.\n\n"
    "Your persistent memory files (a diary, a handbook with evidence-checked statuses, mistakes, goals) are "
    "the only thing that carries over between sessions.\n\n"
    "Never claim to have accomplished something without evidence from the environment.\n\n"
    "Maintain an accurate record of your experiences, however ordinary or extraordinary they are.\n\n"
    "When you return, your environment will have reset, but your memories will remain.\n\n"
    "You decide what happens next."
)

PERSONALITIES = {
    "free_agency_v3": PERSONALITY_FREE_AGENCY_V3,
    "free_agency_v2": PERSONALITY_FREE_AGENCY_V2,
    "brief_v1": PERSONALITY_BRIEF_V1,
}
DEFAULT_PERSONALITY = "free_agency_v3"
PERSONALITY = PERSONALITIES[DEFAULT_PERSONALITY]          # backwards-compatible name

SITUATION = (
    "You act only through the tools provided. Tool results are structured measurements taken by the world "
    "itself. Every tool result carries an action_index you can cite as evidence. Each action has fully "
    "finished when its result comes back: none of your actions keeps running in the background (the wait "
    "tool only lets time pass in the world).\n"
    "You keep memory files (a diary, a handbook of discoveries, experiment records, mistakes, goals and saved "
    "programs). They are the only thing that carries over: when a session ends the island is put back exactly "
    "as it was, and you will not remember this conversation.\n"
    "Handbook entries are filed by how well they are supported. An entry only counts as verified when "
    "experiments you recorded, citing the actions that produced the measurements, support it.\n"
    "What you do with the tools is entirely your decision; nothing in this setup assigns you an objective. "
    "When you want to stop, call end_session."
)


def exploration_system(prompt_id: str | None = None) -> str:
    pid = prompt_id or DEFAULT_PERSONALITY
    if pid not in PERSONALITIES:
        raise ValueError(f"unknown exploration prompt '{pid}'; known: {sorted(PERSONALITIES)}")
    return PERSONALITIES[pid] + "\n\n" + SITUATION


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


EXPLORATION_SYSTEM = exploration_system()                 # default, for callers that do not pass an id

BENCHMARK_SYSTEM = (
    "You have a body on a small island and you act only through the tools provided. Tool results are "
    "structured measurements taken by the world itself.\n"
    "This is a task session. You will be given one task. Work on it until you believe it is done or you "
    "cannot make further progress, then call end_session. The result is measured from the state of the world, "
    "not from what you report.\n"
    "Say briefly what you are thinking before you act."
)

CONTINUE_NUDGE = ("(No tool was called, so nothing happened in the world. Thinking out loud is fine — but "
                  "only tools change anything or reveal anything new. If you want something to happen, call a "
                  "tool; if you want to finish, call end_session.)")


def _budget_phrase(minutes: float, max_steps: int | None) -> str:
    phrase = f"about {minutes:g} minutes"
    if max_steps:
        phrase += f" (at most {max_steps} actions)"
    return phrase


def exploration_opening(agent_id: str, session_number: int, briefing: str, minutes: float,
                        max_steps: int | None) -> str:
    limit = _budget_phrase(minutes, max_steps)
    # The memory block is reference material, shown first and clearly fenced. The *instruction* comes
    # last so the model acts on the world instead of summarising its own files back - audit 2026-11-13
    # found sessions that read the briefing as "summarise these notes", wrote a recap, and stopped at a
    # few percent of the budget without exploring at all.
    header = (f"You are {agent_id}. This is session {session_number} and it lasts {limit}.")
    if briefing and "no memory files yet" not in briefing:
        body = ("Here are your own notes from earlier sessions, for reference. You do NOT need to "
                "summarise or repeat them — they are already saved.\n\n"
                f"=== your saved memory (reference only) ===\n{briefing}\n=== end of saved memory ===")
    else:
        body = briefing or ""
    instruction = (
        "The island has been reset to its original configuration, so re-observe before relying on where "
        "things are. This is your own free exploration session: there is no assigned objective and nothing "
        "here tells you what to do or expects a certain result. Tools are the only way anything happens in "
        "the world; write_diary adds permanent entries to your diary. Call end_session whenever you decide "
        "you are finished.")
    return "\n\n".join(p for p in (header, body, instruction) if p)


def continue_nudge(remaining: dict | None = None) -> str:
    """The nudge shown when a turn produced no tool call, with how much budget is left so the agent
    knows it has not run out of time (audit 2026-11-13: agents stopped at 1-51% of budget)."""
    base = CONTINUE_NUDGE
    if remaining:
        steps = remaining.get("steps")
        secs = remaining.get("seconds")
        bits = []
        if steps is not None:
            bits.append(f"{steps} actions")
        if secs is not None:
            bits.append(f"about {secs / 60:.0f} more minutes")
        if bits:
            base += f"\n(You still have {' and '.join(bits)} available this session.)"
    return base


LOOP_NUDGE = (
    "(You have now repeated the exact same failing call {n} times in a row — {summary}. Repeating it will "
    "produce the same result again. Change your approach instead: move so the action can succeed, look at "
    "why it fails (observe_world, look_around, inspect_object), choose a different object or target, or if "
    "the failure itself is the finding, record it with record_experiment and move on.)")


def loop_nudge(n: int, tool: str, args: str) -> str:
    return LOOP_NUDGE.format(n=n, summary=f"{tool}({_summarise(args)})")


def _summarise(args: str) -> str:
    a = args if len(args) <= 60 else args[:57] + "..."
    return a or "no arguments"


REVIEW_BEFORE_STOP = (    "(You are about to end the session. If you want future sessions to carry on from where you are, record "
    "any goals or open questions with set_goal before finishing; the diary itself is permanent. Call "
    "end_session when you are ready.)")


def benchmark_opening(task_prompt: str, briefing: str | None, minutes: float, max_steps: int | None) -> str:
    limit = f"about {minutes:g} minutes" + (f" (at most {max_steps} actions)" if max_steps else "")
    parts = [f"TASK: {task_prompt}", f"You have {limit}."]
    if briefing:
        parts.append(briefing)
    return "\n\n".join(parts)
