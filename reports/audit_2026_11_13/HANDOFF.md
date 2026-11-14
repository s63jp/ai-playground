# AI Playground — Behaviour / Neural-Network / Autonomy Audit

**Date:** 2026-11-13
**Scope:** throwing fixation, neural-network architecture, agent autonomy, session termination,
observation loop, memory/learning, experimental integrity, model comparison.
**Evidence base:** experiments/playground.db (38 sessions, 1,633 agent actions), LM Studio server logs
2026-11-12..13 (43 files, 1,658 playground requests), agent memory dirs experiments/agents/*.

> **Do not read passing tests as proof of intelligence.** The tests below prove software correctness.
> Whether the agents now explore better must be judged from a fresh live run (see "How to verify live").

---

## Status at a glance

| # | Issue | Classification | Status |
|---|-------|---------------|--------|
| 1 | Throwing fixation | **Confirmed bug** (resume-adapter contamination) + confirmed architectural (throw-only NN) | Fixed + mitigated |
| 2 | NN is throw-centric | **Confirmed architectural limitation** | General transition architecture added |
| 3 | "Self-selected objectives: none set" | **Confirmed bug** (dashboard reads session-only) + design gap | Goals restructured; dashboard note below |
| 4 | Agents end sessions early | **Confirmed bug** (briefing read as summary task) + design gap | Fixed |
| 5 | 2B observation loop | **Confirmed bug** (context overflow) | Fixed |
| 6 | Memory between sessions | Partly working; **confirmed bug** (unbounded briefing) | Fixed |
| 7 | Dashboard vs reality | **Confirmed bug** (evidence coincidence) | Fixed |
| 8 | Model comparison | **Not yet established** (unequal conditions) | Harness built; live run pending |

**Tests:** full suite `240 passed, 2 deselected` (baseline before audit: 163). New suite
`tests/test_audit_2026_11_13.py` = **49 passed** across 12 classes. All work is in the working tree,
uncommitted (this is your repo; your own spire_climbing changes are interleaved and untouched).

---

## 1 & 2. Throwing fixation — ROOT CAUSE FOUND

**The agents were NOT choosing to throw.** Across 1,633 logged actions, `throw` was **23 (1.4%)**;
`move` was 32%, `look_around` 16%. The throwing dominance came from three places, none of them the
agents freely choosing to throw:

1. **Contaminated memory (confirmed bug).** agent_003's "confirmed" jump-throw discovery (D-002), the
   ring-generation rule (D-003), goal G-001, mistake M-001 and experiments E-002..E-006 were **all
   written by the scripted reference policy** (not an AI) during session 19. That session started on
   `lmstudio_unsloth4b`, hit an adapter error, and was **resumed with the CLI's default adapter
   `scripted`** — which is the built-in reference policy, a throwing demo. 70 of its 71 actions were
   scripted, yet the session row still said `lmstudio_unsloth4b`. Those throw "findings" then seeded
   every later briefing (agent_003's briefing grew to 22,169 chars, ~29 "throw" mentions), so real LLMs
   inherited a throwing agenda they never discovered.
   - **Fix:** `check_resume_adapter` refuses to resume a session with a different brain; `--resume`
     defaults to the session's own adapter (`resolve_resume_adapter`); every memory item is now stamped
     with its authoring `adapter`/`model` (`_prov`); `scripts/audit_provenance.py` finds and can
     quarantine the existing contaminated items (detects exactly D-002,D-003,E-002..E-006,G-001,M-001).

2. **The neural network is throw-only by construction (confirmed architectural limitation).** Every ML
   input and target is throw-specific (`features.py` INPUT_COLUMNS = force/elevation/release_height…,
   TARGET_COLUMNS = max_height/flight_time/range…); the dataset generator only reads `throw_release` +
   `flight_end` events; the controller is a throw policy; all 5 researcher models predict throw outcomes;
   the one benchmark task was `throw_height`. No LLM ever called the `nn_*` tools. The network literally
   could not represent movement, placement, or object relationships.
   - **Fix:** added `playground/ml/transition_features.py` — a **general (state, action) → next-state**
     representation covering all 9 physical actions (move, walk_to, jump, climb, pick_up, put_down,
     place, throw, catch, rotate). `validate_spec`/`train` now take a column vocabulary so the same
     trainer serves both. **Proven:** built 1,784 transition rows from the real DB spanning all action
     types and trained a general motion model end-to-end (test RMSE 2.95 m, r²=0.26). The throw pipeline
     is untouched and remains available as a focused special case.

**On recognising diminishing returns / habit:** the environment now tells an agent when an observation
is a verbatim repeat of an unchanged world (§5), which is the signal it previously lacked.

## 3. Self-selected objectives

- **Confirmed bug:** the dashboard's "Self-selected objectives" reads only *this session's* goal
  records, so a fresh session always shows "none set". (Dashboard JS fix still TODO — see Remaining.)
- **Design gap:** `set_goal` was used once in 1,633 actions (by the scripted policy). LLMs stated goals
  in prose but never recorded them.
  - **Fix:** `set_goal` now takes `kind` (question / hypothesis / build / explore / objective) and
    `next_step`, and these render in the briefing. The reframed opening and the end-of-session review
    (§4) explicitly prompt the agent to record goals and open questions.

## 4. Session termination & autonomous continuation

- **Confirmed bug:** no completed session used more than 51% of its budget (most 1–14%). The opening put
  the memory block last, so models read "here are your notes" as *summarise these* — 17 text-only turns
  asked a human what to do, 14 wrote a recap and stopped. Three text-only turns auto-end a session
  (`idle_turn_limit`), and there was **no continuation mechanism** at all.
  - **Fixes:**
    - Opening reframed: memory is fenced as *reference only ("you do NOT need to summarise")* and the
      **action directive is now the last thing the model reads** ("your job is to DO something, not
      write a report").
    - The no-tool nudge now reports remaining budget ("you still have 55 actions and about 40 more
      minutes").
    - One turn before the idle limit would end the session, exploration agents get a single
      `REVIEW_BEFORE_STOP` prompt: review goals, note unfinished experiments, decide if more is worth
      doing — a real chance to continue instead of drifting to a stop.
    - `end_session(continue_next_session=true)` lets an agent say it has unfinished work; `run_agents`
      honours it with a bounded `--max-continuations` cap (no uncontrolled loop), resuming the SAME
      agent identity and accumulated memory.

## 5. Observation loop (2B)

- **Confirmed bug, caused by context overflow.** The 2B model once made **126 identical observations in
  a row** (session 9). After its prompt overflowed the loaded context window, **78% of its calls were
  exact duplicates and it took zero physical actions**. Two compounding causes:
  1. **Context budget (confirmed bug).** The tool schema alone is ~5k tokens (37–49% of the 2B's
     12.5–14.6k window; 80–86% of a fresh agent's first request). When the request overflowed, LM Studio
     **silently dropped the middle of the conversation**. On top of that, the adapter's history-trim had
     a bug: it deleted and re-inserted its own marker each call and **never actually shrank the
     history** (simulated: 20k-char limit → 92k after 60 steps).
     - **Fix:** `_trim` rewritten to drop whole oldest exchanges until the request fits; the budget is
       now derived from the model's **real loaded context window** (auto-probed from LM Studio's
       `/api/v0/models`), accounting for system + tool schema + reply reserve — the whole request must
       fit, not just the history.
  2. **No "nothing changed" signal.** The agent had no way to know it was repeating itself.
     - **Fix:** a read-only observation that returns the same answer about an unchanged world is
       annotated `repeat_of_action` + a notice telling the agent to act; it never blocks (autonomy
       preserved) and resets after any physical action.

## 6. Memory & learning between sessions

- Memory persists and is evidence-gated; that part works.
- **Confirmed bug:** the briefing was **unbounded** (one reached 22,169 chars) — on a 12–16k-token
  model it crowds out the conversation.
  - **Fix:** `briefing_char_budget` (default 6,000). Durable parts (handbook, open goals, mistakes) are
    always kept; diary/experiment detail is trimmed to fit. agent_003's briefing: **22,169 → 6,911
    chars** with handbook/goals/mistakes intact.
- **Repeated experiments:** agent_003 re-ran the same platform survey across sessions 3,4,9,11,13,14,16.
  The repeat-observation signal (§5) plus structured goals/next_step (§3) address the proximate cause;
  deeper "don't re-test a confirmed fact" guidance is a prompt refinement for the next pass.

## 7. Experimental integrity & dashboard

- **Confirmed bug:** an experiment counted as "measured" evidence when a claimed number matched **any**
  numeric leaf of a cited result by value alone. `platform_count = 3` was accepted because it equalled
  the ring's x-coordinate `objects.ring[0] = 3.0` (E-009/E-010). This manufactured "measured" evidence
  from coincidences.
  - **Fix:** a value match into a **positional array element** (a coordinate) is demoted to a weak match
    that *grounds* the record but is **not** counted as a measurement; matches to a **named scalar
    field** (same name, or a descriptive name like spire_height↔top_height) still count. Re-verifying
    the live DB: exactly the 4 coincidental coordinate matches (E-007..E-010) drop from
    "consistent/measured" to "grounded/cited"; the genuine measurements (E-002..E-006, E-013, E-014) are
    preserved. Mis-named fabrications are still "contradicted".
- **Disproven = 0, mistakes = 1** everywhere; 20% of 4B actions failed and were never recorded. These
  are behavioural outcomes to watch in the next live run, not code bugs.

## 8. Model comparison — harness built, live run pending

The historical 2B/4B/9B runs are **not comparable**: different context windows (2B 12.5k, 4B 8k–119k,
9B unknown), different agents, different memory sizes; the 9B produced only 32 actions (every session
ended with "model unloaded"). Two things now fix this:

- **`playground/session/model_compare.py`** — `session_profile(db, sid)` computes a behavioural profile
  from the log alone (distinct tools = diversity, observation_share = looping, goals_set = objective
  generation, experiments/discoveries/confirmed = experimentation, memory_writes = memory use,
  budget_used, continue_requested, throw/physical/failed shares). `compare_models(...)` runs the SAME
  number of FRESH agents per model with the SAME pinned context window, action cap and empty starting
  memory, then aggregates — so any difference is attributable to the model, not the conditions.
- **`playground compare-models --models a,b,c --adapter lmstudio --agents 3 --max-steps 60
  --context-tokens 12000`** runs it and writes `experiments/model_comparison.json`.

Applying `session_profile` to the EXISTING (unequal) logs already shows the shape of the story — to be
confirmed under equal conditions with a live run:
- 2B `lowkeyqwen3.52b`: observation_share **0.76**, experiments **0**, goals **0** — the loop.
- 4B `unslothqen`: observation **0.22**, experiments **0.68**, budget_used **0.78** — balanced.
- 9B: only 2 usable sessions (kept unloading) — not enough to conclude.

A faster model is explicitly **not** assumed to be a better explorer: the profile measures exploration
quality directly, and the harness holds speed-irrelevant conditions equal.

---

## Steer items (asked for mid-audit)

- **"turn neural networks off / set each network to each agent"** — `ml.enabled` (global) and
  `ml.agents.<id>.enabled` (per agent) switch the `nn_*` tools off; `--adapter neural` with
  `owner=<agent_id>` already loads a specific agent's trained network as its brain.
- **"see available LM Studio models instead of only 2"** — `playground models` lists every model the
  server offers with context length and load state; any id works with `--model '<id>'`. Context window
  is auto-probed so the budget is correct per model.

---

## Files changed

Code: `playground/agents/adapters.py` (trim/budget, model discovery, probe), `agents/prompts.py`
(opening, nudge, review), `interface/environment.py` (repeat-obs, continuation, goal kind),
`interface/schema.py` (set_goal, end_session), `memory/store.py` (provenance, goal kind, briefing
budget), `session/runner.py` (resume guard, ml switch, lifecycle), `session/manager.py` (continuations),
`ml/nets.py` (vocabulary), `ml/transition_features.py` (NEW, general NN),
`session/model_compare.py` (NEW, equal-conditions comparison), `cli.py` (models, compare-models, resume,
--max-continuations), `config/default.yaml` (ml, context_tokens, briefing budget).
Scripts: `scripts/audit_lmstudio_logs.py`, `scripts/audit_provenance.py`.
Tests: `tests/test_audit_2026_11_13.py` (49 tests).

## Remaining work (not yet done)

- **Live model-comparison run:** the harness and metrics are built and tested (`compare-models`); it
  needs a live run with the models actually loaded in LM Studio to produce the equal-conditions numbers.
- **Dashboard:** show an agent's *open goals from memory* (not session-only) so "Self-selected
  objectives" reflects reality; add the LM Studio model dropdown; surface `continue_requested` and the
  repeat-observation streak.
- **Apply the provenance quarantine:** run `python scripts/audit_provenance.py --db experiments/playground.db --apply`
  to flag the session-19 scripted items in agent_003's live memory (left for you — it mutates memory).
- **Prompt refinement:** discourage re-testing already-confirmed facts; encourage citing specific
  open questions.
- **Live acceptance run:** the real proof (see below).

## How to verify live (the real acceptance gate)

1. Load a small model in LM Studio. `playground models` to confirm it and its context length.
2. `playground run-agents --agents agent_020 --sessions 2 --adapter lmstudio --max-continuations 1`
   (use a FRESH agent id so no contaminated memory leaks in).
3. Check: does it use most of the budget? does it set goals with `set_goal`? does the observation loop
   recur? Does the briefing stay within budget? Does it ask to continue when it has unfinished work?
4. Compare behaviour counts (throw share, distinct tools, obs%, goals set) against the pre-fix numbers
   in this report. **That** is whether the autonomy improved — not the test count.
