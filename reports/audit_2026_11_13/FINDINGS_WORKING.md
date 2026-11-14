# Working findings (evidence log) — audit 2026-11-13

Source: experiments/playground.db (38 sessions, 1633 actions, read-only), LM Studio server logs
2026-11-12..13 (43 files, 1658 playground requests, parsed by scripts/audit_lmstudio_logs.py),
agent memory dirs experiments/agents/*.

## Throwing
- Action logs: throw = 23/1633 (1.4%). 4B 18/1207 (1.5%), 9B 3/32, 2B 0/378. move 32%, look_around 16%.
- Throwing dominates: (a) ML subsystem 100% throw (features.INPUT/TARGET_COLUMNS, data.generate_throw_dataset,
  data.dataset_from_session_events = throw_release+flight_end only, nn_* tool descriptions "from your own throws",
  controller = throw policy, 5/5 researcher models throw targets, benchmark 'throw_height');
  (b) agent_003 memory: D-002 jump-throw "confirmed", D-003 ring rule "confirmed", G-001, M-001, E-002..E-006,
  program jump_throw — ALL written by the scripted reference policy in session 19 (agent_003 #5), which was
  started as lmstudio_unsloth4b, hit adapter_error, then RESUMED with the CLI default adapter 'scripted'
  (session_started events n=2 lmstudio_unsloth4b, n=7 scripted resumed=True). 70/71 actions scripted.
  Session row still says adapter lmstudio_unsloth4b.
- agent_003 briefing grew 242 -> 22,169 chars; 'throw' mentions 0 -> 23. Session 38 (4B) re-ran jump-throw (E-014).
- No LLM ever called nn_* tools.

## Context
- LM Studio ctx: 2B loaded 12,544-14,592 tokens; 4B 8,192 / 15,360 / 15,872 / 77,056 / 119,040 (varied by session).
- Tool schema 17,493 chars = ~5,000 tokens (46 tools). First request of a fresh agent = 4,998-5,511 tokens.
  agent_003 session 38 first request 14,080 tokens (briefing 22,169 chars).
- Server truncation (truncated = 1 in logs); prompt_tokens fall from ~12-15k to ~6.2k (= tools+system only).
  sid 9 (2B): sent 30x more chars than processed by the end.
- adapters.OpenAICompatAdapter._trim: confirmed bug — after first trim it deletes only the marker it inserted
  and re-inserts it; history never shrinks (sim: limit 20k chars -> 92k after 60 steps). Default limit 120k chars
  anyway far above 15k-token windows.

## Observation loop (2B)
- 2B obs share 87%; longest obs run 126 (sid 9). sid 9 PRE overflow (16 calls) 94% obs dup 0%;
  POST overflow (141 calls) 99% obs, 78% exact duplicate (tool,args,result) calls, 0 physical actions.
  sid 15: dup 4% -> 37%. 4B sid 10 under overflow: dup 0% -> 6%, physical acts continued.

## Objectives
- set_goal: 1 call in 1633 actions — by scripted policy. LLMs: 0. LLM text states objectives
  ("Climb the central tower", "test the ring generation mechanic properly", "full perimeter survey").
- Dashboard 'Self-selected objectives' = goal records of the session only -> "none set".

## Termination
- 21 non-error completed sessions: none used the time budget (1-51%). agent_stopped (3 text-only turns,
  idle_turn_limit=3) x13, agent_ended x7. Sessions 18, 20, 36: 0 actions, model summarised its memory files
  ("All your memory files are read and summarized above") then stopped -> briefing in user role read as a
  request to summarise. Auto-captured diary summaries "Session N complete" feed next briefing.
- No continuation mechanism; run_agents runs fixed N sessions; --resume only for interrupted sessions.

## Evidence/dashboard
- verify_experiment 'value' match: any claimed number within 2% of ANY numeric leaf of a cited result counts
  as measured. E-009/E-010 'platform_count=3' matched 'objects.ring[0]=3.0' (ring x-coordinate) -> consistent.
- Disproven: 0 items anywhere. Mistakes: 1 (scripted). 222/1137 4B actions failed (20%), never recorded.
- agent_003 repeated platform survey across sessions 3,4,9,11,13,14,16 (E-007..E-012 same hypothesis).
- behaviour 'activity for its own sake' = physical actions outside goals/experiments -> with 0 goals set
  this counts absence of goal-tool usage.

## Model comparison
- Not comparable: different ctx windows, n_slots, memory sizes, agents, 9B only 32 actions (all sessions
  ended by 'Model unloaded'). Errors: 'Context size has been exceeded' x4 (4B), model unloaded x3.
