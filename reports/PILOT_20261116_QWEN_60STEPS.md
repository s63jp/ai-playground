# Pilot: qwen 60-step sessions on agents 001–002 (2026-11-16)

Four sessions, same island, same 60-action budget, all integrity chains OK, zero runner errors.
Comparison is descriptive only (2 agents, single sessions — no significance claims).

| | session 51 | session 52 | session 53 | session 54 |
|---|---|---|---|---|
| model | lowkeyqwen3.52b | lowkeyqwen3.52b | **qwen/qwen3.5-9b** | **qwen/qwen3.5-9b** |
| agent | agent_001 (13 prior sessions) | agent_002 | agent_001 | agent_002 |
| actions | 16 / 60 | 63* | 47 / 60 | 57 / 60 |
| end reason | `agent_stopped` (idle×3) | `agent_ended` | `agent_ended` | `agent_ended` |
| physical actions | 9 | — | 21 | 26 |
| failed actions | 8 | — | 12 | 9 |
| experiments recorded | 0 | 0 | 0 | 0 |
| template leak turns | 0 (empty completions instead) | 0 | 0 | 0 |

*63 = agent took its stop-continuation path (`agent_ended`).

## Findings

1. **The 9B used its budget and ended itself properly.** 47/60 and 57/60 actions, `agent_ended`
   both times, no idle cutoffs — vs the 2B's 16-action idle miss (session 51) and its own
   early end (session 52). Context budgeting (`context_tokens: auto`, reply reserve) behaved:
   long memory briefings + 40-tool schemas stayed inside the window.
2. **No template leak on either 9B session** — the `<function=…>` markup phenomenon is
   specific to `liquid/lfm2-24b-a2b` (33 leaked turns across sessions 32–41) and
   `nemotron4b` (4, sessions 49–50); the recovery parser (commit 0e1e2c8) is in place for
   them, untested against a live endpoint yet.
3. **The record loop works — it is an action-budget phenomenon.** The system has recorded
   experiments before, and on these same models: agent_002 on `qwen/qwen3.5-9b` recorded
   15 experiment/discovery records across its longer, uncapped sessions 11–13 (61–66
   actions each), and agent_001 recorded 9 items on `unslothqen 4b` (sessions 3–5, 33–47
   actions). The same 9B recorded **nothing** under the 60-step cap (sessions 53–54,
   47/57 actions). Combined with the earlier 2B runs, the differentiator is session
   length, not model or interface: reaching the hypothesis → `record_experiment` →
   evidence-check loop takes tens of actions, and a 60-step cap that also spends turns on
   exploration reaches it only by luck. The evidence gate did its job in those longer
   sessions, too: agent_002 requested `confirmed` six times and was granted it **never** —
   `hypothesis` twice (no checked evidence), `probable` three times (one checked
   experiment each). Claiming is not knowing, by construction.
4. **Memory reads: 0** in the four pilot sessions (`counters.memory_reads`). The briefing itself
   loads memory at session start, but no in-session recall was made by either model.

## Validity notes

- Sessions 51–52 were accidentally on the wrong model (the `lmstudio` adapter default in
  `config/default.yaml` points at `lowkeyqwen3.52b`); they are still valid as 2B datapoints
  but the label `pilot_qwen9b_60steps` is misleading — kept for honesty, superseded by
  `pilot_qwen9b_60steps_v2` (sessions 53–54).
- agent_010's sessions 32–41 remain flagged as invalid data by
  [ADAPTER_TOOLCALL_LEAK.md](ADAPTER_TOOLCALL_LEAK.md).
