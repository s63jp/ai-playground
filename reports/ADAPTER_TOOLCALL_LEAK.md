# Adapter anomaly log: chat-template / tool-call markup leaking into agent output

Recorded 2026-11-16. The leaked markup is a **recurring, cross-model phenomenon**,
predating the 2026-11-16 pilot runs. Database search across ALL stored messages:
43 messages in 10 sessions contain `<function=…>` / `<|start|>` markup:

| sessions | agent | model | leaked turns | sessions ended `agent_stopped` |
|---|---|---|---|---|
| 32–41 | agent_010 | `liquid/lfm2-24b-a2b` | 33 | 9 of 10 (after only 1–6 actions) |
| 49–50 | agent_012 | `nemotron4b` | 4 | (incl. the never-terminated session-50 row) |

The pilot runs 51–54 (`lowkeyqwen3.52b`, `qwen/qwen3.5-9b`) contain **no** leaked markup —
but session 51 (2B) shows the sibling anomaly: no-tool turns whose completions were fully
empty (no text, no tool call, no `messages` row content).

## Observed

The dashboard "Agent's words" panel shows the model's raw chat-template markup,
verbatim, e.g.:

```
<|start|>assistant<|end|> <function=end_session> <parameter=reason> … </parameter>
<parameter=continue_next_session> False </parameter> </function>
```

Seen repeated (the model resent the identical attempted call each time it was nudged).

## Impact (measured)

For agent_010 on `liquid/lfm2-24b-a2b` the leak is not cosmetic: 9 of its 10 sessions
(32–41) terminated as `agent_stopped` after **1–6 actions** — the model wrote template
markup instead of tool calls, the calls never registered, the three-strike idle rule
ended the session. Almost that agent's complete recent session history is empty-run data,
and `agent_stopped` ("the agent chose to stop") is recorded where the agent's own
`end_session` attempts could not register at all. Any benchmark or behaviour totals for
that agent over those sessions measure the parsing gap, not the model.

## Interpretation

`<function=name> <parameter=…>` is the model GGUF's own function-calling template, not
OpenAI structured tool_calls. On these turns LM Studio handed the template text through
as ordinary message content instead of parsing it. Where that happens the adapter sees a
no-tool turn:

* the runner counts it against `session.idle_turn_limit` (3 even with acting turns in
  between per its cumulative idling rule) and can end a session with `agent_stopped`
  while 44 actions of budget remained (session 51, agent_001, 16 actions of 60);
* an attempted `end_session` inside such text never registers (session could end
  `agent_stopped` where the model actually *asked* to end), and the same attempted call
  is re-narrated instead of executed;
* the two earliest no-tool turns in session 51 stored *fully empty* completions (no text,
  no tool calls, nothing in `messages`) — consistent with the same model spending its
  reply budget inside reasoning/template output.

## Suggested fixes (status, 2026-11-16)

1. ~~**Fallback parser**~~ — **DONE** (commit `0e1e2c8`): fully-formed `<function=…>` and
   fenced-JSON blocks are recovered as schema-validated calls when the API returns none;
   `reasoning_content` fallback for empty completions; 14 HVE tests.
2. ~~**Sanitise what is stored**~~ — **DONE, display-side** (commit `34df7f1`): the dashboard's
   "Agent's words" strips template tokens and markup before rendering (with a pointer to
   Recent actions when nothing clean remains); the database keeps raw content verbatim.
   Display-only by choice: raw is evidence, the DB is the authority.
3. **Measure it** — open: count template-leak turns per session as a gateway/monitor counter
   so cross-model comparisons include "how often model X tries to call tools as text".
4. ~~**Idle accounting**~~ — **MOOT for the leaking models**: with recovery in place, a turn
   with embedded tool markup becomes an acted turn and no longer counts as idle.

## Not a defect in the harness

The environment never executed anything from the leaked markup — `write_diary`,
`climb`, `walk_to` in the same session were real, schema-validated tool calls. This is a
provider-side parsing gap (LM Studio + GGUF preset) meeting an adapter that only reads
structured `tool_calls`.

## Addendum, 2026-11-16: the nemotron-3 shape leaks PAST the dashboard sanitiser

**Seen live** (agent_008, session 79 s4, `nvidia/nemotron-3-nano-4b` via LM Studio): the
"Agent's words" panel showed 14 identical `"/>"`-shaped fragments — the leftover tail of
the model's own tool-call template.

This model's text-written calls have **two** markup parts per call:

```
<function=move>
<parameter=direction>
east
</function>
<tool_call>          ← the close tag, emitted as plain TEXT after </function>
```

`parse_embedded_tool_calls` already recovers the calls (fix 1), and the dashboard
sanitiser stripped the `<function=…>…</function>` block — but nothing stripped the
trailing close tag, so every markup-only turn rendered as that fragment.

**Measured in the database:** 24 message rows contain the stray close tag —
sessions 49–50 (agent_012, `nemotron4b`: 4 rows) and sessions 79–84 (agent_008,
`nemotron-3-nano-4b`: 20 rows, incl. the still-running session 82). So the lfm-2-era
sessions carried it too; it was hidden behind the larger `<function=…>…</function>`
block which the sanitiser then stripped whole — the nemotron-3 rows have short param
value tails, no.

**Fixed (display-side, same commit policy as fix 2):** the sanitiser now also strips a
bare `<tool_call>` / `
