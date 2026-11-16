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

## Suggested fixes (in priority order)

1. **Fallback parser in `OpenAICompatAdapter`**: after a no-tool turn whose text contains
   `<function=…>` / `<parameter=…>` (or `<tool_call>` variants), parse the embedded calls
   like any other tool call. An attempted `end_session` then behaves as one. Log the
   recovery as `adapter.tool_call_recovered` so provenance is honest.
2. **Sanitise what is stored**: strip template tokens (`<|start|>`, `<|end|>`,
   `<|im_start|>` …) from anything written to `messages` / "say" so the dashboard shows
   the agent's words, the template's. Keep the raw content in the DB row for the audit
   trail (raw is evidence; the dashboard view renders it clean).
3. **Measure it**: count template-leak turns per session (a gateway/monitor counter)
   so cross-model comparisons include "how often model X tried to call tools as text".
4. **Idle accounting**: today a no-tool turn counts against 3 even when the model
   immediately afterwards acts again (rule is cumulative-per-streak but resets only on
   a tool-calling turn in `session/runner.py:284`). With a fallback parser in place this
   matters less, but consider treating a turn with *any* tool-attempt markup as acted.

## Not a defect in the harness

The environment never executed anything from the leaked markup — `write_diary`,
`climb`, `walk_to` in the same session were real, schema-validated tool calls. This is a
provider-side parsing gap (LM Studio + GGUF preset) meeting an adapter that only reads
structured `tool_calls`.
