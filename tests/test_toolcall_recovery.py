"""Embedded tool-call recovery (reports/ADAPTER_TOOLCALL_LEAK.md).

Some models emit their function-call template as plain text instead of
structured tool_calls; 9 of agent_010's 10_sessions on lfm2-24b ended after
1-6 actions because those attempts never registered. These tests pin the
recovery behaviour: parse only fully-formed markup, never widen what a model
can do (the environment schema still validates every call), and leave plain
text untouched.
"""
import json

import httpx
import pytest

from playground.agents.adapters import OpenAICompatAdapter, ToolCall, parse_embedded_tool_calls

BQ = chr(96)
TC_OPEN = BQ + "antml:tool_calls" + BQ
TC_CLOSE = BQ + "/antml:tool_calls" + BQ

# The exact shape observed in stored agent messages (agent_010, lfm2-24b).
LEAK_SAMPLE = ("<|start|>assistant<|end|> <function=end_session> "
               "<parameter=reason> Exploration session concluded after multiple attempts. </parameter> "
               "<parameter=continue_next_session> False </parameter> </function>")

# --------------------------------------------------------------------------- happy
def test_recovers_the_observed_parameter_style():
    calls = parse_embedded_tool_calls(LEAK_SAMPLE)
    assert len(calls) == 1
    assert isinstance(calls[0], ToolCall)
    assert calls[0].name == "end_session"
    args = json.loads(calls[0].args)
    assert args == {"reason": "Exploration session concluded after multiple attempts.",
                    "continue_next_session": False}


def test_parameter_values_parse_as_json_when_valid():
    calls = parse_embedded_tool_calls('<function=throw><parameter=directions>{"directions": "N 30 E"}</parameter></function>')
    assert json.loads(calls[0].args)["directions"] == {"directions": "N 30 E"}


def test_recovers_multiple_function_blocks():
    text = '<function=observe_world></function> then <function=move><parameter=direction>N</parameter></function>'
    calls = parse_embedded_tool_calls(text)
    assert [c.name for c in calls] == ["observe_world", "move"]
    assert json.loads(calls[1].args) == {"direction": "N"}


def test_recovers_fenced_json_blocks_single_and_array():
    single = TC_OPEN + json.dumps({"name": "jump", "arguments": {"wait": "landing"}}) + TC_CLOSE
    calls = parse_embedded_tool_calls("thinking... " + single + " more")
    assert len(calls) == 1 and calls[0].name == "jump"
    array = TC_OPEN + json.dumps([{"name": "observe_world", "arguments": {}},
                                  {"name": "end_session", "arguments": {"reason": "done"}}]) + TC_CLOSE
    calls = parse_embedded_tool_calls(array)
    assert [c.name for c in calls] == ["observe_world", "end_session"]


def test_recovery_cap_twelve_calls():
    text = " ".join(f"<function=observe_world></function>" for _ in range(30))
    assert len(parse_embedded_tool_calls(text)) == 12


# --------------------------------------------------------------------------- validate
def test_unterminated_function_block_is_left_alone():
    assert parse_embedded_tool_calls('<function=jump><parameter=wait>landing') == []
    assert parse_embedded_tool_calls('<function=jump>') == []


def test_garbage_marker_block_is_left_alone():
    bad = TC_OPEN + "not json at all" + TC_CLOSE
    assert parse_embedded_tool_calls(bad) == []


def test_marker_block_needs_a_name():
    bad = TC_OPEN + json.dumps({"arguments": {}}) + TC_CLOSE
    assert parse_embedded_tool_calls(bad) == []


def test_plain_text_yields_nothing():
    assert parse_embedded_tool_calls("I should probably climb the terrace next.") == []
    assert parse_embedded_tool_calls("") == []
    assert parse_embedded_tool_calls("<|start|>assistant<|end|> narration only") == []


# --------------------------------------------------------------------------- error/effect
def _canned_adapter(content, tool_calls=None, reasoning=None):
    message = {"content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    payload = {"choices": [{"message": message}], "usage": {"total_tokens": 5}}
    handler = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    adapter = OpenAICompatAdapter("http://fake.local/v1", "fake-model")
    adapter.client = httpx.Client(transport=handler)
    adapter.begin("system text", ["end_session", "observe_world"])
    return adapter


def test_adapter_step_recovers_calls_from_leaked_content():
    adapter = _canned_adapter(LEAK_SAMPLE)
    turn = adapter.step("go", [])
    assert [c.name for c in turn.tool_calls] == ["end_session"]
    # the raw text stays in the conversation history as evidence
    assert "<function=end_session>" in adapter.messages[-1]["content"]
    assert turn.text == LEAK_SAMPLE


def test_adapter_step_prefers_structured_calls_over_markup():
    structured = [{"id": "c1", "type": "function",
                   "function": {"name": "observe_world", "arguments": "{}"}}]
    turn = _canned_adapter(LEAK_SAMPLE, tool_calls=structured).step("go", [])
    assert [c.name for c in turn.tool_calls] == ["observe_world"]


def test_adapter_step_falls_back_to_reasoning_content():
    turn = _canned_adapter("", reasoning=LEAK_SAMPLE).step("go", [])
    assert [c.name for c in turn.tool_calls] == ["end_session"]


def test_recovered_unknown_tool_still_fails_like_any_call():
    """Recovery must not widen what a model can do: an unknown name reaches the
    environment and receives the ordinary tool error. Here we only assert the
    call leaves the adapter with its name intact - the environment's schema
    does the refusing (covered by the suite's Validate tests)."""
    calls = parse_embedded_tool_calls("<function=not_a_real_tool></function>")
    assert calls[0].name == "not_a_real_tool"
    assert calls[0].args is None


def test_adapter_without_markup_behaves_as_before():
    turn = _canned_adapter("I will climb the terrace.").step("go", [])
    assert turn.tool_calls == []
    assert turn.text == "I will climb the terrace."
    assert "tool_calls_recovered_from_text" not in turn.usage


# --------------------------------------------------------------------------- leak counter (fix 3)
def test_recovered_calls_are_counted_in_usage():
    turn = _canned_adapter(LEAK_SAMPLE).step("go", [])
    assert turn.usage.get("tool_calls_recovered_from_text") == 1


def test_structured_calls_are_not_counted_as_recovered():
    structured = [{"id": "c1", "type": "function",
                   "function": {"name": "observe_world", "arguments": "{}"}}]
    turn = _canned_adapter("note", tool_calls=structured).step("go", [])
    assert "tool_calls_recovered_from_text" not in turn.usage
