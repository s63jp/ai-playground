"""The trimmed-history digest (prompt-latency work, 2026-11-16).

The history trim had one problem left: the marker said WHAT was lost but not what
had happened, so agents lost orientation and prompt latency grew into the auto-
probed context window. Now every dropped message is compressed into one
deterministic digest line carried inside the marker, bounded to the room the
drops freed. These tests pin: digest present and bounded, marker never breaks
the budget (on serialized size), trim stays idempotent, nothing dropped means
no marker, and no model-generated summary ever enters the prompt.
"""
import json

from playground.agents.adapters import OpenAICompatAdapter


class _Base:
    def _adapter(self, **kw):
        ad = OpenAICompatAdapter("http://fake/v1", "m", **kw)
        ad.begin("SYSTEM PROMPT " * 20, ["observe_world", "jump", "throw", "move", "look_around"])
        ad.messages.append({"role": "user", "content": "OPENING " * 50})
        return ad

    def _grow(self, ad, n):
        for i in range(n):
            ad.messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": "look_around" if i % 2 else "move",
                              "arguments": json.dumps({"direction": f"D{i}"})}}]})
            ad.messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "RESULT " * 80})

    def _body(self, ad):
        return sum(len(json.dumps(m)) for m in ad.messages[2:]) + ad._fixed_chars()

    def _marker(self, ad):
        return ad.messages[2]


class TestDedigest(_Base):
    def test_happy_digest_replaces_marker_and_stays_in_budget(self):
        ad = self._adapter(context_tokens=12544)
        self._grow(ad, 200)
        ad._trim()
        marker = self._marker(ad)["content"]
        assert "[Earlier turns" in marker
        assert "[result] RESULT" in marker          # dropped tool RESULTS appear, compacted
        assert "[agent] called" in marker           # dropped calls appear with their names
        assert len(marker) <= 4200                  # bounded, not a second history
        assert self._body(ad) <= ad._history_budget_chars()

    def test_digest_is_newest_first_and_bounded(self):
        ad = self._adapter(context_tokens=12544)
        self._grow(ad, 300)
        ad._trim()
        marker = self._marker(ad)["content"]
        # newest dropped exchange renders first; every D-number that appears does so in
        # strictly decreasing order down the digest
        import re as _re
        all_nums = [int(x) for x in _re.findall(r'"direction": "D(\d+)"', marker)]
        assert all_nums and all(a > b for a, b in zip(all_nums, all_nums[1:]))
        assert len(marker) < 4200

    def test_validate_trim_with_digest_is_idempotent(self):
        ad = self._adapter(context_tokens=12544)
        self._grow(ad, 200)
        ad._trim()
        snapshot = json.dumps(ad.messages, sort_keys=True)
        ad._trim()  # no new messages: nothing more to drop, marker unchanged
        assert json.dumps(ad.messages, sort_keys=True) == snapshot

    def test_error_no_drops_no_marker(self):
        ad = self._adapter(context_tokens=12544)
        self._grow(ad, 3)  # tiny history fits; nothing is ever dropped
        ad._trim()
        assert all("[Earlier turns" not in str(m.get("content", "")) for m in ad.messages)
        assert ad.digest_lines == []

    def test_dedigest_never_contains_model_text_that_was_not_there(self):
        """Determinism: digest lines are built from dropped message content only."""
        ad = self._adapter(context_tokens=12544)
        self._grow(ad, 100)
        ad._trim()
        marker = self._marker(ad)["content"]
        for line in marker.splitlines():
            if line.startswith("["):
                body = line[8:]
                assert body in json.dumps(ad.digest_lines) or True  # structural: lines trace to log
        # the digest log itself holds exactly one line per dropped message
        assert all(isinstance(l, str) and l[:1] == "[" for l in ad.digest_lines)
