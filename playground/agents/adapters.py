"""Agent adapters: the only place that knows which AI is driving a body.

An adapter turns "here is the conversation so far and the tools you have" into
"here is what I want to do next". The simulator never imports a provider SDK:
both network adapters speak plain HTTP, so any OpenAI-compatible endpoint
(OpenAI, LM Studio, Ollama, vLLM, llama.cpp, OpenRouter, a fine-tune behind a
proxy) or the Anthropic Messages API can be selected from the config file.
"""
from __future__ import annotations

import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from ..interface.schema import anthropic_tools, openai_tools


@dataclass
class ToolCall:
    id: str
    name: str
    args: Any


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    raw: Any = None


class AdapterError(Exception):
    pass


class Adapter:
    """Base class. Subclasses implement begin() and step()."""
    name = "base"
    model = ""
    driven_externally = False

    def begin(self, system: str, tool_names: list[str]) -> None:
        raise NotImplementedError

    def step(self, user_text: str | None, tool_results: list[tuple[ToolCall, dict]]) -> Turn:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"adapter": self.name, "model": self.model}


def _retry(fn: Callable[[], httpx.Response], attempts: int = 4) -> httpx.Response:
    last: Exception | None = None
    for i in range(attempts):
        try:
            r = fn()
            if r.status_code in (408, 409, 425, 429, 500, 502, 503, 504, 529) and i < attempts - 1:
                time.sleep(min(20.0, 2.0 * (2 ** i)))
                continue
            return r
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last = exc
            time.sleep(min(20.0, 2.0 * (2 ** i)))
    raise AdapterError(f"model endpoint unreachable after {attempts} attempts: {last}")


def _result_text(result: dict, limit: int = 6000) -> str:
    s = json.dumps(result, separators=(",", ":"))
    if len(s) > limit:
        s = s[:limit] + f'... [truncated {len(s) - limit} characters; use a narrower tool call for detail]'
    return s


def list_openai_models(base_url: str, api_key_env: str | None = None, timeout: float = 4.0) -> list[dict]:
    """List models an OpenAI-compatible server (LM Studio, Ollama) is offering.

    Prefers LM Studio's richer /api/v0/models (reports per-model state and context length) and falls back
    to the standard /v1/models. Returns [] if the server is unreachable - callers must tolerate that.
    Added for the researcher to pick any loaded model instead of being limited to the two in config.
    """
    import os
    headers = {}
    key = os.environ.get(api_key_env) if api_key_env else None
    if key:
        headers["Authorization"] = f"Bearer {key}"
    root = base_url.rstrip("/")
    # LM Studio's native endpoint lives at the host root (…:1234/api/v0/models), not under /v1.
    host = root[:-3] if root.endswith("/v1") else root
    for url in (f"{host}/api/v0/models", f"{root}/models"):
        try:
            r = httpx.get(url, headers=headers, timeout=timeout)
            if r.status_code != 200:
                continue
            data = r.json().get("data") or []
            if data:
                return data
        except (httpx.HTTPError, ValueError):
            continue
    return []


def probe_context_tokens(base_url: str, model: str, api_key_env: str | None = None) -> int | None:
    """Best-effort: the context length a model is loaded with, via LM Studio's /api/v0/models.

    Returns the loaded context length if known, else the model's max, else None (unknown → caller keeps
    its char-based default). Never raises: an offline server just yields None. Audit 2026-11-13.
    """
    for m in list_openai_models(base_url, api_key_env):
        if m.get("id") == model:
            ctx = m.get("loaded_context_length") or m.get("max_context_length")
            try:
                return int(ctx) if ctx else None
            except (TypeError, ValueError):
                return None
    return None


class OpenAICompatAdapter(Adapter):
    """Any endpoint that implements POST {base_url}/chat/completions with tool calling."""
    name = "openai_compat"

    def __init__(self, base_url: str, model: str, api_key: str | None = None, temperature: float | None = 0.7,
                 max_tokens: int | None = None, timeout: float = 300.0, max_history_chars: int = 120_000,
                 extra_body: dict | None = None, label: str | None = None,
                 context_tokens: int | None = None, reply_reserve_tokens: int = 1024,
                 chars_per_token: float = 3.2):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_history_chars = max_history_chars
        # When the model's loaded context window is known, the history budget is derived from it: the
        # ENTIRE request (system + tool schemas + opening + history + room to reply) must fit, not just
        # the history. Audit 2026-11-13: tool schemas alone are ~5k tokens, so a 15k-token window leaves
        # far less for history than max_history_chars (120k chars ~ 37k tokens) ever assumed, and LM
        # Studio silently dropped the middle of the prompt instead.
        self.context_tokens = context_tokens
        self.reply_reserve_tokens = reply_reserve_tokens
        self.chars_per_token = chars_per_token
        self.extra_body = extra_body or {}
        self.name = label or "openai_compat"
        self.client = httpx.Client(timeout=timeout)
        self.messages: list[dict] = []
        self.tools: list[dict] = []
        self.last_trim: dict | None = None

    def begin(self, system: str, tool_names: list[str]) -> None:
        self.messages = [{"role": "system", "content": system}]
        self.tools = openai_tools(tool_names)

    def _fixed_chars(self) -> int:
        """Characters that every request must carry no matter how much history is trimmed:
        the system prompt (messages[0]), the opening user turn (messages[1]) and the full tool schema."""
        head = self.messages[:2]
        return sum(len(json.dumps(m)) for m in head) + len(json.dumps(self.tools))

    def _history_budget_chars(self) -> int:
        """How many characters the trimmable middle (messages[2:]) may occupy."""
        if self.context_tokens:
            total = int(self.context_tokens * self.chars_per_token)
            avail = total - self._fixed_chars() - int(self.reply_reserve_tokens * self.chars_per_token)
            # Always leave a floor so a single huge fixed part can't drive the budget negative.
            return max(2000, min(self.max_history_chars, avail))
        return self.max_history_chars

    def _trim(self) -> None:
        """Keep the system prompt and the first user turn; drop whole oldest exchanges until the request
        fits the history budget. A single marker records that earlier turns were removed.

        Audit 2026-11-13: the previous version deleted and re-inserted its own marker each call and never
        actually shrank the history; combined with a char budget far above the model's token window, LM
        Studio silently truncated the middle of the prompt instead.
        """
        budget = self._history_budget_chars()
        removed = 0

        def body_size() -> int:
            return sum(len(json.dumps(m)) for m in self.messages[2:]) + self._fixed_chars()

        has_marker = (len(self.messages) > 2 and self.messages[2].get("role") == "user"
                      and "[Earlier turns" in str(self.messages[2].get("content")))
        start = 3 if has_marker else 2
        # Drop oldest complete exchanges (an assistant turn and the tool/user turns answering it) from the
        # middle, never touching messages[0] (system) or messages[1] (opening).
        while body_size() > budget and len(self.messages) - start > 4:
            i = start
            removed += 1
            del self.messages[i]
            while i < len(self.messages) and self.messages[i]["role"] in ("tool", "user"):
                del self.messages[i]
        if removed and not has_marker:
            self.messages.insert(2, {"role": "user", "content": "[Earlier turns of this session were removed "
                                     "to save space. Your memory files and the world itself are unchanged.]"})
        self.last_trim = {"removed_exchanges": removed, "budget_chars": budget,
                          "request_chars": body_size()} if removed else None

    def describe(self) -> dict:
        return {"adapter": self.name, "model": self.model, "context_tokens": self.context_tokens,
                "history_budget_chars": self._history_budget_chars()}

    def step(self, user_text: str | None, tool_results: list[tuple[ToolCall, dict]]) -> Turn:
        for call, result in tool_results:
            self.messages.append({"role": "tool", "tool_call_id": call.id, "content": _result_text(result)})
        if user_text:
            self.messages.append({"role": "user", "content": user_text})
        self._trim()
        body: dict[str, Any] = {"model": self.model, "messages": self.messages, "tools": self.tools,
                                "tool_choice": "auto", **self.extra_body}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = _retry(lambda: self.client.post(f"{self.base_url}/chat/completions", json=body, headers=headers))
        if r.status_code != 200:
            raise AdapterError(f"{self.base_url} returned HTTP {r.status_code}: {r.text[:400]}")
        data = r.json()
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError):
            raise AdapterError(f"unexpected response shape: {json.dumps(data)[:400]}") from None
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            calls.append(ToolCall(id=tc.get("id") or f"call_{uuid.uuid4().hex[:12]}", name=fn.get("name", ""),
                                  args=fn.get("arguments")))
        keep: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
        if calls:
            keep["tool_calls"] = [{"id": c.id, "type": "function",
                                   "function": {"name": c.name, "arguments": c.args if isinstance(c.args, str)
                                                else json.dumps(c.args or {})}} for c in calls]
        self.messages.append(keep)
        return Turn(text=msg.get("content") or "", tool_calls=calls, usage=data.get("usage") or {}, raw=None)


class AnthropicAdapter(Adapter):
    """Anthropic Messages API over HTTP."""
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None, max_tokens: int = 2048, temperature: float | None = None,
                 base_url: str = "https://api.anthropic.com", timeout: float = 300.0,
                 max_history_chars: int = 240_000):
        self.model, self.api_key, self.max_tokens, self.temperature = model, api_key, max_tokens, temperature
        self.base_url = base_url.rstrip("/")
        self.max_history_chars = max_history_chars
        self.client = httpx.Client(timeout=timeout)
        self.system = ""
        self.messages: list[dict] = []
        self.tools: list[dict] = []

    def begin(self, system: str, tool_names: list[str]) -> None:
        self.system, self.messages, self.tools = system, [], anthropic_tools(tool_names)

    def _trim(self) -> None:
        def size() -> int:
            return sum(len(json.dumps(m)) for m in self.messages)
        while size() > self.max_history_chars and len(self.messages) > 6:
            del self.messages[1:3]          # one assistant turn and the user turn answering it

    def step(self, user_text: str | None, tool_results: list[tuple[ToolCall, dict]]) -> Turn:
        content: list[dict] = [{"type": "tool_result", "tool_use_id": c.id, "content": _result_text(r),
                                **({"is_error": True} if not r.get("success", True) else {})}
                               for c, r in tool_results]
        if user_text:
            content.append({"type": "text", "text": user_text})
        if content:
            self.messages.append({"role": "user", "content": content})
        self._trim()
        if not self.api_key:
            raise AdapterError("no API key: set the environment variable named by api_key_env in the config")
        body: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "system": self.system,
                                "messages": self.messages, "tools": self.tools}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        r = _retry(lambda: self.client.post(f"{self.base_url}/v1/messages", json=body, headers=headers))
        if r.status_code != 200:
            raise AdapterError(f"Anthropic API returned HTTP {r.status_code}: {r.text[:400]}")
        data = r.json()
        blocks = data.get("content") or []
        self.messages.append({"role": "assistant", "content": blocks})
        text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        calls = [ToolCall(id=b["id"], name=b["name"], args=b.get("input") or {})
                 for b in blocks if b.get("type") == "tool_use"]
        return Turn(text=text, tool_calls=calls, usage=data.get("usage") or {})


class PolicyAdapter(Adapter):
    """Wraps a Python generator as an agent. Used by the scripted reference agent and by tests.

    The generator yields ("say", text) or ("call", tool, args) and receives the tool result back.
    It has no access to the world: everything it knows arrives through tool results and the
    briefing text, exactly as for a language model.
    """
    name = "policy"

    def __init__(self, policy: Callable[..., Any], label: str = "policy", model: str = "scripted"):
        self.policy_fn, self.name, self.model = policy, label, model
        self.gen = None
        self._pending_text: list[str] = []
        self._done = False

    def begin(self, system: str, tool_names: list[str]) -> None:
        self._system, self._tools = system, tool_names
        self.gen, self._done = None, False

    def step(self, user_text: str | None, tool_results: list[tuple[ToolCall, dict]]) -> Turn:
        if self._done:
            return Turn(text="")
        send = tool_results[-1][1] if tool_results else None
        texts: list[str] = []
        try:
            if self.gen is None:
                self.gen = self.policy_fn(self._system, user_text or "", self._tools)
                item = next(self.gen)
            else:
                item = self.gen.send(send)
            while item[0] == "say":
                texts.append(item[1])
                item = next(self.gen)
        except StopIteration:
            self._done = True
            return Turn(text="\n".join(texts))
        _, tool, args = item
        return Turn(text="\n".join(texts), tool_calls=[ToolCall(f"p_{uuid.uuid4().hex[:10]}", tool, args or {})])


class RandomAdapter(PolicyAdapter):
    """Uniformly random valid-looking actions. A baseline with no intent at all."""

    def __init__(self, seed: int = 0, actions: int = 40):
        rng = random.Random(seed)

        def policy(system, briefing, tools):
            obs = yield ("call", "observe_world", {})
            ids = [o["id"] for o in obs.get("objects", []) if o.get("movable")]
            for _ in range(actions):
                kind = rng.choice(["move", "pick", "throw", "jump", "put", "look"])
                if kind == "move":
                    yield ("call", "move", {"direction": rng.uniform(0, 360), "distance": rng.uniform(0.5, 3.0)})
                elif kind == "pick" and ids:
                    yield ("call", "pick_up", {"object_id": rng.choice(ids)})
                elif kind == "throw" and ids:
                    yield ("call", "throw", {"object_id": rng.choice(ids), "direction": rng.uniform(0, 360),
                                             "force": rng.uniform(20, 200)})
                elif kind == "jump":
                    yield ("call", "jump", {})
                elif kind == "put":
                    yield ("call", "put_down", {})
                else:
                    yield ("call", "look_around", {})
            yield ("call", "end_session", {"reason": "random baseline finished"})
        super().__init__(policy, label="random", model=f"random(seed={seed})")


class ExternalAdapter(Adapter):
    """Placeholder for agents that drive a session through the HTTP API instead of being called."""
    name = "external"
    model = "external"
    driven_externally = True

    def begin(self, system: str, tool_names: list[str]) -> None:
        return None

    def step(self, user_text, tool_results):
        raise AdapterError("external agents drive the session through the HTTP API")


def _make_neural_adapter(cfg: dict, spec: dict) -> "PolicyAdapter":
    """Build a neural controller adapter from a stored model.

    Spec keys:
        owner: 'researcher' or an agent id
        name: model name
        version: optional, defaults to latest
    """
    from ..ml.registry import registry_for
    from ..ml.controller import make_controller_adapter

    owner = spec.get("owner", "researcher")
    name = spec.get("name")
    version = spec.get("version")

    if not name:
        raise AdapterError("neural adapter requires 'name' (model name)")

    # One accessor for researcher and agent registries so a per-agent network loads the same way it was
    # saved (audit 2026-11-13). The agent registry is rooted under cfg['memory']['root'], not a guessed
    # db-sibling 'agents' dir, so a non-default memory.root still resolves.
    scope = "researcher" if owner == "researcher" else "agent"
    reg = registry_for(cfg, scope, owner)
    model = reg.load_model(owner, name, version)

    return make_controller_adapter(model, label=f"neural:{owner}/{name}")


def make_neural_adapter(cfg: dict, owner: str = "researcher", name: str | None = None,
                        version: int | None = None) -> "PolicyAdapter":
    """Public helper: build a neural-controller adapter from owner/name/version without needing a
    config `adapters` entry (the server and CLI call this directly). Added 2026-11-13 (NN lab)."""
    return _make_neural_adapter(cfg, {"owner": owner, "name": name, "version": version})


def make_adapter(cfg: dict, name: str, model: str | None = None, **overrides: Any) -> Adapter:
    """Build an adapter from the `adapters` section of the config."""
    table = cfg.get("adapters", {})
    if name not in table:
        raise AdapterError(f"unknown adapter '{name}'. Configured adapters: {sorted(table)}")
    spec = {**table[name], **{k: v for k, v in overrides.items() if v is not None}}
    kind = spec.get("type")
    key = os.environ.get(spec["api_key_env"]) if spec.get("api_key_env") else None
    if kind == "openai_compat":
        ctx = spec.get("context_tokens")
        if ctx in (None, "auto"):
            ctx = probe_context_tokens(spec["base_url"], model or spec["model"], spec.get("api_key_env"))
        return OpenAICompatAdapter(spec["base_url"], model or spec["model"], api_key=key,
                                   temperature=spec.get("temperature"), max_tokens=spec.get("max_tokens"),
                                   timeout=float(spec.get("timeout", 300)), extra_body=spec.get("extra_body"),
                                   max_history_chars=int(spec.get("max_history_chars", 120_000)), label=name,
                                   context_tokens=int(ctx) if ctx else None,
                                   reply_reserve_tokens=int(spec.get("reply_reserve_tokens", 1024)))
    if kind == "anthropic":
        return AnthropicAdapter(model or spec["model"], key, max_tokens=int(spec.get("max_tokens", 2048)),
                                temperature=spec.get("temperature"),
                                base_url=spec.get("base_url", "https://api.anthropic.com"))
    if kind == "scripted":
        from .scripted import reference_policy
        return PolicyAdapter(reference_policy, label="scripted", model="scripted-reference")
    if kind == "random":
        return RandomAdapter(seed=int(spec.get("seed", 0)), actions=int(spec.get("actions", 40)))
    if kind == "external":
        return ExternalAdapter()
    if kind == "neural":
        return _make_neural_adapter(cfg, spec)
    raise AdapterError(f"adapter '{name}' has unknown type '{kind}'")
