"""Stage C: backend-enforced concurrency limits + the fair, prioritised request queue.

The ResourceManager is exercised directly with threads (deterministic via snapshot-polling,
no fixed sleeps for the assertions), and one end-to-end test proves the limit is enforced in
the real gateway request path, not just the UI.
"""
from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from playground.config import load_config
from playground.dashboard import create_app
from playground.gateway.concurrency import Cancelled, QueueFull, ResourceManager


def wait_until(pred, timeout=3.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}
        self.text = ""

    def json(self):
        return self._payload


def _completion():
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}


# --------------------------------------------------------------------------- disabled fast path
def test_disabled_by_default():
    rm = ResourceManager({})
    assert rm.enabled is False
    adm = rm.acquire("p", "m", "a")        # returns immediately, no accounting
    assert adm.scopes == []
    rm.release(adm)


def test_enabled_when_any_limit_set():
    assert ResourceManager({"max_concurrent": 1}).enabled is True
    assert ResourceManager({"per_model": {"m": 2}}).enabled is True


# --------------------------------------------------------------------------- serialisation + queue
def test_per_model_limit_serialises_and_queues():
    rm = ResourceManager({"per_model": {"m": 1}})
    release1 = threading.Event()
    held = {}

    def holder():
        held["adm"] = rm.acquire("p", "m", "agent1")
        release1.wait(3)
        rm.release(held["adm"])

    threading.Thread(target=holder, daemon=True).start()
    assert wait_until(lambda: rm.snapshot()["active"].get("model:m") == 1)

    result = {}

    def second():
        try:
            adm = rm.acquire("p", "m", "agent2", poll=0.02)
            result["got"] = True
            rm.release(adm)
        except Exception as e:  # noqa: BLE001
            result["err"] = e

    threading.Thread(target=second, daemon=True).start()
    assert wait_until(lambda: rm.snapshot()["queue_depth"] == 1)   # second is waiting
    assert "got" not in result                                     # ... and has NOT been admitted
    release1.set()
    assert wait_until(lambda: result.get("got") is True)           # frees -> second admitted


# --------------------------------------------------------------------------- priority
def test_priority_order():
    rm = ResourceManager({"max_concurrent": 1})
    hold = threading.Event()

    def holder():
        a = rm.acquire("p", "m", "holder")
        hold.wait(3)
        rm.release(a)

    threading.Thread(target=holder, daemon=True).start()
    assert wait_until(lambda: rm.snapshot()["active"].get("global") == 1)

    admitted = []

    def worker(name, pr):
        a = rm.acquire("p", "m", name, priority=pr, poll=0.02)
        admitted.append(name)
        rm.release(a)

    threading.Thread(target=worker, args=("low", 1), daemon=True).start()
    assert wait_until(lambda: rm.snapshot()["queue_depth"] == 1)
    threading.Thread(target=worker, args=("high", 9), daemon=True).start()
    assert wait_until(lambda: rm.snapshot()["queue_depth"] == 2)
    hold.set()
    assert wait_until(lambda: len(admitted) == 2)
    assert admitted[0] == "high"            # higher priority served first despite arriving later


# --------------------------------------------------------------------------- queue_max
def test_queue_full_rejects():
    rm = ResourceManager({"max_concurrent": 1, "queue_max": 1})
    hold = threading.Event()

    def holder():
        a = rm.acquire("p", "m", "holder")
        hold.wait(3)
        rm.release(a)

    threading.Thread(target=holder, daemon=True).start()
    assert wait_until(lambda: rm.snapshot()["active"].get("global") == 1)
    threading.Thread(target=lambda: rm.acquire("p", "m", "w1", poll=0.05), daemon=True).start()
    assert wait_until(lambda: rm.snapshot()["queue_depth"] == 1)
    with pytest.raises(QueueFull):           # queue already has its one allowed waiter
        rm.acquire("p", "m", "w2")
    hold.set()


# --------------------------------------------------------------------------- cancellation
def test_cancel_while_queued():
    rm = ResourceManager({"max_concurrent": 1})
    hold = threading.Event()
    threading.Thread(target=lambda: (lambda a: (hold.wait(3), rm.release(a)))(rm.acquire("p", "m", "h")),
                     daemon=True).start()
    assert wait_until(lambda: rm.snapshot()["active"].get("global") == 1)
    flag = {"cancel": False}
    result = {}

    def waiter():
        try:
            rm.acquire("p", "m", "x", is_cancelled=lambda: flag["cancel"], poll=0.02)
        except Cancelled:
            result["cancelled"] = True

    threading.Thread(target=waiter, daemon=True).start()
    assert wait_until(lambda: rm.snapshot()["queue_depth"] == 1)
    flag["cancel"] = True
    assert wait_until(lambda: result.get("cancelled") is True)
    assert rm.snapshot()["queue_depth"] == 0      # the cancelled waiter left the queue
    hold.set()


# --------------------------------------------------------------------------- per-agent anti-monopoly
def test_per_agent_cap_prevents_monopoly():
    rm = ResourceManager({"max_concurrent": 2, "per_agent": {"greedy": 1}})
    hold = threading.Event()

    def holder():
        a = rm.acquire("p", "m", "greedy")           # greedy takes its one allowed slot
        hold.wait(3)
        rm.release(a)

    threading.Thread(target=holder, daemon=True).start()
    assert wait_until(lambda: rm.snapshot()["active"].get("agent:greedy") == 1)

    greedy2 = {}

    def greedy_again():
        try:
            rm.acquire("p", "m", "greedy", poll=0.02, is_cancelled=lambda: greedy2.get("stop", False))
            greedy2["got"] = True
        except Cancelled:
            greedy2["cancelled"] = True

    threading.Thread(target=greedy_again, daemon=True).start()
    assert wait_until(lambda: rm.snapshot()["queue_depth"] == 1)      # greedy's 2nd request is blocked

    other = {}

    def other_agent():
        a = rm.acquire("p", "m", "other", poll=0.02)                  # a different agent can still run
        other["got"] = True
        rm.release(a)

    threading.Thread(target=other_agent, daemon=True).start()
    assert wait_until(lambda: other.get("got") is True)               # not starved by greedy
    assert "got" not in greedy2                                       # greedy still capped at 1
    greedy2["stop"] = True                                            # unblock the test
    hold.set()


# --------------------------------------------------------------------------- live config + snapshot
def test_update_limits_and_snapshot():
    rm = ResourceManager({"max_concurrent": 1})
    snap = rm.snapshot()
    assert snap["enabled"] and snap["limits"]["max_concurrent"] == 1 and snap["queue_depth"] == 0
    new = rm.update_limits({"max_concurrent": 5, "per_model": {"m": 2}})
    assert new["max_concurrent"] == 5 and new["per_model"] == {"m": 2}


# --------------------------------------------------------------------------- end-to-end enforcement
def test_endpoint_enforces_per_model_limit(cfg, db, monkeypatch):
    monkeypatch.setattr("playground.gateway.providers.list_openai_models",
                        lambda base, *a, **k: [{"id": "test-model", "state": "loaded"}] if "1234" in base else [])
    c = {**cfg, "gateway": {"concurrency": {"per_model": {"test-model": 1}}}}
    app = create_app(c, db)
    gw = app.state.gateway["chat_gateway"]

    gate = threading.Event()
    active = {"n": 0, "max": 0}
    lock = threading.Lock()

    def slow_post(url, json, headers):
        with lock:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        gate.wait(3)
        with lock:
            active["n"] -= 1
        return FakeResp(200, _completion())

    monkeypatch.setattr(gw.client, "post", slow_post)
    body = {"model": "test-model", "messages": [{"role": "user", "content": "x"}]}
    results = []

    def fire():
        results.append(TestClient(app).post("/v1/chat/completions", json=body).status_code)

    threading.Thread(target=fire, daemon=True).start()
    assert wait_until(lambda: active["n"] == 1)                 # first request reached the provider
    threading.Thread(target=fire, daemon=True).start()
    assert wait_until(lambda: gw.rm.snapshot()["queue_depth"] == 1)   # second is queued, not sent
    assert active["n"] == 1
    gate.set()
    assert wait_until(lambda: len(results) == 2)
    assert active["max"] == 1                                   # provider NEVER saw two at once
    assert results == [200, 200]
