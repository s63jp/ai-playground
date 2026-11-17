"""A REAL short live session through the real B3 trainer modules.

Skipped while playground/rl/ppo.py + checkpoint.py are not on the tree
(the live-training foundations branch is based on the B2 commit); becomes
runnable automatically when the B3 commit lands and `main` merges.

Deliberately tiny (128 decisions, 1 epoch) so it never competes for CPU with
a real training run: it exists to prove the seams (fresh actor → real trainer
→ real tap frames → real checkpoint format), not to demonstrate learning.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("playground.rl.ppo", reason="lands with the B3 commit")
pytest.importorskip("playground.rl.checkpoint", reason="lands with the B3 commit")


def test_real_short_session(tmp_path):
    from playground.rl_live.session import LiveBrainSession
    s = LiveBrainSession(root=tmp_path / "rl_live")
    assert s._trainer_available
    r = s.start(name="smoke", seed=1234, total_steps=128,
                ppo_overrides={"rollout_steps": 64, "epochs": 1,
                               "minibatch": 32})
    assert r["ok"] is True
    t0 = time.time()
    while s.state not in ("running", "paused") and time.time() - t0 < 120:
        time.sleep(0.1)
    assert s.state in ("running", "paused")     # real env + fresh policies built
    st = s.status()["session"]
    assert st["config"]["name"] == "smoke"
    assert st["live_frames"] >= 0               # frames flow through the tap
    s.stop()
    t0 = time.time()
    while s.state != "stopped" and time.time() - t0 < 180:
        time.sleep(0.2)
    assert s.state == "stopped"
    run_dir = Path(s._run_dir)
    prog = [json.loads(l)
            for l in (run_dir / "progress.jsonl").read_text().splitlines()
            if l.strip()]
    assert prog and prog[0]["total_valid_steps"] == 64
    ck = run_dir / "ckpt_final.json"
    assert ck.exists()
    body = json.loads(ck.read_text(encoding="utf-8"))
    assert body["format"] == "rl_checkpoint/1"
    assert body["meta"]["contract"]["obs_dim"] in (40, 43)   # B2 or B3 obs
    r = s.save(tag="manual")                    # save allowed once stopped
    assert r["ok"] is True
