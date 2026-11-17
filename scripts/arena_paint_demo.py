"""Record a REAL scored paintball duel into the live db (M5.5 visual-demo evidence).

Same production path as `playground arena`: run_arena_match + the real factories.
Combatants are PolicyAdapter-driven (like the M5 recording): they equip their marker,
measure the opponent with look_around and fire per-shot from the MEASURED data
(look_around gives the held item's position = the muzzle and the opponent's position).

Usage:  .venv/Scripts/python.exe scripts/arena_paint_demo.py [--duration 20]
The sessions land under agent_arena_* ids with label arena:m55-demo (never the roster).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playground.agents.adapters import PolicyAdapter  # noqa: E402
from playground.arena.factories import run_arena_match  # noqa: E402
from playground.config import load_config  # noqa: E402
from playground.storage import Database  # noqa: E402


def make_policy(item_id: str):
    """Equip the marker, then per shot: measure, aim at the nearest standing foe, fire."""
    def policy(system: str, opening: str, tools: list[str]):
        yield ("call", "equip", {"object_id": item_id})
        yield ("say", f"Equipped {item_id}; scanning for opposing team.")
        while True:
            look = yield ("call", "look_around", {"radius": 9.0})
            muzzle = None
            for ob in (look or {}).get("objects", []):
                if ob.get("held") and ob.get("position"):
                    muzzle = ob["position"]
            foes = [a for a in (look or {}).get("agents_nearby", [])
                    if not a.get("fallen") and a.get("position")]
            if muzzle is None or not foes:
                yield ("say", "Still scanning; nothing measurable yet.")
                continue
            t = foes[0]["position"]
            m = muzzle
            d = np.array([t[0] - m[0], t[1] - m[1], (t[2] + 0.35) - m[2]], dtype=float)
            n = float(np.linalg.norm(d)) or 1.0
            aim: List[float] = [round(float(v / n), 4) for v in d]
            res = yield ("call", "fire", {"direction": aim})
            if isinstance(res, dict) and res.get("success") is False:
                yield ("say", "Out of paint."); break
    return policy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--turn-interval", type=float, default=1.0)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "arena_paint_demo_report.json"))
    args = ap.parse_args()

    cfg = load_config()                     # the REAL config/db
    db = Database(cfg["database"])
    match_cfg = {
        "teams": [
            {"name": "Red", "agents": [{"id": "agent_arena_r1"}], "equipment": ["paintball_rifle"]},
            {"name": "Blue", "agents": [{"id": "agent_arena_b1"}], "equipment": ["paintball_pistol"]},
        ],
        "rules": {"duration_s": args.duration, "pellets": 24, "score_target": 100, "friendly_fire": False},
    }
    def adapter_factory(agent_id: str, plan_item: dict):
        # the item each team's first slot carries (matching the factory layout order)
        if str(agent_id).startswith("agent_arena_r"):
            return PolicyAdapter(make_policy("red_item_0"), label="paint-demo", model="policy-duel")
        return PolicyAdapter(make_policy("blue_item_0"), label="paint-demo", model="policy-duel")

    from playground.arena.match import export_match_report
    result = run_arena_match(match_cfg, adapter_factory, db=db, root_cfg=cfg,
                             turn_interval_s=args.turn_interval, match_name="m55-demo")
    export_match_report(result, args.out)
    print(json.dumps({
        "end_reason": result.get("end_reason"), "winner": result.get("winner"),
        "team_stats": result.get("team_stats"), "agent_stats": result.get("agent_stats"),
        "timeline_events": len(result.get("timeline") or []),
        "sessions": result.get("sessions"), "report": args.out,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
