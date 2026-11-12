"""Probe: run every benchmark task once with a fresh and an experienced scripted agent."""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playground.agents import make_adapter  # noqa: E402
from playground.config import load_config  # noqa: E402
from playground.session import run_session  # noqa: E402
from playground.session.benchmark import run_ablation, run_benchmark  # noqa: E402
from playground.storage import Database  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="pg_bench_"))
cfg = load_config(overrides={"_root": str(tmp), "session": {"max_steps": 70}})
db = Database(tmp / "pg.db")
t = time.perf_counter()
for _ in range(3):
    s = run_session(cfg, db, "agent_001", make_adapter(cfg, "scripted"))
    print(f"explore session {s['session_number']}: end={s['end_reason']} calls={s['agent_calls']} "
          f"discoveries={s['memory']['discoveries']}")
print(f"exploration wall {time.perf_counter() - t:.1f}s")

t = time.perf_counter()
res = run_benchmark(cfg, db, lambda a: make_adapter(cfg, "scripted"), ["agent_001"], ["fresh_001"],
                    progress=lambda r: print(f"  {r['group']} {r['task_id']:22s} completed={r['completed']!s:5} "
                                             f"score={r['score']} actions={r['actions']} failed={r['failed_actions']} "
                                             f"sim={r['sim_time']} exps={r['experiments']} "
                                             f"mem={r['memory_use']['relevant_handbook_entries_at_start']} end={r['end_reason']} "
                                             f"err={(r['error'] or '')[:150]}"))
print(f"benchmark wall {time.perf_counter() - t:.1f}s")
print(json.dumps(res["summary"]["differences"], indent=1))

from playground.memory import MemoryStore  # noqa: E402
from playground.session import memory_root  # noqa: E402
m = MemoryStore(memory_root(cfg), "agent_001", cfg)
print("memory items:", [(x["id"], x["kind"], x["status"], (x["text"] or "")[:50]) for x in m.all_items()
                        if x["kind"] == "discovery"])
ring = next(x["id"] for x in m.all_items() if x["kind"] == "discovery" and "ring" in (x["text"] or "").lower())
ab = run_ablation(cfg, db, lambda a: make_adapter(cfg, "scripted"), "agent_001", [ring],
                  tasks=["generate_blocks", "five_towers"],
                  progress=lambda r: print(f"  {r['condition']:12s} {r['task_id']:18s} completed={r['completed']} "
                                           f"score={r['score']} actions={r['actions']}"))
print("ablation diff:", json.dumps(ab["summary"]["differences"]["per_task"]))
print("tmp:", tmp)
