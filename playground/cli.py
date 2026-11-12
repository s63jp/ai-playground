"""Command-line interface.

    playground serve                         dashboard + API
    playground explore --agent agent_001     one exploration session
    playground run-agents --sessions 3       all 13 agents
    playground memory show agent_001         inspect memory
    playground benchmark --group-a agent_001 Group A vs Group B
    playground ablate agent_001 --remove D-002
    playground demo                          the end-to-end demonstration
    playground verify-log 12                 check a session's hash chain
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .config import agent_ids, load_config, resolve_path


def _db(cfg):
    from .storage import Database
    return Database(resolve_path(cfg, cfg["database"]))


def _cfg(args):
    over = {}
    if getattr(args, "minutes", None) is not None:
        over.setdefault("session", {})["duration_minutes"] = args.minutes
    if getattr(args, "max_steps", None) is not None:
        over.setdefault("session", {})["max_steps"] = args.max_steps
    return load_config(args.config, over or None)


def _adapter_factory(cfg, args):
    from .agents import make_adapter
    return lambda agent_id: make_adapter(cfg, args.adapter, model=args.model)


def _print_summary(s: dict) -> None:
    iv = s["integrity"]["what_was_independently_verified"]
    print(f"[{s['agent_id']}] session {s['session_number']} (id {s['session_id']}): end={s['end_reason']} "
          f"calls={s['agent_calls']} sim={s['sim_time']}s wall={s['wall_seconds']}s")
    if s.get("error"):
        print(f"  ERROR: {s['error'].splitlines()[0]}")
    print(f"  experiments consistent/contradicted: {iv['experiments_consistent_with_engine']}/"
          f"{iv['experiments_contradicted_by_engine']} · claims verified/rejected: {iv['claims_verified']}/"
          f"{iv['claims_rejected']} · behaviour: {s['behaviour']}")
    print(f"  memory now: {s['memory']['discoveries']} · {s['memory']['experiments']} experiments · "
          f"{s['memory']['mistakes']} mistakes · {s['memory']['goals_open']} open goals")


def cmd_serve(args):
    import uvicorn

    from .dashboard import create_app
    cfg = _cfg(args)
    host = args.host or cfg["server"]["host"]
    port = args.port or cfg["server"]["port"]
    print(f"Dashboard: http://{host}:{port}/   API docs: http://{host}:{port}/docs")
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")


def cmd_explore(args):
    from .session import run_session
    cfg = _cfg(args)
    db = _db(cfg)
    for _ in range(args.sessions):
        s = run_session(cfg, db, args.agent, _adapter_factory(cfg, args)(args.agent), label=args.label,
                        resume_session_id=args.resume)
        _print_summary(s)
        args.resume = None
        if s["end_reason"] in ("adapter_error", "runner_error", "interrupted"):
            print(f"  session {s['session_id']} can be resumed: playground explore --agent {args.agent} "
                  f"--resume {s['session_id']}")
            return 1
    return 0


def cmd_run_agents(args):
    from .session import run_agents
    cfg = _cfg(args)
    db = _db(cfg)
    ids = args.agents.split(",") if args.agents else agent_ids(cfg)
    print(f"Running {len(ids)} agents x {args.sessions} session(s), concurrency {args.concurrency}, "
          f"adapter {args.adapter}")
    res = run_agents(cfg, db, _adapter_factory(cfg, args), ids, args.sessions, args.concurrency, args.label,
                     progress=_print_summary)
    print(json.dumps({k: res[k] for k in ("sessions_run", "completed", "wall_seconds", "errors")}, indent=2))
    return 0 if not res["errors"] else 1


def cmd_memory(args):
    from .memory import MemoryStore
    from .session import memory_root
    cfg = _cfg(args)
    db = _db(cfg)
    m = MemoryStore(memory_root(cfg), args.agent, cfg, db=db)
    if args.op == "show":
        print(json.dumps(m.stats(), indent=2))
        for i in m.all_items():
            print(f"  {i['id']:12s} {i['kind']:10s} {str(i['status'] or ''):12s} {(i['text'] or '')[:90]}")
        print(f"\nFiles: {m.dir}")
    elif args.op == "file":
        print((m.dir / f"{args.item}.md").read_text(encoding="utf-8"))
    elif args.op == "remove":
        print(json.dumps(m.remove_item(args.item, args.reason), indent=2, default=str))
    elif args.op == "challenge":
        print(json.dumps(m.challenge(args.item, args.reason or "challenged by researcher"), indent=2))
    elif args.op == "unverify":
        print(json.dumps(m.mark_unverified(args.item, args.reason), indent=2))
    return 0


def cmd_benchmark(args):
    from .session.benchmark import run_benchmark
    cfg = _cfg(args)
    db = _db(cfg)
    a = args.group_a.split(",")
    b = args.group_b.split(",") if args.group_b else None
    res = run_benchmark(cfg, db, _adapter_factory(cfg, args), a, b,
                        args.tasks.split(",") if args.tasks else None, args.label, args.repeats,
                        progress=lambda r: print(f"  {r['group']} {r['agent_id']:12s} {r['task_id']:22s} "
                                                 f"completed={r['completed']!s:5} score={r['score']} "
                                                 f"actions={r['actions']} failed={r['failed_actions']}"))
    print(json.dumps({"run_id": res["run_id"], "overall": res["summary"]["overall"],
                      "differences": res["summary"].get("differences")}, indent=2))
    return 0


def cmd_ablate(args):
    from .session.benchmark import run_ablation
    cfg = _cfg(args)
    db = _db(cfg)
    res = run_ablation(cfg, db, _adapter_factory(cfg, args), args.agent, args.remove.split(","),
                       args.tasks.split(",") if args.tasks else None, args.label, args.repeats,
                       progress=lambda r: print(f"  {r['condition']:12s} {r['task_id']:22s} completed={r['completed']!s:5}"
                                                f" score={r['score']} actions={r['actions']}"))
    print(json.dumps({"run_id": res["run_id"], "removed": res["summary"]["removed_items"],
                      "overall": res["summary"]["overall"], "differences": res["summary"].get("differences")},
                     indent=2, default=str))
    return 0


def cmd_verify_log(args):
    cfg = _cfg(args)
    db = _db(cfg)
    r = db.verify_chain(args.session)
    print(json.dumps(r, indent=2))
    return 0 if r["ok"] else 2


def cmd_demo(args):
    from .demo import run_demo
    cfg = _cfg(args)
    return run_demo(cfg, args.out, adapter=args.adapter, model=args.model)


def cmd_init(args):
    from .session import ensure_agents
    cfg = _cfg(args)
    db = _db(cfg)
    ids = ensure_agents(cfg, db)
    print(f"Database: {db.path}\nAgents ({len(ids)}): {', '.join(ids)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="playground", description="AI Experimental Playground")
    p.add_argument("--config", default=None, help="extra YAML merged over config/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    def adapter_args(sp):
        sp.add_argument("--adapter", default="scripted", help="name from the adapters section of the config")
        sp.add_argument("--model", default=None, help="override the adapter's model")
        sp.add_argument("--minutes", type=float, default=None, help="session length (default from config: 60)")
        sp.add_argument("--max-steps", type=int, default=None, help="cap on agent tool calls per session")
        sp.add_argument("--label", default=None)

    s = sub.add_parser("init", help="create the database and the 13 agent memory directories")
    s.set_defaults(fn=cmd_init)
    s = sub.add_parser("serve", help="dashboard and HTTP API")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("explore", help="free exploration sessions for one agent")
    s.add_argument("--agent", default="agent_001")
    s.add_argument("--sessions", type=int, default=1)
    s.add_argument("--resume", type=int, default=None, help="resume an interrupted session id")
    adapter_args(s)
    s.set_defaults(fn=cmd_explore)
    s = sub.add_parser("run-agents", help="exploration sessions for many agents")
    s.add_argument("--agents", default=None, help="comma-separated ids (default: all configured)")
    s.add_argument("--sessions", type=int, default=1)
    s.add_argument("--concurrency", type=int, default=1)
    adapter_args(s)
    s.set_defaults(fn=cmd_run_agents)
    s = sub.add_parser("memory", help="inspect or edit an agent's memory")
    s.add_argument("op", choices=["show", "file", "remove", "challenge", "unverify"])
    s.add_argument("agent")
    s.add_argument("item", nargs="?", help="item id (or file name for 'file': handbook, diary, mistakes, "
                                           "unfinished_goals)")
    s.add_argument("--reason", default=None)
    s.set_defaults(fn=cmd_memory)
    s = sub.add_parser("benchmark", help="Group A (experienced) vs Group B (fresh)")
    s.add_argument("--group-a", required=True)
    s.add_argument("--group-b", default=None)
    s.add_argument("--tasks", default=None)
    s.add_argument("--repeats", type=int, default=1)
    adapter_args(s)
    s.set_defaults(fn=cmd_benchmark)
    s = sub.add_parser("ablate", help="full memory vs memory with items removed")
    s.add_argument("agent")
    s.add_argument("--remove", required=True, help="comma-separated memory item ids")
    s.add_argument("--tasks", default=None)
    s.add_argument("--repeats", type=int, default=1)
    adapter_args(s)
    s.set_defaults(fn=cmd_ablate)
    s = sub.add_parser("verify-log", help="check a session's event hash chain")
    s.add_argument("session", type=int)
    s.set_defaults(fn=cmd_verify_log)
    s = sub.add_parser("demo", help="end-to-end demonstration in a separate workspace")
    s.add_argument("--out", default="experiments/demo")
    s.add_argument("--adapter", default="scripted")
    s.add_argument("--model", default=None)
    s.set_defaults(fn=cmd_demo)
    args = p.parse_args(argv)
    t = time.time()
    rc = args.fn(args) or 0
    if args.cmd not in ("serve",):
        print(f"({time.time() - t:.1f}s)", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
