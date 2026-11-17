"""Command-line interface.

    playground serve                         dashboard + API
    playground explore --agent agent_001     one exploration session
    playground run-agents --sessions 3       all 13 agents
    playground memory show agent_001         inspect memory
    playground benchmark --group-a agent_001 Group A vs Group B
    playground ablate agent_001 --remove D-002
    playground arena match.json              duel/coop match via the production factories
    playground demo                          the end-to-end demonstration
    playground verify-log 12                 check a session's hash chain
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from .arena.factories import cmd_arena

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


def _db_reconcile(cfg):
    """Open the database and reconcile orphaned 'running' sessions left by dead processes.
    Must run BEFORE this process starts any session of its own."""
    db = _db(cfg)
    fixed = db.reconcile_orphans()
    if fixed:
        print(f"reconciled {len(fixed)} orphaned session(s) left 'running' by a dead process: "
              f"{', '.join('#' + str(s) for s in fixed)} (marked interrupted; resumable with --resume)")
    return db


def _adapter_factory(cfg, args):
    from .agents import make_adapter
    return lambda agent_id: make_adapter(cfg, args.adapter or "scripted", model=args.model)


def _make_adapter(cfg, adapter_name, model):
    from .agents import make_adapter
    return make_adapter(cfg, adapter_name, model=model)


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
    db = _db_reconcile(cfg)  # orphaned external sessions from a dead server become 'interrupted'
    host = args.host or cfg["server"]["host"]
    port = args.port or cfg["server"]["port"]
    print(f"Dashboard: http://{host}:{port}/   API docs: http://{host}:{port}/docs")
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")


def cmd_explore(args):
    from .session import run_session
    from .session.runner import resolve_resume_adapter
    cfg = _cfg(args)
    db = _db_reconcile(cfg)
    # On --resume, default to the adapter/model the session was started with (not the CLI default,
    # which is 'scripted' — the reference policy, not an AI). Audit 2026-11-13.
    adapter_name, model = args.adapter, args.model
    if args.resume is not None:
        adapter_name, model = resolve_resume_adapter(db, args.resume, args.adapter, args.model)
    for _ in range(args.sessions):
        factory = lambda agent_id: _make_adapter(cfg, adapter_name or "scripted", model)
        s = run_session(cfg, db, args.agent, factory(args.agent), label=args.label,
                        resume_session_id=args.resume)
        _print_summary(s)
        args.resume = None
        adapter_name, model = args.adapter, args.model
        if s["end_reason"] in ("adapter_error", "runner_error", "interrupted"):
            print(f"  session {s['session_id']} can be resumed: playground explore --agent {args.agent} "
                  f"--resume {s['session_id']}")
            return 1
    return 0


def cmd_run_agents(args):
    from .session import run_agents
    cfg = _cfg(args)
    db = _db_reconcile(cfg)
    ids = args.agents.split(",") if args.agents else agent_ids(cfg)
    print(f"Running {len(ids)} agents x {args.sessions} session(s), concurrency {args.concurrency}, "
          f"adapter {args.adapter or 'scripted'}")
    res = run_agents(cfg, db, _adapter_factory(cfg, args), ids, args.sessions, args.concurrency, args.label,
                     progress=_print_summary, max_continuations=args.max_continuations)
    print(json.dumps({k: res[k] for k in ("sessions_run", "completed", "continuations_granted",
                                          "wall_seconds", "errors")}, indent=2))
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


def cmd_messages(args):
    """Dump a session's whole transcript: every message and every action, in turn order.
    With --out, writes a UTF-8 JSONL file (messages and actions annotated by kind); otherwise
    prints a human-readable transcript. The database remains the authority; this is a read-only view."""
    import io
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    cfg = _cfg(args)
    db = _db(cfg)
    sid = args.session_id
    sess = db.q("SELECT * FROM sessions WHERE id=?", (sid,))
    if not sess:
        print(f"no session {sid}")
        return 1
    sess = sess[0]
    messages = db.q("SELECT turn, role, content FROM messages WHERE session_id=? ORDER BY turn, id", (sid,))
    actions = db.q("SELECT idx, turn, tool, category, args_json, success, error_code, result_json "
                   "FROM actions WHERE session_id=? ORDER BY idx", (sid,))

    def record(kind: str, row: dict) -> dict:
        return {"kind": kind, "session_id": sid, **row}

    if args.out:
        path = args.out
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for r in messages:
                fh.write(json.dumps(record("message", dict(r)), ensure_ascii=False, default=str) + "\n")
            for r in actions:
                fh.write(json.dumps(record("action", dict(r)), ensure_ascii=False, default=str) + "\n")
        print(f"wrote {len(messages)} messages and {len(actions)} actions for session {sid} to {path}")
        return 0

    print(f"session {sid}: agent {sess['agent_id']}  adapter {sess['adapter']}  model {sess['model']}"
          f"  status {sess['status']}  end {sess['end_reason']}  actions {sess['action_count']}")
    ev_by_turn: dict = {}
    for r in actions:
        ev_by_turn.setdefault(r["turn"], []).append(r)
    for r in messages:
        head = f"--- turn {r['turn']} [{r['role']}]"
        print(head)
        print(r["content"])
        for a in ev_by_turn.get(r["turn"], []):
            ok = "ok" if a["success"] else f"FAIL {a['error_code']}"
            print(f"    action #{a['idx']} {a['tool']}  -> {ok}")
    msg_turns = {r["turn"] for r in messages}
    for turn in sorted(ev_by_turn):
        if turn not in msg_turns:  # actions on turns with no stored message (tool-only turns)
            for a in ev_by_turn[turn]:
                ok = "ok" if a["success"] else f"FAIL {a['error_code']}"
                print(f"--- turn {turn}")
                print(f"    action #{a['idx']} {a['tool']}  -> {ok}")
    return 0


def cmd_benchmark(args):
    from .session.benchmark import run_benchmark
    cfg = _cfg(args)
    db = _db_reconcile(cfg)
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
    db = _db_reconcile(cfg)
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


def cmd_models(args):
    """List the models an openai_compat adapter's server is offering (e.g. every model loaded in LM
    Studio), so the researcher can pick any of them instead of only the ones named in the config."""
    from .agents.adapters import list_openai_models
    cfg = _cfg(args)
    table = cfg.get("adapters", {})
    spec = table.get(args.adapter_name)
    if not spec or spec.get("type") != "openai_compat":
        print(f"'{args.adapter_name}' is not an openai_compat adapter. Available: "
              f"{sorted(a for a, s in table.items() if s.get('type') == 'openai_compat')}")
        return 1
    models = list_openai_models(spec["base_url"], spec.get("api_key_env"))
    if not models:
        print(f"No models returned by {spec['base_url']} (is the server running?).")
        return 1
    print(f"{len(models)} model(s) at {spec['base_url']}:")
    for m in models:
        ctx = m.get("loaded_context_length") or m.get("max_context_length")
        state = m.get("state", "")
        extra = f"  [{state}]" if state else ""
        ctx_s = f"  ctx={ctx}" if ctx else ""
        print(f"  {m.get('id')}{ctx_s}{extra}")
    print(f"\nUse any id with:  playground explore --adapter {args.adapter_name} --model '<id>'")
    return 0


def cmd_demo(args):
    from .demo import run_demo
    cfg = _cfg(args)
    return run_demo(cfg, args.out, adapter=args.adapter, model=args.model)


def cmd_compare_models(args):
    """Run several models under identical conditions (fresh agents, same action cap, same context window)
    and report a per-model behavioural profile. A faster model is not assumed to be a better explorer."""
    from .agents import make_adapter
    from .session.model_compare import compare_models
    cfg = _cfg(args)
    db = _db(cfg)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("Give --models as a comma-separated list of model ids (see: playground models).")
        return 1
    adapter_name = args.adapter or "lmstudio"

    def adapter_for(model, agent_id):
        return make_adapter(cfg, adapter_name, model=model)

    print(f"Comparing {len(models)} model(s) via adapter '{adapter_name}': {models}")
    print(f"Conditions: {args.agents} fresh agent(s)/model x {args.sessions} session(s), "
          f"max_steps={args.max_steps}, context_tokens={args.context_tokens}")
    res = compare_models(cfg, db, adapter_for, models, agents_per_model=args.agents,
                         sessions=args.sessions, context_tokens=args.context_tokens,
                         max_steps=args.max_steps,
                         progress=lambda p: print(f"  [{p['model']}] {p['agent_id']} s{p['session_id']}: "
                                                  f"{p['actions']} acts, {p['distinct_tools']} tools, "
                                                  f"obs={p['observation_share']}, goals={p['goals_set']}, "
                                                  f"exp={p['experiments']}, end={p['end_reason']}"))
    print("\n=== Per-model profile (equal conditions) ===")
    for model, agg in res["by_model"].items():
        print(f"\n{model}:")
        for k in ("sessions", "actions_mean", "distinct_tools_mean", "throw_share_mean",
                  "observation_share_mean", "physical_share_mean", "failed_share_mean", "goals_set_mean",
                  "experiments_mean", "confirmed_discoveries_mean", "budget_used_mean", "continued", "errored"):
            if k in agg:
                print(f"  {k} = {agg[k]}")
    import json as _json
    out = resolve_path(cfg, "experiments") / "model_comparison.json"
    out.write_text(_json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"\nFull profiles written to {out}")
    return 0


def cmd_datasets(args):
    """List the behaviour datasets the ML lab can build (not just throws)."""
    from .ml.datasets import available_datasets
    print("Behaviour datasets (build via dashboard, or an agent's nn_make_dataset kind=…):")
    print(f"  {'throw':13s} throw outcomes: force/angle -> height/range")
    for d in available_datasets():
        print(f"  {d['name']:13s} {d['description']}")
        print(f"  {'':13s} inputs {d['inputs']} · targets {d['targets']}")
    return 0


def cmd_ml(args):
    """Neural network subcommand handler."""
    cfg = _cfg(args)

    if args.op == "status":
        from .ml import torch_status
        print(json.dumps(torch_status(), indent=2))
        return 0

    if args.op == "generate":
        from .ml.data import generate_throw_dataset
        from .ml.registry import registry_for
        if not args.name:
            print("--name required", file=sys.stderr)
            return 1
        print(f"Generating {args.n} throws (seed={args.seed})...", file=sys.stderr)
        ds = generate_throw_dataset(cfg, args.n, seed=args.seed)
        reg = registry_for(cfg, args.scope, args.owner)
        info = reg.save_dataset(args.name, ds)
        print(json.dumps({**info, "scope": args.scope, "owner": args.owner}, indent=2))
        return 0

    if args.op == "train":
        from .ml.registry import registry_for
        from .ml.nets import train, validate_spec, validate_train
        if not args.dataset or not args.name:
            print("--dataset and --name required", file=sys.stderr)
            return 1
        reg = registry_for(cfg, args.scope, args.owner)
        ds = reg.load_dataset(args.dataset)
        inputs = args.inputs.split(",") if args.inputs else [
            "force", "dir_x", "dir_y", "dir_z", "elevation_deg",
            "agent_vx", "agent_vy", "agent_vz", "object_mass",
            "feet_height", "on_ground", "release_height"
        ]
        hidden = [int(h) for h in args.hidden.split(",")]
        spec = validate_spec({"name": args.name, "inputs": inputs, "target": args.target, "hidden": hidden})
        tc = validate_train({"epochs": args.epochs, "device": args.device})
        print(f"Training on {ds['manifest']['rows']} rows (owner={args.owner})...", file=sys.stderr)
        result = train(spec, ds["rows"], tc)
        info = reg.save_model(args.owner, result, ds)
        print(json.dumps({
            **info, "scope": args.scope, "owner": args.owner,
            "epochs_run": result["epochs_run"],
            "best_epoch": result["best_epoch"],
            "test_rmse": result["metrics"]["test"]["rmse"],
            "test_r2": result["metrics"]["test"]["r2"],
            "parameters": result["parameters"],
        }, indent=2))
        return 0

    if args.op == "list":
        from .ml.registry import all_models
        owner = args.owner if args.owner != "researcher" else None
        mroot = resolve_path(cfg, cfg["memory"]["root"])
        models = all_models(cfg, mroot if mroot.exists() else None)
        if owner:
            models = [m for m in models if m["owner"] == owner]
        for m in models:
            m.pop("state_dict", None)
        print(json.dumps(models, indent=2))
        return 0

    if args.op == "evaluate":
        from .ml.registry import registry_for
        from .ml.evaluate import fresh_engine_test, closed_loop_height
        if not args.name:
            print("--name required", file=sys.stderr)
            return 1
        reg = registry_for(cfg, args.scope, args.owner)
        model = reg.load_model(args.owner, args.name, args.version)
        result = {"owner": args.owner, "name": args.name, "version": model["version"]}
        print(f"Fresh engine test ({args.fresh} throws)...", file=sys.stderr)
        result["fresh"] = fresh_engine_test(cfg, model, args.fresh)
        if args.closed_loop:
            print("Closed-loop test...", file=sys.stderr)
            dsets = reg.list_datasets()
            linear_rows = reg.load_dataset(dsets[0])["rows"] if dsets else None
            result["closed_loop"] = closed_loop_height(cfg, model, linear_rows=linear_rows)
        print(json.dumps(result, indent=2))
        return 0

    if args.op == "controller":
        from .ml.registry import registry_for
        from .agents.adapters import make_neural_adapter
        from .session.benchmark import run_controller_benchmark
        from .storage import Database
        if not args.name:
            print("--name required", file=sys.stderr)
            return 1
        reg = registry_for(cfg, args.scope, args.owner)
        model = reg.load_model(args.owner, args.name, args.version)   # fail fast if missing
        adapter = make_neural_adapter(cfg, owner=args.owner, name=args.name, version=args.version)
        db = Database(resolve_path(cfg, cfg["database"]))
        task_id = getattr(args, "task", None) or "throw_height"
        print(f"Running {task_id} benchmark with neural controller "
              f"{args.owner}/{args.name}/v{model['version']}...", file=sys.stderr)
        result = run_controller_benchmark(cfg, db, adapter, task_id=task_id)
        print(json.dumps(result, indent=2))
        return 0

    print(f"Unknown operation: {args.op}", file=sys.stderr)
    return 1


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
        sp.add_argument("--adapter", default=None, help="name from the adapters section of the config "
                                                        "(default: scripted, or the resumed session's own adapter)")
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
    s.add_argument("--max-continuations", type=int, default=0,
                   help="extra sessions an agent may earn by asking to continue (default 0; keeps it bounded)")
    adapter_args(s)
    s.set_defaults(fn=cmd_run_agents)
    s = sub.add_parser("memory", help="inspect or edit an agent's memory")
    s.add_argument("op", choices=["show", "file", "remove", "challenge", "unverify"])
    s.add_argument("agent")
    s.add_argument("item", nargs="?", help="item id (or file name for 'file': handbook, diary, mistakes, "
                                           "unfinished_goals)")
    s.add_argument("--reason", default=None)
    s.set_defaults(fn=cmd_memory)
    s = sub.add_parser("messages", help="dump a session's full transcript (messages + actions)")
    s.add_argument("session_id", type=int)
    s.add_argument("--out", default=None, help="write a JSONL dump to this file instead of printing")
    s.set_defaults(fn=cmd_messages)
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
    s = sub.add_parser("models", help="list models an openai_compat server (e.g. LM Studio) is offering")
    s.add_argument("adapter_name", nargs="?", default="lmstudio",
                   help="which configured openai_compat adapter's server to query (default: lmstudio)")
    s.set_defaults(fn=cmd_models)
    s = sub.add_parser("datasets", help="list the behaviour datasets the ML lab can build (not just throws)")
    s.set_defaults(fn=cmd_datasets)
    s = sub.add_parser("compare-models", help="run several models under identical conditions and profile them")
    s.add_argument("--models", required=True, help="comma-separated model ids (see: playground models)")
    s.add_argument("--adapter", default=None, help="openai_compat adapter whose server hosts them (default: lmstudio)")
    s.add_argument("--agents", type=int, default=3, help="fresh agents per model (default 3)")
    s.add_argument("--sessions", type=int, default=1, help="sessions per agent (default 1)")
    s.add_argument("--max-steps", type=int, default=60, help="action cap per session, kept equal (default 60)")
    s.add_argument("--context-tokens", type=int, default=None,
                   help="pin the same context window for every model (default: each adapter's own)")
    s.set_defaults(fn=cmd_compare_models)
    s = sub.add_parser("demo", help="end-to-end demonstration in a separate workspace")
    s.add_argument("--out", default="experiments/demo")
    s.add_argument("--adapter", default="scripted")
    s.add_argument("--model", default=None)
    s.set_defaults(fn=cmd_demo)
    # ---- ml subcommand
    s = sub.add_parser("ml", help="neural network operations")
    s.add_argument("op", choices=["status", "generate", "train", "list", "evaluate", "controller"])
    s.add_argument("--name", help="dataset or model name")
    s.add_argument("--dataset", help="dataset to train on")
    s.add_argument("--n", type=int, default=1500, help="throws to generate (default 1500)")
    s.add_argument("--seed", type=int, default=7, help="random seed (default 7)")
    s.add_argument("--inputs", help="comma-separated input columns")
    s.add_argument("--target", default="max_height", help="target column (default max_height)")
    s.add_argument("--hidden", default="64,64", help="hidden layer sizes (default 64,64)")
    s.add_argument("--epochs", type=int, default=400, help="training epochs (default 400)")
    s.add_argument("--device", default="cpu", choices=["cpu", "xpu", "auto"],
                   help="device for training (default cpu)")
    s.add_argument("--owner", default="researcher", help="model owner: 'researcher' or an agent id (default researcher)")
    s.add_argument("--scope", default="researcher", choices=["researcher", "agent"],
                   help="registry scope: 'researcher' (shared) or 'agent' (per-agent, under memory root). "
                        "Use --scope agent --owner <agent_id> for a per-agent network (default researcher)")
    s.add_argument("--task", default="throw_height", help="benchmark task for 'controller' (default throw_height)")
    s.add_argument("--version", type=int, default=None, help="model version (default latest)")
    s.add_argument("--fresh", type=int, default=200, help="throws for fresh evaluation (default 200)")
    s.add_argument("--closed-loop", action="store_true", help="run closed-loop evaluation")
    s.set_defaults(fn=cmd_ml)

    # ---- arena subcommand
    s = sub.add_parser("arena", help="run an arena match from a JSON config file")
    s.add_argument("config", help="path to match configuration JSON file")
    s.add_argument("--out", default=None, help="output file for match report (default: same name with .report.json)")
    s.add_argument("--adapter", default="scripted", help="per-combatant adapter (scripted, or a live model name)")
    s.add_argument("--model", default=None, help="model override for the adapter")
    s.add_argument("--db", default=None, help="session database override (default: the configured one)")
    s.add_argument("--root", default=None, help="project root override (default: the configured one)")
    s.set_defaults(fn=cmd_arena)

    args = p.parse_args(argv)
    t = time.time()
    rc = args.fn(args) or 0
    if args.cmd not in ("serve",):
        print(f"({time.time() - t:.1f}s)", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
