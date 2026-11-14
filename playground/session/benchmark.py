"""Controlled benchmarking.

Group A: agents with memory from free exploration. Group B: fresh agents with an
empty memory. Both get identical tasks on identically prepared islands. Results
are measured from the final world and the event log by the task's own verifier;
nothing the agent says about its result is used.

Benchmarks never modify an agent's real memory: each run works on a copy, which
for ablation runs has specific items removed.
"""
from __future__ import annotations

import copy
import json
import shutil
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..agents.adapters import Adapter
from ..integrity.claims import find_structures, find_towers, verify_claim
from ..memory import MemoryStore
from ..world.physics import PhysicsWorld
from ..world.mathutil import quat_to_mat
from .runner import memory_root, run_session


@dataclass
class Task:
    id: str
    title: str
    prompt: str
    verify: Callable[[PhysicsWorld, list[dict]], dict]
    scene: Callable[[PhysicsWorld], None] | None = None
    max_steps: int = 80
    minutes: float = 30.0
    relevant_categories: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)


# ------------------------------------------------------------------ verifiers
def _blocks(world) -> list[str]:
    return [o for o in world.active if world.specs[o].kind == "block"]


def v_five_towers(world, events) -> dict:
    towers = [t for t in find_towers(world) if t["height_blocks"] >= 5]
    return {"verified": len(towers) >= 1, "score": len(towers), "accuracy": None,
            "measured": {"five_block_towers": len(towers), "towers": towers},
            "criteria": "columns of 5+ blocks at rest; score = number of such towers"}


def v_spire(world, events) -> dict:
    on = [o for o in _blocks(world) if "spire" in world.support_graph().get(o, [])]
    climbed = any(e["type"] == "climb" and e["data"].get("climbed_onto") == "spire" for e in events)
    stood = any(e["type"] == "agent_landing" and "spire" in (e["data"].get("landed_on") or []) for e in events)
    ok = bool(on) and not climbed and not stood
    return {"verified": ok, "score": len(on), "accuracy": None,
            "measured": {"blocks_on_spire": on, "agent_climbed_spire": climbed, "agent_stood_on_spire": stood},
            "criteria": "a block at rest on the spire's top; the body never on the spire"}



def _agent_support_paths_to_spire(world) -> list[list[str]]:
    """Find physical support chains from the agent to the Spire."""
    if not world.grounded:
        return []

    graph = world.support_graph()
    paths = []

    def walk(oid: str, path: list[str]) -> None:
        if oid == "spire":
            paths.append(path)
            return
        for lower in graph.get(oid, []):
            if lower not in path:
                walk(lower, path + [lower])

    for oid in sorted(world.support):
        walk(oid, [oid])

    return paths


def v_spire_climbing(world, events) -> dict:
    """Success requires the agent to be physically standing on the Spire."""
    paths = _agent_support_paths_to_spire(world)
    state = world.agent_state()
    spire = world.specs.get("spire")
    properties = getattr(spire, "properties", {}) if spire else {}
    top = float(properties.get("top_height", 6.0))
    rest_speed = float(world.rc["rest_speed"])
    height_ok = float(state["feet_height"]) >= top - 0.15
    stationary = float(state["speed"]) <= rest_speed

    verified = bool(paths) and height_ok and stationary and bool(state["on_ground"])

    return {
        "verified": verified,
        "score": 1 if verified else 0,
        "accuracy": None,
        "measured": {
            "position": state["position"],
            "feet_height": state["feet_height"],
            "spire_top_height": top,
            "grounded": state["on_ground"],
            "speed": state["speed"],
            "standing_on": state["standing_on"],
            "support_paths": paths,
            "direct_support": any(len(path) == 1 for path in paths),
            "indirect_support": any(len(path) > 1 for path in paths),
            "height_requirement_met": height_ok,
            "stationary": stationary,
        },
        "criteria": "the agent is stationary and grounded on the Spire, directly or through a physically supported structure",
    }


def v_generate(world, events) -> dict:
    appeared = [e["data"]["object"] for e in events if e["type"] == "object_appeared"]
    n = len(_blocks(world))
    return {"verified": bool(appeared), "score": len(appeared), "accuracy": None,
            "measured": {"blocks_in_world": n, "blocks_generated": appeared},
            "criteria": "at least one block that was not in the world at the start appeared"}


def make_v_height(target: float, tol: float = 0.25) -> Callable:
    def v(world, events) -> dict:
        peaks = [e["data"]["max_height"] for e in events if e["type"] == "flight_end"
                 and e["data"].get("kind") == "throw" and e["data"].get("max_height") is not None]
        errs = [abs(p - target) for p in peaks]
        best = min(errs) if errs else None
        return {"verified": best is not None and best <= tol, "score": None if best is None else round(best, 4),
                "accuracy": None if best is None else round(best, 4),
                "measured": {"target": target, "tolerance": tol, "peaks": peaks, "best_error": best,
                             "throws": len(peaks)},
                "criteria": f"a throw whose recorded peak is within {tol} m of {target} m"}
    return v


def make_v_object_at(target_xy, tol: float = 0.6, obj: str = "block_01") -> Callable:
    """Success = `obj` comes to rest within tol m of a target point, moved by CONTACT (push/pull/
    slide/roll/kick), not thrown. Rewards the new contact actions instead of throwing."""
    tx, ty = float(target_xy[0]), float(target_xy[1])

    def v(world, events) -> dict:
        thrown = any(e["type"] == "throw_release" and e["data"].get("object") == obj for e in events)
        by_contact = any(e["type"] == "object_impulse" and e["data"].get("object") == obj for e in events)
        if obj in world.specs and obj in getattr(world, "active", []):
            p = world.obj_pos(obj)
            dist = float(((p[0] - tx) ** 2 + (p[1] - ty) ** 2) ** 0.5)
        else:
            dist = None
        ok = dist is not None and dist <= tol and by_contact and not thrown
        return {"verified": ok, "score": None if dist is None else round(dist, 4),
                "accuracy": None if dist is None else round(dist, 4),
                "measured": {"object": obj, "target": [tx, ty], "final_distance": dist, "tolerance": tol,
                             "moved_by_contact": by_contact, "was_thrown": thrown},
                "criteria": f"'{obj}' at rest within {tol} m of ({tx}, {ty}), moved by contact not throwing"}
    return v


def make_v_build_height(target: float, tol: float = 0.3) -> Callable:
    """Success = a resting multi-object structure whose top is within tol of target m. Rewards
    stacking/building to a controlled height, not a single tall throw."""
    def v(world, events) -> dict:
        structs = find_structures(world)
        tops = [s["top_height"] for s in structs if len(s["objects"]) >= 2]
        best = max(tops) if tops else None
        err = None if best is None else abs(best - target)
        return {"verified": err is not None and err <= tol, "score": None if best is None else round(best, 3),
                "accuracy": None if err is None else round(err, 4),
                "measured": {"target": target, "tolerance": tol, "tallest_structure_top": best,
                             "structures": structs},
                "criteria": f"a resting structure of 2+ objects whose top is within {tol} m of {target} m"}
    return v


def v_topple(world, events) -> dict:
    """Success = at least one block that started upright is now tipped on its side, moved by contact."""
    tipped = []
    for oid in sorted(getattr(world, "active", [])):
        s = world.specs.get(oid)
        if s is None or s.kind != "block":
            continue
        up = quat_to_mat(world.obj_quat(oid))[:, 2]
        if abs(float(up[2])) < 0.8 and world.obj_speed(oid) < 0.1:
            tipped.append(oid)
    by_contact = any(e["type"] == "object_impulse" for e in events)
    ok = bool(tipped) and by_contact
    return {"verified": ok, "score": len(tipped), "accuracy": None,
            "measured": {"toppled_blocks": tipped, "moved_by_contact": by_contact},
            "criteria": "at least one block knocked onto its side by contact (not thrown)"}


def v_limited(world, events) -> dict:
    structs = find_structures(world)
    tops = [s["top_height"] for s in structs if any(world.specs[o].kind == "block" for o in s["objects"])]
    best = max(tops) if tops else 0.0
    return {"verified": best >= 1.99, "score": round(best, 3), "accuracy": None,
            "measured": {"tallest_structure_top": best, "structures": structs,
                         "blocks_in_world": len(_blocks(world))},
            "criteria": "tallest resting multi-object structure; success = top at 2.0 m or more"}


def v_complete(world, events) -> dict:
    base = ["block_04", "block_05", "block_06"]
    towers = [t for t in find_towers(world) if t["height_blocks"] >= 5 and t["blocks"][:3] == base]
    return {"verified": bool(towers), "score": max([t["height_blocks"] for t in find_towers(world)
                                                    if t["blocks"][:3] == base] or [0]), "accuracy": None,
            "measured": {"completed_towers": towers, "all_towers": find_towers(world)},
            "criteria": "the unfinished column (block_04, block_05, block_06 at its base) extended to 5+ blocks"}


# ------------------------------------------------------------------ scenes
def scene_limited(world) -> None:
    keep = {"block_01", "block_02", "block_03", "block_04"}
    for oid in sorted(world.active):
        if world.specs[oid].kind == "block" and oid not in keep:
            world.setup_remove(oid)


def scene_unfinished(world) -> None:
    for i, oid in enumerate(["block_04", "block_05", "block_06"]):
        world.setup_place(oid, [-5.0, 1.5, 0.25 + 0.5 * i + 0.002 * i])


TARGET_HEIGHT = 6.5


def default_tasks() -> dict[str, Task]:
    t = [
        Task("five_towers", "Build five-block towers",
             "Build as many towers as you can, each a single column of at least five blocks standing on the "
             "ground. Task reference: five_towers", v_five_towers, max_steps=140,
             relevant_categories=("ring_generation",)),
        Task("spire_landing", "Land a block on the spire",
             "Make a block come to rest on top of THE SPIRE without climbing onto the spire yourself. "
             "Task reference: spire_landing", v_spire, max_steps=60, relevant_categories=("jump_throw",)),
        Task("spire_climbing", "Reach the top of the Spire",
             "Find a way to get yourself standing on top of the Spire.",
             v_spire_climbing, max_steps=80, minutes=30.0),
        Task("generate_blocks", "Generate additional blocks",
             "Find a way to make an additional block exist on the island, beyond the ones there now. "
             "Task reference: generate_blocks", v_generate, max_steps=60, relevant_categories=("ring_generation",)),
        Task("throw_height", "Throw to a specified height",
             f"Throw a block so that the highest point of its flight is as close as possible to "
             f"{TARGET_HEIGHT} m above the ground (within 0.25 m counts). Target height: {TARGET_HEIGHT}. "
             "Task reference: throw_height", make_v_height(TARGET_HEIGHT), max_steps=60,
             relevant_categories=("jump_throw",)),
        Task("push_to_target", "Move a block to a marked spot without throwing",
             "Get block_01 to come to rest on the marked spot at x=4, y=0 WITHOUT throwing it "
             "(within 0.6 m counts). Task reference: push_to_target",
             make_v_object_at([4.0, 0.0], tol=0.6, obj="block_01"), max_steps=70),
        Task("roll_to_target", "Roll the rock to a marked spot",
             "Get the rock to come to rest on the marked spot at x=0, y=5 WITHOUT throwing it "
             "(within 0.8 m counts). Task reference: roll_to_target",
             make_v_object_at([0.0, 5.0], tol=0.8, obj="rock"), max_steps=70),
        Task("build_to_height", "Build to a specified height",
             "Build a resting structure of two or more objects whose top is as close as possible to "
             "1.5 m (within 0.3 m counts). Task reference: build_to_height",
             make_v_build_height(1.5, tol=0.3), max_steps=90),
        Task("topple", "Knock a block onto its side",
             "Knock one of the blocks over so it comes to rest on its side, without throwing it. "
             "Task reference: topple", v_topple, max_steps=60),
        Task("limited_structure", "Build with limited resources",
             "Only four blocks are on the island this time. Build the tallest structure you can. "
             "Task reference: limited_structure", v_limited, scene=scene_limited, max_steps=80,
             relevant_categories=("ring_generation",)),
        Task("complete_construction", "Complete an unfinished construction",
             "Someone started a column of blocks on the island and left it unfinished. Complete it so that it "
             "is at least five blocks tall. Task reference: complete_construction", v_complete,
             scene=scene_unfinished, max_steps=80),
    ]
    return {x.id: x for x in t}


TASKS = default_tasks()


# ------------------------------------------------------------------ metrics
def _metrics(db, sid: int, summary: dict, task: Task, memory: MemoryStore, briefed: bool,
             relevant_at_start: list[str]) -> dict:
    acts = db.actions(sid)
    events = db.events(sid, limit=10 ** 9)
    agent_acts = [a for a in acts if a["source"] in ("agent", "program")]
    objects_used = sorted({a["args"].get("object_id") for a in agent_acts
                           if a["tool"] in ("pick_up", "throw", "place") and a["args"].get("object_id")})
    tr = summary.get("task_result") or {}
    first_success_t = None
    if tr.get("verified"):
        first_success_t = summary["sim_time"]
    return {
        "completed": bool(tr.get("verified")), "score": tr.get("score"), "accuracy": tr.get("accuracy"),
        "sim_time": summary["sim_time"], "wall_seconds": summary["wall_seconds"],
        "time_to_completion_sim": first_success_t,
        "attempts": sum(1 for a in agent_acts if a["tool"] in ("throw", "place", "put_down", "climb", "jump")),
        "actions": len(agent_acts),
        "failed_actions": sum(1 for a in agent_acts if not a["success"]),
        "resources": {"objects_handled": objects_used, "objects_handled_count": len(objects_used),
                      "objects_lost": sum(1 for e in events if e["type"] == "object_lost"),
                      "blocks_generated": sum(1 for e in events if e["type"] == "object_appeared")},
        "route_metrics": {
            "ring_related_actions": sum(
                1 for a in agent_acts
                if "ring" in json.dumps(a.get("args", {})).lower()
            ),
            "ring_generated_objects": sum(
                1 for e in events
                if e["type"] == "object_appeared"
                and (e.get("data", {}).get("hidden") or {}).get("cause") == "ring_generation"
            ),
            "climb_attempts": sum(1 for a in agent_acts if a["tool"] == "climb"),
            "climb_events": sum(1 for e in events if e["type"] == "climb"),
            "jump_attempts": sum(1 for a in agent_acts if a["tool"] == "jump"),
            "placement_actions": sum(
                1 for a in agent_acts if a["tool"] in ("place", "put_down")
            ),
            "successful_placement_actions": sum(
                1 for a in agent_acts
                if a["tool"] in ("place", "put_down") and a["success"]
            ),
            "agent_landing_events": sum(
                1 for e in events if e["type"] == "agent_landing"
            ),
            "experimentation_actions": sum(
                1 for a in agent_acts
                if a["tool"] in ("record_experiment", "run_physics_test")
            ),
        },
        "experiments": sum(1 for a in acts if a["tool"] == "record_experiment") +
                       sum(1 for a in acts if a["tool"] == "run_physics_test"),
        "memory_use": {"briefed_with_memory": briefed and bool(relevant_at_start or summary["counters"]["memory_reads"]),
                       "relevant_handbook_entries_at_start": relevant_at_start,
                       "memory_reads": summary["counters"]["memory_reads"],
                       "prior_programs_run": summary["counters"]["prior_programs_run"]},
        "end_reason": summary["end_reason"], "error": summary["error"],
    }


# ------------------------------------------------------------------ runner
def run_task(cfg: dict, db, task: Task, agent_id: str, adapter: Adapter, memory: MemoryStore,
             run_id: int, group: str, condition: str, briefed: bool = True) -> dict:
    c = copy.deepcopy(cfg)
    c["session"]["max_steps"] = task.max_steps
    c["session"]["duration_minutes"] = task.minutes
    relevant_at_start = [d["id"] for d in memory.discoveries["items"]
                         if d.get("category") in task.relevant_categories and d["status"] in ("confirmed", "probable")]
    memory_was_empty = memory.is_empty()

    def verify(env) -> dict:
        evs = db.events(env.sid, limit=10 ** 9)
        res = task.verify(env.world, evs)
        res["task_id"] = task.id
        return res

    summary = run_session(c, db, agent_id, adapter, mode="benchmark", memory=memory, task_prompt=task.prompt,
                          scene=task.scene, include_briefing=briefed, on_finish=verify,
                          label=f"bench:{run_id}:{group}:{condition}:{task.id}",
                          extra={"benchmark_run_id": run_id, "task_id": task.id, "condition": condition})
    metrics = _metrics(db, summary["session_id"], summary, task, memory, briefed, relevant_at_start)
    metrics["memory_use"]["memory_empty_at_start"] = memory_was_empty
    db.x("INSERT INTO benchmark_results(run_id,session_id,agent_id,grp,condition,task_id,metrics_json,"
         "verified_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
         (run_id, summary["session_id"], agent_id, group, condition, task.id, json.dumps(metrics, default=str),
          json.dumps(summary.get("task_result"), default=str), time.time()))
    return {"agent_id": agent_id, "group": group, "condition": condition, "task_id": task.id,
            "session_id": summary["session_id"], **metrics}


def _bench_root(cfg: dict, run_id: int) -> Path:
    return memory_root(cfg).parent / "benchmark_memory" / f"run_{run_id:04d}"


def run_benchmark(cfg: dict, db, adapter_factory: Callable[[str], Adapter], group_a: list[str],
                  group_b: list[str] | None = None, tasks: list[str] | None = None, label: str | None = None,
                  repeats: int = 1, progress: Callable[[dict], None] | None = None) -> dict:
    """Group A (experienced, memory copied) vs Group B (fresh, empty memory) on identical tasks."""
    task_ids = tasks or list(TASKS)
    unknown = [t for t in task_ids if t not in TASKS]
    if unknown:
        raise ValueError(f"unknown task(s) {unknown}; available: {sorted(TASKS)}")
    group_b = group_b if group_b is not None else [f"fresh_{i + 1:03d}" for i in range(len(group_a))]
    run_id = db.x("INSERT INTO benchmark_runs(label,kind,created_at,params_json) VALUES(?,?,?,?)",
                  (label, "groups", time.time(), json.dumps({"group_a": group_a, "group_b": group_b,
                                                             "tasks": task_ids, "repeats": repeats})))
    root = _bench_root(cfg, run_id)
    real = memory_root(cfg)
    rows = []
    for rep in range(repeats):
        for tid in task_ids:
            task = TASKS[tid]
            for agent in group_a:
                src = MemoryStore(real, agent, cfg)
                mem = src.copy_to(root / f"A_r{rep}_{tid}", reason="benchmark copy")
                r = run_task(cfg, db, task, agent, adapter_factory(agent), mem, run_id, "A", "experienced")
                rows.append(r)
                if progress:
                    progress(r)
            for agent in group_b:
                d = root / f"B_r{rep}_{tid}" / agent
                if d.exists():
                    shutil.rmtree(d)
                mem = MemoryStore(root / f"B_r{rep}_{tid}", agent, cfg)
                r = run_task(cfg, db, task, agent, adapter_factory(agent), mem, run_id, "B", "fresh")
                rows.append(r)
                if progress:
                    progress(r)
    summary = compare(rows, ("group",))
    db.x("UPDATE benchmark_runs SET summary_json=? WHERE id=?", (json.dumps(summary, default=str), run_id))
    return {"run_id": run_id, "results": rows, "summary": summary}


def run_ablation(cfg: dict, db, adapter_factory: Callable[[str], Adapter], agent_id: str, remove_ids: list[str],
                 tasks: list[str] | None = None, label: str | None = None, repeats: int = 1,
                 progress: Callable[[dict], None] | None = None) -> dict:
    """The same experienced agent with its full memory vs with specific items removed."""
    task_ids = tasks or list(TASKS)
    real = MemoryStore(memory_root(cfg), agent_id, cfg)
    missing = [i for i in remove_ids if i not in {x["id"] for x in real.all_items()}]
    if missing:
        raise ValueError(f"{agent_id} has no memory item(s) {missing}")
    removed = [x for x in real.all_items() if x["id"] in remove_ids]
    run_id = db.x("INSERT INTO benchmark_runs(label,kind,created_at,params_json) VALUES(?,?,?,?)",
                  (label, "ablation", time.time(), json.dumps({"agent_id": agent_id, "removed": remove_ids,
                                                               "removed_items": removed, "tasks": task_ids,
                                                               "repeats": repeats})))
    root = _bench_root(cfg, run_id)
    rows = []
    for rep in range(repeats):
        for tid in task_ids:
            task = TASKS[tid]
            for cond, rm in (("full_memory", []), ("ablated", remove_ids)):
                mem = real.copy_to(root / f"{cond}_r{rep}_{tid}", remove_ids=rm, reason=f"ablation run {run_id}")
                r = run_task(cfg, db, task, agent_id, adapter_factory(agent_id), mem, run_id, "A", cond)
                rows.append(r)
                if progress:
                    progress(r)
    summary = compare(rows, ("condition",))
    summary["removed_items"] = removed
    db.x("UPDATE benchmark_runs SET summary_json=? WHERE id=?", (json.dumps(summary, default=str), run_id))
    return {"run_id": run_id, "results": rows, "summary": summary}


def run_controller_benchmark(cfg: dict, db, adapter: Adapter, task_id: str = "throw_height",
                             label: str | None = None, agent_id: str = "neural_controller") -> dict:
    """Run a single benchmark task with any adapter (used for a neural controller) through the REAL
    run_task pipeline, so the task's own independent verifier judges the final world — never the agent.

    Returns the verified result. The neural controller only has a policy for throw_height; other tasks
    make it observe and end, which the verifier will simply score as not completed. Added 2026-11-13
    (stage: NN lab) to replace the broken CLI 'controller' path that called run_task with the wrong
    signature. The benchmark uses a throwaway empty memory under benchmark_memory/, so no agent's real
    memory is touched.
    """
    if task_id not in TASKS:
        raise ValueError(f"unknown task '{task_id}'; available: {sorted(TASKS)}")
    task = TASKS[task_id]
    run_id = db.x("INSERT INTO benchmark_runs(label,kind,created_at,params_json) VALUES(?,?,?,?)",
                  (label or f"controller:{adapter.model}", "controller", time.time(),
                   json.dumps({"task": task_id, "adapter": adapter.describe()})))
    root = _bench_root(cfg, run_id)
    mem = MemoryStore(root / f"controller_{task_id}", agent_id, cfg)
    r = run_task(cfg, db, task, agent_id, adapter, mem, run_id, "controller", "neural")
    db.x("UPDATE benchmark_runs SET summary_json=? WHERE id=?",
         (json.dumps({"controller": r}, default=str), run_id))
    return {"run_id": run_id, "task_id": task_id, "verified": bool(r.get("completed")),
            "completed": r.get("completed"), "score": r.get("score"),
            "accuracy": r.get("accuracy"), "actions": r.get("actions"),
            "failed_actions": r.get("failed_actions"), "session_id": r.get("session_id"),
            "adapter": adapter.describe()}


def compare(rows: list[dict], by: tuple[str, ...]) -> dict:
    """Aggregate per task and per group/condition. Reports differences in both directions."""
    out: dict[str, Any] = {"by_task": {}, "overall": {}}

    def agg(rs: list[dict]) -> dict:
        def mean(key, sub=None):
            vals = [(r[key] if sub is None else r[key][sub]) for r in rs]
            vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
            return round(statistics.mean(vals), 3) if vals else None
        return {"runs": len(rs), "completion_rate": round(sum(r["completed"] for r in rs) / len(rs), 3) if rs else None,
                "mean_score": mean("score"), "mean_accuracy_error": mean("accuracy"),
                "mean_sim_time": mean("sim_time"), "mean_actions": mean("actions"),
                "mean_attempts": mean("attempts"), "mean_failed_actions": mean("failed_actions"),
                "mean_experiments": mean("experiments"),
                "mean_objects_handled": round(statistics.mean([r["resources"]["objects_handled_count"] for r in rs]), 3)
                if rs else None,
                "used_memory": sum(1 for r in rs if r["memory_use"]["memory_reads"] or r["memory_use"]["prior_programs_run"])}

    key = by[0]
    groups = sorted({r[key] for r in rows})
    for tid in sorted({r["task_id"] for r in rows}):
        out["by_task"][tid] = {g: agg([r for r in rows if r["task_id"] == tid and r[key] == g]) for g in groups}
    out["overall"] = {g: agg([r for r in rows if r[key] == g]) for g in groups}
    if len(groups) == 2:
        a, b = groups
        diffs = {}
        for tid, per in out["by_task"].items():
            ca, cb = per[a]["completion_rate"], per[b]["completion_rate"]
            diffs[tid] = {"completion_rate_difference": None if ca is None or cb is None else round(ca - cb, 3),
                          "better": None if ca is None or cb is None or ca == cb else (a if ca > cb else b)}
        out["differences"] = {"compared": f"{a} minus {b}", "per_task": diffs,
                              "note": "Small samples. A difference here is an observation, not a significance test."}
    return out
