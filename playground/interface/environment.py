"""The environment: one agent, one session, one island.

Everything an agent does goes through Environment.call(). Each call is
  * validated, executed and answered with a structured result (never a silent success),
  * written to the append-only action log together with the physics events it caused,
  * bound into the session's hash chain.

The same object is used by in-process adapters and by the HTTP API, so every kind
of agent sees exactly the same world through exactly the same interface.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from typing import Any, Callable

from ..config import canonical_json, config_hash
from ..integrity import CLAIM_HELP, VERIFIERS, check_claim_limits, check_flight, verify_claim, verify_experiment
from ..memory import MemoryError_, MemoryStore
from ..world.physics import ActionError, PhysicsWorld
from .physical import AGENT_EVENT_TYPES, PHYSICAL_TOOL_NAMES, PhysicalTools, visible_event
from .schema import SPECS, ToolError, normalise_args

PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
MAX_PROGRAM_STEPS = 40
MAX_PROGRAM_ACTIONS = 200
MAX_SANDBOX_STEPS = 12
WRAPUP_CALLS = 12


def _sha(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def _substitute(value: Any, params: dict) -> Any:
    if isinstance(value, str):
        m = PLACEHOLDER.fullmatch(value.strip())
        if m:
            if m.group(1) not in params:
                raise ToolError("missing_parameter", f"The program needs a value for '{m.group(1)}'.",
                                parameter=m.group(1))
            return params[m.group(1)]

        def sub(mm):
            if mm.group(1) not in params:
                raise ToolError("missing_parameter", f"The program needs a value for '{mm.group(1)}'.",
                                parameter=mm.group(1))
            return str(params[mm.group(1)])
        return PLACEHOLDER.sub(sub, value)
    if isinstance(value, list):
        return [_substitute(v, params) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, params) for k, v in value.items()}
    return value


def _placeholders(value: Any, found: set[str]) -> None:
    if isinstance(value, str):
        found.update(PLACEHOLDER.findall(value))
    elif isinstance(value, list):
        for v in value:
            _placeholders(v, found)
    elif isinstance(value, dict):
        for v in value.values():
            _placeholders(v, found)


class Environment:
    def __init__(self, cfg: dict, db, memory: MemoryStore, agent_id: str, session_id: int,
                 session_number: int, mode: str = "exploration", allowed_tools: list[str] | None = None,
                 scene: Callable[[PhysicsWorld], None] | None = None,
                 clock: Callable[[], float] | None = None, wall_used: float = 0.0):
        self.cfg, self.db, self.memory = cfg, db, memory
        self.agent_id, self.sid, self.session_number, self.mode = agent_id, session_id, session_number, mode
        self.allowed = set(allowed_tools) if allowed_tools else None
        self.clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._ev_buf: list[dict] = []
        self._fr_buf: list[dict] = []
        self._sink_on = True
        self._idx = 0
        self._cur_idx: int | None = None
        self._n, self._prev = db.last_hash(session_id)
        self.agent_calls = 0
        self.phase = "active"
        self.wrapup_left = WRAPUP_CALLS
        self.end_reason: str | None = None
        self.continue_requested = False
        self.duration_s = float(cfg["session"]["duration_minutes"]) * 60.0
        self.max_steps = cfg["session"].get("max_steps")
        self._wall_prior = float(wall_used)
        self._t0 = self.clock()
        self.counters = {"failed_actions": 0, "physical_actions": 0, "memory_reads": 0,
                         "prior_programs_run": 0, "sandbox_tests": 0}
        # Repeat-observation detection (audit 2026-11-13): map an observation's (tool, args, world-state)
        # signature to the action index that first produced it this session. A 2B model once made 126
        # identical observations in a row because nothing ever told it the answer had not changed.
        self._obs_seen: dict[tuple, int] = {}
        self._repeat_obs_streak = 0
        self.world = PhysicsWorld(cfg, event_sink=self._on_event, frame_sink=self._on_frame)
        if scene is not None:
            scene(self.world)
            self.world.setup_finish()
        self.tools = PhysicalTools(self.world)
        self.initial_state_hash = self.world.state_hash()
        self._flush_only()

    # ------------------------------------------------------------ sinks and logging
    def _on_event(self, ev: dict) -> None:
        if self._sink_on:
            self._ev_buf.append({"source": "physics", "t": ev["t"], "step": ev["step"], "type": ev["type"],
                                 "action_idx": self._cur_idx, "data": ev["data"], "hidden": ev.get("hidden")})

    def _on_frame(self, frame: dict) -> None:
        if self._sink_on:
            self._fr_buf.append(frame)

    def note(self, etype: str, **data: Any) -> None:
        """An interface-level event (what the agent recorded, claimed or was told)."""
        self._ev_buf.append({"source": "interface", "t": round(self.world.time, 4), "step": self.world.steps,
                             "type": etype, "action_idx": self._cur_idx, "data": data, "hidden": None})

    def _flush_only(self) -> None:
        with self.db.tx():
            self._n, self._prev = self.db.append_events(self.sid, self._ev_buf, self._n, self._prev)
            self.db.append_frames(self.sid, self._fr_buf)
        self._ev_buf, self._fr_buf = [], []

    def elapsed(self) -> float:
        return self._wall_prior + (self.clock() - self._t0)

    def remaining(self) -> dict:
        rem_s = max(0.0, self.duration_s - self.elapsed())
        rem_steps = None if not self.max_steps else max(0, int(self.max_steps) - self.agent_calls)
        return {"seconds": round(rem_s, 1), "steps": rem_steps, "phase": self.phase}

    def _update_phase(self) -> None:
        if self.phase != "active":
            return
        if self.elapsed() >= self.duration_s:
            self.phase, self.end_reason = "wrapup", "time_limit"
        elif self.max_steps and self.agent_calls >= int(self.max_steps):
            self.phase, self.end_reason = "wrapup", "step_limit"
        if self.phase == "wrapup":
            self.note("session_time_over", reason=self.end_reason)

    def _notice(self) -> str | None:
        if self.phase == "wrapup":
            return (f"This session's time is over. Only memory tools still work ({self.wrapup_left} more calls); "
                    "save anything you want to keep. The island will then be reset.")
        if self.phase != "active":
            return None
        rem = self.remaining()
        s_cfg = self.cfg["session"]
        if rem["seconds"] <= float(s_cfg["wrap_up_notice_minutes"]) * 60.0:
            return (f"About {max(1, round(rem['seconds'] / 60))} minute(s) of this session remain. "
                    "The island is reset at the end; only your memory files are kept.")
        if rem["steps"] is not None and rem["steps"] <= int(s_cfg["wrap_up_notice_steps"]):
            return (f"{rem['steps']} action(s) remain in this session. "
                    "The island is reset at the end; only your memory files are kept.")
        return None

    # ------------------------------------------------------------ the single entry point
    def call(self, tool: str, args: Any = None, source: str = "agent", turn: int | None = None) -> dict:
        with self._lock:
            if source == "agent":
                self._update_phase()
            res = self._call(str(tool), args, source, turn, None)
            if source == "agent":
                if self.phase == "active":
                    self.agent_calls += 1
                elif self.phase == "wrapup":
                    self.wrapup_left -= 1
                    if self.wrapup_left <= 0:
                        self.phase = "closed"
                self._update_phase()
                notice = self._notice()
                if notice:
                    res["notice"] = notice
                every = int(self.cfg["recording"]["checkpoint_every_calls"])
                if every and self.agent_calls % every == 0:
                    self.checkpoint()
            return res

    def _call(self, tool: str, args: Any, source: str, turn: int | None, program: str | None) -> dict:
        self._idx += 1
        idx = self._idx
        prev_cur = self._cur_idx
        self._cur_idx = idx
        w = self.world
        t_start, n_from, wall0 = w.time, self._n + 1, time.perf_counter()
        spec = SPECS.get(tool)
        category = spec["category"] if spec else "unknown"
        clean: dict = {}
        try:
            if spec is None:
                raise ToolError("unknown_tool", f"There is no tool named '{tool[:60]}'.",
                                available_tools=sorted(self.allowed or SPECS))
            if self.allowed is not None and tool not in self.allowed:
                raise ToolError("tool_not_available", f"'{tool}' is not available in this session.",
                                available_tools=sorted(self.allowed))
            if self.phase == "closed":
                raise ToolError("session_closed", "This session has ended.")
            if self.phase == "wrapup" and category != "memory":
                raise ToolError("session_time_over", "This session's time is over. Only memory tools still work; "
                                "the island is about to be reset.")
            clean = normalise_args(tool, args)
            if tool in PHYSICAL_TOOL_NAMES:
                body = getattr(self.tools, tool)(**clean)
                if category in ("movement", "object"):
                    self.counters["physical_actions"] += 1
            else:
                body = getattr(self, f"_t_{tool}")(clean, turn)
            result = {"success": True, "action": tool, "action_index": idx, **body}
            code = None
        except (ToolError, ActionError, MemoryError_) as e:
            code = e.code
            result = {"success": False, "action": tool, "action_index": idx,
                      "error": {"code": e.code, "message": e.message}}
            if e.details:
                result["details"] = e.details
        except Exception as e:  # noqa: BLE001 - a crash in a tool must surface as a failure, never a success
            code = "internal_error"
            result = {"success": False, "action": tool, "action_index": idx,
                      "error": {"code": code, "message": f"{type(e).__name__}: {e}"}}
            self.db.add_record(self.agent_id, "integrity_flag",
                               {"kind": "internal_error", "tool": tool, "message": f"{type(e).__name__}: {e}"},
                               session_id=self.sid, action_idx=idx, t=w.time)
        result["sim_time"] = round(w.time, 4)
        if not result["success"]:
            self.counters["failed_actions"] += 1
        # Engine-side plausibility: every completed flight is checked against ballistics.
        for ev in self._ev_buf:
            if ev["source"] == "physics" and ev["type"] == "flight_end" and ev["data"].get("kind") == "throw":
                problems = check_flight(ev["data"], float(self.cfg["world"]["gravity"]))
                if problems:
                    self.db.add_record(self.agent_id, "integrity_flag",
                                       {"kind": "implausible_flight", "problems": problems,
                                        "flight_id": ev["data"].get("flight_id")},
                                       session_id=self.sid, action_idx=idx, t=w.time)
        state_hash = w.state_hash()
        # Repeat-observation feedback. A read-only observation that returns the same answer about an
        # unchanged world as an earlier call adds no information; tell the agent so, rather than letting
        # it loop. Only annotates (never blocks) - autonomy is preserved. Audit 2026-11-13.
        if result["success"] and source == "agent" and category == "observation":
            sig = (tool, _sha(clean or {}), state_hash)
            first = self._obs_seen.get(sig)
            if first is not None:
                self._repeat_obs_streak += 1
                result["repeat_of_action"] = first
                result["repeat_notice"] = (
                    f"This is the same observation as action {first}, and the world has not changed since "
                    f"(repeated {self._repeat_obs_streak}x). Observing again will keep returning this. To "
                    f"learn something new, act on it: move somewhere, pick up or move an object, run an "
                    f"experiment, or record what you have found and decide what to investigate next.")
            else:
                self._obs_seen[sig] = idx
                self._repeat_obs_streak = 0
        elif category in ("movement", "object") and result["success"]:
            # A physical action changes the world, so old observation signatures are stale: let the agent
            # re-observe freely without being told it is repeating.
            self._obs_seen.clear()
            self._repeat_obs_streak = 0
        self._ev_buf.append({"source": "interface", "t": round(w.time, 4), "step": w.steps, "type": "action",
                             "action_idx": idx, "hidden": None,
                             "data": {"idx": idx, "tool": tool[:60], "success": result["success"],
                                      "error_code": code, "source": source, "args_sha": _sha(clean or {}),
                                      "result_sha": _sha(result), "state_hash": state_hash}})
        with self.db.tx():
            self._n, self._prev = self.db.append_events(self.sid, self._ev_buf, self._n, self._prev)
            self.db.append_frames(self.sid, self._fr_buf)
            self.db.append_action(self.sid, idx=idx, tool=tool[:60], category=category, args=clean or {},
                                  result=result, success=1 if result["success"] else 0, error_code=code,
                                  source=source, program=program, turn=turn, t_start=round(t_start, 4),
                                  t_end=round(w.time, 4), n_from=n_from, n_to=self._n, wall_ts=time.time(),
                                  wall_ms=round((time.perf_counter() - wall0) * 1000, 2))
            self.db.update_session(self.sid, sim_time=round(w.time, 4), physics_steps=w.steps,
                                   action_count=self._idx, wall_seconds=round(self.elapsed(), 2))
        self._ev_buf, self._fr_buf = [], []
        self._cur_idx = prev_cur
        return result

    # ------------------------------------------------------------ checkpoints and recovery
    def checkpoint(self) -> None:
        self.db.save_checkpoint(self.sid, self._idx, {
            "world": self.world.get_state(), "n": self._n, "agent_calls": self.agent_calls,
            "phase": self.phase, "wrapup_left": self.wrapup_left, "counters": self.counters,
            "wall_used": round(self.elapsed(), 2)})

    def restore(self) -> dict:
        """Rebuild the world of an interrupted session: latest checkpoint, then a deterministic replay
        of every physical action logged after it. Returns a report including a state-hash check."""
        ck = self.db.latest_checkpoint(self.sid)
        acts = self.db.actions(self.sid)
        self._sink_on = False
        start_idx = 0
        if ck:
            st = ck["state"]
            self.world.set_state(st["world"])
            start_idx = int(ck["action_idx"])
            self.agent_calls = int(st["agent_calls"])
            self.counters.update(st.get("counters", {}))
            n_ck = int(st["n"])
        else:
            n_ck = 0
        past = self.db.events(self.sid, limit=10 ** 9)
        self.world.events = [{"seq": e["n"], "t": e["t"], "step": e["step"], "type": e["type"], "data": e["data"]}
                             for e in past if e["source"] == "physics" and e["n"] <= n_ck]
        self.world.completed_flights = [e["data"] for e in self.world.events if e["type"] == "flight_end"]
        replayed = 0
        for a in acts:
            if a["idx"] <= start_idx or a["tool"] not in PHYSICAL_TOOL_NAMES or a["error_code"] in (
                    "invalid_argument", "tool_not_available", "session_time_over", "session_closed"):
                continue
            try:
                getattr(self.tools, a["tool"])(**a["args"])
            except (ToolError, ActionError):
                pass
            replayed += 1
            if a["source"] == "agent" and a["idx"] > start_idx:
                pass
        self.agent_calls += sum(1 for a in acts if a["idx"] > start_idx and a["source"] == "agent")
        self._sink_on = True
        self._ev_buf, self._fr_buf = [], []
        self._idx = max([a["idx"] for a in acts] or [0])
        self._n, self._prev = self.db.last_hash(self.sid)
        logged = None
        for e in reversed(past):
            if e["type"] == "action":
                logged = e["data"].get("state_hash")
                break
        now = self.world.state_hash()
        report = {"checkpoint_action": start_idx, "actions_replayed": replayed, "resumed_at_action": self._idx,
                  "state_hash": now, "logged_state_hash": logged,
                  "state_matches_log": logged is None or logged == now}
        self.note("session_resumed", **report)
        self._flush_only()
        return report

    # ------------------------------------------------------------ world tools
    def _session_events(self, types=None, object_id=None, since_action=None, limit=30) -> list[dict]:
        types = [t for t in (types or AGENT_EVENT_TYPES) if t in AGENT_EVENT_TYPES] or list(AGENT_EVENT_TYPES)
        rows = self.db.events(self.sid, types=types, include_hidden=False, limit=10 ** 9)
        out = []
        for r in rows:
            if r["source"] != "physics":
                continue
            if since_action is not None and (r["action_idx"] or 0) < since_action:
                continue
            if object_id:
                d = r["data"]
                if d.get("object") != object_id and object_id not in (d.get("objects") or []):
                    continue
            out.append(visible_event(r))
        return out[-limit:]

    def _t_inspect_world_history(self, a: dict, turn) -> dict:
        limit = max(1, min(int(a.get("limit", 30)), 200))
        bad = [t for t in a.get("types") or [] if t not in AGENT_EVENT_TYPES]
        if bad:
            raise ToolError("invalid_argument", f"Unknown event type(s) {bad}.", event_types=list(AGENT_EVENT_TYPES))
        evs = self._session_events(a.get("types"), a.get("object_id"), a.get("since_action"), limit)
        return {"events": evs, "count": len(evs), "event_types": list(AGENT_EVENT_TYPES),
                "actions_so_far": self._idx - 1, "sim_time_now": round(self.world.time, 4)}

    def _t_inspect_recent_events(self, a: dict, turn) -> dict:
        evs = self._session_events(limit=max(1, min(int(a.get("limit", 12)), 100)))
        return {"events": evs, "count": len(evs)}

    def _validate_steps(self, steps: Any, sandbox: bool) -> list[dict]:
        if not isinstance(steps, list) or not steps:
            raise ToolError("invalid_argument", "steps must be a non-empty list of {\"tool\": ..., \"args\": {...}}.")
        limit = MAX_SANDBOX_STEPS if sandbox else MAX_PROGRAM_STEPS
        if len(steps) > limit:
            raise ToolError("invalid_argument", f"At most {limit} steps are allowed here.", max_steps=limit)
        out = []
        for i, s in enumerate(steps):
            if not isinstance(s, dict) or not isinstance(s.get("tool"), str):
                raise ToolError("invalid_argument", f"Step {i + 1} must be an object with a 'tool' name.")
            name, sargs = s["tool"], s.get("args") or {}
            spec = SPECS.get(name)
            if spec is None:
                raise ToolError("invalid_argument", f"Step {i + 1}: there is no tool named '{name}'.")
            if not spec["program_ok"] or (sandbox and name not in PHYSICAL_TOOL_NAMES):
                raise ToolError("invalid_argument", f"Step {i + 1}: '{name}' cannot be used inside "
                                f"{'a physics test' if sandbox else 'a program'}.")
            if not isinstance(sargs, dict):
                raise ToolError("invalid_argument", f"Step {i + 1}: args must be an object.")
            unknown = sorted(set(sargs) - set(spec["parameters"]["properties"]))
            if unknown:
                raise ToolError("invalid_argument", f"Step {i + 1}: {name} has no argument(s) {unknown}.")
            out.append({"tool": name, "args": sargs})
        return out

    def _t_run_physics_test(self, a: dict, turn) -> dict:
        params = a["parameters"]
        steps = self._validate_steps(params.get("steps"), sandbox=True)
        sandbox = self.world.clone()
        tools = PhysicalTools(sandbox)
        t0, results = sandbox.time, []
        for i, s in enumerate(steps):
            try:
                body = getattr(tools, s["tool"])(**normalise_args(s["tool"], s["args"]))
                results.append({"step": i + 1, "tool": s["tool"], "success": True, **body})
            except (ToolError, ActionError) as e:
                results.append({"step": i + 1, "tool": s["tool"], "success": False,
                                "error": {"code": e.code, "message": e.message}, **({"details": e.details} if e.details else {})})
        self.counters["sandbox_tests"] += 1
        evs = [visible_event(e) for e in sandbox.events if e["type"] in AGENT_EVENT_TYPES]
        return {"sandbox": True, "label": params.get("label"), "results": results,
                "steps_run": len(results), "steps_failed": sum(1 for r in results if not r["success"]),
                "events": evs[-25:], "test_duration": round(sandbox.time - t0, 4),
                "note": "This ran in a private copy. The real world is unchanged."}

    def _t_save_program(self, a: dict, turn) -> dict:
        steps = self._validate_steps(a["steps"], sandbox=False)
        found: set[str] = set()
        _placeholders(steps, found)
        prog = self.memory.save_program(self.session_number, a["name"], a.get("description"), steps)
        self.note("program_saved", name=prog["name"], version=prog["version"], steps=len(steps))
        self.db.add_record(self.agent_id, "program", {"name": prog["name"], "version": prog["version"],
                                                      "steps": steps, "description": prog["description"]},
                           session_id=self.sid, key=prog["name"], action_idx=self._cur_idx, t=self.world.time)
        return {"program": prog["name"], "version": prog["version"], "steps": len(steps),
                "parameters": sorted(found), "saved": True}

    def _t_execute_program(self, a: dict, turn) -> dict:
        prog = self.memory.get_program(a["name"])
        if prog is None:
            raise ToolError("unknown_program", f"There is no saved program '{a['name']}'.",
                            programs=[p["name"] for p in self.memory.list_programs()])
        repeat = int(a.get("repeat", 1))
        if not (1 <= repeat <= 20):
            raise ToolError("invalid_argument", "repeat must be between 1 and 20.")
        if repeat * len(prog["steps"]) > MAX_PROGRAM_ACTIONS:
            raise ToolError("invalid_argument", f"That would run {repeat * len(prog['steps'])} actions; the limit "
                            f"per call is {MAX_PROGRAM_ACTIONS}.")
        params = a.get("params") or {}
        on_error = a.get("on_error", "stop")
        prior = prog["created_session"] < self.session_number
        # Resolve every placeholder before anything runs, so a missing value cannot leave the
        # program half executed.
        resolved = [{"tool": s["tool"], "args": _substitute(s["args"], params)} for s in prog["steps"]]
        if prior:
            self.counters["prior_programs_run"] += 1
        results, failed, stopped = [], 0, False
        for it in range(repeat):
            for i, s in enumerate(resolved):
                r = self._call(s["tool"], s["args"], "program", turn, prog["name"])
                results.append({"iteration": it + 1, "step": i + 1, **r})
                if not r["success"]:
                    failed += 1
                    if on_error == "stop":
                        stopped = True
                        break
            if stopped:
                break
        if len(results) > 12:
            compact = [{"iteration": r["iteration"], "step": r["step"], "action": r["action"],
                        "action_index": r["action_index"], "success": r["success"],
                        **({"error": r["error"]} if not r["success"] else {})} for r in results[:-1]]
            results = compact + [results[-1]]
            detail = "Only the last step is shown in full; use the measure_* and history tools for the others."
        else:
            detail = None
        self.note("program_executed", name=prog["name"], repeat=repeat, steps_run=len(results),
                  steps_failed=failed, created_in_earlier_session=prior)
        return {"program": prog["name"], "version": prog["version"], "repeat": repeat,
                "steps_run": len(results), "steps_failed": failed, "stopped_early": stopped,
                "results": results, **({"note": detail} if detail else {})}

    def _t_list_programs(self, a: dict, turn) -> dict:
        progs = self.memory.list_programs()
        out = []
        for p in progs:
            found: set[str] = set()
            _placeholders(p["steps"], found)
            out.append({"name": p["name"], "description": p["description"], "steps": len(p["steps"]),
                        "parameters": sorted(found), "created_session": p["created_session"]})
        return {"programs": out, "count": len(out)}

    # ------------------------------------------------------------ memory tools
    def _t_read_memory(self, a: dict, turn) -> dict:
        self.counters["memory_reads"] += 1
        m, f = self.memory, a["file"]
        self.note("memory_read", file=f)
        if f in ("handbook", "diary", "mistakes"):
            return {"file": f, "content": (m.dir / f"{f}.md").read_text(encoding="utf-8")}
        if f == "goals":
            return {"file": f, "goals": m.state["goals"]}
        if f == "discoveries":
            return {"file": f, "discoveries": [{k: d[k] for k in ("id", "statement", "status", "status_reason",
                                                                  "evidence", "category", "created_session", "flags")}
                                               for d in m.discoveries["items"]]}
        if f == "experiments":
            return {"file": f, "experiments": m.experiments["items"][-30:], "total": len(m.experiments["items"])}
        return {"file": f, "programs": m.list_programs()}

    def _t_write_diary(self, a: dict, turn) -> dict:
        if not a["text"].strip():
            raise ToolError("invalid_argument", "The diary entry is empty.")
        e = self.memory.add_diary(self.session_number, self.world.time, a["text"])
        self.db.add_record(self.agent_id, "diary", e, session_id=self.sid, key=e["id"],
                           action_idx=self._cur_idx, t=self.world.time)
        self.note("diary_entry", id=e["id"], text=e["text"])
        return {"entry_id": e["id"], "saved": True}

    def _t_add_note(self, a: dict, turn) -> dict:
        if not a["text"].strip():
            raise ToolError("invalid_argument", "The note is empty.")
        n = self.memory.add_note(self.session_number, self.world.time, a["text"])
        self.note("note_added", id=n["id"], text=n["text"])
        return {"note_id": n["id"], "saved": True,
                "filed_under": "Working notes (unverified). Use record_discovery with experiments for findings."}

    def _t_record_experiment(self, a: dict, turn) -> dict:
        refs = sorted({int(r) for r in a["action_refs"]})
        cited = {r: self.db.action(self.sid, r) for r in refs}
        fields = {k: a.get(k) for k in ("question", "hypothesis", "action_description", "expected_result",
                                        "actual_result", "measurements", "supported", "needs_followup",
                                        "conclusion")}
        fields["action_refs"] = refs
        ver = verify_experiment(fields, cited)
        limits = check_claim_limits(fields.get("measurements") or {}, {"max_height": 60.0, "max_speed": 60.0})
        if limits:
            ver["issues"].extend(limits)
            ver.update({"verified": False, "verdict": "implausible", "level": None})
        e = self.memory.add_experiment(self.session_number, self.world.time, fields, ver)
        self.db.add_record(self.agent_id, "experiment", e, session_id=self.sid, key=e["id"],
                           action_idx=self._cur_idx, t=self.world.time)
        if ver["verdict"] in ("contradicted", "implausible", "invalid_references"):
            self.db.add_record(self.agent_id, "integrity_flag",
                               {"kind": f"experiment_{ver['verdict']}", "experiment": e["id"],
                                "issues": ver["issues"], "contradictions": ver.get("contradictions", [])},
                               session_id=self.sid, action_idx=self._cur_idx, t=self.world.time)
        self.note("experiment_recorded", id=e["id"], hypothesis=fields["hypothesis"], supported=fields["supported"],
                  verdict=ver["verdict"], verified=ver["verified"], action_refs=refs)
        return {"experiment_id": e["id"], "saved": True,
                "evidence_check": {"verdict": ver["verdict"], "counts_as_evidence": ver["verified"],
                                   "matched_measurements": ver["matches"],
                                   "contradicted_measurements": ver["contradictions"],
                                   "not_found_in_cited_actions": ver["unmatched"], "issues": ver["issues"]}}

    def _discovery_result(self, d: dict, requested: str | None) -> dict:
        self.db.add_record(self.agent_id, "discovery",
                           {k: d[k] for k in ("id", "statement", "status", "requested_status", "status_reason",
                                              "evidence", "category")},
                           session_id=self.sid, key=d["id"], action_idx=self._cur_idx, t=self.world.time)
        self.note("discovery_recorded", id=d["id"], statement=d["statement"], requested_status=requested,
                  granted_status=d["status"], evidence=d["evidence"])
        return {"discovery_id": d["id"], "requested_status": requested, "granted_status": d["status"],
                "status_granted_as_requested": requested is None or requested == d["status"],
                "reason": d["status_reason"], "evidence": d["evidence"], "saved": True}

    def _t_record_discovery(self, a: dict, turn) -> dict:
        if not a["statement"].strip():
            raise ToolError("invalid_argument", "The statement is empty.")
        d = self.memory.add_discovery(self.session_number, a["statement"], a["status"],
                                      [str(x) for x in a.get("experiment_ids") or []], a.get("category"))
        return self._discovery_result(d, a["status"])

    def _t_update_discovery(self, a: dict, turn) -> dict:
        d = self.memory.update_discovery(self.session_number, a["discovery_id"], a.get("status"),
                                         [str(x) for x in a.get("add_experiment_ids") or []], a.get("note"))
        return self._discovery_result(d, a.get("status"))

    def _t_record_mistake(self, a: dict, turn) -> dict:
        m = self.memory.add_mistake(self.session_number, self.world.time, a["what"], a.get("why"),
                                    a.get("lesson"), [int(r) for r in a.get("action_refs") or []])
        self.db.add_record(self.agent_id, "mistake", m, session_id=self.sid, key=m["id"],
                           action_idx=self._cur_idx, t=self.world.time)
        self.note("mistake_recorded", id=m["id"], what=m["what"])
        return {"mistake_id": m["id"], "saved": True}

    def _t_set_goal(self, a: dict, turn) -> dict:
        if not a["title"].strip():
            raise ToolError("invalid_argument", "The goal needs a title.")
        g = self.memory.add_goal(self.session_number, self.world.time, a["title"], a.get("description"),
                                 kind=a.get("kind"), next_step=a.get("next_step"))
        self.db.add_record(self.agent_id, "goal", {"id": g["id"], "title": g["title"], "status": "active",
                                                   "description": g["description"], "kind": g["kind"],
                                                   "next_step": g["next_step"], "event": "set"},
                           session_id=self.sid, key=g["id"], action_idx=self._cur_idx, t=self.world.time)
        self.note("goal_set", id=g["id"], title=g["title"], kind=g["kind"])
        return {"goal_id": g["id"], "status": "active", "kind": g["kind"], "saved": True}

    def _check_claim(self, ctype: str, params: dict, description: str | None, goal_id: str | None = None) -> dict:
        if ctype not in VERIFIERS:
            raise ToolError("invalid_argument", f"Unknown claim type '{ctype}'.", claim_types=CLAIM_HELP)
        if not isinstance(params, dict):
            raise ToolError("invalid_argument", "Claim params must be an object.", claim_types=CLAIM_HELP)
        events = self.db.events(self.sid, limit=10 ** 9, include_hidden=False)
        events = [e for e in events if e["source"] == "physics"] + [
            {"type": e["type"], "t": e["t"], "data": e["data"]} for e in self._ev_buf if e["source"] == "physics"]
        try:
            res = verify_claim(ctype, params, self.world, events)
        except TypeError as exc:
            raise ToolError("invalid_argument", f"Bad parameters for a '{ctype}' claim: {exc}.",
                            expected=CLAIM_HELP[ctype]) from None
        rec = {"type": ctype, "params": params, "description": description, "goal_id": goal_id,
               "claimed": True, "verified": res["verified"], "measured": res["measured"],
               "criteria": res["criteria"]}
        self.db.add_record(self.agent_id, "claim", rec, session_id=self.sid, action_idx=self._cur_idx,
                           t=self.world.time)
        if not res["verified"]:
            self.db.add_record(self.agent_id, "integrity_flag",
                               {"kind": "claim_not_verified", "claim": ctype, "params": params,
                                "measured": res["measured"]},
                               session_id=self.sid, action_idx=self._cur_idx, t=self.world.time)
        self.note("claim_checked", claim=ctype, params=params, verified=res["verified"], goal_id=goal_id)
        return res

    def _t_update_goal(self, a: dict, turn) -> dict:
        ver = None
        if a.get("claim") is not None:
            c = a["claim"]
            if not isinstance(c, dict) or "type" not in c:
                raise ToolError("invalid_argument", "claim must be {\"type\": ..., \"params\": {...}}.",
                                claim_types=CLAIM_HELP)
            if self.memory.get_goal(a["goal_id"]) is None:
                raise ToolError("unknown_goal", f"There is no goal '{a['goal_id']}'.")
            res = self._check_claim(str(c["type"]), c.get("params") or {}, a.get("note"), a["goal_id"])
            ver = {"claim": c, "verified": res["verified"], "measured": res["measured"],
                   "session": self.session_number}
        g = self.memory.update_goal(self.session_number, self.world.time, a["goal_id"], a["status"],
                                    a.get("note"), ver)
        self.db.add_record(self.agent_id, "goal", {"id": g["id"], "title": g["title"], "status": g["status"],
                                                   "note": a.get("note"), "event": "update",
                                                   "independently_verified": None if ver is None else ver["verified"]},
                           session_id=self.sid, key=g["id"], action_idx=self._cur_idx, t=self.world.time)
        self.note("goal_updated", id=g["id"], status=g["status"],
                  independently_verified=None if ver is None else ver["verified"])
        out = {"goal_id": g["id"], "status": g["status"], "saved": True}
        if ver is not None:
            out["claim_check"] = {"verified": ver["verified"], "measured": ver["measured"]}
            if a["status"] == "completed" and not ver["verified"]:
                out["warning"] = ("The goal is recorded as completed by you, but the independent check did not "
                                  "confirm the claim. Both facts are kept.")
        return out

    def _t_assign_meaning(self, a: dict, turn) -> dict:
        targets = a["target"] if isinstance(a["target"], list) else [a["target"]]
        unknown = [t for t in targets if t not in self.world.specs or (
            self.world.specs[t].movable and t not in self.world.active)]
        if unknown:
            raise ToolError("unknown_object", f"Unknown object id(s): {unknown}.",
                            known_ids=sorted(self.world.visible_ids()))
        if not a["meaning"].strip():
            raise ToolError("invalid_argument", "The meaning is empty.")
        rec = {"targets": targets, "meaning": a["meaning"].strip(),
               "positions": {t: self.tools.get_object_positions()["objects"].get(t) for t in targets}}
        self.db.add_record(self.agent_id, "label", rec, session_id=self.sid, action_idx=self._cur_idx,
                           t=self.world.time)
        self.note("meaning_assigned", targets=targets, meaning=rec["meaning"])
        return {"targets": targets, "meaning": rec["meaning"], "saved": True}

    def _t_claim_achievement(self, a: dict, turn) -> dict:
        res = self._check_claim(a["type"], a.get("params") or {}, a.get("description"))
        return {"claim": a["type"], "verified": res["verified"], "criteria": res["criteria"],
                "measured": res["measured"],
                "note": None if res["verified"] else "The physical record does not support this claim."}

    def _t_end_session(self, a: dict, turn) -> dict:
        self.phase = "closed"
        self.end_reason = "agent_ended"
        wants_more = bool(a.get("continue_next_session"))
        self.continue_requested = wants_more
        self.note("session_end_requested", reason=a.get("reason"), continue_next_session=wants_more)
        note = "Your memory files are saved. The island will be reset before your next session."
        if wants_more:
            note += (" You asked to continue: if the researcher is running multiple sessions, another one "
                     "will start with this same memory. Make sure any unfinished goal or open question is "
                     "recorded so you can resume it.")
        return {"ended": True, "reason": a.get("reason"), "continue_next_session": wants_more, "note": note}

    # ------------------------------------------------------------ learning tools (neural networks)
    def _t_nn_list(self, a: dict, turn) -> dict:
        from ..ml.agent_tools import nn_list
        return nn_list(self)

    def _t_nn_make_dataset(self, a: dict, turn) -> dict:
        from ..ml.agent_tools import nn_make_dataset
        return nn_make_dataset(self, a["name"], a.get("min_rows", 20), kind=a.get("kind", "throw"))

    def _t_nn_train(self, a: dict, turn) -> dict:
        from ..ml.agent_tools import nn_train
        return nn_train(self, a["dataset_name"], a["spec"], a.get("train_config"))

    def _t_nn_predict(self, a: dict, turn) -> dict:
        from ..ml.agent_tools import nn_predict
        return nn_predict(self, a["model_name"], a["inputs"], a.get("version"))

    def _t_nn_evaluate(self, a: dict, turn) -> dict:
        from ..ml.agent_tools import nn_evaluate
        return nn_evaluate(self, a["model_name"], a.get("version"))

    # ------------------------------------------------------------ researcher-side views
    def snapshot(self) -> dict:
        """Full current state for the dashboard and for the end-of-session record."""
        with self._lock:
            d = self.world.describe()
            d.update({"session_id": self.sid, "agent_id": self.agent_id, "mode": self.mode,
                      "phase": self.phase, "actions": self._idx, "remaining": self.remaining(),
                      "config_hash": config_hash(self.cfg)})
            return d
