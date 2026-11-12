"""Agent memory: the only thing that persists for an agent between sessions.

Canonical data is JSON (discoveries.json, experiments.json, state.json, programs/).
The Markdown files (diary.md, handbook.md, mistakes.md, unfinished_goals.md) are
rendered from it after every change, and the start-of-session briefing is built
by reading those rendered files back from disk.

Epistemic status is never taken on the author's word. A discovery's status is
*granted* by evaluate_status() from experiments whose evidence was checked against
the engine's own action log (see integrity/evidence.py).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

STATUSES = ("confirmed", "probable", "hypothesis", "disproven")
STATUS_TITLES = {
    "confirmed": "Verified discoveries (at least two independently checked supporting experiments)",
    "probable": "Probable explanations (some checked evidence; repeat before relying on them)",
    "hypothesis": "Untested hypotheses (no checked evidence yet; do not treat these as facts)",
    "disproven": "Disproven theories (tested and contradicted by a checked experiment)",
}
GOAL_OPEN = ("active", "paused")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,47}$")


class MemoryError_(Exception):
    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def evaluate_status(requested: str, experiments: list[dict], cfg_memory: dict) -> tuple[str, str]:
    """Decide which status the evidence supports. Returns (granted, reason)."""
    ok = [e for e in experiments if e.get("verification", {}).get("verified")]
    supporting = [e for e in ok if e.get("supported") is True]
    refuting = [e for e in ok if e.get("supported") is False]

    def distinct(items: list[dict]) -> int:
        # The same actions cited twice are one piece of evidence, not two.
        return len({(e.get("session"), tuple(sorted(e.get("action_refs") or []))) for e in items})

    n_sup = distinct(supporting)
    # "Measured" evidence: at least one claimed number matched a value the engine recorded.
    n_measured = distinct([e for e in supporting if e.get("verification", {}).get("level") == "measured"])
    need_c = int(cfg_memory.get("confirm_min_experiments", 2))
    need_p = int(cfg_memory.get("probable_min_experiments", 1))
    unverified = len(experiments) - len(ok)
    tail = f" ({unverified} cited experiment(s) failed the evidence check and were not counted)" if unverified else ""
    if requested == "disproven":
        if refuting:
            return "disproven", f"{len(refuting)} checked experiment(s) contradict it{tail}"
        return "hypothesis", "no checked experiment contradicts it yet, so it stays an untested hypothesis" + tail
    if refuting and supporting:
        return "probable" if n_sup > len(refuting) else "hypothesis", \
            f"evidence conflicts: {n_sup} supporting vs {len(refuting)} contradicting checked experiment(s){tail}"
    if refuting:
        return "disproven", f"{len(refuting)} checked experiment(s) contradict it{tail}"
    if requested == "confirmed":
        if n_measured >= need_c:
            return "confirmed", f"{n_measured} independent experiments with engine-matched measurements support it{tail}"
        if n_sup >= need_p:
            return "probable", (f"{n_measured} supporting experiment(s) have measurements that match the engine's "
                                f"record; {need_c} with different evidence are required for 'confirmed'{tail}")
        return "hypothesis", "no checked supporting experiment was cited" + tail
    if requested == "probable":
        if n_sup >= need_p:
            return "probable", f"{n_sup} checked supporting experiment(s){tail}"
        return "hypothesis", "no checked supporting experiment was cited" + tail
    return "hypothesis", "recorded as an untested hypothesis"


class MemoryStore:
    def __init__(self, root: str | Path, agent_id: str, cfg: dict, db=None):
        self.root = Path(root)
        self.agent_id = agent_id
        self.cfg = cfg
        self.mcfg = cfg.get("memory", {})
        self.db = db
        self.dir = self.root / agent_id
        (self.dir / "programs").mkdir(parents=True, exist_ok=True)
        (self.dir / "session_history").mkdir(parents=True, exist_ok=True)
        self.discoveries = self._load("discoveries.json", {"agent_id": agent_id, "next_id": 1, "items": []})
        self.experiments = self._load("experiments.json", {"agent_id": agent_id, "next_id": 1, "items": []})
        self.state = self._load("state.json", {"agent_id": agent_id, "sessions_completed": 0,
                                               "next": {"Y": 1, "M": 1, "G": 1, "N": 1},
                                               "diary": [], "mistakes": [], "goals": [], "notes": []})
        if not (self.dir / "handbook.md").exists():
            self.save()

    # ------------------------------------------------------------ persistence
    def _load(self, name: str, default: dict) -> dict:
        p = self.dir / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return default

    def save(self) -> None:
        _write_atomic(self.dir / "discoveries.json", json.dumps(self.discoveries, indent=2))
        _write_atomic(self.dir / "experiments.json", json.dumps(self.experiments, indent=2))
        _write_atomic(self.dir / "state.json", json.dumps(self.state, indent=2))
        self.render()

    def _audit(self, actor: str, op: str, item_id: str | None, before: Any, after: Any, reason: str | None) -> None:
        line = {"ts": time.time(), "actor": actor, "op": op, "item_id": item_id, "reason": reason,
                "before": before, "after": after}
        with open(self.dir / "audit.log", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, default=str) + "\n")
        if self.db is not None:
            self.db.audit(self.agent_id, actor, op, item_id, before, after, reason)

    def is_empty(self) -> bool:
        s = self.state
        return not (self.discoveries["items"] or self.experiments["items"] or s["diary"] or s["mistakes"]
                    or s["goals"] or s["notes"] or self.list_programs())

    def _next(self, prefix: str) -> str:
        n = self.state["next"][prefix]
        self.state["next"][prefix] = n + 1
        return f"{prefix}-{n:03d}"

    # ------------------------------------------------------------ rendering
    def render(self) -> None:
        a = self.agent_id
        gen = "_Rendered from the agent's memory store. Edit with `playground memory ...`, not by hand._\n"
        # handbook
        out = [f"# Handbook: {a}\n", gen,
               "_Status is granted by an evidence check against the physics log, not by the author._\n"]
        for st in STATUSES:
            items = [d for d in self.discoveries["items"] if d["status"] == st]
            out.append(f"\n## {STATUS_TITLES[st]}\n")
            if not items:
                out.append("_None._\n")
            for d in items:
                ev = ", ".join(d["evidence"]) if d["evidence"] else "none"
                out.append(f"- **{d['id']}** [{d.get('category') or 'general'}] {d['statement']}  \n"
                           f"  evidence: {ev} | first recorded in session {d['created_session']}\n")
                if d.get("flags", {}).get("challenged"):
                    out.append(f"  - CHALLENGED by the researcher: {d['flags'].get('challenge_reason') or 'no reason given'}."
                               " Re-test before relying on it.\n")
                if d.get("flags", {}).get("marked_unverified"):
                    out.append("  - Marked unverified by the researcher: earlier evidence no longer counts.\n")
                for n in d.get("notes", []):
                    out.append(f"  - note (session {n['session']}): {n['text']}\n")
        out.append("\n## Working notes (the agent's own, unverified)\n")
        if not self.state["notes"]:
            out.append("_None._\n")
        for n in self.state["notes"]:
            out.append(f"- **{n['id']}** (session {n['session']}): {n['text']}\n")
        _write_atomic(self.dir / "handbook.md", "".join(out))
        # diary
        out = [f"# Diary: {a}\n", gen]
        cur = None
        for e in self.state["diary"]:
            if e["session"] != cur:
                cur = e["session"]
                out.append(f"\n## Session {cur}\n")
            out.append(f"- **{e['id']}** (t={e['t']:.1f}s) {e['text']}\n")
        _write_atomic(self.dir / "diary.md", "".join(out))
        # mistakes
        out = [f"# Mistakes and lessons: {a}\n", gen, "\n"]
        if not self.state["mistakes"]:
            out.append("_None recorded._\n")
        for m in self.state["mistakes"]:
            out.append(f"- **{m['id']}** (session {m['session']}, t={m['t']:.1f}s) {m['what']}\n")
            if m.get("why"):
                out.append(f"  - why: {m['why']}\n")
            if m.get("lesson"):
                out.append(f"  - lesson: {m['lesson']}\n")
        _write_atomic(self.dir / "mistakes.md", "".join(out))
        # goals
        out = [f"# Unfinished objectives: {a}\n", gen, "\n"]
        open_goals = [g for g in self.state["goals"] if g["status"] in GOAL_OPEN]
        if not open_goals:
            out.append("_None._\n")
        for g in open_goals:
            out.append(f"- **{g['id']}** [{g['status']}] {g['title']} (set in session {g['origin_session']})\n")
            if g.get("description"):
                out.append(f"  - {g['description']}\n")
            for h in g["history"][-3:]:
                if h.get("note"):
                    out.append(f"  - session {h['session']}: {h['note']}\n")
        _write_atomic(self.dir / "unfinished_goals.md", "".join(out))

    def briefing(self) -> str:
        """The text an agent is given at the start of a session: its own files, read from disk."""
        if self.is_empty():
            return ("You have no memory files yet. This is your first session on this island, "
                    "so nothing is known in advance.")
        parts = [f"YOUR MEMORY FILES ({self.agent_id}; sessions completed so far: "
                 f"{self.state['sessions_completed']}).\n"
                 "You wrote these in earlier sessions. The island has been reset to its original configuration "
                 "since then, so objects are back where they started and anything you built is gone. "
                 "Entries are only as reliable as their stated status."]
        for name in ("handbook.md", "unfinished_goals.md", "mistakes.md"):
            parts.append(f"----- {name} -----\n" + (self.dir / name).read_text(encoding="utf-8").strip())
        diary = self.state["diary"][-int(self.mcfg.get("diary_entries_in_briefing", 12)):]
        if diary:
            lines = [f"- (session {e['session']}, t={e['t']:.1f}s) {e['text']}" for e in diary]
            parts.append("----- diary.md (most recent entries) -----\n" + "\n".join(lines))
        progs = self.list_programs()
        if progs:
            lines = [f"- {p['name']}: {p.get('description') or 'no description'} ({len(p['steps'])} steps)" for p in progs]
            parts.append("----- programs/ (saved programs you can run with execute_program) -----\n" + "\n".join(lines))
        exps = self.experiments["items"][-6:]
        if exps:
            lines = []
            for e in exps:
                v = e.get("verification", {})
                lines.append(f"- {e['id']} (session {e['session']}): {e.get('hypothesis') or e.get('question')} -> "
                             f"{e.get('actual_result')} [evidence check: {v.get('verdict', 'n/a')}]")
            parts.append("----- experiments.json (most recent) -----\n" + "\n".join(lines))
        return "\n\n".join(parts)

    # ------------------------------------------------------------ agent-side writes
    def add_diary(self, session: int, t: float, text: str) -> dict:
        e = {"id": self._next("Y"), "session": session, "t": round(t, 2), "text": text.strip()}
        self.state["diary"].append(e)
        self.save()
        return e

    def add_note(self, session: int, t: float, text: str) -> dict:
        n = {"id": self._next("N"), "session": session, "t": round(t, 2), "text": text.strip()}
        self.state["notes"].append(n)
        self.save()
        return n

    def add_experiment(self, session: int, t: float, fields: dict, verification: dict) -> dict:
        n = self.experiments["next_id"]
        self.experiments["next_id"] = n + 1
        e = {"id": f"E-{n:03d}", "session": session, "t": round(t, 2), **fields, "verification": verification}
        self.experiments["items"].append(e)
        self.save()
        return e

    def get_experiment(self, eid: str) -> dict | None:
        return next((e for e in self.experiments["items"] if e["id"] == eid), None)

    def get_discovery(self, did: str) -> dict | None:
        return next((d for d in self.discoveries["items"] if d["id"] == did), None)

    def _resolve_experiments(self, ids: list[str]) -> list[dict]:
        out, missing = [], []
        for i in ids:
            e = self.get_experiment(str(i))
            (out if e else missing).append(e or i)
        if missing:
            raise MemoryError_("unknown_experiment", f"Unknown experiment id(s): {', '.join(map(str, missing))}.",
                               known=[e["id"] for e in self.experiments["items"]][-20:])
        return out

    def add_discovery(self, session: int, statement: str, requested: str, experiment_ids: list[str],
                      category: str | None = None) -> dict:
        if requested not in STATUSES:
            raise MemoryError_("invalid_status", f"status must be one of: {', '.join(STATUSES)}.")
        exps = self._resolve_experiments(experiment_ids or [])
        granted, reason = evaluate_status(requested, exps, self.mcfg)
        n = self.discoveries["next_id"]
        self.discoveries["next_id"] = n + 1
        d = {"id": f"D-{n:03d}", "statement": statement.strip(), "category": category, "status": granted,
             "requested_status": requested, "status_reason": reason, "evidence": [e["id"] for e in exps],
             "created_session": session, "updated_session": session, "flags": {}, "notes": [],
             "history": [{"session": session, "from": None, "to": granted, "requested": requested,
                          "reason": reason, "by": "agent"}]}
        self.discoveries["items"].append(d)
        self.save()
        return d

    def update_discovery(self, session: int, did: str, requested: str | None = None,
                         add_experiment_ids: list[str] | None = None, note: str | None = None) -> dict:
        d = self.get_discovery(did)
        if d is None:
            raise MemoryError_("unknown_discovery", f"There is no discovery '{did}'.",
                               known=[x["id"] for x in self.discoveries["items"]])
        if requested is not None and requested not in STATUSES:
            raise MemoryError_("invalid_status", f"status must be one of: {', '.join(STATUSES)}.")
        new = self._resolve_experiments(add_experiment_ids or [])
        for e in new:
            if e["id"] not in d["evidence"]:
                d["evidence"].append(e["id"])
        if note:
            d["notes"].append({"session": session, "text": note.strip()})
        exps = [e for e in (self.get_experiment(i) for i in d["evidence"]) if e]
        if d.get("flags", {}).get("marked_unverified"):
            # Evidence recorded before the researcher's reset no longer counts.
            cut = d["flags"].get("unverified_at_experiment", 0)
            exps = [e for e in exps if int(e["id"].split("-")[1]) > cut]
        req = requested or d.get("requested_status") or d["status"]
        granted, reason = evaluate_status(req, exps, self.mcfg)
        if granted != d["status"] or requested:
            d["history"].append({"session": session, "from": d["status"], "to": granted, "requested": req,
                                 "reason": reason, "by": "agent"})
        d.update({"status": granted, "requested_status": req, "status_reason": reason, "updated_session": session})
        if granted in ("confirmed", "disproven") and new:
            d["flags"].pop("challenged", None)
            d["flags"].pop("challenge_reason", None)
        self.save()
        return d

    def add_mistake(self, session: int, t: float, what: str, why: str | None, lesson: str | None,
                    action_refs: list[int] | None = None) -> dict:
        m = {"id": self._next("M"), "session": session, "t": round(t, 2), "what": what.strip(),
             "why": (why or "").strip() or None, "lesson": (lesson or "").strip() or None,
             "action_refs": action_refs or []}
        self.state["mistakes"].append(m)
        self.save()
        return m

    def add_goal(self, session: int, t: float, title: str, description: str | None) -> dict:
        g = {"id": self._next("G"), "title": title.strip(), "description": (description or "").strip() or None,
             "origin_session": session, "status": "active", "verification": None,
             "history": [{"session": session, "t": round(t, 2), "status": "active", "note": "goal set"}]}
        self.state["goals"].append(g)
        self.save()
        return g

    def get_goal(self, gid: str) -> dict | None:
        return next((g for g in self.state["goals"] if g["id"] == gid), None)

    def update_goal(self, session: int, t: float, gid: str, status: str, note: str | None,
                    verification: dict | None = None) -> dict:
        g = self.get_goal(gid)
        if g is None:
            raise MemoryError_("unknown_goal", f"There is no goal '{gid}'.",
                               known=[x["id"] for x in self.state["goals"]])
        if status not in ("active", "paused", "completed", "abandoned"):
            raise MemoryError_("invalid_status", "status must be one of: active, paused, completed, abandoned.")
        g["status"] = status
        g["history"].append({"session": session, "t": round(t, 2), "status": status, "note": (note or "").strip() or None})
        if verification is not None:
            g["verification"] = verification
        self.save()
        return g

    # ------------------------------------------------------------ programs
    def save_program(self, session: int, name: str, description: str | None, steps: list[dict]) -> dict:
        if not NAME_RE.match(name or ""):
            raise MemoryError_("invalid_name", "Program names use lowercase letters, digits, '_' or '-' (max 48).")
        p = self.dir / "programs" / f"{name}.json"
        prev = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        prog = {"name": name, "description": (description or "").strip() or None, "steps": steps,
                "created_session": prev["created_session"] if prev else session, "updated_session": session,
                "version": (prev["version"] + 1) if prev else 1}
        _write_atomic(p, json.dumps(prog, indent=2))
        return prog

    def get_program(self, name: str) -> dict | None:
        p = self.dir / "programs" / f"{name}.json"
        return json.loads(p.read_text(encoding="utf-8")) if NAME_RE.match(name or "") and p.exists() else None

    def list_programs(self) -> list[dict]:
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((self.dir / "programs").glob("*.json"))]

    # ------------------------------------------------------------ session boundaries
    def close_session(self, session: int, final_world_state: dict, summary: dict) -> Path:
        """Snapshot every memory file plus the final world state into session_history/."""
        self.state["sessions_completed"] = max(self.state["sessions_completed"], session)
        self.save()
        dest = self.dir / "session_history" / f"session_{session:03d}"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for name in ("diary.md", "handbook.md", "mistakes.md", "unfinished_goals.md", "discoveries.json",
                     "experiments.json", "state.json"):
            shutil.copy2(self.dir / name, dest / name)
        shutil.copytree(self.dir / "programs", dest / "programs")
        _write_atomic(dest / "final_world_state.json", json.dumps(final_world_state, indent=2))
        _write_atomic(dest / "summary.json", json.dumps(summary, indent=2, default=str))
        return dest

    # ------------------------------------------------------------ researcher-side operations
    def all_items(self) -> list[dict]:
        out = []
        for d in self.discoveries["items"]:
            out.append({"id": d["id"], "kind": "discovery", "status": d["status"], "text": d["statement"],
                        "session": d["created_session"], "evidence": d["evidence"], "flags": d.get("flags", {})})
        for e in self.experiments["items"]:
            out.append({"id": e["id"], "kind": "experiment", "status": e["verification"].get("verdict"),
                        "text": e.get("hypothesis") or e.get("question"), "session": e["session"]})
        for key, kind, field in (("mistakes", "mistake", "what"), ("goals", "goal", "title"),
                                 ("diary", "diary", "text"), ("notes", "note", "text")):
            for x in self.state[key]:
                out.append({"id": x["id"], "kind": kind, "status": x.get("status"), "text": x[field],
                            "session": x.get("session", x.get("origin_session"))})
        for p in self.list_programs():
            out.append({"id": f"P-{p['name']}", "kind": "program", "status": None,
                        "text": p.get("description") or p["name"], "session": p["created_session"]})
        return out

    def remove_item(self, item_id: str, reason: str | None = None, actor: str = "researcher") -> dict:
        """Remove one memory item of any kind. Returns the removed item."""
        prefix = item_id.split("-", 1)[0]
        removed = None
        if prefix == "P":
            p = self.dir / "programs" / f"{item_id[2:]}.json"
            if p.exists():
                removed = json.loads(p.read_text(encoding="utf-8"))
                p.unlink()
        else:
            coll = {"D": self.discoveries["items"], "E": self.experiments["items"], "M": self.state["mistakes"],
                    "G": self.state["goals"], "Y": self.state["diary"], "N": self.state["notes"]}.get(prefix)
            if coll is not None:
                for i, x in enumerate(coll):
                    if x["id"] == item_id:
                        removed = coll.pop(i)
                        break
        if removed is None:
            raise MemoryError_("unknown_item", f"There is no memory item '{item_id}' for {self.agent_id}.")
        if prefix == "E":
            # A discovery cannot keep a status that rested on evidence which no longer exists.
            for d in self.discoveries["items"]:
                if item_id in d["evidence"]:
                    d["evidence"].remove(item_id)
                    exps = [e for e in (self.get_experiment(i) for i in d["evidence"]) if e]
                    granted, why = evaluate_status(d.get("requested_status") or d["status"], exps, self.mcfg)
                    if granted != d["status"]:
                        d["history"].append({"session": None, "from": d["status"], "to": granted,
                                             "reason": f"evidence {item_id} removed; {why}", "by": actor})
                        d["status"], d["status_reason"] = granted, why
        self._audit(actor, "remove", item_id, removed, None, reason)
        self.save()
        return removed

    def mark_unverified(self, did: str, reason: str | None = None, actor: str = "researcher") -> dict:
        d = self.get_discovery(did)
        if d is None:
            raise MemoryError_("unknown_discovery", f"There is no discovery '{did}'.")
        before = json.loads(json.dumps(d))
        last = max([int(e["id"].split("-")[1]) for e in self.experiments["items"]] or [0])
        d["history"].append({"session": None, "from": d["status"], "to": "hypothesis",
                             "reason": reason or "marked unverified by researcher", "by": actor})
        d["status"] = "hypothesis"
        d["status_reason"] = "marked unverified by the researcher; needs fresh evidence"
        d["flags"].update({"marked_unverified": True, "unverified_at_experiment": last})
        self._audit(actor, "mark_unverified", did, before, d, reason)
        self.save()
        return d

    def challenge(self, did: str, reason: str, actor: str = "researcher") -> dict:
        d = self.get_discovery(did)
        if d is None:
            raise MemoryError_("unknown_discovery", f"There is no discovery '{did}'.")
        before = json.loads(json.dumps(d))
        d["flags"].update({"challenged": True, "challenge_reason": reason})
        self._audit(actor, "challenge", did, before, d, reason)
        self.save()
        return d

    def copy_to(self, dest_root: str | Path, remove_ids: list[str] | None = None, db=None,
                reason: str = "ablation copy") -> "MemoryStore":
        """Copy this memory to another root (benchmarks never touch the original), optionally ablated."""
        dest_dir = Path(dest_root) / self.agent_id
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(self.dir, dest_dir, ignore=shutil.ignore_patterns("session_history", "audit.log", "*.tmp"))
        other = MemoryStore(dest_root, self.agent_id, self.cfg, db=db)
        for item in remove_ids or []:
            other.remove_item(item, reason=reason, actor="benchmark")
        other.save()
        return other

    def stats(self) -> dict:
        by = {s: 0 for s in STATUSES}
        for d in self.discoveries["items"]:
            by[d["status"]] += 1
        exps = self.experiments["items"]
        return {"agent_id": self.agent_id, "sessions_completed": self.state["sessions_completed"],
                "discoveries": by, "discoveries_total": len(self.discoveries["items"]),
                "experiments": len(exps),
                "experiments_verified": sum(1 for e in exps if e["verification"].get("verified")),
                "experiments_contradicted": sum(1 for e in exps if e["verification"].get("verdict") == "contradicted"),
                "mistakes": len(self.state["mistakes"]), "diary_entries": len(self.state["diary"]),
                "goals_open": sum(1 for g in self.state["goals"] if g["status"] in GOAL_OPEN),
                "goals_total": len(self.state["goals"]), "programs": len(self.list_programs()),
                "challenged": sum(1 for d in self.discoveries["items"] if d.get("flags", {}).get("challenged"))}
