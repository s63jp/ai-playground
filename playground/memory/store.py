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
import threading
import time
from pathlib import Path
from typing import Any

from ..integrity.evidence import evidence_outcome

AGENT_STATUSES = ("hypothesis", "probable", "confirmed", "disproven")
# Evidence statuses an agent may request and the verification system may grant.
# "unverified" and "challenged" are authoritative researcher overlays: only a
# researcher action may set or clear them, and agent evidence can never clear
# one by itself (the original claim and its full history stay intact either
# way; fresh evidence is recorded as evidence_status until someone authorised
# resolves the overlay).
STATUSES = AGENT_STATUSES + ("unverified", "challenged")
RESEARCHER_OVERLAYS = ("unverified", "challenged")
GOAL_ORIGINS = ("agent", "assigned", "derived_from_discovery", "carried_forward")
STATUS_TITLES = {
    "confirmed": "Verified discoveries (at least two independently checked supporting experiments)",
    "probable": "Probable explanations (some checked evidence; repeat before relying on them)",
    "hypothesis": "Untested hypotheses (no checked evidence yet; do not treat these as facts)",
    "disproven": "Disproven theories (tested and contradicted by a checked experiment)",
    "unverified": "Marked unverified by the researcher (earlier evidence no longer counts; fresh work is recorded but not reinstated until the researcher re-verifies)",
    "challenged": "Challenged by the researcher (kept for scrutiny; evidence is recorded but nothing becomes confirmed until the researcher resolves the challenge)",
}
GOAL_OPEN = ("active", "paused")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,47}$")


class MemoryError_(Exception):
    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details


_REPLACE_RETRY_BACKOFF_S = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32)


def _write_atomic(path: Path, text: str) -> None:
    """Write-or-replace that is safe under concurrency: a unique temp file per writer (two sessions for
    the same agent used to collide on the shared '<file>.tmp' name and crash with PermissionError), a
    per-path lock so the rename sequence is serialised within the process, and a bounded retry on
    PermissionError — on Windows os.replace fails with Access denied while ANY handle is open on the
    destination (an in-process reader mid-turn, or a Defender/indexer scan of the freshly written
    target). Those locks are transient (the Nov 16 2026 suite hit one on unfinished_goals.md with
    concurrency=2), so retrying resolves them; POSIX replace never raises here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _path_lock(path):
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        for backoff in _REPLACE_RETRY_BACKOFF_S:
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if backoff:
                    time.sleep(backoff)
        os.replace(tmp, path)  # retries exhausted: let the genuine error surface


_path_locks: dict[str, threading.Lock] = {}
_path_locks_guard = threading.Lock()


def _path_lock(path: Path) -> threading.Lock:
    key = str(path).lower()
    with _path_locks_guard:
        lock = _path_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _path_locks[key] = lock
        return lock


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
        # Who is writing memory this session. Stamped onto every item created through the agent-side
        # writes so a record can never again misattribute the scripted reference policy's output to an
        # LLM (audit 2026-11-13, session 19). Set by run_session; None for direct/legacy use.
        self.author: dict | None = None
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
                    or s["goals"] or s["notes"] or self.list_programs() or self.list_networks())

    def _prov(self, session: int) -> dict:
        """The provenance stamp for an item written this session: which brain (adapter/model) produced it.

        Audit 2026-11-13: without this, a resumed session that swapped the adapter wrote items the record
        later attributed to the original brain. `self.author` is set by run_session each session.
        """
        a = self.author or {}
        return {"session": session, "adapter": a.get("adapter"), "model": a.get("model"),
                "source": a.get("source", "agent")}

    # ------------------------------------------------------------ nn/ integration
    def _nn_models_dir(self) -> Path:
        """Return the nn/models/<agent_id>/ directory path."""
        return self.dir / "nn" / "models" / self.agent_id

    def list_networks(self) -> list[dict]:
        """List all neural networks this agent has trained (reads JSON, no torch import)."""
        models_dir = self._nn_models_dir()
        if not models_dir.exists():
            return []
        out = []
        for model_dir in sorted(models_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            versions = sorted(model_dir.glob("v*.json"), key=lambda p: int(p.stem[1:]))
            if not versions:
                continue
            try:
                latest = json.loads(versions[-1].read_text(encoding="utf-8"))
                spec = latest.get("spec", {})
                metrics = latest.get("metrics", {})
                out.append({
                    "name": model_dir.name,
                    "version": latest.get("version", int(versions[-1].stem[1:])),
                    "target": spec.get("target"),
                    "inputs": spec.get("inputs", []),
                    "hidden": spec.get("hidden", []),
                    "test_rmse": metrics.get("test", {}).get("rmse"),
                    "linear_rmse": metrics.get("baseline_linear", {}).get("rmse"),
                    "beats_linear": latest.get("beats_linear_baseline", False),
                    "created_at": latest.get("created_at"),
                })
            except (json.JSONDecodeError, KeyError):
                # Skip malformed model files
                continue
        return out

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
            items = [d for d in self.discoveries["items"]
                     if d["status"] == st and not d.get("flags", {}).get("removed")]
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
                if d.get("evidence_status") and d["status"] in RESEARCHER_OVERLAYS:
                    out.append(f"  - Fresh evidence currently supports '{d.get('evidence_status')}' "
                               f"({d.get('evidence_status_reason') or 'no detail'}), but the authoritative status "
                               "stays as shown until the researcher resolves it.\n")
                for n in d.get("notes", []):
                    out.append(f"  - note (session {n['session']}): {n['text']}\n")
        skipped = sum(1 for d in self.discoveries["items"] if d.get("flags", {}).get("removed"))
        if skipped:
            out.append(f"\n_{skipped} removed by the researcher (tombstoned; history kept in discoveries.json and "
                       "the audit trail)_\n")
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
            label = f"[{g['status']}]" + (f" [{g['kind']}]" if g.get("kind") and g["kind"] != "objective" else "")
            origin = g.get("origin") or "agent"
            origin_label = {"agent": "you", "assigned": "an external task", "derived_from_discovery": "a discovery",
                            "carried_forward": "an earlier session"}.get(origin, origin)
            out.append(f"- **{g['id']}** {label} {g['title']} (set in session {g['origin_session']}, "
                       f"by {origin_label})\n")
            for h in g["history"]:
                if h.get("reason"):
                    out.append(f"  - why (session {h['session']}): {h['reason']}\n")
                    break
            if g.get("description"):
                out.append(f"  - {g['description']}\n")
            if g.get("next_step"):
                out.append(f"  - next step: {g['next_step']}\n")
            for h in g["history"][-3:]:
                if h.get("note"):
                    out.append(f"  - session {h['session']}: {h['note']}\n")
        _write_atomic(self.dir / "unfinished_goals.md", "".join(out))

    def briefing(self) -> str:
        """The text an agent is given at the start of a session: its own files, read from disk.

        Bounded by memory.briefing_char_budget (audit 2026-11-13): the durable parts (handbook, open
        goals, mistakes) are always included; diary, programs, experiments and networks are added while
        there is room, trimming the oldest diary/experiment detail first so a long history cannot crowd
        a small context window.
        """
        if self.is_empty():
            return ("You have no memory files yet. This is your first session on this island, "
                    "so nothing is known in advance.")
        header = (f"YOUR MEMORY FILES ({self.agent_id}; sessions completed so far: "
                  f"{self.state['sessions_completed']}).\n"
                  "You wrote these in earlier sessions. The island has been reset to its original configuration "
                  "since then, so objects are back where they started and anything you built is gone. "
                  "Entries are only as reliable as their stated status.")
        core = [header]
        for name in ("handbook.md", "unfinished_goals.md", "mistakes.md"):
            core.append(f"----- {name} -----\n" + (self.dir / name).read_text(encoding="utf-8").strip())

        budget = int(self.mcfg.get("briefing_char_budget", 0) or 0)
        used = sum(len(p) for p in core)
        optional: list[str] = []

        def fits(block: str) -> bool:
            nonlocal used
            if budget and used + len(block) > budget:
                return False
            used += len(block)
            return True

        def diary_block(n: int) -> str:
            diary = self.state["diary"][-n:]
            lines = [f"- (session {e['session']}, t={e['t']:.1f}s) {e['text']}" for e in diary]
            return "----- diary.md (most recent entries) -----\n" + "\n".join(lines)

        n_diary = int(self.mcfg.get("diary_entries_in_briefing", 12))
        if self.state["diary"]:
            # Shrink the diary window until it fits (keep at least the single most recent entry).
            while n_diary > 1 and budget and used + len(diary_block(n_diary)) > budget:
                n_diary -= 1
            block = diary_block(n_diary)
            if not budget or used + len(block) <= budget:
                used += len(block)
                optional.append(block)

        progs = self.list_programs()
        if progs:
            lines = [f"- {p['name']}: {p.get('description') or 'no description'} ({len(p['steps'])} steps)" for p in progs]
            block = "----- programs/ (saved programs you can run with execute_program) -----\n" + "\n".join(lines)
            if fits(block):
                optional.append(block)
        exps = self.experiments["items"][-6:]
        if exps:
            lines = []
            for e in exps:
                v = e.get("verification", {})
                oc = e.get("outcome") or {}
                check = oc.get("evidence_check") or v.get("verdict", "n/a")
                sci = oc.get("scientific") or ("the agent calls it supported" if e.get("supported") is True else
                                               "the agent says it does not support" if e.get("supported") is False
                                               else "n/a")
                lines.append(f"- {e['id']} (session {e['session']}): {e.get('hypothesis') or e.get('question')} -> "
                             f"{e.get('actual_result')} [evidence check: {check}; scientific conclusion: {sci}]")
            block = "----- experiments.json (most recent) -----\n" + "\n".join(lines)
            if fits(block):
                optional.append(block)
        networks = self.list_networks()
        if networks:
            lines = []
            for n in networks:
                inputs = ", ".join(n.get("inputs", [])[:3])
                if len(n.get("inputs", [])) > 3:
                    inputs += ", ..."
                test_rmse = n.get("test_rmse")
                linear_rmse = n.get("linear_rmse")
                rmse_str = f"test RMSE={test_rmse:.3f}" if test_rmse else "no metrics"
                if linear_rmse:
                    rmse_str += f" (linear: {linear_rmse:.3f})"
                lines.append(f"- {n['name']} v{n['version']}: {n.get('target')} <- [{inputs}] ({rmse_str})")
            block = "----- nn/ (networks you trained; run nn_list / nn_predict) -----\n" + "\n".join(lines)
            if fits(block):
                optional.append(block)
        return "\n\n".join(core + optional)

    # ------------------------------------------------------------ agent-side writes
    def add_diary(self, session: int, t: float, text: str, author: str = "agent") -> dict:
        # Diary entries are immutable history; 'author' records who wrote the line
        # ("agent" for tool calls, "auto" for session-runner auto-captures).
        e = {"id": self._next("Y"), "session": session, "t": round(t, 2), "text": text.strip(),
             "author": author if author in ("agent", "auto") else "agent"}
        self.state["diary"].append(e)
        self.save()
        return e

    def add_note(self, session: int, t: float, text: str) -> dict:
        n = {"id": self._next("N"), "session": session, "t": round(t, 2), "text": text.strip()}
        self.state["notes"].append(n)
        self.save()
        return n

    def add_experiment(self, session: int, t: float, fields: dict, verification: dict,
                       session_id: int | None = None) -> dict:
        n = self.experiments["next_id"]
        self.experiments["next_id"] = n + 1
        e = {"id": f"E-{n:03d}", "session": session, "t": round(t, 2), **fields, "verification": verification,
             # evidence_outcome separates the mechanical evidence check from any
             # scientific conclusion: passing the check never implies the claim
             # itself is true; inconclusive stays inconclusive.
             "outcome": evidence_outcome(verification, fields.get("supported")),
             "agent_id": self.agent_id, "created_at": round(time.time(), 3),
             "author": self._prov(session)}
        if session_id is not None:
            e["session_id"] = session_id
        self.experiments["items"].append(e)
        self.save()
        return e

    def get_experiment(self, eid: str) -> dict | None:
        return next((e for e in self.experiments["items"] if e["id"] == eid), None)

    def get_discovery(self, did: str) -> dict | None:
        return next((d for d in self.discoveries["items"] if d["id"] == did), None)

    def _resolve_experiments(self, ids: list[str], strict_removed: bool = True) -> list[dict]:
        out, missing = [], []
        for i in ids:
            e = self.get_experiment(str(i))
            if e is not None and e.get("flags", {}).get("removed"):
                # Researcher-removed evidence no longer counts; citing it fresh is an error.
                if strict_removed:
                    raise MemoryError_("experiment_removed",
                                       f"Experiment {e['id']} was removed by the researcher and can no longer "
                                       "be cited as evidence.",
                                       known=[x["id"] for x in self.experiments["items"]
                                              if not x.get("flags", {}).get("removed")][-20:])
                continue
            (out if e else missing).append(e or i)
        if missing:
            raise MemoryError_("unknown_experiment", f"Unknown experiment id(s): {', '.join(map(str, missing))}.",
                               known=[e["id"] for e in self.experiments["items"]][-20:])
        return out

    def add_discovery(self, session: int, statement: str, requested: str, experiment_ids: list[str],
                      category: str | None = None, session_id: int | None = None) -> dict:
        if requested not in AGENT_STATUSES:
            raise MemoryError_("invalid_status",
                               f"status must be one of: {', '.join(AGENT_STATUSES)}. "
                               "'unverified' and 'challenged' are researcher states, not choices an agent makes.")
        exps = self._resolve_experiments(experiment_ids or [])
        granted, reason = evaluate_status(requested, exps, self.mcfg)
        n = self.discoveries["next_id"]
        self.discoveries["next_id"] = n + 1
        d = {"id": f"D-{n:03d}", "statement": statement.strip(), "category": category, "status": granted,
             "requested_status": requested, "status_reason": reason, "evidence": [e["id"] for e in exps],
             "created_session": session, "updated_session": session, "flags": {}, "notes": [],
             "agent_id": self.agent_id, "created_at": round(time.time(), 3),
             "author": self._prov(session),
             "history": [{"session": session, "from": None, "to": granted, "requested": requested,
                          "reason": reason, "by": "agent"}]}
        if session_id is not None:
            d["session_id"] = session_id
        self.discoveries["items"].append(d)
        self.save()
        return d

    def update_discovery(self, session: int, did: str, requested: str | None = None,
                         add_experiment_ids: list[str] | None = None, note: str | None = None,
                         session_id: int | None = None) -> dict:
        d = self.get_discovery(did)
        if d is None:
            raise MemoryError_("unknown_discovery", f"There is no discovery '{did}'.",
                               known=[x["id"] for x in self.discoveries["items"]])
        if requested is not None and requested not in AGENT_STATUSES:
            raise MemoryError_("invalid_status",
                               f"status must be one of: {', '.join(AGENT_STATUSES)}. "
                               "'unverified' and 'challenged' are set or cleared only by the researcher.")
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
            exps = [e for e in exps if int(e["id"].split("-")[1]) > cut
                    and not e.get("flags", {}).get("removed")]
        req = requested or d.get("requested_status") or d["status"]
        req_basis = req if req in AGENT_STATUSES else "hypothesis"
        granted, reason = evaluate_status(req_basis, exps, self.mcfg)
        if session_id is not None:
            d["session_id"] = session_id
        d["updated_session"] = session
        if d["status"] in RESEARCHER_OVERLAYS:
            # The authoritative state is the researcher's. Record what the fresh
            # evidence would support, but never change the status and never clear
            # the researcher's overlay from the agent side.
            d["evidence_status"], d["evidence_status_reason"] = granted, reason
            if requested is not None:
                d["requested_status"] = requested
            d["history"].append({"session": session, "from": d["status"], "to": d["status"],
                                 "requested": requested, "requested_status": granted,
                                 "reason": f"fresh evidence checked; would support '{granted}', status remains "
                                           f"'{d['status']}' until a researcher resolves it",
                                 "by": "agent"})
        else:
            if granted != d["status"] or requested:
                d["history"].append({"session": session, "from": d["status"], "to": granted, "requested": req,
                                     "reason": reason, "by": "agent"})
            d.update({"status": granted, "requested_status": req, "status_reason": reason})
            d.pop("evidence_status", None)
            d.pop("evidence_status_reason", None)
        self.save()
        return d

    def add_mistake(self, session: int, t: float, what: str, why: str | None, lesson: str | None,
                    action_refs: list[int] | None = None) -> dict:
        m = {"id": self._next("M"), "session": session, "t": round(t, 2), "what": what.strip(),
             "why": (why or "").strip() or None, "lesson": (lesson or "").strip() or None,
             "action_refs": action_refs or [], "author": self._prov(session)}
        self.state["mistakes"].append(m)
        self.save()
        return m

    def add_goal(self, session: int, t: float, title: str, description: str | None,
                 kind: str | None = None, next_step: str | None = None,
                 origin: str = "agent", reason: str | None = None,
                 session_id: int | None = None) -> dict:
        if origin not in GOAL_ORIGINS:
            raise MemoryError_("invalid_origin", f"origin must be one of: {', '.join(GOAL_ORIGINS)}.")
        g = {"id": self._next("G"), "title": title.strip(), "description": (description or "").strip() or None,
             "kind": kind or "objective", "next_step": (next_step or "").strip() or None,
             "origin": origin, "origin_session": session, "status": "active", "verification": None,
             "agent_id": self.agent_id, "author": self._prov(session),
             "history": [{"session": session, "t": round(t, 2), "status": "active",
                          "origin": origin, "reason": (reason or "").strip() or None,
                          "note": f"goal set by {origin}"}]}
        if session_id is not None:
            g["session_id"] = session_id
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
        # Copy nn/ directory if it exists (model JSONs only, not weights)
        nn_dir = self.dir / "nn"
        if nn_dir.exists():
            shutil.copytree(nn_dir, dest / "nn")
        _write_atomic(dest / "final_world_state.json", json.dumps(final_world_state, indent=2))
        _write_atomic(dest / "summary.json", json.dumps(summary, indent=2, default=str))
        return dest

    # ------------------------------------------------------------ researcher-side operations
    def all_items(self) -> list[dict]:
        out = []
        for d in self.discoveries["items"]:
            out.append({"id": d["id"], "kind": "discovery", "status": d["status"], "text": d["statement"],
                        "session": d["created_session"], "evidence": d["evidence"], "flags": d.get("flags", {}),
                        "agent_id": self.agent_id,
                        "evidence_status": d.get("evidence_status"),
                        "history_len": len(d.get("history", []))})
        for e in self.experiments["items"]:
            out.append({"id": e["id"], "kind": "experiment", "status": e["verification"].get("verdict"),
                        "text": e.get("hypothesis") or e.get("question"), "session": e["session"],
                        "agent_id": self.agent_id, "evidence_check": e.get("verification")
                        and e["verification"].get("verified"),
                        "outcome": e.get("outcome")})
        for key, kind, field in (("mistakes", "mistake", "what"), ("goals", "goal", "title"),
                                 ("diary", "diary", "text"), ("notes", "note", "text")):
            for x in self.state[key]:
                out.append({"id": x["id"], "kind": kind, "status": x.get("status"), "text": x[field],
                            "session": x.get("session", x.get("origin_session"))})
        for p in self.list_programs():
            out.append({"id": f"P-{p['name']}", "kind": "program", "status": None,
                        "text": p.get("description") or p["name"], "session": p["created_session"]})
        # Include neural networks
        for n in self.list_networks():
            out.append({"id": f"NN-{n['name']}", "kind": "network", "status": None,
                        "text": f"{n.get('target')} from {', '.join(n.get('inputs', [])[:3])}",
                        "session": None, "version": n.get("version")})
        return out

    def remove_item(self, item_id: str, reason: str | None = None, actor: str = "researcher",
                    tombstone: bool = True) -> dict:
        """Remove one memory item of any kind.

        Discoveries and experiments are tombstoned (kept in their JSON files with
        a `flags.removed` marker) so the scientific record is never silently
        discarded — the handbook simply stops listing them. Other kinds (notes,
        diary, mistakes, goals, programs, networks) are deleted outright; the
        audit trail still records the deletion. Synthetic copies (benchmarks and
        ablations) pass tombstone=False for a genuine hard remove.
        """
        prefix = item_id.split("-", 1)[0]
        removed = None
        if prefix == "P":
            p = self.dir / "programs" / f"{item_id[2:]}.json"
            if p.exists():
                removed = json.loads(p.read_text(encoding="utf-8"))
                p.unlink()
        elif prefix == "NN":
            # Remove neural network folder
            name = item_id[3:]  # Strip 'NN-' prefix
            model_dir = self._nn_models_dir() / name
            if model_dir.exists():
                # Read the latest version for audit before removing
                versions = sorted(model_dir.glob("v*.json"), key=lambda p: int(p.stem[1:]))
                if versions:
                    try:
                        removed = json.loads(versions[-1].read_text(encoding="utf-8"))
                        removed["_name"] = name
                    except json.JSONDecodeError:
                        removed = {"_name": name, "_error": "could not read model JSON"}
                shutil.rmtree(model_dir)
        else:
            coll = {"D": self.discoveries["items"], "E": self.experiments["items"], "M": self.state["mistakes"],
                    "G": self.state["goals"], "Y": self.state["diary"], "N": self.state["notes"]}.get(prefix)
            if coll is not None:
                for i, x in enumerate(coll):
                    if x["id"] == item_id:
                        if prefix in ("D", "E") and tombstone:
                            # Keep the record: an append-only tombstone preserves
                            # the original claim, its evidence links and history.
                            x["flags"] = {**x.get("flags", {}), "removed": True, "removed_by": actor,
                                          "removed_at": round(time.time(), 3)}
                            if reason:
                                x["flags"]["removal_reason"] = reason
                            if prefix == "D":
                                x["history"].append({"session": None, "from": x["status"], "to": "removed",
                                                     "requested": None,
                                                     "reason": f"removed by {actor}: {reason or 'no reason given'}",
                                                     "by": actor})
                            removed = {**x, "_tombstoned": True}
                        else:
                            removed = coll.pop(i)
                        break
        if removed is None:
            raise MemoryError_("unknown_item", f"There is no memory item '{item_id}' for {self.agent_id}.")
        if prefix == "E":
            # A discovery cannot keep a status that rested on evidence which no
            # longer counts. The tombstoned id stays in the evidence list (the
            # citation is history); evaluation simply ignores removed evidence.
            for d in self.discoveries["items"]:
                if d.get("flags", {}).get("removed"):
                    continue
                if item_id in d["evidence"]:
                    exps = [e for e in (self.get_experiment(i) for i in d["evidence"])
                            if e and not e.get("flags", {}).get("removed")]
                    granted, why = evaluate_status(d.get("requested_status") or d["status"], exps, self.mcfg)
                    if d["status"] in RESEARCHER_OVERLAYS:
                        d["evidence_status"], d["evidence_status_reason"] = granted, why
                    elif granted != d["status"]:
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
        now = round(time.time(), 3)
        h = d["history"]
        h.append({"session": None, "from": before["status"], "to": "unverified",
                  "requested": None, "reason": reason or "marked unverified by researcher",
                  "by": actor, "at": now, "kind": "mark_unverified"})
        d["status"] = "unverified"
        d["status_reason"] = "marked unverified by the researcher; earlier evidence no longer counts, " \
                             "fresh work is recorded but the discovery is not reinstated until re-verification"
        d["flags"].update({"marked_unverified": True, "unverified_at_experiment": last, "unverified_by": actor,
                           "unverified_at": now})
        self._audit(actor, "mark_unverified", did, before, d, reason)
        self.save()
        return d

    def reverify(self, did: str, reason: str | None = None, actor: str = "researcher",
                 status: str | None = None) -> dict:
        """Researcher action: end an 'unverified' overlay.

        By default the resulting authoritative status is what the fresh evidence
        (recorded after the unverified cutoff) currently supports; the researcher
        may also pin an explicit status. Every unverified-then-reverified cycle
        stays in the discovery's history.
        """
        d = self.get_discovery(did)
        if d is None:
            raise MemoryError_("unknown_discovery", f"There is no discovery '{did}'.")
        if d["status"] != "unverified":
            raise MemoryError_("not_unverified",
                               f"Discovery '{did}' is '{d['status']}', not 'unverified'; nothing to re-verify.")
        if status is not None and status not in AGENT_STATUSES:
            raise MemoryError_("invalid_status",
                               f"status must be one of: {', '.join(AGENT_STATUSES)}.")
        before = json.loads(json.dumps(d))
        exps = [self.get_experiment(i) for i in d["evidence"]]
        if d.get("flags", {}).get("marked_unverified"):
            cut = d["flags"].get("unverified_at_experiment", 0)
            exps = [e for e in exps if e and int(e["id"].split("-")[1]) > cut
                    and not e.get("flags", {}).get("removed")]
        granted, why = evaluate_status(d.get("requested_status") or "hypothesis", exps, self.mcfg)
        target = status or granted
        now = round(time.time(), 3)
        d["flags"] = {k: v for k, v in d.get("flags", {}).items()
                      if k not in ("marked_unverified", "unverified_at_experiment", "unverified_by",
                                   "unverified_at", "unverified_prev_status")}
        d["history"].append({"session": None, "from": "unverified", "to": target,
                             "requested": None, "reason": reason or
                             f"re-verified by researcher; fresh evidence supports '{granted}'",
                             "by": actor, "at": now, "kind": "reverify",
                             "evidence_status": granted})
        d["status"] = target
        d["status_reason"] = f"re-verified by researcher ({reason or 'no reason given'}); {why}"
        d.pop("evidence_status", None)
        d.pop("evidence_status_reason", None)
        self._audit(actor, "reverify", did, before, d, reason)
        self.save()
        return d

    def challenge(self, did: str, reason: str, actor: str = "researcher") -> dict:
        d = self.get_discovery(did)
        if d is None:
            raise MemoryError_("unknown_discovery", f"There is no discovery '{did}'.")
        before = json.loads(json.dumps(d))
        now = round(time.time(), 3)
        f = d["flags"]
        if d["status"] != "challenged":
            f["challenge_prev_status"] = before["status"]
        f.update({"challenged": True, "challenge_reason": reason, "challenge_by": actor, "challenge_at": now})
        f["challenge_count"] = int(f.get("challenge_count", 0)) + 1
        d["history"].append({"session": None, "from": before["status"], "to": "challenged",
                             "requested": None, "reason": reason, "by": actor, "at": now,
                             "kind": "challenge"})
        d["status"] = "challenged"
        d["status_reason"] = f"challenged by {actor}: {reason}"
        self._audit(actor, "challenge", did, before, d, reason)
        self.save()
        return d

    def resolve_challenge(self, did: str, reason: str | None = None, actor: str = "researcher",
                          status: str | None = None) -> dict:
        """Researcher action: end a 'challenged' overlay.

        Default outcome: the discovery returns to the status the (fresh) evidence
        currently supports — which may be weaker than the challenged claim. The
        researcher may pin any explicit evidence status. The challenge record,
        its reason and the outcome stay in the history for good.
        """
        d = self.get_discovery(did)
        if d is None:
            raise MemoryError_("unknown_discovery", f"There is no discovery '{did}'.")
        if d["status"] != "challenged":
            raise MemoryError_("not_challenged", f"Discovery '{did}' is '{d['status']}', not 'challenged'.")
        if status is not None and status not in AGENT_STATUSES:
            raise MemoryError_("invalid_status", f"status must be one of: {', '.join(AGENT_STATUSES)}.")
        before = json.loads(json.dumps(d))
        exps = [e for e in (self.get_experiment(i) for i in d["evidence"])
                if e and not e.get("flags", {}).get("removed")]
        granted, why = evaluate_status(d.get("requested_status") or
                                       (d.get("flags", {}).get("challenge_prev_status") or "hypothesis"),
                                       exps, self.mcfg)
        target = status or (d.get("evidence_status") if d.get("evidence_status") in AGENT_STATUSES else granted)
        now = round(time.time(), 3)
        f = d["flags"]
        d["flags"] = {k: v for k, v in f.items()
                      if k not in ("challenged", "challenge_reason", "challenge_by", "challenge_at",
                                   "challenge_prev_status", "challenge_count")}
        d["history"].append({"session": None, "from": "challenged", "to": target,
                             "requested": None, "reason": reason or
                             f"challenge resolved by researcher; evidence supports '{granted}'",
                             "by": actor, "at": now, "kind": "resolve_challenge",
                             "evidence_status": granted})
        d["status"] = target
        d["status_reason"] = f"challenge resolved by researcher ({reason or 'no reason given'}); {why}"
        d.pop("evidence_status", None)
        d.pop("evidence_status_reason", None)
        self._audit(actor, "resolve_challenge", did, before, d, reason)
        self.save()
        return d

    def set_status(self, did: str, status: str, reason: str | None = None,
                   actor: str = "researcher") -> dict:
        """Researcher action: explicitly set the authoritative status of a discovery.

        The only way a status becomes 'confirmed'/'disproven' outside the evidence
        rule or 'unverified'/'challenged' outside the overlay methods — and it is
        always recorded in the discovery's history and the audit trail.
        """
        d = self.get_discovery(did)
        if d is None:
            raise MemoryError_("unknown_discovery", f"There is no discovery '{did}'.")
        if status not in STATUSES:
            raise MemoryError_("invalid_status", f"status must be one of: {', '.join(STATUSES)}.")
        before = json.loads(json.dumps(d))
        now = round(time.time(), 3)
        f = d["flags"]
        if status in AGENT_STATUSES:
            d["flags"] = {k: v for k, v in f.items()
                          if k not in ("challenged", "challenge_reason", "challenge_by", "challenge_at",
                                       "challenge_prev_status", "challenge_count", "marked_unverified",
                                       "unverified_at_experiment", "unverified_by", "unverified_at",
                                       "unverified_prev_status")}
            d.pop("evidence_status", None)
            d.pop("evidence_status_reason", None)
        d["history"].append({"session": None, "from": before["status"], "to": status,
                             "requested": None, "reason": reason or "status set by researcher",
                             "by": actor, "at": now, "kind": "set_status"})
        d["status"] = status
        d["status_reason"] = f"status set by researcher ({reason or 'no reason given'})"
        self._audit(actor, "set_status", did, before, d, reason)
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
            # Ablation copies are synthetic workspaces: genuinely hard-remove
            # (no tombstones) so benchmark agents never see the removed items.
            other.remove_item(item, reason=reason, actor="benchmark", tombstone=False)
        other.save()
        return other

    def stats(self) -> dict:
        by = {s: 0 for s in STATUSES}
        items = self.discoveries["items"]
        for d in items:
            if d.get("flags", {}).get("removed"):
                continue
            by[d["status"]] += 1
        exps = self.experiments["items"]
        removed = sum(1 for x in items if x.get("flags", {}).get("removed")) + \
            sum(1 for e in exps if e.get("flags", {}).get("removed"))
        return {"agent_id": self.agent_id, "sessions_completed": self.state["sessions_completed"],
                "discoveries": by, "discoveries_total": len(items),
                "experiments": len(exps),
                "experiments_verified": sum(1 for e in exps if e["verification"].get("verified")),
                "experiments_contradicted": sum(1 for e in exps if e["verification"].get("verdict") == "contradicted"),
                "mistakes": len(self.state["mistakes"]), "diary_entries": len(self.state["diary"]),
                "goals_open": sum(1 for g in self.state["goals"] if g["status"] in GOAL_OPEN),
                "goals_total": len(self.state["goals"]), "programs": len(self.list_programs()),
                "challenged": by["challenged"],
                "records_removed": removed,
                "networks": len(self.list_networks())}
