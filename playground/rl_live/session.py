"""LiveBrainSession — one genuinely untrained network learning to walk, in
front of the user, driven by the ONE PPO trainer.

What "live" means here (the honest version):
- a FRESH TanhGaussianActor/Critic is built when the user starts a session;
- training runs through the real `PPOTrainer` (the B3 algorithm; no second
  implementation and no modifications to it — pause/stop ride its
  `should_stop` callback, which fires at iteration boundaries, i.e. at most
  one 2048-decision rollout later, and telemetry rides `on_iteration`);
- the env the trainer steps is the real `LocoEnv` wrapped in the read-only
  `FrameTapEnv`, so what the viewer shows IS the network being trained.

State machine: idle → starting → running ⇄ paused → stopped | error.
`stop` saves a final checkpoint; `save` is manual and only allowed while
paused/stopped (torch state must not be read mid-update).

Only one live session per sidecar process, by construction.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .interop import build_session_env, build_session_trainer, save_session_checkpoint
from .tap import FrameTapEnv

# First supported objective (B3's ladder); more objectives later register here.
WALKING_STAGE_BUDGETS = {"standing": 200000, "controlled": 500000}
OBJECTIVES = {
    "walking": {
        "label": "Walking (standing → controlled → sustained)",
        "stage_budgets": dict(WALKING_STAGE_BUDGETS),
        "env_overrides": {},
    },
}

_ALLOWED_PPO_KEYS = ("lr", "ent_coef", "clip_eps", "gamma", "gae_lambda",
                     "rollout_steps", "epochs", "minibatch", "max_grad_norm", "hidden")

ITERATION_CACHE = 200


@dataclass
class SessionConfig:
    name: str
    objective: str
    seed: int
    total_steps: int
    stage_budgets: dict = field(default_factory=dict)
    ppo_overrides: dict = field(default_factory=dict)
    env_overrides: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class LiveBrainSession:
    """Owns the training thread for one live run.

    Injected dependencies (all optional, defaults are the real ones):
      trainer_factory  -> interop.build_session_trainer
      saver            -> interop.save_session_checkpoint
      broadcaster      -> callable(kind: str, payload: dict) for server events
      frame_sink       -> callable(frame_dict) for viewer frames (the bus)
      root             -> Path for run dirs (default <repo>/experiments/rl_live)
    """

    def __init__(self, *, trainer_factory=None, saver=None, broadcaster=None,
                 frame_sink=None, root=None):
        self._trainer_factory = trainer_factory or build_session_trainer
        self._saver = saver or save_session_checkpoint
        self._broadcast = broadcaster or (lambda kind, payload: None)
        self._frame_sink = frame_sink
        self.root = Path(root) if root else Path(__file__).resolve().parents[2] \
            / "experiments" / "rl_live"
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()        # resume / phase events
        self._paused = False
        self._stopping = False
        self.state = "idle"                    # idle|starting|running|paused|stopped|error
        self._error: str | None = None
        self.config: SessionConfig | None = None
        self._bundle: dict | None = None       # trainer/actor/critic/ppo cfg
        self._curriculum = None
        self._tap: FrameTapEnv | None = None
        self._iterations: list[dict] = []
        self._stage: str | None = None
        self._progress_file = None
        self._run_dir: Path | None = None
        self._saved: list[dict] = []
        from . import interop
        self._trainer_available = bool(trainer_factory is not None or
                                       interop.trainer_available())
        if not self._trainer_available:
            # honest display: the default interop path cannot work on a tree
            # without the B3 modules; injected factories (tests/hosts) bypass
            self.state = "error"
            self._error = ("B3 trainer modules (playground/rl/ppo.py, "
                           "playground/rl/checkpoint.py) are not on this "
                           "tree — live training starts once the B3 commit "
                           "lands")

    # ------------------------------------------------------------- inquiries
    def status(self) -> dict:
        with self._lock:
            last = self._iterations[-1] if self._iterations else None
            tap = self._tap
            return {
                "session": {
                    "state": self.state,
                    "error": self._error,
                    "stage": self._stage,
                    "config": self.config.as_dict() if self.config else None,
                    "run_dir": str(self._run_dir) if self._run_dir else None,
                    "iteration": last.get("iteration") if last else 0,
                    "total_valid_steps": last.get("total_valid_steps", 0) if last else 0,
                    "last_iteration": last,
                    "checkpoints": list(self._saved[-10:]),
                    "live_frames": tap.live_frames if tap else 0,
                    "dropped_frames": tap.dropped_frames if tap else 0,
                },
                "trainer_available": self._trainer_available,
            }

    def recent_iterations(self, limit: int = 200) -> list[dict]:
        with self._lock:
            return list(self._iterations[-limit:])

    def busy(self) -> bool:
        return self.state in ("starting", "running", "paused")

    # ------------------------------------------------------------- controls
    def start(self, *, objective="walking", name=None, seed=None,
              total_steps=None, ppo_overrides=None) -> dict:
        with self._lock:
            if self.busy():
                return {"ok": False, "error": {"code": "busy",
                        "message": "a live session is already active"}}
            if not self._trainer_available:
                return {"ok": False, "error": {"code": "trainer_unavailable",
                        "message": "the B3 trainer modules are not on this "
                                   "tree; live training starts once the B3 "
                                   "commit lands"}}
            preset = OBJECTIVES.get(objective)
            if preset is None:
                return {"ok": False, "error": {"code": "unknown_objective",
                        "message": f"objectives available: {sorted(OBJECTIVES)}"}}
            try:
                seed = int(seed) if seed is not None else 20261118
                total_steps = int(total_steps) if total_steps is not None else 2_000_000
                if total_steps <= 0 or seed < 0:
                    raise ValueError
            except (TypeError, ValueError):
                return {"ok": False, "error": {"code": "invalid_config",
                        "message": "seed must be an int >= 0 and total_steps > 0"}}
            overrides = dict(ppo_overrides or {})
            bad = set(overrides) - set(_ALLOWED_PPO_KEYS)
            if bad:
                return {"ok": False, "error": {"code": "invalid_config",
                        "message": f"unknown ppo keys: {sorted(bad)}"}}
            name = (name or f"live_{time.strftime('%Y%m%d_%H%M%S')}").strip()
            if "/" in name or "\\" in name or name in ("", ".", ".."):
                return {"ok": False, "error": {"code": "invalid_config",
                        "message": "run name must be a plain folder name"}}
            self.config = SessionConfig(
                name=name, objective=objective, seed=seed,
                total_steps=total_steps,
                stage_budgets=dict(preset["stage_budgets"]),
                ppo_overrides=overrides,
                env_overrides=dict(preset.get("env_overrides") or {}))
            self._paused = False
            self._stopping = False
            self._error = None
            self._iterations = []
            self._saved = []
            self.state = "starting"
            self._wake.clear()
            t = threading.Thread(target=self._run, name="rl_live_session",
                                 daemon=True)
            self._thread = t
            t.start()
        self._broadcast("status", self._status_payload())
        return {"ok": True, "config": self.config.as_dict()}

    def pause(self) -> dict:
        with self._lock:
            if self.state != "running":
                return {"ok": False, "error": {"code": "not_running",
                        "message": f"state is {self.state}"}}
            self._paused = True
        return {"ok": True, "state": self.state,
                "note": "pause applies at the current rollout boundary "
                        "(up to one rollout later)"}

    def resume(self) -> dict:
        with self._lock:
            if self.state != "paused":
                return {"ok": False, "error": {"code": "not_paused",
                        "message": f"state is {self.state}"}}
            self._paused = False
            self._wake.set()
        return {"ok": True}

    def stop(self) -> dict:
        with self._lock:
            if not self.busy():
                return {"ok": False, "error": {"code": "not_active",
                        "message": f"state is {self.state}"}}
            self._stopping = True
            self._paused = False
            self._wake.set()
        return {"ok": True}

    def save(self, tag: str | None = None) -> dict:
        with self._lock:
            bundle = self._bundle
            cur = self._curriculum
            cfg = self.config
            run_dir = self._run_dir
            if not (bundle and cfg and run_dir):
                return {"ok": False, "error": {"code": "nothing_to_save",
                        "message": "no session has been started on this server"}}
            if self.state not in ("paused", "stopped"):
                return {"ok": False, "error": {"code": "save_requires_paused",
                        "message": f"saving is allowed while paused or stopped "
                                   f"(state is {self.state})"}}
            steps = trainer_steps(bundle)
            fname = f"ckpt_{tag}.json" if tag else f"ckpt_{steps:09d}.json"
            if "/" in fname or "\\" in fname:
                return {"ok": False, "error": {"code": "invalid_config",
                        "message": "tag must be a plain filename fragment"}}
        try:
            result = self._saver(run_dir / fname, trainer=bundle["trainer"],
                                 curriculum=cur, seed=cfg.seed, name=cfg.name,
                                 total_steps=cfg.total_steps,
                                 stage_budgets=cfg.stage_budgets,
                                 ppo_cfg_dict=bundle.get("ppo_cfg_dict"))
        except Exception as exc:                      # saver may be unbindable
            return {"ok": False, "error": {"code": "save_failed",
                    "message": str(exc)}}
        with self._lock:
            self._saved.append({"path": result["path"],
                                "steps": steps,
                                "bytes": result.get("bytes")})
            saved_now = dict(self._saved[-1])
        self._broadcast("saved", saved_now)
        return {"ok": True, "checkpoint": saved_now}

    # ------------------------------------------------------------ internals
    def _status_payload(self) -> dict:
        return self.status()["session"]

    def _run(self):
        try:
            env, cur = build_session_env(self.config.env_overrides)
            tap = FrameTapEnv(env, sink=self._frame_sink or (lambda m: None))
            run_dir = self.root / self.config.name
            run_dir.mkdir(parents=True, exist_ok=True)
            self._run_dir = run_dir
            self._tap = tap

            def on_iteration(rec: dict):
                with self._lock:
                    self._iterations.append(rec)
                    self._iterations = self._iterations[-ITERATION_CACHE:]
                    if rec.get("stage"):
                        self._stage = rec["stage"]
                self._write_iteration(run_dir, rec)
                self._broadcast("iteration", rec)
                if rec.get("stage_transition_to"):
                    self._broadcast("stage_transition",
                                    {"from_stage": rec.get("stage"),
                                     "to_stage": rec["stage_transition_to"],
                                     "iteration": rec.get("iteration"),
                                     "total_valid_steps": rec.get("total_valid_steps")})

            def should_stop():
                return self._paused or self._stopping

            bundle = self._trainer_factory(
                tap, cur, name=self.config.name, seed=self.config.seed,
                total_steps=self.config.total_steps,
                stage_budgets=self.config.stage_budgets,
                ppo_overrides=self.config.ppo_overrides,
                on_iteration=on_iteration, should_stop=should_stop)
            self._bundle = bundle
            self._curriculum = cur
            (run_dir / "run_config.json").write_text(json.dumps(
                {**self.config.as_dict(), "ppo": bundle.get("ppo_cfg_dict")},
                indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            self._fail(exc)
            return
        if self._frame_sink is not None:
            # one scene spec per session start; the model is unchanged after that
            from .frames import scene_spec
            try:
                self._broadcast("spec", scene_spec(tap._env))
            except Exception:
                self._broadcast("spec", {"spec_version": 0, "bodies": [],
                                         "meshes_skipped": 0,
                                         "geoms_skipped": 0, "joints": [],
                                         "policy_joints": [], "notes": [],
                                         "sim": {}, "world_size_hint": 30.0})
        with self._lock:
            self.state = "running" if not self._paused else "paused"
        self._broadcast("status", self._status_payload())
        trainer = self._bundle["trainer"]
        try:
            while True:
                if self._stopping:
                    break
                if self._paused:
                    with self._lock:
                        self.state = "paused"
                    self._broadcast("status", self._status_payload())
                    while self._paused and not self._stopping:
                        self._wake.wait(timeout=0.5)
                    if self._stopping:
                        break
                    with self._lock:
                        self.state = "running"
                    self._broadcast("status", self._status_payload())
                    continue
                trainer.train()   # returns when should_stop() fires or the total is reached
                if trainer.total_valid_steps >= self.config.total_steps:
                    self._write_final_checkpoint(reason="complete")
                    with self._lock:
                        self.state = "stopped"
                        self._stopping = True
                    self._broadcast("status", self._status_payload())
                    return
        except Exception as exc:
            self._fail(exc)
            return
        # stop path: final checkpoint, then leave the thread
        try:
            self._write_final_checkpoint(reason="stopped_by_user")
        except Exception as exc:
            self._fail(exc)
            return
        with self._lock:
            self.state = "stopped"
        self._broadcast("status", self._status_payload())

    def _write_final_checkpoint(self, *, reason: str) -> None:
        bundle, cfg, cur, run_dir = self._bundle, self.config, self._curriculum, self._run_dir
        if not (bundle and cfg and run_dir):
            return
        try:
            result = self._saver(
                run_dir / "ckpt_final.json", trainer=bundle["trainer"],
                curriculum=cur, seed=cfg.seed, name=cfg.name,
                total_steps=cfg.total_steps,
                stage_budgets=cfg.stage_budgets,
                ppo_cfg_dict=bundle.get("ppo_cfg_dict"))
        except Exception as exc:
            self._error = f"final checkpoint failed: {exc}"
            self._broadcast("status", self._status_payload())
            return
        with self._lock:
            self._saved.append({"path": result["path"],
                                "steps": trainer_steps(bundle),
                                "bytes": result.get("bytes"),
                                "reason": reason})
        self._broadcast("saved", dict(self._saved[-1]))

    def _write_iteration(self, run_dir: Path, rec: dict) -> None:
        path = run_dir / "progress.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def _fail(self, exc: Exception) -> None:
        with self._lock:
            self.state = "error"
            self._error = f"{type(exc).__name__}: {exc}"
        self._broadcast("status", self._status_payload())


def trainer_steps(bundle: dict) -> int:
    t = bundle.get("trainer")
    return int(getattr(t, "total_valid_steps", 0) or 0)
