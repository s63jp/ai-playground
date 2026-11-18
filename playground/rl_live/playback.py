"""Checkpoint playback — watch a saved trained brain act in the real viewer.

A dedicated READ-ONLY surface for replaying a verified training checkpoint
(the B3 two-million-step brain) through a real MuJoCo simulation into the
existing live-viewer pipeline. This module is a playback adapter, not a
trainer:

- the checkpoint is loaded through THE `playground.rl.checkpoint` loader
  (content-SHA-256 tamper check + full obs/action contract fingerprint), and
  its optimiser state is discarded immediately: nothing here can update any
  weight. There is NO import of `ppo.PPOTrainer` anywhere in this module;
- the actor hash (every float32 parameter, log_std included) is recomputed
  after EVERY episode and broadcast in the status payload — a byte change
  would be visible and would abort the session;
- the rig is the B3-ERA humanoid asset: the launcher extracts it from the
  repository's own git history (commit e7e81ac), refuses the bytes unless
  SHA-256 matches the recorded value, copies it into the playback run dir
  and points `playground.world.spec.HUMANOID_ASSET_PATH` at the copy BEFORE
  any world is built. The default asset path is never modified on disk; the
  compiled model of every episode is re-checked against the expected B3 head
  inertial (6.370626302704502 kg @ +0.26 m) before a single decision runs.
- inference modes are explicit and labelled: "deterministic" (a = tanh(mu(s)),
  the recorded held-out evaluation convention) is the primary demonstration;
  "stochastic" samples z = mu + sigma*eps with sigma exactly as trained.
  A mode switch takes effect at the next episode start and is broadcast as
  `mode_next` until then.
- frames stream from THE same simulation the actor controls, through the
  existing read-only `FrameTapEnv` -> bus -> SSE path the live viewers
  already consume (`scene_spec` once per session, `frame` per decision,
  `reset`/`episode_end` per episode).

Runs are isolated on their own port (default 8770) with their own run dir
(default experiments/rl_playback/<name>/); the checkpoint's run dir
(experiments/rl/b3_run1) is only ever READ, never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# ---- B3-era rig identity (recorded once, verified every start) ----
B3_ASSET_GIT_REF = "e7e81ac"
B3_ASSET_GIT_PATH = "assets/humanoid_mujoco.xml"
B3_ASSET_SHA256 = "1d00ea2b6fee73139ccd7c5432fd5d22fb35d2515907d33991a04dda5bbe9496"
# Compiled from that exact asset (audit: b3_playback_audit_head_inertial.json):
B3_HEAD = {"mass": 6.370626302704502,
           "ipos": (0.0, 0.0, 0.26),
           "inertia": (0.03370061314130682, 0.03370061314130682,
                       0.03370061314130682)}

STAGES = {"standing": 0, "controlled": 1, "sustained": 2}
MODES = ("deterministic", "stochastic")
SPEED_MIN, SPEED_MAX = 0.25, 4.0
EVAL_SEED_BASE = 97_010_000      # the held-out evaluation seed space

SERVER_VERSION = "rl_playback/1"


class PlaybackError(RuntimeError):
    """Playback cannot proceed honestly (rig, checkpoint or mode problem)."""


# ------------------------------------------------------------ the B3-era rig
def extract_b3_rig_xml(dest_dir: Path) -> dict:
    """Extract the B3-era asset from git history, verify its SHA-256, copy it
    under the playback run dir and return {"path", "sha256", "source"}.

    Playback never reads `assets/` for its own rig: the copy is the one the
    seam points at, and its bytes are re-verified against the recorded hash
    on every session start.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "humanoid_mujoco_b3_era.xml"
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "show",
         f"{B3_ASSET_GIT_REF}:{B3_ASSET_GIT_PATH}"],
        capture_output=True, timeout=30)
    if proc.returncode != 0:
        raise PlaybackError(
            f"cannot extract the B3-era rig from git ({B3_ASSET_GIT_REF}"
            f":{B3_ASSET_GIT_PATH}): "
            f"{proc.stderr.decode(errors='replace').strip()}")
    raw = proc.stdout
    sha = hashlib.sha256(raw).hexdigest()
    if sha != B3_ASSET_SHA256:
        raise PlaybackError(
            "B3-era asset hash mismatch: expected "
            f"{B3_ASSET_SHA256}, git gave {sha}; refusing to build a rig")
    dest.write_bytes(raw)
    return {"path": str(dest), "sha256": sha,
            "source": f"git {B3_ASSET_GIT_REF}:{B3_ASSET_GIT_PATH}"}


def activate_b3_rig(rig_path: str) -> str:
    """Point the world builder at the verified B3-era copy (this process
    only). Returns the previous path — the caller MUST restore it when its
    session ends: the seam is a process-global constant and a leaked B3 rig
    would silently change every later humanoid build in the same process
    (this exact leak once broke the B1 combat test in the full suite)."""
    from ..world import spec as world_spec
    previous = str(world_spec.HUMANOID_ASSET_PATH)
    world_spec.HUMANOID_ASSET_PATH = rig_path
    return previous


def rig_fingerprint(m) -> dict:
    """Fingerprint of the compiled humanoid rig (head identity + counts)."""
    bid = int(m.body("head").id)
    return {
        "head_mass_kg": float(m.body_mass[bid]),
        "head_ipos": [float(x) for x in np.asarray(m.body_ipos[bid])],
        "head_inertia": [float(x) for x in np.asarray(m.body_inertia[bid])],
        "nbody": int(m.nbody), "njnt": int(m.njnt),
    }


def verify_head_inertial(m) -> dict:
    """Assert the compiled rig carries the B3-era head; raises PlaybackError."""
    fp = rig_fingerprint(m)
    if fp["head_mass_kg"] != B3_HEAD["mass"] \
            or any(float(a) != float(b) for a, b in zip(fp["head_ipos"],
                                                        B3_HEAD["ipos"])) \
            or any(float(a) != float(b) for a, b in zip(fp["head_inertia"],
                                                        B3_HEAD["inertia"])):
        raise PlaybackError(
            f"compiled rig is NOT the B3-era head: measured {fp}; "
            f"expected {B3_HEAD}")
    return fp


# ------------------------------------------------------------ actor identity
def actor_sha256(actor) -> str:
    """Stable digest of every actor parameter (float32 bytes, sorted keys).

    Recomputed after every episode and at session end; any weight change —
    including log_std — changes this value and aborts playback. This is the
    read-only proof."""
    h = hashlib.sha256()
    sd = actor.state_dict()
    for k in sorted(sd):
        h.update(k.encode())
        arr = np.ascontiguousarray(sd[k].detach().cpu().numpy()
                                   .astype(np.float32))
        h.update(arr.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- the session
class PlaybackSession:
    """One playback thread re-playing the checkpoint's actor in LocoEnv.

    Same wiring contract as LiveBrainSession: the server's App assigns
    `_frame_sink` (the bus) and `_broadcast` (server events) after
    construction; `start()` launches the thread. Status exposes the
    checkpoint/rig identity, the labelled inference mode, speeds and
    per-episode measurements; `playback_control` is the server's
    /api/rl_playback/* dispatch target.
    """

    def __init__(self, *, checkpoint: str, stage: str = "sustained",
                 phase: dict | None = None, inference: str = "deterministic",                 seed_base: int = EVAL_SEED_BASE, speed: float = 1.0,
                 name: str | None = None, root: str | Path | None = None,
                 env_overrides: dict | None = None,
                 broadcaster=None, frame_sink=None):
        self._broadcast = broadcaster or (lambda kind, payload: None)
        self._frame_sink = frame_sink            # the App sets the bus here
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()
        ck_path = Path(checkpoint)
        if not ck_path.exists():
            raise PlaybackError(f"no checkpoint at {ck_path}")
        # a KNOWN stage id without an explicit phase selects that stage's
        # phase; an explicit phase always decides behaviour, so a known stage
        # id + a phase dict is a configuration mistake (loud, silent-never)
        if stage in STAGES and phase is not None:
            raise PlaybackError("pass either a stage id or an explicit "
                                "phase dict, not both")
        if stage not in STAGES and (phase is None or not isinstance(phase, dict)):
            raise PlaybackError(f"unknown stage {stage!r} (known: {STAGES}); "
                                f"stages: standing|controlled|sustained")
        if inference not in MODES:
            raise PlaybackError(f"inference mode must be one of {MODES}")
        self._checkpoint_path = str(ck_path.resolve())
        self._stage = stage                    # display label; phase decides
        self._phase = phase                      # explicit (tests) or None
        self._mode = inference                   # running episode's mode
        self._mode_next: str | None = None        # pending mode switch
        self._seed_base = int(seed_base)
        self._speed = float(min(max(float(speed), SPEED_MIN), SPEED_MAX))
        self._env_overrides = dict(env_overrides or {})
        self.name = name or f"playback_{ck_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.root = Path(root) if root else ROOT / "experiments" / "rl_playback"
        self._run_dir: Path | None = None
        self.state = "idle"          # idle|starting|running|paused|stopped|error
        self._paused = False
        self._error: str | None = None
        self._identity: dict = {}
        self._totals = {"episodes_started": 0, "completed": 0,
                        "falls": 0, "faulted": 0}
        self._episode_records: list[dict] = []
        self._last_episode: dict | None = None
        self._reset_req = False
        self._stopping = False

    # ------------------------------------------------------------- inquiries
    def status(self) -> dict:
        with self._lock:
            return {
                "session": {
                    "state": self.state,
                    "error": self._error,
                    "server_version": SERVER_VERSION,
                    "stage": self._stage,
                    "mode": self._mode,
                    "mode_next": self._mode_next,
                    "speed": self._speed,
                    "episode": self._totals["episodes_started"],
                    "config": {
                        "checkpoint": self._checkpoint_path,
                        "stage": self._stage,
                        "inference": self._mode,
                        "seed_base": self._seed_base,
                        "name": self.name,
                        "speed": self._speed,
                    },
                    "identity": self._identity,
                    "totals": dict(self._totals),
                    "last_episode": self._last_episode,
                    "episodes": list(self._episode_records[-5:]),
                    "run_dir": str(self._run_dir) if self._run_dir else None,
                    "is_playback": True,
                },
                "trainer_available": True,
            }

    def recent_iterations(self, limit: int = 200) -> list[dict]:
        """Playback has no PPO iterations; the latest episode records are the
        closest analog (the playback page reads its identity from status)."""
        with self._lock:
            return list(self._episode_records[-int(limit):])

    def busy(self) -> bool:
        return self.state in ("starting", "running", "paused")

    # ------------------------------------------------------------- controls
    def start(self) -> dict:
        with self._lock:
            if self.state != "idle":
                return {"ok": False, "error": {"code": "busy",
                        "message": "playback has already been started"}}
            self.state = "starting"
            self._wake.clear()
            t = threading.Thread(target=self._run, name="rl_playback",
                                 daemon=True)
            self._thread = t
            t.start()
        self._broadcast("status", self.status()["session"])
        return {"ok": True, "name": self.name}

    def stop(self) -> dict:
        return self.playback_control("stop")

    def playback_control(self, op: str, body: dict | None = None) -> dict:
        """The /api/rl_playback/* dispatch target (server route seam)."""
        body = body or {}
        with self._lock:
            if op == "pause":
                if self.state != "running":
                    return {"ok": False, "error": {"code": "not_running",
                            "message": f"state is {self.state}"}}
                self._paused = True
                self._wake.clear()
                self.state = "paused"
            elif op == "resume":
                if self.state != "paused" or not self._paused:
                    return {"ok": False, "error": {"code": "not_paused",
                            "message": f"state is {self.state} "
                                       f"(paused={self._paused})"}}
                self._paused = False
                self._wake.set()
                self.state = "running"
            elif op == "reset":
                if self.state not in ("running", "paused"):
                    return {"ok": False, "error": {"code": "not_running",
                            "message": f"state is {self.state}"}}
                self._reset_req = True          # consumed at the next
                # decision-loop check (on resume, if parked mid-episode)
            elif op == "speed":
                try:
                    v = float(body.get("speed"))
                except (TypeError, ValueError):
                    return {"ok": False, "error": {"code": "invalid_config",
                            "message": "speed must be a number"}}
                if not (SPEED_MIN <= v <= SPEED_MAX):
                    return {"ok": False, "error": {"code": "invalid_config",
                            "message": f"speed must be in "
                                       f"[{SPEED_MIN}, {SPEED_MAX}]"}}
                self._speed = v
            elif op == "mode":
                want = body.get("mode")
                if want not in MODES:
                    return {"ok": False, "error": {"code": "invalid_config",
                            "message": f"mode must be one of {MODES}"}}
                self._mode_next = want          # takes effect at next episode
            elif op == "stop":
                if not self.busy():
                    return {"ok": False, "error": {"code": "not_active",
                            "message": f"state is {self.state}"}}
                self._stopping = True
                self._paused = False
                self._reset_req = False
                self._wake.set()
            else:
                return {"ok": False, "error": {"code": "not_found",
                        "message": f"unknown playback op {op!r}"}}
        st = self.status()["session"]
        self._broadcast("status", st)
        return {"ok": True, **{k: st[k] for k in
                               ("state", "mode", "mode_next", "speed")}}

    # ------------------------------------------------------------- internals
    def _status_payload(self) -> dict:
        return self.status()["session"]

    def _fail(self, exc: Exception) -> None:
        import traceback
        with self._lock:
            self.state = "error"
            detail = traceback.format_exc(limit=10)
            self._error = f"{type(exc).__name__}: {exc}\n{detail}"
        self._broadcast("status", self._status_payload())

    def _record_episode(self, tap, env, *, seed: int, mode: str,
                        reset_requested: bool, wall_s: float) -> dict:
        """One measured episode row from the env's own engine summary.

        The reward column carries an explicit non-comparability note: the
        current tree's reward pipeline is the post-B4 chest-aware rule, while
        the checkpoint was trained under the B3-era rule."""
        # noqa: E501 — _summary_with_axis is the recorded eval's own metric
        from playground.rl.eval import _summary_with_axis
        s = _summary_with_axis(env)
        warnings_now = dict(env.world.physics_warnings())
        rec = {
            "t": "playback_episode",
            "episode": tap.episode_count,
            "seed": seed,
            "mode": mode,
            "falls": int(s.get("falls", 0)),
            "faulted_windows": int(s.get("faulted_windows", 0)),
            "physics_warnings": sorted(w for w in warnings_now
                                       if warnings_now[w]
                                       > self._warn_baseline.get(w, 0)),
            "axis_displacement_m": s.get("axis_displacement_m"),
            "net_displacement_m": s.get("net_displacement_m"),
            "mean_speed_m_s": s.get("mean_speed_m_s"),
            "sim_time_s": s.get("sim_time_s"),
            "steps": s.get("steps"),
            "truncated": s.get("truncated_at") is not None,
            "upright_fraction": s.get("upright_fraction"),
            "min_up_z": s.get("min_up_z"),
            "effort_total_j": s.get("effort_total_j"),
            "reward_total": s.get("reward_total"),
            "reward_total_note": ("current-tree pipeline (post-B4 chest-aware "
                                  "rule); NOT comparable to B3-era rewards"),
            "reset_requested": bool(reset_requested),
            "wall_s": round(float(wall_s), 3),
            "actor_sha256": actor_sha256(self._actor),
        }
        self._warn_baseline = dict(warnings_now)
        sha = rec["actor_sha256"]
        with self._lock:
            self._identity["actor_sha256"] = sha
            self._identity["actor_unmodified"] = (sha == self._actor_sha_start)
            if sha != self._actor_sha_start:
                raise PlaybackError(
                    "ACTOR WEIGHTS CHANGED DURING PLAYBACK — aborting "
                    f"({sha} != {self._actor_sha_start})")
        return rec

    def _run(self):
        import torch
        torch.set_num_threads(1)   # same convention as the trainer
        prev_seam = None          # the verified rig seam; restored on any exit
        try:
            from .frames import scene_spec
            from .tap import FrameTapEnv
            from playground.rl.env import LocoEnv
            from playground.rl.checkpoint import load_checkpoint
            from playground.rl.curriculum import DEFAULT_STAGES

            self._run_dir = self.root / self.name
            self._run_dir.mkdir(parents=True, exist_ok=True)
            # ---- the B3-era rig: extract -> SHA-verify -> seam -> build ----
            rig = extract_b3_rig_xml(self._run_dir / "rig")
            prev_seam = activate_b3_rig(rig["path"])   # restored in finally
            # env_overrides are RL-BLOCK keys (rl.*); LocoEnv takes the
            # already-merged block (load_config deep-merges the rl section
            # wholesale, so a flat {action_repeat: n} override would be
            # silently ignored — the exact silent-mis-config this project
            # refuses)
            from playground.config import load_config
            base_cfg = load_config(None, overrides={})
            rl = dict(base_cfg.get("rl") or {})
            rl.update(self._env_overrides)
            env = LocoEnv(cfg={"agent_body": {"type": "humanoid"}, "rl": rl})
            # Explicit phase: the named stage's phase, or the caller's dict
            # (tests). set_phase is the one non-cumulative appliance.
            phase = (dict(self._phase) if self._phase is not None
                     else dict(DEFAULT_STAGES[STAGES[self._stage]].phase))
            env.set_phase(dict(phase))
            fp0 = verify_head_inertial(env.world.m)
            self._identity["rig"] = {
                **rig,
                "compiled_head_mass_kg": fp0["head_mass_kg"],
                "compiled_head_ipos": fp0["head_ipos"],
                "compiled_head_inertia": fp0["head_inertia"],
                "verified": True,
            }
            # ---- the checkpoint, through THE loader, read-only ----
            ck = load_checkpoint(self._checkpoint_path, env=env)
            ck.pop("optimizer_actor", None)   # exists in the file; playback
            ck.pop("optimizer_critic", None)  # discards it: no update path
            actor = ck["actor"]
            self._actor = actor
            self._actor_sha_start = actor_sha256(actor)
            meta = ck["meta"]
            log_std = np.asarray(meta["policy"]["log_std"], dtype=np.float64)
            self._identity.update({
                "checkpoint_name": Path(self._checkpoint_path).name,
                "content_sha256": ck["content_sha256"],
                "trained_steps": meta.get("training_steps"),
                "run": meta.get("run"),
                "seed": meta.get("seed"),
                "stage_at_save": meta.get("stage"),
                "obs_dim": int(meta["contract"]["obs_dim"]),
                "action_dim": int(meta["contract"]["action_dim"]),
                "contract_version": int(meta["contract"]["contract_version"]),
                "actor_sha256_start": self._actor_sha_start,
                "actor_sha256": self._actor_sha_start,
                "actor_unmodified": True,
                "sigma_mean": float(np.exp(log_std).mean()),
                "sigma_range": [float(np.exp(log_std).min()),
                                float(np.exp(log_std).max())],
                "mode": self._mode,
                "era_note": ("B3-era rig: git e7e81ac assets/humanoid_mujoco "
                             "(geom-derived head 6.3706 kg @ +0.26 m), "
                             "SHA-verified every start"),
                "phase": phase,
            })
            (self._run_dir / "playback_info.json").write_text(json.dumps(
                {"identity": self._identity, "server_version": SERVER_VERSION},
                indent=2, default=str), encoding="utf-8")
            self._warn_baseline = dict(env.world.physics_warnings())
        except Exception as exc:
            self._fail(exc)
            # a failed setup must not leak the process-global rig seam
            if prev_seam is not None:
                try:
                    from ..world import spec as world_spec
                    world_spec.HUMANOID_ASSET_PATH = prev_seam
                except Exception:
                    pass
            return
        self._prev_seam = prev_seam

        # ---- streaming: the SAME env the driver steps, through the tap ----
        ep_log = (self._run_dir / "episodes.jsonl").open("a", encoding="utf-8")
        self._ep_log = ep_log
        tap = FrameTapEnv(env, sink=self._frame_sink or (lambda m: None))
        self._tap = tap
        try:
            self._broadcast("spec", scene_spec(env))
        except Exception as exc:
            self._fail(exc)
            return
        with self._lock:
            self.state = "running"
        self._broadcast("status", self._status_payload())

        ep_idx = 0
        try:
            while not self._stopping:
                seed = self._seed_base + ep_idx
                reset_req = False
                # apply a pending mode switch at this episode boundary
                with self._lock:
                    if self._mode_next is not None:
                        self._mode = self._mode_next
                        self._mode_next = None
                    mode = self._mode
                # NOTE: a pending `reset` flag is deliberately NOT cleared
                # here — its one consumer is the decision-loop check below, so
                # a press that arrives between episodes is consumed at the
                # next episode's first decision and replays that episode's
                # seed from the start (a press is never silently lost)
                # reproducible sampling RNG per episode (stochastic); the
                # actor's parameters are never touched by this
                torch.manual_seed((int(seed) * 1_000_003 + 0x9E37) % (2 ** 31 - 1))
                obs = tap.reset(seed)
                verify_head_inertial(env.world.m)     # per-reset rig guard
                self._totals["episodes_started"] += 1
                t0 = time.monotonic()
                dt_window = (float(env.action_repeat)
                             * float(env.world.m.opt.timestep))
                while True:
                    if self._paused:
                        while self._paused and not self._stopping:
                            self._wake.wait(timeout=0.5)
                        if self._stopping:
                            break
                    if self._stopping:
                        break
                    with self._lock:
                        if self._reset_req:
                            self._reset_req = False
                            reset_req = True
                            break                      # restart this episode
                    it_start = time.monotonic()
                    if mode == "deterministic":
                        a = self._actor.deterministic_action(obs)
                    else:
                        a, _z, _lp = self._actor.sample(obs, deterministic=False)
                    obs, _r, term, trunc, info = tap.step(a)
                    delay = dt_window / max(self._speed, 1e-6) \
                        - (time.monotonic() - it_start)
                    if delay > 0:
                        time.sleep(delay)
                    if term or trunc:
                        break
                if reset_req and env.step_count == 0:
                    # a reset pressed at the very start of an episode: nothing
                    # was measured (zero decisions, no physics), so no record
                    # row is written — the replay starts immediately
                    continue
                rec = self._record_episode(
                    tap, env, seed=seed, mode=mode, reset_requested=reset_req,
                    wall_s=time.monotonic() - t0)
                with self._lock:
                    self._episode_records.append(rec)
                    self._episode_records = self._episode_records[-50:]
                    self._last_episode = rec
                    self._totals["falls"] += int(rec["falls"])
                    self._totals["faulted"] += int(rec["faulted_windows"])
                    if rec["truncated"]:
                        self._totals["completed"] += 1
                try:
                    ep_log.write(json.dumps(rec, default=str) + "\n")
                    ep_log.flush()
                except Exception:
                    pass
                if reset_req:
                    continue                 # replay the same seed
                ep_idx += 1
        except Exception as exc:
            self._fail(exc)
            return
        finally:
            close_it = getattr(self, "_ep_log", None)
            if close_it is not None:
                try:
                    close_it.close()
                except Exception:
                    pass
            # restore the process-global rig seam — a leaked B3-era path
            # would silently change every later humanoid build in this
            # process (the recorded B1-combat full-suite failure)
            try:
                from ..world import spec as world_spec
                world_spec.HUMANOID_ASSET_PATH = self._prev_seam
            except Exception:
                pass
            # final read-only proof: did the actor survive the whole session?
            try:
                end_sha = actor_sha256(self._actor)
                with self._lock:
                    self._identity["actor_sha256_end"] = end_sha
                    self._identity["actor_unmodified"] = \
                        (end_sha == self._identity["actor_sha256_start"])
            except Exception:
                pass
        with self._lock:
            self.state = "stopped"
        self._broadcast("status", self._status_payload())


# ---------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stage", default="sustained", choices=sorted(STAGES))
    ap.add_argument("--inference", default="deterministic", choices=list(MODES))
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-root", default=str(ROOT / "experiments" / "rl_playback"))
    ap.add_argument("--frame-hz", type=float, default=30.0)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="loopback only by policy; any other value is refused")
    args = ap.parse_args(argv)
    if args.bind not in ("127.0.0.1", "localhost", "::1"):
        ap.error("refusing to bind anything but loopback (project policy)")
    from .bus import Bus
    from .server import App, LiveHTTPServer
    session = PlaybackSession(checkpoint=args.checkpoint, stage=args.stage,
                              inference=args.inference, speed=args.speed,
                              seed_base=args.seed_base, name=args.name,
                              root=args.out_root)
    bus = Bus(frame_hz=args.frame_hz)
    # App wires the session's _frame_sink (the bus) and _broadcast itself.
    app = App(session=session, bus=bus, b3=None, out_root=Path(args.out_root),
              frame_hz=args.frame_hz, refresher=False)
    start = session.start()
    if not start.get("ok"):
        raise SystemExit(f"playback failed to start: {start}")
    httpd = LiveHTTPServer(("127.0.0.1", args.port), app)
    print(f"[rl_playback] {SERVER_VERSION} on "
          f"http://127.0.0.1:{httpd.server_address[1]}/playback.html",
          flush=True)
    print(f"[rl_playback] checkpoint: {args.checkpoint}", flush=True)
    print("[rl_playback] read-only: no PPO updates; the checkpoint's run dir "
          "is never written; the live-training sidecar on :8766 is a separate "
          "process and is untouched", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
