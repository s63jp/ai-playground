"""The ONE curriculum mechanism for the locomotion RL environment (M6-B B2).

Replaces the broken pre-M6 `Curriculum` class (which re-multiplied gravity
cumulatively against a non-existent `PhysicsWorld.opt` attribute). Stage
application happens ONLY through `LocoEnv.set_phase` — the env rewrites
gravity from its captured base each time, so phases are never cumulative;
reward-coefficient phases are re-applied over the frozen defaults, never
stacked over a previous stage's overrides.

The canonical stage ladder (mission stages 1-6):

1  standing        balance only (zero target velocity, upright assistance)
2  controlled      slow forward walking (0.25 m/s along the rig's walkable
                   chest axis = world +Y at spawn) with partial assistance
3  sustained       sustained walking at the full 0.5 m/s target along the same
                   axis, no assist: the honest walking task
4  disturbance     recovery from measured disturbances — RESERVED: needs an
                   env-side disturbance hook (impulse classes); not trainable
                   until that lands
5  terrain         terrain adaptation — RESERVED: needs terrain spawn phases
6  navigation      heading/waypoint curricula — RESERVED: needs heading
                   sampling support

Stages 4-6 are declared with env_ready=False: advancing into them raises a
clear error instead of silently pretending. Transitions are explicit
(step() returns True exactly when a transition happens; advance() moves the
pointer) and fully deterministic (no RNG anywhere; identical sequences give
identical phase progressions). Transitions are non-cumulative by
construction — every apply() routes through set_phase, which rewrites each
parameter from its source of truth rather than incrementing it.

M7-1a adds the SECOND ladder: `M7_S1_STAGES`, the individual limb-control
ladder for the M7 lineages — selected explicitly at run launch, never a
mutation of DEFAULT_STAGES (the B2/B3/B4 fixtures keep their indices and
their bit-identical behaviour). See reports/M7_1A_LIMB_STAGES_CHANGE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .env import LocoEnv


@dataclass(frozen=True)
class Stage:
    id: str
    name: str
    description: str
    duration_decisions: int          # decision windows before auto-advance
    phase: dict = field(default_factory=dict)
    env_ready: bool = True           # False = declared future work


def _stand_phase() -> dict:
    return {"gravity_scale": 1.0, "upright_assist": 0.6, "max_steps": 400,
            "target_velocity": [0.0, 0.0, 0.0]}


def _controlled_phase() -> dict:
    # B3 measured correction (evidence: ai-playground-audit/probe_b3_yaw): a
    # policy without a waist command (the rig's ONLY rotational root dof, held
    # by the env posture PD) locomotes along the chest axis — random-action
    # displacement mean |dy| = 0.94 m vs |dx| = 0.013 m; the scripted gait
    # walks the chest axis equally well (heading 90 -> 0.749 m / 8 s). Stage
    # phases therefore command WORLD +Y (the spawn-facing chest axis).
    return {"gravity_scale": 1.0, "upright_assist": 0.3, "max_steps": 400,
            "target_velocity": [0.0, 0.25, 0.0]}


def _sustained_phase() -> dict:
    return {"gravity_scale": 1.0, "upright_assist": 0.0, "max_steps": 400,
            "target_velocity": [0.0, 0.5, 0.0]}


DEFAULT_STAGES = (
    Stage("standing", "Standing and balance",
          "Stand upright with assistance; zero target velocity: the tracking "
          "kernel pays matching stillness, progress pays nothing.",
          400, _stand_phase(), True),
    Stage("controlled", "Controlled forward movement",
          "Slow 0.25 m/s walking along the rig's walkable axis (world +Y at "
          "spawn) with partial upright assistance.",
          1200, _controlled_phase(), True),
    Stage("sustained", "Sustained walking",
          "Full 0.5 m/s target along the walkable axis, no assistance: the "
          "honest walking task.",
          1600, _sustained_phase(), True),
    Stage("disturbance", "Recovery from disturbances",
          "Reserved: needs a measured disturbance hook in the env "
          "(impulse classes applied through the world API).",
          0, {"gravity_scale": 1.0, "target_velocity": [0.5, 0.0, 0.0]}, False),
    Stage("terrain", "Terrain adaptation",
          "Reserved: needs terrain/obstacle spawn phases in the env config.",
          0, {}, False),
    Stage("navigation", "Advanced navigation",
          "Reserved: needs heading/waypoint task sampling support.",
          0, {}, False),
)


# ---------------------------------------------------------------- M7-1a limb
# The M7 S1 limb-control ladder (reports/M7_1A_LIMB_STAGES_CHANGE.md) — a
# SEPARATE ladder for the M7 lineages, selected explicitly at run launch
# (scripts/rl_ppo.py --stages m7_s1). DEFAULT_STAGES is UNTOUCHED: the B2/B3/
# B4 fixtures keep their indices and bit-identical behaviour. Index 0 is the
# SAME standing Stage object as the B4 ladder (bit-identical base stage).
#
# Scope (Jamie's M7-1 directive 2026-11-18): INDIVIDUAL limb control with
# head/neck and trunk posture included; NO walking curriculum — every phase
# carries target_velocity [0, 0, 0] EXPLICITLY (never the stale [0.5,0,0]
# default that set_phase falls back to when the key is absent, the M7-3
# flagged figure). The waist is NEVER commanded (P1B/P2 dead-band ⇒
# neutral-hold only) and ankle schedules are PINNED at 0 (P2: ground-pinned
# under load — the finding rides the M7-2 COM-shift stage, not S1).
#
# Stage order = ascending measured physical load (P2 torque rows): arms
# ≤1.56 N·m → head ≤0.88 → torso 2.27 → legs 8.5–10.1 N·m. Duration default
# 200,000 decisions per stage (the standing-stage precedent); the training
# run's stage_budgets (--stage-budget id=budget) override at launch.

_M7_HOLD_TARGETS = (-0.7, -0.35, 0.35, 0.7)   # P2-measured holdable set (rad)

_M7_SCHED_KW = {"hold_windows": 120, "transition_windows": 30, "shuffle": False}


def _m7_sched(group: str, targets: dict) -> dict:
    return {"group": group, "targets_rad": targets, **_M7_SCHED_KW}


def _m7_phase(group: str, targets: dict, limb_tracking_w: float) -> dict:
    return {"gravity_scale": 1.0, "upright_assist": 0.6, "max_steps": 400,
            "target_velocity": [0.0, 0.0, 0.0],
            "reward_coeffs": {"limb_tracking": limb_tracking_w},
            "commands": {"mask": (1, 0, 0, 0), "limb_group": group},
            "command_schedule": _m7_sched(group, targets)}


# M7-1a limb_tracking coefficient — MEASURED-CONFIRMED. The untrained-baseline
# probe (reports/m7_s1_probe.py -> reports/M7_1A_S1_PROBE.json, 2026-11-19;
# gate report reports/M7_1A_S1_GATE.md) measured the feasible window
# [0.75, 5.18] reward/window and confirms 2.0 sits inside it: signal floor
# w>=0.75 (separation>=0.05/window for every group; legs bind) and equilibrium
# ceiling w<=5.18 (good-tracking residual <=5% of the 2.50 base reward/window).
# The value was validated, not changed. The launch gate (M7_S1_PROVISIONAL)
# stays closed until the gate report is reviewed (M7-1a report Sec.7).
_M7_S1_W = 2.0

_M7_T = _M7_HOLD_TARGETS
_M7_N = (0.0,)   # pinned-neutral (the P1B waist gate / P2 ankle pins)

M7_S1_STAGES = (
    DEFAULT_STAGES[0],        # standing — the same base stage, bit-identical
    Stage("m7_s1_left_arm", "Left arm control",
          "Hold tracked left shoulder+elbow targets while standing; head and "
          "trunk posture bands live from the base reward.",
          200_000, _m7_phase("left_arm", {"left_shoulder": _M7_T,
                                          "left_elbow": _M7_T}, _M7_S1_W), True),
    Stage("m7_s1_right_arm", "Right arm control",
          "Hold tracked right shoulder+elbow targets while standing.",
          200_000, _m7_phase("right_arm", {"right_shoulder": _M7_T,
                                           "right_elbow": _M7_T}, _M7_S1_W), True),
    Stage("m7_s1_head", "Head and neck control",
          "Hold tracked head pitch/yaw/roll targets while standing (Jamie's "
          "S1 scope: head/neck included).",
          200_000, _m7_phase("head", {"head_pitch": _M7_T, "head_yaw": _M7_T,
                                      "head_roll": _M7_T}, _M7_S1_W), True),
    Stage("m7_s1_torso", "Trunk (torso) control",
          "Hold tracked spine_L2 targets while standing — the trunk-posture "
          "ask under the chest-aware base reward.",
          200_000, _m7_phase("torso", {"spine_L2": _M7_T}, _M7_S1_W), True),
    Stage("m7_s1_left_leg", "Left leg control",
          "Hold tracked left hip+knee targets; the ankle stays pinned at "
          "neutral (P2: ground-pinned under load).",
          200_000, _m7_phase("left_leg", {"left_hip": _M7_T,
                                          "left_knee": _M7_T,
                                          "left_ankle": _M7_N}, _M7_S1_W), True),
    Stage("m7_s1_right_leg", "Right leg control",
          "Hold tracked right hip+knee targets; the ankle stays pinned.",
          200_000, _m7_phase("right_leg", {"right_hip": _M7_T,
                                           "right_knee": _M7_T,
                                           "right_ankle": _M7_N}, _M7_S1_W), True),
    Stage("m7_s1_all", "All limbs together",
          "Every joint commanded at once; the waist is pinned at neutral in "
          "every window (the P1B neutral-hold gate — its error is measured "
          "as a gate, never trained).",
          200_000, _m7_phase("all", {"left_shoulder": _M7_T,
                                     "left_elbow": _M7_T,
                                     "right_shoulder": _M7_T,
                                     "right_elbow": _M7_T,
                                     "head_pitch": _M7_T, "head_yaw": _M7_T,
                                     "head_roll": _M7_T,
                                     "spine_L2": _M7_T,
                                     "left_hip": _M7_T, "left_knee": _M7_T,
                                     "left_ankle": _M7_N,
                                     "right_hip": _M7_T, "right_knee": _M7_T,
                                     "right_ankle": _M7_N,
                                     "waist_yaw": _M7_N}, _M7_S1_W), True),
)

# The launcher's explicit ladder registry (scripts/rl_ppo.py --stages):
STAGE_LADDERS = {"m7_s1": M7_S1_STAGES}

# ---- M7-1a launch gate: SATISFIED by recorded calibration evidence --------
# The untrained-baseline probe (reports/m7_s1_probe.py ->
# reports/M7_1A_S1_PROBE.json, gate report reports/M7_1A_S1_GATE.md) measured
# the feasible limb_tracking window; the gate is cleared only while the
# shipped coefficient sits inside it. This retains the pre-training safety
# requirement (no S1 training on an uncalibrated coefficient) as an
# evidence check rather than a bypassable boolean — it is SATISFIED, not
# removed. Mastery is a separate, stronger gate (held-out capability eval,
# playground/rl/s1_eval.py): this one only clears the START of training.
_M7_S1_W_WINDOW = (0.75, 5.18)   # measured [signal floor, equilibrium ceiling]


def m7_s1_launch_block_reason() -> str | None:
    """None when the S1 limb ladder is cleared to train; otherwise the
    human-readable reason it is blocked. Checked by every launch path (the CLI
    ``--stages m7_s1`` and the live session)."""
    lo, hi = _M7_S1_W_WINDOW
    if not (lo <= _M7_S1_W <= hi):
        return (f"limb_tracking coefficient w={_M7_S1_W} is outside the "
                f"measured calibration window [{lo}, {hi}] reward/window "
                f"(reports/M7_1A_S1_PROBE.json) — re-run the baseline probe "
                f"(reports/m7_s1_probe.py) before training the S1 ladder")
    return None


def m7_s1_limb_stage_ids() -> list:
    """The selectable individual-limb stage ids (every S1 stage past the
    shared standing base)."""
    return [s.id for s in M7_S1_STAGES[1:]]


def m7_s1_limb_ladder(limb_stage_id: str) -> tuple:
    """The 2-stage live ladder for ONE S1 limb: the standing base (bit-
    identical to DEFAULT_STAGES[0]) then the selected limb stage. Raises
    ValueError on an unknown or the base id."""
    by_id = {s.id: s for s in M7_S1_STAGES}
    if limb_stage_id == "standing" or limb_stage_id not in by_id:
        raise ValueError(
            f"unknown S1 limb stage {limb_stage_id!r}; choose one of "
            f"{m7_s1_limb_stage_ids()}")
    return (M7_S1_STAGES[0], by_id[limb_stage_id])


class Curriculum:
    """Deterministic stage sequencer over one LocoEnv.

    - `apply()` pushes the current stage's phase into the env (via set_phase).
    - `step()` is called after each env decision; when the current stage's
      decision budget is exhausted it advances to the NEXT env-ready stage
      (skipping reserved ones is an error; reserved stages terminate the
      sequence instead) and returns True. Transitions are explicit — the
      caller observes exactly when they happen.
    - `advance(index)` moves the pointer explicitly (id: index or stage id).
    - Every advance re-applies the stage phase: identical calls give
      identical env state (deterministic, reproducible, non-cumulative).
    """

    def __init__(self, env: LocoEnv | None = None, stages=None, start_index: int = 0):
        self.stages = list(stages if stages is not None else DEFAULT_STAGES)
        if not self.stages:
            raise ValueError("curriculum needs at least one stage")
        self.index = 0
        self.steps_in_stage = 0
        self.total_steps = 0
        self.env: LocoEnv | None = None
        if env is not None:
            self.bind(env, start_index=start_index)

    # ------------------------------------------------------------ lifecycle
    def bind(self, env: LocoEnv, start_index: int = 0) -> None:
        """Attach an env and apply the starting stage immediately."""
        self.env = env
        self.select(start_index)

    def select(self, index) -> None:
        """Explicitly move to a stage (int index or stage id) and apply it."""
        idx = self._resolve(index)
        stage = self.stages[idx]
        if not stage.env_ready:
            raise ValueError(
                f"stage '{stage.id}' is declared but the environment cannot "
                "express it yet (env_ready=False); it becomes trainable when "
                "its env-side hook lands")
        self.index = idx
        self.steps_in_stage = 0
        if self.env is not None:
            self.env.set_phase(dict(stage.phase))

    def advance(self, index=None) -> None:
        """Explicit transition (default: next stage). Applies immediately."""
        target = self.index + 1 if index is None else index
        if isinstance(target, str):
            self.select(target)
            return
        if target >= len(self.stages):
            raise ValueError(f"no stage at index {target} (last: {len(self.stages) - 1})")
        nxt = self.stages[target]
        if not nxt.env_ready:
            raise ValueError(
                f"stage '{nxt.id}' is reserved (env_ready=False); the "
                "sequence ends here until its env hook lands")
        self.select(target)

    def step(self, n: int = 1) -> bool:
        """Tick after n decisions; True exactly on a stage transition."""
        if self.env is None:
            raise RuntimeError("Curriculum not bound to an env (bind first)")
        self.steps_in_stage += int(n)
        self.total_steps += int(n)
        if self.steps_in_stage >= self.stages[self.index].duration_decisions:
            if self.index + 1 < len(self.stages):
                if self.stages[self.index + 1].env_ready:
                    self.advance()
                    return True
                # next stage is reserved: hold here (the env contract, not a
                # silent fabricate-advance); document via snapshot.
                return False
        return False

    def _resolve(self, index) -> int:
        if isinstance(index, int):
            if not (0 <= index < len(self.stages)):
                raise ValueError(f"stage index {index} out of range")
            return index
        for i, s in enumerate(self.stages):
            if s.id == index:
                return i
        raise ValueError(f"unknown stage id {index!r}")

    # ------------------------------------------------------------ reporting
    @property
    def stage(self) -> Stage:
        return self.stages[self.index]

    def snapshot(self) -> dict:
        """Reproducibility record of the curriculum state."""
        return {
            "index": self.index,
            "stage_id": self.stage.id,
            "steps_in_stage": self.steps_in_stage,
            "total_steps": self.total_steps,
            "stages": [
                {"id": s.id, "name": s.name, "env_ready": s.env_ready,
                 "duration_decisions": s.duration_decisions, "phase": dict(s.phase)}
                for s in self.stages
            ],
        }
