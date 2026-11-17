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
