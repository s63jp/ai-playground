"""Initial catalog of 16 physical skills.

Each entry is a ``SkillSpec``. All sixteen dispatch through the agent adapter
(``playground/skills/agent_adapter.py``) onto the EXISTING world implementations;
none introduces new actuation. ``controller.defaults`` fills parameters the caller
does not give (explicit arguments always win).
"""
from __future__ import annotations
from .spec import SkillSpec


def _base(id_suffix: str, name: str, category: str, method: str,
          defaults: dict | None = None) -> SkillSpec:
    controller = {"method": method}
    if defaults:
        controller["defaults"] = dict(defaults)
    return SkillSpec(
        id=f"skill-{id_suffix}",
        name=name,
        category=category,
        inputs={"params": {}},
        outputs={"result": "dict"},
        preconditions={},
        success_criteria={"status": "succeeded"},
        failure_conditions={"status": "failed"},
        controller=controller,
        status="available",
        version="1.1.0",
        metrics={},
    )


standing = _base("standing", "Standing", "locomotion", "stand", {"seconds": 2.0})
walking = _base("walking", "Walking", "locomotion", "set_gait", {"mode": "walk", "distance": 0.5})
running = _base("running", "Running", "locomotion", "set_gait", {"mode": "walk", "speed": 2.8, "distance": 1.0})
turning = _base("turning", "Turning", "locomotion", "turn")
jumping = _base("jumping", "Jumping", "locomotion", "jump")
landing = _base("landing", "Landing", "locomotion", "jump_landing")
balancing = _base("balancing", "Balancing", "stability", "balance")
falling = _base("falling", "Falling", "stability", "agent_fallen")
recovery = _base("recovery", "Recovery", "stability", "stand_up")
reaching = _base("reaching", "Reaching", "manipulation", "reach")
grasping = _base("grasping", "Grasping", "manipulation", "pick_up")
lifting = _base("lifting", "Lifting", "manipulation", "lift")
carrying = _base("carrying", "Carrying", "manipulation", "carry", {"distance": 0.5})
throwing = _base("throwing", "Throwing", "manipulation", "throw", {"force": 60.0})
catching = _base("catching", "Catching", "manipulation", "catch")
climbing = _base("climbing", "Climbing", "locomotion", "climb")

ALL_SKILLS = [
    standing, walking, running, turning, jumping, landing,
    balancing, falling, recovery, reaching, grasping,
    lifting, carrying, throwing, catching, climbing,
]
