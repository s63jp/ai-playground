
"""Initial catalog of 16 physical skills.
Each entry is a ``SkillSpec``. For capabilities not yet implemented the
status is ``unavailable``.
"""
from __future__ import annotations
import pathlib
from .spec import SkillSpec

def _base(id_suffix: str, name: str, category: str, method: str, status: str) -> SkillSpec:
    return SkillSpec(
        id=f"skill-{id_suffix}",
        name=name,
        category=category,
        inputs={"params": {}},
        outputs={"result": "dict"},
        preconditions={},
        success_criteria={"status": "succeeded"},
        failure_conditions={"status": "failed"},
        controller={"method": method},
        status=status,
        version="1.0.0",
        metrics={}
    )

# Available capabilities (assuming world provides set_gait, stand_up, etc.)
standing = _base("standing", "Standing", "locomotion", "stand_up", "available")
walking = _base("walking", "Walking", "locomotion", "set_gait", "available")
running = _base("running", "Running", "locomotion", "set_gait", "available")
turning = _base("turning", "Turning", "locomotion", "set_gait", "available")
jumping = _base("jumping", "Jumping", "locomotion", "set_gait", "available")
landing = _base("landing", "Landing", "locomotion", "set_gait", "unavailable")
balancing = _base("balancing", "Balancing", "stability", "set_gait", "available")
falling = _base("falling", "Falling", "stability", "agent_fallen", "available")
recovery = _base("recovery", "Recovery", "stability", "stand_up", "available")
reaching = _base("reaching", "Reaching", "manipulation", "reach", "available")
grasping = _base("grasping", "Grasping", "manipulation", "grasp", "available")
lifting = _base("lifting", "Lifting", "manipulation", "pick_up", "unavailable")
carrying = _base("carrying", "Carrying", "manipulation", "pick_up", "unavailable")
throwing = _base("throwing", "Throwing", "manipulation", "throw", "available")
catching = _base("catching", "Catching", "manipulation", "catch", "unavailable")
climbing = _base("climbing", "Climbing", "locomotion", "set_gait", "unavailable")

ALL_SKILLS = [
    standing, walking, running, turning, jumping, landing,
    balancing, falling, recovery, reaching, grasping,
    lifting, carrying, throwing, catching, climbing,
]
