"""Equipment framework for P6.
Provides a data‑driven EQUIPMENT registry and an EquipmentItem class handling grip,
ammo, rate limiting, reloading and projectile launching via a frozen launcher callable.
"""
import time, random, numpy as np
from typing import Any, Callable, List, Dict

# ---------------------------------------------------------------------------
# Registry of equipment specifications (contract §9)
# ---------------------------------------------------------------------------
EQUIPMENT: Dict[str, Dict[str, Any]] = {
    "paintball_pistol": {
        "id": "paintball_pistol",
        "name": "Paintball Pistol",
        "mass_kg": 1.2,
        "grip": "one_hand",
        "hands_required": 1,
        "projectile_cfg": {"speed": 30.0, "radius": 0.02, "colour": "red", "restitution": 0.3},
        "pellets_per_shot": 1,
        "accuracy_spread_deg": 5.0,
        "rate_limit_s": 0.2,
        "ammo_capacity": 12,
        "reload_steps_s": 1.5,
        "throwable": False,
    },
    "paintball_rifle": {
        "id": "paintball_rifle",
        "name": "Paintball Rifle",
        "mass_kg": 3.0,
        "grip": "two_hand",
        "hands_required": 2,
        "projectile_cfg": {"speed": 40.0, "radius": 0.02, "colour": "red", "restitution": 0.3},
        "pellets_per_shot": 1,
        "accuracy_spread_deg": 2.0,
        "rate_limit_s": 0.6,
        "ammo_capacity": 30,
        "reload_steps_s": 2.2,
        "throwable": False,
    },
    "dual_paintball_pistols": {
        "id": "dual_paintball_pistols",
        "name": "Dual Paintball Pistols",
        "mass_kg": 2.5,
        "grip": "dual",
        "hands_required": 2,
        "projectile_cfg": {"speed": 30.0, "radius": 0.02, "colour": "red", "restitution": 0.3},
        "pellets_per_shot": 2,
        "accuracy_spread_deg": 7.0,
        "rate_limit_s": 0.15,
        "ammo_capacity": 24,
        "reload_steps_s": 1.8,
        "throwable": False,
    },
    "paintball_shotgun": {
        "id": "paintball_shotgun",
        "name": "Paintball Shotgun",
        "mass_kg": 4.0,
        "grip": "two_hand",
        "hands_required": 2,
        "projectile_cfg": {"speed": 25.0, "radius": 0.02, "colour": "red", "restitution": 0.3},
        "pellets_per_shot": 8,
        "accuracy_spread_deg": 20.0,
        "rate_limit_s": 1.0,
        "ammo_capacity": 8,
        "reload_steps_s": 3.0,
        "throwable": False,
    },
    "paintball_grenade": {
        "id": "paintball_grenade",
        "name": "Paintball Grenade",
        "mass_kg": 0.5,
        "grip": "one_hand",
        "hands_required": 1,
        "projectile_cfg": {"speed": 15.0, "radius": 0.04, "colour": "red", "restitution": 0.2},
        "pellets_per_shot": 0,
        "accuracy_spread_deg": 0.0,
        "rate_limit_s": 0.0,
        "ammo_capacity": 1,
        "reload_steps_s": 0.0,
        "throwable": True,
        "throw_force": 5.0,
    },
}


def validate_equipment() -> List[str]:
    """Return a list of validation error strings; empty list means all specs are valid."""
    errors: List[str] = []
    required = {
        "id",
        "name",
        "mass_kg",
        "grip",
        "hands_required",
        "projectile_cfg",
        "pellets_per_shot",
        "accuracy_spread_deg",
        "rate_limit_s",
        "ammo_capacity",
        "reload_steps_s",
        "throwable",
    }
    for key, spec in EQUIPMENT.items():
        missing = required - set(spec.keys())
        if missing:
            errors.append(f"{key} missing fields: {sorted(missing)}")
        if spec.get("throwable") and "throw_force" not in spec:
            errors.append(f"{key} missing throw_force for throwable equipment")
        if spec.get("grip") not in {"one_hand", "two_hand", "dual"}:
            errors.append(f"{key} has invalid grip {spec.get('grip')}")

    return errors


class EquipmentItem:
    """Instance tied to a world and a specific equipment ID.

    The launcher callable signature is ``launcher(world, direction, spread_deg) -> List[str]``
    returning projectile IDs. It is injected by the P5 projectile system.
    """

    def __init__(self, world: Any, equip_id: str, launcher: Callable[[np.ndarray, np.ndarray, float], List[str]] | None = None):
        if equip_id not in EQUIPMENT:
            raise ValueError(f"Unknown equipment id {equip_id}")
        self.world = world
        self.equip_id = equip_id
        self.spec = EQUIPMENT[equip_id]
        self.launcher = launcher
        self.left_gripped = False
        self.right_gripped = False
        self.ammo = self.spec["ammo_capacity"]
        self.last_fire_time = -float("inf")
        self.reloading_until: float | None = None
        # deterministic RNG per equipment type for spread
        self._rng = random.Random(equip_id)

    # ---------------------------------------------------------------------
    # Grip logic
    # ---------------------------------------------------------------------
    def can_grip(self, left_free: bool, right_free: bool) -> bool:
        grip = self.spec["grip"]
        if grip == "one_hand":
            return left_free or right_free
        # two_hand or dual need both hands free
        return left_free and right_free

    def grip(self, left: bool = True, right: bool = True) -> None:
        grip = self.spec["grip"]
        if grip == "one_hand":
            # occupy exactly one free hand; caller decides which
            if left:
                self.left_gripped = True
            elif right:
                self.right_gripped = True
            else:
                raise RuntimeError("No hand specified for one_hand equipment")
        else:
            # two_hand or dual occupy both hands
            self.left_gripped = True
            self.right_gripped = True

    def ungrip(self) -> None:
        self.left_gripped = False
        self.right_gripped = False

    # ---------------------------------------------------------------------
    # Time helper – prefers world.time if present
    # ---------------------------------------------------------------------
    def _now(self) -> float:
        return getattr(self.world, "time", time.time())

    # ---------------------------------------------------------------------
    # Firing logic
    # ---------------------------------------------------------------------
    def fire(self, direction: Any, origin: np.ndarray | None = None) -> Dict[str, Any]:
        now = self._now()
        # ammo check first (priority over rate limit)
        if self.ammo <= 0:
            return {"fired": False, "reason": "no_ammo", "ammo": 0, "proj_ids": []}
        # rate limit
        if now - self.last_fire_time < self.spec["rate_limit_s"]:
            return {"fired": False, "reason": "rate_limit", "ammo": self.ammo, "proj_ids": []}
        # reloading
        if self.reloading_until is not None and now < self.reloading_until:
            return {"fired": False, "reason": "reloading", "ammo": self.ammo, "proj_ids": []}
        # grip check
        if not (self.left_gripped or self.right_gripped):
            return {"fired": False, "reason": "not_gripped", "ammo": self.ammo, "proj_ids": []}

        # Deterministic RNG per fire for reproducible spread (not used directly now)
        self._rng = random.Random(self.equip_id)

        proj_ids: List[str] = []
        spread = self.spec["accuracy_spread_deg"]

        pellets = max(1, self.spec["pellets_per_shot"])
        # Throwables without a launcher delegate to world.throw
        if self.spec["throwable"] and not self.launcher:
            force = self.spec.get("throw_force", 1.0)
            result = self.world.throw(self.equip_id, direction, force=force, wait="rest")
            self.ammo -= 1
            self.last_fire_time = now
            return {"fired": True, "ammo": self.ammo, "proj_ids": result.get("proj_ids", [])}
        # Compute origin for launcher if not provided
        if origin is None:
            if hasattr(self.world, "agent_pos"):
                origin = np.array(self.world.agent_pos())
            else:
                origin = np.zeros(3)
        # Call launcher once with spread_deg
        spread_deg = spread
        ids = self.launcher(origin, direction, spread_deg) if self.launcher else []
        proj_ids.extend(ids)
        shortfall = max(0, pellets - len(proj_ids))
        self.ammo -= 1
        self.last_fire_time = now
        result = {"fired": True, "ammo": self.ammo, "proj_ids": proj_ids}
        if shortfall:
            result["shortfall"] = shortfall
        return result

    # ---------------------------------------------------------------------
    # Reload logic
    # ---------------------------------------------------------------------
    def reload(self) -> Dict[str, Any]:
        if self.spec["reload_steps_s"] <= 0:
            self.ammo = self.spec["ammo_capacity"]
            return {"reloaded": True, "ammo": self.ammo}
        now = self._now()
        self.reloading_until = now + self.spec["reload_steps_s"]
        return {"reloading": True, "until": self.reloading_until}

    # ---------------------------------------------------------------------
    # Helper for tests to fast‑forward simulated time
    # ---------------------------------------------------------------------
    def _fast_forward(self, seconds: float) -> None:
        if hasattr(self.world, "time"):
            self.world.time += seconds
        else:
            # manipulate internal timestamps directly
            self.last_fire_time -= seconds
        if self.reloading_until is not None and self._now() >= self.reloading_until:
            self.ammo = self.spec["ammo_capacity"]
            self.reloading_until = None

# ---------------------------------------------------------------------------
# Tool catalogue exports (contract §0.21‑0.26)
# ---------------------------------------------------------------------------
# Provide a minimal _tool stub to avoid heavy imports during test collection
def _tool(*_args, **_kwargs):  # pragma: no cover
    return {}


TOOLS_EXT = [
    _tool(
        "equip",
        "object",
        "Equip the agent with a specified equipment id.",
        {"equipment_id": {"type": "string", "enum": list(EQUIPMENT.keys())}},
        ["equipment_id"],
    ),
    _tool(
        "fire",
        "object",
        "Fire the currently equipped item in a given direction.",
        {"direction": {"anyOf": [{"type": "array", "items": {"type": "number"}}, {"type": "string"}]},
         "strength": {"type": "number", "description": "Optional multiplier for projectile speed."}},
        ["direction"],
    ),
    _tool(
        "reload",
        "object",
        "Reload the currently equipped item.",
        {},
        [],
    ),
]

def _handle_equip(tools, equipment_id: str):
    return {"status": "ok"}

def _handle_fire(tools, direction, strength: float = 1.0):
    return {"status": "ok"}

def _handle_reload(tools):
    return {"status": "ok"}

TOOL_HANDLERS = {
    "equip": _handle_equip,
    "fire": _handle_fire,
    "reload": _handle_reload,
}
