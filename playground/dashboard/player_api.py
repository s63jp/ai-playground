"""Player API router.
Provides endpoints for a human player to join the simulation, send input, and query state.
"""

from fastapi import APIRouter, HTTPException
from typing import Callable, Any

router = APIRouter(prefix="/api/player")

_world_provider: Callable[[], Any] | None = None
_controller: Any | None = None


def set_world_provider(fn: Callable[[], Any]):
    """Inject a callable that returns the live PhysicsWorld instance.
    The server can call this to retrieve the current world for player actions.
    """
    global _world_provider
    _world_provider = fn


def _get_world():
    if _world_provider is None:
        raise HTTPException(status_code=500, detail="World provider not configured")
    world = _world_provider()
    if world is None:
        raise HTTPException(status_code=500, detail="World instance not available")
    return world


@router.post("/join")
def join():
    """Attach a PlayerController to the live world.
    Returns error dict if world provider missing.
    """
    try:
        world = _get_world()
    except HTTPException as e:
        return {"error": e.detail}
    global _controller
    from ..player import PlayerController
    _controller = PlayerController(world)
    return {"joined": True}


@router.post("/input")
def input(state: dict):
    """Accept player input state dict and forward to controller.
    Expected keys: move, yaw, pitch, jump, run.
    """
    if _controller is None:
        return {"error": "player not joined"}
    try:
        result = _controller.apply_input(state)
        return {"state": result}
    except Exception as e:
        return {"error": str(e)}


@router.get("/state")
def get_state():
    if _controller is None:
        return {"error": "player not joined"}
    try:
        out = {"state": _controller.state()}
    except Exception as e:
        return {"error": str(e)}
    # Humanoid view feed: the skeleton (bodies + joints) lets static/js/
    # humanoid_view.js and debug overlays render the figure; agent_skeleton()
    # works for box agents too (single body) so the panel degrades gracefully.
    try:
        world = _get_world()
        out["skeleton"] = world.agent_skeleton()
        out["agent_state"] = world.agent_state()
    except HTTPException:
        pass
    except Exception as e:
        out["skeleton_error"] = f"{type(e).__name__}: {e}"
    return out
