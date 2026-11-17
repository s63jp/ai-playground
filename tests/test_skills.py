
import json
import pathlib
import pytest
from playground.skills.spec import SkillSpec
from playground.skills.library import SkillsLibrary
from playground.skills.adapter import WorldCapabilities, CapabilityAdapter

# Simple stub world that implements a few methods
class StubWorld:
    def __init__(self):
        self.position = 0.0
        self.fallen = False
    def set_gait(self, mode, speed=1.0, heading=None):
        # Simulate movement for walking/running
        if mode in {"walk", "run", "turn", "jump"}:
            self.position += speed * 0.5
            return {"status": "succeeded", "detail": f"Moved {mode}"}
        return {"status": "failed", "detail": "Mode not supported"}
    def move(self, direction=None, distance=0.5, speed=None, **extra):
        # The catalogue's walking/running skill dispatches to move(); the stub
        # advances by the requested distance.
        self.position += float(distance)
        return {"status": "succeeded", "detail": f"Moved {distance}"}
    def stand_up(self):
        self.fallen = False
        return {"status": "succeeded", "detail": "Standing up"}
    def agent_fallen(self):
        return {"status": "succeeded", "detail": "Agent is fallen"}

# Stub manipulation controller with only reach/grasp present
class StubManip:
    def reach(self, object_id):
        return {"status": "succeeded", "detail": f"Reached {object_id}"}
    def grasp(self, object_id):
        return {"status": "succeeded", "detail": f"Grasped {object_id}"}
    def release(self):
        return {"status": "succeeded", "detail": "Released"}
    def pick_up(self, object_id):
        return {"status": "succeeded", "detail": f"Picked {object_id}"}
    def throw(self, object_id, direction, force):
        return {"status": "succeeded", "detail": f"Threw {object_id}"}

def make_adapter(with_manip=True):
    world = StubWorld()
    manip = StubManip() if with_manip else None
    return WorldCapabilities(world, manip)

def test_spec_validation():
    # valid spec
    spec = SkillSpec(
        id="test",
        name="Test",
        category="test",
        inputs={},
        outputs={},
        preconditions={},
        success_criteria={},
        failure_conditions={},
        controller={},
        status="available",
        version="1.0",
        metrics={},
    )
    assert spec.id == "test"
    # invalid status
    with pytest.raises(ValueError):
        SkillSpec(**{**spec.__dict__, "status": "bad"})

def test_library_roundtrip(tmp_path):
    lib = SkillsLibrary(tmp_path)
    from playground.skills.catalog import standing, walking
    lib.register(standing)
    lib.register(walking)
    lib.save()
    # new instance load
    lib2 = SkillsLibrary(tmp_path)
    lib2.load()
    assert lib2.get(standing.id).name == standing.name
    assert len(lib2.list()) == 2

def test_invoke_available_skill():
    lib = SkillsLibrary(pathlib.Path('.'))
    from playground.skills.catalog import walking
    lib.register(walking)
    adapter = make_adapter()
    result = lib.invoke(walking.id, adapter, {"mode": "walk", "speed": 2.0})
    assert result["status"] == "succeeded"
    # position should have increased
    assert adapter.world.position > 0

def test_unavailable_due_to_missing_capability():
    lib = SkillsLibrary(pathlib.Path('.'))
    from playground.skills.catalog import reaching
    lib.register(reaching)
    adapter = make_adapter(with_manip=False)  # no manipulation controller
    result = lib.invoke(reaching.id, adapter, {"object_id": "obj1"})
    assert result["status"] == "unavailable"

def test_tool_handlers_shape():
    from TOOLS_EXT.TOOL_HANDLERS import TOOL_HANDLERS
    assert "invoke_skill" in TOOL_HANDLERS
    schema = TOOL_HANDLERS["invoke_skill"]["params_schema"]
    # ensure required field 'skill' exists
    assert 'skill' in schema.__annotations__
