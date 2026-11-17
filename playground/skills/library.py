
"""Skill registry and executor.
The library stores SkillSpec objects on disk as JSON files and can invoke
them via a CapabilityAdapter implementation.
"""
from __future__ import annotations
import json
import pathlib
from typing import Dict, List, Any
from .spec import SkillSpec
import dataclasses

class SkillsLibrary:
    def __init__(self, root: pathlib.Path):
        self.root = pathlib.Path(root)
        self._registry: Dict[str, SkillSpec] = {}
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, spec: SkillSpec) -> None:
        if spec.id in self._registry:
            raise ValueError(f"Skill with id {spec.id} already registered")
        self._registry[spec.id] = spec

    def get(self, skill_id: str) -> SkillSpec:
        return self._registry[skill_id]

    def list(self) -> List[SkillSpec]:
        return list(self._registry.values())

    def save(self) -> None:
        """Write each skill spec to a JSON file named ``<id>.json`` under ``root``."""
        for spec in self._registry.values():
            out_path = self.root / f"{spec.id}.json"
            # dataclasses.asdict works for frozen dataclasses
            with out_path.open('w', encoding='utf-8') as f:
                json.dump(dataclasses.asdict(spec), f, indent=2)

    def load(self) -> None:
        """Load all ``*.json`` files from ``root`` into the registry."""
        self._registry.clear()
        for p in self.root.glob('*.json'):
            data = json.load(p.open())
            spec = SkillSpec(**data)
            self._registry[spec.id] = spec

    def invoke(self, skill_id: str, adapter: Any, params: Dict[str, Any] | None = None) -> dict:
        """Execute a skill against *adapter*.

        Returns a dict with keys ``skill``, ``status`` and ``detail``; when the
        controller method returns a dict, the full measured payload is carried
        under ``result`` (additive — callers of the old shape keep working).
        """
        params = params or {}
        spec = self.get(skill_id)
        # Simple precondition check: each key/value must match an attribute on the adapter.
        for key, expected in spec.preconditions.items():
            actual = getattr(adapter, key, None)
            if actual != expected:
                return {"skill": skill_id, "status": "unavailable", "detail": f"Precondition {key}={expected} not met"}
        # Dispatch to the appropriate adapter method based on skill name.
        method_name = spec.controller.get('method')
        if not method_name or not hasattr(adapter, method_name):
            return {"skill": skill_id, "status": "unavailable", "detail": f"Adapter lacks method {method_name}"}
        method = getattr(adapter, method_name)
        # Catalog defaults fill unset parameters; explicit params win.
        merged = {**spec.controller.get("defaults", {}), **params}
        try:
            result = method(**merged)
        except Exception as exc:
            return {"skill": skill_id, "status": "failed", "detail": str(exc)}
        # Expect result dict with a ``status`` key.
        status = result.get('status', 'failed') if isinstance(result, dict) else 'failed'
        detail = result.get('detail', '') if isinstance(result, dict) else ''
        out = {"skill": skill_id, "status": status, "detail": detail}
        if isinstance(result, dict):
            out["result"] = result
        return out

    def bump_version(self, skill_id: str, new_version: str) -> None:
        spec = self.get(skill_id)
        # because SkillSpec is frozen we need to create a new instance
        updated = dataclasses.replace(spec, version=new_version)
        self._registry[skill_id] = updated
        # also write to disk if already saved
        out_path = self.root / f"{skill_id}.json"
        if out_path.exists():
            out_path.write_text(json.dumps(dataclasses.asdict(updated), indent=2), encoding='utf-8')
