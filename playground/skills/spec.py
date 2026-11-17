
"""Skill specification dataclass."""
from __future__ import annotations
import dataclasses
from typing import Dict, Any

@dataclasses.dataclass(frozen=True)
class SkillSpec:
    id: str
    name: str
    category: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    preconditions: Dict[str, Any]
    success_criteria: Dict[str, Any]
    failure_conditions: Dict[str, Any]
    controller: Dict[str, Any]
    status: str
    version: str
    metrics: Dict[str, Any]

    def __post_init__(self):
        # Basic validation: required non-empty strings and dicts
        if not self.id:
            raise ValueError('SkillSpec.id must be non‑empty')
        if not self.name:
            raise ValueError('SkillSpec.name must be non‑empty')
        if not self.category:
            raise ValueError('SkillSpec.category must be non‑empty')
        if self.status not in {"available", "unavailable"}:
            raise ValueError('SkillSpec.status must be "available" or "unavailable"')
        # ensure dict fields are dicts
        for field_name in ["inputs", "outputs", "preconditions", "success_criteria",
                           "failure_conditions", "controller", "metrics"]:
            if not isinstance(getattr(self, field_name), dict):
                raise TypeError(f'SkillSpec.{field_name} must be a dict')
