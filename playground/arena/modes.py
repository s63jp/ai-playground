
"""Utilities to construct world configuration snippets for different match modes.
"""
from __future__ import annotations

from typing import Dict, Any


def build_world_cfg(arena_cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return a minimal world configuration dictionary.

    The real project supplies a complex spec; for our purposes we only forward the geometry and
    obstacle information that the caller passes in.
    """
    arena_cfg = arena_cfg or {}
    # Ensure keys exist; callers may omit them.
    obstacles = arena_cfg.get('obstacles', [])
    cover = arena_cfg.get('cover', [])
    return {'obstacles': obstacles, 'cover': cover}
