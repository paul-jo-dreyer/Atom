"""Renderer implementations.

Each renderer consumes a `SceneSpec` + `StyleConfig` and produces output —
either to an interactive window (live) or to a numpy RGB frame (headless).
The base.py protocol is the contract; new backends just implement it.
"""

from .base import Renderer
from .pygame_headless import PygameHeadlessRenderer
from .pygame_live import PygameLiveRenderer

__all__ = ["PygameHeadlessRenderer", "PygameLiveRenderer", "Renderer"]
