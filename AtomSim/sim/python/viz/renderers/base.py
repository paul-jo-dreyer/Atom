"""Renderer protocol — the contract every backend implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..scene import SceneSpec


@runtime_checkable
class Renderer(Protocol):
    """A renderer draws SceneSpecs.

    Live backends present to a window and return None.
    Headless backends return an RGB ndarray of shape (H, W, 3), uint8.
    """

    def render(
        self, scene: SceneSpec, hud_lines: list[str] | None = None
    ) -> np.ndarray | None:
        """Draw one frame. `hud_lines` is honored only by live renderers."""
        ...

    def close(self) -> None:
        """Release any resources (windows, surfaces, encoders)."""
        ...
