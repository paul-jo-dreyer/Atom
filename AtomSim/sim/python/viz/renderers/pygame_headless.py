"""Pygame headless renderer — draws SceneSpecs to off-screen surfaces and
returns RGB ndarrays. Same drawing code as the live backend; output is
bit-identical (modulo any output-resolution scaling).

Does NOT call pygame.display — only `pygame.font` is initialised, so this
works on truly headless machines (no X server, no `SDL_VIDEODRIVER` env
var required).
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from ..scene import SceneSpec  # noqa: E402
from ..style import StyleConfig  # noqa: E402
from ._pygame_draw import PygameSceneDrawer  # noqa: E402


class PygameHeadlessRenderer:
    """Headless renderer. `render(scene)` returns an (H, W, 3) uint8 RGB array."""

    def __init__(
        self,
        style: StyleConfig,
        field_x_half: float | None = None,
        field_y_half: float | None = None,
        show_hud: bool = False,
    ) -> None:
        """`show_hud` opts in to drawing the HUD strip (control panel + any
        hud_lines passed to render()). Default is False — most exported
        videos don't want a HUD overlay on top. Set True for tooling that
        wants the live-style display (e.g. random-action demos)."""
        self.style = style
        if not pygame.font.get_init():
            pygame.font.init()
        self._render_surface = pygame.Surface(
            (style.resolution.render_w, style.resolution.render_h)
        )
        hud_strip_px = 110 if (show_hud and style.show_hud) else 0
        self._drawer = PygameSceneDrawer(
            style,
            style.resolution.render_w,
            style.resolution.render_h,
            field_x_half,
            field_y_half,
            hud_strip_px=hud_strip_px,
        )

    def render(
        self, scene: SceneSpec, hud_lines: list[str] | None = None
    ) -> np.ndarray:
        self._drawer.draw(self._render_surface, scene, hud_lines)
        out = self._render_surface
        if (
            self.style.resolution.render_w != self.style.resolution.output_w
            or self.style.resolution.render_h != self.style.resolution.output_h
        ):
            out = pygame.transform.smoothscale(
                self._render_surface,
                (self.style.resolution.output_w, self.style.resolution.output_h),
            )
        # pygame.surfarray.array3d returns (W, H, 3); imageio + numpy use (H, W, 3).
        return pygame.surfarray.array3d(out).swapaxes(0, 1)

    def close(self) -> None:
        # Surfaces are GC'd; nothing else to release in headless mode.
        pass
