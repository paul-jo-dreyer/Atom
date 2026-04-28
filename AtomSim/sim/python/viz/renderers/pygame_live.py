"""Pygame live renderer — opens a window, draws SceneSpecs in real time.

Owns a window + a render surface; drawing is delegated to PygameSceneDrawer
so this backend is pixel-identical to the headless backend.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from ..scene import SceneSpec  # noqa: E402
from ..style import StyleConfig  # noqa: E402
from ._pygame_draw import PygameSceneDrawer  # noqa: E402


class PygameLiveRenderer:
    """Interactive pygame renderer. Opens a window on construction."""

    def __init__(
        self,
        style: StyleConfig,
        title: str = "AtomSim",
        field_x_half: float | None = None,
        field_y_half: float | None = None,
    ) -> None:
        self.style = style
        pygame.init()
        pygame.display.set_caption(title)
        self._window = pygame.display.set_mode(
            (style.resolution.output_w, style.resolution.output_h)
        )
        self._render_surface = pygame.Surface(
            (style.resolution.render_w, style.resolution.render_h)
        )
        # Reserve top of the canvas for HUD (~6 lines × 16 px + padding).
        hud_strip_px = 110 if style.show_hud else 0
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
    ) -> None:
        self._drawer.draw(self._render_surface, scene, hud_lines)
        if (
            self.style.resolution.render_w != self.style.resolution.output_w
            or self.style.resolution.render_h != self.style.resolution.output_h
        ):
            scaled = pygame.transform.smoothscale(
                self._render_surface,
                (self.style.resolution.output_w, self.style.resolution.output_h),
            )
            self._window.blit(scaled, (0, 0))
        else:
            self._window.blit(self._render_surface, (0, 0))
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()
