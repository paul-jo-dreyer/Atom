"""Shared pygame drawing module — used by both live and headless backends.

Owns scene-to-pixel coordinate conversion (y-up world → y-down screen) and
all primitive drawing (field border, robots, balls, body axes, HUD). The
two public renderers (`PygameLiveRenderer`, `PygameHeadlessRenderer`) own
their own surfaces and forward to a single `PygameSceneDrawer.draw()` call,
guaranteeing pixel-identical output between live and exported video.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402
from pygame import gfxdraw  # noqa: E402

from ..scene import BallSpec, RobotSpec, SceneSpec  # noqa: E402
from ..style import RGB, BallStyle, RobotStyle, StyleConfig  # noqa: E402


class PygameSceneDrawer:
    """Stateless-ish drawer: holds style + computed scale, draws onto any surface."""

    def __init__(
        self,
        style: StyleConfig,
        render_w: int,
        render_h: int,
        field_x_half: float | None = None,
        field_y_half: float | None = None,
        hud_strip_px: int = 0,
    ) -> None:
        """`hud_strip_px` reserves the top N pixels of the render surface for HUD,
        shifting the field down so they don't overlap. Live renderers pass a
        positive value; headless renderers pass 0 (no HUD ever drawn)."""
        self.style = style
        self._render_w = render_w
        self._render_h = render_h
        self._hud_strip_px = max(0, hud_strip_px)
        self._scale_px_per_m: float | None = None
        if field_x_half is not None and field_y_half is not None:
            self._scale_px_per_m = self._compute_scale(field_x_half, field_y_half)
        # Font init requires pygame.init() / pygame.font.init() to have been called.
        if not pygame.font.get_init():
            pygame.font.init()
        self._font = pygame.font.SysFont("monospace", 14)

    # ---- public ----------------------------------------------------------

    def draw(
        self,
        surface: pygame.Surface,
        scene: SceneSpec,
        hud_lines: list[str] | None = None,
    ) -> None:
        if self._scale_px_per_m is None:
            self._scale_px_per_m = self._compute_scale(
                scene.field.x_half, scene.field.y_half
            )

        surface.fill(self.style.field.background)
        self._draw_field(surface, scene)
        for ball in scene.balls:
            self._draw_ball(surface, ball)
        for robot in scene.robots:
            self._draw_robot(surface, robot)
        if self.style.show_hud and hud_lines:
            self._draw_hud(surface, hud_lines)
        if self.style.show_hud and self._hud_strip_px > 0 and scene.controls:
            self._draw_control_panel(surface, scene)

    # ---- coordinate helpers ----------------------------------------------

    def _compute_scale(self, x_half: float, y_half: float) -> float:
        margin = 0.95
        avail_h = self._render_h - self._hud_strip_px
        sw = self._render_w * margin / (2.0 * x_half)
        sh = avail_h * margin / (2.0 * y_half)
        return min(sw, sh)

    def _w2s(self, x: float, y: float) -> tuple[int, int]:
        """World metres → render-surface pixels. Y-down flip + HUD strip offset."""
        assert self._scale_px_per_m is not None
        s = self._scale_px_per_m
        cx = self._render_w / 2
        # Field is centred in the area BELOW the HUD strip.
        cy = self._hud_strip_px + (self._render_h - self._hud_strip_px) / 2
        return (int(cx + x * s), int(cy - y * s))

    def _m2px(self, d: float) -> int:
        assert self._scale_px_per_m is not None
        return max(1, int(d * self._scale_px_per_m))

    # ---- primitives ------------------------------------------------------

    def _draw_field(self, surf: pygame.Surface, scene: SceneSpec) -> None:
        xh, yh = scene.field.x_half, scene.field.y_half
        tl = self._w2s(-xh, yh)
        br = self._w2s(xh, -yh)
        rect = pygame.Rect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
        pygame.draw.rect(
            surf, self.style.field.walls, rect, width=self.style.field.walls_width_px
        )

    def _draw_ball(self, surf: pygame.Surface, ball: BallSpec) -> None:
        bs: BallStyle = self.style.ball
        cx, cy = self._w2s(ball.px, ball.py)
        if bs.shape == "point":
            r = bs.point_radius_px
            gfxdraw.filled_circle(surf, cx, cy, r, bs.color)
            gfxdraw.aacircle(surf, cx, cy, r, bs.color)
            return
        r = self._m2px(ball.radius)
        gfxdraw.filled_circle(surf, cx, cy, r, bs.color)
        gfxdraw.aacircle(
            surf, cx, cy, r, bs.outline_color if bs.outline_color else bs.color
        )

    def _draw_robot(self, surf: pygame.Surface, robot: RobotSpec) -> None:
        rs: RobotStyle = self.style.robot_style_for(robot.team)

        if rs.shape == "point":
            cx, cy = self._w2s(robot.px, robot.py)
            r = rs.point_radius_px
            gfxdraw.filled_circle(surf, cx, cy, r, rs.body_color)
            gfxdraw.aacircle(surf, cx, cy, r, rs.body_color)
            if rs.show_axes:
                self._draw_axes(surf, robot, rs)
            return

        half = robot.chassis_side * 0.5
        chassis_local = ((-half, -half), (half, -half), (half, half), (-half, half))
        chassis_screen = [
            self._w2s(*self._body_to_world(p, robot)) for p in chassis_local
        ]
        self._fill_aapolygon(surf, chassis_screen, rs.body_color, rs.body_outline_color)

        if rs.shape == "full":
            for part in robot.manipulator_parts:
                if len(part) < 3:
                    continue
                part_screen = [self._w2s(*self._body_to_world(v, robot)) for v in part]
                self._fill_aapolygon(
                    surf,
                    part_screen,
                    rs.manipulator_color,
                    rs.manipulator_outline_color,
                )

        if rs.show_axes:
            self._draw_axes(surf, robot, rs)

    def _draw_axes(
        self, surf: pygame.Surface, robot: RobotSpec, rs: RobotStyle
    ) -> None:
        """Red x-axis (forward), green y-axis (left), drawn from body origin."""
        origin = self._w2s(robot.px, robot.py)
        x_tip = self._w2s(*self._body_to_world((rs.axes_length_m, 0.0), robot))
        y_tip = self._w2s(*self._body_to_world((0.0, rs.axes_length_m), robot))
        red: RGB = (220, 30, 30)
        green: RGB = (30, 180, 30)
        pygame.draw.line(surf, red, origin, x_tip, rs.axes_width_px)
        pygame.draw.line(surf, green, origin, y_tip, rs.axes_width_px)

    def _draw_hud(self, surf: pygame.Surface, lines: list[str]) -> None:
        for i, line in enumerate(lines):
            txt = self._font.render(line, True, self.style.hud_color)
            surf.blit(txt, (12, 10 + i * 16))

    # ---- control indicator panel ----------------------------------------

    # Control panel layout: cells fan out from screen centre, blue to the
    # LEFT and orange to the RIGHT. First cell of each team sits closest to
    # centre (at ±spacing·0.5); the next sits at ±spacing·1.5, then ±2.5.
    # 1 robot total                → centred (the lone team's first cell is at ±0.5).
    # 1 blue + 1 orange (2 total)  → blue at −0.5·spacing, orange at +0.5·spacing.
    # 3 + 3 (6 total)              → blue at −0.5/−1.5/−2.5, orange at +0.5/+1.5/+2.5.
    _CELL_W = 100  # per-robot cell width (px)
    _CELL_SPACING = 80  # px between adjacent cell centres
    _CELLS_PER_ROW = 3  # max robots per team row
    _VBAR_W = 16  # vertical (forward) bar width
    _VBAR_H = 36  # vertical bar height
    _HBAR_W = 40  # horizontal (turn) bar width
    _HBAR_H = 16  # horizontal bar height
    # Horizontal lean per team. -1 = left of centre, +1 = right.
    _TEAM_SIDE = {"blue": -1, "orange": +1}
    _BAR_FRAME_COLOR: RGB = (180, 180, 180)
    _BAR_ZERO_COLOR: RGB = (90, 90, 90)
    _LABEL_COLOR: RGB = (60, 60, 60)
    # Top-row team is blue, bottom-row team is orange. Per-team override slots
    # in `style.teams` provide the fill color.
    _ROW_TEAMS: tuple[str, str] = ("blue", "orange")

    def _draw_control_panel(self, surf: pygame.Surface, scene: SceneSpec) -> None:
        row_h = self._hud_strip_px // 2
        center_x = self._render_w / 2.0
        rosters = {t: [r for r in scene.robots if r.team == t] for t in self._ROW_TEAMS}
        total_cells = sum(min(len(rs), self._CELLS_PER_ROW) for rs in rosters.values())
        for row_idx, team_name in enumerate(self._ROW_TEAMS):
            team_robots = rosters[team_name]
            if not team_robots:
                continue
            team_color = self.style.robot_style_for(team_name).body_color
            sign = self._TEAM_SIDE.get(team_name, -1)
            for col_idx, robot in enumerate(team_robots[: self._CELLS_PER_ROW]):
                # Special case: a lone cell sits dead-centre regardless of team.
                # Otherwise, cell `col_idx` for this team is at
                # sign · spacing · (col_idx + 0.5):
                #   blue  (sign −1):  −0.5·s, −1.5·s, −2.5·s
                #   orange (sign +1): +0.5·s, +1.5·s, +2.5·s
                if total_cells == 1:
                    offset = 0.0
                else:
                    offset = sign * self._CELL_SPACING * (col_idx + 0.5)
                cx = int(center_x + offset - self._CELL_W / 2)
                # cy = row_idx * row_h
                cy = row_h - 5
                fwd, turn = scene.controls.get(robot.name, (0.0, 0.0))
                self._draw_one_indicator(
                    surf, cx, cy, self._CELL_W, row_h, fwd, turn, team_color, robot.name
                )

    def _draw_one_indicator(
        self,
        surf: pygame.Surface,
        x: int,
        y: int,
        w: int,
        h: int,
        forward: float,
        turn: float,
        team_color: RGB,
        label: str,
    ) -> None:
        """Single robot's indicator: vertical (forward) bar atop horizontal (turn) bar."""
        forward = max(-1.0, min(1.0, float(forward)))
        turn = max(-1.0, min(1.0, float(turn)))

        # ---- vertical bar (forward) ----
        vbar_x = x + (w - self._VBAR_W) // 2
        vbar_y = y + 4
        vbar_cy = vbar_y + self._VBAR_H // 2
        # frame
        pygame.draw.rect(
            surf,
            self._BAR_FRAME_COLOR,
            (vbar_x, vbar_y, self._VBAR_W, self._VBAR_H),
            1,
        )
        # zero line (horizontal across the vertical bar)
        pygame.draw.line(
            surf,
            self._BAR_ZERO_COLOR,
            (vbar_x, vbar_cy),
            (vbar_x + self._VBAR_W - 1, vbar_cy),
            1,
        )
        # fill
        half_h = self._VBAR_H // 2
        fill_h = int(round(abs(forward) * (half_h - 1)))
        if fill_h > 0:
            if forward > 0:  # +forward grows UP from centre
                fill_rect = (vbar_x + 1, vbar_cy - fill_h, self._VBAR_W - 2, fill_h)
            else:  # −forward grows DOWN from centre
                fill_rect = (vbar_x + 1, vbar_cy + 1, self._VBAR_W - 2, fill_h)
            pygame.draw.rect(surf, team_color, fill_rect)

        # ---- horizontal bar (turn) ----
        hbar_x = x + (w - self._HBAR_W) // 2
        hbar_y = vbar_y + self._VBAR_H + 4
        hbar_cx = hbar_x + self._HBAR_W // 2
        pygame.draw.rect(
            surf,
            self._BAR_FRAME_COLOR,
            (hbar_x, hbar_y, self._HBAR_W, self._HBAR_H),
            1,
        )
        pygame.draw.line(
            surf,
            self._BAR_ZERO_COLOR,
            (hbar_cx, hbar_y),
            (hbar_cx, hbar_y + self._HBAR_H - 1),
            1,
        )
        # Visual inversion: +turn (CCW) fills LEFT of centre, −turn fills RIGHT.
        half_w = self._HBAR_W // 2
        fill_w = int(round(abs(turn) * (half_w - 1)))
        if fill_w > 0:
            if turn > 0:  # +turn = CCW = fills LEFT
                fill_rect = (hbar_cx - fill_w, hbar_y + 1, fill_w, self._HBAR_H - 2)
            else:  # −turn = CW = fills RIGHT
                fill_rect = (hbar_cx + 1, hbar_y + 1, fill_w, self._HBAR_H - 2)
            pygame.draw.rect(surf, team_color, fill_rect)

        # ---- label ----
        # Truncate to fit; centred horizontally below the horizontal bar.
        small = pygame.font.SysFont("monospace", 11)
        max_chars = max(1, w // 7)
        text = label if len(label) <= max_chars else label[: max_chars - 1] + "…"
        txt = small.render(text, True, self._LABEL_COLOR)
        label_x = x + (w - txt.get_width()) // 2
        label_y = hbar_y + self._HBAR_H + 2
        surf.blit(txt, (label_x, label_y))

    # ---- geometry --------------------------------------------------------

    @staticmethod
    def _body_to_world(
        local: tuple[float, float], robot: RobotSpec
    ) -> tuple[float, float]:
        c, s = np.cos(robot.theta), np.sin(robot.theta)
        return (
            robot.px + c * local[0] - s * local[1],
            robot.py + s * local[0] + c * local[1],
        )

    @staticmethod
    def _fill_aapolygon(
        surf: pygame.Surface,
        points: list[tuple[int, int]],
        fill: RGB,
        outline: RGB | None,
    ) -> None:
        gfxdraw.filled_polygon(surf, points, fill)
        gfxdraw.aapolygon(surf, points, outline if outline is not None else fill)
        if outline is not None:
            pygame.draw.aalines(surf, outline, True, points)
