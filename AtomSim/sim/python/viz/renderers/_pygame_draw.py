"""Shared pygame drawing module — used by both live and headless backends.

Owns scene-to-pixel coordinate conversion (y-up world → y-down screen) and
all primitive drawing (field border, robots, balls, body axes, HUD). The
two public renderers (`PygameLiveRenderer`, `PygameHeadlessRenderer`) own
their own surfaces and forward to a single `PygameSceneDrawer.draw()` call,
guaranteeing pixel-identical output between live and exported video.
"""

from __future__ import annotations

import math
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

        # Draw order is critical: surround fill → green turf rect → white
        # interior markings → white perimeter walls. If markings were drawn
        # before the turf, the turf rectangle would paint over them.
        surface.fill(self.style.field.background)
        self._draw_turf(surface, scene)
        self._draw_markings(surface, scene)
        self._draw_walls(surface, scene)
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
        margin = 0.8
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

    def _draw_turf(self, surf: pygame.Surface, scene: SceneSpec) -> None:
        """Filled rectangle of grass colour, slightly larger than the field
        so the white perimeter walls sit ON the turf rather than at its edge.

        If `field.mowed_stripes_n > 0`, overlays alternating-brightness bands
        on top of the base rect for a mowed-grass look. The base fill is
        still drawn first so any stripe-rounding gap reveals the canonical
        turf colour rather than the background."""
        xh, yh = scene.field.x_half, scene.field.y_half
        buffer_x = 0.08
        buffer_y = 0.04
        x0, y0 = self._w2s(-xh - buffer_x, yh + buffer_y)
        xf, yf = self._w2s(xh + buffer_x, -yh - buffer_y)
        pygame.draw.rect(
            surf,
            color=self.style.field.field_color,
            rect=(x0, y0, xf - x0, yf - y0),
            width=0,
        )

        n = self.style.field.mowed_stripes_n
        if n <= 0:
            return
        delta = self.style.field.mowed_stripes_delta
        base = self.style.field.field_color

        def _shift(c: int, d: int) -> int:
            return max(0, min(255, c + d))

        # Two tones flanking the base by ±delta/2 so the visual mean
        # equals the configured turf colour.
        half = delta // 2
        light = (_shift(base[0], +half), _shift(base[1], +half), _shift(base[2], +half))
        dark = (_shift(base[0], -half), _shift(base[1], -half), _shift(base[2], -half))

        if self.style.field.mowed_stripes_axis == "vertical":
            total = xf - x0
            for i in range(n):
                a = int(round(x0 + i * total / n))
                b = int(round(x0 + (i + 1) * total / n))
                pygame.draw.rect(
                    surf,
                    color=light if i % 2 == 0 else dark,
                    rect=(a, y0, b - a, yf - y0),
                    width=0,
                )
        else:  # horizontal
            total = yf - y0
            for i in range(n):
                a = int(round(y0 + i * total / n))
                b = int(round(y0 + (i + 1) * total / n))
                pygame.draw.rect(
                    surf,
                    color=light if i % 2 == 0 else dark,
                    rect=(x0, a, xf - x0, b - a),
                    width=0,
                )

    def _draw_walls(self, surf: pygame.Surface, scene: SceneSpec) -> None:
        """Field perimeter + goal-chamber outlines + nets. Drawn AFTER
        markings so the wall lines sit on top of the centre circle and
        goalie boxes.

        The perimeter is always drawn in `field.walls`. Each goal
        chamber (frame + net) can optionally override its colour via
        `field.goal_color_left` / `_right` — typically copying a team
        colour from `teams:` to give each side a visual identity."""
        xh, yh = scene.field.x_half, scene.field.y_half
        gh, gx = scene.field.goal_y_half, scene.field.goal_extension
        wall_color = self.style.field.walls
        left_goal_color = self.style.field.goal_color_left or wall_color
        right_goal_color = self.style.field.goal_color_right or wall_color
        boarder_width = self.style.field.walls_width_px
        net_width = max(1, int(boarder_width * 0.75))
        has_goals = gh > 0.0 and gx > 0.0

        def line(
            color: RGB,
            p1: tuple[float, float],
            p2: tuple[float, float],
            thickness: int,
        ) -> None:
            pygame.draw.line(surf, color, self._w2s(*p1), self._w2s(*p2), thickness)

        # Top + bottom walls (full-width). Always white perimeter.
        line(wall_color, (-xh, yh), (xh, yh), boarder_width)
        line(wall_color, (-xh, -yh), (xh, -yh), boarder_width)

        if not has_goals:
            line(wall_color, (-xh, -yh), (-xh, yh), boarder_width)
            line(wall_color, (xh, -yh), (xh, yh), boarder_width)
            return

        # Field walls split around the goal mouth — perimeter colour.
        line(wall_color, (-xh, -yh), (-xh, yh), boarder_width)
        line(wall_color, (xh, -yh), (xh, yh), boarder_width)

        # Left net.
        net_gap = 0.01
        x0 = -xh - gx
        x = x0
        while x < -xh:
            x = min(-xh, x + net_gap)
            line(wall_color, (x, gh), (x, -gh), net_width)
        y = gh
        while y > -gh:
            y = max(-gh, y - net_gap)
            line(wall_color, (x0, y), (x0 + gx, y), net_width)

        # Right net.
        x0 = xh + gx
        x = x0
        while x > xh:
            x = max(xh, x - net_gap)
            line(wall_color, (x, gh), (x, -gh), net_width)
        y = gh
        while y > -gh:
            y = max(-gh, y - net_gap)
            line(wall_color, (x0, y), (x0 - gx, y), net_width)

        # Left goal chamber — U opening to the right.
        line(left_goal_color, (-xh - gx, gh), (-xh, gh), (int(boarder_width * 1.5)))
        line(left_goal_color, (-xh - gx, -gh), (-xh, -gh), (int(boarder_width * 1.5)))
        line(
            left_goal_color, (-xh - gx, -gh), (-xh - gx, gh), (int(boarder_width * 1.5))
        )

        # Right goal chamber — U opening to the left.
        line(right_goal_color, (xh, gh), (xh + gx, gh), (int(boarder_width * 1.5)))
        line(right_goal_color, (xh, -gh), (xh + gx, -gh), (int(boarder_width * 1.5)))
        line(
            right_goal_color, (xh + gx, -gh), (xh + gx, gh), (int(boarder_width * 1.5))
        )

    def _draw_markings(self, surf: pygame.Surface, scene: SceneSpec) -> None:
        """Cosmetic interior soccer-field lines: halfway line, centre circle,
        goalie boxes. Drawn before the perimeter walls and physical objects
        so they sit beneath everything that matters."""
        m = self.style.markings
        if not m.enabled:
            return

        xh, yh = scene.field.x_half, scene.field.y_half
        color = m.color
        width = max(3, m.line_width_px)

        # Halfway line.
        if m.halfway_line:
            pygame.draw.line(
                surf, color, self._w2s(0.0, -yh), self._w2s(0.0, yh), width
            )

        # Centre circle.
        if m.center_circle_radius_m > 0.0:
            cx, cy = self._w2s(0.0, 0.0)
            r_px = self._m2px(m.center_circle_radius_m)
            pygame.draw.circle(surf, color, (cx, cy), r_px, width)
            pygame.draw.circle(surf, color, (cx, cy), r_px // 10, width * 5)

        # Goalie boxes — open "U" shape with the goal-line side absent.
        # Two interior corners (where horizontal meets vertical) can be
        # rounded via tangent quarter-arcs of radius
        # `goalie_box_corner_radius_m`. r=0 ⟹ sharp legacy corners.
        bd = m.goalie_box_depth_m
        bh = 0.5 * m.goalie_box_height_m
        if bd > 0.0 and bh > 0.0:
            # Clamp the radius so the arcs always fit in the box. The
            # interior vertical line needs 2r of vertical extent for the
            # two arcs, so r ≤ bh; each horizontal stub needs r of x
            # extent, so r ≤ bd.
            r = max(0.0, min(m.goalie_box_corner_radius_m, bd, bh))
            left_box_color = m.box_color_left or color
            right_box_color = m.box_color_right or color

            # Fill (translucent) is drawn FIRST so the outline lands on
            # top of it — the outline reads as the boundary of the
            # filled region rather than a separate stroke.
            if m.box_fill_alpha > 0:
                self._draw_one_goalie_box_fill(
                    surf,
                    left_box_color,
                    m.box_fill_alpha,
                    side=-1,
                    xh=xh,
                    bd=bd,
                    bh=bh,
                    r=r,
                )
                self._draw_one_goalie_box_fill(
                    surf,
                    right_box_color,
                    m.box_fill_alpha,
                    side=+1,
                    xh=xh,
                    bd=bd,
                    bh=bh,
                    r=r,
                )
            self._draw_one_goalie_box(
                surf, left_box_color, width, side=-1, xh=xh, bd=bd, bh=bh, r=r
            )
            self._draw_one_goalie_box(
                surf, right_box_color, width, side=+1, xh=xh, bd=bd, bh=bh, r=r
            )

    def _draw_one_goalie_box(
        self,
        surf: pygame.Surface,
        color: RGB,
        width: int,
        *,
        side: int,
        xh: float,
        bd: float,
        bh: float,
        r: float,
    ) -> None:
        """Render the open-U marking outline for one goalie box.

        side = +1 → right box (opens toward -x, against the +x wall)
        side = -1 → left  box (opens toward +x, against the -x wall)

        With r=0 this reduces to three straight segments (the legacy
        sharp-cornered behaviour). With r>0 the two interior corners
        become quarter-arcs tangent to the adjoining segments.

        Implementation: build the entire U as a single polyline of
        world-frame points (corners replaced by arc samples) and draw
        it via pygame.draw.lines. Single rasteriser ⟹ consistent
        thickness and clean joins where the arcs meet the straight
        segments. Drawing line + arc separately produces 1-2 px gaps
        because pygame's line/arc primitives round sub-pixel positions
        differently."""
        path = self._goalie_box_path(side=side, xh=xh, bd=bd, bh=bh, r=r)
        screen_pts = [self._w2s(x, y) for x, y in path]
        # `closed=False` ⟹ open polyline: the goal-line edge (the
        # implicit closing segment from last point back to first) is
        # NOT drawn — that's where the box opens onto the goal mouth.
        pygame.draw.lines(surf, color, False, screen_pts, width)

    def _draw_one_goalie_box_fill(
        self,
        surf: pygame.Surface,
        color: RGB,
        alpha: int,
        *,
        side: int,
        xh: float,
        bd: float,
        bh: float,
        r: float,
    ) -> None:
        """Translucent polygon fill of the goalie-box interior.

        Same path geometry as the outline (so the boundary lines up
        exactly), but rendered as a closed polygon with per-pixel
        alpha. The implicit closing segment from the bottom goal-line
        endpoint back to the top one becomes the goal-line side of the
        fill — interior fully enclosed."""
        path = self._goalie_box_path(side=side, xh=xh, bd=bd, bh=bh, r=r)
        screen_pts = [self._w2s(x, y) for x, y in path]
        # Per-pixel alpha requires a SRCALPHA-flagged surface. We draw
        # onto a transparent overlay sized to the main surface, then
        # blit it down. Cheap (one alloc per goalie box per frame —
        # the fill is render-time, not policy-time, so this isn't on
        # the hot path).
        overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        rgba = (color[0], color[1], color[2], int(alpha))
        pygame.draw.polygon(overlay, rgba, screen_pts)
        surf.blit(overlay, (0, 0))

    def _goalie_box_path(
        self,
        *,
        side: int,
        xh: float,
        bd: float,
        bh: float,
        r: float,
    ) -> list[tuple[float, float]]:
        """Build the world-frame polyline path for one goalie box.

        Walks from the top goal-line endpoint, along the open-U side
        (top horizontal → top arc → interior vertical → bottom arc →
        bottom horizontal) to the bottom goal-line endpoint. The
        implicit closing edge from last point back to first IS the
        goal-line side (only drawn if the caller closes the polygon —
        the outline `lines` call passes closed=False so it stays open)."""
        x_open = side * xh
        x_edge = side * (xh - bd)
        path: list[tuple[float, float]] = [(x_open, +bh)]
        if r > 0.0:
            # Top arc — sweep in path-order. side=+1 ⟹ CCW π/2→π;
            # side=-1 ⟹ CW π/2→0.
            cx_top = x_edge + side * r
            cy_top = +bh - r
            a_start, a_end = (
                (math.pi / 2, math.pi) if side == +1 else (math.pi / 2, 0.0)
            )
            path.extend(self._sample_arc(cx_top, cy_top, r, a_start, a_end))
            # The vertical segment from top-arc-end to bottom-arc-start
            # is drawn implicitly by pygame.draw.lines/polygon between
            # consecutive points. No explicit breakpoint needed.
            cx_bot = x_edge + side * r
            cy_bot = -bh + r
            a_start, a_end = (
                (math.pi, 3 * math.pi / 2) if side == +1 else (0.0, -math.pi / 2)
            )
            path.extend(self._sample_arc(cx_bot, cy_bot, r, a_start, a_end))
        else:
            # Sharp corners — three explicit points (legacy geometry).
            path.append((x_edge, +bh))
            path.append((x_edge, -bh))
        path.append((x_open, -bh))
        return path

    @staticmethod
    def _sample_arc(
        cx: float,
        cy: float,
        r: float,
        a_start: float,
        a_end: float,
        n_samples: int = 24,
    ) -> list[tuple[float, float]]:
        """Sample a circular arc as a list of (x, y) world-frame points.
        `np.linspace(a_start, a_end, n)` handles either sweep direction
        (positive or negative angular change) automatically. n_samples=24
        gives ~4° steps over a quarter-circle — at typical render
        scales (~250 px/m, r=60 mm ⟹ 15 px) the resulting chord lengths
        are sub-pixel, so the polyline reads as a smooth arc."""
        return [
            (cx + r * math.cos(t), cy + r * math.sin(t))
            for t in np.linspace(a_start, a_end, n_samples)
        ]

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
        row_h = self._hud_strip_px // 3
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
