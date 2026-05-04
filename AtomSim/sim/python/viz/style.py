"""Style configuration: colors, shape modes, resolution. Loaded from YAML.

A `StyleConfig` says *how* to draw the scene; a `SceneSpec` says *what* to
draw. They are independent — swap styles freely without re-running the sim,
swap renderers freely without changing either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

RobotShape = Literal["full", "square_only", "point"]
BallShape = Literal["circle", "point"]
RGB = tuple[int, int, int]


def parse_color(s: str) -> RGB:
    """Parse an "#RRGGBB" hex color into an (R, G, B) tuple."""
    s = s.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Expected '#RRGGBB' hex color, got {s!r}")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


@dataclass(frozen=True)
class Resolution:
    render_w: int  # internal raster width (pixels)
    render_h: int  # internal raster height
    output_w: int  # final blitted/exported width
    output_h: int  # final blitted/exported height


@dataclass(frozen=True)
class FieldStyle:
    background: RGB
    field_color: RGB
    walls: RGB
    walls_width_px: int
    # Mowed-grass overlay: alternating-brightness stripes across the turf.
    # Set `mowed_stripes_n=0` to disable (single-tone turf).
    mowed_stripes_n: int = 12
    mowed_stripes_delta: int = 14  # ±RGB shift between adjacent stripes
    mowed_stripes_axis: Literal["vertical", "horizontal"] = "vertical"


@dataclass(frozen=True)
class RobotStyle:
    shape: RobotShape
    body_color: RGB
    manipulator_color: RGB
    body_outline_color: RGB | None
    manipulator_outline_color: RGB | None
    show_axes: bool
    axes_length_m: float  # red x-axis and green y-axis line length
    axes_width_px: int
    point_radius_px: int  # used when shape == "point"


@dataclass(frozen=True)
class BallStyle:
    shape: BallShape
    color: RGB
    outline_color: RGB | None
    point_radius_px: int


@dataclass(frozen=True)
class FieldMarkings:
    """Cosmetic interior field lines (center circle, halfway line, goalie boxes).
    Pure visual overlay — no physics interaction."""

    enabled: bool = True
    color: RGB = (255, 255, 255)
    line_width_px: int = 2
    center_circle_radius_m: float = 0.07
    halfway_line: bool = True
    goalie_box_depth_m: float = 0.06  # how far box extends into the field
    goalie_box_height_m: float = 0.18  # full vertical extent (top to bottom)


@dataclass(frozen=True)
class TeamStyle:
    """Per-team color overrides applied on top of the default RobotStyle."""

    body_color: RGB | None = None
    manipulator_color: RGB | None = None
    body_outline_color: RGB | None = None
    manipulator_outline_color: RGB | None = None


@dataclass
class StyleConfig:
    resolution: Resolution
    field: FieldStyle
    robot: RobotStyle
    ball: BallStyle
    markings: FieldMarkings = field(default_factory=FieldMarkings)
    teams: dict[str, TeamStyle] = field(default_factory=dict)
    show_hud: bool = True
    hud_color: RGB = (30, 30, 30)

    def robot_style_for(self, team: str | None) -> RobotStyle:
        """Return RobotStyle with team overrides folded in."""
        if team is None or team not in self.teams:
            return self.robot
        ovr = self.teams[team]
        return RobotStyle(
            shape=self.robot.shape,
            body_color=ovr.body_color or self.robot.body_color,
            manipulator_color=ovr.manipulator_color or self.robot.manipulator_color,
            body_outline_color=ovr.body_outline_color or self.robot.body_outline_color,
            manipulator_outline_color=(
                ovr.manipulator_outline_color or self.robot.manipulator_outline_color
            ),
            show_axes=self.robot.show_axes,
            axes_length_m=self.robot.axes_length_m,
            axes_width_px=self.robot.axes_width_px,
            point_radius_px=self.robot.point_radius_px,
        )


def _opt_color(d: dict, key: str) -> RGB | None:
    v = d.get(key)
    return parse_color(v) if v else None


def load_style(path: str | Path) -> StyleConfig:
    """Load a YAML style file into a StyleConfig."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())

    res = raw["resolution"]
    render_w, render_h = res["render"]
    output_w, output_h = res.get("output", res["render"])
    resolution = Resolution(int(render_w), int(render_h), int(output_w), int(output_h))

    f = raw["field"]
    stripes_axis = f.get("mowed_stripes_axis", "vertical")
    if stripes_axis not in ("vertical", "horizontal"):
        raise ValueError(
            f"mowed_stripes_axis must be 'vertical' or 'horizontal', got {stripes_axis!r}"
        )
    field_style = FieldStyle(
        background=parse_color(f["background"]),
        field_color=parse_color(f["field_color"]),
        walls=parse_color(f["walls"]),
        walls_width_px=int(f.get("walls_width_px", 3)),
        mowed_stripes_n=int(f.get("mowed_stripes_n", 12)),
        mowed_stripes_delta=int(f.get("mowed_stripes_delta", 14)),
        mowed_stripes_axis=stripes_axis,
    )

    r = raw["robot"]
    robot_style = RobotStyle(
        shape=r.get("shape", "full"),
        body_color=parse_color(r["body_color"]),
        manipulator_color=parse_color(r["manipulator_color"]),
        body_outline_color=_opt_color(r, "body_outline_color"),
        manipulator_outline_color=_opt_color(r, "manipulator_outline_color"),
        show_axes=bool(r.get("show_axes", True)),
        axes_length_m=float(r.get("axes_length_m", 0.04)),
        axes_width_px=int(r.get("axes_width_px", 2)),
        point_radius_px=int(r.get("point_radius_px", 6)),
    )

    b = raw["ball"]
    ball_style = BallStyle(
        shape=b.get("shape", "circle"),
        color=parse_color(b["color"]),
        outline_color=_opt_color(b, "outline_color"),
        point_radius_px=int(b.get("point_radius_px", 4)),
    )

    mk = raw.get("markings") or {}
    markings = FieldMarkings(
        enabled=bool(mk.get("enabled", True)),
        color=parse_color(mk.get("color", "#FFFFFF")),
        line_width_px=int(mk.get("line_width_px", 2)),
        center_circle_radius_m=float(mk.get("center_circle_radius_m", 0.07)),
        halfway_line=bool(mk.get("halfway_line", True)),
        goalie_box_depth_m=float(mk.get("goalie_box_depth_m", 0.06)),
        goalie_box_height_m=float(mk.get("goalie_box_height_m", 0.18)),
    )

    teams: dict[str, TeamStyle] = {}
    for name, td in (raw.get("teams") or {}).items():
        teams[name] = TeamStyle(
            body_color=_opt_color(td, "body_color"),
            manipulator_color=_opt_color(td, "manipulator_color"),
            body_outline_color=_opt_color(td, "body_outline_color"),
            manipulator_outline_color=_opt_color(td, "manipulator_outline_color"),
        )

    return StyleConfig(
        resolution=resolution,
        field=field_style,
        robot=robot_style,
        ball=ball_style,
        markings=markings,
        teams=teams,
        show_hud=bool(raw.get("show_hud", True)),
        hud_color=parse_color(raw["hud_color"]) if "hud_color" in raw else (30, 30, 30),
    )
