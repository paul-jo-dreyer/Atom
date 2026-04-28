"""Pure-data description of a scene at one timestep.

A `SceneSpec` is everything a renderer needs to draw a frame: field bounds,
robot poses + geometry, ball poses + radii, sim time. It carries no Box2D
handles, no `sim_py` references, nothing tied to the live simulation. This
makes it the natural intermediate format between:

    live sim  ─→  SceneSpec  ─→  live renderer (pygame window)
    .npz      ─→  SceneSpec  ─→  headless renderer (mp4/gif)

Both paths feed identical SceneSpecs to identical renderer code. The only
difference is upstream: one builds from `sim_py` objects, the other from a
recorded episode array.

The builder `build_scene()` lives here for convenience; it lazily imports
`sim_py` so the rest of `viz/` can be imported before the build path is on
`sys.path`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sim_py  # noqa: F401  (only for type hints)


@dataclass(frozen=True)
class FieldSpec:
    x_half: float          # half-extent along x, metres
    y_half: float          # half-extent along y, metres
    goal_y_half: float = 0.06        # half-height of goal opening (0 = no goals)
    goal_extension: float = 0.06     # depth of goal box behind the wall


@dataclass(frozen=True)
class RobotSpec:
    name: str                                            # unique identifier
    team: str | None                                     # team key for style lookup; None = default style
    px: float                                            # body origin x, metres (world frame)
    py: float                                            # body origin y, metres (world frame)
    theta: float                                         # heading, radians
    chassis_side: float                                  # square chassis edge length, metres
    manipulator_parts: tuple[tuple[tuple[float, float], ...], ...]  # body-local convex polygons


@dataclass(frozen=True)
class BallSpec:
    name: str
    px: float
    py: float
    radius: float          # metres


@dataclass
class SceneSpec:
    """Complete scene at one frame. Mutable container; the *Spec children are frozen."""
    field: FieldSpec
    robots: list[RobotSpec] = field(default_factory=list)
    balls: list[BallSpec] = field(default_factory=list)
    t: float = 0.0         # sim time, seconds — for HUD or episode timestamps
    # Per-robot normalized control inputs at this frame, keyed by robot name:
    #   forward ∈ [-1, 1] (+1 full forward), turn ∈ [-1, 1] (+1 full CCW).
    # Read by the HUD's control indicator panel. Robots not in the dict get no bar.
    controls: dict[str, tuple[float, float]] = field(default_factory=dict)


def build_scene(
    world: "sim_py.World",
    robots: list[tuple[str, "sim_py.Robot"]] | list["sim_py.Robot"],
    balls: list[tuple[str, "sim_py.Ball"]] | list["sim_py.Ball"],
    t: float = 0.0,
    teams: dict[str, str] | None = None,
) -> SceneSpec:
    """Snapshot a live sim into a SceneSpec.

    `robots` and `balls` accept either bare objects (auto-named "robot_0", "ball_0", ...)
    or (name, object) pairs. `teams` maps robot name → team key for style lookup.
    """
    teams = teams or {}
    field_cfg = world.config

    robot_specs: list[RobotSpec] = []
    for i, item in enumerate(robots):
        name, robot = (item if isinstance(item, tuple) else (f"robot_{i}", item))
        s = robot.state
        cfg = robot.config
        parts = tuple(
            tuple((float(v[0]), float(v[1])) for v in part)
            for part in cfg.manipulator_parts
        )
        robot_specs.append(RobotSpec(
            name=name,
            team=teams.get(name),
            px=float(s[0]),
            py=float(s[1]),
            theta=float(s[2]),
            chassis_side=float(cfg.chassis_side),
            manipulator_parts=parts,
        ))

    ball_specs: list[BallSpec] = []
    for i, item in enumerate(balls):
        name, ball = (item if isinstance(item, tuple) else (f"ball_{i}", item))
        s = ball.state
        ball_specs.append(BallSpec(
            name=name,
            px=float(s[0]),
            py=float(s[1]),
            radius=float(ball.config.dynamics_params.radius),
        ))

    return SceneSpec(
        field=FieldSpec(
            x_half=float(field_cfg.field_x_half),
            y_half=float(field_cfg.field_y_half),
            goal_y_half=float(field_cfg.goal_y_half),
            goal_extension=float(field_cfg.goal_extension),
        ),
        robots=robot_specs,
        balls=ball_specs,
        t=t,
    )
