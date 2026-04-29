"""Per-reset domain-randomization config for env initial states.

`InitialStateRanges` holds (low, high) bounds for everything an env
randomizes on `reset()`. Envs accept one in their constructor and sample
uniformly from each range every time `reset()` is called.

Defaults reproduce the original "random pose, zero velocity" behavior —
existing code that doesn't pass a ranges object sees no change. Curriculum
stages set non-zero velocity ranges (or wider/narrower position ranges) to
broaden or narrow the policy's training distribution.

Pose-position randomization is expressed as a margin (the half-extent
shrinkage along each axis), so it scales naturally to whatever field
config the env is using. Velocity ranges are absolute (m/s, rad/s).

Ball velocity uses polar (speed, direction) rather than Cartesian
(vx, vy) so the magnitude is directly controllable — easy to write
"ball moves at up to 1 m/s in any direction" or "ball at rest".
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class InitialStateRanges:
    """Uniform-sampling bounds for every initial-state DOF the env can
    randomize at reset. All `(low, high)` tuples; `low <= high` enforced.

    Defaults: random pose within the field, zero initial velocities."""

    # ---- robot pose -----------------------------------------------------
    # xy is sampled uniformly inside the field with this margin from each
    # wall. Same value applied to both x and y.
    robot_xy_margin: float = 0.04
    robot_theta: tuple[float, float] = (-math.pi, math.pi)

    # ---- robot velocities ----------------------------------------------
    robot_speed: tuple[float, float] = (0.0, 0.0)  # body-frame longitudinal, m/s
    robot_omega: tuple[float, float] = (0.0, 0.0)  # rad/s, body-frame yaw rate

    # ---- ball position --------------------------------------------------
    ball_xy_margin: float = 0.04

    # ---- ball velocity (polar) -----------------------------------------
    ball_speed: tuple[float, float] = (0.0, 0.0)  # magnitude, m/s
    ball_direction: tuple[float, float] = (-math.pi, math.pi)  # rad, world frame

    def __post_init__(self) -> None:
        for name in (
            "robot_theta",
            "robot_speed",
            "robot_omega",
            "ball_speed",
            "ball_direction",
        ):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(
                    f"InitialStateRanges.{name}: low ({low}) must be <= high ({high})"
                )
        for name in ("robot_xy_margin", "ball_xy_margin"):
            v = getattr(self, name)
            if v < 0.0:
                raise ValueError(
                    f"InitialStateRanges.{name}: must be >= 0, got {v}"
                )
