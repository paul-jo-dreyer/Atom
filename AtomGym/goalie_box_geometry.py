"""Shared rounded-rectangle goalie-box geometry.

Both the env (box-entry detection + episode termination) and the
`GoalieBoxPenalty` reward (depth-weighted warning ramp) need the same
notion of "what does it mean to be inside the goalie box?". When the
box has rounded interior corners, the answer is more involved than a
simple axis-aligned rectangle test, and divergent implementations
across env/reward would produce hard-to-debug behaviours (timer
ticking while reward depth = 0 at corners, etc.).

This module is the single source of truth. The two public helpers:

  * `signed_depth_into_box(x, y, ...)` — signed-distance function (SDF)
    to the rounded-box boundary. > 0 inside, ≤ 0 outside. Used by the
    reward to compute the depth factor.
  * `is_in_opp_goalie_box(x, y, ...)` — bool wrapper around the SDF.
    Used by the env to decide whether to tick the box-time timer.

Geometry
--------
The opposing goalie box on side `s ∈ {+1, -1}` is the rectangle:

    x ∈ [s·(xh - bd), s·xh],   y ∈ [-bh, +bh]

with the TWO interior corners (those facing the field, at x = s·(xh−bd))
replaced by quarter-circle arcs of radius `r`, tangent to the adjoining
straight edges. The two corners on the GOAL-LINE side (at x = s·xh)
remain sharp — the goal line is a wall, not a field-facing edge.

The goal-line side itself is NOT a boundary for depth purposes: a
robot deep against the goal line is considered fully inside relative
to the field-facing perimeter, with depth equal to its distance from
the nearest field-facing edge.

`r = 0` recovers the plain-rectangle (sharp-cornered) geometry.

Implementation note: the asymmetry of "rounded only on the field side"
means we can't use the standard symmetric-rounded-box SDF directly.
The piecewise approach below handles each region (main body /
corner-arc) explicitly. We also exploit y-symmetry (only `±side` cases
matter) by mirroring x in the side=-1 case so the body of the
function only handles side=+1 geometry.
"""

from __future__ import annotations

import math


def signed_depth_into_box(
    x: float,
    y: float,
    *,
    field_x_half: float,
    goalie_box_depth: float,
    goalie_box_y_half: float,
    goalie_box_corner_radius: float = 0.0,
    side: int = +1,
) -> float:
    """Signed distance from (x, y) to the rounded-box boundary of the
    opposing goalie box on side `side` (∈ {+1, -1}).

    Returns:
        > 0  inside the box (value = depth from the nearest field-facing
             edge or arc, in metres).
        = 0  on the boundary.
        < 0  outside the box (value = signed distance to the nearest
             boundary; magnitude useful only for monotonicity, not for
             precise SDF guarantees).

    `r=0` ⟹ legacy sharp-cornered rectangle.
    """
    if side == -1:
        x = -x
    elif side != +1:
        raise ValueError(f"side must be +1 or -1, got {side}")

    xh = field_x_half
    bd = goalie_box_depth
    bh = goalie_box_y_half
    r = goalie_box_corner_radius
    x_edge = xh - bd

    # Past the goal line ⟹ outside. The goal line is at x = xh; points
    # with x > xh are in the goal chamber, not in the box.
    if x > xh:
        return xh - x

    # Distances to the three field-facing edges. Positive ⟹ inside.
    d_inner = x - x_edge
    d_top = bh - y
    d_bottom = y + bh

    # If outside any field-facing edge, return the most-negative signed
    # distance (i.e. SDF to the nearest such edge from the outside).
    if d_inner < 0 or d_top < 0 or d_bottom < 0:
        return min(d_inner, d_top, d_bottom)

    # Inside the bounding rectangle. With r > 0, points within `r` of
    # both the inner edge AND a y-edge are in the corner-arc region.
    # Their signed distance is `r - dist_to_inscribed_circle_centre`.
    if r > 0.0 and d_inner <= r:
        if d_top <= r:
            cx = x_edge + r
            cy = bh - r
            return r - math.hypot(x - cx, y - cy)
        if d_bottom <= r:
            cx = x_edge + r
            cy = -bh + r
            return r - math.hypot(x - cx, y - cy)

    # Main body — depth = min distance to the three field-facing edges.
    # The goal-line side at x = xh is NOT included: a robot deep
    # against the goal line is treated as fully inside.
    return min(d_inner, d_top, d_bottom)


def is_in_opp_goalie_box(
    x: float,
    y: float,
    *,
    field_x_half: float,
    goalie_box_depth: float,
    goalie_box_y_half: float,
    goalie_box_corner_radius: float = 0.0,
    side: int = +1,
) -> bool:
    """Bool wrapper around `signed_depth_into_box`. Strict inequality:
    a point on the boundary is NOT considered inside (no double-
    counting at exact boundary)."""
    return signed_depth_into_box(
        x, y,
        field_x_half=field_x_half,
        goalie_box_depth=goalie_box_depth,
        goalie_box_y_half=goalie_box_y_half,
        goalie_box_corner_radius=goalie_box_corner_radius,
        side=side,
    ) > 0.0
