"""Unit tests for `AtomGym.goalie_box_geometry` — the shared rounded-
rect SDF that drives both env box-detection and reward depth factor.

Coverage:
    * legacy r=0 case behaves identically to the old sharp-rectangle
    * with r > 0, the four corner regions are correctly carved away
    * symmetry: side=+1 and side=-1 give mirror-image results
    * goal-line side is NOT a depth boundary
    * `is_in_opp_goalie_box` agrees with `signed_depth > 0`
"""

from __future__ import annotations

import math

import pytest

from AtomGym.goalie_box_geometry import (
    is_in_opp_goalie_box,
    signed_depth_into_box,
)


XH = 0.375
BD = 0.12
BH = 0.10
R = 0.06
EDGE = XH - BD  # = 0.255 — field-facing inner x of the right box


def _depth(x, y, *, r=R, side=+1):
    return signed_depth_into_box(
        x, y,
        field_x_half=XH,
        goalie_box_depth=BD,
        goalie_box_y_half=BH,
        goalie_box_corner_radius=r,
        side=side,
    )


# ---------------------------------------------------------------------------
# Legacy r=0 case — must match the old sharp-rect behaviour exactly
# ---------------------------------------------------------------------------


def test_sharp_rect_centroid_inside() -> None:
    centroid = (XH - 0.5 * BD, 0.0)
    assert _depth(*centroid, r=0) > 0


def test_sharp_rect_outside_in_x_returns_negative() -> None:
    just_outside = (EDGE - 0.01, 0.0)
    assert _depth(*just_outside, r=0) < 0


def test_sharp_rect_outside_in_y_returns_negative() -> None:
    just_outside = (XH - 0.5 * BD, BH + 0.01)
    assert _depth(*just_outside, r=0) < 0


def test_sharp_rect_corner_is_on_boundary() -> None:
    """At the sharp corner (x_edge, ±bh), depth = 0 (exactly on
    boundary) — d_inner = 0 dominates."""
    assert _depth(EDGE, BH, r=0) == pytest.approx(0.0, abs=1e-9)
    assert _depth(EDGE, -BH, r=0) == pytest.approx(0.0, abs=1e-9)


def test_sharp_rect_inside_depth_min_of_three_edges() -> None:
    """At (EDGE + 0.02, 0), depth = min(0.02, BH, BH) = 0.02."""
    assert _depth(EDGE + 0.02, 0.0, r=0) == pytest.approx(0.02, abs=1e-9)


# ---------------------------------------------------------------------------
# Rounded-corner case — the corner regions are carved away
# ---------------------------------------------------------------------------


def test_rounded_sharp_corner_is_OUTSIDE() -> None:
    """At (EDGE, BH) — the legacy sharp corner — the rounded box
    treats this as OUTSIDE (the corner is carved away)."""
    assert _depth(EDGE, BH, r=R) < 0
    assert _depth(EDGE, -BH, r=R) < 0


def test_rounded_inscribed_circle_centre_is_max_corner_depth() -> None:
    """The centre of the inscribed circle is at (EDGE+r, BH-r) — the
    deepest point inside the corner-arc region."""
    cx, cy = EDGE + R, BH - R
    # At the centre, depth = r (maximal). Outside r-radius, decreasing.
    assert _depth(cx, cy, r=R) == pytest.approx(R, abs=1e-9)


def test_rounded_perimeter_arc_is_zero() -> None:
    """Points on the inscribed circle (the rounded boundary itself)
    have signed depth = 0."""
    cx, cy = EDGE + R, BH - R
    # 45° point on the arc that bounds the carved region.
    # The "outside-the-arc" direction is toward (EDGE, BH); on the
    # arc itself: cx + r*cos(135°), cy + r*sin(135°).
    px = cx + R * math.cos(math.radians(135))
    py = cy + R * math.sin(math.radians(135))
    assert _depth(px, py, r=R) == pytest.approx(0.0, abs=1e-9)


def test_rounded_main_body_unchanged() -> None:
    """At (EDGE + 0.02, 0) — well inside the main body, away from
    corner regions — depth is the same as for sharp rect (= 0.02)."""
    assert _depth(EDGE + 0.02, 0.0, r=R) == pytest.approx(0.02, abs=1e-9)


def test_rounded_corner_region_uses_arc_distance() -> None:
    """Inside the corner-arc region, depth = r - dist_to_inscribed_centre."""
    cx, cy = EDGE + R, BH - R
    # Pick a point INSIDE the inscribed circle (offset 0.5*R from centre).
    angle = math.radians(150)  # somewhere in the upper-left of centre
    dx = 0.5 * R * math.cos(angle)
    dy = 0.5 * R * math.sin(angle)
    px, py = cx + dx, cy + dy
    expected_depth = R - math.hypot(dx, dy)
    assert _depth(px, py, r=R) == pytest.approx(expected_depth, abs=1e-9)


# ---------------------------------------------------------------------------
# Side mirror: side=-1 gives mirror-image (in x) results vs side=+1
# ---------------------------------------------------------------------------


def test_side_mirror_centroid() -> None:
    """Right-box centroid at (EDGE + bd/2, 0); left-box centroid at
    (-EDGE - bd/2, 0). Both should give the same depth."""
    right = _depth(XH - 0.5 * BD, 0.0, side=+1)
    left = _depth(-(XH - 0.5 * BD), 0.0, side=-1)
    assert right == pytest.approx(left, abs=1e-9)


def test_side_mirror_corner_carved() -> None:
    """Sharp legacy corner of the LEFT box at (-EDGE, BH) is
    carved away same as on the right side."""
    assert _depth(-EDGE, BH, r=R, side=-1) < 0


# ---------------------------------------------------------------------------
# Goal-line side is NOT a depth boundary
# ---------------------------------------------------------------------------


def test_goal_line_edge_is_not_a_boundary() -> None:
    """A point at the goal-line edge (x = XH, y = 0) is fully inside
    the box for depth purposes — depth should equal min(BD, BH) = BH
    (the closest field-facing edge is the y-edge at distance BH)."""
    depth = _depth(XH, 0.0)
    assert depth == pytest.approx(BH, abs=1e-9)


def test_past_goal_line_is_outside() -> None:
    """A point past the goal line is in the goal chamber, not in the
    box — should return a negative signed distance."""
    assert _depth(XH + 0.01, 0.0) < 0


# ---------------------------------------------------------------------------
# is_in_opp_goalie_box ↔ signed_depth_into_box agreement
# ---------------------------------------------------------------------------


def test_inside_check_agrees_with_sdf() -> None:
    sample_points = [
        (XH - 0.5 * BD, 0.0, True),       # centroid — inside
        (EDGE + 0.01, 0.0, True),         # near inner edge — inside
        (EDGE - 0.01, 0.0, False),        # outside inner edge
        (EDGE, BH, False),                # legacy sharp corner — carved away
        (EDGE + R, BH - R, True),         # inscribed-circle centre — inside
        (XH + 0.01, 0.0, False),          # past goal line — outside
    ]
    for x, y, expected in sample_points:
        assert is_in_opp_goalie_box(
            x, y,
            field_x_half=XH,
            goalie_box_depth=BD,
            goalie_box_y_half=BH,
            goalie_box_corner_radius=R,
            side=+1,
        ) is expected, f"(x={x}, y={y}) expected {expected}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_side_raises() -> None:
    with pytest.raises(ValueError, match="side"):
        signed_depth_into_box(
            0.0, 0.0,
            field_x_half=XH,
            goalie_box_depth=BD,
            goalie_box_y_half=BH,
            side=0,
        )
