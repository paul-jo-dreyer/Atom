"""Unit tests for BallAlignmentReward (pure-python, no sim_py).

The reward reads positions and heading directly from the obs vector,
so we can construct synthetic obs and exercise every regime: outside
the band (silent), inside contact (silent), at the band midpoint with
various heading orientations, and the constructor's argument
validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from AtomGym.action_observation import (
    ActionView,
    ObsView,
    sincos_from_theta,
)
from AtomGym.rewards import BallAlignmentReward, RewardContext


_FIELD_X_HALF = 0.375
_FIELD_Y_HALF = 0.225


def _ctx_for(
    *,
    robot_xy: tuple[float, float],
    ball_xy: tuple[float, float],
    robot_theta: float,
) -> RewardContext:
    """Synthesise an obs vector with the requested geometry, return a
    RewardContext suitable for evaluating the alignment reward.

    Positions are in world-frame metres; we apply the same per-axis
    normalisation the env's `build_observation` would have applied.
    """
    obs = np.zeros(11, dtype=np.float32)
    # Ball block (idx 0..3): [px/Fx, py/Fy, vx/Vmax, vy/Vmax]
    obs[0] = ball_xy[0] / _FIELD_X_HALF
    obs[1] = ball_xy[1] / _FIELD_Y_HALF
    # Robot block (idx 4..10): [px/Fx, py/Fy, sin θ, cos θ, dx, dy, dθ]
    obs[4] = robot_xy[0] / _FIELD_X_HALF
    obs[5] = robot_xy[1] / _FIELD_Y_HALF
    sin_th, cos_th = sincos_from_theta(robot_theta)
    obs[6] = sin_th
    obs[7] = cos_th

    return RewardContext(
        obs=obs,
        action=np.zeros(2, dtype=np.float32),
        prev_obs=None,
        prev_action=None,
        info={},
        obs_view=ObsView(n_robots=1),
        action_view=ActionView(),
        field_x_half=_FIELD_X_HALF,
        field_y_half=_FIELD_Y_HALF,
        goal_y_half=0.06,
        goal_extension=0.06,
        dt=1.0 / 30.0,
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_inner_must_be_nonneg() -> None:
    with pytest.raises(ValueError, match="inner_radius"):
        BallAlignmentReward(inner_radius=-0.01)


def test_outer_must_exceed_inner() -> None:
    with pytest.raises(ValueError, match="outer_radius"):
        BallAlignmentReward(inner_radius=0.05, outer_radius=0.05)
    with pytest.raises(ValueError, match="outer_radius"):
        BallAlignmentReward(inner_radius=0.10, outer_radius=0.05)


# ---------------------------------------------------------------------------
# Distance gate
# ---------------------------------------------------------------------------


def test_far_from_ball_returns_zero() -> None:
    """Beyond outer radius: silent regardless of alignment."""
    term = BallAlignmentReward(inner_radius=0.044, outer_radius=0.10)
    # 0.20 m apart, perfectly axis-aligned ⟹ would be peak if gate didn't apply
    ctx = _ctx_for(robot_xy=(0.0, 0.0), ball_xy=(0.20, 0.0), robot_theta=0.0)
    assert term(ctx) == pytest.approx(0.0)


def test_inside_contact_returns_zero() -> None:
    """Below inner radius (in contact / very-near contact): silent.
    Preserves dribbling — alignment doesn't fight ball-at-side
    behaviour."""
    term = BallAlignmentReward(inner_radius=0.044, outer_radius=0.10)
    # 0.03 m apart, perfectly axis-aligned
    ctx = _ctx_for(robot_xy=(0.0, 0.0), ball_xy=(0.03, 0.0), robot_theta=0.0)
    assert term(ctx) == pytest.approx(0.0)


def test_at_inner_boundary_returns_zero() -> None:
    """Exactly on the inner boundary: still zero (closed at inner)."""
    term = BallAlignmentReward(inner_radius=0.044, outer_radius=0.10)
    ctx = _ctx_for(robot_xy=(0.0, 0.0), ball_xy=(0.044, 0.0), robot_theta=0.0)
    assert term(ctx) == pytest.approx(0.0)


def test_at_outer_boundary_returns_zero() -> None:
    """Exactly on the outer boundary: still zero (closed at outer)."""
    term = BallAlignmentReward(inner_radius=0.044, outer_radius=0.10)
    ctx = _ctx_for(robot_xy=(0.0, 0.0), ball_xy=(0.10, 0.0), robot_theta=0.0)
    assert term(ctx) == pytest.approx(0.0)


def test_band_midpoint_aligned_returns_peak() -> None:
    """At (inner+outer)/2 with perfect alignment: parabolic gate peak
    = 1.0, |cos| = 1.0, reward = 1.0."""
    term = BallAlignmentReward(inner_radius=0.044, outer_radius=0.10)
    midpoint = 0.5 * (0.044 + 0.10)  # 0.072 m
    ctx = _ctx_for(
        robot_xy=(0.0, 0.0), ball_xy=(midpoint, 0.0), robot_theta=0.0
    )
    assert term(ctx) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Heading / alignment
# ---------------------------------------------------------------------------


def test_perpendicular_heading_returns_zero() -> None:
    """At band midpoint with body axis perpendicular to ball direction:
    |cos(π/2)| = 0 ⟹ reward zero. This is the credit-assignment
    failure case the term is designed to surface a gradient out of —
    by giving 0 here and >0 once rotation reduces |Δθ|."""
    term = BallAlignmentReward(inner_radius=0.044, outer_radius=0.10)
    midpoint = 0.5 * (0.044 + 0.10)
    # Ball in +x, robot facing +y (perpendicular)
    ctx = _ctx_for(
        robot_xy=(0.0, 0.0),
        ball_xy=(midpoint, 0.0),
        robot_theta=math.pi / 2.0,
    )
    assert term(ctx) == pytest.approx(0.0, abs=1e-6)


def test_back_aligned_matches_front_aligned() -> None:
    """|cos| symmetry: facing the ball (θ=0) and facing 180° away
    should give identical reward. Pushing a ball with the back face
    is just as effective as the front for this task."""
    term = BallAlignmentReward(inner_radius=0.044, outer_radius=0.10)
    midpoint = 0.5 * (0.044 + 0.10)
    front = _ctx_for(
        robot_xy=(0.0, 0.0),
        ball_xy=(midpoint, 0.0),
        robot_theta=0.0,
    )
    back = _ctx_for(
        robot_xy=(0.0, 0.0),
        ball_xy=(midpoint, 0.0),
        robot_theta=math.pi,
    )
    assert term(front) == pytest.approx(term(back), abs=1e-6)


def test_partial_alignment_scales_with_cos() -> None:
    """At 45° off axis, |cos(π/4)| = √2/2; reward should be peak ×
    that factor."""
    term = BallAlignmentReward(inner_radius=0.044, outer_radius=0.10)
    midpoint = 0.5 * (0.044 + 0.10)
    ctx = _ctx_for(
        robot_xy=(0.0, 0.0),
        ball_xy=(midpoint, 0.0),
        robot_theta=math.pi / 4.0,
    )
    expected = math.sqrt(2.0) / 2.0
    assert term(ctx) == pytest.approx(expected, abs=1e-6)


def test_aligned_in_y_direction() -> None:
    """Sanity: alignment isn't accidentally x-axis biased. Ball in +y,
    robot facing +y, mid-band distance ⟹ peak alignment."""
    term = BallAlignmentReward(inner_radius=0.044, outer_radius=0.10)
    midpoint = 0.5 * (0.044 + 0.10)
    ctx = _ctx_for(
        robot_xy=(0.0, 0.0),
        ball_xy=(0.0, midpoint),
        robot_theta=math.pi / 2.0,
    )
    assert term(ctx) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Composition with weight
# ---------------------------------------------------------------------------


def test_weight_scales_contribution() -> None:
    """Standard convention: positive return × weight = contribution."""
    term = BallAlignmentReward(weight=0.3, inner_radius=0.044, outer_radius=0.10)
    midpoint = 0.5 * (0.044 + 0.10)
    ctx = _ctx_for(
        robot_xy=(0.0, 0.0),
        ball_xy=(midpoint, 0.0),
        robot_theta=0.0,
    )
    raw = term(ctx)
    assert raw == pytest.approx(1.0, abs=1e-6)
    assert term.weight * raw == pytest.approx(0.3, abs=1e-6)


def test_term_name() -> None:
    """Stable name for TB log key under reward/."""
    assert BallAlignmentReward.name == "ball_alignment"
