"""Unit tests for the three concrete reward terms (BallProgressReward,
DistanceToBallReward, GoalScoredReward). Pure-python — no sim_py needed."""

from __future__ import annotations

import numpy as np
import pytest

from AtomGym.action_observation import (
    ActionView,
    BALL_BLOCK_DIM,
    ObsView,
    V_BALL_MAX,
    build_observation,
)
from AtomGym.rewards import (
    BallProgressReward,
    DistanceToBallReward,
    GoalScoredReward,
    RewardContext,
)


FIELD_X_HALF = 0.375
FIELD_Y_HALF = 0.225
GOAL_Y_HALF = 0.06
GOAL_EXTENSION = 0.06


def _ctx_from_states(
    *,
    ball_state: np.ndarray,
    self_state_5d: np.ndarray,
    info: dict | None = None,
    prev_obs: np.ndarray | None = None,
) -> RewardContext:
    """Build a RewardContext from raw sim-frame states. Goes through
    `build_observation` so the obs encoding is exercised end-to-end."""
    obs = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=ball_state,
        self_state_5d=self_state_5d,
    )
    return RewardContext(
        obs=obs,
        action=np.zeros(2, dtype=np.float32),
        prev_obs=prev_obs,
        prev_action=None,
        info=info if info is not None else {},
        obs_view=ObsView(n_robots=1),
        action_view=ActionView(),
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        goal_y_half=GOAL_Y_HALF,
        goal_extension=GOAL_EXTENSION,
        dt=1.0 / 60.0,
    )


# ---------------------------------------------------------------------------
# BallProgressReward
# ---------------------------------------------------------------------------


def test_ball_progress_zero_when_ball_at_rest() -> None:
    term = BallProgressReward()
    ctx = _ctx_from_states(
        ball_state=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        self_state_5d=np.zeros(5, dtype=np.float32),
    )
    assert term(ctx) == pytest.approx(0.0)


def test_ball_progress_positive_when_moving_toward_goal() -> None:
    """Ball at origin moving in +x → reward = +|vx| (cos = 1)."""
    term = BallProgressReward()
    vx = 1.5
    ctx = _ctx_from_states(
        ball_state=np.array([0.0, 0.0, vx, 0.0], dtype=np.float32),
        self_state_5d=np.zeros(5, dtype=np.float32),
    )
    assert term(ctx) == pytest.approx(vx, abs=1e-5)


def test_ball_progress_negative_when_moving_away_from_goal() -> None:
    """Ball at origin moving in -x → reward = -|vx| (cos = -1)."""
    term = BallProgressReward()
    vx = 1.5
    ctx = _ctx_from_states(
        ball_state=np.array([0.0, 0.0, -vx, 0.0], dtype=np.float32),
        self_state_5d=np.zeros(5, dtype=np.float32),
    )
    assert term(ctx) == pytest.approx(-vx, abs=1e-5)


def test_ball_progress_zero_when_moving_perpendicular() -> None:
    """Ball at origin, target also at y=0 (in mouth band) → +y motion is
    perpendicular to the goal direction, cos = 0."""
    term = BallProgressReward()
    ctx = _ctx_from_states(
        ball_state=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        self_state_5d=np.zeros(5, dtype=np.float32),
    )
    assert term(ctx) == pytest.approx(0.0, abs=1e-5)


def test_ball_progress_scales_with_velocity_magnitude() -> None:
    """Doubling velocity should double the reward (linearity)."""
    term = BallProgressReward()
    vx_low = 0.5
    vx_hi = 2.0
    ctx_low = _ctx_from_states(
        ball_state=np.array([0.0, 0.0, vx_low, 0.0], dtype=np.float32),
        self_state_5d=np.zeros(5, dtype=np.float32),
    )
    ctx_hi = _ctx_from_states(
        ball_state=np.array([0.0, 0.0, vx_hi, 0.0], dtype=np.float32),
        self_state_5d=np.zeros(5, dtype=np.float32),
    )
    assert term(ctx_hi) == pytest.approx(term(ctx_low) * (vx_hi / vx_low), abs=1e-5)


def test_ball_progress_target_snaps_to_mouth_band() -> None:
    """Ball at y=0.20 (above the mouth) — the target snaps to (xh, +gh).
    A ball moving in pure +x direction is no longer perfectly aligned, so
    the reward should be less than |vx|."""
    term = BallProgressReward()
    vx = 1.0
    ctx = _ctx_from_states(
        ball_state=np.array([0.0, 0.20, vx, 0.0], dtype=np.float32),
        self_state_5d=np.zeros(5, dtype=np.float32),
    )
    # Manual computation: dx = 0.375 - 0 = 0.375, dy = 0.06 - 0.20 = -0.14
    # dist = sqrt(0.375² + 0.14²) ≈ 0.4003
    # cos = 0.375 / 0.4003 ≈ 0.9367
    # reward = 1.0 * 0.9367 = 0.9367
    expected = 1.0 * 0.375 / (0.375**2 + 0.14**2) ** 0.5
    assert term(ctx) == pytest.approx(expected, abs=1e-3)
    # And strictly less than the perfectly-aligned case.
    assert term(ctx) < vx


def test_ball_progress_clipped_velocity_at_boundary() -> None:
    """Ball moving faster than V_BALL_MAX → obs clips to ±1, reward should
    saturate at V_BALL_MAX (in m/s, after de-normalization)."""
    term = BallProgressReward()
    huge_v = V_BALL_MAX * 3.0
    ctx = _ctx_from_states(
        ball_state=np.array([0.0, 0.0, huge_v, 0.0], dtype=np.float32),
        self_state_5d=np.zeros(5, dtype=np.float32),
    )
    # The obs clips vx to +1 (i.e. V_BALL_MAX m/s after de-normalization).
    assert term(ctx) == pytest.approx(V_BALL_MAX, abs=1e-5)


# ---------------------------------------------------------------------------
# DistanceToBallReward
# ---------------------------------------------------------------------------


def test_distance_zero_when_robot_at_ball() -> None:
    term = DistanceToBallReward()
    ctx = _ctx_from_states(
        ball_state=np.array([0.10, 0.05, 0.0, 0.0], dtype=np.float32),
        self_state_5d=np.array([0.10, 0.05, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert term(ctx) == pytest.approx(0.0, abs=1e-5)


def test_distance_pythagorean() -> None:
    """3-4-5 triangle: robot at origin, ball at (0.03, 0.04) → dist = 0.05."""
    term = DistanceToBallReward()
    ctx = _ctx_from_states(
        ball_state=np.array([0.03, 0.04, 0.0, 0.0], dtype=np.float32),
        self_state_5d=np.zeros(5, dtype=np.float32),
    )
    assert term(ctx) == pytest.approx(0.05, abs=1e-5)


def test_distance_independent_of_velocities_and_orientations() -> None:
    """Reward depends only on positions; vx, vy, theta, omega should not
    influence it."""
    term = DistanceToBallReward()
    ball_a = np.array([0.10, 0.0, 0.0, 0.0], dtype=np.float32)
    ball_b = np.array([0.10, 0.0, 4.0, -3.0], dtype=np.float32)  # different velocity
    self_a = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    self_b = np.array([0.0, 0.0, 1.5, 0.2, -0.3], dtype=np.float32)  # different theta/v/omega
    ctx_a = _ctx_from_states(ball_state=ball_a, self_state_5d=self_a)
    ctx_b = _ctx_from_states(ball_state=ball_b, self_state_5d=self_b)
    assert term(ctx_a) == pytest.approx(term(ctx_b), abs=1e-5)


def test_distance_returns_positive_value() -> None:
    """Sign convention: term returns POSITIVE distance. Use negative weight
    to penalise distance."""
    term = DistanceToBallReward()
    ctx = _ctx_from_states(
        ball_state=np.array([+0.20, -0.10, 0.0, 0.0], dtype=np.float32),
        self_state_5d=np.array([-0.20, +0.10, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert term(ctx) > 0


def test_distance_handles_anisotropic_field_normalization() -> None:
    """field_x_half=0.375 and field_y_half=0.225 differ — distance must be
    computed in metric coords, not normalised coords. Robot at (xh, 0),
    ball at (0, yh) → metric distance is sqrt(0.375² + 0.225²)."""
    term = DistanceToBallReward()
    ctx = _ctx_from_states(
        ball_state=np.array([0.0, FIELD_Y_HALF, 0.0, 0.0], dtype=np.float32),
        self_state_5d=np.array([FIELD_X_HALF, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    expected = (FIELD_X_HALF**2 + FIELD_Y_HALF**2) ** 0.5
    assert term(ctx) == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# GoalScoredReward
# ---------------------------------------------------------------------------


def test_goal_scored_zero_when_no_event() -> None:
    term = GoalScoredReward()
    ctx = _ctx_from_states(
        ball_state=np.zeros(4, dtype=np.float32),
        self_state_5d=np.zeros(5, dtype=np.float32),
        info={},
    )
    assert term(ctx) == 0.0


def test_goal_scored_positive_on_for_us() -> None:
    term = GoalScoredReward()
    ctx = _ctx_from_states(
        ball_state=np.zeros(4, dtype=np.float32),
        self_state_5d=np.zeros(5, dtype=np.float32),
        info={"scored_for_us": True, "scored_against_us": False},
    )
    assert term(ctx) == pytest.approx(1.0)


def test_goal_scored_negative_on_against_us() -> None:
    term = GoalScoredReward()
    ctx = _ctx_from_states(
        ball_state=np.zeros(4, dtype=np.float32),
        self_state_5d=np.zeros(5, dtype=np.float32),
        info={"scored_for_us": False, "scored_against_us": True},
    )
    assert term(ctx) == pytest.approx(-1.0)


def test_goal_scored_ignores_other_info_keys() -> None:
    """Term should not be confused by unrelated info keys."""
    term = GoalScoredReward()
    ctx = _ctx_from_states(
        ball_state=np.zeros(4, dtype=np.float32),
        self_state_5d=np.zeros(5, dtype=np.float32),
        info={"some_other_event": True, "collision": True},
    )
    assert term(ctx) == 0.0


def test_goal_scored_weight_scaling() -> None:
    """Sparse-but-large is set via weight at construction, not by the term
    returning a different magnitude."""
    term = GoalScoredReward(weight=20.0)
    assert term.weight == 20.0
    # The unweighted return is still ±1; the composite applies the scaling.
    ctx = _ctx_from_states(
        ball_state=np.zeros(4, dtype=np.float32),
        self_state_5d=np.zeros(5, dtype=np.float32),
        info={"scored_for_us": True},
    )
    assert term(ctx) == pytest.approx(1.0)
