"""Unit tests for StallPenaltyReward (pure-python, no sim_py)."""

from __future__ import annotations

import numpy as np
import pytest

from AtomGym.action_observation import ActionView, ObsView
from AtomGym.rewards import RewardContext, StallPenaltyReward


def _ctx_with_action(v: float, omega: float) -> RewardContext:
    """Build a RewardContext with the given action; everything else is zero."""
    return RewardContext(
        obs=np.zeros(11, dtype=np.float32),
        action=np.array([v, omega], dtype=np.float32),
        prev_obs=None,
        prev_action=None,
        info={},
        obs_view=ObsView(n_robots=1),
        action_view=ActionView(),
        field_x_half=0.375,
        field_y_half=0.225,
        goal_y_half=0.06,
        goal_extension=0.06,
        dt=1.0 / 30.0,
    )


# ---------------------------------------------------------------------------
# Boundary cases
# ---------------------------------------------------------------------------


def test_zero_action_returns_one() -> None:
    """The "do nothing" action: maximum stall penalty (1.0)."""
    term = StallPenaltyReward()
    assert term(_ctx_with_action(0.0, 0.0)) == pytest.approx(1.0)


def test_full_forward_returns_zero() -> None:
    term = StallPenaltyReward()
    assert term(_ctx_with_action(1.0, 0.0)) == pytest.approx(0.0)


def test_full_reverse_returns_zero() -> None:
    term = StallPenaltyReward()
    assert term(_ctx_with_action(-1.0, 0.0)) == pytest.approx(0.0)


def test_full_spin_returns_zero() -> None:
    term = StallPenaltyReward()
    assert term(_ctx_with_action(0.0, 1.0)) == pytest.approx(0.0)
    assert term(_ctx_with_action(0.0, -1.0)) == pytest.approx(0.0)


def test_diagonal_full_action_clips_to_zero() -> None:
    """(1, 1) has L2 magnitude sqrt(2) ≈ 1.41 > 1; the clip prevents the
    term from going negative (which would be a bonus rather than a penalty
    when used with a negative weight)."""
    term = StallPenaltyReward()
    assert term(_ctx_with_action(1.0, 1.0)) == pytest.approx(0.0)
    assert term(_ctx_with_action(-1.0, -1.0)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Linearity / proportional values
# ---------------------------------------------------------------------------


def test_half_pure_forward() -> None:
    """V=0.5, Ω=0 → magnitude 0.5 → penalty 0.5."""
    term = StallPenaltyReward()
    assert term(_ctx_with_action(0.5, 0.0)) == pytest.approx(0.5)


def test_quarter_pure_spin() -> None:
    term = StallPenaltyReward()
    assert term(_ctx_with_action(0.0, 0.25)) == pytest.approx(0.75)


def test_diagonal_partial_action() -> None:
    """V=Ω=0.3 → magnitude sqrt(0.18) ≈ 0.424 → penalty ≈ 0.576."""
    term = StallPenaltyReward()
    expected = 1.0 - (0.3**2 + 0.3**2) ** 0.5
    assert term(_ctx_with_action(0.3, 0.3)) == pytest.approx(expected, abs=1e-6)


def test_symmetry_of_signs() -> None:
    """Magnitude is signless — V vs -V give the same penalty."""
    term = StallPenaltyReward()
    assert term(_ctx_with_action(0.4, 0.7)) == pytest.approx(
        term(_ctx_with_action(-0.4, -0.7)), abs=1e-7
    )
    assert term(_ctx_with_action(0.4, 0.7)) == pytest.approx(
        term(_ctx_with_action(0.4, -0.7)), abs=1e-7
    )


# ---------------------------------------------------------------------------
# Weight + composite integration
# ---------------------------------------------------------------------------


def test_negative_weight_produces_negative_contribution() -> None:
    """The convention: positive return × negative weight = penalty in the
    composite breakdown."""
    term = StallPenaltyReward(weight=-0.5)
    raw = term(_ctx_with_action(0.0, 0.0))
    weighted = term.weight * raw
    assert raw == pytest.approx(1.0)
    assert weighted == pytest.approx(-0.5)


def test_zero_action_dominates_over_full_action() -> None:
    """Sanity: a stalling step always gets a more-negative contribution
    than a full-action step, when used with negative weight."""
    term = StallPenaltyReward(weight=-1.0)
    stall_contrib = term.weight * term(_ctx_with_action(0.0, 0.0))
    full_contrib = term.weight * term(_ctx_with_action(1.0, 0.0))
    assert stall_contrib < full_contrib
