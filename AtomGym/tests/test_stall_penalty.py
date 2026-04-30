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


def test_diagonal_full_action_returns_zero() -> None:
    """(1, 1): max(|V|, |Ω|) = 1, penalty = 0. With L∞ this is naturally
    zero rather than relying on a clip."""
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
    """V=Ω=0.3 → max(|V|, |Ω|) = 0.3 → penalty 0.7. The L∞ formulation
    gives no benefit to adding Ω alongside V — same penalty as (0.3, 0)
    or (0, 0.3) — which is the whole point of the rewrite."""
    term = StallPenaltyReward()
    assert term(_ctx_with_action(0.3, 0.3)) == pytest.approx(0.7, abs=1e-6)


def test_pure_rotation_matches_pure_thrust() -> None:
    """L∞ symmetry: (V, 0) and (0, V) must give identical penalties.
    The L2 formulation already shared this property at full magnitude
    (both → 0), but it differed at intermediate values; L∞ matches at
    every magnitude."""
    term = StallPenaltyReward()
    for mag in (0.1, 0.25, 0.5, 0.75):
        assert term(_ctx_with_action(mag, 0.0)) == pytest.approx(
            term(_ctx_with_action(0.0, mag)), abs=1e-6
        )


def test_adding_omega_to_v_does_not_help() -> None:
    """Diagnostic for the failure mode that motivated the rewrite: in
    L2, going from (0.3, 0) to (0.3, 0.3) reduced penalty from 0.7 to
    ~0.576; in L∞ both give 0.7. This is what removes the spurious
    radial pull toward the (1, 1) corner."""
    term = StallPenaltyReward()
    pure   = term(_ctx_with_action(0.3, 0.0))
    mixed  = term(_ctx_with_action(0.3, 0.3))
    assert pure == pytest.approx(mixed, abs=1e-6)


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
