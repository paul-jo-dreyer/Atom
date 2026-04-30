"""Tests for ObstacleContactPenalty.

Two layers:
  1. Pure-python unit tests on the reward term: it just forwards
     `info["obstacle_contact_frac"]`, so the contract is straightforward.
  2. Env-integration test that drives a dynamic robot into the wall and
     verifies the env populates `info["robot_contacts"]` and
     `info["obstacle_contact_frac"]` correctly. This one imports sim_py.
"""

from __future__ import annotations

import numpy as np
import pytest

from AtomGym.action_observation import ActionView, ObsView
from AtomGym.rewards import ObstacleContactPenalty, RewardContext


def _ctx_with_info(info: dict) -> RewardContext:
    """Build a RewardContext with the given info; everything else is zero."""
    return RewardContext(
        obs=np.zeros(11, dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
        prev_obs=None,
        prev_action=None,
        info=info,
        obs_view=ObsView(n_robots=1),
        action_view=ActionView(),
        field_x_half=0.375,
        field_y_half=0.225,
        goal_y_half=0.06,
        goal_extension=0.06,
        dt=1.0 / 30.0,
    )


# ---------------------------------------------------------------------------
# Unit tests on the reward term
# ---------------------------------------------------------------------------


def test_no_contact_returns_zero() -> None:
    term = ObstacleContactPenalty()
    assert term(_ctx_with_info({"obstacle_contact_frac": 0.0})) == pytest.approx(0.0)


def test_full_contact_returns_one() -> None:
    """Pinned-to-wall case: every substep had an obstacle contact."""
    term = ObstacleContactPenalty()
    assert term(_ctx_with_info({"obstacle_contact_frac": 1.0})) == pytest.approx(1.0)


def test_half_contact_returns_half() -> None:
    """Brief bounce off a wall mid control-step: 1 of 2 substeps in contact."""
    term = ObstacleContactPenalty()
    assert term(_ctx_with_info({"obstacle_contact_frac": 0.5})) == pytest.approx(0.5)


def test_missing_info_key_returns_zero() -> None:
    """If the env never populated the field, the term degrades to zero
    rather than crashing — useful for hand-built contexts in tests."""
    term = ObstacleContactPenalty()
    assert term(_ctx_with_info({})) == pytest.approx(0.0)


def test_negative_weight_produces_negative_contribution() -> None:
    """Standard convention: positive return × negative weight = penalty."""
    term = ObstacleContactPenalty(weight=-0.5)
    raw = term(_ctx_with_info({"obstacle_contact_frac": 1.0}))
    assert raw == pytest.approx(1.0)
    assert term.weight * raw == pytest.approx(-0.5)


def test_term_name() -> None:
    """Name must be stable — used as the TB log key under reward/."""
    assert ObstacleContactPenalty.name == "obstacle_contact"


# ---------------------------------------------------------------------------
# Env-integration: drive into a wall, check info is populated correctly
# ---------------------------------------------------------------------------


try:
    from AtomGym.environments import AtomSoloEnv
except RuntimeError as e:
    pytest.skip(f"AtomSim release build not available: {e}", allow_module_level=True)


def _make_env_at_wall() -> AtomSoloEnv:
    """Construct a Level-1 env, then teleport the robot to just left of
    the +x wall, slightly above the goal mouth so contacting the wall is
    actually possible (the goal mouth is a gap in the wall)."""
    env = AtomSoloEnv(
        physics_dt=1.0 / 60.0,
        control_dt=1.0 / 30.0,  # action_repeat = 2
        max_episode_steps=400,
        seed=0,
    )
    env.reset(seed=0)
    # Position 5 mm shy of the right wall, y above the goal mouth, heading
    # +x so a forward command drives straight into the wall.
    env.robot.set_state(np.array([0.370, 0.10, 0.0, 0.0, 0.0], dtype=np.float32))
    return env


def test_env_populates_robot_contacts_and_frac_when_pinned() -> None:
    """Drive into the wall for many steps; once pinned, every substep
    should have an obstacle contact ⟹ frac → 1.0."""
    env = _make_env_at_wall()
    full_forward = np.array([1.0, 0.0], dtype=np.float32)

    # Step long enough that the robot is fully wedged (motor lag has
    # settled, body has come to rest against the wall).
    final_info = None
    for _ in range(40):
        _obs, _r, term, trunc, info = env.step(full_forward)
        final_info = info
        if term or trunc:
            break

    assert final_info is not None
    assert "robot_contacts" in final_info
    assert "obstacle_contact_frac" in final_info
    assert final_info["obstacle_contact_frac"] == pytest.approx(1.0)
    # At least one of the contacts should be a wall contact with non-zero
    # impulse — sanity that the per-contact data is also coming through.
    contacts = final_info["robot_contacts"]
    assert len(contacts) > 0
    import sim_py
    wall_hits = [
        c for c in contacts
        if c.other_category & sim_py.CATEGORY_WALL and c.normal_impulse > 0.0
    ]
    assert len(wall_hits) > 0


def test_env_populates_zero_frac_in_open_field() -> None:
    """A robot driving forward starting from the origin doesn't touch
    anything for a while; frac should stay 0."""
    env = AtomSoloEnv(
        physics_dt=1.0 / 60.0,
        control_dt=1.0 / 30.0,
        max_episode_steps=400,
        seed=0,
    )
    env.reset(seed=0)
    # Force the robot to a known clean spot in the middle of the field.
    env.robot.set_state(np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    for _ in range(5):
        _obs, _r, _term, _trunc, info = env.step(
            np.array([0.5, 0.0], dtype=np.float32)
        )
        assert info["obstacle_contact_frac"] == pytest.approx(0.0)
        assert info["robot_contacts"] == []


def test_env_includes_obstacle_contact_in_reward_breakdown() -> None:
    """End-to-end sanity: with ObstacleContactPenalty on the reward list,
    the breakdown has the term, and its sign matches the weight × frac."""
    env = AtomSoloEnv(
        physics_dt=1.0 / 60.0,
        control_dt=1.0 / 30.0,
        max_episode_steps=400,
        rewards=[ObstacleContactPenalty(weight=-0.5)],
        seed=0,
    )
    env.reset(seed=0)
    env.robot.set_state(np.array([0.370, 0.10, 0.0, 0.0, 0.0], dtype=np.float32))

    full_forward = np.array([1.0, 0.0], dtype=np.float32)
    # Settle into the wall.
    for _ in range(40):
        _obs, _r, term, trunc, info = env.step(full_forward)
        if term or trunc:
            break

    breakdown = info["reward_breakdown"]
    assert "obstacle_contact" in breakdown
    # Weight is -0.5, frac is ~1.0 once pinned ⟹ contribution ~ -0.5.
    assert breakdown["obstacle_contact"] == pytest.approx(-0.5, abs=1e-6)
