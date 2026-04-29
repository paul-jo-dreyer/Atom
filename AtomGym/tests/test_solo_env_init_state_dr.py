"""Integration tests for the SoloEnv reset DR pipeline. Imports sim_py."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from AtomGym.environments import AtomSoloEnv, InitialStateRanges
except RuntimeError as e:
    pytest.skip(f"AtomSim release build not available: {e}", allow_module_level=True)


def _robot_state(env: AtomSoloEnv) -> np.ndarray:
    return np.asarray(env.robot.state, dtype=np.float32)


def _ball_state(env: AtomSoloEnv) -> np.ndarray:
    return np.asarray(env.ball.state, dtype=np.float32)


# ---------------------------------------------------------------------------
# Default behavior — zero initial velocities
# ---------------------------------------------------------------------------


def test_default_ranges_produce_zero_velocities() -> None:
    """With no `init_state_ranges` argument, reset should leave the robot
    and ball at rest (preserves original behavior before DR was added)."""
    env = AtomSoloEnv(seed=0)
    for s in range(5):
        env.reset(seed=s)
        rs = _robot_state(env)
        bs = _ball_state(env)
        assert rs[3] == 0.0  # body-frame v
        assert rs[4] == 0.0  # omega
        assert bs[2] == 0.0  # vx
        assert bs[3] == 0.0  # vy


def test_default_ranges_produce_random_poses() -> None:
    """The default config still randomizes pose; only velocities are zero."""
    env_a = AtomSoloEnv(seed=1)
    env_b = AtomSoloEnv(seed=2)
    env_a.reset()
    env_b.reset()
    assert not np.allclose(_robot_state(env_a)[:3], _robot_state(env_b)[:3])


# ---------------------------------------------------------------------------
# Velocity randomization — bounds and variance
# ---------------------------------------------------------------------------


def test_robot_speed_within_range() -> None:
    ranges = InitialStateRanges(robot_speed=(0.05, 0.15))
    env = AtomSoloEnv(init_state_ranges=ranges, seed=42)
    for s in range(20):
        env.reset(seed=s)
        v = _robot_state(env)[3]
        assert 0.05 <= v <= 0.15


def test_robot_omega_within_range() -> None:
    ranges = InitialStateRanges(robot_omega=(-0.5, 0.5))
    env = AtomSoloEnv(init_state_ranges=ranges, seed=42)
    for s in range(20):
        env.reset(seed=s)
        omega = _robot_state(env)[4]
        assert -0.5 <= omega <= 0.5


def test_ball_speed_within_range_polar() -> None:
    """Polar parameterization: speed bounds the magnitude of (vx, vy)."""
    speed_max = 1.5
    ranges = InitialStateRanges(ball_speed=(0.5, speed_max))
    env = AtomSoloEnv(init_state_ranges=ranges, seed=42)
    for s in range(20):
        env.reset(seed=s)
        bs = _ball_state(env)
        magnitude = float(np.hypot(bs[2], bs[3]))
        assert 0.5 - 1e-6 <= magnitude <= speed_max + 1e-6


def test_ball_zero_speed_yields_zero_velocity() -> None:
    """Speed range = (0,0) ⟹ vx = vy = 0 regardless of direction range."""
    ranges = InitialStateRanges(ball_speed=(0.0, 0.0))
    env = AtomSoloEnv(init_state_ranges=ranges, seed=42)
    for s in range(5):
        env.reset(seed=s)
        bs = _ball_state(env)
        assert bs[2] == 0.0 and bs[3] == 0.0


def test_velocity_randomization_has_variance() -> None:
    """Wide range should produce visible variance across resets — sanity
    check that we're actually sampling, not stuck on one value."""
    ranges = InitialStateRanges(
        robot_speed=(-0.2, 0.2),
        robot_omega=(-2.0, 2.0),
        ball_speed=(0.0, 1.0),
    )
    env = AtomSoloEnv(init_state_ranges=ranges, seed=42)
    speeds: list[float] = []
    omegas: list[float] = []
    ball_speeds: list[float] = []
    for s in range(50):
        env.reset(seed=s)
        rs = _robot_state(env)
        bs = _ball_state(env)
        speeds.append(float(rs[3]))
        omegas.append(float(rs[4]))
        ball_speeds.append(float(np.hypot(bs[2], bs[3])))
    assert np.std(speeds) > 0.05
    assert np.std(omegas) > 0.5
    assert np.std(ball_speeds) > 0.1


def test_different_seeds_produce_different_initial_velocities() -> None:
    """Two envs with the same DR config but different seeds must produce
    different initial velocities."""
    ranges = InitialStateRanges(robot_speed=(-0.2, 0.2), ball_speed=(0.5, 1.0))
    env_a = AtomSoloEnv(init_state_ranges=ranges)
    env_b = AtomSoloEnv(init_state_ranges=ranges)
    env_a.reset(seed=11)
    env_b.reset(seed=22)
    assert _robot_state(env_a)[3] != _robot_state(env_b)[3]
    a_ball = np.hypot(*_ball_state(env_a)[2:4])
    b_ball = np.hypot(*_ball_state(env_b)[2:4])
    assert a_ball != b_ball


# ---------------------------------------------------------------------------
# Mid-training swap
# ---------------------------------------------------------------------------


def test_init_state_ranges_can_be_swapped_mid_training() -> None:
    """For curriculum: reassign env.init_state_ranges between resets and
    the next reset should sample from the new config."""
    env = AtomSoloEnv(seed=0)  # defaults: zero velocities
    env.reset(seed=0)
    assert _robot_state(env)[3] == 0.0  # initial config: zero speed

    # Swap to a wide range and reset.
    env.init_state_ranges = InitialStateRanges(robot_speed=(0.10, 0.20))
    env.reset(seed=0)
    v = _robot_state(env)[3]
    assert 0.10 <= v <= 0.20
