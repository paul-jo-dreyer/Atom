"""Unit tests for InitialStateRanges (pure-python, no env / sim_py)."""

from __future__ import annotations

import math

import pytest

from AtomGym.environments import InitialStateRanges


def test_defaults_are_zero_velocity_random_pose() -> None:
    r = InitialStateRanges()
    assert r.robot_speed == (0.0, 0.0)
    assert r.robot_omega == (0.0, 0.0)
    assert r.ball_speed == (0.0, 0.0)
    assert r.robot_theta == (-math.pi, math.pi)
    assert r.ball_direction == (-math.pi, math.pi)
    assert r.robot_xy_margin == pytest.approx(0.04)
    assert r.ball_xy_margin == pytest.approx(0.04)


def test_dataclass_is_frozen() -> None:
    """Frozen so reset() can't accidentally mutate the config; users who want
    to swap configs reassign the whole attribute on the env."""
    r = InitialStateRanges()
    with pytest.raises(Exception):
        r.robot_speed = (0.1, 0.2)  # type: ignore[misc]


def test_validation_rejects_inverted_range() -> None:
    with pytest.raises(ValueError):
        InitialStateRanges(robot_speed=(0.5, 0.1))


def test_validation_rejects_inverted_theta() -> None:
    with pytest.raises(ValueError):
        InitialStateRanges(robot_theta=(1.0, -1.0))


def test_validation_rejects_negative_margin() -> None:
    with pytest.raises(ValueError):
        InitialStateRanges(robot_xy_margin=-0.01)


def test_equal_low_high_is_allowed() -> None:
    """A degenerate range (low == high) is fine — it just always samples
    the same value. Useful for pinning one DOF while randomizing others."""
    r = InitialStateRanges(robot_speed=(0.5, 0.5), ball_speed=(1.0, 1.0))
    assert r.robot_speed == (0.5, 0.5)
    assert r.ball_speed == (1.0, 1.0)
