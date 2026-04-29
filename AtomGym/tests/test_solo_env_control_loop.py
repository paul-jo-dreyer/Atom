"""Integration tests for control-vs-physics rate, action repeat, fully-
crosses goal detection, and the new 800-step truncation default. Imports
sim_py."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from AtomGym.environments import AtomSoloEnv
except RuntimeError as e:
    pytest.skip(f"AtomSim release build not available: {e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_default_action_repeat_is_one() -> None:
    env = AtomSoloEnv()
    assert env.physics_dt == pytest.approx(1.0 / 60.0)
    assert env.control_dt == pytest.approx(env.physics_dt)
    assert env.action_repeat == 1


def test_control_dt_two_x_physics_dt_gives_repeat_two() -> None:
    env = AtomSoloEnv(physics_dt=1.0 / 60.0, control_dt=1.0 / 30.0)
    assert env.action_repeat == 2
    assert env.control_dt == pytest.approx(1.0 / 30.0)


def test_control_dt_three_x_physics_dt_gives_repeat_three() -> None:
    env = AtomSoloEnv(physics_dt=1.0 / 60.0, control_dt=1.0 / 20.0)
    assert env.action_repeat == 3


def test_non_integer_ratio_rejected() -> None:
    """0.025 / 0.01 = 2.5 — fractional substep count is meaningless."""
    with pytest.raises(ValueError, match="integer multiple"):
        AtomSoloEnv(physics_dt=0.01, control_dt=0.025)


def test_control_dt_below_physics_dt_rejected() -> None:
    """Can't run control faster than physics — ratio < 1."""
    with pytest.raises(ValueError):
        AtomSoloEnv(physics_dt=1.0 / 60.0, control_dt=1.0 / 120.0)


def test_zero_physics_dt_rejected() -> None:
    with pytest.raises(ValueError):
        AtomSoloEnv(physics_dt=0.0)


def test_default_max_episode_steps_is_800() -> None:
    """User-requested default — episodes truncate at 800 control steps."""
    env = AtomSoloEnv()
    assert env.max_episode_steps == 800


# ---------------------------------------------------------------------------
# Step counting and timing
# ---------------------------------------------------------------------------


def test_step_count_advances_one_per_control_step() -> None:
    """Even with action_repeat > 1, env.step() advances step_count by 1."""
    env = AtomSoloEnv(physics_dt=1.0 / 60.0, control_dt=1.0 / 20.0)  # repeat=3
    env.reset(seed=0)
    for i in range(5):
        env.step(np.zeros(2, dtype=np.float32))
    assert env._step_count == 5


def test_t_uses_control_dt() -> None:
    """env.t = step_count × control_dt — total elapsed sim time."""
    env = AtomSoloEnv(physics_dt=1.0 / 60.0, control_dt=1.0 / 30.0)  # repeat=2
    env.reset(seed=0)
    for _ in range(4):
        env.step(np.zeros(2, dtype=np.float32))
    # 4 control steps × 1/30 s = 0.133 s
    assert env.t == pytest.approx(4.0 / 30.0)


def test_truncation_at_800_steps() -> None:
    """Drive in circles until truncation triggers — should be exactly 800."""
    env = AtomSoloEnv(seed=0)
    env.reset(seed=0)
    truncated = False
    steps = 0
    # Use a reset config that won't accidentally score a goal.
    while not truncated:
        steps += 1
        _, _, terminated, truncated, _ = env.step(
            np.array([0.0, 1.0], dtype=np.float32)  # spin in place — no scoring possible
        )
        if terminated:
            break
        if steps > 1000:
            pytest.fail("Truncation did not trigger by step 1000")
    assert steps == 800
    assert truncated is True


# ---------------------------------------------------------------------------
# Action repeat actually advances the sim more
# ---------------------------------------------------------------------------


def test_action_repeat_advances_sim_proportionally() -> None:
    """Two envs with the same starting state, same action, but different
    action_repeat — the higher-repeat env should advance the robot
    further per env.step() (since the same action was held for more
    physics ticks)."""
    env_fast = AtomSoloEnv(physics_dt=1.0 / 60.0, control_dt=1.0 / 60.0, seed=0)  # repeat=1
    env_slow = AtomSoloEnv(physics_dt=1.0 / 60.0, control_dt=1.0 / 20.0, seed=0)  # repeat=3

    # Plant the same starting state in both envs.
    init = np.array([-0.20, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    env_fast.reset(seed=0); env_slow.reset(seed=0)
    env_fast.robot.set_state(init.copy())
    env_slow.robot.set_state(init.copy())
    env_fast.ball.set_state(np.array([0.30, 0.0, 0.0, 0.0], dtype=np.float32))
    env_slow.ball.set_state(np.array([0.30, 0.0, 0.0, 0.0], dtype=np.float32))

    # One env.step() with full forward thrust.
    a = np.array([1.0, 0.0], dtype=np.float32)
    env_fast.step(a)
    env_slow.step(a)

    px_fast = float(env_fast.robot.state[0])
    px_slow = float(env_slow.robot.state[0])
    # The 3-substep env should have moved noticeably further than the 1-substep one.
    assert px_slow > px_fast
    assert (px_slow - init[0]) > 2.0 * (px_fast - init[0])


# ---------------------------------------------------------------------------
# Goal-line "fully crosses" semantics
# ---------------------------------------------------------------------------


def test_ball_center_just_past_line_does_NOT_score() -> None:
    """If the ball CENTER is past the goal line but the ball is still
    straddling it (radius extends back inside the field), this should
    NOT count as a goal under the "fully crosses" rule."""
    env = AtomSoloEnv(seed=0)
    env.reset(seed=0)
    # Center past line by less than one radius — ball still straddling.
    radius = env.ball_radius
    bx = env.field_x_half + 0.5 * radius
    env.ball.set_state(np.array([bx, 0.0, 0.0, 0.0], dtype=np.float32))
    _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["scored_for_us"] is False
    assert terminated is False


def test_ball_fully_past_line_DOES_score() -> None:
    """Trailing edge of ball past the goal line in x AND ball in y-band:
    fully crossed → goal."""
    env = AtomSoloEnv(seed=0)
    env.reset(seed=0)
    radius = env.ball_radius
    # Place the ball so its trailing (left) edge is just past the line.
    bx = env.field_x_half + radius + 0.005
    env.ball.set_state(np.array([bx, 0.0, 0.0, 0.0], dtype=np.float32))
    _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["scored_for_us"] is True
    assert terminated is True


def test_ball_fully_past_line_outside_y_band_does_NOT_score() -> None:
    """Even if the ball is past the line in x, if it's outside the goal
    mouth in y, no goal."""
    env = AtomSoloEnv(seed=0)
    env.reset(seed=0)
    radius = env.ball_radius
    bx = env.field_x_half + radius + 0.005
    by = env.goal_y_half + 0.05  # well above the goal mouth
    env.ball.set_state(np.array([bx, by, 0.0, 0.0], dtype=np.float32))
    _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["scored_for_us"] is False
    assert terminated is False


# ---------------------------------------------------------------------------
# Early-exit on goal during a multi-substep control step
# ---------------------------------------------------------------------------


def test_goal_during_substep_terminates_immediately() -> None:
    """With action_repeat > 1, plant the ball just BEFORE the line. After a
    couple of substeps with the ball drifting in (we plant a velocity), the
    goal fires partway through the macro-step. The env should still report
    terminated=True at the end of step()."""
    env = AtomSoloEnv(physics_dt=1.0 / 60.0, control_dt=1.0 / 20.0, seed=0)  # repeat=3
    env.reset(seed=0)
    radius = env.ball_radius
    # Ball one radius's worth INSIDE the line, moving in +x at speed enough
    # to fully cross within 1 physics tick (1/60 s × 5 m/s = 0.083 m).
    bx = env.field_x_half - radius * 0.1
    env.ball.set_state(np.array([bx, 0.0, 5.0, 0.0], dtype=np.float32))
    _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is True
    assert truncated is False
    assert info["scored_for_us"] is True
