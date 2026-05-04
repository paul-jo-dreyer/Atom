"""Tests for AtomTeamEnv (1v1 self-play scaffolding).

Covers: construction, observation shape, reset placement, step roundtrip,
goal events from the learner's POV, opponent_policy hook contract, and
robot-robot contact accounting in info."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from AtomGym.environments import AtomTeamEnv
except RuntimeError as e:
    pytest.skip(f"AtomSim release build not available: {e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Construction & shapes
# ---------------------------------------------------------------------------


def test_action_space_is_2d() -> None:
    env = AtomTeamEnv()
    assert env.action_space.shape == (2,)
    assert env.action_view.total_dim == 2


def test_observation_space_is_20d() -> None:
    """[ball(4) | learner(8) | opp(8)] = 20."""
    env = AtomTeamEnv()
    assert env.observation_space.shape == (20,)
    assert env.obs_view.total_dim == 20
    assert env.obs_view.n_robots == 2


def test_two_robots_constructed() -> None:
    env = AtomTeamEnv()
    assert env.robot is not None
    assert env.opponent is not None
    assert env.robot is not env.opponent


# ---------------------------------------------------------------------------
# Reset placement
# ---------------------------------------------------------------------------


def test_reset_places_learner_on_negative_x_half() -> None:
    """Learner attacks +x → spawns in -x half. Opponent in +x half.
    Margin pulls in from the centre line so no robot starts exactly at x=0."""
    env = AtomTeamEnv(seed=0)
    for seed in range(20):
        env.reset(seed=seed)
        learner_x = float(env.robot.state[0])
        opponent_x = float(env.opponent.state[0])
        assert learner_x < 0.0, f"seed {seed}: learner_x={learner_x}"
        assert opponent_x > 0.0, f"seed {seed}: opponent_x={opponent_x}"


def test_reset_returns_correct_shape() -> None:
    env = AtomTeamEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (20,)
    assert obs.dtype == np.float32
    assert info == {}


def test_reset_obs_in_unit_box() -> None:
    """Every component of the obs is normalized to [-1, 1]."""
    env = AtomTeamEnv(seed=0)
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        assert np.all(obs >= -1.0 - 1e-6), f"seed {seed}: min={obs.min()}"
        assert np.all(obs <= +1.0 + 1e-6), f"seed {seed}: max={obs.max()}"


# ---------------------------------------------------------------------------
# Step roundtrip
# ---------------------------------------------------------------------------


def test_step_returns_correct_tuple() -> None:
    env = AtomTeamEnv(seed=0)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert obs.shape == (20,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "scored_for_us" in info
    assert "scored_against_us" in info
    assert "robot_contacts" in info
    assert "obstacle_contact_frac" in info


def test_step_count_advances_one_per_control_step() -> None:
    env = AtomTeamEnv(physics_dt=1.0 / 60.0, control_dt=1.0 / 20.0, seed=0)  # repeat=3
    env.reset(seed=0)
    for _ in range(5):
        env.step(np.zeros(2, dtype=np.float32))
    assert env._step_count == 5


def test_truncation_at_default_max_steps() -> None:
    """Spin in place — no goal possible — should truncate at exactly 800."""
    env = AtomTeamEnv(seed=0)
    env.reset(seed=0)
    truncated = False
    steps = 0
    while not truncated:
        steps += 1
        _, _, terminated, truncated, _ = env.step(
            np.array([0.0, 1.0], dtype=np.float32)
        )
        if terminated or steps > 1000:
            break
    assert steps == 800
    assert truncated is True


# ---------------------------------------------------------------------------
# Goal events from the LEARNER's perspective
# ---------------------------------------------------------------------------


def test_ball_in_plus_x_goal_is_for_us() -> None:
    """Learner attacks +x, so ball fully past +x line = scored_for_us."""
    env = AtomTeamEnv(seed=0)
    env.reset(seed=0)
    radius = env.ball_radius
    bx = env.field_x_half + radius + 0.005
    env.ball.set_state(np.array([bx, 0.0, 0.0, 0.0], dtype=np.float32))
    _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["scored_for_us"] is True
    assert info["scored_against_us"] is False
    assert terminated is True


def test_ball_in_minus_x_goal_is_against_us() -> None:
    """Ball fully past -x line = scored_against_us (opponent scored)."""
    env = AtomTeamEnv(seed=0)
    env.reset(seed=0)
    radius = env.ball_radius
    bx = -(env.field_x_half + radius + 0.005)
    env.ball.set_state(np.array([bx, 0.0, 0.0, 0.0], dtype=np.float32))
    _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["scored_for_us"] is False
    assert info["scored_against_us"] is True
    assert terminated is True


# ---------------------------------------------------------------------------
# Opponent policy hook
# ---------------------------------------------------------------------------


def test_default_opponent_policy_keeps_opponent_still() -> None:
    """With zero opponent action and zero initial velocity, opponent should
    barely move over a few steps (any motion comes from sim noise / contact)."""
    env = AtomTeamEnv(seed=0)
    env.reset(seed=0)
    # Park the opponent at a known spot with zero velocity.
    env.opponent.set_state(np.array([0.10, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    initial_x = float(env.opponent.state[0])
    initial_y = float(env.opponent.state[1])
    for _ in range(10):
        env.step(np.zeros(2, dtype=np.float32))
    # Should not have drifted more than a millimetre or two without contact.
    drift = abs(float(env.opponent.state[0]) - initial_x) + abs(float(env.opponent.state[1]) - initial_y)
    assert drift < 0.01, f"opponent drifted {drift} m with zero action"


def test_opponent_policy_receives_20d_observation() -> None:
    """The hook is called with the opponent's perspective obs, shape (20,)."""
    captured = {}

    def recording_policy(obs):
        captured["obs"] = obs.copy()
        return np.zeros(2, dtype=np.float32)

    env = AtomTeamEnv(seed=0, opponent_policy=recording_policy)
    env.reset(seed=0)
    env.step(np.zeros(2, dtype=np.float32))
    assert "obs" in captured
    assert captured["obs"].shape == (20,)
    assert captured["obs"].dtype == np.float32


def test_set_opponent_policy_swap() -> None:
    env = AtomTeamEnv(seed=0)
    n_calls = [0]

    def counting_policy(_obs):
        n_calls[0] += 1
        return np.zeros(2, dtype=np.float32)

    env.set_opponent_policy(counting_policy)
    env.reset(seed=0)
    env.step(np.zeros(2, dtype=np.float32))
    env.step(np.zeros(2, dtype=np.float32))
    assert n_calls[0] == 2

    # Swap back to default — counting should stop.
    env.set_opponent_policy(None)
    env.step(np.zeros(2, dtype=np.float32))
    assert n_calls[0] == 2


def test_opponent_returning_wrong_shape_raises() -> None:
    def bad_policy(_obs):
        return np.zeros(3, dtype=np.float32)

    env = AtomTeamEnv(seed=0, opponent_policy=bad_policy)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros(2, dtype=np.float32))


def test_opponent_can_actually_drive() -> None:
    """If the opponent emits forward thrust, it should move in +x_body
    direction. Place it facing +x at origin and verify."""
    def forward_policy(_obs):
        return np.array([1.0, 0.0], dtype=np.float32)

    env = AtomTeamEnv(seed=0, opponent_policy=forward_policy)
    env.reset(seed=0)
    # Plant opponent facing +x at a known spot, away from walls.
    env.opponent.set_state(np.array([0.10, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    initial_x = float(env.opponent.state[0])
    for _ in range(10):
        env.step(np.zeros(2, dtype=np.float32))
    # Should have moved noticeably in +x (facing +x, full thrust).
    assert float(env.opponent.state[0]) > initial_x + 0.01


# ---------------------------------------------------------------------------
# Contact accounting
# ---------------------------------------------------------------------------


def test_robot_robot_contact_appears_in_info() -> None:
    """Plant the two robots overlapping. Box2D will resolve the contact;
    info["robot_contacts"] should report at least one CATEGORY_ROBOT contact."""
    import sim_py
    env = AtomTeamEnv(seed=0)
    env.reset(seed=0)
    # Place them at the same point, both facing +x. Box2D will eject.
    env.robot.set_state(np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    env.opponent.set_state(np.array([0.02, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    # Move ball away so it doesn't tangle in the contact list.
    env.ball.set_state(np.array([0.30, 0.20, 0.0, 0.0], dtype=np.float32))
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    contacts = info["robot_contacts"]
    robot_contacts = [c for c in contacts if c.other_category == sim_py.CATEGORY_ROBOT]
    assert len(robot_contacts) > 0, (
        f"expected at least one robot-robot contact, got categories: "
        f"{[c.other_category for c in contacts]}"
    )


def test_obstacle_contact_frac_includes_robot_robot() -> None:
    """Robot-robot contact counts as an obstacle (per _OBSTACLE_CATEGORIES)."""
    env = AtomTeamEnv(seed=0)
    env.reset(seed=0)
    env.robot.set_state(np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    env.opponent.set_state(np.array([0.02, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    env.ball.set_state(np.array([0.30, 0.20, 0.0, 0.0], dtype=np.float32))
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["obstacle_contact_frac"] > 0.0


# ---------------------------------------------------------------------------
# opponent_view canonical-view helper
# ---------------------------------------------------------------------------


def test_opponent_view_validates_shape() -> None:
    env = AtomTeamEnv()
    with pytest.raises(ValueError, match="shape"):
        env.opponent_view(np.zeros(17, dtype=np.float32))


def test_opponent_view_swaps_robot_blocks() -> None:
    """The opp's robot block (slot 1 in learner view) should move to the
    self slot (slot 0) in opponent view, with the mirror sign-flips applied."""
    env = AtomTeamEnv(seed=0)
    env.reset(seed=0)
    # Plant distinguishable states for learner and opp, ball off-axis.
    env.robot.set_state(np.array([-0.10, 0.05, 0.5, 0.10, 0.20], dtype=np.float32))
    env.opponent.set_state(np.array([+0.15, -0.08, 1.2, 0.05, -0.30], dtype=np.float32))
    env.ball.set_state(np.array([0.05, 0.10, 0.50, -0.20], dtype=np.float32))

    learner_obs = env._build_learner_obs()
    opp_view = env.opponent_view(learner_obs)

    # The "self" block of opp_view (indices 4..11) must be the opp's state,
    # mirrored (px and dx flipped, cos and ω flipped).
    # opp_view self block: px should be -(+0.15)/field_x_half (mirrored).
    expected_opp_self_px = -(+0.15) / env.field_x_half
    assert opp_view[4] == pytest.approx(expected_opp_self_px, abs=1e-5)
    # py unchanged
    expected_opp_self_py = -0.08 / env.field_y_half
    assert opp_view[5] == pytest.approx(expected_opp_self_py, abs=1e-5)


def test_opponent_view_mirrors_ball_x_components() -> None:
    """ball.px and ball.vx should be sign-flipped; py and vy preserved."""
    env = AtomTeamEnv(seed=0)
    env.reset(seed=0)
    env.ball.set_state(np.array([0.10, 0.05, 0.40, -0.25], dtype=np.float32))
    learner_obs = env._build_learner_obs()
    opp_view = env.opponent_view(learner_obs)

    assert opp_view[0] == pytest.approx(-learner_obs[0], abs=1e-7)  # px flipped
    assert opp_view[1] == pytest.approx(+learner_obs[1], abs=1e-7)  # py kept
    assert opp_view[2] == pytest.approx(-learner_obs[2], abs=1e-7)  # vx flipped
    assert opp_view[3] == pytest.approx(+learner_obs[3], abs=1e-7)  # vy kept


def test_opponent_view_matches_direct_build_with_mirror() -> None:
    """opponent_view(learner_obs) must equal build_observation(self=opp,
    others=[learner], mirror=True) up to float precision. This is the
    correctness guarantee — we're producing the same vector that the
    canonical build path would produce, just via permutation+flip on the
    already-built learner obs."""
    from AtomGym.action_observation import (
        OMEGA_MAX_DEFAULT,
        V_MAX_DEFAULT,
        build_observation,
    )

    env = AtomTeamEnv(seed=0)
    for seed in range(5):
        env.reset(seed=seed)
        # Vary states across iterations.
        env.robot.set_state(np.array([-0.10 + 0.01 * seed, 0.05, 0.5, 0.10, 0.20], dtype=np.float32))
        env.opponent.set_state(np.array([+0.15 - 0.02 * seed, -0.08, 1.2, 0.05, -0.30], dtype=np.float32))
        env.ball.set_state(np.array([0.05 + 0.03 * seed, 0.10, 0.50, -0.20], dtype=np.float32))

        learner_obs = env._build_learner_obs()
        opp_via_helper = env.opponent_view(learner_obs)
        opp_via_direct = build_observation(
            field_x_half=env.field_x_half,
            field_y_half=env.field_y_half,
            ball_state=np.asarray(env.ball.state, dtype=np.float32),
            self_state_5d=np.asarray(env.opponent.state, dtype=np.float32),
            others_states_5d=(np.asarray(env.robot.state, dtype=np.float32),),
            mirror=True,
            v_max=V_MAX_DEFAULT,
            omega_max=OMEGA_MAX_DEFAULT,
        )
        np.testing.assert_allclose(opp_via_helper, opp_via_direct, atol=1e-6)


def test_opponent_view_is_involution() -> None:
    """Applying the helper twice should round-trip to the original obs:
    swap-then-swap returns the slots, mirror-then-mirror cancels the signs."""
    env = AtomTeamEnv(seed=0)
    env.reset(seed=42)
    obs = env._build_learner_obs()
    twice = env.opponent_view(env.opponent_view(obs))
    np.testing.assert_allclose(twice, obs, atol=1e-6)


def test_opponent_view_does_not_mutate_input() -> None:
    env = AtomTeamEnv(seed=0)
    env.reset(seed=0)
    obs = env._build_learner_obs()
    snapshot = obs.copy()
    env.opponent_view(obs)
    np.testing.assert_array_equal(obs, snapshot)


def test_last_opponent_action_starts_none_and_populates_after_step() -> None:
    """The GIF eval callback reads `env.last_opponent_action` to render
    the opponent's control panel. Pin the contract: None before any
    step (so the panel falls back to zeros for the first frame), set
    to the policy's raw output after step."""
    def constant_opponent(_obs):
        return np.array([0.6, -0.4], dtype=np.float32)

    env = AtomTeamEnv(seed=0, opponent_policy=constant_opponent)
    env.reset(seed=0)
    assert env.last_opponent_action is None

    env.step(np.zeros(2, dtype=np.float32))
    assert env.last_opponent_action is not None
    np.testing.assert_allclose(
        env.last_opponent_action, np.array([0.6, -0.4], dtype=np.float32)
    )

    # Reset clears it again.
    env.reset(seed=0)
    assert env.last_opponent_action is None


def test_manipulator_kwarg_attaches_pusher_polygon() -> None:
    """`manipulator='default_pusher'` should load the JSON polygon and
    attach it to BOTH robots in the team env. Default None ⟹ empty
    manipulator_parts (bare body)."""
    env_bare = AtomTeamEnv()
    assert list(env_bare._robot_cfg.manipulator_parts) == []

    env_pusher = AtomTeamEnv(manipulator="default_pusher")
    parts = list(env_pusher._robot_cfg.manipulator_parts)
    assert len(parts) == 1, f"expected 1 part, got {len(parts)}"
    # default_pusher.json has a 4-vertex trapezoid.
    assert len(parts[0]) == 4


def test_manipulator_kwarg_unknown_name_raises() -> None:
    with pytest.raises(FileNotFoundError, match="manipulator config not found"):
        AtomTeamEnv(manipulator="this_pusher_does_not_exist")


# ---------------------------------------------------------------------------
# ball_touched flag — credit-hack guard (team variant)
# ---------------------------------------------------------------------------


def test_ball_touched_starts_false_in_team_env() -> None:
    env = AtomTeamEnv(seed=0)
    env.reset(seed=0)
    env.robot.set_state(np.array([-0.20, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    env.opponent.set_state(np.array([+0.20, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    env.ball.set_state(np.array([0.0, 0.10, 0.0, 0.0], dtype=np.float32))
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["ball_touched"] is False


def test_ball_touched_flips_on_learner_contact_in_team_env() -> None:
    env = AtomTeamEnv(seed=0)
    env.reset(seed=0)
    env.robot.set_state(np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    env.opponent.set_state(np.array([+0.30, 0.20, 0.0, 0.0, 0.0], dtype=np.float32))
    env.ball.set_state(np.array([0.02, 0.0, 0.0, 0.0], dtype=np.float32))
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["ball_touched"] is True


def test_ball_touched_flips_on_opponent_contact_in_team_env() -> None:
    """Opponent influence is part of the environment from the learner's POV.
    A goal where only the opponent touched the ball is a defensive failure
    the learner needs to learn from — so the touch flag must flip True even
    when the learner stays clear."""
    env = AtomTeamEnv(seed=0)
    env.reset(seed=0)
    # Learner stashed far from the ball.
    env.robot.set_state(np.array([-0.30, 0.20, 0.0, 0.0, 0.0], dtype=np.float32))
    # Opponent overlapping ball at +x side.
    env.opponent.set_state(np.array([+0.15, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    env.ball.set_state(np.array([+0.17, 0.0, 0.0, 0.0], dtype=np.float32))
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["ball_touched"] is True


def test_team_spurious_goal_suppresses_reward_when_neither_robot_touches() -> None:
    """Random fling into +x goal with NO contact from either robot must not
    credit the policy — pure init noise."""
    from AtomGym.rewards import GoalScoredReward
    env = AtomTeamEnv(seed=0, rewards=[GoalScoredReward(weight=20.0)])
    env.reset(seed=0)
    radius = env.ball_radius
    env.robot.set_state(np.array([-0.30, 0.20, 0.0, 0.0, 0.0], dtype=np.float32))
    env.opponent.set_state(np.array([+0.30, -0.20, 0.0, 0.0, 0.0], dtype=np.float32))
    bx = env.field_x_half + radius + 0.005
    env.ball.set_state(np.array([bx, 0.0, 0.0, 0.0], dtype=np.float32))
    _, reward, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["scored_for_us"] is True
    assert info["ball_touched"] is False
    assert terminated is True
    assert reward == 0.0


def test_team_opponent_scored_against_us_credits_negative_reward() -> None:
    """Opponent touches the ball, then the ball ends up in the learner's
    own goal. The learner SHOULD see the negative reward — defending
    against opponent attacks is a real skill it needs to learn.

    Uses control_dt=1/20 (action_repeat=3) so the ball-velocity-based
    goal crossing on step 2 has multiple substeps to be detected at —
    a teleport-into-goal-chamber doesn't work cleanly after Box2D has
    resolved contact in step 1, and a single-substep velocity flight
    can be bounced back by goal-chamber walls before the substep ends."""
    from AtomGym.rewards import GoalScoredReward
    env = AtomTeamEnv(
        physics_dt=1.0 / 60.0,
        control_dt=1.0 / 20.0,  # action_repeat=3
        seed=0,
        rewards=[GoalScoredReward(weight=20.0)],
    )
    env.reset(seed=0)
    # Step 1: opponent overlaps ball, learner stays clear. Latches
    # ball_touched=True.
    env.robot.set_state(np.array([-0.30, 0.20, 0.0, 0.0, 0.0], dtype=np.float32))
    env.opponent.set_state(np.array([+0.15, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    env.ball.set_state(np.array([+0.17, 0.0, 0.0, 0.0], dtype=np.float32))
    _, _, _, _, info_1 = env.step(np.zeros(2, dtype=np.float32))
    assert info_1["ball_touched"] is True
    # Step 2: separate the robots; place ball just inside -x line moving
    # in -x at 5 m/s. With substep dt=1/60, the ball travels ~0.083 m per
    # substep — fully crosses the line on substep 1, goal fires.
    env.robot.set_state(np.array([+0.30, 0.20, 0.0, 0.0, 0.0], dtype=np.float32))
    env.opponent.set_state(np.array([+0.30, -0.20, 0.0, 0.0, 0.0], dtype=np.float32))
    radius = env.ball_radius
    bx_start = -env.field_x_half + radius * 0.1
    env.ball.set_state(np.array([bx_start, 0.0, -5.0, 0.0], dtype=np.float32))
    _, reward, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["scored_against_us"] is True
    assert info["ball_touched"] is True
    assert terminated is True
    assert reward == pytest.approx(-20.0)


def test_opponent_canonical_turn_inverts_in_world() -> None:
    """An opponent policy emitting "+CCW in canonical view" (Ω=+1) should
    cause the opponent to turn CW in WORLD frame, since the action mirror
    negates Ω on application. This is the core check that the
    obs-mirror + action-mirror pair is wired correctly."""
    def turn_ccw_canonical(_obs):
        return np.array([0.0, 1.0], dtype=np.float32)

    env = AtomTeamEnv(seed=0, opponent_policy=turn_ccw_canonical)
    env.reset(seed=0)
    # Park opp at a known pose, away from walls/ball.
    env.opponent.set_state(np.array([0.10, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    env.ball.set_state(np.array([-0.30, 0.20, 0.0, 0.0], dtype=np.float32))
    initial_theta = float(env.opponent.state[2])
    for _ in range(5):
        env.step(np.zeros(2, dtype=np.float32))
    final_theta = float(env.opponent.state[2])
    # Ω flipped on application → world frame turn rate is negative → θ decreases.
    assert final_theta < initial_theta - 0.05, (
        f"expected θ to decrease (CW in world), got {initial_theta} → {final_theta}"
    )
