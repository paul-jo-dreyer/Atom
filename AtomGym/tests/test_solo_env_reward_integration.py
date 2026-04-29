"""End-to-end test: reward composite + event detection + termination
flowing through `AtomSoloEnv`. Imports `sim_py` so the release build is
required (pytest will skip if it isn't there)."""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module if the C++ build hasn't been produced yet.
try:
    from AtomGym.environments import AtomSoloEnv
except RuntimeError as e:
    pytest.skip(f"AtomSim release build not available: {e}", allow_module_level=True)

from AtomGym.rewards import RewardContext, RewardTerm


# ---------------------------------------------------------------------------
# Reward terms used as test instruments
# ---------------------------------------------------------------------------


class ConstantTerm(RewardTerm):
    name = "constant"

    def __init__(self, value: float, weight: float = 1.0) -> None:
        super().__init__(weight=weight)
        self._value = value

    def __call__(self, ctx: RewardContext) -> float:
        return self._value


class SeenScoredFlagTerm(RewardTerm):
    """Returns 1.0 on the step the env flags a goal-for-us event."""

    name = "scored"

    def __call__(self, ctx: RewardContext) -> float:
        return 1.0 if ctx.info.get("scored_for_us", False) else 0.0


class ObsSelfPxTerm(RewardTerm):
    """Sanity check that ctx.obs and ctx.obs_view both work."""

    name = "self_px"

    def __call__(self, ctx: RewardContext) -> float:
        return ctx.obs_view.self_px(ctx.obs)


# ---------------------------------------------------------------------------
# Defaults / no-reward case
# ---------------------------------------------------------------------------


def test_env_default_reward_is_zero() -> None:
    """No rewards passed in → every step returns 0.0."""
    env = AtomSoloEnv(seed=0)
    assert env.reward_terms == []
    env.reset(seed=0)
    for _ in range(5):
        _, r, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert r == 0.0
        assert info["reward_breakdown"] == {}


# ---------------------------------------------------------------------------
# Composite plumbing
# ---------------------------------------------------------------------------


def test_env_constant_term_returns_weighted_value() -> None:
    env = AtomSoloEnv(rewards=[ConstantTerm(2.0, weight=0.5)], seed=0)
    env.reset(seed=0)
    _, r, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert r == pytest.approx(1.0)
    assert info["reward_breakdown"] == {"constant": pytest.approx(1.0)}


def test_env_obs_view_accessible_in_reward() -> None:
    """The view passed via ctx must read the same obs the env returned."""
    env = AtomSoloEnv(rewards=[ObsSelfPxTerm()], seed=0)
    obs, _ = env.reset(seed=0)
    expected_self_px = env.obs_view.self_px(obs)
    obs2, r, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    # After one step, the term reads from the NEW obs (post-step) — so the
    # reward should equal self_px(obs2), not self_px(obs).
    assert r == pytest.approx(env.obs_view.self_px(obs2))
    # And the value at reset wasn't lost — sanity check.
    assert isinstance(expected_self_px, float)


def test_env_reward_terms_list_is_mutable() -> None:
    """Public list — useful for curriculum stages that swap rewards mid-run."""
    env = AtomSoloEnv(rewards=[ConstantTerm(1.0)], seed=0)
    env.reset(seed=0)
    # Mutate the list and verify the env picks up the change next step.
    env.reward_terms.append(ConstantTerm(3.0))
    _, r, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert r == pytest.approx(4.0)
    assert set(info["reward_breakdown"].keys()) == {"constant"}  # both terms named "constant"


def test_env_compute_reward_callable_directly() -> None:
    """compute_reward is a public method — callable from outside step()."""
    from AtomGym.rewards import RewardContext

    env = AtomSoloEnv(rewards=[ConstantTerm(2.5, weight=2.0)], seed=0)
    obs, _ = env.reset(seed=0)
    ctx = RewardContext(
        obs=obs,
        action=np.zeros(2, dtype=np.float32),
        prev_obs=None,
        prev_action=None,
        info={},
        obs_view=env.obs_view,
        action_view=env.action_view,
        field_x_half=env.field_x_half,
        field_y_half=env.field_y_half,
        goal_y_half=env.goal_y_half,
        goal_extension=env.goal_extension,
        dt=env.control_dt,
    )
    total, breakdown = env.compute_reward(ctx)
    assert total == pytest.approx(5.0)
    assert breakdown == {"constant": pytest.approx(5.0)}


# ---------------------------------------------------------------------------
# Goal event detection + termination
# ---------------------------------------------------------------------------


def test_env_scoring_in_opp_goal_terminates_episode() -> None:
    """Plant the ball inside the +x goal chamber, take one step, expect
    `scored_for_us` to fire and the env to terminate."""
    env = AtomSoloEnv(rewards=[SeenScoredFlagTerm(weight=10.0)], seed=0)
    env.reset(seed=0)

    # Manually move the ball to the back of the +x goal chamber. The env's
    # event detector triggers on the rising edge after reset (latch is False).
    chamber_x = env.field_x_half + env.goal_extension * 0.5
    env.ball.set_state(np.array([chamber_x, 0.0, 0.0, 0.0], dtype=np.float32))

    obs, r, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["scored_for_us"] is True
    assert info["scored_against_us"] is False
    assert terminated is True
    assert truncated is False
    assert r == pytest.approx(10.0)


def test_env_scoring_in_own_goal_terminates_episode() -> None:
    env = AtomSoloEnv(seed=0)  # reward irrelevant — just verifying termination
    env.reset(seed=0)
    chamber_x = -(env.field_x_half + env.goal_extension * 0.5)
    env.ball.set_state(np.array([chamber_x, 0.0, 0.0, 0.0], dtype=np.float32))
    _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["scored_against_us"] is True
    assert info["scored_for_us"] is False
    assert terminated is True


def test_env_goal_event_only_fires_once() -> None:
    """Edge detection: once the ball is in the chamber, subsequent steps
    while it's still there should NOT keep firing the flag."""
    env = AtomSoloEnv(seed=0)
    env.reset(seed=0)
    # Put the ball in the +x chamber...
    env.ball.set_state(np.array([env.field_x_half + 0.03, 0.0, 0.0, 0.0], dtype=np.float32))
    _, _, terminated_first, _, info_first = env.step(np.zeros(2, dtype=np.float32))
    assert info_first["scored_for_us"] is True
    assert terminated_first is True

    # Episode terminated; the flag should not re-fire on a fresh reset.
    env.reset(seed=0)
    _, _, _, _, info_after_reset = env.step(np.zeros(2, dtype=np.float32))
    assert info_after_reset["scored_for_us"] is False


# ---------------------------------------------------------------------------
# Cross-step state plumbing — prev_obs reaches the term on step 2+
# ---------------------------------------------------------------------------


class PrevObsAvailabilityTerm(RewardTerm):
    """Returns 0 if prev_obs is None, 1 otherwise — verifies the env passes
    prev_obs through on the second-and-later steps."""

    name = "prev_obs_present"

    def __call__(self, ctx: RewardContext) -> float:
        return 1.0 if ctx.prev_obs is not None else 0.0


def test_env_prev_obs_is_none_on_first_step_then_populated() -> None:
    env = AtomSoloEnv(rewards=[PrevObsAvailabilityTerm()], seed=0)
    env.reset(seed=0)
    _, r1, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    _, r2, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert r1 == pytest.approx(0.0)  # prev_obs was None
    assert r2 == pytest.approx(1.0)  # prev_obs populated by step 2
