"""Tests for RefEvalCallback — the win-rate-gated curriculum eval.

Uses a fake env + mock model so tests are fast and deterministic.
Outcome classification (W/L/T from info) is tested directly; goal-line
detection itself is covered by the team-env tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.logger import Logger

from AtomGym.training.ref_eval_callback import RefEvalCallback
from AtomGym.training.reference_opponent import ReferenceOpponent
from AtomGym.training.snapshot_pool import Snapshot, SnapshotPool
from AtomGym.training.win_rate_tracker import Outcome, WinRateTracker


_OBS_SPACE = spaces.Box(low=-1.0, high=+1.0, shape=(20,), dtype=np.float32)
_ACT_SPACE = spaces.Box(low=-1.0, high=+1.0, shape=(2,), dtype=np.float32)
_POLICY_KWARGS: dict[str, Any] = dict(
    net_arch=dict(pi=[128, 128], vf=[128, 128]),
    log_std_init=0.3,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeEnv:
    """Stands in for AtomTeamEnv in callback tests. step() returns a
    fixed (terminated, truncated, info) tuple supplied at construction
    so we can drive the outcome classifier deterministically."""

    def __init__(self, info: dict[str, Any], terminated: bool = True, truncated: bool = False) -> None:
        self._info = info
        self._terminated = terminated
        self._truncated = truncated
        self._opp_policy: Any = None

    def reset(self, **_kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        return np.zeros(18, dtype=np.float32), {}

    def step(self, _action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return np.zeros(18, dtype=np.float32), 0.0, self._terminated, self._truncated, self._info

    def set_opponent_policy(self, policy: Any) -> None:
        self._opp_policy = policy


class _MockModel:
    """Minimal stand-in for SB3 BaseAlgorithm used by BaseCallback."""

    def __init__(self) -> None:
        self.num_timesteps = 0
        self.logger = Logger(folder=None, output_formats=[])  # silent

    def get_env(self) -> None:
        return None

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        return np.zeros(2, dtype=np.float32), None


def _learner_state_dict_cpu() -> dict:
    from stable_baselines3.common.policies import ActorCriticPolicy
    policy = ActorCriticPolicy(
        observation_space=_OBS_SPACE,
        action_space=_ACT_SPACE,
        lr_schedule=lambda _: 0.0,
        **_POLICY_KWARGS,
    ).to("cpu")
    return {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}


def _make_snapshot(iteration: int) -> Snapshot:
    return Snapshot(
        state_dict=_learner_state_dict_cpu(),
        iteration=iteration,
        elo=1000.0,
        episodes_played=0,
        wins=0,
    )


def _make_callback(
    info_to_return: dict[str, Any],
    pool: SnapshotPool | None = None,
    *,
    eval_every: int = 100,
    episodes_per_cycle: int = 5,
    window_size: int = 10,
    promotion_threshold: float = 0.8,
) -> tuple[RefEvalCallback, _MockModel, SnapshotPool, ReferenceOpponent, WinRateTracker]:
    pool = pool if pool is not None else SnapshotPool(capacity=10)
    ref = ReferenceOpponent(_OBS_SPACE, _ACT_SPACE, _POLICY_KWARGS)
    tracker = WinRateTracker(window_size=window_size)

    cb = RefEvalCallback(
        eval_env_factory=lambda: _FakeEnv(info_to_return),
        pool=pool,
        reference=ref,
        tracker=tracker,
        eval_every=eval_every,
        episodes_per_cycle=episodes_per_cycle,
        promotion_threshold=promotion_threshold,
        verbose=0,
    )
    model = _MockModel()
    cb.init_callback(model)  # type: ignore[arg-type]
    return cb, model, pool, ref, tracker


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_invalid_eval_every() -> None:
    with pytest.raises(ValueError):
        _make_callback({}, eval_every=0)


def test_constructor_rejects_invalid_episodes_per_cycle() -> None:
    with pytest.raises(ValueError):
        _make_callback({}, episodes_per_cycle=0)


def test_constructor_rejects_invalid_promotion_threshold() -> None:
    with pytest.raises(ValueError):
        _make_callback({}, promotion_threshold=-0.1)
    with pytest.raises(ValueError):
        _make_callback({}, promotion_threshold=1.5)


# ---------------------------------------------------------------------------
# Empty pool: eval cycle no-ops
# ---------------------------------------------------------------------------


def test_empty_pool_no_ref_set() -> None:
    cb, model, pool, ref, tracker = _make_callback({"scored_for_us": True})
    model.num_timesteps = cb._eval_every  # trigger eval
    cb.on_step()
    assert ref.is_set is False
    assert len(tracker) == 0


# ---------------------------------------------------------------------------
# First eval cycle initialises the reference and runs episodes
# ---------------------------------------------------------------------------


def test_first_cycle_initialises_reference_to_pool_latest() -> None:
    pool = SnapshotPool(capacity=10)
    pool.add(_learner_state_dict_cpu(), iteration=10)
    pool.add(_learner_state_dict_cpu(), iteration=20)
    cb, model, _, ref, tracker = _make_callback(
        {"scored_for_us": True}, pool=pool, episodes_per_cycle=3, window_size=10
    )
    model.num_timesteps = cb._eval_every
    cb.on_step()
    # Ref initialised to latest pool entry.
    assert ref.loaded_iteration == 20
    # 3 episodes recorded (all WINs given the fake info).
    assert len(tracker) == 3


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------


def test_outcome_classification_win() -> None:
    pool = SnapshotPool()
    pool.add(_learner_state_dict_cpu(), iteration=1)
    cb, model, _, _, tracker = _make_callback(
        {"scored_for_us": True}, pool=pool, episodes_per_cycle=4, window_size=8
    )
    model.num_timesteps = cb._eval_every
    cb.on_step()
    assert all(s == Outcome.WIN.value for s in tracker._scores)


def test_outcome_classification_loss() -> None:
    pool = SnapshotPool()
    pool.add(_learner_state_dict_cpu(), iteration=1)
    cb, model, _, _, tracker = _make_callback(
        {"scored_against_us": True}, pool=pool, episodes_per_cycle=4, window_size=8
    )
    model.num_timesteps = cb._eval_every
    cb.on_step()
    assert all(s == Outcome.LOSS.value for s in tracker._scores)


def test_outcome_classification_draw() -> None:
    """Truncation with no goal events → DRAW."""
    pool = SnapshotPool()
    pool.add(_learner_state_dict_cpu(), iteration=1)
    # Build callback with a fake env that truncates rather than terminates.
    pool_ = pool
    ref = ReferenceOpponent(_OBS_SPACE, _ACT_SPACE, _POLICY_KWARGS)
    tracker = WinRateTracker(window_size=8)
    cb = RefEvalCallback(
        eval_env_factory=lambda: _FakeEnv({}, terminated=False, truncated=True),
        pool=pool_,
        reference=ref,
        tracker=tracker,
        eval_every=100,
        episodes_per_cycle=4,
        promotion_threshold=0.8,
        verbose=0,
    )
    model = _MockModel()
    cb.init_callback(model)  # type: ignore[arg-type]
    model.num_timesteps = cb._eval_every
    cb.on_step()
    assert all(s == Outcome.DRAW.value for s in tracker._scores)


# ---------------------------------------------------------------------------
# Window fill across cycles
# ---------------------------------------------------------------------------


def test_window_fills_across_multiple_cycles() -> None:
    pool = SnapshotPool()
    pool.add(_learner_state_dict_cpu(), iteration=1)
    cb, model, _, _, tracker = _make_callback(
        {"scored_for_us": True},
        pool=pool,
        eval_every=100,
        episodes_per_cycle=4,
        window_size=10,
    )
    # Cycle 1: 4 episodes
    model.num_timesteps = 100
    cb.on_step()
    assert len(tracker) == 4 and not tracker.is_full
    # Cycle 2: 8 total
    model.num_timesteps = 200
    cb.on_step()
    assert len(tracker) == 8 and not tracker.is_full
    # Cycle 3: should fill to 10 (capped) — we record 4 more, oldest 2 evicted.
    model.num_timesteps = 300
    cb.on_step()
    assert len(tracker) == 10 and tracker.is_full


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


def test_promotion_fires_when_threshold_met_and_new_snapshot_available() -> None:
    """Tracker full at 1.0 win rate, pool has a newer snapshot → promote."""
    pool = SnapshotPool(capacity=10)
    pool.add(_learner_state_dict_cpu(), iteration=10)  # initial ref
    cb, model, _, ref, tracker = _make_callback(
        {"scored_for_us": True},
        pool=pool,
        eval_every=100,
        episodes_per_cycle=10,
        window_size=10,
        promotion_threshold=0.8,
    )
    # Cycle 1: ref initialised to iter=10, window fills with 10 wins.
    model.num_timesteps = 100
    cb.on_step()
    assert ref.loaded_iteration == 10
    assert tracker.is_full
    assert tracker.win_rate == 1.0

    # Add a newer snapshot to the pool (simulating training progress).
    pool.add(_learner_state_dict_cpu(), iteration=20)

    # Cycle 2: should promote — but first the 10 fresh WIN outcomes will
    # arrive in the tracker. Since the deque is already full, the new
    # WINs replace old ones — rate stays 1.0. After episodes, the
    # promotion check fires.
    model.num_timesteps = 200
    cb.on_step()
    assert ref.loaded_iteration == 20  # promoted
    assert len(tracker) == 0  # tracker reset on promotion


def test_no_promote_when_pool_latest_equals_current_ref() -> None:
    """If pool hasn't grown since current ref was set, no promotion
    even with a perfect win rate."""
    pool = SnapshotPool()
    pool.add(_learner_state_dict_cpu(), iteration=10)
    cb, model, _, ref, tracker = _make_callback(
        {"scored_for_us": True},
        pool=pool,
        eval_every=100,
        episodes_per_cycle=10,
        window_size=10,
        promotion_threshold=0.5,  # easy threshold
    )
    model.num_timesteps = 100
    cb.on_step()
    assert ref.loaded_iteration == 10
    assert tracker.is_full and tracker.win_rate == 1.0
    # No newer snapshot — should NOT promote, tracker should NOT reset.
    assert len(tracker) == 10


def test_no_promote_below_threshold() -> None:
    """Window full but win rate < threshold → no promotion."""
    # Use Loss info so all outcomes are 0.0 → rate=0.0.
    pool = SnapshotPool()
    pool.add(_learner_state_dict_cpu(), iteration=10)
    pool.add(_learner_state_dict_cpu(), iteration=20)  # eligible for promotion
    cb, model, _, ref, tracker = _make_callback(
        {"scored_against_us": True},
        pool=pool,
        eval_every=100,
        episodes_per_cycle=10,
        window_size=10,
        promotion_threshold=0.8,
    )
    model.num_timesteps = 100
    cb.on_step()
    # Ref initialised to iter=20 (latest at time of init). Window full
    # of LOSSes → rate=0.0. No promotion.
    assert ref.loaded_iteration == 20
    assert tracker.is_full and tracker.win_rate == 0.0
    # Tracker not reset — keeps sliding.
    assert len(tracker) == 10


def test_tracker_resets_on_promotion() -> None:
    """After a promotion, the next eval cycle should start with a 0-len tracker."""
    pool = SnapshotPool()
    pool.add(_learner_state_dict_cpu(), iteration=10)
    cb, model, _, ref, tracker = _make_callback(
        {"scored_for_us": True},
        pool=pool,
        eval_every=100,
        episodes_per_cycle=10,
        window_size=10,
        promotion_threshold=0.5,
    )
    # Cycle 1: ref → 10, fill window with WINs.
    model.num_timesteps = 100
    cb.on_step()
    assert tracker.is_full
    # Add new snapshot, run cycle 2 → promote.
    pool.add(_learner_state_dict_cpu(), iteration=20)
    model.num_timesteps = 200
    cb.on_step()
    assert ref.loaded_iteration == 20
    assert len(tracker) == 0  # cooldown / reset


# ---------------------------------------------------------------------------
# Eval scheduling
# ---------------------------------------------------------------------------


def test_eval_schedules_to_next_interval() -> None:
    pool = SnapshotPool()
    pool.add(_learner_state_dict_cpu(), iteration=1)
    cb, model, _, _, _ = _make_callback(
        {"scored_for_us": True},
        pool=pool,
        eval_every=100,
        episodes_per_cycle=2,
        window_size=10,
    )
    assert cb._next_eval_at == 100
    model.num_timesteps = 100
    cb.on_step()
    assert cb._next_eval_at == 200


def test_eval_does_not_run_before_threshold() -> None:
    pool = SnapshotPool()
    pool.add(_learner_state_dict_cpu(), iteration=1)
    cb, model, _, ref, tracker = _make_callback(
        {"scored_for_us": True},
        pool=pool,
        eval_every=100,
        episodes_per_cycle=2,
        window_size=10,
    )
    model.num_timesteps = 50  # below threshold
    cb.on_step()
    # No eval should have run.
    assert ref.is_set is False
    assert len(tracker) == 0
