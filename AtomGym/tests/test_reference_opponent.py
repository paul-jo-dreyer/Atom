"""Tests for ReferenceOpponent — the single-snapshot benchmark opponent."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from AtomGym.training.reference_opponent import ReferenceOpponent
from AtomGym.training.snapshot_pool import Snapshot


_OBS_SPACE = spaces.Box(low=-1.0, high=+1.0, shape=(20,), dtype=np.float32)
_ACT_SPACE = spaces.Box(low=-1.0, high=+1.0, shape=(2,), dtype=np.float32)
_POLICY_KWARGS: dict[str, Any] = dict(
    net_arch=dict(pi=[128, 128], vf=[128, 128]),
    log_std_init=0.3,
)


def _learner_state_dict_cpu() -> dict:
    """Fresh learner-shaped policy → CPU state_dict (mimics what
    training will do when adding a snapshot)."""
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


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_initial_state_unset() -> None:
    ref = ReferenceOpponent(_OBS_SPACE, _ACT_SPACE, _POLICY_KWARGS)
    assert ref.loaded_iteration is None
    assert ref.is_set is False


def test_predict_returns_zeros_when_unset() -> None:
    ref = ReferenceOpponent(_OBS_SPACE, _ACT_SPACE, _POLICY_KWARGS)
    obs = np.zeros(20, dtype=np.float32)
    action = ref.predict(obs)
    assert action.shape == (2,)
    assert action.dtype == np.float32
    np.testing.assert_array_equal(action, np.zeros(2, dtype=np.float32))


# ---------------------------------------------------------------------------
# set_snapshot
# ---------------------------------------------------------------------------


def test_set_snapshot_marks_loaded() -> None:
    ref = ReferenceOpponent(_OBS_SPACE, _ACT_SPACE, _POLICY_KWARGS)
    ref.set_snapshot(_make_snapshot(iteration=42))
    assert ref.loaded_iteration == 42
    assert ref.is_set is True


def test_set_snapshot_replaces_previous() -> None:
    """Promotion operation: the new ref takes over completely."""
    ref = ReferenceOpponent(_OBS_SPACE, _ACT_SPACE, _POLICY_KWARGS)
    ref.set_snapshot(_make_snapshot(iteration=10))
    ref.set_snapshot(_make_snapshot(iteration=20))
    assert ref.loaded_iteration == 20


def test_predict_after_set_returns_correct_shape() -> None:
    ref = ReferenceOpponent(_OBS_SPACE, _ACT_SPACE, _POLICY_KWARGS)
    ref.set_snapshot(_make_snapshot(iteration=1))
    obs = np.random.default_rng(0).standard_normal(20).astype(np.float32)
    action = ref.predict(obs)
    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0 - 1e-6)
    assert np.all(action <= +1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_predict_is_deterministic() -> None:
    """Same obs, same loaded snapshot → identical action across calls
    (deterministic mode = action distribution mean, no RNG)."""
    ref = ReferenceOpponent(_OBS_SPACE, _ACT_SPACE, _POLICY_KWARGS)
    ref.set_snapshot(_make_snapshot(iteration=1))
    obs = np.random.default_rng(0).standard_normal(20).astype(np.float32)
    a1 = ref.predict(obs)
    a2 = ref.predict(obs)
    a3 = ref.predict(obs)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(a1, a3)


# ---------------------------------------------------------------------------
# Round-trip equivalence with a real learner
# ---------------------------------------------------------------------------


def test_state_dict_round_trip_matches_learner() -> None:
    """Reference loaded with a learner's state_dict must produce the
    same deterministic action as the learner. Pins arch matching."""
    torch.manual_seed(11)
    learner = ActorCriticPolicy(
        observation_space=_OBS_SPACE,
        action_space=_ACT_SPACE,
        lr_schedule=lambda _: 0.0,
        **_POLICY_KWARGS,
    ).to("cpu")
    learner.set_training_mode(False)

    learner_sd = {k: v.detach().cpu().clone() for k, v in learner.state_dict().items()}
    snap = Snapshot(state_dict=learner_sd, iteration=1, elo=1000.0,
                    episodes_played=0, wins=0)

    ref = ReferenceOpponent(_OBS_SPACE, _ACT_SPACE, _POLICY_KWARGS)
    ref.set_snapshot(snap)

    rng = np.random.default_rng(0)
    for _ in range(5):
        obs = rng.standard_normal(20).astype(np.float32)
        learner_action, _ = learner.predict(obs, deterministic=True)
        ref_action = ref.predict(obs)
        np.testing.assert_allclose(ref_action, learner_action, atol=1e-6)
