"""Tests for OpponentRunner — the worker-side bundle of {pool replica,
CPU shadow policy, sampling RNG} used to run frozen opponents during
self-play training."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from AtomGym.training.opponent_runner import OpponentRunner
from AtomGym.training.snapshot_pool import SnapshotPool


# Match the team env's spaces (18-dim obs, 2-dim action).
_OBS_SPACE = spaces.Box(low=-1.0, high=+1.0, shape=(20,), dtype=np.float32)
_ACT_SPACE = spaces.Box(low=-1.0, high=+1.0, shape=(2,), dtype=np.float32)
# Same kwargs as train.py uses for the learner.
_POLICY_KWARGS: dict[str, Any] = dict(
    net_arch=dict(pi=[128, 128], vf=[128, 128]),
    log_std_init=0.3,
)


def _make_runner(seed: int = 0, eps_latest: float = 0.5) -> OpponentRunner:
    return OpponentRunner(
        observation_space=_OBS_SPACE,
        action_space=_ACT_SPACE,
        policy_kwargs=_POLICY_KWARGS,
        eps_latest=eps_latest,
        seed=seed,
    )


def _learner_state_dict_cpu() -> dict:
    """Build a fresh learner-shaped policy and grab its CPU state_dict.
    The pool stores CPU tensors; this helper mimics what train.py will do
    when adding a snapshot."""
    policy = ActorCriticPolicy(
        observation_space=_OBS_SPACE,
        action_space=_ACT_SPACE,
        lr_schedule=lambda _: 0.0,
        **_POLICY_KWARGS,
    ).to("cpu")
    return {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}


# ---------------------------------------------------------------------------
# Construction & invariants
# ---------------------------------------------------------------------------


def test_constructs_with_empty_pool() -> None:
    r = _make_runner()
    assert r.pool_size == 0
    assert r.loaded_iteration is None


def test_invalid_eps_latest_raises() -> None:
    with pytest.raises(ValueError, match="eps_latest"):
        _make_runner(eps_latest=-0.1)
    with pytest.raises(ValueError, match="eps_latest"):
        _make_runner(eps_latest=1.5)


def test_eps_latest_property() -> None:
    r = _make_runner(eps_latest=0.7)
    assert r.eps_latest == pytest.approx(0.7)


def test_shadow_policy_is_in_eval_mode() -> None:
    """training_mode flag should be False on the shadow — no dropout /
    batchnorm side-effects, and forward passes don't need autograd."""
    r = _make_runner()
    # Access the shadow directly. set_training_mode(False) sets module.training=False.
    assert r._shadow_policy.training is False


# ---------------------------------------------------------------------------
# Empty-pool semantics
# ---------------------------------------------------------------------------


def test_predict_returns_zeros_when_unloaded() -> None:
    r = _make_runner()
    obs = np.zeros(20, dtype=np.float32)
    action = r.predict(obs)
    assert action.shape == (2,)
    assert action.dtype == np.float32
    np.testing.assert_array_equal(action, np.zeros(2, dtype=np.float32))


def test_update_with_empty_pool_keeps_unloaded() -> None:
    r = _make_runner()
    r.update_pool(SnapshotPool())
    assert r.loaded_iteration is None
    np.testing.assert_array_equal(
        r.predict(np.zeros(20, dtype=np.float32)),
        np.zeros(2, dtype=np.float32),
    )


def test_unload_after_pool_becomes_empty() -> None:
    """If the pool was non-empty and gets replaced with an empty pool,
    the runner should unload (not keep stale snapshot loaded)."""
    r = _make_runner()
    pool = SnapshotPool()
    pool.add(_learner_state_dict_cpu(), iteration=42)
    r.update_pool(pool)
    assert r.loaded_iteration == 42
    r.update_pool(SnapshotPool())
    assert r.loaded_iteration is None


# ---------------------------------------------------------------------------
# update_pool loads a snapshot
# ---------------------------------------------------------------------------


def test_update_pool_loads_single_snapshot() -> None:
    r = _make_runner()
    pool = SnapshotPool()
    pool.add(_learner_state_dict_cpu(), iteration=100)
    r.update_pool(pool)
    assert r.loaded_iteration == 100
    assert r.pool_size == 1


def test_predict_after_load_returns_correct_shape_and_dtype() -> None:
    r = _make_runner()
    pool = SnapshotPool()
    pool.add(_learner_state_dict_cpu(), iteration=100)
    r.update_pool(pool)
    obs = np.random.default_rng(0).standard_normal(20).astype(np.float32)
    action = r.predict(obs)
    assert action.shape == (2,)
    assert action.dtype == np.float32
    # Action should be clipped to action_space bounds.
    assert np.all(action >= -1.0 - 1e-6)
    assert np.all(action <= +1.0 + 1e-6)


def test_predict_after_load_is_not_zero() -> None:
    """Once a snapshot is loaded, predict should produce non-trivial
    actions (the network is initialized with non-zero weights, log_std
    is finite, so sampled actions are non-zero with probability 1)."""
    r = _make_runner(seed=0)
    pool = SnapshotPool()
    pool.add(_learner_state_dict_cpu(), iteration=100)
    r.update_pool(pool)
    obs = np.random.default_rng(0).standard_normal(20).astype(np.float32)
    # Sample a few actions; at least one should be clearly non-zero.
    nonzero_count = 0
    for _ in range(5):
        action = r.predict(obs)
        if np.linalg.norm(action) > 1e-3:
            nonzero_count += 1
    assert nonzero_count > 0


# ---------------------------------------------------------------------------
# eps_latest sampling behaviour
# ---------------------------------------------------------------------------


def test_eps_latest_one_always_picks_latest() -> None:
    r = _make_runner(eps_latest=1.0, seed=0)
    pool = SnapshotPool(capacity=10)
    iterations = [10, 20, 30, 40, 50]
    for i in iterations:
        pool.add(_learner_state_dict_cpu(), iteration=i)

    # Many resyncs — should always land on iteration=50 (latest).
    for _ in range(20):
        r.update_pool(pool)
        assert r.loaded_iteration == 50


def test_eps_latest_zero_uses_uniform_sampling() -> None:
    """With eps_latest=0, all entries should be reachable across many
    resyncs. (Not a strict uniformity check — that's tested in the pool
    tests; here we just want to confirm the latest isn't always picked.)"""
    r = _make_runner(eps_latest=0.0, seed=0)
    pool = SnapshotPool(capacity=10)
    iterations = [10, 20, 30, 40, 50]
    for i in iterations:
        pool.add(_learner_state_dict_cpu(), iteration=i)

    seen: set[int] = set()
    for _ in range(50):
        r.update_pool(pool)
        seen.add(r.loaded_iteration)
    # With 50 trials over 5 entries we should see at least 4 distinct ones.
    assert len(seen) >= 4, f"saw only {seen}"


def test_single_entry_pool_always_loads_that_entry() -> None:
    """eps_latest is irrelevant when there's only one option."""
    for eps in (0.0, 0.5, 1.0):
        r = _make_runner(eps_latest=eps, seed=0)
        pool = SnapshotPool()
        pool.add(_learner_state_dict_cpu(), iteration=99)
        for _ in range(5):
            r.update_pool(pool)
            assert r.loaded_iteration == 99


def test_seeded_rng_is_reproducible() -> None:
    """Two runners with the same seed and the same pool sequence should
    produce the same iteration trajectory."""
    pool = SnapshotPool(capacity=10)
    for i in [10, 20, 30, 40, 50]:
        pool.add(_learner_state_dict_cpu(), iteration=i)

    r_a = _make_runner(eps_latest=0.5, seed=42)
    r_b = _make_runner(eps_latest=0.5, seed=42)
    seq_a, seq_b = [], []
    for _ in range(10):
        r_a.update_pool(pool)
        r_b.update_pool(pool)
        seq_a.append(r_a.loaded_iteration)
        seq_b.append(r_b.loaded_iteration)
    assert seq_a == seq_b


# ---------------------------------------------------------------------------
# Round-trip: shadow policy with a learner's state_dict produces
# equivalent deterministic outputs to the learner itself.
# ---------------------------------------------------------------------------


def test_state_dict_round_trip_deterministic_outputs() -> None:
    """If the learner's state_dict is loaded into the runner's shadow,
    deterministic predictions on the same obs must match exactly. This
    pins the architecture-matching contract: same policy_kwargs ⟹
    state_dicts are interchangeable."""
    # Build a "learner" — separate ActorCriticPolicy with the same args.
    torch.manual_seed(7)
    learner = ActorCriticPolicy(
        observation_space=_OBS_SPACE,
        action_space=_ACT_SPACE,
        lr_schedule=lambda _: 0.0,
        **_POLICY_KWARGS,
    ).to("cpu")
    learner.set_training_mode(False)

    learner_sd = {k: v.detach().cpu().clone() for k, v in learner.state_dict().items()}
    pool = SnapshotPool()
    pool.add(learner_sd, iteration=1)

    r = _make_runner(eps_latest=1.0, seed=0)
    r.update_pool(pool)

    # Probe with several random obs vectors; deterministic predictions
    # should be elementwise-equal between the learner and the shadow.
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs = rng.standard_normal(20).astype(np.float32)
        learner_action, _ = learner.predict(obs, deterministic=True)
        shadow_action, _ = r._shadow_policy.predict(obs, deterministic=True)
        np.testing.assert_allclose(learner_action, shadow_action, atol=1e-6)


# ---------------------------------------------------------------------------
# State_dict storage — sanity that snapshots are CPU tensors
# ---------------------------------------------------------------------------


def test_pool_stores_cpu_tensors() -> None:
    """The CPU pre-clone in callers (and the deepcopy in pool.add) should
    leave us with CPU tensors in stored snapshots — required for the
    shadow policy to load them without device mismatch."""
    pool = SnapshotPool()
    pool.add(_learner_state_dict_cpu(), iteration=1)
    snap = pool.latest()
    for k, v in snap.state_dict.items():
        if isinstance(v, torch.Tensor):
            assert v.device.type == "cpu", f"{k} on {v.device}"
