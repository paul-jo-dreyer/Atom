"""Tests for PoolSyncCallback — snapshot cadence and env_method dispatch."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.policies import ActorCriticPolicy

from AtomGym.training.pool_sync_callback import PoolSyncCallback
from AtomGym.training.snapshot_pool import SnapshotPool


_OBS_SPACE = spaces.Box(low=-1.0, high=+1.0, shape=(18,), dtype=np.float32)
_ACT_SPACE = spaces.Box(low=-1.0, high=+1.0, shape=(2,), dtype=np.float32)
_POLICY_KWARGS: dict[str, Any] = dict(
    net_arch=dict(pi=[128, 128], vf=[128, 128]),
    log_std_init=0.3,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeVecEnv:
    """Captures env_method calls for inspection."""

    def __init__(self, n_envs: int = 2) -> None:
        self.num_envs = n_envs
        self.env_method_calls: list[tuple[str, tuple, dict]] = []

    def env_method(self, method_name: str, *args: Any, **kwargs: Any) -> list[None]:
        self.env_method_calls.append((method_name, args, kwargs))
        return [None] * self.num_envs


class _MockModel:
    """Real torch policy under .policy so state_dict() works correctly."""

    def __init__(self) -> None:
        self.policy = ActorCriticPolicy(
            observation_space=_OBS_SPACE,
            action_space=_ACT_SPACE,
            lr_schedule=lambda _: 0.0,
            **_POLICY_KWARGS,
        ).to("cpu")
        self.num_timesteps = 0
        self.logger = Logger(folder=None, output_formats=[])

    def get_env(self) -> None:
        return None


def _make_callback(
    *,
    snapshot_every: int = 100,
    pool_capacity: int = 10,
) -> tuple[PoolSyncCallback, _MockModel, SnapshotPool, _FakeVecEnv]:
    pool = SnapshotPool(capacity=pool_capacity)
    vec_env = _FakeVecEnv(n_envs=4)
    cb = PoolSyncCallback(
        pool=pool,
        vec_env=vec_env,  # type: ignore[arg-type]
        snapshot_every=snapshot_every,
        verbose=0,
    )
    model = _MockModel()
    cb.init_callback(model)  # type: ignore[arg-type]
    return cb, model, pool, vec_env


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_invalid_snapshot_every() -> None:
    with pytest.raises(ValueError):
        _make_callback(snapshot_every=0)
    with pytest.raises(ValueError):
        _make_callback(snapshot_every=-100)


# ---------------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------------


def test_no_sync_before_threshold() -> None:
    cb, model, pool, vec_env = _make_callback(snapshot_every=100)
    model.num_timesteps = 50
    cb.on_rollout_start()
    assert len(pool) == 0
    assert vec_env.env_method_calls == []


def test_sync_fires_at_threshold() -> None:
    cb, model, pool, vec_env = _make_callback(snapshot_every=100)
    model.num_timesteps = 100
    cb.on_rollout_start()
    assert len(pool) == 1
    assert pool.latest().iteration == 100
    assert len(vec_env.env_method_calls) == 1
    method_name, args, kwargs = vec_env.env_method_calls[0]
    assert method_name == "update_opponent_pool"
    assert args[0] is pool
    assert kwargs == {}


def test_schedule_advances_correctly() -> None:
    cb, model, pool, vec_env = _make_callback(snapshot_every=100)
    model.num_timesteps = 100
    cb.on_rollout_start()
    assert cb._next_snapshot_at == 200

    # Below new threshold → no sync
    model.num_timesteps = 150
    cb.on_rollout_start()
    assert len(pool) == 1

    # At new threshold → another sync
    model.num_timesteps = 200
    cb.on_rollout_start()
    assert len(pool) == 2
    assert pool.latest().iteration == 200
    assert cb._next_snapshot_at == 300


def test_jumping_past_multiple_thresholds_takes_one_snapshot_only() -> None:
    """If a rollout is long enough to cross several boundaries (rare but
    possible), we take only ONE snapshot — the cadence advances to the
    next boundary past the current step. Avoids piling up duplicate
    snapshots from a single rollout."""
    cb, model, pool, _ = _make_callback(snapshot_every=100)
    model.num_timesteps = 350  # crosses 100, 200, 300 in one go
    cb.on_rollout_start()
    assert len(pool) == 1
    assert pool.latest().iteration == 350
    assert cb._next_snapshot_at == 400


# ---------------------------------------------------------------------------
# State-dict independence
# ---------------------------------------------------------------------------


def test_snapshot_is_detached_from_model() -> None:
    """After a snapshot lands in the pool, mutating the model's weights
    must not affect the stored snapshot. Pins the detach + clone
    behaviour."""
    cb, model, pool, _ = _make_callback(snapshot_every=100)
    model.num_timesteps = 100
    cb.on_rollout_start()

    stored_value = float(
        pool.latest().state_dict["mlp_extractor.policy_net.0.weight"][0, 0]
    )

    # Mutate the model's weight in place.
    with torch.no_grad():
        for p in model.policy.parameters():
            p.add_(1.0)

    # The stored snapshot should be unaffected.
    after_value = float(
        pool.latest().state_dict["mlp_extractor.policy_net.0.weight"][0, 0]
    )
    assert after_value == stored_value


def test_pool_stores_numpy_not_tensors() -> None:
    """Critical: pool entries must be numpy arrays, not torch tensors.
    Tensors pickled across `multiprocessing.Pipe` use torch's
    file-descriptor sharing scheme — across many syncs × many workers
    that blows past `ulimit -n` (Errno 24). Numpy pickles as plain
    bytes with no FDs."""
    cb, model, pool, _ = _make_callback(snapshot_every=100)
    model.num_timesteps = 100
    cb.on_rollout_start()
    for key, value in pool.latest().state_dict.items():
        assert isinstance(value, np.ndarray), (
            f"{key} is {type(value).__name__}, expected numpy.ndarray. "
            f"This regresses the FD-leak fix."
        )
        # No torch.Tensor anywhere in the dict.
        assert not isinstance(value, torch.Tensor)


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


def test_env_method_failure_does_not_kill_callback() -> None:
    """If env_method throws (e.g. transient IPC issue), the callback
    should log + continue, not propagate the exception. Pool may end
    up partially out-of-sync but training survives."""
    pool = SnapshotPool(capacity=10)

    class _ThrowingVecEnv:
        num_envs = 4
        def env_method(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("simulated IPC failure")

    cb = PoolSyncCallback(
        pool=pool,
        vec_env=_ThrowingVecEnv(),  # type: ignore[arg-type]
        snapshot_every=100,
        verbose=0,
    )
    model = _MockModel()
    cb.init_callback(model)  # type: ignore[arg-type]
    model.num_timesteps = 100
    # Should NOT raise.
    cb.on_rollout_start()
    # Schedule still advanced even though sync failed — we don't want
    # to retry the same step forever.
    assert cb._next_snapshot_at == 200
