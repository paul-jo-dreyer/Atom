"""SB3 callback: snapshot the learner's policy at a fixed cadence and
sync the pool replicas across all SubprocVec workers.

Hooks into PPO's training loop. Every `snapshot_every` env steps, at
the start of the next rollout (so snapshots reflect post-update
weights):
  1. Detach + CPU-clone the learner's policy `state_dict`.
  2. Add it to the master `SnapshotPool` (which deep-copies).
  3. Push the updated pool to all workers via
     `vec_env.env_method('update_opponent_pool', pool)`. Each worker's
     `OpponentRunner` atomically replaces its replica AND resamples
     the loaded opponent — so the next rollout runs against fresh
     opponents drawn from the just-synced pool.

Why `_on_rollout_start` (not `_on_rollout_end`)?
    `_on_rollout_end` fires AFTER rollout collection but BEFORE the
    PPO update. Snapshotting then would freeze pre-update weights.
    `_on_rollout_start` fires before the next rollout begins, by which
    point the previous update is complete — so snapshots are
    post-update.

The first call to `_on_rollout_start` happens at `num_timesteps == 0`,
which is below any reasonable `snapshot_every` — so the first rollout
naturally runs with an empty pool (zero-action opponents) per the
empty-pool fallback design.
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from AtomGym.training._shadow_policy import state_dict_to_numpy
from AtomGym.training.snapshot_pool import SnapshotPool


class PoolSyncCallback(BaseCallback):
    """Snapshot + broadcast at every `snapshot_every` boundary.

    Parameters
    ----------
    pool
        Master `SnapshotPool` (shared with `RefEvalCallback`). The
        callback both *adds* to it and *broadcasts* it; everyone else
        reads it.
    vec_env
        The training `VecEnv`. Must be a `SubprocVecEnv` or
        `DummyVecEnv` whose underlying envs are `TeamWorkerWrapper`s
        (i.e. expose `update_opponent_pool`). Required as a constructor
        arg because the callback's `self.training_env` only resolves
        via `model.get_env()` after `init_callback`, and that VecEnv
        instance can be in either subproc or dummy form — passing it
        explicitly removes ambiguity.
    snapshot_every
        Cadence in total env steps.
    verbose
        Standard SB3 verbosity. >=1 prints a line per snapshot+sync.
    """

    def __init__(
        self,
        *,
        pool: SnapshotPool,
        vec_env: VecEnv,
        snapshot_every: int,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        if snapshot_every <= 0:
            raise ValueError(
                f"snapshot_every must be > 0, got {snapshot_every}"
            )
        self._pool = pool
        self._vec_env = vec_env
        self._snapshot_every = int(snapshot_every)
        self._next_snapshot_at: int = self._snapshot_every

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        # `self.num_timesteps` only updates inside `on_step`; read from
        # `self.model.num_timesteps` here for an up-to-date counter.
        timesteps = int(self.model.num_timesteps)
        if timesteps < self._next_snapshot_at:
            return
        try:
            self._snapshot_and_sync(timesteps)
        except Exception as e:
            # Don't kill training over a sync hiccup; log and continue.
            print(f"[pool_sync] snapshot/sync failed at step {timesteps:,}: {e}")
            import traceback
            traceback.print_exc()
        n = (timesteps // self._snapshot_every) + 1
        self._next_snapshot_at = n * self._snapshot_every

    def _snapshot_and_sync(self, timesteps: int) -> None:
        # Convert tensors to numpy before pool storage. Critical: torch
        # tensors pickled across multiprocessing pipes use the
        # `file_descriptor` sharing scheme, which dups an OS FD per
        # tensor. Across many syncs × many workers this blows past
        # `ulimit -n` and crashes with Errno 24 (observed in long runs).
        # Numpy pickles as plain bytes — no FDs involved.
        sd = state_dict_to_numpy(self.model.policy.state_dict())
        self._pool.add(sd, iteration=timesteps)
        # Broadcast pool replicas to all workers; each worker's runner
        # resamples on receipt.
        self._vec_env.env_method("update_opponent_pool", self._pool)
        if self.verbose >= 1:
            print(
                f"[pool_sync] step {timesteps:>9,}: snapshot added "
                f"(pool size {len(self._pool)}), synced to "
                f"{self._vec_env.num_envs} workers"
            )
