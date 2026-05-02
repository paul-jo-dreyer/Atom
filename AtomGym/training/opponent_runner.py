"""Worker-side opponent inference for self-play.

Each subproc owns one `OpponentRunner`. It bundles:
  * a `SnapshotPool` replica (worker's local copy, synced from main at
    the PPO update boundary)
  * a CPU "shadow" policy (architecturally identical to the learner —
    same MLP layout, same log_std_init, same activation, etc.)
  * a worker-local RNG for opponent sampling
  * the iteration tag of the currently-loaded snapshot (or None)

The lifecycle:
  1. **Construction.** Worker spins up; runner builds an empty pool and
     a fresh CPU shadow policy. No snapshot is loaded → `predict` returns
     zeros (the empty-pool fallback).
  2. **Sync (once per PPO update).** Main calls
     `vec_env.env_method('update_opponent_pool', new_pool)`. Each worker
     atomically replaces its replica AND resamples (ε chance latest, else
     uniform from the pool), then loads that snapshot's state_dict into
     the shadow policy. The opponent stays fixed for the whole next
     rollout.
  3. **Per-step inference.** `predict(obs)` is bound to
     `AtomTeamEnv.opponent_policy`. CPU forward pass + action sample +
     numpy convert. Microseconds.

Design notes
------------
*Why CPU?* The learner trains on GPU. Putting torch + CUDA on every
SubprocVec worker is a multi-GB memory hit (this is what caused our
earlier OOM). CPU inference for a 128×128 MLP on 18-d obs is fast
enough that this is a strict win.

*Why per-rollout sampling, not per-reset?* Decided in step 4 design:
load_state_dict cost is small but per-reset wrapping was unjustified
machinery. Rollout-boundary sampling is what AlphaStar/OpenAI Five do —
plenty of opponent diversity comes from `n_envs` workers each picking
their own.

*Why ε-greedy "latest else uniform"?* Default ε=0.5. Keeps the gradient
signal sharp at the policy frontier (latest) while spreading rollouts
across the history (uniform). PFSP / Elo-weighted sampling is a future
optimization noted in CLAUDE.md.

*Why returns zeros when empty?* Decided in design: don't poison early
training with a randomly-weighted opponent that just bumps into the
learner. Stationary body lets the learner build basic ball/goal skill
in a 2-robot scene before the opponent gets serious.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from AtomGym.training._shadow_policy import (
    build_shadow_policy,
    shadow_predict,
    state_dict_to_tensors,
)
from AtomGym.training.snapshot_pool import SnapshotPool


class OpponentRunner:
    """Worker-side bundle of {pool replica, shadow policy, RNG}.

    Construct once per worker. Call `update_pool(new_pool)` at the PPO
    update boundary; call `predict(obs)` per step (this is what gets
    bound to `AtomTeamEnv.opponent_policy`).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        policy_kwargs: dict[str, Any] | None = None,
        eps_latest: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= eps_latest <= 1.0:
            raise ValueError(
                f"eps_latest must be in [0, 1], got {eps_latest}"
            )
        self._eps_latest = float(eps_latest)

        self._shadow_policy = build_shadow_policy(
            observation_space, action_space, policy_kwargs
        )

        self._action_dim = int(np.prod(action_space.shape))
        self._pool: SnapshotPool = SnapshotPool()  # placeholder, replaced on first update
        self._loaded_iteration: int | None = None
        self._rng = np.random.default_rng(seed)

    # ---- public API ----------------------------------------------------

    def update_pool(self, new_pool: SnapshotPool) -> None:
        """Atomically replace the pool replica and resample the loaded
        opponent. Called at the PPO update boundary by main, via
        `vec_env.env_method('update_opponent_pool', new_pool)`.

        Empty pool ⟹ unload (predict will return zeros until a non-empty
        pool comes through). Non-empty ⟹ pick latest with prob
        `eps_latest`, else uniform-sample, and load the snapshot's
        state_dict into the shadow policy.
        """
        self._pool = new_pool
        if len(self._pool) == 0:
            self._loaded_iteration = None
            return
        if self._rng.random() < self._eps_latest:
            snap = self._pool.latest()
        else:
            snap = self._pool.sample(self._rng)
        # Pool stores numpy arrays (see _shadow_policy.py for the
        # rationale); convert back to tensors for load_state_dict.
        self._shadow_policy.load_state_dict(state_dict_to_tensors(snap.state_dict))
        self._loaded_iteration = int(snap.iteration)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Inference for the currently-loaded opponent. Returns
        zero-action when no snapshot is loaded (empty-pool case).

        Stochastic by default — same convention as the learner during
        rollouts; keeps the opponent from being deterministically
        exploitable.
        """
        if self._loaded_iteration is None:
            return np.zeros(self._action_dim, dtype=np.float32)
        return shadow_predict(self._shadow_policy, obs, deterministic=False)

    # ---- introspection -------------------------------------------------

    @property
    def loaded_iteration(self) -> int | None:
        """Iteration tag of the currently-loaded snapshot, or None if
        nothing is loaded (empty pool)."""
        return self._loaded_iteration

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    @property
    def eps_latest(self) -> float:
        return self._eps_latest
