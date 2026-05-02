"""Single fixed-snapshot opponent used as the win-rate benchmark during
self-play training.

Plays the same architectural role as `OpponentRunner` but for the
*reference* in the curriculum, not the rolling pool:
  * Holds exactly one `Snapshot` at a time (or none, before the first
    snapshot is added to the pool).
  * Inference is **deterministic** by default — eval episodes against
    the reference are meant to be a low-variance progress signal, so
    we sample the action distribution mean rather than draw from it.
  * `set_snapshot(snap)` swaps in a new reference; this is the
    promotion operation, called by `RefEvalCallback` when the win
    rate gate fires.

Predict returns zeros if no snapshot has ever been set (initial state
before the very first snapshot lands in the pool). In normal training
this only matters for the very first eval cycle — by the time the
callback runs at all, the pool will typically have at least one
entry, which gets promoted to "first reference."
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
from AtomGym.training.snapshot_pool import Snapshot


class ReferenceOpponent:
    """Single-snapshot CPU opponent used by the win-rate benchmark.

    Architecture must match the learner exactly — same `policy_kwargs`
    that train.py uses for the learner. Constructed once per training
    run; `set_snapshot()` is the only mutation API.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        policy_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._shadow_policy = build_shadow_policy(
            observation_space, action_space, policy_kwargs
        )
        self._action_dim = int(np.prod(action_space.shape))
        self._loaded_iteration: int | None = None

    # ---- public API -----------------------------------------------------

    def set_snapshot(self, snap: Snapshot) -> None:
        """Load `snap`'s state_dict into the shadow policy. This is the
        promotion operation — called when the win rate gate fires.

        Pool snapshots are stored as numpy arrays (see _shadow_policy.py
        for why); convert to tensors for `load_state_dict`."""
        self._shadow_policy.load_state_dict(state_dict_to_tensors(snap.state_dict))
        self._loaded_iteration = int(snap.iteration)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic CPU forward → action of shape (action_dim,).
        Returns zeros if no snapshot has been set yet."""
        if self._loaded_iteration is None:
            return np.zeros(self._action_dim, dtype=np.float32)
        return shadow_predict(self._shadow_policy, obs, deterministic=True)

    # ---- introspection --------------------------------------------------

    @property
    def loaded_iteration(self) -> int | None:
        return self._loaded_iteration

    @property
    def is_set(self) -> bool:
        return self._loaded_iteration is not None
