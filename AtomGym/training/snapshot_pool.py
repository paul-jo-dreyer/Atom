"""In-memory snapshot pool for self-play opponent sampling.

A `Snapshot` is an immutable record of a policy's weights at a point in
training, plus bookkeeping (elo, episodes played, wins). The
`SnapshotPool` stores up to `capacity` most-recent snapshots; older
entries are FIFO-evicted on add.

The pool itself is pure data — no torch dependency, no policy logic. The
shadow-policy machinery (loading a snapshot's state_dict into a prebuilt
module for inference) lives in the opponent-sampling layer, separately.
This split lets the pool be tested without torch and shipped through
pickle (e.g. for SubprocVecEnv distribution) without dragging a heavy
import along.

Tensor storage notes
--------------------
`add()` deep-copies the `state_dict`, so caller-side mutation of the
original after adding cannot leak in. Storing on CPU is recommended (GPU
memory is expensive for snapshots that sit idle 99% of the time); the
pool does not enforce this — pre-CPU at the call site if needed:

    pool.add({k: v.cpu() for k, v in policy.state_dict().items()},
             iteration=step)

Bookkeeping mutability
----------------------
`Snapshot` is a NamedTuple (immutable). To update `episodes_played` /
`wins`, the pool replaces the entry in place via `_replace`. This keeps
the entries themselves immutable from the caller's perspective —
external references to a `Snapshot` retrieved via `latest()` /
`sample()` are *not* updated when the pool's internal copy is rewritten.
For win-rate logging that's fine: the caller logs once and discards.
"""

from __future__ import annotations

import copy
from typing import Iterator, NamedTuple

import numpy as np


class Snapshot(NamedTuple):
    """Immutable record of a policy state at a point in training."""

    state_dict: dict
    iteration: int
    elo: float
    episodes_played: int
    wins: int


class SnapshotPool:
    """Bounded FIFO pool of policy snapshots.

    Add new snapshots via `add()`; sample uniformly via `sample()`; query
    the most recent via `latest()`. Once the pool is at capacity, the
    next `add()` evicts the oldest entry.
    """

    def __init__(self, capacity: int = 20) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self._capacity = int(capacity)
        self._entries: list[Snapshot] = []

    # ---- introspection --------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Snapshot]:
        """Yield entries oldest-first."""
        return iter(self._entries)

    # ---- mutation -------------------------------------------------------

    def add(
        self,
        state_dict: dict,
        iteration: int,
        *,
        elo: float = 1000.0,
    ) -> Snapshot:
        """Add a new snapshot, deep-copying the state_dict so post-add
        mutation of the source dict cannot leak in. Evicts the oldest
        entry if the pool is at capacity. Returns the stored `Snapshot`."""
        snap = Snapshot(
            state_dict=copy.deepcopy(state_dict),
            iteration=int(iteration),
            elo=float(elo),
            episodes_played=0,
            wins=0,
        )
        self._entries.append(snap)
        if len(self._entries) > self._capacity:
            self._entries.pop(0)
        return snap

    def record_outcome(self, iteration: int, learner_won: bool) -> bool:
        """Increment `episodes_played` for the snapshot at `iteration`,
        and `wins` if `learner_won` is False (the snapshot won, learner
        lost). Returns True if the snapshot was found, False if it has
        been evicted (silent no-op)."""
        for i, snap in enumerate(self._entries):
            if snap.iteration == iteration:
                self._entries[i] = snap._replace(
                    episodes_played=snap.episodes_played + 1,
                    wins=snap.wins + (0 if learner_won else 1),
                )
                return True
        return False

    # ---- queries --------------------------------------------------------

    def latest(self) -> Snapshot:
        """Most recently added snapshot. Raises IndexError if empty."""
        if not self._entries:
            raise IndexError("SnapshotPool is empty")
        return self._entries[-1]

    def sample(self, rng: np.random.Generator) -> Snapshot:
        """Uniformly sample one snapshot. Raises IndexError if empty.
        Pass an explicit `rng` so seed-controlled experiments are
        reproducible."""
        if not self._entries:
            raise IndexError("SnapshotPool is empty")
        idx = int(rng.integers(0, len(self._entries)))
        return self._entries[idx]
