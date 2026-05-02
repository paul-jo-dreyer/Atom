"""Sliding-window win-rate tracker for self-play reference evaluation.

Records episode outcomes against a fixed reference opponent and
exposes a chess-style score: W=1.0, L=0.0, T=0.5. The window has a
fixed maximum length; once full, each new outcome evicts the oldest.

Two consumer rules from the design discussion:
  * Win rate is **only valid when the window is full** — early-training
    estimates with K<50 are too noisy to gate promotion on. Callers
    check `is_full` before reading `win_rate`.
  * Cooldown after promotion = `reset()` clears the window. The next
    promotion can't fire until the window has refilled to capacity.
"""

from __future__ import annotations

from collections import deque
from enum import Enum


class Outcome(Enum):
    """Per-episode outcome from the learner's perspective."""

    WIN = 1.0
    DRAW = 0.5
    LOSS = 0.0


class WinRateTracker:
    """Bounded deque of episode scores; reports the running mean as a
    chess-style win rate when the window is full.

    Parameters
    ----------
    window_size
        Maximum number of recent outcomes kept. Must be >= 1.
    """

    def __init__(self, window_size: int = 50) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        self._window_size = int(window_size)
        self._scores: deque[float] = deque(maxlen=self._window_size)

    # ---- introspection --------------------------------------------------

    @property
    def window_size(self) -> int:
        return self._window_size

    def __len__(self) -> int:
        return len(self._scores)

    @property
    def is_full(self) -> bool:
        return len(self._scores) == self._window_size

    @property
    def win_rate(self) -> float:
        """Mean score across the window (chess-style: W=1, L=0, T=0.5).
        Raises if the window is not yet full — early estimates with too
        few samples are not stable enough to act on."""
        if not self.is_full:
            raise RuntimeError(
                f"win_rate undefined: window has {len(self._scores)}/"
                f"{self._window_size} entries"
            )
        return sum(self._scores) / self._window_size

    # ---- mutation -------------------------------------------------------

    def record(self, outcome: Outcome) -> None:
        """Append a single episode outcome. Once the window is full,
        appending evicts the oldest entry (deque maxlen semantics)."""
        if not isinstance(outcome, Outcome):
            raise TypeError(
                f"record() expects Outcome enum, got {type(outcome).__name__}"
            )
        self._scores.append(outcome.value)

    def reset(self) -> None:
        """Clear all recorded outcomes. Called on promotion — the new
        reference's win rate is computed from scratch."""
        self._scores.clear()
