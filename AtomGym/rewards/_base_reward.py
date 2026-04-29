"""Reward base machinery — context bundle, term protocol, composite summer.

The contract is small and intentional:

1. The env owns all state. Each step it builds a `RewardContext` (read-only
   snapshot) and hands it to a `RewardComposite`.
2. `RewardTerm`s are stateless functions of context. They never hold
   `self._something_from_last_step`; if a term needs cross-step info (prev
   obs, event flags, episode time) it reads it from the context.
3. The composite returns both the scalar total AND a per-term breakdown
   dict, so the env can include the breakdown in `info` for TensorBoard
   logging.

Why this split: it keeps reward terms trivially testable (pure functions),
keeps state concerns owned by exactly one class (the env), and works
correctly under SB3's `SubprocVecEnv` because nothing in a reward term is
mutable per-process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from AtomGym.action_observation import ActionView, ObsView


# ---------------------------------------------------------------------------
# Context bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardContext:
    """Read-only snapshot of everything a reward term might need at this step.

    The env constructs one per step via lightweight reference-passing
    (no copies). Reward terms read from it; they do not mutate it.

    Attributes
    ----------
    obs, action
        Current step's observation and action arrays.
    prev_obs, prev_action
        Previous step's. `None` on the first step after `reset()`. Reward
        terms must handle `None` (typically by returning 0.0 — there's no
        "delta" to compute on the first step).
    info
        Per-step event flags raised by the env (e.g. `scored_for_us`,
        `scored_against_us`, `collision`). Reward terms read these to
        emit one-shot rewards on transitions.
    obs_view, action_view
        The schema views from `AtomGym.action_observation`. Saves every
        reward from holding its own.
    field_x_half, field_y_half, goal_y_half, goal_extension
        Physical field parameters (metres). Avoids hard-coding constants
        inside reward terms.
    dt
        Sim timestep (seconds). Useful for time-derivative or
        time-discount rewards.
    """

    obs: np.ndarray
    action: np.ndarray
    prev_obs: np.ndarray | None
    prev_action: np.ndarray | None
    info: dict[str, Any]
    obs_view: ObsView
    action_view: ActionView
    field_x_half: float
    field_y_half: float
    goal_y_half: float
    goal_extension: float
    dt: float


# ---------------------------------------------------------------------------
# Term protocol
# ---------------------------------------------------------------------------


class RewardTerm(ABC):
    """Base class for reward terms. Subclasses override `name` (class attr)
    and implement `__call__`. Construction takes a `weight` only — any other
    hyperparams are subclass-specific.

    Stateless contract: `__call__` is a pure function of `RewardContext`.
    No attribute writes inside `__call__`. No `self._prev_*` fields.
    """

    name: str = "unnamed"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = float(weight)

    @abstractmethod
    def __call__(self, ctx: RewardContext) -> float:
        """Return the unweighted scalar reward contribution for this step.
        The composite multiplies this by `self.weight`."""
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, weight={self.weight})"


# ---------------------------------------------------------------------------
# Composite — sums weighted contributions, returns breakdown
# ---------------------------------------------------------------------------


@dataclass
class RewardComposite:
    """Sums weighted contributions from a list of `RewardTerm`s.

    The breakdown dict it returns alongside the total is the input we'd want
    to log to TensorBoard one-key-at-a-time, so each term's contribution is
    visible during training. Keys are term names; values are the WEIGHTED
    contribution (i.e., `weight * term(ctx)`). If two terms share a name,
    their weighted contributions accumulate under the one key (so the total
    is always correct; the breakdown just groups same-named terms).
    """

    terms: list[RewardTerm] = field(default_factory=list)

    def __call__(self, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        breakdown: dict[str, float] = {}
        for term in self.terms:
            contribution = term.weight * term(ctx)
            breakdown[term.name] = breakdown.get(term.name, 0.0) + contribution
        total = float(sum(breakdown.values()))
        return total, breakdown

    def add(self, term: RewardTerm) -> "RewardComposite":
        """Append a term and return self, for fluent construction:
        `composite.add(TermA(...)).add(TermB(...))`."""
        self.terms.append(term)
        return self

    def __len__(self) -> int:
        return len(self.terms)

    def names(self) -> list[str]:
        """Term names in evaluation order — useful for fixed-shape logging."""
        return [t.name for t in self.terms]
