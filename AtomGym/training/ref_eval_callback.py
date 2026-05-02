"""SB3 callback: periodic eval against the curriculum reference, with
win-rate-gated promotion.

Hooks into PPO's training loop. Every `eval_every` env steps:
  1. If no reference is set yet and the pool has at least one snapshot,
     load `pool.latest()` as the initial reference (per CLAUDE.md, "first
     reference is the first snapshot").
  2. Run `episodes_per_cycle` deterministic episodes against the
     reference. Classify each as W (scored_for_us), L (scored_against_us),
     or T (truncated with no goal).
  3. Record each outcome in the sliding-window tracker.
  4. Log `eval/reference_iteration` and `eval/window_size` to TB every
     cycle; log `eval/win_rate_vs_reference` once the window is full.
  5. **Promotion gate.** When the window is full AND win rate ≥
     `promotion_threshold` AND `pool.latest().iteration` differs from
     the current reference's iteration: replace the reference with
     `pool.latest()` and reset the tracker. The tracker reset is the
     cooldown — no further promotion can fire until the new ref has
     accumulated `window_size` fresh outcomes.

Errors during eval are caught and logged — a viz/eval hiccup never
kills training (same defensive pattern as GifEvalCallback).
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

from stable_baselines3.common.callbacks import BaseCallback

from AtomGym.environments import AtomTeamEnv
from AtomGym.training.reference_opponent import ReferenceOpponent
from AtomGym.training.snapshot_pool import SnapshotPool
from AtomGym.training.win_rate_tracker import Outcome, WinRateTracker


class RefEvalCallback(BaseCallback):
    """Periodic curriculum-reference eval + promotion.

    Parameters
    ----------
    eval_env_factory
        No-arg callable that returns a fresh `AtomTeamEnv`. The env is
        constructed at training start; its `opponent_policy` is bound
        to the reference's `predict` method once and stays bound (the
        reference's internal state changes via `set_snapshot`, but the
        callable identity is stable).
    pool
        The master `SnapshotPool` (shared with the training loop). The
        callback reads `latest()` only — it never mutates the pool.
    reference
        Pre-built `ReferenceOpponent` whose architecture matches the
        learner. Caller is responsible for matching `policy_kwargs` to
        train.py's; the round-trip test in test_reference_opponent.py
        pins the contract.
    tracker
        Sliding-window `WinRateTracker`. Window size determines both
        the variance of the win-rate estimate and the post-promotion
        cooldown length.
    eval_every
        Eval cadence in total env steps (counted against
        `self.num_timesteps`).
    episodes_per_cycle
        Number of episodes to run per eval cycle. Window of
        `tracker.window_size` fills after `ceil(window_size /
        episodes_per_cycle)` cycles.
    promotion_threshold
        Win rate (chess-style) at or above which promotion fires. 0.80
        is a good starting value; tune via CLI.
    verbose
        Standard SB3 verbosity. >=1 prints one-line eval summaries and
        promotion events.
    """

    def __init__(
        self,
        *,
        eval_env_factory: Callable[[], AtomTeamEnv],
        pool: SnapshotPool,
        reference: ReferenceOpponent,
        tracker: WinRateTracker,
        eval_every: int,
        episodes_per_cycle: int,
        promotion_threshold: float,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        if eval_every <= 0:
            raise ValueError(f"eval_every must be > 0, got {eval_every}")
        if episodes_per_cycle <= 0:
            raise ValueError(
                f"episodes_per_cycle must be > 0, got {episodes_per_cycle}"
            )
        if not 0.0 <= promotion_threshold <= 1.0:
            raise ValueError(
                f"promotion_threshold must be in [0, 1], got {promotion_threshold}"
            )
        self._eval_env_factory = eval_env_factory
        self._pool = pool
        self._reference = reference
        self._tracker = tracker
        self._eval_every = int(eval_every)
        self._episodes_per_cycle = int(episodes_per_cycle)
        self._promotion_threshold = float(promotion_threshold)

        self._next_eval_at: int = self._eval_every
        self._eval_env: AtomTeamEnv | None = None

    # ---- SB3 lifecycle hooks --------------------------------------------

    def _init_callback(self) -> None:
        self._eval_env = self._eval_env_factory()
        # Bind the reference's predict ONCE. Its internal state changes
        # on set_snapshot but the callable identity stays the same.
        self._eval_env.set_opponent_policy(self._reference.predict)
        if self.verbose >= 1:
            print(
                f"[ref_eval] every {self._eval_every:,} steps, "
                f"{self._episodes_per_cycle} episodes/cycle, "
                f"window={self._tracker.window_size}, "
                f"promote@{self._promotion_threshold:.2f}"
            )

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval_at:
            try:
                self._run_eval_cycle()
            except Exception as e:
                print(f"[ref_eval] cycle failed at step {self.num_timesteps:,}: {e}")
                traceback.print_exc()
            n = (self.num_timesteps // self._eval_every) + 1
            self._next_eval_at = n * self._eval_every
        return True

    # ---- eval cycle -----------------------------------------------------

    def _run_eval_cycle(self) -> None:
        assert self._eval_env is not None

        # 1. Initialize reference if pool has at least one snapshot.
        if not self._reference.is_set and len(self._pool) > 0:
            self._reference.set_snapshot(self._pool.latest())
            self._tracker.reset()
            if self.verbose >= 1:
                print(
                    f"[ref_eval] initial reference set @ iter "
                    f"{self._reference.loaded_iteration}"
                )

        if not self._reference.is_set:
            # Pool still empty; nothing to evaluate against.
            return

        # 2. Run K deterministic episodes; record outcomes.
        for _ in range(self._episodes_per_cycle):
            outcome = self._run_one_episode()
            self._tracker.record(outcome)

        # 3. Log every cycle: ref iteration, window fill.
        self.logger.record(
            "eval/reference_iteration", float(self._reference.loaded_iteration)
        )
        self.logger.record("eval/window_size", float(len(self._tracker)))

        # 4. Win-rate-gated promotion (only when window is full).
        if self._tracker.is_full:
            rate = self._tracker.win_rate
            self.logger.record("eval/win_rate_vs_reference", rate)
            if self.verbose >= 1:
                print(
                    f"[ref_eval] step {self.num_timesteps:>9,}: "
                    f"win_rate={rate:.3f} vs ref_iter={self._reference.loaded_iteration}"
                )
            self._maybe_promote(rate)

    def _maybe_promote(self, rate: float) -> None:
        if rate < self._promotion_threshold:
            return
        latest = self._pool.latest()
        if latest.iteration == self._reference.loaded_iteration:
            # Pool hasn't grown since current ref was set — nothing to
            # promote to. Tracker keeps sliding; we'll re-evaluate next
            # cycle and promote as soon as a newer snapshot lands.
            return
        old_iter = self._reference.loaded_iteration
        self._reference.set_snapshot(latest)
        self._tracker.reset()
        if self.verbose >= 1:
            print(
                f"[ref_eval] PROMOTED: ref iter {old_iter} → "
                f"{self._reference.loaded_iteration} "
                f"(win_rate {rate:.3f} ≥ {self._promotion_threshold:.2f})"
            )

    def _run_one_episode(self) -> Outcome:
        """Run a single eval episode to termination/truncation. Returns
        the outcome from the learner's POV: WIN / LOSS / DRAW."""
        assert self._eval_env is not None
        obs, _ = self._eval_env.reset()
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        while not (terminated or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = self._eval_env.step(action)
        if info.get("scored_for_us", False):
            return Outcome.WIN
        if info.get("scored_against_us", False):
            return Outcome.LOSS
        return Outcome.DRAW
