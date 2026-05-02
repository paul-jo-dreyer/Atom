"""Per-worker wrapper that bundles `AtomTeamEnv` with an `OpponentRunner`.

Each subproc constructs one of these via the env factory passed to
SubprocVecEnv. The runner's `predict` is bound as the env's
`opponent_policy` at construction; from then on, every env step runs
the opponent's forward pass on the worker's CPU shadow policy.

Main process syncs the pool replica via
    vec_env.env_method('update_opponent_pool', pool)
which dispatches to `update_opponent_pool` on each worker's wrapper.
SB3's vec_env wraps each env in `Monitor`; gym.Wrapper's `__getattr__`
delegation lets `env_method` find this method through the Monitor
chain.

The wrapper is intentionally minimal — it adds one method on top of
gym.Wrapper's pass-through behaviour. All env attribute access
(`env.world`, `env.robot`, `env.opponent`, `env.ball`, `env.field_x_half`,
etc.) passes through to the underlying `AtomTeamEnv`.
"""

from __future__ import annotations

import gymnasium as gym

from AtomGym.environments import AtomTeamEnv
from AtomGym.training.opponent_runner import OpponentRunner
from AtomGym.training.snapshot_pool import SnapshotPool


class TeamWorkerWrapper(gym.Wrapper):
    """Bundles a team env with its CPU opponent runner. Lifetime of
    both is tied to the wrapper's lifetime (typically the subproc's
    lifetime)."""

    def __init__(self, team_env: AtomTeamEnv, runner: OpponentRunner) -> None:
        super().__init__(team_env)
        self._runner = runner
        team_env.set_opponent_policy(runner.predict)

    def update_opponent_pool(self, new_pool: SnapshotPool) -> None:
        """Sync hook called from main via env_method. Atomically replaces
        the worker's pool replica AND resamples the opponent (per
        `OpponentRunner.update_pool`)."""
        self._runner.update_pool(new_pool)

    @property
    def runner(self) -> OpponentRunner:
        """Exposed for tests / debugging — production code should not
        reach into the runner directly; use `env_method` instead."""
        return self._runner
