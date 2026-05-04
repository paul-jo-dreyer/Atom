"""YAML-driven training config: reward weights + env shaping params in
one versioned file.

Why this exists
---------------
Tracking the hyperparameters of a long-running training experiment via
shell history (or hand-rolled `--this --that` flag soup) is fragile.
Move them into a YAML committed alongside the run, and the diff between
two experiments is a single readable file.

What lives in YAML vs CLI
-------------------------
- **YAML** owns *what defines the experiment*: reward weights and
  per-term shaping params, env shaping (goalie-box geometry, episode
  cap, manipulator), DR ranges. Things you'd want to compare across
  runs to understand a behavioural delta.
- **CLI** owns *how the experiment runs*: `--n-envs`, `--total-timesteps`,
  `--learning-rate`, `--seed`, `--run-name`, `--resume`, render flags.
  Things that change wall-clock or output paths but don't change the
  optimisation problem itself.

This split is opinionated but matches the user's framing
("tracking results over time easier rather than tracking cli inputs").

Schema
------
See `AtomGym/configs/default_solo.yaml` for a full example.

```yaml
env:
  max_episode_steps: 800
  goalie_box_depth: 0.12
  goalie_box_y_half: 0.10
  goalie_box_terminal_time: 0.0   # 0 disables the box-time rule
  manipulator: null

rewards:
  ball_progress:
    weight: 1.0
  ball_alignment:
    weight: 0.3
    # inner_radius / outer_radius / back_weight: class defaults
  goalie_box:
    weight: -20.0
    trigger_time: 2.0
    terminal_time: 3.0
    power: 3.0
    termination_penalty: 1.0
    depth_saturation: 0.06
  # Omit a key entirely to disable that reward term.
```

Validation
----------
`load_training_config` introspects each reward's `__init__` signature
via `inspect.signature` and verifies that the YAML keys are a subset
of the constructor's accepted kwargs. Missing required kwargs and
unknown keys both surface as clear `ConfigError`s — preferable to a
bare `TypeError` deep inside model construction or, worse, a silent
defaulting that masks the typo.

Adding a new reward term
------------------------
Add an entry to `REWARD_REGISTRY` mapping its `name` to its class. The
loader picks it up automatically — no schema-file editing needed,
since the validator pulls allowed kwargs from `inspect.signature` at
load time.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from AtomGym.rewards import (
    BallAlignmentReward,
    BallProgressReward,
    DistanceToBallReward,
    GoalScoredReward,
    GoalieBoxPenalty,
    ObstacleContactPenalty,
    RewardTerm,
    StallPenaltyReward,
    StaticFieldPenalty,
)


# Map YAML keys to reward classes. The key MUST match `RewardTerm.name`
# so the per-term TensorBoard breakdown lines up with the config file.
REWARD_REGISTRY: dict[str, type[RewardTerm]] = {
    BallAlignmentReward.name: BallAlignmentReward,
    BallProgressReward.name: BallProgressReward,
    DistanceToBallReward.name: DistanceToBallReward,
    GoalScoredReward.name: GoalScoredReward,
    GoalieBoxPenalty.name: GoalieBoxPenalty,
    ObstacleContactPenalty.name: ObstacleContactPenalty,
    StallPenaltyReward.name: StallPenaltyReward,
    StaticFieldPenalty.name: StaticFieldPenalty,
}


# Whitelist of kwargs the env constructors accept from YAML. Anything
# else under `env:` is a typo and we want to surface it. Keep this in
# sync with `AtomSoloEnv.__init__` / `AtomTeamEnv.__init__`. Inspecting
# at load time is also possible (and would be drift-proof) but the
# fixed list makes the schema self-documenting.
_ALLOWED_ENV_KEYS: frozenset[str] = frozenset({
    "max_episode_steps",
    "goalie_box_depth",
    "goalie_box_y_half",
    "goalie_box_terminal_time",
    "manipulator",
    "physics_dt",
    "control_dt",
})


class ConfigError(ValueError):
    """Raised on any structural problem in a training-config YAML —
    unknown keys, missing required reward kwargs, type-mismatched
    values, etc. Inherits from ValueError so existing `except
    ValueError` callers still catch it."""


# ---------------------------------------------------------------------------
# Validation utility — also useful as a stand-alone helper for tests
# ---------------------------------------------------------------------------


def validate_and_construct(
    cls: type[RewardTerm], params: dict[str, Any]
) -> RewardTerm:
    """Construct a reward term from a YAML-decoded params dict, with
    schema validation against the class's `__init__` signature.

    Checks:
      * No unknown keys (typo guard).
      * All required (no-default) parameters present.

    `cls.__init__` is inspected via `inspect.signature`; defaulted
    parameters are optional in YAML. The class's own `__init__`
    validation (e.g. value-range checks) still runs as the final
    line of defence — the validator here only catches *schema*
    errors, not value-range errors."""
    sig = inspect.signature(cls.__init__)
    allowed = {name for name in sig.parameters if name != "self"}
    required = {
        name for name, p in sig.parameters.items()
        if name != "self" and p.default is inspect.Parameter.empty
    }
    provided = set(params.keys())

    unknown = provided - allowed
    if unknown:
        raise ConfigError(
            f"reward {cls.__name__}: unknown keys {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}."
        )
    missing = required - provided
    if missing:
        raise ConfigError(
            f"reward {cls.__name__}: missing required keys {sorted(missing)}."
        )
    try:
        return cls(**params)
    except TypeError as e:
        raise ConfigError(
            f"reward {cls.__name__}: failed to construct with {params!r}: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Top-level config dataclass + loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """Validated, ready-to-use config for an env factory.

    `rewards` is a list of constructed RewardTerms; `env_kwargs` is a
    dict that can be `**`-spread into `AtomSoloEnv` / `AtomTeamEnv`
    constructors. The factory just merges this with the runtime kwargs
    (seed, etc.) coming from the CLI.

    `raw` retains the parsed YAML so consumers (e.g. logging code that
    wants to dump the experiment config to TensorBoard) can serialise
    the entire config back out without re-reading the file."""

    rewards: list[RewardTerm]
    env_kwargs: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load + validate a YAML config. Raises `ConfigError` for any
    schema issue."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raise ConfigError(f"config file is empty: {path}")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config file root must be a mapping, got {type(raw).__name__}: {path}"
        )

    # --- env section --------------------------------------------------------
    env_raw = raw.get("env", {}) or {}
    if not isinstance(env_raw, dict):
        raise ConfigError(f"`env:` must be a mapping, got {type(env_raw).__name__}")
    unknown_env = set(env_raw.keys()) - _ALLOWED_ENV_KEYS
    if unknown_env:
        raise ConfigError(
            f"env: unknown keys {sorted(unknown_env)}. "
            f"Allowed: {sorted(_ALLOWED_ENV_KEYS)}."
        )
    env_kwargs = dict(env_raw)

    # --- rewards section ----------------------------------------------------
    rewards_raw = raw.get("rewards", {}) or {}
    if not isinstance(rewards_raw, dict):
        raise ConfigError(
            f"`rewards:` must be a mapping, got {type(rewards_raw).__name__}"
        )
    rewards: list[RewardTerm] = []
    for name, params in rewards_raw.items():
        if name not in REWARD_REGISTRY:
            raise ConfigError(
                f"unknown reward {name!r}. Known: {sorted(REWARD_REGISTRY.keys())}."
            )
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ConfigError(
                f"reward {name!r}: params must be a mapping, got {type(params).__name__}"
            )
        cls = REWARD_REGISTRY[name]
        rewards.append(validate_and_construct(cls, params))

    return TrainingConfig(
        rewards=rewards,
        env_kwargs=env_kwargs,
        raw=raw,
    )
