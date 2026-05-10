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
import warnings
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
    "goalie_box_corner_radius",
    "goalie_box_terminal_time",
    "manipulator",
    "physics_dt",
    "control_dt",
})


# Game-rule parameters that exist on BOTH the env constructor and a
# reward term. The split is principled:
#
#   * env owns the GAME RULE — what's legal, when termination fires.
#     `goalie_box_terminal_time` (the budget) and the box geometry are
#     properties of the game itself.
#   * reward owns the TRAINING SHAPING — `weight`, `trigger_time`,
#     `power`, `termination_penalty`, `depth_saturation`. These define
#     HOW we shape the learning signal *within* the game's rules. They
#     have no game-mechanical meaning.
#
# Game-rule values flow from env → reward at config-load time so
# there's a single source of truth. Setting them inside the reward
# block is forbidden — the loader rejects it with a clear error.
#
# Schema: { reward_name: { reward_kwarg: env_kwarg } }
_REWARD_INHERITS_FROM_ENV: dict[str, dict[str, str]] = {
    "goalie_box": {
        "terminal_time": "goalie_box_terminal_time",
        "goalie_box_depth": "goalie_box_depth",
        "goalie_box_y_half": "goalie_box_y_half",
        "goalie_box_corner_radius": "goalie_box_corner_radius",
    },
}


# Cross-section auto-fill: parameters that exist on both env constructor
# AND a reward term, where they MUST agree for the system to behave
# correctly. The env is the source of truth; the loader copies the
# value into the reward's params dict before construction so the user
# never specifies the same number twice.
#
# Schema: { reward_name: { reward_kwarg_name: env_kwarg_name } }
#
# If the user explicitly sets one of these in the reward block, we raise
# ConfigError — there's only one place to set it. If `env` doesn't
# supply the value, the reward's __init__ will reject the auto-filled
# default (e.g. GoalieBoxPenalty rejects terminal_time=0 since
# terminal_time must exceed trigger_time), giving a clear error.
_REWARD_AUTOFILL_FROM_ENV: dict[str, dict[str, str]] = {
    "goalie_box": {
        "terminal_time": "goalie_box_terminal_time",
        "goalie_box_depth": "goalie_box_depth",
        "goalie_box_y_half": "goalie_box_y_half",
    },
}


class ConfigError(ValueError):
    """Raised on any structural problem in a training-config YAML —
    unknown keys, missing required reward kwargs, type-mismatched
    values, etc. Inherits from ValueError so existing `except
    ValueError` callers still catch it."""


class ConfigSignWarning(UserWarning):
    """Emitted (NOT raised) when a YAML weight contradicts the
    reward term's `expected_weight_sign`. Likely a sign-flip bug
    that would silently train the wrong objective. Surfaces via
    Python's `warnings` machinery so it shows up on stdout / in
    pytest output by default but doesn't refuse to construct (some
    legitimate ablations want a deliberately reversed sign)."""


def _inherit_env_rules(
    name: str, params: dict[str, Any], env_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Apply the env-owns-game-rules inheritance pattern: for each
    reward kwarg listed in `_REWARD_INHERITS_FROM_ENV`, pull the value
    from the env section and inject it into the reward's params.

    Errors if the user explicitly set a game-rule value inside the
    reward block — there's only one place to set it.

    Errors with a specific message if the goalie-box rule is configured
    via the reward but the env hasn't enabled it (terminal_time = 0
    means rule disabled). Without this special case the user would see
    a confusing "GoalieBoxPenalty: terminal_time must be > trigger_time"
    error, which is technically correct but doesn't say where to fix it."""
    inheritance = _REWARD_INHERITS_FROM_ENV.get(name)
    if not inheritance:
        return params
    for reward_kwarg, env_kwarg in inheritance.items():
        if reward_kwarg in params:
            raise ConfigError(
                f"reward {name!r}: must not set {reward_kwarg!r} in the "
                f"reward block — that's a game-rule value owned by "
                f"env.{env_kwarg!r}. Set env.{env_kwarg} once and the "
                f"reward will inherit it."
            )
        if env_kwarg in env_kwargs:
            params[reward_kwarg] = env_kwargs[env_kwarg]
    # Targeted check for the most likely user-facing error: rule
    # disabled at the env level but the shaping reward is configured.
    if name == "goalie_box":
        if env_kwargs.get("goalie_box_terminal_time", 0.0) <= 0.0:
            raise ConfigError(
                "rewards.goalie_box is configured but env.goalie_box_terminal_time "
                "is 0 (rule disabled). The reward shapes behaviour against the "
                "game rule — without the rule there's nothing to shape. "
                "Set env.goalie_box_terminal_time > 0 to enable, or remove the "
                "rewards.goalie_box block."
            )
    return params


def _check_weight_sign(cls: type, weight: float) -> None:
    """If the reward class declares `expected_weight_sign != 0` and
    the supplied weight has the opposite sign, emit a warning. Zero
    weight skips the check (a no-op term doesn't need an opinion on
    sign). The +1 / -1 convention is documented on `RewardTerm`."""
    expected = getattr(cls, "expected_weight_sign", 0)
    if expected == 0 or weight == 0.0:
        return
    actual = +1 if weight > 0 else -1
    if actual != expected:
        wanted = "POSITIVE" if expected > 0 else "NEGATIVE"
        warnings.warn(
            f"{cls.__name__}: weight={weight} has the wrong sign — this "
            f"term expects a {wanted} weight (see its docstring). The "
            f"agent will be optimised against the term, not for it. "
            f"If this is intentional (e.g. an ablation), ignore this "
            f"warning. Otherwise flip the sign in your YAML.",
            ConfigSignWarning,
            stacklevel=3,
        )


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
    if "weight" in params:
        _check_weight_sign(cls, float(params["weight"]))
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
        # Game-rule inheritance: env is the source of truth for game-rule
        # parameters (terminal_time, box geometry). Copy them into the
        # reward params before construction so the reward block stays
        # focused on training-shaping knobs.
        params = _inherit_env_rules(name, dict(params), env_kwargs)
        cls = REWARD_REGISTRY[name]
        rewards.append(validate_and_construct(cls, params))

    return TrainingConfig(
        rewards=rewards,
        env_kwargs=env_kwargs,
        raw=raw,
    )
