"""Tests for `AtomGym.training.config` — YAML loader + signature-based
validator + default config files.

Coverage:
    * round-trip a minimal YAML
    * unknown reward key raises with a helpful message
    * unknown reward kwarg (typo guard) raises
    * missing required reward kwarg raises
    * unknown env key raises
    * the shipped default configs (`default_solo.yaml`,
      `default_team.yaml`) are valid and produce sensible output
    * `validate_and_construct` works as a stand-alone helper
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from AtomGym.rewards import (
    BallAlignmentReward,
    BallProgressReward,
    GoalieBoxPenalty,
    StaticFieldPenalty,
)
from AtomGym.training.config import (
    REWARD_REGISTRY,
    ConfigError,
    TrainingConfig,
    load_training_config,
    validate_and_construct,
)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body).lstrip())
    return p


# ---------------------------------------------------------------------------
# validate_and_construct — direct unit tests
# ---------------------------------------------------------------------------


def test_validate_and_construct_uses_class_defaults() -> None:
    """Only `weight` is required; other params fall back to defaults."""
    term = validate_and_construct(BallAlignmentReward, {"weight": 0.5})
    assert isinstance(term, BallAlignmentReward)
    assert term.weight == pytest.approx(0.5)


def test_validate_and_construct_passes_through_all_kwargs() -> None:
    term = validate_and_construct(
        BallAlignmentReward,
        {"weight": 0.7, "inner_radius": 0.0, "outer_radius": 0.20, "back_weight": 0.5},
    )
    assert term.outer_radius == pytest.approx(0.20)
    assert term.back_weight == pytest.approx(0.5)


def test_validate_and_construct_unknown_kwarg_raises() -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        validate_and_construct(
            BallAlignmentReward,
            {"weight": 0.3, "outer_radius_typo": 0.18},
        )


def test_validate_and_construct_propagates_value_errors() -> None:
    """Class-level value-range checks (e.g. outer_radius > inner_radius)
    surface as ConfigError-wrapped TypeErrors are NOT triggered by
    range issues — those raise ValueError directly. Verify the wrapper
    leaves ValueError alone."""
    with pytest.raises(ValueError):  # not necessarily ConfigError
        validate_and_construct(
            BallAlignmentReward,
            {"weight": 0.3, "outer_radius": 0.05, "inner_radius": 0.10},
        )


# ---------------------------------------------------------------------------
# load_training_config — file-level happy path
# ---------------------------------------------------------------------------


def test_minimal_config_loads(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        env:
          max_episode_steps: 400
        rewards:
          ball_progress:
            weight: 1.0
    """)
    cfg = load_training_config(p)
    assert isinstance(cfg, TrainingConfig)
    assert cfg.env_kwargs == {"max_episode_steps": 400}
    assert len(cfg.rewards) == 1
    assert isinstance(cfg.rewards[0], BallProgressReward)


def test_omitted_reward_disables_term(tmp_path: Path) -> None:
    """A reward key not present in YAML ⟹ that term is not added.
    The composite is only as long as the YAML keys."""
    p = _write(tmp_path, """
        rewards:
          ball_progress:
            weight: 1.0
    """)
    cfg = load_training_config(p)
    assert len(cfg.rewards) == 1
    assert all(not isinstance(t, BallAlignmentReward) for t in cfg.rewards)


def test_empty_env_section_ok(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        rewards:
          ball_progress:
            weight: 1.0
    """)
    cfg = load_training_config(p)
    assert cfg.env_kwargs == {}


def test_complete_config_constructs_all_rewards(tmp_path: Path) -> None:
    """Smoke-check: every reward in the registry can be configured
    with weight: 0.1 and constructs successfully."""
    body = "rewards:\n"
    for name in REWARD_REGISTRY:
        body += f"  {name}:\n    weight: 0.1\n"
    p = _write(tmp_path, body)
    cfg = load_training_config(p)
    assert len(cfg.rewards) == len(REWARD_REGISTRY)


# ---------------------------------------------------------------------------
# load_training_config — error paths
# ---------------------------------------------------------------------------


def test_unknown_reward_name_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        rewards:
          ball_alignment_typo:
            weight: 0.3
    """)
    with pytest.raises(ConfigError, match="unknown reward"):
        load_training_config(p)


def test_unknown_reward_kwarg_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        rewards:
          ball_alignment:
            weight: 0.3
            outer_radius_typo: 0.18
    """)
    with pytest.raises(ConfigError, match="unknown keys"):
        load_training_config(p)


def test_unknown_env_key_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        env:
          max_episode_steps_typo: 400
    """)
    with pytest.raises(ConfigError, match="env: unknown"):
        load_training_config(p)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_training_config(tmp_path / "does_not_exist.yaml")


def test_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(ConfigError, match="empty"):
        load_training_config(p)


def test_non_mapping_root_raises(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_training_config(p)


def test_rewards_section_not_a_mapping_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        rewards:
          - ball_progress
          - ball_alignment
    """)
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_training_config(p)


def test_per_reward_params_not_a_mapping_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        rewards:
          ball_alignment: 0.3
    """)
    with pytest.raises(ConfigError, match="params must be a mapping"):
        load_training_config(p)


# ---------------------------------------------------------------------------
# Shipped default configs validate
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_default_solo_config_loads() -> None:
    """Shipped default config must always be valid — it's the example
    config that users will copy from."""
    cfg = load_training_config(_REPO_ROOT / "AtomGym" / "configs" / "default_solo.yaml")
    # Default has the goalie-box rule disabled.
    assert cfg.env_kwargs.get("goalie_box_terminal_time", 0.0) == 0.0
    # Default reward set: 6 terms (no static_field, no goalie_box).
    names = {t.name for t in cfg.rewards}
    assert "ball_progress" in names
    assert "ball_alignment" in names
    assert "goal_scored" in names
    assert "goalie_box" not in names  # disabled by default


def test_default_team_config_loads() -> None:
    cfg = load_training_config(_REPO_ROOT / "AtomGym" / "configs" / "default_team.yaml")
    assert cfg.env_kwargs.get("goalie_box_terminal_time", 0.0) == 0.0
    names = {t.name for t in cfg.rewards}
    assert "ball_progress" in names
    assert "ball_alignment" in names


# ---------------------------------------------------------------------------
# Schema-driven discovery — adding a new reward to REWARD_REGISTRY is
# enough for the loader to accept it. This test guards the contract.
# ---------------------------------------------------------------------------


def test_registry_keys_match_class_name_attribute() -> None:
    """The YAML key MUST equal the class's `name` attribute. Otherwise
    the per-term TensorBoard breakdown won't line up with the config
    file (the breakdown is keyed by `name`)."""
    for key, cls in REWARD_REGISTRY.items():
        assert cls.name == key, (
            f"REWARD_REGISTRY key {key!r} != {cls.__name__}.name "
            f"({cls.name!r}) — config and TB log will diverge."
        )


# ---------------------------------------------------------------------------
# Specific GoalieBoxPenalty smoke (most-complex term — make sure all
# its fields plumb through cleanly)
# ---------------------------------------------------------------------------


def test_goalie_box_full_config_round_trip(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        rewards:
          goalie_box:
            weight: -50.0
            trigger_time: 1.5
            terminal_time: 2.5
            power: 4.0
            termination_penalty: 1.5
            depth_saturation: 0.08
            goalie_box_depth: 0.10
            goalie_box_y_half: 0.08
    """)
    cfg = load_training_config(p)
    assert len(cfg.rewards) == 1
    term = cfg.rewards[0]
    assert isinstance(term, GoalieBoxPenalty)
    assert term.weight == pytest.approx(-50.0)
    assert term.trigger_time == pytest.approx(1.5)
    assert term.terminal_time == pytest.approx(2.5)
    assert term.power == pytest.approx(4.0)
    assert term.termination_penalty == pytest.approx(1.5)
    assert term.depth_saturation == pytest.approx(0.08)
    assert term.goalie_box_depth == pytest.approx(0.10)
    assert term.goalie_box_y_half == pytest.approx(0.08)


def test_static_field_with_include_goalie_box_round_trip(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        rewards:
          static_field:
            weight: -0.5
            include_goalie_box: true
            penalize_own_box: false
    """)
    cfg = load_training_config(p)
    assert len(cfg.rewards) == 1
    term = cfg.rewards[0]
    assert isinstance(term, StaticFieldPenalty)
    assert term.include_goalie_box is True
    assert term.penalize_own_box is False
