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

import warnings
from pathlib import Path
import textwrap

import pytest

from AtomGym.rewards import (
    BallAlignmentReward,
    BallProgressReward,
    DistanceToBallReward,
    GoalieBoxPenalty,
    StallPenaltyReward,
    StaticFieldPenalty,
)
from AtomGym.training.config import (
    REWARD_REGISTRY,
    ConfigError,
    ConfigSignWarning,
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
    """Smoke-check: every reward in the registry constructs successfully
    when given the correctly-signed weight per its
    `expected_weight_sign`. Includes `env.goalie_box_terminal_time > 0`
    so the goalie_box reward (which requires the rule to be enabled)
    can construct."""
    body = "env:\n  goalie_box_terminal_time: 3.0\nrewards:\n"
    for name, cls in REWARD_REGISTRY.items():
        sign = getattr(cls, "expected_weight_sign", 0)
        # +1 or 0 → positive default; -1 → negative default.
        weight = -0.1 if sign < 0 else 0.1
        body += f"  {name}:\n    weight: {weight}\n"
    p = _write(tmp_path, body)
    # No sign warnings should fire on a correctly-signed config.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigSignWarning)
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
    names = {t.name for t in cfg.rewards}
    assert "ball_progress" in names
    assert "ball_alignment" in names
    assert "goal_scored" in names
    # If the goalie_box reward is configured, the env rule MUST be
    # enabled for the config to be self-consistent (the loader enforces
    # this — see test_goalie_box_reward_with_explicit_env_zero_errors).
    if "goalie_box" in names:
        assert cfg.env_kwargs.get("goalie_box_terminal_time", 0.0) > 0.0


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


def test_goalie_box_inherits_game_rules_from_env(tmp_path: Path) -> None:
    """Game-rule values (terminal_time, box geometry) flow from env →
    reward at load time. The reward block should NOT contain them."""
    p = _write(tmp_path, """
        env:
          goalie_box_terminal_time: 2.5
          goalie_box_depth: 0.10
          goalie_box_y_half: 0.08
        rewards:
          goalie_box:
            weight: -50.0
            trigger_time: 1.5
            power: 4.0
            termination_penalty: 1.5
            depth_saturation: 0.08
    """)
    cfg = load_training_config(p)
    assert len(cfg.rewards) == 1
    term = cfg.rewards[0]
    assert isinstance(term, GoalieBoxPenalty)
    # Training-shaping params from the reward block.
    assert term.weight == pytest.approx(-50.0)
    assert term.trigger_time == pytest.approx(1.5)
    assert term.power == pytest.approx(4.0)
    assert term.termination_penalty == pytest.approx(1.5)
    assert term.depth_saturation == pytest.approx(0.08)
    # Game-rule params inherited from env.
    assert term.terminal_time == pytest.approx(2.5)
    assert term.goalie_box_depth == pytest.approx(0.10)
    assert term.goalie_box_y_half == pytest.approx(0.08)


def test_setting_game_rule_in_reward_block_errors(tmp_path: Path) -> None:
    """terminal_time / box geometry MUST come from env. Setting them
    inside the reward block is a single-source-of-truth violation."""
    p = _write(tmp_path, """
        env:
          goalie_box_terminal_time: 3.0
        rewards:
          goalie_box:
            weight: -50.0
            terminal_time: 5.0       # WRONG — owned by env
    """)
    with pytest.raises(ConfigError, match="game-rule value owned by env"):
        load_training_config(p)


def test_setting_box_geometry_in_reward_block_errors(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        env:
          goalie_box_terminal_time: 3.0
        rewards:
          goalie_box:
            weight: -50.0
            goalie_box_depth: 0.10   # WRONG — owned by env
    """)
    with pytest.raises(ConfigError, match="goalie_box_depth"):
        load_training_config(p)


def test_goalie_box_reward_without_env_rule_enabled_errors(tmp_path: Path) -> None:
    """Configuring the reward while the env rule is disabled
    (terminal_time = 0, the default) is the most common foot-gun. The
    loader catches it with a specific, actionable error message rather
    than letting the user see the cryptic 'terminal_time must be >
    trigger_time' from the reward constructor."""
    p = _write(tmp_path, """
        rewards:
          goalie_box:
            weight: -50.0
            trigger_time: 1.5
    """)
    with pytest.raises(ConfigError, match="env.goalie_box_terminal_time"):
        load_training_config(p)


def test_goalie_box_reward_with_explicit_env_zero_errors(tmp_path: Path) -> None:
    """Same as above but with the env value explicitly set to 0 — the
    error path must trigger regardless of whether the user just
    omitted the env key or actively wrote 0.0."""
    p = _write(tmp_path, """
        env:
          goalie_box_terminal_time: 0.0
        rewards:
          goalie_box:
            weight: -50.0
            trigger_time: 1.5
    """)
    with pytest.raises(ConfigError, match="rule disabled"):
        load_training_config(p)


def test_env_only_box_termination_without_reward_is_allowed(tmp_path: Path) -> None:
    """Setting `env.goalie_box_terminal_time` without the
    `goalie_box` reward is a legitimate ablation — pure rule-based
    termination, no shaping. The loader should not reject it."""
    p = _write(tmp_path, """
        env:
          goalie_box_terminal_time: 3.0
        rewards:
          ball_progress:
            weight: 1.0
    """)
    cfg = load_training_config(p)
    assert cfg.env_kwargs.get("goalie_box_terminal_time") == 3.0
    assert all(t.name != "goalie_box" for t in cfg.rewards)


# ---------------------------------------------------------------------------
# Weight-sign validator (ConfigSignWarning)
# ---------------------------------------------------------------------------


def test_sign_check_warns_on_positive_weight_for_penalty_term() -> None:
    """StallPenaltyReward expects NEGATIVE weight (it's an unsigned-
    magnitude penalty). A positive weight would actively reward
    stalling — exactly the bug the warning catches."""
    with pytest.warns(ConfigSignWarning, match="StallPenaltyReward"):
        validate_and_construct(StallPenaltyReward, {"weight": 0.5})


def test_sign_check_warns_on_negative_weight_for_signed_term() -> None:
    """BallProgressReward returns SIGNED progress (m/s toward goal).
    Negative weight inverts the goal direction — likely a bug."""
    with pytest.warns(ConfigSignWarning, match="BallProgressReward"):
        validate_and_construct(BallProgressReward, {"weight": -1.0})


def test_sign_check_silent_on_correct_signs() -> None:
    """No warning when YAML signs match the term's expectation."""
    pairs = [
        (StallPenaltyReward, -0.5),
        (DistanceToBallReward, -0.25),
        (BallProgressReward, +1.0),
        (BallAlignmentReward, +0.3),
        (GoalieBoxPenalty, -50.0),
        (StaticFieldPenalty, -0.5),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigSignWarning)
        for cls, w in pairs:
            validate_and_construct(cls, {"weight": w})


def test_sign_check_silent_on_zero_weight() -> None:
    """A zero weight is a no-op — sign is meaningless. Skip the check
    so 'I want to disable this term but keep it in the config for
    documentation' patterns don't spam warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigSignWarning)
        # StallPenaltyReward expects negative; 0 is neither positive
        # nor negative and shouldn't warn.
        validate_and_construct(StallPenaltyReward, {"weight": 0.0})


def test_sign_check_fires_through_yaml_loader(tmp_path: Path) -> None:
    """End-to-end: a wrongly-signed weight in the YAML produces the
    warning when `load_training_config` runs."""
    p = _write(tmp_path, """
        rewards:
          stall_penalty:
            weight: 0.3      # WRONG SIGN — should be negative
    """)
    with pytest.warns(ConfigSignWarning, match="StallPenaltyReward"):
        load_training_config(p)


def test_every_registered_term_declares_expected_sign() -> None:
    """Drift guard: every reward term in `REWARD_REGISTRY` should
    declare an explicit `expected_weight_sign` (not inherit the base
    class's default 0). 0 is the 'no opinion' value; if a new term
    really wants no opinion, override here to 0 explicitly so this
    test acknowledges the choice."""
    for cls in REWARD_REGISTRY.values():
        own = cls.__dict__.get("expected_weight_sign")
        assert own is not None and own in (-1, 0, +1), (
            f"{cls.__name__}: must declare `expected_weight_sign` as a "
            f"class attribute (-1, 0, or +1). Inheriting the base "
            f"default leaves the validator silent — set explicitly."
        )


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
