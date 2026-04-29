"""Unit tests for the reward base machinery (RewardContext, RewardTerm,
RewardComposite). Pure-python — no sim_py needed."""

from __future__ import annotations

import numpy as np
import pytest

from AtomGym.action_observation import ActionView, ObsView
from AtomGym.rewards import RewardComposite, RewardContext, RewardTerm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    *,
    obs: np.ndarray | None = None,
    prev_obs: np.ndarray | None = None,
    info: dict | None = None,
) -> RewardContext:
    obs = obs if obs is not None else np.zeros(11, dtype=np.float32)
    return RewardContext(
        obs=obs,
        action=np.zeros(2, dtype=np.float32),
        prev_obs=prev_obs,
        prev_action=None,
        info=info if info is not None else {},
        obs_view=ObsView(n_robots=1),
        action_view=ActionView(),
        field_x_half=0.375,
        field_y_half=0.225,
        goal_y_half=0.06,
        goal_extension=0.06,
        dt=1.0 / 60.0,
    )


class ConstantTerm(RewardTerm):
    """Always returns the same value — useful for testing the composite."""

    def __init__(self, name: str, value: float, weight: float = 1.0) -> None:
        super().__init__(weight=weight)
        self.name = name
        self._value = value

    def __call__(self, ctx: RewardContext) -> float:
        return self._value


class ObsScalarTerm(RewardTerm):
    """Returns obs[index] — for testing that ctx.obs is plumbed correctly."""

    name = "obs_scalar"

    def __init__(self, index: int, weight: float = 1.0) -> None:
        super().__init__(weight=weight)
        self._index = index

    def __call__(self, ctx: RewardContext) -> float:
        return float(ctx.obs[self._index])


# ---------------------------------------------------------------------------
# RewardTerm — abstract base behavior
# ---------------------------------------------------------------------------


def test_reward_term_is_abstract() -> None:
    """Can't instantiate the base class — `__call__` is abstract."""
    with pytest.raises(TypeError):
        RewardTerm()  # type: ignore[abstract]


def test_reward_term_default_weight_is_one() -> None:
    term = ConstantTerm("x", 5.0)
    assert term.weight == 1.0


def test_reward_term_weight_is_settable() -> None:
    term = ConstantTerm("x", 5.0, weight=3.0)
    assert term.weight == 3.0


# ---------------------------------------------------------------------------
# RewardComposite — summing, breakdown, weighting
# ---------------------------------------------------------------------------


def test_composite_empty_returns_zero() -> None:
    composite = RewardComposite()
    total, breakdown = composite(_make_ctx())
    assert total == 0.0
    assert breakdown == {}


def test_composite_single_term() -> None:
    composite = RewardComposite([ConstantTerm("x", 1.5)])
    total, breakdown = composite(_make_ctx())
    assert total == pytest.approx(1.5)
    assert breakdown == {"x": pytest.approx(1.5)}


def test_composite_sums_terms() -> None:
    composite = RewardComposite([
        ConstantTerm("a", 1.0),
        ConstantTerm("b", 2.0),
        ConstantTerm("c", -0.5),
    ])
    total, breakdown = composite(_make_ctx())
    assert total == pytest.approx(2.5)
    assert breakdown == {
        "a": pytest.approx(1.0),
        "b": pytest.approx(2.0),
        "c": pytest.approx(-0.5),
    }


def test_composite_applies_weights() -> None:
    composite = RewardComposite([
        ConstantTerm("a", 1.0, weight=2.0),
        ConstantTerm("b", 3.0, weight=-1.0),
    ])
    total, breakdown = composite(_make_ctx())
    # weight × value: 2·1 + (-1)·3 = -1
    assert total == pytest.approx(-1.0)
    assert breakdown == {"a": pytest.approx(2.0), "b": pytest.approx(-3.0)}


def test_composite_fluent_add() -> None:
    composite = RewardComposite()
    returned = composite.add(ConstantTerm("x", 1.0)).add(ConstantTerm("y", 2.0))
    assert returned is composite
    assert len(composite) == 2
    assert composite.names() == ["x", "y"]


def test_composite_term_can_read_obs() -> None:
    obs = np.array([0.7] + [0.0] * 10, dtype=np.float32)
    composite = RewardComposite([ObsScalarTerm(index=0)])
    total, breakdown = composite(_make_ctx(obs=obs))
    assert total == pytest.approx(0.7)


def test_composite_term_can_read_info() -> None:
    """Reward terms read event flags from ctx.info — verify plumbing."""

    class FlagTerm(RewardTerm):
        name = "flag"

        def __call__(self, ctx: RewardContext) -> float:
            return 1.0 if ctx.info.get("scored", False) else 0.0

    composite = RewardComposite([FlagTerm()])
    total_off, _ = composite(_make_ctx(info={"scored": False}))
    total_on, _ = composite(_make_ctx(info={"scored": True}))
    assert total_off == 0.0
    assert total_on == pytest.approx(1.0)


def test_composite_duplicate_names_accumulate() -> None:
    """If two terms share a name, contributions accumulate under the one key
    rather than the last-wins clobbering. Total stays correct; breakdown
    groups them together."""
    composite = RewardComposite([
        ConstantTerm("dup", 1.0),
        ConstantTerm("dup", 3.0, weight=2.0),  # contributes 6.0
        ConstantTerm("solo", 5.0),
    ])
    total, breakdown = composite(_make_ctx())
    assert total == pytest.approx(12.0)  # 1 + 6 + 5
    assert breakdown == {
        "dup": pytest.approx(7.0),    # 1 + 6
        "solo": pytest.approx(5.0),
    }


def test_composite_term_handles_no_prev_obs() -> None:
    """Reward terms must accept ctx.prev_obs is None on the first step."""

    class DeltaTerm(RewardTerm):
        name = "delta"

        def __call__(self, ctx: RewardContext) -> float:
            if ctx.prev_obs is None:
                return 0.0
            return float(ctx.obs[0] - ctx.prev_obs[0])

    obs = np.array([0.5] + [0.0] * 10, dtype=np.float32)
    prev = np.array([0.3] + [0.0] * 10, dtype=np.float32)
    composite = RewardComposite([DeltaTerm()])

    # First step (no prev) → 0
    first, _ = composite(_make_ctx(obs=obs, prev_obs=None))
    assert first == 0.0

    # Subsequent step → delta
    after, _ = composite(_make_ctx(obs=obs, prev_obs=prev))
    assert after == pytest.approx(0.2)
