"""Unit tests for GoalieBoxPenalty (pure-python, no sim_py).

Coverage:
    * shape — silent below trigger, polynomial ramp from trigger to terminal
    * depth weighting — at-boundary 0, deep inside ~1, outside the box 0
    * sparse termination cost — fires exactly when info flag is set
    * sign — term returns POSITIVE magnitude (composite multiplies in sign)
    * weighted total at violation = ramp + termination_penalty
"""

from __future__ import annotations

import numpy as np
import pytest

from AtomGym.action_observation import ActionView, ObsView, build_observation
from AtomGym.rewards import GoalieBoxPenalty, RewardContext


FIELD_X_HALF = 0.375
FIELD_Y_HALF = 0.225
GOAL_Y_HALF = 0.06
BOX_DEPTH = 0.12
BOX_Y_HALF = 0.10
TERMINAL_TIME = 3.0
TRIGGER_TIME = 2.0


def _ctx(
    *,
    rx: float,
    ry: float,
    time_in_box_norm: float,
    info: dict | None = None,
) -> RewardContext:
    """Build a RewardContext with one robot at (rx, ry) and a given
    timer reading. Ball position doesn't matter for this term."""
    obs = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=np.zeros(4, dtype=np.float32),
        self_state_5d=np.array([rx, ry, 0.0, 0.0, 0.0], dtype=np.float32),
        self_time_in_box_norm=time_in_box_norm,
    )
    return RewardContext(
        obs=obs,
        action=np.zeros(2, dtype=np.float32),
        prev_obs=None,
        prev_action=None,
        info=info if info is not None else {},
        obs_view=ObsView(n_robots=1),
        action_view=ActionView(),
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        goal_y_half=GOAL_Y_HALF,
        goal_extension=0.06,
        dt=1.0 / 60.0,
    )


def _term(**overrides) -> GoalieBoxPenalty:
    kwargs = dict(
        weight=1.0,
        trigger_time=TRIGGER_TIME,
        terminal_time=TERMINAL_TIME,
        power=3.0,
        termination_penalty=1.0,
        goalie_box_depth=BOX_DEPTH,
        goalie_box_y_half=BOX_Y_HALF,
    )
    kwargs.update(overrides)
    return GoalieBoxPenalty(**kwargs)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_terminal_must_exceed_trigger() -> None:
    with pytest.raises(ValueError, match="terminal_time"):
        GoalieBoxPenalty(trigger_time=2.0, terminal_time=2.0)
    with pytest.raises(ValueError, match="terminal_time"):
        GoalieBoxPenalty(trigger_time=2.0, terminal_time=1.5)


def test_negative_trigger_rejected() -> None:
    with pytest.raises(ValueError, match="trigger_time"):
        GoalieBoxPenalty(trigger_time=-0.1, terminal_time=1.0)


def test_zero_power_rejected() -> None:
    with pytest.raises(ValueError, match="power"):
        GoalieBoxPenalty(power=0.0)


def test_negative_termination_penalty_rejected() -> None:
    with pytest.raises(ValueError, match="termination_penalty"):
        GoalieBoxPenalty(termination_penalty=-1.0)


def test_depth_floor_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="depth_floor"):
        GoalieBoxPenalty(depth_floor=-0.1)
    with pytest.raises(ValueError, match="depth_floor"):
        GoalieBoxPenalty(depth_floor=1.1)


# ---------------------------------------------------------------------------
# depth_floor — boundary penalty rises smoothly from 0 (legacy) to u^p
# ---------------------------------------------------------------------------


def test_depth_floor_zero_recovers_legacy_boundary_silence() -> None:
    """depth_floor=0 (default) ⟹ boundary still pays 0 even at near-
    terminal time, matching the original potential-field design."""
    term = _term(depth_floor=0.0)
    inner_edge_x = FIELD_X_HALF - BOX_DEPTH
    ctx = _ctx(rx=inner_edge_x, ry=0.0, time_in_box_norm=0.99)
    assert term(ctx) == pytest.approx(0.0, abs=1e-6)


def test_depth_floor_one_makes_penalty_uniform_inside() -> None:
    """depth_floor=1 ⟹ the per-step penalty is uniform across the box
    interior; boundary, centroid, and deep all give u^p."""
    term = _term(depth_floor=1.0, power=3.0)
    trigger_norm = TRIGGER_TIME / TERMINAL_TIME
    # u=0.5 ⟹ u^p = 0.125 (matches test_ramp_polynomial_shape but
    # now everywhere inside the box, not just at centroid).
    time_norm = trigger_norm + 0.5 * (1.0 - trigger_norm)
    inner_edge_x = FIELD_X_HALF - BOX_DEPTH + 1e-4   # just inside
    centroid_x = FIELD_X_HALF - 0.5 * BOX_DEPTH
    deep_x = FIELD_X_HALF - BOX_DEPTH + 0.06          # 1 robot deep
    for x in (inner_edge_x, centroid_x, deep_x):
        ctx = _ctx(rx=x, ry=0.0, time_in_box_norm=time_norm)
        assert term(ctx) == pytest.approx(0.125, abs=1e-3), (
            f"x={x}: expected 0.125, got {term(ctx)}"
        )


def test_depth_floor_half_blends_linearly() -> None:
    """depth_floor=0.5 with u^p=1 (terminal time, just below violation):
    boundary should be ~0.5, centroid ~1.0."""
    term = _term(depth_floor=0.5, power=1.0)  # linear ramp for cleaner math
    # u=1 ⟹ u^p = 1. time_norm = 1.0 (just below terminal trigger).
    # Use 0.999 to avoid the violation flag interaction.
    time_norm = 0.99999
    inner_edge_x = FIELD_X_HALF - BOX_DEPTH + 1e-4
    centroid_x = FIELD_X_HALF - 0.5 * BOX_DEPTH
    val_boundary = term(_ctx(rx=inner_edge_x, ry=0.0, time_in_box_norm=time_norm))
    val_centroid = term(_ctx(rx=centroid_x, ry=0.0, time_in_box_norm=time_norm))
    assert val_boundary == pytest.approx(0.5, abs=5e-3)
    assert val_centroid == pytest.approx(1.0, abs=5e-3)


def test_depth_floor_does_not_change_outside_zero() -> None:
    """Outside the box, penalty is still 0 regardless of depth_floor.
    The boundary cliff is preserved."""
    term = _term(depth_floor=1.0)
    just_outside_x = FIELD_X_HALF - BOX_DEPTH - 0.005
    ctx = _ctx(rx=just_outside_x, ry=0.0, time_in_box_norm=0.99)
    assert term(ctx) == pytest.approx(0.0, abs=1e-6)


def test_depth_floor_centroid_unchanged() -> None:
    """The peak per-step penalty (centroid / depth-saturated interior)
    is the same at all depth_floor values — calibration invariant."""
    cx = FIELD_X_HALF - 0.5 * BOX_DEPTH
    trigger_norm = TRIGGER_TIME / TERMINAL_TIME
    time_norm = trigger_norm + 0.5 * (1.0 - trigger_norm)
    for floor in (0.0, 0.25, 0.5, 0.75, 1.0):
        term = _term(depth_floor=floor, power=3.0)
        val = term(_ctx(rx=cx, ry=0.0, time_in_box_norm=time_norm))
        # u^p = 0.5^3 = 0.125 at centroid (depth_factor=1 regardless of floor)
        assert val == pytest.approx(0.125, abs=1e-3), (
            f"depth_floor={floor}: centroid penalty changed; got {val}"
        )


def test_depth_floor_pre_trigger_is_silent() -> None:
    """Even with depth_floor=1, no penalty fires before the trigger
    time — entry remains free."""
    term = _term(depth_floor=1.0)
    cx = FIELD_X_HALF - 0.5 * BOX_DEPTH
    ctx = _ctx(rx=cx, ry=0.0, time_in_box_norm=0.5 * TRIGGER_TIME / TERMINAL_TIME)
    assert term(ctx) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Time ramp — silent below trigger, polynomial above
# ---------------------------------------------------------------------------


def test_silent_when_timer_zero() -> None:
    """Robot in the box but timer reads 0 (just entered) ⟹ no penalty."""
    term = _term()
    cx = FIELD_X_HALF - 0.5 * BOX_DEPTH  # box centroid x
    ctx = _ctx(rx=cx, ry=0.0, time_in_box_norm=0.0)
    assert term(ctx) == 0.0


def test_silent_below_trigger() -> None:
    """At t = 1.0 s of 3.0 s budget (trigger at 2.0 s) ⟹ still 0."""
    term = _term()
    cx = FIELD_X_HALF - 0.5 * BOX_DEPTH
    ctx = _ctx(rx=cx, ry=0.0, time_in_box_norm=1.0 / TERMINAL_TIME)
    assert term(ctx) == 0.0


def test_silent_at_trigger_boundary() -> None:
    """At exactly the trigger time, penalty is 0 (u=0 ⟹ u^p=0)."""
    term = _term()
    cx = FIELD_X_HALF - 0.5 * BOX_DEPTH
    ctx = _ctx(rx=cx, ry=0.0, time_in_box_norm=TRIGGER_TIME / TERMINAL_TIME)
    assert term(ctx) == pytest.approx(0.0, abs=1e-6)


def test_ramp_increases_monotonically_in_warning_zone() -> None:
    """In [trigger, terminal], the ramp is strictly increasing in time."""
    term = _term()
    cx = FIELD_X_HALF - 0.5 * BOX_DEPTH  # constant depth, vary time
    times = [
        TRIGGER_TIME + 0.1,
        TRIGGER_TIME + 0.3,
        TRIGGER_TIME + 0.5,
        TRIGGER_TIME + 0.7,
        TRIGGER_TIME + 0.9,
    ]
    values = [term(_ctx(rx=cx, ry=0.0, time_in_box_norm=t / TERMINAL_TIME)) for t in times]
    for a, b in zip(values, values[1:]):
        assert b > a, f"ramp should be monotonically increasing, got {values}"


def test_ramp_polynomial_shape() -> None:
    """At u=0.5 with power=3, ramp factor = 0.125 (= 0.5^3). Combined
    with depth_factor=1 (centroid) and unit weight, the term value is
    0.125 unweighted."""
    term = _term(power=3.0)
    cx = FIELD_X_HALF - 0.5 * BOX_DEPTH  # centroid → depth_factor = 1
    # u = 0.5 ⟹ time_norm = trigger_norm + 0.5 * (1 - trigger_norm)
    trigger_norm = TRIGGER_TIME / TERMINAL_TIME
    time_norm = trigger_norm + 0.5 * (1.0 - trigger_norm)
    ctx = _ctx(rx=cx, ry=0.0, time_in_box_norm=time_norm)
    assert term(ctx) == pytest.approx(0.125, abs=1e-4)


# ---------------------------------------------------------------------------
# Depth weighting
# ---------------------------------------------------------------------------


def test_depth_factor_zero_at_box_inner_edge() -> None:
    """At the field-facing inner x-edge of the box, depth_factor = 0
    even when the timer is at terminal — the policy can sit on the
    edge without paying ramp cost (sparse cost still fires though)."""
    term = _term()
    inner_edge_x = FIELD_X_HALF - BOX_DEPTH
    ctx = _ctx(rx=inner_edge_x, ry=0.0, time_in_box_norm=0.95)
    assert term(ctx) == pytest.approx(0.0, abs=1e-6)


def test_depth_factor_zero_outside_box() -> None:
    term = _term()
    just_outside_x = FIELD_X_HALF - BOX_DEPTH - 0.001
    ctx = _ctx(rx=just_outside_x, ry=0.0, time_in_box_norm=0.95)
    assert term(ctx) == 0.0


def test_depth_factor_zero_above_box_y() -> None:
    """Robot above the box in y ⟹ outside the box ⟹ depth_factor = 0."""
    term = _term()
    inside_x = FIELD_X_HALF - 0.5 * BOX_DEPTH
    ctx = _ctx(rx=inside_x, ry=BOX_Y_HALF + 0.01, time_in_box_norm=0.95)
    assert term(ctx) == 0.0


def test_depth_factor_grows_monotonically_with_intrusion() -> None:
    """Moving from boundary toward the centroid in x increases depth."""
    term = _term()
    inner_x = FIELD_X_HALF - BOX_DEPTH
    intrusions = [0.005, 0.02, 0.04, 0.05, 0.06]
    values = [
        term(_ctx(rx=inner_x + d, ry=0.0, time_in_box_norm=0.95))
        for d in intrusions
    ]
    for a, b in zip(values, values[1:]):
        assert b >= a, f"depth-weighted ramp should grow with intrusion, got {values}"


# ---------------------------------------------------------------------------
# Sparse termination cost
# ---------------------------------------------------------------------------


def test_sparse_penalty_fires_only_when_info_flag_set() -> None:
    """The discrete cost is gated on `info["box_violation_self"]`."""
    term = _term(termination_penalty=1.0)
    cx = FIELD_X_HALF - 0.5 * BOX_DEPTH
    ctx_no = _ctx(rx=cx, ry=0.0, time_in_box_norm=0.5, info={})
    assert term(ctx_no) == 0.0
    ctx_yes = _ctx(
        rx=cx,
        ry=0.0,
        time_in_box_norm=0.5,
        info={"box_violation_self": True},
    )
    # Below trigger so ramp is 0; total = sparse only.
    assert term(ctx_yes) == pytest.approx(1.0, abs=1e-6)


def test_sparse_penalty_independent_of_depth() -> None:
    """Sparse cost fires even at the box edge (depth=0). The discrete
    signal is "you violated the rule," not a spatial value."""
    term = _term(termination_penalty=1.0)
    edge_x = FIELD_X_HALF - BOX_DEPTH  # depth = 0
    ctx = _ctx(
        rx=edge_x,
        ry=0.0,
        time_in_box_norm=1.0,
        info={"box_violation_self": True},
    )
    assert term(ctx) == pytest.approx(1.0, abs=1e-6)


def test_sparse_does_not_fire_for_opponent_violation() -> None:
    """`box_violation_opp=True` (in team play) is the *opponent* having
    violated. The learner's GoalieBoxPenalty must NOT fire from that."""
    term = _term(termination_penalty=1.0)
    cx = FIELD_X_HALF - 0.5 * BOX_DEPTH
    ctx = _ctx(
        rx=cx,
        ry=0.0,
        time_in_box_norm=0.5,
        info={"box_violation_opp": True, "box_violation": True},
    )
    assert term(ctx) == 0.0


# ---------------------------------------------------------------------------
# Composition with weight (sign convention check)
# ---------------------------------------------------------------------------


def test_term_returns_unsigned_magnitude() -> None:
    """Term returns POSITIVE numbers; the composite multiplies by
    `weight` (negative in production) to deliver the penalty."""
    term = _term(weight=-20.0, termination_penalty=1.0)
    cx = FIELD_X_HALF - 0.5 * BOX_DEPTH
    ctx = _ctx(
        rx=cx,
        ry=0.0,
        time_in_box_norm=1.0,
        info={"box_violation_self": True},
    )
    # Returned value (unweighted) is sparse + ramp(u=1)*depth(centroid=1)
    # = 1.0 (sparse) + 1.0 (ramp at u=1) = 2.0. Sign comes from weight.
    raw = term(ctx)
    assert raw == pytest.approx(2.0, abs=1e-4)
    # Weighted via composite logic:
    weighted = term.weight * raw
    assert weighted == pytest.approx(-40.0, abs=1e-4)


def test_total_violation_cost_is_weight_times_two() -> None:
    """Worst case (full ramp + sparse, depth=1 at centroid, u=1) gives
    `2 × weight` in absolute units. With weight=-20 ⟹ total = -40,
    i.e. twice the magnitude of a goal scored."""
    term = _term(weight=-20.0, termination_penalty=1.0)
    cx = FIELD_X_HALF - 0.5 * BOX_DEPTH
    ctx = _ctx(
        rx=cx,
        ry=0.0,
        time_in_box_norm=1.0,
        info={"box_violation_self": True},
    )
    assert term.weight * term(ctx) == pytest.approx(-40.0, abs=1e-4)
