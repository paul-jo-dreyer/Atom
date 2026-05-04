"""Tests for StaticFieldPenalty — sigmoid math, grid build correctness,
bilinear interpolation, source overlay."""

from __future__ import annotations

import math

import numpy as np
import pytest

from AtomGym.action_observation import ActionView, ObsView
from AtomGym.rewards import RewardContext
from AtomGym.rewards.static_field_penalty import StaticFieldPenalty


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_safe_le_unavoidable() -> None:
    with pytest.raises(ValueError, match="unavoidable_dist"):
        StaticFieldPenalty(safe_dist=0.030, unavoidable_dist=0.030)
    with pytest.raises(ValueError, match="unavoidable_dist"):
        StaticFieldPenalty(safe_dist=0.020, unavoidable_dist=0.030)


def test_constructor_rejects_zero_unavoidable() -> None:
    with pytest.raises(ValueError):
        StaticFieldPenalty(unavoidable_dist=0.0)


def test_constructor_rejects_zero_grid_resolution() -> None:
    with pytest.raises(ValueError, match="grid_resolution"):
        StaticFieldPenalty(grid_resolution=0.0)


def test_constructor_rejects_zero_field_dims() -> None:
    with pytest.raises(ValueError, match="field"):
        StaticFieldPenalty(field_x_half=0.0)


def test_constructor_rejects_oversized_goalie_box() -> None:
    with pytest.raises(ValueError, match="goalie_box_depth"):
        StaticFieldPenalty(field_x_half=0.05, goalie_box_depth=0.20)


# ---------------------------------------------------------------------------
# Sigmoid behaviour
# ---------------------------------------------------------------------------


def _make_term(**kwargs):
    """Default term with coarse grid; per-test overrides via kwargs."""
    base: dict = dict(grid_resolution=0.01)
    base.update(kwargs)
    return StaticFieldPenalty(**base)


def test_sigmoid_clamps_below_unavoidable() -> None:
    term = _make_term(unavoidable_dist=0.03, safe_dist=0.085)
    assert term._sigmoid(0.0) == 1.0
    assert term._sigmoid(-0.10) == 1.0  # negative ⟹ inside hazard
    assert term._sigmoid(0.029) == 1.0  # just below threshold


def test_sigmoid_clamps_above_safe() -> None:
    term = _make_term(unavoidable_dist=0.03, safe_dist=0.085)
    assert term._sigmoid(0.085) == 0.0
    assert term._sigmoid(0.5) == 0.0


def test_sigmoid_monotone_decreasing_in_band() -> None:
    """As distance from hazard grows, penalty should decrease monotonically."""
    term = _make_term(unavoidable_dist=0.03, safe_dist=0.085)
    ds = np.linspace(0.031, 0.084, 30)
    vals = [term._sigmoid(float(d)) for d in ds]
    for a, b in zip(vals[:-1], vals[1:]):
        assert b <= a + 1e-9


def test_sigmoid_midpoint_value_near_half() -> None:
    """At the midpoint of the band, the logistic should be ~0.5."""
    term = _make_term(unavoidable_dist=0.03, safe_dist=0.085)
    midpoint = 0.5 * (0.030 + 0.085)
    assert abs(term._sigmoid(midpoint) - 0.5) < 1e-6


def test_sigmoid_endpoints_close_to_target() -> None:
    """ln(99) calibration ⟹ ~0.99 at unavoidable, ~0.01 at safe."""
    term = _make_term(unavoidable_dist=0.03, safe_dist=0.085)
    # Hard clamps trigger AT the endpoints, so probe just inside.
    eps = 1e-6
    assert abs(term._sigmoid(0.030 + eps) - 0.99) < 1e-3
    assert abs(term._sigmoid(0.085 - eps) - 0.01) < 1e-3


# ---------------------------------------------------------------------------
# Intrusion sigmoid — goalie box shaping
# ---------------------------------------------------------------------------


def test_intrusion_sigmoid_zero_at_or_outside_boundary() -> None:
    """At the box boundary (intrusion=0) and outside (negative
    intrusion), penalty must be exactly 0 — corners of the field stay
    reachable."""
    term = _make_term(goalie_box_full_depth=0.06)
    assert term._intrusion_sigmoid(0.0) == 0.0
    assert term._intrusion_sigmoid(-0.05) == 0.0
    assert term._intrusion_sigmoid(-1.0) == 0.0


def test_intrusion_sigmoid_one_at_full_depth_or_deeper() -> None:
    term = _make_term(goalie_box_full_depth=0.06)
    assert term._intrusion_sigmoid(0.06) == 1.0
    assert term._intrusion_sigmoid(0.10) == 1.0


def test_intrusion_sigmoid_midpoint_is_half() -> None:
    term = _make_term(goalie_box_full_depth=0.06)
    assert abs(term._intrusion_sigmoid(0.03) - 0.5) < 1e-6


def test_intrusion_sigmoid_monotone_increasing() -> None:
    term = _make_term(goalie_box_full_depth=0.06)
    ds = np.linspace(1e-6, 0.06 - 1e-6, 30)
    vals = [term._intrusion_sigmoid(float(d)) for d in ds]
    for a, b in zip(vals[:-1], vals[1:]):
        assert b >= a - 1e-9


# ---------------------------------------------------------------------------
# Grid build — known values
# ---------------------------------------------------------------------------


def test_grid_centre_is_zero() -> None:
    """Field centre (0, 0) is far from every wall and from both goalie
    boxes, so penalty should be 0."""
    term = _make_term(grid_resolution=0.01)
    assert term.lookup(0.0, 0.0) == 0.0


def test_grid_corner_is_one() -> None:
    """Field corner is near two walls; penalty should saturate to 1."""
    term = _make_term(grid_resolution=0.01)
    assert term.lookup(term.field_x_half, term.field_y_half) == 1.0


def test_grid_deep_inside_opposing_box_is_one() -> None:
    """Past `goalie_box_full_depth` of intrusion, the box penalty
    saturates to 1. We use a deep box and a small full-depth so the
    saturation zone is reachable without hitting the wall sigmoid.
    Requires `include_goalie_box=True` (the default disables the
    spatial box source — `GoalieBoxPenalty` handles it instead)."""
    term = StaticFieldPenalty(
        field_x_half=0.5,
        field_y_half=0.3,
        goalie_box_depth=0.20,
        goalie_box_y_half=0.10,
        goalie_box_full_depth=0.05,  # saturates 50mm into the box
        safe_dist=0.085,
        unavoidable_dist=0.030,
        grid_resolution=0.005,
        include_goalie_box=True,
    )
    # Box on +x: x in [0.30, 0.50]. Pick x = 0.36: 60mm intrusion ⟹
    # well past the 50mm saturation threshold. Wall distance 140mm ⟹
    # wall penalty = 0.
    assert term.lookup(0.36, 0.0) == pytest.approx(1.0, abs=1e-6)


def test_box_boundary_alone_has_zero_penalty() -> None:
    """At the goalie box boundary (and far from any wall), the box
    penalty is exactly 0 — the policy can scrape along the perimeter
    and round the corner without paying for it."""
    term = StaticFieldPenalty(
        field_x_half=0.5,
        field_y_half=0.3,
        goalie_box_depth=0.20,
        goalie_box_y_half=0.10,
        goalie_box_full_depth=0.05,
        safe_dist=0.085,
        unavoidable_dist=0.030,
        grid_resolution=0.005,
        include_goalie_box=True,
    )
    # Just outside the inner edge of the +x box (x=0.30 is the edge;
    # x=0.295 is 5mm outside). Wall distance >> safe_dist ⟹ wall
    # contribution is 0; SDF > 0 ⟹ intrusion clamp is 0. Total 0.
    assert term.lookup(0.295, 0.0) == pytest.approx(0.0, abs=1e-6)


def test_grid_field_corner_outside_box_is_zero() -> None:
    """Field corner (just outside the goalie box but in the field
    corner) should be reachable when far enough from the walls — the
    new shaping doesn't lock it out as a hazard zone."""
    # Use a small box so the corner is well clear. With box_y_half=0.05
    # and field_y_half=0.225, the corner at (x_half - small, y_half - small)
    # sits outside the box in y but near +x wall. Push x inward enough
    # to clear the wall sigmoid.
    term = StaticFieldPenalty(
        field_x_half=0.375,
        field_y_half=0.225,
        goalie_box_depth=0.07,
        goalie_box_y_half=0.05,
        goalie_box_full_depth=0.06,
        safe_dist=0.085,
        unavoidable_dist=0.030,
        grid_resolution=0.005,
    )
    # x=0.27: 105mm from +x wall, well past safe_dist; outside the box
    # in x (box starts at x=0.305). y=0.15: 75mm from top wall (within
    # band, so wall penalty > 0) — we still want it strictly < 1.
    val = term.lookup(0.27, 0.15)
    assert val < 1.0
    # x=0.27, y=0.10: 125mm from top wall, well clear; box-clear too.
    assert term.lookup(0.27, 0.10) == pytest.approx(0.0, abs=1e-6)


def test_grid_far_from_opposing_box_is_zero_when_far_from_walls() -> None:
    """Mirror-image point on the -x side should be safe — there is no
    own goalie box by default, and the point is far from any wall."""
    term = _make_term(
        grid_resolution=0.005,
        penalize_own_box=False,
    )
    # Far from -x wall AND far from any other wall AND no own-box penalty.
    x = -term.field_x_half + 0.15  # 150mm in from the left wall
    y = 0.0
    assert term.lookup(x, y) == 0.0


def test_own_box_penalty_when_enabled() -> None:
    """penalize_own_box=True ⟹ -x box also penalises intrusion. Also
    needs `include_goalie_box=True` to enable the spatial source at
    all (default off)."""
    term = StaticFieldPenalty(
        field_x_half=0.5,
        field_y_half=0.3,
        goalie_box_depth=0.20,
        goalie_box_y_half=0.10,
        goalie_box_full_depth=0.05,
        safe_dist=0.085,
        unavoidable_dist=0.030,
        grid_resolution=0.005,
        penalize_own_box=True,
        include_goalie_box=True,
    )
    # Mirror of the deep-intrusion test on the -x side.
    assert term.lookup(-0.36, 0.0) == pytest.approx(1.0, abs=1e-6)


def test_own_box_disabled_default() -> None:
    """penalize_own_box=False (default) ⟹ -x box has no contribution.
    A point deep inside the -x box but well clear of the -x wall should
    return 0."""
    term = StaticFieldPenalty(
        field_x_half=0.5,
        field_y_half=0.3,
        goalie_box_depth=0.20,
        goalie_box_y_half=0.10,
        goalie_box_full_depth=0.05,
        safe_dist=0.085,
        unavoidable_dist=0.030,
        grid_resolution=0.005,
        penalize_own_box=False,
    )
    assert term.lookup(-0.36, 0.0) == 0.0


def test_max_overlay_no_double_count() -> None:
    """Where wall and goalie box both saturate, max keeps it at 1
    (no additive overshoot)."""
    term = _make_term(grid_resolution=0.005)
    x = term.field_x_half  # on opposing wall AND inside opposing box
    y = 0.0
    assert term.lookup(x, y) == 1.0  # not 2.0


def test_grid_shape_matches_resolution() -> None:
    """Grid size should be 2*half/dx + 1 in each axis."""
    term = StaticFieldPenalty(
        field_x_half=0.30,
        field_y_half=0.20,
        grid_resolution=0.01,
    )
    expected_nx = int(round(2 * 0.30 / 0.01)) + 1
    expected_ny = int(round(2 * 0.20 / 0.01)) + 1
    assert term.grid.shape == (expected_nx, expected_ny)


# ---------------------------------------------------------------------------
# Bilinear interpolation
# ---------------------------------------------------------------------------


def test_bilerp_at_cell_centres_matches_grid() -> None:
    """Querying exactly at a grid cell's world coord must return the
    grid value (no interpolation needed)."""
    term = _make_term(grid_resolution=0.01)
    # Pick a few cells across the grid.
    for (i, j) in [(0, 0), (10, 5), (20, 15), (term.grid.shape[0] - 1, term.grid.shape[1] - 1)]:
        x = -term.field_x_half + i * term._dx
        y = -term.field_y_half + j * term._dx
        assert term.lookup(x, y) == pytest.approx(term.grid[i, j], abs=1e-7)


def test_bilerp_midpoint_is_average_of_neighbours() -> None:
    """At the centre of a cell quadrant, bilerp returns the simple
    average of the 4 surrounding cell values. Pins the bilinear
    weighting math."""
    term = _make_term(grid_resolution=0.01)
    # Pick a known interior cell pair.
    i, j = 30, 15
    x = -term.field_x_half + (i + 0.5) * term._dx
    y = -term.field_y_half + (j + 0.5) * term._dx
    expected = 0.25 * (
        term.grid[i, j] + term.grid[i + 1, j]
        + term.grid[i, j + 1] + term.grid[i + 1, j + 1]
    )
    assert term.lookup(x, y) == pytest.approx(expected, abs=1e-6)


def test_lookup_clamps_out_of_grid() -> None:
    """A query outside the grid bounds returns the boundary value
    rather than crashing or extrapolating."""
    term = _make_term(grid_resolution=0.01)
    # Far past the +x wall — clamp to the grid corner value.
    edge_value = term.lookup(term.field_x_half, 0.0)
    out_value = term.lookup(term.field_x_half + 1.0, 0.0)
    assert out_value == edge_value


# ---------------------------------------------------------------------------
# RewardTerm protocol — exercised via RewardContext
# ---------------------------------------------------------------------------


def _make_ctx(self_px_norm: float, self_py_norm: float, n_robots: int = 1) -> RewardContext:
    """Build a minimal RewardContext with the learner placed at
    (self_px_norm * field_x_half, self_py_norm * field_y_half)."""
    obs_view = ObsView(n_robots=n_robots)
    obs = np.zeros(obs_view.total_dim, dtype=np.float32)
    obs[4 + 0] = self_px_norm  # self_px slot
    obs[4 + 1] = self_py_norm  # self_py slot
    return RewardContext(
        obs=obs,
        action=np.zeros(2, dtype=np.float32),
        prev_obs=None,
        prev_action=None,
        info={},
        obs_view=obs_view,
        action_view=ActionView(),
        field_x_half=0.375,
        field_y_half=0.225,
        goal_y_half=0.06,
        goal_extension=0.06,
        dt=1.0 / 60.0,
    )


def test_call_at_centre_returns_zero() -> None:
    term = _make_term(grid_resolution=0.005)
    ctx = _make_ctx(0.0, 0.0)
    assert term(ctx) == 0.0


def test_call_at_corner_returns_one() -> None:
    term = _make_term(grid_resolution=0.005)
    # Normalised obs: ±1 = ±field_half_extent.
    ctx = _make_ctx(1.0, 1.0)
    assert term(ctx) == pytest.approx(1.0, abs=1e-6)


def test_call_works_with_team_obs_layout() -> None:
    """In TeamEnv (n_robots=2), the self block still starts at index 4
    — the term should read the same slot regardless of how many
    'other' robots follow."""
    term = _make_term(grid_resolution=0.005)
    ctx = _make_ctx(0.0, 0.0, n_robots=2)
    assert term(ctx) == 0.0
    ctx = _make_ctx(1.0, 1.0, n_robots=2)
    assert term(ctx) == pytest.approx(1.0, abs=1e-6)


def test_call_respects_weight() -> None:
    """The base class multiplies term value by weight at composite-eval
    time; the term itself returns the unweighted value."""
    term = _make_term(weight=2.5, grid_resolution=0.005)
    ctx = _make_ctx(1.0, 1.0)
    raw = term(ctx)
    weighted = term.weight * raw
    assert raw == pytest.approx(1.0, abs=1e-6)
    assert weighted == pytest.approx(2.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Sigmoid endpoint sanity outside the grid path
# ---------------------------------------------------------------------------


def test_sigmoid_endpoints_consistent_across_param_choices() -> None:
    """Re-tuning the band should still give ~99% / ~1% at the endpoints."""
    for (u, s) in [(0.020, 0.060), (0.040, 0.120), (0.005, 0.040)]:
        term = _make_term(unavoidable_dist=u, safe_dist=s, grid_resolution=0.01)
        eps = 1e-7
        assert abs(term._sigmoid(u + eps) - 0.99) < 1e-3
        assert abs(term._sigmoid(s - eps) - 0.01) < 1e-3
