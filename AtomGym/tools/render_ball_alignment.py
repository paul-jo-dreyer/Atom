"""Render BallAlignmentReward as heatmaps for visual inspection.

Usage
-----
    python -m AtomGym.tools.render_ball_alignment --out /tmp/align.png
    python -m AtomGym.tools.render_ball_alignment \
        --back-weight 0.3 --inner-radius 0.044 --outer-radius 0.10 \
        --out /tmp/align.png

Builds a `BallAlignmentReward` with the supplied params and renders TWO
complementary views, side by side:

  (1) Spatial — robot fixed at origin facing +x. Heatmap of the reward
      over ball position (bx, by). Overlay shows the robot footprint,
      forward-axis arrow, and the annular gate (inner / outer circles).
      Front/back asymmetry is visually obvious here.

  (2) Polar — distance d on x-axis, signed angle α (=ball direction
      relative to robot forward axis) on y-axis. Heatmap of reward.
      The annular gate is two vertical bands; the alignment kink at
      α = ±π/2 is a clean horizontal zero-line.

Also calls the real `BallAlignmentReward.__call__` (not a re-derivation)
so what you see is what training sees. As a sanity check the script
prints the reward at a handful of canonical poses (front-aligned,
back-aligned, tangential left, tangential right) at the band midpoint.
"""

from __future__ import annotations

import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from AtomGym.action_observation import ActionView, ObsView, build_observation
from AtomGym.rewards._base_reward import RewardContext
from AtomGym.rewards.ball_alignment import BallAlignmentReward


# ---------------------------------------------------------------------------
# Reward evaluation harness
# ---------------------------------------------------------------------------


def _make_ctx(
    *,
    obs: np.ndarray,
    obs_view: ObsView,
    field_x_half: float,
    field_y_half: float,
) -> RewardContext:
    """Build a minimal RewardContext suitable for BallAlignmentReward.

    The term only reads ball / self positions and self heading from `obs`,
    plus `field_x_half` / `field_y_half`. Everything else gets a benign
    default."""
    return RewardContext(
        obs=obs,
        action=np.zeros(2, dtype=np.float32),
        prev_obs=None,
        prev_action=None,
        info={},
        obs_view=obs_view,
        action_view=ActionView(),
        field_x_half=field_x_half,
        field_y_half=field_y_half,
        goal_y_half=0.06,
        goal_extension=0.02,
        dt=0.02,
    )


def _build_obs(
    *,
    robot_xy: tuple[float, float],
    robot_theta: float,
    ball_xy: tuple[float, float],
    field_x_half: float,
    field_y_half: float,
) -> np.ndarray:
    """Build a single-robot observation vector with the given pose."""
    return build_observation(
        field_x_half=field_x_half,
        field_y_half=field_y_half,
        ball_state=np.array(
            [ball_xy[0], ball_xy[1], 0.0, 0.0], dtype=np.float64
        ),
        self_state_5d=np.array(
            [robot_xy[0], robot_xy[1], robot_theta, 0.0, 0.0],
            dtype=np.float64,
        ),
    )


def _eval_reward_grid_xy(
    *,
    term: BallAlignmentReward,
    obs_view: ObsView,
    bx_grid: np.ndarray,
    by_grid: np.ndarray,
    robot_xy: tuple[float, float],
    robot_theta: float,
    field_x_half: float,
    field_y_half: float,
) -> np.ndarray:
    """Evaluate the term over a 2D meshgrid of ball positions.

    Returns array of shape (len(by_grid), len(bx_grid)) — rows index by,
    columns index bx. (matplotlib `imshow` row=y / col=x convention.)"""
    nx, ny = len(bx_grid), len(by_grid)
    out = np.zeros((ny, nx), dtype=np.float32)
    for j, by in enumerate(by_grid):
        for i, bx in enumerate(bx_grid):
            obs = _build_obs(
                robot_xy=robot_xy,
                robot_theta=robot_theta,
                ball_xy=(float(bx), float(by)),
                field_x_half=field_x_half,
                field_y_half=field_y_half,
            )
            ctx = _make_ctx(
                obs=obs,
                obs_view=obs_view,
                field_x_half=field_x_half,
                field_y_half=field_y_half,
            )
            out[j, i] = term(ctx) * term.weight
    return out


def _eval_reward_grid_polar(
    *,
    term: BallAlignmentReward,
    obs_view: ObsView,
    d_grid: np.ndarray,
    alpha_grid: np.ndarray,
    robot_theta: float,
    field_x_half: float,
    field_y_half: float,
) -> np.ndarray:
    """Evaluate the term over (distance, signed-angle) grid.

    α is the angle from the robot's forward axis to the ball direction
    (positive = CCW = ball to the robot's left). The ball is placed at
    (d cos(θ+α), d sin(θ+α)) so the observed bearing in body frame is α.

    Returns array of shape (len(alpha_grid), len(d_grid))."""
    nd, na = len(d_grid), len(alpha_grid)
    out = np.zeros((na, nd), dtype=np.float32)
    for j, alpha in enumerate(alpha_grid):
        bearing = robot_theta + alpha
        cos_b = float(np.cos(bearing))
        sin_b = float(np.sin(bearing))
        for i, d in enumerate(d_grid):
            bx = float(d) * cos_b
            by = float(d) * sin_b
            obs = _build_obs(
                robot_xy=(0.0, 0.0),
                robot_theta=robot_theta,
                ball_xy=(bx, by),
                field_x_half=field_x_half,
                field_y_half=field_y_half,
            )
            ctx = _make_ctx(
                obs=obs,
                obs_view=obs_view,
                field_x_half=field_x_half,
                field_y_half=field_y_half,
            )
            out[j, i] = term(ctx) * term.weight
    return out


# ---------------------------------------------------------------------------
# Sanity-check probe — confirms forward axis lines up with cos/sin θ
# ---------------------------------------------------------------------------


def _probe_canonical_poses(
    term: BallAlignmentReward,
    obs_view: ObsView,
    field_x_half: float,
    field_y_half: float,
) -> None:
    """Probe known-answer poses. Robot at origin with several headings;
    ball placed at the band midpoint in body-frame canonical directions.

    Forward (α=0)        → expect `weight * 1.0`
    Back   (α=π)         → expect `weight * back_weight`
    Tangent left  (α=+π/2) → expect 0
    Tangent right (α=-π/2) → expect 0

    If the heading axis were measured from anything other than the
    forward axis (e.g. from +x globally, ignoring θ), these probes
    would give wrong answers when the robot is rotated to e.g. θ=π/2."""
    midpoint = 0.5 * (term.inner_radius + term.outer_radius)
    print(f"\nSanity probe — ball at body-frame distance {midpoint*1000:.1f} mm:")
    print(f"  weight={term.weight}, back_weight={term.back_weight}")
    print(f"  expected: front≈{term.weight*1.0:.3f}  back≈{term.weight*term.back_weight:.3f}  "
          f"tangent-L≈0.000  tangent-R≈0.000")
    print()
    print(f"  {'robot θ':>10}  {'front α=0':>10}  {'back α=π':>10}  "
          f"{'tan +π/2':>10}  {'tan −π/2':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    headings = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, -np.pi / 2]
    for theta in headings:
        row = []
        for alpha in (0.0, np.pi, np.pi / 2, -np.pi / 2):
            bearing = theta + alpha
            bx = midpoint * np.cos(bearing)
            by = midpoint * np.sin(bearing)
            obs = _build_obs(
                robot_xy=(0.0, 0.0),
                robot_theta=theta,
                ball_xy=(float(bx), float(by)),
                field_x_half=field_x_half,
                field_y_half=field_y_half,
            )
            ctx = _make_ctx(
                obs=obs,
                obs_view=obs_view,
                field_x_half=field_x_half,
                field_y_half=field_y_half,
            )
            row.append(term(ctx) * term.weight)
        print(f"  {theta:>10.3f}  {row[0]:>10.3f}  {row[1]:>10.3f}  "
              f"{row[2]:>10.3f}  {row[3]:>10.3f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _draw_robot_overlay(
    ax: plt.Axes,
    *,
    robot_xy: tuple[float, float],
    robot_theta: float,
    chassis_half: float,
    inner_radius: float,
    outer_radius: float,
) -> None:
    """Draw the robot footprint, forward arrow, and annular gate circles."""
    rx, ry = robot_xy

    # Robot footprint — square aligned with body axes. Drawn unrotated
    # (the robot heading θ=0 in this view), so we just draw a centred
    # square.
    cos_t, sin_t = np.cos(robot_theta), np.sin(robot_theta)
    # Build the four corners in body frame, rotate, translate.
    half = chassis_half
    corners_body = np.array([
        [+half, +half],
        [+half, -half],
        [-half, -half],
        [-half, +half],
    ])
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    corners_world = corners_body @ R.T + np.array([rx, ry])
    ax.fill(
        corners_world[:, 0],
        corners_world[:, 1],
        color="white",
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
        zorder=4,
    )

    # Forward axis — arrow from centre, length = chassis_half * 1.6.
    arrow_len = chassis_half * 1.6
    ax.annotate(
        "",
        xy=(rx + arrow_len * cos_t, ry + arrow_len * sin_t),
        xytext=(rx, ry),
        arrowprops=dict(arrowstyle="->", color="black", linewidth=2.0),
        zorder=5,
    )

    # Annular gate — two dashed circles.
    ax.add_patch(
        Circle(
            (rx, ry),
            inner_radius,
            fill=False,
            edgecolor="cyan",
            linestyle="--",
            linewidth=1.3,
            alpha=0.9,
            zorder=3,
        )
    )
    ax.add_patch(
        Circle(
            (rx, ry),
            outer_radius,
            fill=False,
            edgecolor="cyan",
            linestyle="--",
            linewidth=1.3,
            alpha=0.9,
            zorder=3,
        )
    )

    # Tangential dashes — perpendicular to forward, full annular extent.
    # Helps eye-balling the kink line where alignment crosses 0.
    perp = np.array([-sin_t, cos_t])  # left-perpendicular
    for sign in (+1, -1):
        a = np.array([rx, ry]) + sign * perp * inner_radius
        b = np.array([rx, ry]) + sign * perp * outer_radius
        ax.plot(
            [a[0], b[0]], [a[1], b[1]],
            color="magenta",
            linewidth=1.0,
            linestyle=":",
            alpha=0.8,
            zorder=3,
        )


def _plot_spatial(
    ax: plt.Axes,
    *,
    term: BallAlignmentReward,
    obs_view: ObsView,
    field_x_half: float,
    field_y_half: float,
    chassis_half: float,
    cmap: str,
) -> None:
    # Sample window — generous enough to clearly see beyond the outer
    # radius, but not so big that the central detail gets lost.
    half_extent = term.outer_radius * 1.6
    res = 0.0015  # ~1.5 mm cells; total ~210 × 210 cells
    bx_grid = np.arange(-half_extent, half_extent + res, res)
    by_grid = np.arange(-half_extent, half_extent + res, res)
    grid = _eval_reward_grid_xy(
        term=term,
        obs_view=obs_view,
        bx_grid=bx_grid,
        by_grid=by_grid,
        robot_xy=(0.0, 0.0),
        robot_theta=0.0,  # facing +x
        field_x_half=field_x_half,
        field_y_half=field_y_half,
    )
    im = ax.imshow(
        grid,
        origin="lower",
        extent=(bx_grid[0], bx_grid[-1], by_grid[0], by_grid[-1]),
        cmap=cmap,
        vmin=0.0,
        vmax=term.weight,
        aspect="equal",
        interpolation="nearest",
    )
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("reward", rotation=270, labelpad=12)

    _draw_robot_overlay(
        ax,
        robot_xy=(0.0, 0.0),
        robot_theta=0.0,
        chassis_half=chassis_half,
        inner_radius=term.inner_radius,
        outer_radius=term.outer_radius,
    )

    ax.set_xlabel("ball x (m, robot frame)")
    ax.set_ylabel("ball y (m, robot frame)")
    ax.set_title(
        "Spatial — robot at origin facing →\n"
        "front-aligned region is bright, back-aligned region is dimmer\n"
        "magenta dotted line = tangential (reward = 0)"
    )


def _plot_polar(
    ax: plt.Axes,
    *,
    term: BallAlignmentReward,
    obs_view: ObsView,
    field_x_half: float,
    field_y_half: float,
    cmap: str,
) -> None:
    # Distance: from 0 to a bit past outer_radius; angle: full circle in
    # signed convention (-π, π).
    d_max = term.outer_radius * 1.4
    d_grid = np.linspace(0.0, d_max, 200)
    alpha_grid = np.linspace(-np.pi, np.pi, 360)
    grid = _eval_reward_grid_polar(
        term=term,
        obs_view=obs_view,
        d_grid=d_grid,
        alpha_grid=alpha_grid,
        robot_theta=0.0,
        field_x_half=field_x_half,
        field_y_half=field_y_half,
    )
    im = ax.imshow(
        grid,
        origin="lower",
        extent=(d_grid[0], d_grid[-1], np.degrees(alpha_grid[0]),
                np.degrees(alpha_grid[-1])),
        cmap=cmap,
        vmin=0.0,
        vmax=term.weight,
        aspect="auto",
        interpolation="nearest",
    )
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("reward", rotation=270, labelpad=12)

    # Mark the gate boundaries.
    for d in (term.inner_radius, term.outer_radius):
        ax.axvline(d, color="cyan", linestyle="--", linewidth=1.2, alpha=0.9)
    band_mid = 0.5 * (term.inner_radius + term.outer_radius)
    ax.axvline(band_mid, color="white", linestyle=":", linewidth=0.9, alpha=0.6)

    # Mark the tangential angles (where alignment = 0).
    for a in (-90, 90):
        ax.axhline(a, color="magenta", linestyle=":", linewidth=1.0, alpha=0.85)

    ax.set_xlabel("distance to ball (m)")
    ax.set_ylabel("body-frame angle α to ball (deg)")
    ax.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.set_title(
        "Polar — d on x, α on y (α = 0 ⟹ ball straight ahead)\n"
        "cyan dashed = gate inner/outer; magenta dotted = tangential α=±90°\n"
        "back band (|α|>90°) is dimmer by `back_weight`"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weight", type=float, default=1.0)
    p.add_argument("--inner-radius", type=float, default=0.0)
    p.add_argument("--outer-radius", type=float, default=0.18)
    p.add_argument("--back-weight", type=float, default=0.3)
    p.add_argument(
        "--chassis-half",
        type=float,
        default=0.030,
        help="Robot chassis half-side (drawn as overlay only).",
    )
    p.add_argument("--field-x-half", type=float, default=0.375)
    p.add_argument("--field-y-half", type=float, default=0.225)
    p.add_argument(
        "--out",
        type=str,
        default="AtomGym/research/ball_alignment/ball_alignment_heatmap.png",
    )
    p.add_argument("--cmap", type=str, default="viridis")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    term = BallAlignmentReward(
        weight=args.weight,
        inner_radius=args.inner_radius,
        outer_radius=args.outer_radius,
        back_weight=args.back_weight,
    )
    obs_view = ObsView(n_robots=1)

    print("Building BallAlignmentReward:")
    print(f"  weight        : {term.weight}")
    print(f"  inner_radius  : {term.inner_radius*1000:.1f} mm")
    print(f"  outer_radius  : {term.outer_radius*1000:.1f} mm")
    print(f"  band midpoint : {0.5*(term.inner_radius+term.outer_radius)*1000:.1f} mm")
    print(f"  back_weight   : {term.back_weight}")

    _probe_canonical_poses(
        term,
        obs_view,
        field_x_half=args.field_x_half,
        field_y_half=args.field_y_half,
    )

    fig, (ax_xy, ax_pol) = plt.subplots(1, 2, figsize=(15, 6))
    _plot_spatial(
        ax_xy,
        term=term,
        obs_view=obs_view,
        field_x_half=args.field_x_half,
        field_y_half=args.field_y_half,
        chassis_half=args.chassis_half,
        cmap=args.cmap,
    )
    _plot_polar(
        ax_pol,
        term=term,
        obs_view=obs_view,
        field_x_half=args.field_x_half,
        field_y_half=args.field_y_half,
        cmap=args.cmap,
    )
    fig.suptitle(
        f"BallAlignmentReward — back_weight={term.back_weight}, "
        f"gate {term.inner_radius*1000:.0f}–{term.outer_radius*1000:.0f} mm, "
        f"weight={term.weight}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved heatmap → {args.out}")

    if not args.no_show:
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
