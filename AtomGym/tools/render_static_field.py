"""Render the StaticFieldPenalty as a heatmap for visual inspection.

Usage
-----
    python -m AtomGym.tools.render_static_field --out /tmp/field.png
    python -m AtomGym.tools.render_static_field \
        --safe-dist 0.10 --unavoidable-dist 0.04 \
        --goalie-box-depth 0.08 --goalie-box-y-half 0.12 \
        --both-boxes \
        --out /tmp/field.png

Builds a `StaticFieldPenalty` with the supplied parameters, renders the
precomputed grid as a heatmap, and overlays the field walls and goalie
boxes for reference. Useful for iterating on shaping params before
spending compute on a training run.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from AtomGym.rewards.static_field_penalty import StaticFieldPenalty


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Field geometry — defaults match WorldConfig.
    p.add_argument("--field-x-half", type=float, default=0.375)
    p.add_argument("--field-y-half", type=float, default=0.225)
    p.add_argument(
        "--goal-y-half",
        type=float,
        default=0.06,
        help="Goal mouth half-height (drawn as overlay only).",
    )

    # Goalie box
    p.add_argument("--goalie-box-depth", type=float, default=0.12)
    p.add_argument("--goalie-box-y-half", type=float, default=0.10)
    p.add_argument(
        "--both-boxes",
        action="store_true",
        help="Penalize own goalie box too (default: opposing only).",
    )

    # Sigmoid shaping
    p.add_argument("--safe-dist", type=float, default=0.065)
    p.add_argument("--unavoidable-dist", type=float, default=0.035)

    # Grid + render
    p.add_argument(
        "--grid-resolution",
        type=float,
        default=0.005,
        help="Grid cell size in metres for the rendered heatmap. "
        "Finer = sharper visualisation.",
    )
    p.add_argument("--out", type=str, default="static_field_heatmap.png")
    p.add_argument("--cmap", type=str, default="jet")
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Save without trying to open an interactive window.",
    )
    return p.parse_args()


def _draw_overlay(ax: plt.Axes, args: argparse.Namespace) -> None:
    """Draw field walls, goalie boxes, and goal mouths on top of the
    heatmap so the geometry is self-evident."""
    fxh = args.field_x_half
    fyh = args.field_y_half
    gxh = args.goalie_box_depth
    gyh = args.goalie_box_y_half
    gmy = args.goal_y_half

    # Field perimeter (white).
    ax.add_patch(
        Rectangle(
            (-fxh, -fyh),
            2 * fxh,
            2 * fyh,
            fill=False,
            edgecolor="white",
            linewidth=2.0,
            zorder=3,
        )
    )

    # Halfway line + centre mark.
    ax.plot(
        [0, 0],
        [-fyh, fyh],
        color="white",
        linewidth=1.0,
        linestyle="--",
        alpha=0.6,
        zorder=3,
    )
    ax.plot(0, 0, marker="+", color="white", markersize=8, zorder=3)

    # Goal mouths (cyan dashes — these are gaps in the wall, the ball
    # passes through them).
    for x in (-fxh, +fxh):
        ax.plot([x, x], [-gmy, gmy], color="cyan", linewidth=2.5, alpha=0.9, zorder=4)

    # Goalie boxes — opposing in lime, own in dim grey if also drawn.
    opp_box = Rectangle(
        (fxh - gxh, -gyh),
        gxh,
        2 * gyh,
        fill=False,
        edgecolor="lime",
        linewidth=1.5,
        linestyle="-",
        zorder=3,
        label="opposing goalie box",
    )
    ax.add_patch(opp_box)

    own_color = "lime" if args.both_boxes else "grey"
    own_alpha = 1.0 if args.both_boxes else 0.5
    own_label = (
        "own goalie box (penalized)"
        if args.both_boxes
        else "own goalie box (not penalized)"
    )
    own_box = Rectangle(
        (-fxh, -gyh),
        gxh,
        2 * gyh,
        fill=False,
        edgecolor=own_color,
        linewidth=1.5,
        linestyle="--",
        alpha=own_alpha,
        zorder=3,
        label=own_label,
    )
    ax.add_patch(own_box)


def main() -> None:
    args = _parse_args()

    print(f"Building StaticFieldPenalty:")
    print(f"  field      : {2 * args.field_x_half:.3f} × {2 * args.field_y_half:.3f} m")
    print(
        f"  sigmoid    : 0 at d={args.safe_dist:.3f} m, 1 at d={args.unavoidable_dist:.3f} m"
    )
    print(
        f"  goalie box : {args.goalie_box_depth:.3f} m deep × "
        f"{2 * args.goalie_box_y_half:.3f} m wide ({'both' if args.both_boxes else 'opposing only'})"
    )
    print(f"  grid res   : {args.grid_resolution * 1000:.1f} mm")

    term = StaticFieldPenalty(
        field_x_half=args.field_x_half,
        field_y_half=args.field_y_half,
        goalie_box_depth=args.goalie_box_depth,
        goalie_box_y_half=args.goalie_box_y_half,
        safe_dist=args.safe_dist,
        unavoidable_dist=args.unavoidable_dist,
        grid_resolution=args.grid_resolution,
        penalize_own_box=args.both_boxes,
    )
    print(f"  grid shape : {term.grid.shape} = {term.grid.nbytes / 1024:.1f} KB")

    # imshow expects (rows=y, cols=x); our grid is (nx, ny) so transpose.
    # extent maps array indices to data coords (left, right, bottom, top).
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        term.grid.T,
        origin="lower",
        extent=(
            -term.field_x_half,
            term.field_x_half,
            -term.field_y_half,
            term.field_y_half,
        ),
        cmap=args.cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("penalty", rotation=270, labelpad=12)

    _draw_overlay(ax, args)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"StaticFieldPenalty — sigmoid({args.safe_dist * 1000:.0f}→"
        f"{args.unavoidable_dist * 1000:.0f} mm), "
        f"goalie box {args.goalie_box_depth * 1000:.0f}×"
        f"{2 * args.goalie_box_y_half * 1000:.0f} mm "
        f"({'both' if args.both_boxes else 'opposing'})"
    )
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved heatmap → {args.out}")

    if not args.no_show:
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
