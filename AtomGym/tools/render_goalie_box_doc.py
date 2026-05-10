"""Documentation-quality renderer for the goalie-box rule + reward.

Two output modes:

  --static (default): a single 2×3 multi-panel PNG suitable for embedding
    in a doc:
       Row 1: SDF heatmap | Time ramp at centroid | Joint depth×time
       Row 2: Spatial penalty at three progressive time slices

  --animate: an animated GIF that sweeps `time_in_box` from 0 to terminal,
    showing how the spatial penalty field evolves. Good for demos / social.

Usage
-----
    # Static multi-panel doc figure with the production parameters
    python -m AtomGym.tools.render_goalie_box_doc \\
        --goalie-box-depth 0.13 --goalie-box-y-half 0.11 \\
        --goalie-box-corner-radius 0.07 \\
        --weight 0.5 --termination-penalty 20.0 \\
        --out AtomGym/research/goalie_box/doc_static.png

    # Animated GIF showing time evolution
    python -m AtomGym.tools.render_goalie_box_doc --animate \\
        --goalie-box-depth 0.13 --goalie-box-y-half 0.11 \\
        --goalie-box-corner-radius 0.07 \\
        --out AtomGym/research/goalie_box/doc_animation.gif

Both modes use the SAME SDF the env uses for box-entry termination —
visual and rule are guaranteed consistent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from AtomGym.action_observation import ActionView, ObsView, build_observation
from AtomGym.goalie_box_geometry import signed_depth_into_box
from AtomGym.rewards._base_reward import RewardContext
from AtomGym.rewards.goalie_box_penalty import GoalieBoxPenalty


# ---------------------------------------------------------------------------
# Reward evaluation harness
# ---------------------------------------------------------------------------


def _make_ctx(
    *,
    rx: float,
    ry: float,
    time_in_box_norm: float,
    field_x_half: float,
    field_y_half: float,
    info: dict | None = None,
) -> RewardContext:
    obs = build_observation(
        field_x_half=field_x_half,
        field_y_half=field_y_half,
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
        field_x_half=field_x_half,
        field_y_half=field_y_half,
        goal_y_half=0.06,
        goal_extension=0.06,
        dt=1.0 / 60.0,
    )


def _spatial_grid(
    term: GoalieBoxPenalty,
    fxh: float,
    fyh: float,
    *,
    t_norm: float,
    show_violation: bool = False,
    res: float = 0.002,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Evaluate the (unsigned) penalty over a 2D grid spanning the +x box
    plus a small buffer. Returns (grid, extent) where grid is shape
    (ny, nx) suitable for `imshow(origin='lower')` and extent is the
    matching world-frame (xmin, xmax, ymin, ymax)."""
    x_lo = fxh - term.goalie_box_depth - 0.04
    x_hi = fxh + 0.005
    y_lo = -term.goalie_box_y_half - 0.04
    y_hi = +term.goalie_box_y_half + 0.04
    xs = np.arange(x_lo, x_hi + res, res)
    ys = np.arange(y_lo, y_hi + res, res)
    grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
    info = {"box_violation_self": True} if show_violation else {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            grid[j, i] = term(
                _make_ctx(
                    rx=float(x),
                    ry=float(y),
                    time_in_box_norm=t_norm,
                    field_x_half=fxh,
                    field_y_half=fyh,
                    info=info,
                )
            )
    return grid, (xs[0], xs[-1], ys[0], ys[-1])


# ---------------------------------------------------------------------------
# Box-outline overlay
# ---------------------------------------------------------------------------


def _box_outline_world_pts(
    term: GoalieBoxPenalty, fxh: float
) -> tuple[list[float], list[float]]:
    """Sample points along the field-facing perimeter of the rounded
    box (right side, side=+1). Useful as a matplotlib overlay so the
    heatmap reader can see the rule boundary at a glance."""
    bd = term.goalie_box_depth
    bh = term.goalie_box_y_half
    r = term.goalie_box_corner_radius
    x_edge = fxh - bd
    x_open = fxh

    xs: list[float] = []
    ys: list[float] = []

    # Top horizontal — from goal-line edge to start of top arc.
    xs.append(x_open)
    ys.append(+bh)
    if r > 0:
        cx_top = x_edge + r
        cy_top = bh - r
        # arc from π/2 to π (CCW), i.e. top to left
        for t in np.linspace(np.pi / 2, np.pi, 32):
            xs.append(cx_top + r * np.cos(t))
            ys.append(cy_top + r * np.sin(t))
        cx_bot = x_edge + r
        cy_bot = -bh + r
        for t in np.linspace(np.pi, 3 * np.pi / 2, 32):
            xs.append(cx_bot + r * np.cos(t))
            ys.append(cy_bot + r * np.sin(t))
    else:
        xs.append(x_edge)
        ys.append(+bh)
        xs.append(x_edge)
        ys.append(-bh)
    xs.append(x_open)
    ys.append(-bh)
    return xs, ys


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_sdf(ax: plt.Axes, term: GoalieBoxPenalty, fxh: float, fyh: float) -> None:
    """Heatmap of the signed-distance function. Diverging colormap
    centred at 0 (the box boundary). Contour lines at 0 (boundary)
    and `depth_saturation` (where the depth factor hits 1)."""
    x_lo = fxh - term.goalie_box_depth - 0.04
    x_hi = fxh + 0.005
    y_lo = -term.goalie_box_y_half - 0.04
    y_hi = +term.goalie_box_y_half + 0.04
    res = 0.0015
    xs = np.arange(x_lo, x_hi + res, res)
    ys = np.arange(y_lo, y_hi + res, res)
    sdf = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            sdf[j, i] = signed_depth_into_box(
                float(x),
                float(y),
                field_x_half=fxh,
                goalie_box_depth=term.goalie_box_depth,
                goalie_box_y_half=term.goalie_box_y_half,
                goalie_box_corner_radius=term.goalie_box_corner_radius,
                side=+1,
            )
    vmax = max(abs(sdf.min()), abs(sdf.max()))
    im = ax.imshow(
        sdf,
        origin="lower",
        extent=(xs[0], xs[-1], ys[0], ys[-1]),
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=+vmax,
        aspect="equal",
        interpolation="bilinear",
    )
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("signed depth (m)", rotation=270, labelpad=12)
    # Contour at 0 (boundary) — black; at depth_saturation — green dashed.
    cs0 = ax.contour(xs, ys, sdf, levels=[0.0], colors="black", linewidths=1.5)
    ax.clabel(cs0, fmt="boundary", fontsize=8)
    if term.depth_saturation < sdf.max():
        css = ax.contour(
            xs,
            ys,
            sdf,
            levels=[term.depth_saturation],
            colors="lime",
            linewidths=1.2,
            linestyles="--",
        )
        ax.clabel(css, fmt=f"sat ({term.depth_saturation * 1000:.0f}mm)", fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"SDF (rounded-box geometry, r={term.goalie_box_corner_radius * 1000:.0f}mm)"
    )


def _plot_time_ramp(
    ax: plt.Axes, term: GoalieBoxPenalty, fxh: float, fyh: float
) -> None:
    ts = np.linspace(0.0, 1.0, 200)
    cx = fxh - 0.5 * term.goalie_box_depth
    vals = [
        term(
            _make_ctx(
                rx=cx, ry=0.0, time_in_box_norm=t, field_x_half=fxh, field_y_half=fyh
            )
        )
        for t in ts
    ]
    ax.plot(ts * term.terminal_time, vals, linewidth=2.0, label="ramp (deep)")
    ax.axvline(
        term.trigger_time,
        color="cyan",
        linestyle="--",
        linewidth=1.2,
        label=f"trigger ({term.trigger_time}s)",
    )
    ax.axvline(
        term.terminal_time,
        color="magenta",
        linestyle="--",
        linewidth=1.2,
        label=f"terminal ({term.terminal_time}s)",
    )
    # Mark the discrete sparse cost as a stem at terminal.
    ax.plot(
        [term.terminal_time],
        [term.termination_penalty],
        marker="^",
        markersize=12,
        color="crimson",
        linestyle="None",
        label=f"sparse +{term.termination_penalty:g} on violation",
    )
    ax.set_xlabel("time in opposing box (s)")
    ax.set_ylabel("penalty (unsigned)")
    ax.set_title(f"Time ramp at centroid (power={term.power})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)


def _plot_joint(ax: plt.Axes, term: GoalieBoxPenalty, fxh: float, fyh: float) -> None:
    intrusions = np.linspace(0.0, term.goalie_box_depth, 80)
    ts = np.linspace(0.0, 1.0, 80)
    grid = np.zeros((len(ts), len(intrusions)), dtype=np.float32)
    for j, t in enumerate(ts):
        for i, d in enumerate(intrusions):
            rx = fxh - term.goalie_box_depth + float(d)
            grid[j, i] = term(
                _make_ctx(
                    rx=rx,
                    ry=0.0,
                    time_in_box_norm=float(t),
                    field_x_half=fxh,
                    field_y_half=fyh,
                )
            )
    im = ax.imshow(
        grid,
        origin="lower",
        extent=(intrusions[0] * 1000, intrusions[-1] * 1000, ts[0], ts[-1]),
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="bilinear",
    )
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("penalty (unsigned)", rotation=270, labelpad=12)
    ax.axhline(
        term.trigger_time / term.terminal_time,
        color="cyan",
        linestyle="--",
        linewidth=1.2,
        label="trigger",
    )
    ax.axvline(
        term.depth_saturation * 1000,
        color="lime",
        linestyle=":",
        linewidth=1.2,
        label=f"depth sat ({term.depth_saturation * 1000:.0f}mm)",
    )
    ax.set_xlabel("intrusion at y=0 (mm)")
    ax.set_ylabel("time_in_box / terminal")
    ax.set_title("Joint depth × time")
    ax.legend(loc="upper left", fontsize=8)


def _plot_spatial_at(
    ax: plt.Axes,
    term: GoalieBoxPenalty,
    fxh: float,
    fyh: float,
    t_norm: float,
    *,
    show_violation: bool = False,
    vmax: float | None = None,
) -> None:
    grid, extent = _spatial_grid(
        term,
        fxh,
        fyh,
        t_norm=t_norm,
        show_violation=show_violation,
    )
    if vmax is None:
        vmax = max(1.0, term.termination_penalty + 1.0) if show_violation else 1.0
    im = ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        cmap="jet",
        vmin=0.0,
        vmax=vmax,
        aspect="equal",
        interpolation="bilinear",
    )
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("penalty", rotation=270, labelpad=12)
    # Overlay the rounded box outline so the rule boundary is visible.
    xs, ys = _box_outline_world_pts(term, fxh)
    ax.plot(xs, ys, color="cyan", linewidth=1.4, linestyle="--", alpha=0.85)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    title = f"Spatial @ t={t_norm:.2f}·terminal"
    if show_violation:
        title += " (+ sparse violation)"
    ax.set_title(title)


# ---------------------------------------------------------------------------
# Top-level renderers
# ---------------------------------------------------------------------------


def render_static(args: argparse.Namespace, term: GoalieBoxPenalty) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    _plot_sdf(axes[0, 0], term, args.field_x_half, args.field_y_half)
    _plot_time_ramp(axes[0, 1], term, args.field_x_half, args.field_y_half)
    _plot_joint(axes[0, 2], term, args.field_x_half, args.field_y_half)
    # Three time slices on the bottom row, plus the violation step.
    trigger_norm = term.trigger_time / term.terminal_time
    t_mid = trigger_norm + 0.5 * (1.0 - trigger_norm)
    t_late = trigger_norm + 0.85 * (1.0 - trigger_norm)
    _plot_spatial_at(
        axes[1, 0], term, args.field_x_half, args.field_y_half, t_norm=t_mid
    )
    _plot_spatial_at(
        axes[1, 1], term, args.field_x_half, args.field_y_half, t_norm=t_late
    )
    _plot_spatial_at(
        axes[1, 2],
        term,
        args.field_x_half,
        args.field_y_half,
        t_norm=1.0,
        show_violation=True,
    )
    fig.suptitle(
        f"GoalieBoxPenalty — trigger {term.trigger_time}s, terminal {term.terminal_time}s, "
        f"power {term.power}, term penalty {term.termination_penalty}, "
        f"corner r={term.goalie_box_corner_radius * 1000:.0f}mm, "
        f"depth_floor={term.depth_floor:.2f}",
        fontsize=13,
    )
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved doc figure → {args.out}")


def render_animation(args: argparse.Namespace, term: GoalieBoxPenalty) -> None:
    """Animated GIF: spatial penalty at progressively-larger
    time_in_box, ending with the sparse violation flash."""
    try:
        import imageio.v2 as imageio  # noqa: F401
    except ImportError:
        import imageio  # type: ignore

    n_frames = max(8, args.n_frames)
    timesteps = list(np.linspace(0.0, 1.0, n_frames - 1))
    timesteps.append(1.0)  # last frame = violation step
    show_viol_per_frame = [False] * (n_frames - 1) + [True]

    # Pin a single colour scale across the animation so the dimming /
    # brightening is interpretable. With sparse violation, the last
    # frame can spike well above the ramp range — clamp consistently.
    vmax = max(1.0, term.termination_penalty + 1.0)

    frames: list[np.ndarray] = []
    fig, ax = plt.subplots(figsize=(7, 6))
    cbar = None
    for k, (t, viol) in enumerate(zip(timesteps, show_viol_per_frame)):
        ax.clear()
        grid, extent = _spatial_grid(
            term,
            args.field_x_half,
            args.field_y_half,
            t_norm=float(t),
            show_violation=bool(viol),
        )
        im = ax.imshow(
            grid,
            origin="lower",
            extent=extent,
            cmap="jet",
            vmin=0.0,
            vmax=vmax,
            aspect="equal",
            interpolation="bilinear",
        )
        if cbar is None:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("penalty", rotation=270, labelpad=12)
        xs, ys = _box_outline_world_pts(term, args.field_x_half)
        ax.plot(xs, ys, color="cyan", linewidth=1.4, linestyle="--", alpha=0.85)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        elapsed = float(t) * term.terminal_time
        suffix = " — VIOLATION (sparse fired)" if viol else ""
        ax.set_title(
            f"GoalieBoxPenalty @ t={elapsed:.2f}s ({float(t):.2f}·terminal)"
            f"  |  depth_floor={term.depth_floor:.2f}{suffix}"
        )
        fig.canvas.draw()
        # Use buffer_rgba (modern matplotlib) and drop the alpha channel.
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[..., :3].copy())
    plt.close(fig)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    duration_per_frame = args.duration / max(1, len(frames))
    try:
        imageio.mimsave(args.out, frames, duration=duration_per_frame, loop=0)
    except TypeError:
        # Older imageio: no `loop` kwarg.
        imageio.mimsave(args.out, frames, duration=duration_per_frame)
    print(f"Saved animation ({len(frames)} frames) → {args.out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Mode
    p.add_argument(
        "--animate",
        action="store_true",
        help="Render an animated GIF instead of the static multi-panel figure.",
    )
    p.add_argument(
        "--n-frames",
        type=int,
        default=40,
        help="Number of frames in the animation (default 40).",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Total duration of the animation in seconds (default 4).",
    )
    # Reward params
    p.add_argument("--weight", type=float, default=2.0)
    p.add_argument("--trigger-time", type=float, default=1.0)
    p.add_argument("--terminal-time", type=float, default=3.0)
    p.add_argument("--power", type=float, default=1.0)
    p.add_argument("--termination-penalty", type=float, default=20.0)
    p.add_argument("--depth-saturation", type=float, default=0.06)
    p.add_argument(
        "--depth-floor",
        type=float,
        default=0.0,
        help="Linear-blend floor on the effective depth factor in [0, 1]. "
        "0 = pure depth-graduated potential field (boundary contributes 0). "
        "1 = uniform per-step penalty across the box interior. 0.5 ⟹ "
        "boundary at 50%% of centroid magnitude.",
    )
    # Box geometry (game-rule)
    p.add_argument("--goalie-box-depth", type=float, default=0.13)
    p.add_argument("--goalie-box-y-half", type=float, default=0.11)
    p.add_argument("--goalie-box-corner-radius", type=float, default=0.07)
    # Field
    p.add_argument("--field-x-half", type=float, default=0.375)
    p.add_argument("--field-y-half", type=float, default=0.225)
    # Output
    p.add_argument(
        "--out",
        type=str,
        default="AtomGym/research/goalie_box/doc_static.png",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    term = GoalieBoxPenalty(
        weight=args.weight,
        trigger_time=args.trigger_time,
        terminal_time=args.terminal_time,
        power=args.power,
        termination_penalty=args.termination_penalty,
        goalie_box_depth=args.goalie_box_depth,
        goalie_box_y_half=args.goalie_box_y_half,
        goalie_box_corner_radius=args.goalie_box_corner_radius,
        depth_saturation=args.depth_saturation,
        depth_floor=args.depth_floor,
    )

    print("Building GoalieBoxPenalty:")
    print(f"  weight             : {term.weight}")
    print(f"  trigger / terminal : {term.trigger_time}s / {term.terminal_time}s")
    print(f"  power              : {term.power}")
    print(f"  termination_penalty: {term.termination_penalty}")
    print(
        f"  box (mm)           : {term.goalie_box_depth * 1000:.0f} × "
        f"{term.goalie_box_y_half * 2 * 1000:.0f}, corner r={term.goalie_box_corner_radius * 1000:.0f}"
    )
    print(f"  depth_saturation   : {term.depth_saturation * 1000:.0f}mm")
    print(f"  depth_floor        : {term.depth_floor}")

    if args.animate:
        if not args.out.endswith((".gif", ".GIF")):
            print(
                f"[warn] --animate set but --out={args.out!r} doesn't end in .gif; "
                f"writing GIF data anyway."
            )
        render_animation(args, term)
    else:
        render_static(args, term)


if __name__ == "__main__":
    main()
