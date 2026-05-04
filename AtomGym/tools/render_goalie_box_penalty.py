"""Render GoalieBoxPenalty as heatmaps for visual inspection.

Usage
-----
    python -m AtomGym.tools.render_goalie_box_penalty \\
        --out AtomGym/research/goalie_box/baseline.png

Builds a `GoalieBoxPenalty` and renders THREE complementary views, side
by side:

  (1) Time ramp — 1D plot, penalty at the centroid of the box vs.
      time-in-box. Shows the trigger / terminal boundaries and the
      polynomial shape.

  (2) Spatial heatmap at fixed timer near terminal — the depth-weighted
      penalty over (x, y) inside (and around) the +x goalie box. The
      potential-field shape pointing OUT of the box is the key signal
      the policy's gradient picks up.

  (3) Joint heatmap — the penalty as a function of (depth_into_box,
      time_in_box) for a fixed y=0 line. Shows trigger / terminal /
      depth-saturation boundaries together. The corner near (deep,
      terminal) is where the worst per-step cost lives.

Sanity probe printed to stdout: penalty at four canonical poses
(boundary / centroid / deep / outside) at fixed t=0.95×terminal.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from AtomGym.action_observation import ActionView, ObsView, build_observation
from AtomGym.rewards._base_reward import RewardContext
from AtomGym.rewards.goalie_box_penalty import GoalieBoxPenalty


def _make_ctx(
    *,
    rx: float,
    ry: float,
    time_in_box_norm: float,
    field_x_half: float,
    field_y_half: float,
    info: dict | None = None,
) -> RewardContext:
    obs_view = ObsView(n_robots=1)
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
        obs_view=obs_view,
        action_view=ActionView(),
        field_x_half=field_x_half,
        field_y_half=field_y_half,
        goal_y_half=0.06,
        goal_extension=0.06,
        dt=1.0 / 60.0,
    )


def _eval(
    term: GoalieBoxPenalty, **ctx_kwargs
) -> float:
    """Returned value is unsigned magnitude (term returns >= 0)."""
    return term(_make_ctx(**ctx_kwargs)) * term.weight * (-1.0)
    # Negate because production weight is negative (penalty); rendered
    # heatmap shows positive penalty magnitude for visual clarity.


def _probe(
    term: GoalieBoxPenalty,
    field_x_half: float,
    field_y_half: float,
) -> None:
    print("\nSanity probe — penalty at canonical poses (positive = penalty magnitude):")
    print(f"  weight={term.weight}, terminal={term.terminal_time}s, "
          f"trigger={term.trigger_time}s, power={term.power}")

    near_terminal = 0.95
    poses = [
        ("at inner edge (just inside)",
         field_x_half - term.goalie_box_depth + 0.001, 0.0),
        ("centroid (cx, 0)",
         field_x_half - 0.5 * term.goalie_box_depth, 0.0),
        ("deep inside (1 robot from edge)",
         field_x_half - term.goalie_box_depth + term.depth_saturation, 0.0),
        ("just outside box",
         field_x_half - term.goalie_box_depth - 0.005, 0.0),
    ]
    print(f"  at time_in_box_norm = {near_terminal}:")
    for name, rx, ry in poses:
        v = term(_make_ctx(
            rx=rx, ry=ry, time_in_box_norm=near_terminal,
            field_x_half=field_x_half, field_y_half=field_y_half,
        ))
        print(f"    {name:42s} → {v:.4f}")
    # And at the terminal (with violation flag)
    print(f"  at violation (time=1.0, info[box_violation_self]=True), centroid:")
    v_term = term(_make_ctx(
        rx=field_x_half - 0.5 * term.goalie_box_depth, ry=0.0,
        time_in_box_norm=1.0, field_x_half=field_x_half, field_y_half=field_y_half,
        info={"box_violation_self": True},
    ))
    print(f"    centroid + sparse → {v_term:.4f}  (≈ 1.0 ramp + {term.termination_penalty} sparse)")


def _plot_time_ramp(ax: plt.Axes, term: GoalieBoxPenalty, fxh: float, fyh: float) -> None:
    ts = np.linspace(0.0, 1.0, 200)
    cx = fxh - 0.5 * term.goalie_box_depth
    vals = [
        term(_make_ctx(rx=cx, ry=0.0, time_in_box_norm=t, field_x_half=fxh, field_y_half=fyh))
        for t in ts
    ]
    ax.plot(ts * term.terminal_time, vals, linewidth=2.0)
    ax.axvline(term.trigger_time, color="cyan", linestyle="--", linewidth=1.2,
               label=f"trigger ({term.trigger_time}s)")
    ax.axvline(term.terminal_time, color="magenta", linestyle="--", linewidth=1.2,
               label=f"terminal ({term.terminal_time}s)")
    ax.set_xlabel("time in opposing box (s)")
    ax.set_ylabel("penalty (unsigned)")
    ax.set_title(f"Time ramp at centroid (power={term.power})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)


def _plot_spatial(
    ax: plt.Axes, term: GoalieBoxPenalty, fxh: float, fyh: float, t_norm: float
) -> None:
    # Heatmap window — focus on the +x box and a buffer around it.
    x_lo = fxh - term.goalie_box_depth - 0.04
    x_hi = fxh + 0.005
    y_lo = -term.goalie_box_y_half - 0.04
    y_hi = +term.goalie_box_y_half + 0.04
    res = 0.002
    xs = np.arange(x_lo, x_hi + res, res)
    ys = np.arange(y_lo, y_hi + res, res)
    grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            grid[j, i] = term(
                _make_ctx(
                    rx=float(x), ry=float(y), time_in_box_norm=t_norm,
                    field_x_half=fxh, field_y_half=fyh,
                )
            )
    im = ax.imshow(
        grid, origin="lower",
        extent=(x_lo, x_hi, y_lo, y_hi),
        cmap="hot", vmin=0.0, vmax=1.0, aspect="equal",
        interpolation="nearest",
    )
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("penalty (unsigned)", rotation=270, labelpad=12)
    # Box outline.
    ax.add_patch(Rectangle(
        (fxh - term.goalie_box_depth, -term.goalie_box_y_half),
        term.goalie_box_depth, 2 * term.goalie_box_y_half,
        fill=False, edgecolor="cyan", linewidth=1.5, linestyle="--",
    ))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Spatial @ t={t_norm:.2f}·terminal")


def _plot_joint(
    ax: plt.Axes, term: GoalieBoxPenalty, fxh: float, fyh: float
) -> None:
    """Heatmap of penalty over (intrusion_x, time_in_box_norm) at y=0."""
    intrusions = np.linspace(0.0, term.goalie_box_depth, 80)
    ts = np.linspace(0.0, 1.0, 80)
    grid = np.zeros((len(ts), len(intrusions)), dtype=np.float32)
    for j, t in enumerate(ts):
        for i, d in enumerate(intrusions):
            rx = fxh - term.goalie_box_depth + float(d)
            grid[j, i] = term(
                _make_ctx(
                    rx=rx, ry=0.0, time_in_box_norm=float(t),
                    field_x_half=fxh, field_y_half=fyh,
                )
            )
    im = ax.imshow(
        grid, origin="lower",
        extent=(intrusions[0] * 1000, intrusions[-1] * 1000, ts[0], ts[-1]),
        cmap="hot", vmin=0.0, vmax=1.0, aspect="auto",
        interpolation="nearest",
    )
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("penalty (unsigned)", rotation=270, labelpad=12)
    # Trigger / depth saturation lines.
    ax.axhline(term.trigger_time / term.terminal_time, color="cyan",
               linestyle="--", linewidth=1.2, label="trigger")
    ax.axvline(term.depth_saturation * 1000, color="lime",
               linestyle=":", linewidth=1.2, label=f"depth sat ({term.depth_saturation*1000:.0f}mm)")
    ax.set_xlabel("intrusion (mm)")
    ax.set_ylabel("time_in_box / terminal")
    ax.set_title("Joint depth × time")
    ax.legend(loc="upper left", fontsize=8)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weight", type=float, default=1.0)
    p.add_argument("--trigger-time", type=float, default=2.0)
    p.add_argument("--terminal-time", type=float, default=3.0)
    p.add_argument("--power", type=float, default=3.0)
    p.add_argument("--termination-penalty", type=float, default=1.0)
    p.add_argument("--goalie-box-depth", type=float, default=0.12)
    p.add_argument("--goalie-box-y-half", type=float, default=0.10)
    p.add_argument("--depth-saturation", type=float, default=0.06)
    p.add_argument("--field-x-half", type=float, default=0.375)
    p.add_argument("--field-y-half", type=float, default=0.225)
    p.add_argument(
        "--out",
        type=str,
        default="AtomGym/research/goalie_box/goalie_box_heatmap.png",
    )
    p.add_argument("--no-show", action="store_true")
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
        depth_saturation=args.depth_saturation,
    )

    print("Building GoalieBoxPenalty:")
    print(f"  weight             : {term.weight}")
    print(f"  trigger_time       : {term.trigger_time} s")
    print(f"  terminal_time      : {term.terminal_time} s")
    print(f"  power              : {term.power}")
    print(f"  termination_penalty: {term.termination_penalty}")
    print(f"  goalie_box_depth   : {term.goalie_box_depth*1000:.0f} mm")
    print(f"  goalie_box_y_half  : {term.goalie_box_y_half*1000:.0f} mm")
    print(f"  depth_saturation   : {term.depth_saturation*1000:.0f} mm")

    _probe(term, args.field_x_half, args.field_y_half)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _plot_time_ramp(axes[0], term, args.field_x_half, args.field_y_half)
    _plot_spatial(axes[1], term, args.field_x_half, args.field_y_half, t_norm=0.95)
    _plot_joint(axes[2], term, args.field_x_half, args.field_y_half)
    fig.suptitle(
        f"GoalieBoxPenalty — trigger {term.trigger_time}s, terminal {term.terminal_time}s, "
        f"power {term.power}, term penalty {term.termination_penalty}",
        fontsize=12,
    )
    plt.tight_layout()

    out_path = args.out
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved heatmap → {out_path}")
    if not args.no_show:
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
