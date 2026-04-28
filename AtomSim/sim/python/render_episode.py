"""Render a recorded .npz episode to mp4 or gif.

Usage:
    .venv/bin/python AtomSim/sim/python/render_episode.py EPISODE.npz [options]

Options:
    --style PATH          Path to a style YAML. Defaults to sim/configs/styles/default.yaml.
    --out PATH            Output path. Extension picks the format (.mp4 or .gif).
                          Defaults to EPISODE.mp4 next to the input.
    --fps FLOAT           Output framerate. Defaults to 1/dt from the episode meta.
    --quality INT         mp4 quality 1-10 (default 8). Ignored for gif.
    --frame-stride INT    Render every Nth frame (default 1, render every frame).
                          Useful for fast previews of long episodes.

The script auto-locates AtomSim and the viz package — run from anywhere.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_atomsim_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "CMakePresets.json").exists():
            return p
    raise RuntimeError("Could not locate AtomSim/ from this script's path.")


_atomsim = _find_atomsim_root()
sys.path.insert(0, str(_atomsim / "sim" / "python"))

from viz import Episode, load_style  # noqa: E402
from viz.recorder import write_video  # noqa: E402
from viz.renderers import PygameHeadlessRenderer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("episode", type=Path, help="Path to a recorded .npz episode")
    p.add_argument(
        "--style",
        type=Path,
        default=_atomsim / "sim" / "configs" / "styles" / "default.yaml",
        help="Style YAML",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (.mp4 or .gif). Defaults to EPISODE.mp4 next to input.",
    )
    p.add_argument("--fps", type=float, default=None, help="Output framerate")
    p.add_argument("--quality", type=int, default=8, help="mp4 quality 1-10")
    p.add_argument(
        "--frame-stride", type=int, default=1, help="Render every Nth frame"
    )
    args = p.parse_args()

    ep = Episode.load(args.episode)
    print(f"Loaded {args.episode.name}: {ep.num_frames} frames, dt={ep.dt:.4f}s "
          f"({ep.num_frames * ep.dt:.2f}s total)")

    style = load_style(args.style)
    world = ep.meta["world"]
    renderer = PygameHeadlessRenderer(
        style,
        field_x_half=float(world["field_x_half"]),
        field_y_half=float(world["field_y_half"]),
    )

    out = args.out or args.episode.with_suffix(".mp4")
    fps = args.fps if args.fps else (1.0 / ep.dt if ep.dt > 0 else 60.0)
    if args.frame_stride > 1:
        fps = fps / args.frame_stride

    indices = range(0, ep.num_frames, max(1, args.frame_stride))
    print(f"Rendering {len(list(indices))} frames @ {fps:.1f} fps → {out}")
    frames = [renderer.render(ep.scene_at(i)) for i in indices]
    write_video(out, frames, fps=fps, quality=args.quality)
    renderer.close()

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Wrote {out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
