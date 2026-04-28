"""
Replay and scrub a recorded .npz episode in a live pygame window.

Usage:
    .venv/bin/python AtomSim/sim/python/replay_episode.py EPISODE.npz [--style PATH]

Controls:
    Space          play / pause
    ← / →          step back / forward 1 frame      (pauses)
    Shift + ← / →  step back / forward 10 frames    (pauses)
    Home / End     jump to start / end
    R              reset to t=0 and play
    [ / ]          halve / double playback speed
    Click on bar   seek to that position
    Click + drag   scrub
    ESC / Q        quit

The control-bar panel at the top shows the inputs that were RECORDED in
the episode (the bars are populated by Episode.scene_at via robot_inputs).
No sim is being stepped — every frame is reconstructed from the .npz.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402


def _find_atomsim_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "CMakePresets.json").exists():
            return p
    raise RuntimeError("Could not locate AtomSim root.")


_atomsim = _find_atomsim_root()
sys.path.insert(0, str(_atomsim / "sim" / "python"))

from viz import Episode, load_style  # noqa: E402
from viz.renderers import PygameLiveRenderer  # noqa: E402


# ---------------------------------------------------------------------------
# Scrubber UI
# ---------------------------------------------------------------------------

SCRUBBER_HEIGHT = 36
SCRUBBER_PAD_X = 16
BAR_HEIGHT = 6


def make_scrubber(window_w: int, window_h: int, n_frames: int):
    """Returns helpers for the bottom-of-window scrubber: layout, hit-test,
    pixel↔frame conversion, and a draw function."""
    bar_y = window_h - SCRUBBER_HEIGHT + 22
    bar_x_start = SCRUBBER_PAD_X
    bar_x_end = window_w - SCRUBBER_PAD_X
    bar_w = bar_x_end - bar_x_start
    hit_y0 = window_h - SCRUBBER_HEIGHT
    hit_y1 = window_h

    def is_in_scrubber(mx: int, my: int) -> bool:
        return hit_y0 <= my <= hit_y1 and bar_x_start - 4 <= mx <= bar_x_end + 4

    def x_to_frame(mx: int) -> int:
        ratio = (mx - bar_x_start) / max(1, bar_w)
        return max(0, min(n_frames - 1, int(round(ratio * (n_frames - 1)))))

    def frame_to_x(idx: int) -> int:
        ratio = idx / max(1, n_frames - 1)
        return bar_x_start + int(round(ratio * bar_w))

    return bar_x_start, bar_x_end, bar_y, bar_w, is_in_scrubber, x_to_frame, frame_to_x


def main() -> None:
    p = argparse.ArgumentParser(description="Scrub-replay an .npz episode.")
    p.add_argument("episode", type=Path, help="Path to a recorded .npz episode")
    p.add_argument(
        "--style",
        type=Path,
        default=_atomsim / "sim" / "configs" / "styles" / "default.yaml",
    )
    args = p.parse_args()

    ep = Episode.load(args.episode)
    if ep.num_frames < 2:
        raise RuntimeError(f"Episode has only {ep.num_frames} frame(s); nothing to scrub.")
    style = load_style(args.style)

    world = ep.meta["world"]
    renderer = PygameLiveRenderer(
        style,
        title=f"AtomSim replay — {args.episode.name}",
        field_x_half=float(world["field_x_half"]),
        field_y_half=float(world["field_y_half"]),
    )
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    DT = ep.dt if ep.dt > 0 else 1.0 / 60.0
    n_frames = ep.num_frames
    total_t = (n_frames - 1) * DT

    out_w = style.resolution.output_w
    out_h = style.resolution.output_h
    (bar_x_start, bar_x_end, bar_y, bar_w,
     is_in_scrubber, x_to_frame, frame_to_x) = make_scrubber(out_w, out_h, n_frames)

    # Playback state
    frame_idx = 0
    playing = True
    speed = 1.0
    accumulated = 0.0   # seconds accumulated since last frame advance
    dragging = False

    # Colors for the scrubber overlay
    BAR_BG = (60, 60, 60)
    BAR_FILL = (200, 200, 200)
    CURSOR = (255, 255, 255)
    LABEL = (240, 240, 240)
    SHADOW_BG = (0, 0, 0)

    def draw_scrubber(window: pygame.Surface) -> None:
        # Translucent dark strip behind the bar so it reads on any field colour.
        strip = pygame.Surface((out_w, SCRUBBER_HEIGHT), pygame.SRCALPHA)
        strip.fill((0, 0, 0, 130))
        window.blit(strip, (0, out_h - SCRUBBER_HEIGHT))

        # Background bar.
        pygame.draw.rect(
            window, BAR_BG, (bar_x_start, bar_y, bar_w, BAR_HEIGHT), border_radius=2
        )
        # Filled progress.
        cursor_x = frame_to_x(frame_idx)
        fill_w = max(0, cursor_x - bar_x_start)
        if fill_w > 0:
            pygame.draw.rect(
                window, BAR_FILL,
                (bar_x_start, bar_y, fill_w, BAR_HEIGHT), border_radius=2,
            )
        # Cursor (vertical handle).
        pygame.draw.line(
            window, CURSOR, (cursor_x, bar_y - 6), (cursor_x, bar_y + BAR_HEIGHT + 6), 3
        )

        # Status label (top-left of the strip).
        state = "▶ PLAY" if playing else "❚❚ PAUSE"
        speed_str = f"  ×{speed:.2f}" if abs(speed - 1.0) > 1e-3 else ""
        cur_t = frame_idx * DT
        label = (
            f"{state}{speed_str}   "
            f"t={cur_t:6.2f}/{total_t:.2f}s   frame {frame_idx}/{n_frames - 1}"
        )
        txt = font.render(label, True, LABEL)
        # Drop a 1-px shadow so it stays readable on any field colour.
        shadow = font.render(label, True, SHADOW_BG)
        window.blit(shadow, (SCRUBBER_PAD_X + 1, out_h - SCRUBBER_HEIGHT + 5))
        window.blit(txt,    (SCRUBBER_PAD_X,     out_h - SCRUBBER_HEIGHT + 4))

    running = True
    last_tick_ms = pygame.time.get_ticks()
    while running:
        # Real-time delta for auto-advance.
        now_ms = pygame.time.get_ticks()
        dt_real = (now_ms - last_tick_ms) / 1000.0
        last_tick_ms = now_ms

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    playing = not playing
                    accumulated = 0.0
                elif ev.key == pygame.K_LEFT:
                    step = 10 if pygame.key.get_mods() & pygame.KMOD_SHIFT else 1
                    frame_idx = max(0, frame_idx - step)
                    playing = False
                elif ev.key == pygame.K_RIGHT:
                    step = 10 if pygame.key.get_mods() & pygame.KMOD_SHIFT else 1
                    frame_idx = min(n_frames - 1, frame_idx + step)
                    playing = False
                elif ev.key == pygame.K_HOME:
                    frame_idx = 0
                elif ev.key == pygame.K_END:
                    frame_idx = n_frames - 1
                elif ev.key == pygame.K_r:
                    frame_idx = 0
                    playing = True
                    accumulated = 0.0
                elif ev.key == pygame.K_LEFTBRACKET:
                    speed = max(0.0625, speed * 0.5)
                elif ev.key == pygame.K_RIGHTBRACKET:
                    speed = min(16.0, speed * 2.0)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                if is_in_scrubber(mx, my):
                    frame_idx = x_to_frame(mx)
                    dragging = True
                    playing = False
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging:
                mx, _ = ev.pos
                frame_idx = x_to_frame(mx)

        # Auto-advance during playback.
        if playing and not dragging:
            accumulated += dt_real * speed
            advance = int(accumulated // DT)
            if advance:
                accumulated -= advance * DT
                frame_idx += advance
                if frame_idx >= n_frames - 1:
                    frame_idx = n_frames - 1
                    playing = False
                    accumulated = 0.0

        scene = ep.scene_at(frame_idx)
        renderer.render(
            scene,
            hud_lines=[f"replay: {args.episode.name}"],
            overlay=draw_scrubber,
        )
        clock.tick(60)

    renderer.close()


if __name__ == "__main__":
    main()
