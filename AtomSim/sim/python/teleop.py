"""
Real-time teleop: drive the robot with the keyboard (and optionally a
gamepad), watch it interact with a ball in the simulated field.

Usage:
    .venv/bin/python AtomSim/sim/python/teleop.py [--record [PATH]]

Requires the release build of sim_py and the `viz` dep group:

    cmake --preset release && cmake --build build/release   # from AtomSim/
    uv sync --group viz                                     # from repo root

Controls:
    Keyboard    W / ↑    drive forward          A / ←   turn CCW (left)
                S / ↓    drive reverse          D / →   turn CW  (right)
                R        reset                  ESC/Q   quit
    Gamepad     Left stick    drive (Y) + turn (X) — analog
                A button (0)  reset
                Back  (6)     quit

If a gamepad is detected, both keyboard and gamepad are active simultaneously
(signals are summed and clipped). Otherwise keyboard-only.

If --record is passed, every frame is captured into an Episode and saved
on quit. Re-render the saved .npz with `render_episode.py` for a video.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402


# --- Locate AtomSim and import sim bindings ---------------------------------


def _find_atomsim_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "CMakePresets.json").exists():
            return p
    raise RuntimeError("Could not locate AtomSim/ from this script's path.")


_atomsim = _find_atomsim_root()
_build_dir = _atomsim / "build" / "release" / "sim" / "bindings"
if not _build_dir.exists():
    raise RuntimeError(
        f"No release build at {_build_dir}.\n"
        f"From {_atomsim}, run: cmake --preset release && cmake --build build/release"
    )
sys.path.insert(0, str(_build_dir))
sys.path.insert(0, str(_atomsim / "sim" / "python"))

import sim_py  # noqa: E402

from viz import EpisodeRecorder, build_scene, load_style  # noqa: E402
from viz.input import (  # noqa: E402
    CompositeInput,
    KeyboardInput,
    detect_gamepad,
    diff_drive_wheels_from_input,
)
from viz.renderers import PygameLiveRenderer  # noqa: E402


# --- Config loading ---------------------------------------------------------


def load_robot_config(name: str) -> tuple[sim_py.RobotConfig, list[list[list[float]]]]:
    """Returns (RobotConfig, manipulator_parts_as_nested_list).

    The nested list copy is used for the episode meta blob — sim_py's polygon
    type is opaque to JSON, so we keep a plain Python view alongside it."""
    cfg_dir = _atomsim / "sim" / "configs"
    data = json.loads((cfg_dir / "robots" / f"{name}.json").read_text())
    cfg = sim_py.RobotConfig()
    cfg.chassis_side = float(data.get("chassis_side", 0.060))
    parts: list[list[list[float]]] = []
    if "manipulator" in data:
        m_data = json.loads(
            (cfg_dir / "manipulators" / f"{data['manipulator']}.json").read_text()
        )
        parts = [[[float(v[0]), float(v[1])] for v in part] for part in m_data["parts"]]
        cfg.manipulator_parts = [
            [(float(v[0]), float(v[1])) for v in part] for part in m_data["parts"]
        ]
    return cfg, parts


# --- Main loop --------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="AtomSim real-time teleop")
    p.add_argument(
        "--record",
        nargs="?",
        const="auto",
        default=None,
        help="Record an episode and save to PATH on quit. With no PATH, "
             "auto-names episode_YYYYMMDD_HHMMSS.npz in the current dir.",
    )
    args = p.parse_args()

    # --- sim setup ---
    robot_cfg, manip_parts = load_robot_config("diff_drive_sidewall")
    robot_cfg.body_type = sim_py.BodyType.Dynamic
    robot_cfg.mass = 0.3
    robot_cfg.yaw_inertia = 5.0e-4
    robot_cfg.x0 = -0.20
    robot_cfg.y0 = 0.00
    robot_cfg.theta0 = 0.00
    robot_cfg.dynamics_params.track_width = 0.060
    robot_cfg.dynamics_params.tau_motor = 0.05

    ball_cfg = sim_py.BallConfig()
    ball_cfg.x0 = 0.10
    ball_cfg.y0 = 0.05
    ball_cfg.field_k = 70.0
    ball_cfg.dynamics_params.radius = 0.014
    ball_cfg.dynamics_params.mass = 0.05
    ball_cfg.dynamics_params.restitution = 0.4
    ball_cfg.dynamics_params.damping = 0.8

    world = sim_py.World()
    robot = sim_py.Robot(world, robot_cfg)
    ball = sim_py.Ball(world, ball_cfg)

    # --- renderer ---
    style = load_style(_atomsim / "sim" / "configs" / "styles" / "default.yaml")
    renderer = PygameLiveRenderer(
        style,
        title="AtomSim teleop — WASD/arrows or gamepad, ESC to quit",
        field_x_half=world.config.field_x_half,
        field_y_half=world.config.field_y_half,
    )
    clock = pygame.time.Clock()

    # --- inputs ---
    devices: list = [KeyboardInput()]
    pad = detect_gamepad()
    if pad is not None:
        print(f"Gamepad detected: {pad.name}")
        devices.append(pad)
    inputs = CompositeInput(*devices)

    # --- control mapping ---
    MAX_WHEEL_SPEED = 0.225
    TURN_RATE_K = 0.4
    DT = 1.0 / 60.0

    # --- episode recording (optional) ---
    rec: EpisodeRecorder | None = None
    if args.record is not None:
        if args.record == "auto":
            stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path.cwd() / f"episode_{stamp}.npz"
        else:
            out_path = Path(args.record)
        rec = EpisodeRecorder(
            dt=DT,
            world={
                "field_x_half": float(world.config.field_x_half),
                "field_y_half": float(world.config.field_y_half),
            },
            agents=[
                {"name": "blue", "type": "diff_drive", "team": "blue",
                 "config": {
                     "chassis_side": float(robot_cfg.chassis_side),
                     "manipulator_parts": manip_parts,
                 }},
                {"name": "main", "type": "ball", "team": None,
                 "config": {"radius": float(ball_cfg.dynamics_params.radius)}},
            ],
        )
        print(f"Recording to {out_path}")

    def reset() -> None:
        rs = np.array(
            [robot_cfg.x0, robot_cfg.y0, robot_cfg.theta0, 0.0, 0.0], dtype=np.float32
        )
        bs = np.array([ball_cfg.x0, ball_cfg.y0, 0.0, 0.0], dtype=np.float32)
        robot.set_state(rs)
        ball.set_state(bs)

    sim_t = 0.0
    running = True
    while running:
        events = pygame.event.get()
        inp = inputs.poll(events)
        if inp.quit:
            running = False
        if inp.reset:
            reset()
            sim_t = 0.0

        v_left, v_right = diff_drive_wheels_from_input(
            inp, max_wheel_speed=MAX_WHEEL_SPEED, turn_rate_k=TURN_RATE_K
        )
        cmd = np.array([v_left, v_right], dtype=np.float32)

        # --- step ---
        robot.pre_step(cmd, DT)
        ball.pre_step(DT)
        world.step(DT)
        robot.post_step()
        ball.post_step()
        sim_t += DT

        # --- record ---
        norm_input = np.array([inp.forward, inp.turn], dtype=np.float32)
        if rec is not None:
            rec.append(
                t=sim_t,
                robot_states={"blue": np.array(robot.state, dtype=np.float32)},
                robot_actions={"blue": cmd},
                robot_inputs={"blue": norm_input},
                ball_states={"main": np.array(ball.state, dtype=np.float32)},
            )

        # --- render ---
        scene = build_scene(
            world, [("blue", robot)], [("main", ball)],
            t=sim_t, teams={"blue": "blue"},
        )
        scene.controls = {"blue": (inp.forward, inp.turn)}
        s, b = robot.state, ball.state
        hud_lines = [
            f"robot: x={float(s[0]):+.3f}  y={float(s[1]):+.3f}  θ={float(s[2]):+.3f}    "
            f"v={float(s[3]):+.3f}  ω={float(s[4]):+.3f}",
            f"ball:  x={float(b[0]):+.3f}  y={float(b[1]):+.3f}                      "
            f"vx={float(b[2]):+.3f}  vy={float(b[3]):+.3f}",
            f"input: fwd={inp.forward:+.2f}  turn={inp.turn:+.2f}    "
            f"cmd: vL={cmd[0]:+.2f}  vR={cmd[1]:+.2f}",
            f"t={sim_t:.2f}s" + ("  [REC]" if rec is not None else ""),
            "",
            "WASD/↑↓←→ = drive   R = reset   ESC/Q = quit",
        ]
        renderer.render(scene, hud_lines=hud_lines)
        clock.tick(60)

    # --- finalize ---
    if rec is not None and len(rec) > 0:
        ep = rec.finalize()
        ep.save(out_path)
        print(f"Saved {ep.num_frames}-frame episode ({ep.num_frames * DT:.1f}s) to {out_path}")
    renderer.close()


if __name__ == "__main__":
    main()
