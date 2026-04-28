"""
Minimal random-action gif recorder.

Drives the robot with random wheel commands for 5 s at 24 fps, renders
each frame headlessly, and writes a gif.

Usage:
    .venv/bin/python AtomSim/sim/python/random_gif.py [out.gif]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _find_atomsim_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "CMakePresets.json").exists():
            return p
    raise RuntimeError("Could not locate AtomSim root.")


_atomsim = _find_atomsim_root()
sys.path.insert(0, str(_atomsim / "build" / "release" / "sim" / "bindings"))
sys.path.insert(0, str(_atomsim / "sim" / "python"))

import sim_py  # noqa: E402

from viz import build_scene, load_style, write_video  # noqa: E402
from viz.input import TeleopInput, diff_drive_wheels_from_input  # noqa: E402
from viz.renderers import PygameHeadlessRenderer  # noqa: E402


def load_robot_config(name: str) -> sim_py.RobotConfig:
    cfg_dir = _atomsim / "sim" / "configs"
    data = json.loads((cfg_dir / "robots" / f"{name}.json").read_text())
    cfg = sim_py.RobotConfig()
    cfg.body_type = sim_py.BodyType.Dynamic
    cfg.chassis_side = float(data.get("chassis_side", 0.060))
    if "manipulator" in data:
        m = json.loads(
            (cfg_dir / "manipulators" / f"{data['manipulator']}.json").read_text()
        )
        cfg.manipulator_parts = [
            [(float(v[0]), float(v[1])) for v in part] for part in m["parts"]
        ]
    cfg.mass = 0.3
    cfg.yaw_inertia = 5.0e-4
    cfg.dynamics_params.track_width = 0.060
    cfg.dynamics_params.tau_motor = 0.05
    return cfg


def main() -> None:
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "random.gif")

    FPS = 24
    DURATION_S = 5.0
    DT = 1.0 / FPS
    N_FRAMES = int(round(DURATION_S * FPS))
    MAX_WHEEL_SPEED = 0.225

    robot_cfg = load_robot_config("diff_drive_default")
    robot_cfg.x0 = -0.10
    robot_cfg.y0 = 0.00

    ball_cfg = sim_py.BallConfig()
    ball_cfg.x0 = 0.10
    ball_cfg.dynamics_params.radius = 0.014

    world = sim_py.World()
    robot = sim_py.Robot(world, robot_cfg)
    ball = sim_py.Ball(world, ball_cfg)

    style = load_style(_atomsim / "sim" / "configs" / "styles" / "default.yaml")
    renderer = PygameHeadlessRenderer(
        style,
        field_x_half=world.config.field_x_half,
        field_y_half=world.config.field_y_half,
        show_hud=True,
    )

    rng = np.random.default_rng(42)
    frames = []
    for i in range(N_FRAMES):
        # Sample normalised actions in [-1, 1] then convert to wheel commands;
        # this way the same (forward, turn) drives the sim AND the on-screen
        # control bars.
        fwd  = float(rng.uniform(-1.0, 1.0))
        turn = float(rng.uniform(-1.0, 1.0))
        v_left, v_right = diff_drive_wheels_from_input(
            TeleopInput(forward=fwd, turn=turn),
            max_wheel_speed=MAX_WHEEL_SPEED,
            turn_rate_k=0.4,
        )
        cmd = np.array([v_left, v_right], dtype=np.float32)

        robot.pre_step(cmd, DT)
        ball.pre_step(DT)
        world.step(DT)
        robot.post_step()
        ball.post_step()

        sim_t = i * DT
        scene = build_scene(
            world, [("blue", robot)], [("main", ball)],
            t=sim_t, teams={"blue": "blue"},
        )
        scene.controls = {"blue": (fwd, turn)}
        frames.append(renderer.render(scene, hud_lines=[f"t={sim_t:5.2f} s"]))

    write_video(out_path, frames, fps=FPS)
    renderer.close()
    print(f"Wrote {out_path} — {N_FRAMES} frames, {DURATION_S}s @ {FPS} fps")


if __name__ == "__main__":
    main()
