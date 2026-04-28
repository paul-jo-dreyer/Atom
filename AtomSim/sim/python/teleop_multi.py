"""
Multi-robot teleop for visually testing the control indicator panel layout.

Spawns N robots (N ∈ 1..6) split across blue and orange teams. The first
robot is driven by keyboard / gamepad; the remaining robots get synthetic
animated control signals (sin/cos), so every cell in the panel animates in
real time and you can verify the layout for any team count.

Usage:
    .venv/bin/python AtomSim/sim/python/teleop_multi.py [N]

    N=1 → 1 blue
    N=2 → 1 blue + 1 orange
    N=3 → 2 blue + 1 orange
    N=4 → 2 blue + 2 orange
    N=5 → 3 blue + 2 orange
    N=6 → 3 blue + 3 orange   (default)

Drives only robot index 0 — the rest stay where they spawn (zero wheel cmd).
You can drive into them to verify physics still works with multiple bodies.

Controls: WASD / arrows + R (reset) + ESC (quit). Gamepad picked up if present.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402


def _find_atomsim_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "CMakePresets.json").exists():
            return p
    raise RuntimeError("Could not locate AtomSim/ root.")


_atomsim = _find_atomsim_root()
_build_dir = _atomsim / "build" / "release" / "sim" / "bindings"
if not _build_dir.exists():
    raise RuntimeError(
        f"No release build at {_build_dir}. From AtomSim/, run: "
        f"cmake --preset release && cmake --build build/release"
    )
sys.path.insert(0, str(_build_dir))
sys.path.insert(0, str(_atomsim / "sim" / "python"))

import sim_py  # noqa: E402

from viz import build_scene, load_style  # noqa: E402
from viz.input import (  # noqa: E402
    CompositeInput,
    KeyboardInput,
    detect_gamepad,
    diff_drive_wheels_from_input,
)
from viz.renderers import PygameLiveRenderer  # noqa: E402


# ----- robot factory --------------------------------------------------------


def _load_robot_config(name: str) -> sim_py.RobotConfig:
    cfg_dir = _atomsim / "sim" / "configs"
    data = json.loads((cfg_dir / "robots" / f"{name}.json").read_text())
    cfg = sim_py.RobotConfig()
    cfg.body_type = sim_py.BodyType.Dynamic
    cfg.chassis_side = float(data.get("chassis_side", 0.060))
    if "manipulator" in data:
        m_data = json.loads(
            (cfg_dir / "manipulators" / f"{data['manipulator']}.json").read_text()
        )
        cfg.manipulator_parts = [
            [(float(v[0]), float(v[1])) for v in p] for p in m_data["parts"]
        ]
    cfg.mass = 0.3
    cfg.yaw_inertia = 5.0e-4
    cfg.dynamics_params.track_width = 0.060
    cfg.dynamics_params.tau_motor = 0.05
    return cfg


# ----- team / pose distribution --------------------------------------------


_TEAM_COUNTS: dict[int, tuple[int, int]] = {
    1: (1, 0),  # blue, orange
    2: (1, 1),
    3: (2, 1),
    4: (2, 2),
    5: (3, 2),
    6: (3, 3),
}


def _row_positions(count: int) -> list[float]:
    """X coords for `count` robots in a row, spread across the field."""
    if count <= 0:
        return []
    if count == 1:
        return [0.0]
    if count == 2:
        return [-0.12, 0.12]
    return [-0.22, 0.0, 0.22]


# ----- main ----------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("n", nargs="?", type=int, default=6, choices=list(_TEAM_COUNTS))
    args = p.parse_args()

    n_blue, n_orange = _TEAM_COUNTS[args.n]
    print(f"Spawning {args.n} robots: {n_blue} blue, {n_orange} orange.")

    world = sim_py.World()

    # robots: list of (name, robot, cfg, team).
    robots: list[tuple[str, "sim_py.Robot", "sim_py.RobotConfig", str]] = []
    for k, x in enumerate(_row_positions(n_blue)):
        cfg = _load_robot_config("diff_drive_sidewall")
        cfg.x0, cfg.y0, cfg.theta0 = x, 0.10, 0.0
        robots.append((f"B{k + 1}", sim_py.Robot(world, cfg), cfg, "blue"))
    for k, x in enumerate(_row_positions(n_orange)):
        cfg = _load_robot_config("diff_drive_sidewall")
        cfg.x0, cfg.y0, cfg.theta0 = x, -0.10, math.pi
        robots.append((f"O{k + 1}", sim_py.Robot(world, cfg), cfg, "orange"))

    active_name = robots[0][0]
    print(f"Driving {active_name}; the rest get synthetic animated bar values.")

    # --- viz ---
    style = load_style(_atomsim / "sim" / "configs" / "styles" / "default.yaml")
    renderer = PygameLiveRenderer(
        style,
        title=f"AtomSim multi-teleop ({args.n} robots) — drive {active_name}",
        field_x_half=world.config.field_x_half,
        field_y_half=world.config.field_y_half,
    )
    clock = pygame.time.Clock()

    devices: list = [KeyboardInput()]
    pad = detect_gamepad()
    if pad is not None:
        print(f"Gamepad detected: {pad.name}")
        devices.append(pad)
    inputs = CompositeInput(*devices)

    MAX_WHEEL_SPEED = 0.225
    TURN_RATE_K = 0.4
    DT = 1.0 / 60.0

    def reset_all() -> None:
        for _, robot, cfg, _ in robots:
            state = np.array([cfg.x0, cfg.y0, cfg.theta0, 0.0, 0.0], dtype=np.float32)
            robot.set_state(state)

    sim_t = 0.0
    running = True
    while running:
        events = pygame.event.get()
        inp = inputs.poll(events)
        if inp.quit:
            running = False
        if inp.reset:
            reset_all()
            sim_t = 0.0

        v_left, v_right = diff_drive_wheels_from_input(
            inp, max_wheel_speed=MAX_WHEEL_SPEED, turn_rate_k=TURN_RATE_K
        )
        active_cmd = np.array([v_left, v_right], dtype=np.float32)
        zero_cmd = np.zeros(2, dtype=np.float32)

        # Step everyone — only the active robot gets a real command.
        for i, (_, robot, _, _) in enumerate(robots):
            robot.pre_step(active_cmd if i == 0 else zero_cmd, DT)
        world.step(DT)
        for _, robot, _, _ in robots:
            robot.post_step()
        sim_t += DT

        # Build scene + control panel signals.
        teams_dict = {name: team for name, _, _, team in robots}
        robots_arg = [(name, robot) for name, robot, _, _ in robots]
        scene = build_scene(world, robots_arg, [], t=sim_t, teams=teams_dict)

        controls: dict[str, tuple[float, float]] = {}
        for i, (name, _, _, _) in enumerate(robots):
            if i == 0:
                controls[name] = (inp.forward, inp.turn)
            else:
                # Synthetic signals so non-active bars animate. Per-robot phase
                # so the cells move out of phase with each other — easy to
                # see they're independent.
                phase = i * 0.85
                controls[name] = (
                    0.85 * math.sin(1.2 * sim_t + phase),
                    0.85 * math.cos(1.7 * sim_t + phase),
                )
        scene.controls = controls

        hud = [
            f"{args.n} robots ({n_blue}B + {n_orange}O) — driving {active_name}",
            f"t={sim_t:.2f}s   live input: fwd={inp.forward:+.2f}  turn={inp.turn:+.2f}",
            "WASD/arrows = drive   R = reset   ESC = quit",
        ]
        renderer.render(scene, hud_lines=hud)
        clock.tick(60)

    renderer.close()


if __name__ == "__main__":
    main()
