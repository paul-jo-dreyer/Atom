"""
Real-time teleop: drive the robot with the keyboard, watch it interact with
a ball in the simulated field. Self-locating — run from anywhere:

    .venv/bin/python AtomSim/sim/python/teleop.py

Requires the release build of sim_py and the `viz` dep group (pygame):

    cmake --preset release && cmake --build build/release   # from AtomSim/
    uv sync --group viz                                     # from repo root

Controls:
    W / ↑    — both wheels forward (drive)
    S / ↓    — both wheels reverse
    A / ←    — turn CCW (left)
    D / →    — turn CW (right)
    R        — reset robot + ball to initial poses
    ESC / Q  — quit

Combinations work the way you'd expect: W+A drives forward while turning left,
S+D reverses while turning right (banks out the back).

Internally the sim runs at 60 Hz with dt = 1/60 s. The diff_drive motor lag
(τ ≈ 50 ms) is well-resolved at this rate; RK4 has ample headroom.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# Suppress the pygame splash before importing it.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402


# --- Locate the AtomSim build and import the sim bindings -------------------


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

import sim_py  # noqa: E402


# --- Render config ----------------------------------------------------------

WINDOW_W, WINDOW_H = 1100, 720
SCALE_PX_PER_M = 1100  # field 0.75 m → 825 px wide, fits comfortably

BG_COLOR = (250, 250, 250)
FIELD_BORDER_COLOR = (60, 60, 60)
ROBOT_FILL_COLOR = (165, 180, 230)
ROBOT_OUTLINE_COLOR = (40, 40, 90)
MANIP_FILL_COLOR = (235, 165, 90)
MANIP_OUTLINE_COLOR = (140, 85, 35)
BALL_FILL_COLOR = (220, 70, 70)
BALL_OUTLINE_COLOR = (140, 35, 35)
HUD_COLOR = (30, 30, 30)


def world_to_screen(x: float, y: float) -> tuple[int, int]:
    """World metres (origin at field centre, +y = up) → window pixels."""
    return (
        int(WINDOW_W / 2 + x * SCALE_PX_PER_M),
        int(WINDOW_H / 2 - y * SCALE_PX_PER_M),
    )


def m_to_px(d: float) -> int:
    return max(1, int(d * SCALE_PX_PER_M))


def _rotate_translate(
    pt: tuple[float, float], theta: float, ox: float, oy: float
) -> tuple[float, float]:
    c, s = np.cos(theta), np.sin(theta)
    return (ox + c * pt[0] - s * pt[1], oy + s * pt[0] + c * pt[1])


# --- Drawing primitives -----------------------------------------------------


def draw_field(surf, world):
    cfg = world.config
    xh, yh = cfg.field_x_half, cfg.field_y_half
    tl = world_to_screen(-xh, yh)
    br = world_to_screen(xh, -yh)
    rect = pygame.Rect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
    pygame.draw.rect(surf, FIELD_BORDER_COLOR, rect, width=3)


def draw_robot(surf, robot):
    s = robot.state
    px, py, theta = float(s[0]), float(s[1]), float(s[2])
    cfg = robot.config

    # Chassis: square centred at body origin
    half = cfg.chassis_side * 0.5
    chassis_local = [(-half, -half), (half, -half), (half, half), (-half, half)]
    chassis_screen = [
        world_to_screen(*_rotate_translate(p, theta, px, py)) for p in chassis_local
    ]
    pygame.draw.polygon(surf, ROBOT_FILL_COLOR, chassis_screen)
    pygame.draw.polygon(surf, ROBOT_OUTLINE_COLOR, chassis_screen, width=2)

    # Manipulator polygons (one or more parts)
    for part in cfg.manipulator_parts:
        part_screen = [
            world_to_screen(
                *_rotate_translate((float(v[0]), float(v[1])), theta, px, py)
            )
            for v in part
        ]
        if len(part_screen) >= 3:
            pygame.draw.polygon(surf, MANIP_FILL_COLOR, part_screen)
            pygame.draw.polygon(surf, MANIP_OUTLINE_COLOR, part_screen, width=2)

    # Heading indicator: short line from centre out the front
    nose_local = (cfg.chassis_side * 0.55, 0.0)
    nose_world = _rotate_translate(nose_local, theta, px, py)
    pygame.draw.line(
        surf,
        ROBOT_OUTLINE_COLOR,
        world_to_screen(px, py),
        world_to_screen(*nose_world),
        2,
    )


def draw_ball(surf, ball):
    s = ball.state
    px, py = float(s[0]), float(s[1])
    radius_px = m_to_px(ball.config.dynamics_params.radius)
    pos = world_to_screen(px, py)
    pygame.draw.circle(surf, BALL_FILL_COLOR, pos, radius_px)
    pygame.draw.circle(surf, BALL_OUTLINE_COLOR, pos, radius_px, width=2)


def draw_hud(surf, font, robot, ball, cmd):
    s = robot.state
    b = ball.state
    lines = [
        f"robot: x={float(s[0]):+.3f}  y={float(s[1]):+.3f}  θ={float(s[2]):+.3f}    v={float(s[3]):+.3f}  ω={float(s[4]):+.3f}",
        f"ball:  x={float(b[0]):+.3f}  y={float(b[1]):+.3f}                      vx={float(b[2]):+.3f}  vy={float(b[3]):+.3f}",
        f"cmd:   v_left={cmd[0]:+.2f}   v_right={cmd[1]:+.2f}",
        "",
        "WASD/↑↓←→ = drive   R = reset   ESC/Q = quit",
    ]
    for i, line in enumerate(lines):
        surf.blit(font.render(line, True, HUD_COLOR), (12, 10 + i * 16))


# --- Config loading ---------------------------------------------------------


def load_robot_config(name: str) -> sim_py.RobotConfig:
    cfg_dir = _atomsim / "sim" / "configs"
    data = json.loads((cfg_dir / "robots" / f"{name}.json").read_text())
    cfg = sim_py.RobotConfig()
    cfg.chassis_side = float(data.get("chassis_side", 0.10))
    if "manipulator" in data:
        m_data = json.loads(
            (cfg_dir / "manipulators" / f"{data['manipulator']}.json").read_text()
        )
        cfg.manipulator_parts = [
            [(float(v[0]), float(v[1])) for v in part] for part in m_data["parts"]
        ]
    return cfg


# --- Main loop --------------------------------------------------------------


def main() -> None:
    pygame.init()
    pygame.display.set_caption("AtomSim teleop — WASD/arrows to drive, ESC to quit")
    surf = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    # --- sim setup ---
    robot_cfg = load_robot_config("diff_drive_sidewall")
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

    # --- control mapping ---
    MAX_WHEEL_SPEED = 0.225  # m/s — caps wheel rim velocity at full key press
    TURN_RATE_K = 0.4  # turn rate (rad/s) per unit of turn input; lower → more forward motion when turning
    DT = 1.0 / 60.0

    def reset() -> None:
        rs = np.array(
            [robot_cfg.x0, robot_cfg.y0, robot_cfg.theta0, 0.0, 0.0], dtype=np.float32
        )
        bs = np.array([ball_cfg.x0, ball_cfg.y0, 0.0, 0.0], dtype=np.float32)
        robot.set_state(rs)
        ball.set_state(bs)

    running = True
    while running:
        # --- events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    reset()

        # --- keyboard → wheel command ---
        keys = pygame.key.get_pressed()
        forward = (keys[pygame.K_w] or keys[pygame.K_UP]) - (
            keys[pygame.K_s] or keys[pygame.K_DOWN]
        )
        turn = (keys[pygame.K_a] or keys[pygame.K_LEFT]) - (
            keys[pygame.K_d] or keys[pygame.K_RIGHT]
        )
        v_left = MAX_WHEEL_SPEED * (forward - turn * TURN_RATE_K)
        v_right = MAX_WHEEL_SPEED * (forward + turn * TURN_RATE_K)
        cmd = np.array([v_left, v_right], dtype=np.float32)

        # --- step ---
        robot.pre_step(cmd, DT)
        ball.pre_step(DT)
        world.step(DT)
        robot.post_step()
        ball.post_step()

        # --- render ---
        surf.fill(BG_COLOR)
        draw_field(surf, world)
        draw_ball(surf, ball)
        draw_robot(surf, robot)
        draw_hud(surf, font, robot, ball, cmd)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
