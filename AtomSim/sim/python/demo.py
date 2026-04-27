"""
Demo: drive a dynamic diff-drive robot at a stationary ball, watch the bounce.

Usage (from anywhere):

    .venv/bin/python AtomSim/sim/python/demo.py

The script auto-locates the AtomSim build directory and prepends it to
sys.path before importing sim_py — no PYTHONPATH manipulation required.
Requires the release build:

    cmake --preset release && cmake --build build/release   # from AtomSim/

The demo:
    1. Loads the diff_drive_default robot config from sim/configs/robots/.
    2. Resolves the manipulator polygon reference.
    3. Drops a ball into the field with default ball params at the origin.
    4. Drives the robot forward at 0.5 m/s for 1.5 seconds.
    5. Saves a plot of the trajectories to sim/python/demo_trajectory.png.
"""

import json
import sys
from pathlib import Path

import numpy as np


def find_atomsim_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "CMakePresets.json").exists():
            return p
    raise RuntimeError("Could not locate AtomSim/ from this script's path.")


# Make sim_py importable without setting PYTHONPATH externally.
_atomsim = find_atomsim_root()
_build_dir = _atomsim / "build" / "release" / "sim" / "bindings"
if not _build_dir.exists():
    raise RuntimeError(
        f"No release build at {_build_dir}.\n"
        f"From {_atomsim}, run: cmake --preset release && cmake --build build/release"
    )
sys.path.insert(0, str(_build_dir))

import sim_py  # noqa: E402  (after sys.path mutation)


def load_manipulator_parts(name: str, configs_dir: Path) -> list:
    """Read sim/configs/manipulators/<name>.json and return parts as nested list."""
    path = configs_dir / "manipulators" / f"{name}.json"
    data = json.loads(path.read_text())
    return [[(float(v[0]), float(v[1])) for v in part] for part in data["parts"]]


def load_robot_config(name: str, configs_dir: Path) -> sim_py.RobotConfig:
    """Read sim/configs/robots/<name>.json and build a RobotConfig.

    Resolves the manipulator reference if present.
    """
    path = configs_dir / "robots" / f"{name}.json"
    data = json.loads(path.read_text())
    cfg = sim_py.RobotConfig()
    cfg.chassis_side = float(data.get("chassis_side", 0.10))
    if "manipulator" in data:
        cfg.manipulator_parts = load_manipulator_parts(data["manipulator"], configs_dir)
    return cfg


def main() -> None:
    atomsim = _atomsim
    configs_dir = atomsim / "sim" / "configs"

    # Robot — dynamic so it can physically push the ball
    robot_cfg = load_robot_config("diff_drive_default", configs_dir)
    robot_cfg.body_type   = sim_py.BodyType.Dynamic
    robot_cfg.mass        = 0.3
    robot_cfg.yaw_inertia = 5e-4
    robot_cfg.x0          = -0.20
    robot_cfg.y0          =  0.00
    robot_cfg.theta0      =  0.00
    robot_cfg.dynamics_params.track_width = 0.10
    robot_cfg.dynamics_params.tau_motor   = 0.05

    # Ball — sitting in the path of the robot
    ball_cfg = sim_py.BallConfig()
    ball_cfg.x0 = 0.00
    ball_cfg.y0 = 0.00
    ball_cfg.field_k = 50.0
    ball_cfg.dynamics_params.radius      = 0.025
    ball_cfg.dynamics_params.mass        = 0.05
    ball_cfg.dynamics_params.restitution = 0.6
    ball_cfg.dynamics_params.damping     = 1.0

    world = sim_py.World()
    robot = sim_py.Robot(world, robot_cfg)
    ball  = sim_py.Ball(world, ball_cfg)

    # Drive forward for 1.5 s
    cmd = np.array([0.5, 0.5], dtype=np.float32)
    dt  = 0.01
    n_steps = 150

    robot_traj = np.zeros((n_steps, 5), dtype=np.float32)
    ball_traj  = np.zeros((n_steps, 4), dtype=np.float32)

    for i in range(n_steps):
        robot.pre_step(cmd, dt)
        ball.pre_step(dt)
        world.step(dt)
        robot.post_step()
        ball.post_step()
        robot_traj[i] = robot.state
        ball_traj[i]  = ball.state

    print(f"Final robot pose:  px={robot_traj[-1, 0]:+.4f}  py={robot_traj[-1, 1]:+.4f}  theta={robot_traj[-1, 2]:+.4f}")
    print(f"Final robot v,ω:   v ={robot_traj[-1, 3]:+.4f}  ω ={robot_traj[-1, 4]:+.4f}")
    print(f"Final ball pose:   px={ball_traj[-1, 0]:+.4f}  py={ball_traj[-1, 1]:+.4f}")
    print(f"Final ball vel:    vx={ball_traj[-1, 2]:+.4f}  vy={ball_traj[-1, 3]:+.4f}")

    # Find when the ball first started moving — that's the bounce moment
    moving = np.where(np.abs(ball_traj[:, 2]) > 1e-3)[0]
    if len(moving):
        i_bounce = int(moving[0])
        print(f"Ball started moving at step {i_bounce} (t = {i_bounce*dt:.3f}s):  "
              f"vx={ball_traj[i_bounce, 2]:+.4f}")
    else:
        print("Ball never moved — the robot didn't reach it.")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        fig, ax = plt.subplots(figsize=(10, 6))
        xh, yh = world.config.field_x_half, world.config.field_y_half
        ax.add_patch(Rectangle((-xh, -yh), 2*xh, 2*yh, fill=False, edgecolor="black", linewidth=1.5))
        ax.plot(robot_traj[:, 0], robot_traj[:, 1], "b-", linewidth=2, label="robot path")
        ax.plot(ball_traj[:, 0],  ball_traj[:, 1],  "r-", linewidth=2, label="ball path")
        ax.add_patch(Circle((ball_cfg.x0, ball_cfg.y0), ball_cfg.dynamics_params.radius,
                            fill=False, edgecolor="red", linestyle="--", alpha=0.6, label="ball start"))
        ax.add_patch(Circle((ball_traj[-1, 0], ball_traj[-1, 1]), ball_cfg.dynamics_params.radius,
                            fill=True, facecolor="red", alpha=0.4, label="ball end"))
        ax.scatter([robot_cfg.x0], [robot_cfg.y0], c="blue", marker="s", s=100, alpha=0.4, label="robot start")
        ax.scatter([robot_traj[-1, 0]], [robot_traj[-1, 1]], c="blue", marker="s", s=100, label="robot end")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Phase 4 demo: dynamic robot pushes ball")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        out = Path(__file__).parent / "demo_trajectory.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"\nSaved trajectory plot to {out.relative_to(atomsim)}")
    except ImportError:
        print("\n(matplotlib not available — skipped plot)")


if __name__ == "__main__":
    main()
