"""Level 1: solo robot in the field with a ball, two goals, no opponent.

Action and observation conventions match `AtomGym/action_observation.py` —
the policy emits normalized (V, Ω); the env converts via anti-windup mixing.

Reward is plug-in via a `RewardComposite` (from `AtomGym.rewards`). If none
is supplied, every step's reward is 0.0 — the env still simulates correctly,
just with no learning signal. The env detects ball-in-goal events (edge
detection on the chamber bounds) and surfaces them as `info["scored_for_us"]`
/ `info["scored_against_us"]` for reward terms to pick up. Goals also
trigger `terminated=True` so episodes end as soon as the game is decided.

`AtomSoloEnv` intentionally exposes its sim handles (`world`, `robot`,
`ball`) so the visualization tools can build a SceneSpec from a live env
without going through the .npz format. The +x goal is "ours to attack";
the -x goal is "ours to defend".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# --- locate the AtomSim release build and import sim_py --------------------


def _find_atomsim_build_dir() -> Path:
    """Walk upward from this file looking for AtomSim/build/release/sim/bindings.
    Lets the env be imported from anywhere as long as AtomSim sits at the same
    repo root as AtomGym (the standard layout)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "AtomSim" / "build" / "release" / "sim" / "bindings"
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "Could not locate AtomSim release build. Run "
        "`cmake --preset release && cmake --build build/release` from AtomSim/."
    )


sys.path.insert(0, str(_find_atomsim_build_dir()))

import sim_py  # noqa: E402

from AtomGym.action_observation import (  # noqa: E402
    OMEGA_MAX_DEFAULT,
    TRACK_WIDTH_DEFAULT,
    V_MAX_DEFAULT,
    ActionView,
    ObsView,
    action_to_wheel_cmds,
    build_observation,
)
from AtomGym.rewards import RewardComposite, RewardContext  # noqa: E402


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class AtomSoloEnv(gym.Env):
    """Single robot, single ball, two goals on the field. No opponent.

    Action space: Box(low=-1, high=+1, shape=(2,), dtype=float32)
        [V, Ω] — normalized forward thrust and yaw rate.

    Observation space: Box(low=-1, high=+1, shape=(11,), dtype=float32)
        Layout: [ball (4) | self (7)]. See action_observation.py.

    Episode termination is currently `truncated`-only at `max_episode_steps`;
    `terminated` is reserved for goal-scored events once the reward function
    is in place.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        reward: RewardComposite | None = None,
        dt: float = 1.0 / 60.0,
        max_episode_steps: int = 1500,  # 25 s at 60 Hz
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.max_episode_steps = max_episode_steps

        # Reward composite. None ⟹ zero reward every step (sim still runs
        # correctly; you just don't get a learning signal).
        self._reward = reward if reward is not None else RewardComposite()

        # Schema views: instantiate once, pass arrays into the methods. Exposed
        # as public attributes so reward functions / callbacks / tests can reuse
        # them without reconstructing.
        self.obs_view = ObsView(n_robots=1)
        self.action_view = ActionView()

        self.action_space = spaces.Box(
            low=-1.0, high=+1.0,
            shape=(self.action_view.total_dim,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=+1.0,
            shape=(self.obs_view.total_dim,), dtype=np.float32,
        )

        # Sim setup. Field + goals come from the default WorldConfig.
        self.world = sim_py.World()
        self.field_x_half = float(self.world.config.field_x_half)
        self.field_y_half = float(self.world.config.field_y_half)
        self.goal_y_half = float(self.world.config.goal_y_half)
        self.goal_extension = float(self.world.config.goal_extension)

        self._robot_cfg = self._make_robot_config()
        self.robot = sim_py.Robot(self.world, self._robot_cfg)

        self._ball_cfg = self._make_ball_config()
        self.ball = sim_py.Ball(self.world, self._ball_cfg)

        # Mutable cross-step state — the env owns it; reward terms read it
        # via `RewardContext`. See `_base_reward.py` for the contract.
        self._step_count = 0
        self._prev_obs: np.ndarray | None = None
        self._prev_action: np.ndarray | None = None
        # Edge-detection latches for goal events.
        self._ball_in_opp_goal_prev: bool = False
        self._ball_in_own_goal_prev: bool = False

        self._rng = np.random.default_rng(seed)

    # ---- gym API ----------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Randomize robot pose and ball position within a margin of the field
        # bounds so neither spawns inside a goal chamber or against a wall.
        margin = 0.04
        x_lim = self.field_x_half - margin
        y_lim = self.field_y_half - margin

        rx = float(self._rng.uniform(-x_lim, x_lim))
        ry = float(self._rng.uniform(-y_lim, y_lim))
        rt = float(self._rng.uniform(-np.pi, np.pi))

        bx = float(self._rng.uniform(-x_lim, x_lim))
        by = float(self._rng.uniform(-y_lim, y_lim))

        self.robot.set_state(np.array([rx, ry, rt, 0.0, 0.0], dtype=np.float32))
        self.ball.set_state(np.array([bx, by, 0.0, 0.0], dtype=np.float32))

        self._step_count = 0
        self._prev_obs = None
        self._prev_action = None
        self._ball_in_opp_goal_prev = False
        self._ball_in_own_goal_prev = False
        return self._build_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        v_norm = self.action_view.v(action)
        omega_norm = self.action_view.omega(action)
        v_left, v_right = action_to_wheel_cmds(
            v_norm,
            omega_norm,
            max_wheel_speed=V_MAX_DEFAULT,
            track_width=TRACK_WIDTH_DEFAULT,
        )
        wheel_cmd = np.array([v_left, v_right], dtype=np.float32)

        self.robot.pre_step(wheel_cmd, self.dt)
        self.ball.pre_step(self.dt)
        self.world.step(self.dt)
        self.robot.post_step()
        self.ball.post_step()

        self._step_count += 1

        obs = self._build_obs()
        info: dict[str, Any] = self._detect_events()

        # Build the per-step context bundle and run the reward composite.
        ctx = RewardContext(
            obs=obs,
            action=np.asarray(action, dtype=np.float32),
            prev_obs=self._prev_obs,
            prev_action=self._prev_action,
            info=info,
            obs_view=self.obs_view,
            action_view=self.action_view,
            field_x_half=self.field_x_half,
            field_y_half=self.field_y_half,
            goal_y_half=self.goal_y_half,
            goal_extension=self.goal_extension,
            dt=self.dt,
        )
        reward, breakdown = self._reward(ctx)
        info["reward_breakdown"] = breakdown

        # Episode ends as soon as a goal goes in (either side). The env owns
        # this decision, not the reward — termination is a game rule.
        terminated = bool(
            info.get("scored_for_us", False)
            or info.get("scored_against_us", False)
        )
        truncated = (self._step_count >= self.max_episode_steps) and not terminated

        # Roll prev-step state forward AFTER context is built.
        self._prev_obs = obs
        self._prev_action = ctx.action
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        # sim_py owns the underlying Box2D world; lifetime is tied to GC.
        pass

    # ---- event detection ------------------------------------------------

    def _detect_events(self) -> dict[str, Any]:
        """Edge-detect ball-in-goal-chamber transitions. Sets `scored_for_us`
        on the rising edge of "ball center inside +x goal chamber",
        `scored_against_us` on the rising edge of -x. Latches prevent the
        same goal from registering on every step the ball sits in the
        chamber."""
        bx = float(self.ball.state[0])
        by = float(self.ball.state[1])
        in_y_band = abs(by) <= self.goal_y_half

        in_opp = in_y_band and (bx > self.field_x_half)
        in_own = in_y_band and (bx < -self.field_x_half)

        info: dict[str, Any] = {
            "scored_for_us": in_opp and not self._ball_in_opp_goal_prev,
            "scored_against_us": in_own and not self._ball_in_own_goal_prev,
        }
        self._ball_in_opp_goal_prev = in_opp
        self._ball_in_own_goal_prev = in_own
        return info

    # ---- helpers ---------------------------------------------------------

    @property
    def t(self) -> float:
        """Sim time elapsed in the current episode, seconds."""
        return self._step_count * self.dt

    def _build_obs(self) -> np.ndarray:
        return build_observation(
            field_x_half=self.field_x_half,
            field_y_half=self.field_y_half,
            ball_state=np.asarray(self.ball.state, dtype=np.float32),
            self_state_5d=np.asarray(self.robot.state, dtype=np.float32),
            others_states_5d=(),
            mirror=False,
            v_max=V_MAX_DEFAULT,
            omega_max=OMEGA_MAX_DEFAULT,
        )

    @staticmethod
    def _make_robot_config() -> sim_py.RobotConfig:
        cfg = sim_py.RobotConfig()
        cfg.body_type = sim_py.BodyType.Dynamic
        cfg.chassis_side = 0.060
        cfg.mass = 0.3
        cfg.yaw_inertia = 5.0e-4
        cfg.dynamics_params.track_width = TRACK_WIDTH_DEFAULT
        cfg.dynamics_params.tau_motor = 0.05
        return cfg

    @staticmethod
    def _make_ball_config() -> sim_py.BallConfig:
        cfg = sim_py.BallConfig()
        cfg.field_k = 70.0
        cfg.dynamics_params.radius = 0.014
        cfg.dynamics_params.mass = 0.05
        cfg.dynamics_params.restitution = 0.4
        cfg.dynamics_params.damping = 0.8
        return cfg
