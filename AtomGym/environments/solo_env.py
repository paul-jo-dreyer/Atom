"""Level 1: solo robot in the field with a ball, two goals, no opponent.

Action and observation conventions match `AtomGym/action_observation.py` —
the policy emits normalized (V, Ω); the env converts via anti-windup mixing.

Control vs physics rate
-----------------------
The sim integrator runs at `physics_dt`; the policy emits an action at
`control_dt`. They can differ — `control_dt` must be an integer multiple
of `physics_dt`, and that ratio is the action-repeat count. One call to
`step()` holds the same action constant and ticks the physics
`action_repeat` times, then returns one observation. Default is 1 — both
rates equal — preserving the simple "one action ⟹ one physics step" loop.

Goal scoring
------------
A goal is detected when the ENTIRE ball is past a goal line in x AND in
the goal-mouth y-band — the same rule used in real soccer (the ball must
fully cross the line). Edge-detected: `info["scored_for_us"]` /
`info["scored_against_us"]` fires only on the rising edge.
`terminated=True` on either event. Episodes truncate at
`max_episode_steps` control steps (default 800).

Rewards are plug-in via a list of `RewardTerm`s passed at construction.
The env's `compute_reward(ctx)` iterates over them, weights, sums, and
returns (total, breakdown). If no rewards are supplied, every step
returns 0.0 — the sim still runs correctly, just with no learning signal.
Reward is computed once per `step()` (i.e. per control tick), with
`ctx.dt = control_dt` so reward terms can reason about the time elapsed
between observations.

`AtomSoloEnv` intentionally exposes its sim handles (`world`, `robot`,
`ball`) so the visualization tools can build a SceneSpec from a live env
without going through the .npz format. The +x goal is "ours to attack";
the -x goal is "ours to defend".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, NamedTuple

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
from AtomGym.environments.initial_state import InitialStateRanges  # noqa: E402
from AtomGym.rewards import RewardContext, RewardTerm  # noqa: E402


# "Obstacle" = anything that isn't the ball. Walls, goal-walls, and (in
# multi-robot envs) other robots are all hostile contacts that should be
# penalised. Reward terms read `info["robot_contacts"]` and mask each
# contact's `other_category` against this; the env also pre-aggregates the
# fraction-in-contact signal so simple terms don't have to walk the list.
_OBSTACLE_CATEGORIES = (
    sim_py.CATEGORY_WALL | sim_py.CATEGORY_GOAL_WALL | sim_py.CATEGORY_ROBOT
)


class ContactRecord(NamedTuple):
    """Pickleable mirror of `sim_py.RobotContactPoint`. Pybind11 classes
    aren't pickle-friendly, but SB3's DummyVecEnv `deepcopy`s every info
    dict and SubprocVecEnv pickles them for IPC, so we copy the C++ struct
    into a plain Python NamedTuple before pushing it through `info`. Field
    names match the C++ struct so reward code reads either type the same
    way."""

    other_category: int
    point_x: float
    point_y: float
    normal_x: float
    normal_y: float
    normal_impulse: float
    tangent_impulse: float
    separation: float


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
        rewards: list[RewardTerm] | None = None,
        init_state_ranges: InitialStateRanges | None = None,
        physics_dt: float = 1.0 / 60.0,
        control_dt: float | None = None,
        max_episode_steps: int = 800,  # ≈13.3 s at 60 Hz control
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        physics_dt
            Sim integrator timestep, seconds.
        control_dt
            Time between successive observations / actions, seconds. Must
            be a positive integer multiple of `physics_dt`. `None` ⟹ same
            as `physics_dt` (no action repeat).
        max_episode_steps
            Truncation limit, in CONTROL steps (i.e. calls to `step()`).
        """
        super().__init__()
        if physics_dt <= 0.0:
            raise ValueError(f"physics_dt must be > 0, got {physics_dt}")
        if control_dt is None:
            control_dt = physics_dt
        ratio = control_dt / physics_dt
        action_repeat = int(round(ratio))
        if action_repeat < 1 or abs(action_repeat - ratio) > 1e-6:
            raise ValueError(
                f"control_dt ({control_dt}) must be a positive integer multiple "
                f"of physics_dt ({physics_dt}); got ratio {ratio}"
            )
        self.physics_dt = physics_dt
        self.control_dt = control_dt
        self.action_repeat = action_repeat
        self.max_episode_steps = max_episode_steps

        # Reward terms applied each step. Public list — mutate freely
        # (e.g. for curriculum stage changes mid-training); `compute_reward`
        # walks this list on each call. Empty list ⟹ zero reward (sim
        # still runs correctly; you just don't get a learning signal).
        self.reward_terms: list[RewardTerm] = (
            list(rewards) if rewards is not None else []
        )

        # Per-reset domain-randomization ranges. Public — reassign to swap
        # DR config mid-training (curriculum stage transitions). Defaults
        # produce random pose with zero initial velocities.
        self.init_state_ranges: InitialStateRanges = (
            init_state_ranges if init_state_ranges is not None else InitialStateRanges()
        )

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

        r = self.init_state_ranges

        # Robot pose: uniform inside the field with a margin on each side.
        rx_lim = max(0.0, self.field_x_half - r.robot_xy_margin)
        ry_lim = max(0.0, self.field_y_half - r.robot_xy_margin)
        rx = float(self._rng.uniform(-rx_lim, rx_lim)) if rx_lim > 0 else 0.0
        ry = float(self._rng.uniform(-ry_lim, ry_lim)) if ry_lim > 0 else 0.0
        rt = float(self._rng.uniform(*r.robot_theta))
        # Body-frame longitudinal velocity and yaw rate.
        rv = float(self._rng.uniform(*r.robot_speed))
        romega = float(self._rng.uniform(*r.robot_omega))

        # Ball position: same margin idea.
        bx_lim = max(0.0, self.field_x_half - r.ball_xy_margin)
        by_lim = max(0.0, self.field_y_half - r.ball_xy_margin)
        bx = float(self._rng.uniform(-bx_lim, bx_lim)) if bx_lim > 0 else 0.0
        by = float(self._rng.uniform(-by_lim, by_lim)) if by_lim > 0 else 0.0
        # Ball velocity in polar: speed × (cos θ, sin θ).
        bspeed = float(self._rng.uniform(*r.ball_speed))
        bdir = float(self._rng.uniform(*r.ball_direction))
        bvx = bspeed * float(np.cos(bdir))
        bvy = bspeed * float(np.sin(bdir))

        self.robot.set_state(np.array([rx, ry, rt, rv, romega], dtype=np.float32))
        self.ball.set_state(np.array([bx, by, bvx, bvy], dtype=np.float32))

        self._step_count = 0
        self._prev_obs = None
        self._prev_action = None
        self._ball_in_opp_goal_prev = False
        self._ball_in_own_goal_prev = False
        return self._build_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Decode the action ONCE per control step — the same wheel cmd is
        # held constant across all `action_repeat` physics substeps.
        v_norm = self.action_view.v(action)
        omega_norm = self.action_view.omega(action)
        v_left, v_right = action_to_wheel_cmds(
            v_norm,
            omega_norm,
            max_wheel_speed=V_MAX_DEFAULT,
            track_width=TRACK_WIDTH_DEFAULT,
        )
        wheel_cmd = np.array([v_left, v_right], dtype=np.float32)

        # Accumulate event flags across substeps. A goal scored on substep
        # k of N is OR'd in here and the loop early-exits — no point
        # ticking the sim further into a finished episode.
        accumulated: dict[str, Any] = {
            "scored_for_us": False,
            "scored_against_us": False,
        }
        # Contact accumulators: a flat list of every RobotContactPoint Box2D
        # reported across all substeps in this control step. Reward terms
        # that scale by impulse magnitude can sum directly over this list;
        # `obstacle_contact_substeps` powers the simple "fraction in
        # contact" signal without forcing every reward to re-walk the list.
        all_contacts: list[Any] = []
        obstacle_contact_substeps = 0
        n_substeps_run = 0
        for _ in range(self.action_repeat):
            self.robot.pre_step(wheel_cmd, self.physics_dt)
            self.ball.pre_step(self.physics_dt)
            self.world.step(self.physics_dt)
            self.robot.post_step()
            self.ball.post_step()
            n_substeps_run += 1

            # Snapshot contacts AFTER post_step. Box2D's contact data
            # reflects the impulse the solver just applied, so we get the
            # impulse for this substep specifically. Across the action_
            # repeat loop, list grows; track whether each substep had at
            # least one obstacle contact so we can return a fraction.
            #
            # Convert the C++ structs to ContactRecord NamedTuples here
            # because SB3's vec envs deepcopy/pickle info dicts and
            # pybind11 classes don't play with either.
            substep_contacts = self.robot.contact_points()
            if substep_contacts:
                substep_had_obstacle = False
                for c in substep_contacts:
                    all_contacts.append(ContactRecord(
                        c.other_category,
                        c.point_x, c.point_y,
                        c.normal_x, c.normal_y,
                        c.normal_impulse, c.tangent_impulse, c.separation,
                    ))
                    if c.other_category & _OBSTACLE_CATEGORIES:
                        substep_had_obstacle = True
                if substep_had_obstacle:
                    obstacle_contact_substeps += 1

            substep_info = self._detect_events()
            terminal_this_substep = False
            for key in ("scored_for_us", "scored_against_us"):
                if substep_info.get(key, False):
                    accumulated[key] = True
                    terminal_this_substep = True
            if terminal_this_substep:
                break

        self._step_count += 1  # one CONTROL step (not physics step).

        obs = self._build_obs()
        info: dict[str, Any] = accumulated
        # Divide by substeps actually run (early-exit on goal can leave
        # n_substeps_run < action_repeat).
        info["robot_contacts"] = all_contacts
        info["obstacle_contact_frac"] = (
            obstacle_contact_substeps / n_substeps_run if n_substeps_run > 0 else 0.0
        )

        # Reward computed once per control step. ctx.dt = control_dt so
        # reward terms see the time elapsed between observations, not the
        # finer physics tick.
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
            dt=self.control_dt,
        )
        reward, breakdown = self.compute_reward(ctx)
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

    # ---- reward ----------------------------------------------------------

    def compute_reward(
        self, ctx: RewardContext
    ) -> tuple[float, dict[str, float]]:
        """Walk `self.reward_terms`, weight each term's contribution, return
        the total scalar reward AND the per-term breakdown dict.

        The breakdown is keyed by term name and holds the WEIGHTED value
        (i.e. `term.weight * term(ctx)`) — that's what makes its way into
        `info["reward_breakdown"]` for TensorBoard logging. If two terms
        share a name, their contributions are accumulated under that one
        key (the total stays correct; the breakdown groups duplicates).

        Pure with respect to the env: doesn't mutate `self`. Reward terms
        are themselves stateless (read from `ctx`), so this method can be
        called from the REPL or debug code with a hand-built context to
        evaluate a composition against a saved scene.
        """
        breakdown: dict[str, float] = {}
        for term in self.reward_terms:
            contribution = term.weight * term(ctx)
            breakdown[term.name] = breakdown.get(term.name, 0.0) + contribution
        return float(sum(breakdown.values())), breakdown

    # ---- event detection ------------------------------------------------

    def _detect_events(self) -> dict[str, Any]:
        """Edge-detect ball-fully-past-goal-line transitions. The ball must
        FULLY cross the goal line — the entire ball, not just its centre,
        past x = ±field_x_half — and be in the goal-mouth y-band. This
        matches real soccer's "ball wholly across the line" rule and
        avoids spurious triggers from a centre-just-touching-line state.

        Latches (`_ball_in_*_goal_prev`) prevent the same goal from firing
        on every substep the ball sits in the chamber."""
        bx = float(self.ball.state[0])
        by = float(self.ball.state[1])
        radius = self.ball_radius
        in_y_band = abs(by) <= self.goal_y_half

        in_opp = in_y_band and (bx - radius > self.field_x_half)
        in_own = in_y_band and (bx + radius < -self.field_x_half)

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
        return self._step_count * self.control_dt

    @property
    def ball_radius(self) -> float:
        return float(self._ball_cfg.dynamics_params.radius)

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
