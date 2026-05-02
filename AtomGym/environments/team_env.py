"""Level 2: 1v1 — single learner + single opponent + ball, two goals.

The Gymnasium API exposes ONE policy: the learner. The opponent is driven
by a callable hook (`opponent_policy`) the env queries each control step.
Default hook ⟹ zero action (opponent stands still). The snapshot-pool
self-play machinery lives outside the env: it samples a snapshot, wraps
it as a callable, and assigns it via `set_opponent_policy()` at episode
boundaries.

Observation layout (learner's perspective, world-frame, 18 dims):

    [ball (4) | self/learner (7) | opp (7)]

Per CLAUDE.md, the canonical "my team in slots 0..N-1" view is achieved
by re-indexing this same vector for the opponent's perspective rather
than running a parallel env — that helper is the next step in the
roadmap. For now, this env builds the opponent's obs the simple way
(re-running `build_observation` from the opponent's perspective) so it
can be queried with the no-op default hook end-to-end.

Goal sides: learner attacks +x; opponent attacks -x. `info["scored_for_us"]`
fires when ball fully crosses +x line; `scored_against_us` for the -x line.
Both terminate the episode. Same semantics as `AtomSoloEnv`.

Reset placement: each robot is randomized within its OWN half of the
field (learner: x ∈ [-fx_half, 0], opponent: x ∈ [0, +fx_half]). This
makes overlap impossible by construction and matches a kickoff layout.
Ball is randomized anywhere on the field.

Reward terms run on the learner only — only the learner contributes
gradients. Per-step contact info comes from the learner's robot; the
existing `_OBSTACLE_CATEGORIES` mask already includes `CATEGORY_ROBOT`
so robot-robot contacts feed `obstacle_contact` rewards naturally.
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# AtomSim build path is added to sys.path when solo_env is imported.
# Importing solo_env first ensures sim_py is on sys.path before we use it.
from AtomGym.environments.solo_env import (
    ContactRecord,
    _OBSTACLE_CATEGORIES,
    AtomSoloEnv,  # for _make_robot_config / _make_ball_config reuse
)
from AtomGym.action_observation import (
    BALL_BLOCK_DIM,
    OMEGA_MAX_DEFAULT,
    ROBOT_BLOCK_DIM,
    TRACK_WIDTH_DEFAULT,
    V_MAX_DEFAULT,
    ActionView,
    ObsView,
    action_to_wheel_cmds,
    build_observation,
)
from AtomGym.environments.initial_state import InitialStateRanges
from AtomGym.rewards import RewardContext, RewardTerm

import sim_py  # noqa: E402  — sys.path was set up by solo_env import above


# Sign-flip masks for the canonical-view (mirror across y-axis) transform.
# A reflection across the y-axis flips x-coordinate components and quantities
# that change sign under reflection (cos θ, ω). y-coordinates, sin θ, and
# y-velocities are unchanged. These masks exactly mirror the per-element
# sign flips that `_encode_*` apply when called with `mirror=True`, but here
# we apply them to an already-encoded obs vector — so we don't need to
# rebuild from raw sim state.
_BALL_MIRROR_MASK = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)
# Robot block layout: [px, py, sin θ, cos θ, dx, dy, ω]
_ROBOT_MIRROR_MASK = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0], dtype=np.float32)


# Type alias for the opponent policy hook. Takes the opponent's observation
# (shape (18,), float32) and returns an action (shape (2,), float32, in
# [-1, 1]). Returning zeros = "stand still". The hook is allowed to be
# stateful (e.g. a frozen torch module wrapped in a closure).
OpponentPolicy = Callable[[np.ndarray], np.ndarray]


def _zero_opponent_policy(_obs: np.ndarray) -> np.ndarray:
    """Default opponent: emit zero action every step (stationary)."""
    return np.zeros(2, dtype=np.float32)


class AtomTeamEnv(gym.Env):
    """1v1 self-play env. Single learner controls one robot; the other is
    driven by a swappable `opponent_policy` callable.

    Action space: Box(low=-1, high=+1, shape=(2,), dtype=float32)
        Learner's [V, Ω].
    Observation space: Box(low=-1, high=+1, shape=(18,), dtype=float32)
        Learner's view: [ball (4) | self (7) | opp (7)].
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        rewards: list[RewardTerm] | None = None,
        init_state_ranges: InitialStateRanges | None = None,
        physics_dt: float = 1.0 / 60.0,
        control_dt: float | None = None,
        max_episode_steps: int = 800,
        seed: int | None = None,
        opponent_policy: OpponentPolicy | None = None,
        manipulator: str | None = None,
    ) -> None:
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

        self.reward_terms: list[RewardTerm] = (
            list(rewards) if rewards is not None else []
        )
        self.init_state_ranges: InitialStateRanges = (
            init_state_ranges if init_state_ranges is not None else InitialStateRanges()
        )

        # Two robots → ObsView(n_robots=2). Total dim = 4 + 7·2 = 18.
        self.obs_view = ObsView(n_robots=2)
        self.action_view = ActionView()

        self.action_space = spaces.Box(
            low=-1.0, high=+1.0,
            shape=(self.action_view.total_dim,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=+1.0,
            shape=(self.obs_view.total_dim,), dtype=np.float32,
        )

        # World + bodies. Reuse solo's robot/ball config builders verbatim
        # — both robots use the same physical config.
        self.world = sim_py.World()
        self.field_x_half = float(self.world.config.field_x_half)
        self.field_y_half = float(self.world.config.field_y_half)
        self.goal_y_half = float(self.world.config.goal_y_half)
        self.goal_extension = float(self.world.config.goal_extension)

        self._robot_cfg = AtomSoloEnv._make_robot_config(manipulator=manipulator)
        self.robot = sim_py.Robot(self.world, self._robot_cfg)
        self.opponent = sim_py.Robot(self.world, self._robot_cfg)

        self._ball_cfg = AtomSoloEnv._make_ball_config()
        self.ball = sim_py.Ball(self.world, self._ball_cfg)

        # Opponent action source. Public-ish via set_opponent_policy(); the
        # snapshot-pool sampler will reassign this at episode boundaries.
        self._opponent_policy: OpponentPolicy = (
            opponent_policy if opponent_policy is not None else _zero_opponent_policy
        )

        self._step_count = 0
        self._prev_obs: np.ndarray | None = None
        self._prev_action: np.ndarray | None = None
        self._ball_in_opp_goal_prev: bool = False
        self._ball_in_own_goal_prev: bool = False
        # Last action emitted by the opponent's policy (canonical-frame
        # (V, Ω) in [-1, +1]). Public attribute so external observers
        # (e.g. GIF eval callback) can render the opponent's control
        # indicator instead of hardcoding zeros. None before the first
        # step of an episode; reset on every reset().
        self.last_opponent_action: np.ndarray | None = None

        self._rng = np.random.default_rng(seed)

    # ---- opponent hook ---------------------------------------------------

    def set_opponent_policy(self, policy: OpponentPolicy | None) -> None:
        """Swap the opponent's policy. None ⟹ zero-action default. Safe to
        call between episodes (e.g. on `reset()`); behavior mid-episode is
        defined but the snapshot-pool sampler should only swap on reset."""
        self._opponent_policy = (
            policy if policy is not None else _zero_opponent_policy
        )

    # ---- gym API ---------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        r = self.init_state_ranges

        # Each robot on its own half of the field. Learner attacks +x, so it
        # spawns in the -x half; opponent attacks -x, spawns in +x half.
        # Margin pulls in from each wall (and from the centre line so the
        # robots can't start kissing centre).
        rx_lim = max(0.0, self.field_x_half - r.robot_xy_margin)
        ry_lim = max(0.0, self.field_y_half - r.robot_xy_margin)

        # Learner: x in [-rx_lim, -margin], y in [-ry_lim, +ry_lim]
        learner_x = -float(self._rng.uniform(r.robot_xy_margin, rx_lim)) \
            if rx_lim > r.robot_xy_margin else 0.0
        learner_y = float(self._rng.uniform(-ry_lim, ry_lim)) if ry_lim > 0 else 0.0
        learner_th = float(self._rng.uniform(*r.robot_theta))
        learner_v = float(self._rng.uniform(*r.robot_speed))
        learner_w = float(self._rng.uniform(*r.robot_omega))

        # Opponent: x in [+margin, +rx_lim]
        opp_x = float(self._rng.uniform(r.robot_xy_margin, rx_lim)) \
            if rx_lim > r.robot_xy_margin else 0.0
        opp_y = float(self._rng.uniform(-ry_lim, ry_lim)) if ry_lim > 0 else 0.0
        opp_th = float(self._rng.uniform(*r.robot_theta))
        opp_v = float(self._rng.uniform(*r.robot_speed))
        opp_w = float(self._rng.uniform(*r.robot_omega))

        # Ball anywhere on field with margin (matches solo).
        bx_lim = max(0.0, self.field_x_half - r.ball_xy_margin)
        by_lim = max(0.0, self.field_y_half - r.ball_xy_margin)
        bx = float(self._rng.uniform(-bx_lim, bx_lim)) if bx_lim > 0 else 0.0
        by = float(self._rng.uniform(-by_lim, by_lim)) if by_lim > 0 else 0.0
        bspeed = float(self._rng.uniform(*r.ball_speed))
        bdir = float(self._rng.uniform(*r.ball_direction))
        bvx = bspeed * float(np.cos(bdir))
        bvy = bspeed * float(np.sin(bdir))

        self.robot.set_state(
            np.array([learner_x, learner_y, learner_th, learner_v, learner_w], dtype=np.float32)
        )
        self.opponent.set_state(
            np.array([opp_x, opp_y, opp_th, opp_v, opp_w], dtype=np.float32)
        )
        self.ball.set_state(np.array([bx, by, bvx, bvy], dtype=np.float32))

        self._step_count = 0
        self._prev_obs = None
        self._prev_action = None
        self._ball_in_opp_goal_prev = False
        self._ball_in_own_goal_prev = False
        self.last_opponent_action = None
        return self._build_learner_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # --- decode learner action --------------------------------------
        v_norm = self.action_view.v(action)
        omega_norm = self.action_view.omega(action)
        vL_l, vR_l = action_to_wheel_cmds(
            v_norm, omega_norm,
            max_wheel_speed=V_MAX_DEFAULT, track_width=TRACK_WIDTH_DEFAULT,
        )
        learner_wheel = np.array([vL_l, vR_l], dtype=np.float32)

        # --- query opponent policy (once per control step, like learner) -
        # The opponent's policy was trained as a learner — i.e. expects the
        # canonical "I attack +x, I am in slot 0" view. We get there via
        # `opponent_view`, which permutes slots AND mirrors x-components so
        # the opponent's goal sits where its policy expects. The action it
        # returns is in that canonical frame, so we apply with `mirror=True`
        # below — this negates Ω on application (CCW-in-canonical maps to
        # CW-in-world). V is body-frame longitudinal and is mirror-invariant.
        opp_obs = self.opponent_view(self._build_learner_obs())
        opp_action = self._opponent_policy(opp_obs)
        opp_action = np.asarray(opp_action, dtype=np.float32).reshape(-1)
        if opp_action.shape != (2,):
            raise ValueError(
                f"opponent_policy returned action with shape {opp_action.shape}; "
                f"expected (2,)"
            )
        # Stash before mirroring so external readers (GIF eval) see the
        # canonical-frame (V, Ω) — that's what the policy emitted, and
        # the team team's control panel renders raw policy output for
        # both robots symmetrically.
        self.last_opponent_action = opp_action.copy()
        vL_o, vR_o = action_to_wheel_cmds(
            float(opp_action[0]), float(opp_action[1]),
            max_wheel_speed=V_MAX_DEFAULT, track_width=TRACK_WIDTH_DEFAULT,
            mirror=True,
        )
        opp_wheel = np.array([vL_o, vR_o], dtype=np.float32)

        # --- substep loop -----------------------------------------------
        accumulated: dict[str, Any] = {
            "scored_for_us": False,
            "scored_against_us": False,
        }
        all_contacts: list[Any] = []  # learner's contact list — opponent's is ignored
        obstacle_contact_substeps = 0
        n_substeps_run = 0
        for _ in range(self.action_repeat):
            self.robot.pre_step(learner_wheel, self.physics_dt)
            self.opponent.pre_step(opp_wheel, self.physics_dt)
            self.ball.pre_step(self.physics_dt)
            self.world.step(self.physics_dt)
            self.robot.post_step()
            self.opponent.post_step()
            self.ball.post_step()
            n_substeps_run += 1

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

        self._step_count += 1

        obs = self._build_learner_obs()
        info: dict[str, Any] = accumulated
        info["robot_contacts"] = all_contacts
        info["obstacle_contact_frac"] = (
            obstacle_contact_substeps / n_substeps_run if n_substeps_run > 0 else 0.0
        )

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

        terminated = bool(
            info.get("scored_for_us", False)
            or info.get("scored_against_us", False)
        )
        truncated = (self._step_count >= self.max_episode_steps) and not terminated

        self._prev_obs = obs
        self._prev_action = ctx.action
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        pass

    # ---- reward (identical to solo) --------------------------------------

    def compute_reward(
        self, ctx: RewardContext
    ) -> tuple[float, dict[str, float]]:
        breakdown: dict[str, float] = {}
        for term in self.reward_terms:
            contribution = term.weight * term(ctx)
            breakdown[term.name] = breakdown.get(term.name, 0.0) + contribution
        return float(sum(breakdown.values())), breakdown

    # ---- event detection (identical to solo) ----------------------------

    def _detect_events(self) -> dict[str, Any]:
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

    # ---- observation builders -------------------------------------------

    def _build_learner_obs(self) -> np.ndarray:
        """Learner's view: [ball | learner | opp]. World-frame, no mirror."""
        return build_observation(
            field_x_half=self.field_x_half,
            field_y_half=self.field_y_half,
            ball_state=np.asarray(self.ball.state, dtype=np.float32),
            self_state_5d=np.asarray(self.robot.state, dtype=np.float32),
            others_states_5d=(np.asarray(self.opponent.state, dtype=np.float32),),
            mirror=False,
            v_max=V_MAX_DEFAULT,
            omega_max=OMEGA_MAX_DEFAULT,
        )

    def opponent_view(self, learner_obs: np.ndarray) -> np.ndarray:
        """Convert a learner-perspective obs into the opponent's canonical
        view: "I attack +x, I am in slot 0".

        Two transforms applied to the (18,) learner obs:
          1. **Slot swap** — the opp's robot block (slot 1) moves to the
             "self" slot (slot 0); the learner's block moves to "other".
          2. **X-axis mirror** — flip ball.px, ball.vx; flip robot.px,
             cos θ, dx, ω in each block. This puts the opponent's goal at
             +x in the obs (where the policy expects its own goal to be).

        Returns a NEW array; `learner_obs` is not mutated. World-frame obs
        makes this safe — the only operations needed are a slot copy and
        elementwise sign flips. No re-encoding, no rebuilt sim queries.

        This is `mathematically equivalent` to calling
        `build_observation(self_state_5d=opponent, others_states_5d=[learner],
        mirror=True)` directly from sim state, but cheaper because it
        skips the trig + clip-norm work — the opponent's network sees the
        same vector either way.

        For 1v1 only. Generalising to NvN requires choosing a slot
        permutation rule, which is a different design conversation.
        """
        if learner_obs.shape != (self.obs_view.total_dim,):
            raise ValueError(
                f"opponent_view: expected obs of shape ({self.obs_view.total_dim},), "
                f"got {learner_obs.shape}"
            )
        out = np.empty_like(learner_obs)
        # Block index ranges. self block starts at BALL_BLOCK_DIM; other_0
        # block starts immediately after (BALL_BLOCK_DIM + ROBOT_BLOCK_DIM).
        self_start = BALL_BLOCK_DIM
        other_start = BALL_BLOCK_DIM + ROBOT_BLOCK_DIM
        block_end = other_start + ROBOT_BLOCK_DIM

        # Ball: same slot, mirror x components.
        out[0:BALL_BLOCK_DIM] = learner_obs[0:BALL_BLOCK_DIM] * _BALL_MIRROR_MASK
        # Opp (was other in learner view) → self in opponent view, mirrored.
        out[self_start:other_start] = (
            learner_obs[other_start:block_end] * _ROBOT_MIRROR_MASK
        )
        # Learner (was self in learner view) → other in opponent view, mirrored.
        out[other_start:block_end] = (
            learner_obs[self_start:other_start] * _ROBOT_MIRROR_MASK
        )
        return out

    # ---- helpers --------------------------------------------------------

    @property
    def t(self) -> float:
        return self._step_count * self.control_dt

    @property
    def ball_radius(self) -> float:
        return float(self._ball_cfg.dynamics_params.radius)
