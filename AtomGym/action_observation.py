"""Action and observation encoding — the single source of truth.

Every env, every reward function, every BC dataset preprocessor goes through
this module. Constants live here; encoding/decoding helpers live here;
indexing layout for the observation block lives here.

Conventions
-----------

**Action** is `(V, Ω) ∈ [-1, 1]²`:
- V is normalized forward thrust (+1 = full forward at v_max).
- Ω is normalized yaw rate (+1 = full CCW spin at ω_max = 2 v_max / track_width).

`action_to_wheel_cmds` maps to `(v_left, v_right)` m/s using anti-windup
scaling: if either wheel would exceed v_max, BOTH wheels are scaled
proportionally so the V/Ω ratio is preserved. Smooth, no clipping
discontinuities.

**Observation** layout (all entries in [-1, 1] after normalization):

    [ ball (4) | self (8) | other_0 (8) | other_1 (8) | ... ]

- ball:  (px, py, vx, vy)
- robot: (px, py, sin θ, cos θ, dx, dy, dθ, time_in_box) — sin/cos to avoid
  the wraparound discontinuity at θ=0/2π. dx, dy are world-frame velocities
  (computed from the body-frame v in the sim's 5D state). dθ = ω.
  `time_in_box` is the fraction of `goalie_box_terminal_time` the robot has
  spent continuously inside ITS OPPOSING goalie box this visit. 0.0 = not
  in box / fresh entry; 1.0 = at the violation threshold. Resets to 0 on
  box exit. Always present in the obs (reads 0 when the rule is disabled).

Positions normalised by field half-extents; robot velocities by `V_MAX`
and `OMEGA_MAX`; ball velocities by `V_BALL_MAX`. Out-of-range values are
clipped to [-1, 1]. `time_in_box` is clipped to [0, 1] (it's
non-negative and capped at terminal by construction).

**Mirror** (for self-play): when one team naturally attacks +x and the
other attacks −x, the same policy plays both sides if we present a
canonical "I attack +x" view. Mirror flips x positions, x velocities,
cos θ, and dθ. Sin θ and y components are unchanged. The action's Ω is
then negated when applied to the actual robot (since "+CCW in policy
view" maps to CW in the unmirrored world).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------

# Robot motion limits — derived from the robot config used in the sim layer.
# `V_MAX` matches `MAX_WHEEL_SPEED` in teleop.py. `OMEGA_MAX` is the yaw rate
# achievable with one wheel at +v_max and the other at -v_max:
#     ω = (v_right - v_left) / track_width = 2 v_max / track_width
V_MAX_DEFAULT: float = 0.225
TRACK_WIDTH_DEFAULT: float = 0.060
OMEGA_MAX_DEFAULT: float = 2.0 * V_MAX_DEFAULT / TRACK_WIDTH_DEFAULT  # = 7.5 rad/s

# Ball velocity normalization — the ball can be struck and exceed the robot's
# top speed. 5 m/s is the agreed theoretical max; faster values are clipped
# to ±1 in the obs (the sim doesn't enforce this — only the obs encoding).
V_BALL_MAX: float = 5.0

# Block dimensions in the observation vector.
BALL_BLOCK_DIM: int = 4
ROBOT_BLOCK_DIM: int = 8


# ---------------------------------------------------------------------------
# Heading <-> sin/cos helpers
# ---------------------------------------------------------------------------
#
# The obs encodes heading as (sin θ, cos θ) — periodic, no wraparound cliff —
# but reward terms, tests, and analysis often want θ in radians. These two
# free helpers are the bridge in both directions; the view's `self_theta` /
# `other_theta` are thin wrappers around `theta_from_sincos`.


def theta_from_sincos(sin_th: float, cos_th: float) -> float:
    """Recover the heading angle (radians, in [-π, π]) from its (sin, cos)
    encoding. Inverse of `sincos_from_theta`."""
    return float(np.arctan2(sin_th, cos_th))


def sincos_from_theta(theta: float) -> tuple[float, float]:
    """Encode a heading angle as (sin θ, cos θ). Matches the obs encoding
    convention exactly — useful for building synthetic obs in tests, scripted
    policies, or demo episodes. Inverse of `theta_from_sincos`."""
    return float(np.sin(theta)), float(np.cos(theta))


# ---------------------------------------------------------------------------
# Schema views — instantiate once, pass arrays into the methods
# ---------------------------------------------------------------------------
#
# Both views are stateless: they hold no array reference. Construct one in
# the env at __init__ time, expose as `env.obs_view` / `env.action_view`,
# and pass observation / action arrays into the methods at read time. Same
# instance is safe to share across processes / vec env workers.
#
# Mutation semantics worth being explicit about:
#  * Scalar accessors return Python `float` — IMMUTABLE; rebinding the
#    returned value cannot affect the source array.
#  * Block accessors return numpy `ndarray` — these are VIEWS into the
#    source array. Writing to the returned slice WILL mutate the source.
#    Each such method's docstring restates this; call `.copy()` if the
#    caller needs an independent buffer.


class ObsView:
    """Schema + accessor for observation arrays of shape (4 + 8·n_robots,).

    Exposes block slices, block-view accessors, and named scalar field
    accessors. See module-level docstring for the observation layout.
    """

    def __init__(self, n_robots: int) -> None:
        if n_robots < 1:
            raise ValueError(
                f"ObsView requires at least 1 robot (self), got {n_robots}"
            )
        self.n_robots = n_robots

    # ---- shape ------------------------------------------------------------

    @property
    def total_dim(self) -> int:
        return BALL_BLOCK_DIM + self.n_robots * ROBOT_BLOCK_DIM

    # ---- slice properties -------------------------------------------------

    @property
    def ball_slice(self) -> slice:
        return slice(0, BALL_BLOCK_DIM)

    @property
    def self_slice(self) -> slice:
        # Trailing underscore on field names that would shadow `self`.
        return slice(BALL_BLOCK_DIM, BALL_BLOCK_DIM + ROBOT_BLOCK_DIM)

    def other_slice(self, idx: int) -> slice:
        """Slice for the idx-th other robot (0 ≤ idx < n_robots − 1)."""
        n_others = self.n_robots - 1
        if not 0 <= idx < n_others:
            raise IndexError(f"other index {idx} out of range [0, {n_others})")
        start = BALL_BLOCK_DIM + ROBOT_BLOCK_DIM * (idx + 1)
        return slice(start, start + ROBOT_BLOCK_DIM)

    # ---- block reads (numpy views — writes mutate the source array) -----

    def ball(self, arr: np.ndarray) -> np.ndarray:
        """Returns a numpy VIEW into `arr`. Writing to the returned slice
        mutates the underlying observation. Call .copy() for an independent
        buffer.

        Block layout: [px, py, vx, vy] (all normalized to [-1, 1])."""
        return arr[self.ball_slice]

    def self_(self, arr: np.ndarray) -> np.ndarray:
        """Returns a numpy VIEW into `arr`. Writing to the returned slice
        mutates the underlying observation. Call .copy() for an independent
        buffer.

        Block layout: [px, py, sin θ, cos θ, dx, dy, dθ]."""
        return arr[self.self_slice]

    def other(self, arr: np.ndarray, idx: int) -> np.ndarray:
        """Returns a numpy VIEW into `arr`. Writing to the returned slice
        mutates the underlying observation. Call .copy() for an independent
        buffer.

        Block layout for other robot `idx`: [px, py, sin θ, cos θ, dx, dy, dθ]."""
        return arr[self.other_slice(idx)]

    # ---- ball field accessors (return immutable Python floats) ----------

    def ball_px(self, arr: np.ndarray) -> float:
        return float(arr[0])

    def ball_py(self, arr: np.ndarray) -> float:
        return float(arr[1])

    def ball_vx(self, arr: np.ndarray) -> float:
        return float(arr[2])

    def ball_vy(self, arr: np.ndarray) -> float:
        return float(arr[3])

    def ball_xy(self, arr: np.ndarray) -> np.ndarray:
        """Returns a numpy VIEW into `arr` — writes mutate the source. The
        2-element position vector [px, py] of the ball."""
        return arr[0:2]

    def ball_v(self, arr: np.ndarray) -> np.ndarray:
        """Returns a numpy VIEW into `arr` — writes mutate the source. The
        2-element velocity vector [vx, vy] of the ball."""
        return arr[2:4]

    # ---- self field accessors -------------------------------------------
    # The self block always begins at index BALL_BLOCK_DIM (= 4).

    def self_px(self, arr: np.ndarray) -> float:
        return float(arr[BALL_BLOCK_DIM + 0])

    def self_py(self, arr: np.ndarray) -> float:
        return float(arr[BALL_BLOCK_DIM + 1])

    def self_sin_th(self, arr: np.ndarray) -> float:
        return float(arr[BALL_BLOCK_DIM + 2])

    def self_cos_th(self, arr: np.ndarray) -> float:
        return float(arr[BALL_BLOCK_DIM + 3])

    def self_dx(self, arr: np.ndarray) -> float:
        return float(arr[BALL_BLOCK_DIM + 4])

    def self_dy(self, arr: np.ndarray) -> float:
        return float(arr[BALL_BLOCK_DIM + 5])

    def self_dth(self, arr: np.ndarray) -> float:
        return float(arr[BALL_BLOCK_DIM + 6])

    def self_time_in_box(self, arr: np.ndarray) -> float:
        """Fraction of the goalie-box budget the robot has used continuously
        in its OPPOSING box this visit. 0.0 = not in box; 1.0 = at terminal."""
        return float(arr[BALL_BLOCK_DIM + 7])

    def self_xy(self, arr: np.ndarray) -> np.ndarray:
        """Returns a numpy VIEW into `arr` — writes mutate the source. The
        2-element world-frame position [px, py] of self."""
        return arr[BALL_BLOCK_DIM : BALL_BLOCK_DIM + 2]

    def self_v(self, arr: np.ndarray) -> np.ndarray:
        """Returns a numpy VIEW into `arr` — writes mutate the source. The
        2-element world-frame velocity [dx, dy] of self."""
        return arr[BALL_BLOCK_DIM + 4 : BALL_BLOCK_DIM + 6]

    def self_theta(self, arr: np.ndarray) -> float:
        """Recovered heading angle in radians via `theta_from_sincos`. Returns
        a fresh Python float — independent of `arr`."""
        return theta_from_sincos(
            arr[BALL_BLOCK_DIM + 2], arr[BALL_BLOCK_DIM + 3]
        )

    # ---- other-robot field accessors -------------------------------------

    def other_px(self, arr: np.ndarray, idx: int) -> float:
        return float(arr[self.other_slice(idx).start + 0])

    def other_py(self, arr: np.ndarray, idx: int) -> float:
        return float(arr[self.other_slice(idx).start + 1])

    def other_sin_th(self, arr: np.ndarray, idx: int) -> float:
        return float(arr[self.other_slice(idx).start + 2])

    def other_cos_th(self, arr: np.ndarray, idx: int) -> float:
        return float(arr[self.other_slice(idx).start + 3])

    def other_dx(self, arr: np.ndarray, idx: int) -> float:
        return float(arr[self.other_slice(idx).start + 4])

    def other_dy(self, arr: np.ndarray, idx: int) -> float:
        return float(arr[self.other_slice(idx).start + 5])

    def other_dth(self, arr: np.ndarray, idx: int) -> float:
        return float(arr[self.other_slice(idx).start + 6])

    def other_time_in_box(self, arr: np.ndarray, idx: int) -> float:
        """Fraction of the goalie-box budget the other robot has used."""
        return float(arr[self.other_slice(idx).start + 7])

    def other_xy(self, arr: np.ndarray, idx: int) -> np.ndarray:
        """Returns a numpy VIEW into `arr` — writes mutate the source. The
        2-element world-frame position [px, py] of other robot `idx`."""
        s = self.other_slice(idx).start
        return arr[s : s + 2]

    def other_v(self, arr: np.ndarray, idx: int) -> np.ndarray:
        """Returns a numpy VIEW into `arr` — writes mutate the source. The
        2-element world-frame velocity [dx, dy] of other robot `idx`."""
        s = self.other_slice(idx).start
        return arr[s + 4 : s + 6]

    def other_theta(self, arr: np.ndarray, idx: int) -> float:
        """Recovered heading angle in radians for other robot `idx`. Returns
        a fresh Python float — independent of `arr`."""
        s = self.other_slice(idx).start
        return theta_from_sincos(arr[s + 2], arr[s + 3])


class ActionView:
    """Schema + accessor for action arrays of shape (2,): [V, Ω].

    Stateless. No constructor args — action shape is fixed."""

    # ---- shape ------------------------------------------------------------

    @property
    def total_dim(self) -> int:
        return 2

    # ---- index properties -------------------------------------------------

    @property
    def v_idx(self) -> int:
        return 0

    @property
    def omega_idx(self) -> int:
        return 1

    # ---- field accessors -------------------------------------------------

    def v(self, arr: np.ndarray) -> float:
        return float(arr[0])

    def omega(self, arr: np.ndarray) -> float:
        return float(arr[1])

    def as_tuple(self, arr: np.ndarray) -> tuple[float, float]:
        return float(arr[0]), float(arr[1])


def obs_dim(n_robots: int) -> int:
    """Total observation dimension for `n_robots` total robots in scene.
    Convenience shortcut around `ObsView(n_robots).total_dim`."""
    return ObsView(n_robots).total_dim


# ---------------------------------------------------------------------------
# Internal encoding helpers
# ---------------------------------------------------------------------------


def _clip_norm(x: float, scale: float) -> float:
    """Normalize by `scale` and clip to [-1, 1]. Scale must be > 0."""
    return float(np.clip(x / scale, -1.0, 1.0))


def _encode_ball(
    ball_state: np.ndarray,
    field_x_half: float,
    field_y_half: float,
    mirror: bool,
) -> np.ndarray:
    """ball_state is (4,) [px, py, vx, vy] in world frame, metres / m·s⁻¹."""
    px, py, vx, vy = float(ball_state[0]), float(ball_state[1]), float(ball_state[2]), float(ball_state[3])
    if mirror:
        px = -px
        vx = -vx
    return np.array(
        [
            _clip_norm(px, field_x_half),
            _clip_norm(py, field_y_half),
            _clip_norm(vx, V_BALL_MAX),
            _clip_norm(vy, V_BALL_MAX),
        ],
        dtype=np.float32,
    )


def _encode_robot(
    state_5d: np.ndarray,
    field_x_half: float,
    field_y_half: float,
    v_max: float,
    omega_max: float,
    mirror: bool,
    time_in_box_norm: float = 0.0,
) -> np.ndarray:
    """state_5d is sim_py's robot.state: (5,) [px, py, theta, v, omega].
    `v` is body-frame longitudinal velocity (m/s); we convert to world-frame
    (dx, dy) for the obs.

    `time_in_box_norm` is the robot's pre-normalised goalie-box timer in
    [0, 1] (0 = not in box, 1 = at terminal violation threshold). Mirror-
    invariant — it's a scalar count, not a coordinate."""
    px = float(state_5d[0])
    py = float(state_5d[1])
    theta = float(state_5d[2])
    v_body = float(state_5d[3])
    omega = float(state_5d[4])

    sin_theta = float(np.sin(theta))
    cos_theta = float(np.cos(theta))
    dx_world = v_body * cos_theta
    dy_world = v_body * sin_theta

    if mirror:
        # Reflect across the y-axis: x flips, y unchanged.
        # heading: θ → π − θ  ⟹  cos flips sign, sin unchanged
        # angular vel: ω flips sign (CCW becomes CW)
        px = -px
        dx_world = -dx_world
        cos_theta = -cos_theta
        omega = -omega

    return np.array(
        [
            _clip_norm(px, field_x_half),
            _clip_norm(py, field_y_half),
            sin_theta,  # mathematically in [-1, 1] — no clip needed
            cos_theta,
            _clip_norm(dx_world, v_max),
            _clip_norm(dy_world, v_max),
            _clip_norm(omega, omega_max),
            float(np.clip(time_in_box_norm, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_observation(
    *,
    field_x_half: float,
    field_y_half: float,
    ball_state: np.ndarray,
    self_state_5d: np.ndarray,
    others_states_5d: Sequence[np.ndarray] = (),
    mirror: bool = False,
    v_max: float = V_MAX_DEFAULT,
    omega_max: float = OMEGA_MAX_DEFAULT,
    self_time_in_box_norm: float = 0.0,
    others_time_in_box_norm: Sequence[float] = (),
) -> np.ndarray:
    """Build a normalized observation vector.

    Parameters
    ----------
    field_x_half, field_y_half
        Field half-extents in metres. Position normalization divides by these.
    ball_state
        Shape (4,). World-frame [px, py, vx, vy].
    self_state_5d
        Shape (5,). Sim's robot state [px, py, theta, v_body, omega].
    others_states_5d
        Sequence of (5,) arrays for every other robot in the scene.
    mirror
        True if we're encoding from the perspective of a team attacking the
        −x goal (so we present a canonical "I attack +x" view to the policy).
    v_max, omega_max
        Robot velocity normalization scales. Defaults match the platform.
    self_time_in_box_norm
        Self's goalie-box timer in [0, 1]. Default 0 (rule-disabled / fresh
        episode). The env passes its tracked value here.
    others_time_in_box_norm
        Same, for each other robot in `others_states_5d` order. If shorter
        than `others_states_5d`, missing entries default to 0.

    Returns
    -------
    np.ndarray, shape (4 + 8·N,), dtype float32
        Where N = 1 + len(others_states_5d).
    """
    parts: list[np.ndarray] = [
        _encode_ball(ball_state, field_x_half, field_y_half, mirror),
        _encode_robot(
            self_state_5d, field_x_half, field_y_half, v_max, omega_max, mirror,
            time_in_box_norm=self_time_in_box_norm,
        ),
    ]
    others_t = list(others_time_in_box_norm)
    for i, s in enumerate(others_states_5d):
        t = others_t[i] if i < len(others_t) else 0.0
        parts.append(
            _encode_robot(s, field_x_half, field_y_half, v_max, omega_max, mirror,
                          time_in_box_norm=t)
        )
    return np.concatenate(parts).astype(np.float32, copy=False)


def action_to_wheel_cmds(
    v_norm: float,
    omega_norm: float,
    *,
    max_wheel_speed: float = V_MAX_DEFAULT,
    track_width: float = TRACK_WIDTH_DEFAULT,
    mirror: bool = False,
) -> tuple[float, float]:
    """Map normalized (V, Ω) ∈ [-1, 1]² to wheel commands (v_left, v_right) m/s.

    Anti-windup: if either wheel command would exceed `max_wheel_speed`, BOTH
    wheels are scaled by the same factor so V/Ω ratio is preserved. Avoids the
    saturation discontinuity of hard per-wheel clipping.

    `mirror=True` is the inverse of the obs mirror flag — used when the policy
    was producing actions in a mirrored reference frame (self-play opponent).
    Negates Ω before mapping; V is body-frame longitudinal speed and is
    unaffected by world reflection.
    """
    v_norm = float(np.clip(v_norm, -1.0, 1.0))
    omega_norm = float(np.clip(omega_norm, -1.0, 1.0))
    if mirror:
        omega_norm = -omega_norm

    v_cmd = v_norm * max_wheel_speed
    omega_max = 2.0 * max_wheel_speed / track_width
    omega_cmd = omega_norm * omega_max

    half_diff = omega_cmd * track_width / 2.0
    v_left = v_cmd - half_diff
    v_right = v_cmd + half_diff

    max_abs = max(abs(v_left), abs(v_right))
    if max_abs > max_wheel_speed and max_abs > 0.0:
        scale = max_wheel_speed / max_abs
        v_left *= scale
        v_right *= scale

    return v_left, v_right
