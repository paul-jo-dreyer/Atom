"""GoalieBoxPenalty — time-based goalie-box constraint with depth-weighted
warning ramp + sparse termination cost.

Why this exists
---------------
The original spatial-only goalie-box penalty (a sigmoid intrusion field
from `StaticFieldPenalty`) suffered from the classic "sparse goal beats
dense penalty" failure mode in self-play: as PPO + GAE distributed the
+R goal reward across many steps, the per-step −k box penalty became
worth paying. Both teams co-evolved into a "ignore the box, score
faster" equilibrium that sim-trained policies wouldn't transfer to
real-world rules.

The fix is a *temporal* constraint instead of a *spatial* one: the
robot may pass through the opposing box freely up to a configurable
budget per visit. Once the budget is exceeded, the episode terminates
(the env owns that decision, see `AtomSoloEnv` / `AtomTeamEnv` step
loop). To give PPO a smooth shaping gradient leading up to the
terminal — instead of a single binary cliff — this term provides a
two-tier warning signal:

    1. Time-based ramp (penalty method):
           t < trigger     → 0
           trigger ≤ t < terminal → polynomial ramp (u^p, u in [0, 1])

    2. Spatial depth weighting (encourages directional behaviour):
           penalty = ramp × (depth_into_box / max_depth)

    3. Sparse cost on violation (interior-point analog):
           t == terminal → fire `info["box_violation_self"]` and the
           term emits a goal-equivalent negative reward on that one
           step, then the episode ends.

The depth factor turns the time-only ramp into a potential-field
shape: at any t > trigger, moving toward the box boundary reduces
both depth and (over the next step) ramp. The gradient points OUT of
the box — exactly the directional signal we want.

Reads / writes
--------------
Reads:
    obs_view.self_time_in_box(obs)  — pre-normalised [0, 1] timer
                                       (1.0 = at terminal)
    obs_view.self_px / self_py      — robot position for depth lookup
    info["box_violation_self"]      — set by env when this robot
                                       violated; gates the sparse cost

Configuration is via constructor params (see `__init__` docstring).
The sign convention is "positive value, negative weight": this term
returns the unsigned magnitude; the composite multiplies by `weight`
which should be NEGATIVE to deliver a penalty.

**Sign convention**: use a NEGATIVE weight (e.g. `weight=-0.5`). The
term returns an UNSIGNED magnitude (ramp + sparse cost ≥ 0); the
negative weight turns "violation" into a penalty. Positive weight
would actively reward lingering in the box.

Calibration (how to pick `weight` + `termination_penalty`)
----------------------------------------------------------
The policy sees the goalie-box rule as a sum of two costs per episode:

 1. **Integrated ramp** — the polynomial fires at every control step
    inside the warning zone, scaled by depth. Worst case (robot stays
    deep through the entire warning window): the integral collapses to

        integrated_ramp ≈ weight * n_steps / (power + 1)

    where `n_steps = (terminal_time - trigger_time) / control_dt`.
    Example: weight=-0.5, power=3, warning=1.8s, control_dt=1/30s
    ⟹ `n_steps = 54`, integrated_ramp ≈ -0.5 * 54 / 4 ≈ -6.8.

 2. **Sparse violation cost** — fires once on the terminal step:

        sparse = weight * termination_penalty

    Example: weight=-0.5, termination_penalty=20.0 ⟹ sparse = -10.

The total worst-case violation cost is `integrated_ramp + sparse`.

**Calibration rule of thumb**: pick `weight` + `termination_penalty`
so the total worst-case violation cost is **1-3× the magnitude of
GoalScoredReward.weight** in absolute terms. Less than 1× and the
policy may decide "loiter, score, exit" is worth it. More than 5-10×
and the policy treats the warning zone as a death region and avoids
the box even when scoring would require briefly entering — over-
correction.

The PPO TensorBoard breakdown (`reward/goalie_box` vs
`reward/goal_scored`) is the diagnostic: if the per-episode means
have a healthy ratio (the cost-of-violation episodes shouldn't
dwarf the reward-of-scoring episodes by orders of magnitude), the
calibration is balanced.
"""

from __future__ import annotations

from AtomGym.rewards._base_reward import RewardContext, RewardTerm


class GoalieBoxPenalty(RewardTerm):
    name = "goalie_box"
    expected_weight_sign = -1

    def __init__(
        self,
        weight: float = 1.0,
        *,
        trigger_time: float = 2.0,
        terminal_time: float = 3.0,
        power: float = 3.0,
        termination_penalty: float = 1.0,
        goalie_box_depth: float = 0.12,
        goalie_box_y_half: float = 0.10,
        goalie_box_corner_radius: float = 0.0,
        depth_saturation: float = 0.06,
        depth_floor: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        weight
            Magnitude scale. Use a NEGATIVE value (e.g. -20.0) — the term
            returns positive numbers and the composite multiplies in the
            sign. Calibrate so the typical worst-case ramp integral plus
            a single termination event produces a net cost roughly equal
            to (or slightly larger than) the goal reward.
        trigger_time
            Seconds of free box-time per visit before the ramp activates.
            Below this, penalty is 0 (free use).
        terminal_time
            Seconds at which the env terminates the episode for box
            violation. Must be > trigger_time. The env's
            `goalie_box_terminal_time` MUST equal this value — the
            normalised timer in obs is `min(elapsed / terminal_time, 1.0)`,
            so trigger_normalized = trigger_time / terminal_time relies
            on these being the same.
        power
            Polynomial steepness of the ramp in [trigger, terminal].
            Higher = more peaked near terminal. Default 3 ⟹ cubic ramp:
            most of the cost is paid in the last ~30% of the warning
            window, encouraging the policy to either exit early or
            commit to scoring fast.
        termination_penalty
            Multiplier on `weight` for the discrete cost fired exactly
            once on the violation step. Default 1.0 ⟹ violator pays the
            full `weight` magnitude in addition to whatever ramp it has
            integrated. With `weight=-goal_reward` this makes a
            box-violation termination strictly worse than a goal scored.
        goalie_box_depth, goalie_box_y_half
            Geometry of the goalie box (inherited from env at config-
            load time — see `_REWARD_INHERITS_FROM_ENV` in
            `training/config.py`). Used to compute the depth factor.
        goalie_box_corner_radius
            Radius of the rounded interior corners. Inherited from
            env. Default 0 ⟹ legacy sharp-cornered rectangle.
        depth_saturation
            Distance from the nearest field-facing box edge at which the
            depth factor saturates to 1. Default 0.06 m ≈ one robot
            chassis side — beyond ~one body-length inside the box, the
            ramp pays full magnitude regardless of how much deeper the
            robot goes. Smaller values make the spatial gradient steeper
            near the boundary.
        depth_floor
            Minimum effective depth factor regardless of position
            inside the box, in [0, 1]. Linearly blends the raw
            depth factor toward 1: `effective = (1−α)·raw + α` where
            α = `depth_floor`. Default 0.0 ⟹ pure depth-graduated
            potential field (legacy: boundary penalty = 0). Set > 0
            to ensure the warning ramp pays meaningful penalty even at
            the box boundary as time approaches terminal — useful when
            you want time-pressure to dominate position. The peak
            per-step penalty (at the centroid / depth-saturated
            interior) is unchanged at `weight × u^p`. Worst-case
            calibration math is therefore unaffected; only the
            best-case (boundary-loitering) integrated cost rises.
        """
        super().__init__(weight=weight)
        if trigger_time < 0.0:
            raise ValueError(f"trigger_time must be >= 0, got {trigger_time}")
        if terminal_time <= trigger_time:
            raise ValueError(
                f"terminal_time ({terminal_time}) must be > "
                f"trigger_time ({trigger_time})"
            )
        if power <= 0.0:
            raise ValueError(f"power must be > 0, got {power}")
        if termination_penalty < 0.0:
            raise ValueError(
                f"termination_penalty must be >= 0, got {termination_penalty}"
            )
        if goalie_box_depth <= 0.0 or goalie_box_y_half <= 0.0:
            raise ValueError(
                f"goalie box dims must be > 0, got "
                f"depth={goalie_box_depth}, y_half={goalie_box_y_half}"
            )
        if goalie_box_corner_radius < 0.0:
            raise ValueError(
                f"goalie_box_corner_radius must be >= 0, got {goalie_box_corner_radius}"
            )
        if depth_saturation <= 0.0:
            raise ValueError(f"depth_saturation must be > 0, got {depth_saturation}")
        if not 0.0 <= depth_floor <= 1.0:
            raise ValueError(
                f"depth_floor must be in [0, 1], got {depth_floor}"
            )
        self.trigger_time = float(trigger_time)
        self.terminal_time = float(terminal_time)
        self.power = float(power)
        self.termination_penalty = float(termination_penalty)
        self.goalie_box_depth = float(goalie_box_depth)
        self.goalie_box_y_half = float(goalie_box_y_half)
        self.goalie_box_corner_radius = float(goalie_box_corner_radius)
        self.depth_saturation = float(depth_saturation)
        self.depth_floor = float(depth_floor)

        # Pre-compute fixed quantities used in __call__.
        self._trigger_norm = self.trigger_time / self.terminal_time

    def __call__(self, ctx: RewardContext) -> float:
        # Sparse termination cost — fires exactly once when the env
        # determined this robot violated. Read first so a valid violation
        # always delivers the discrete signal even if the timer somehow
        # rounded below 1.0 by a tiny amount.
        sparse = (
            self.termination_penalty
            if ctx.info.get("box_violation_self", False)
            else 0.0
        )

        # Per-step ramp. Off entirely below trigger.
        timer_norm = ctx.obs_view.self_time_in_box(ctx.obs)
        if timer_norm <= self._trigger_norm:
            return sparse

        u = (timer_norm - self._trigger_norm) / (1.0 - self._trigger_norm)
        # Numerically-stable monotonic ramp; clip just in case the env
        # writes a slightly out-of-band value.
        u = max(0.0, min(1.0, u))
        ramp = u ** self.power

        # Depth factor — robot's current depth into the opposing box.
        # Belt-and-braces clamp: env resets the timer on box exit, so
        # if timer > trigger we're "in the box" by definition. The depth
        # factor scales the ramp by how deep we are, providing the
        # outward gradient that makes "head for the boundary" the locally
        # rewarding direction.
        rx = ctx.obs_view.self_px(ctx.obs) * ctx.field_x_half
        ry = ctx.obs_view.self_py(ctx.obs) * ctx.field_y_half
        depth_factor = self._depth_factor_at(rx, ry, ctx.field_x_half)

        return ramp * depth_factor + sparse

    # ---- helpers ----------------------------------------------------------

    def _depth_factor_at(
        self, rx: float, ry: float, field_x_half: float
    ) -> float:
        """Effective depth factor in [depth_floor, 1] inside the box,
        0 outside.

        Geometry: rounded-rect SDF from `AtomGym.goalie_box_geometry`
        (shared with env, so box-entry test and depth shaping agree).

        Shaping: raw depth-saturated factor is `min(depth/sat, 1)` in
        [0, 1]. Linearly blended toward 1 by `depth_floor`:

            effective = (1 − depth_floor) · raw + depth_floor

        depth_floor=0 recovers the legacy potential-field (boundary
        contributes 0). depth_floor=1 makes the per-step penalty
        uniform across the box interior. Outside the box the term
        always returns 0 — the boundary cliff is preserved at all
        depth_floor values."""
        from AtomGym.goalie_box_geometry import signed_depth_into_box
        depth = signed_depth_into_box(
            rx, ry,
            field_x_half=field_x_half,
            goalie_box_depth=self.goalie_box_depth,
            goalie_box_y_half=self.goalie_box_y_half,
            goalie_box_corner_radius=self.goalie_box_corner_radius,
            side=+1,
        )
        if depth <= 0.0:
            return 0.0
        raw = min(depth / self.depth_saturation, 1.0)
        return (1.0 - self.depth_floor) * raw + self.depth_floor
