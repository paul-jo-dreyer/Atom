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

Use with a NEGATIVE weight (e.g. `weight=-20.0`).
"""

from __future__ import annotations

from AtomGym.rewards._base_reward import RewardContext, RewardTerm


class GoalieBoxPenalty(RewardTerm):
    name = "goalie_box"

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
        depth_saturation: float = 0.06,
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
            Geometry of the goalie box (matches `StaticFieldPenalty` and
            env defaults). Used to compute the depth factor for the
            ramp.
        depth_saturation
            Distance from the nearest field-facing box edge at which the
            depth factor saturates to 1. Default 0.06 m ≈ one robot
            chassis side — beyond ~one body-length inside the box, the
            ramp pays full magnitude regardless of how much deeper the
            robot goes. Smaller values make the spatial gradient steeper
            near the boundary.
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
        if depth_saturation <= 0.0:
            raise ValueError(f"depth_saturation must be > 0, got {depth_saturation}")
        self.trigger_time = float(trigger_time)
        self.terminal_time = float(terminal_time)
        self.power = float(power)
        self.termination_penalty = float(termination_penalty)
        self.goalie_box_depth = float(goalie_box_depth)
        self.goalie_box_y_half = float(goalie_box_y_half)
        self.depth_saturation = float(depth_saturation)

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
        """Normalised depth into the opposing (+x) goalie box, in [0, 1].

        The opposing box is a rectangle adjacent to the +x goal:
            x ∈ [field_x_half - depth, field_x_half]
            y ∈ [-y_half, +y_half]

        Three of its four edges face the field interior (the fourth abuts
        the goal mouth — there's no "exit" through it). Depth = min
        distance to any of those three field-facing edges, normalised by
        `min(box_depth, box_y_half)` (the smallest "outward" distance
        from the deepest point in the box). Outside the box: 0."""
        if abs(ry) > self.goalie_box_y_half:
            return 0.0
        box_inner_x = field_x_half - self.goalie_box_depth
        box_outer_x = field_x_half
        if rx < box_inner_x or rx > box_outer_x:
            return 0.0
        # Three field-facing edges: x=box_inner_x, y=+y_half, y=-y_half.
        d_left = rx - box_inner_x
        d_top = self.goalie_box_y_half - ry
        d_bottom = ry + self.goalie_box_y_half
        depth = min(d_left, d_top, d_bottom)
        return max(0.0, min(1.0, depth / self.depth_saturation))
