"""BallAlignmentReward — encourages body-axis alignment with the ball
in the credit-assignment hole between the approach and contact regimes.

Why this exists
---------------
When the robot is near but not touching the ball, distance-to-ball is
locally informative for forward / backward thrust but provides no signal
for rotation: rotating in place leaves the distance unchanged. PPO
struggles to credit-assign the multi-step plan "rotate, then push"
through this flat region, slowing learning whenever the robot ends up
in positions where its body axis isn't pointed at (or away from) the
ball direction.

This term provides a small dense gradient in that regime: more aligned
with the ball ⟹ slightly more reward. The shaping is **asymmetric** —
front-aligned (front-face toward ball) earns the full reward, back-
aligned earns a fraction (`back_weight`, default 0.3). Once a pusher
is attached, the front face is mechanically advantaged for ball
control, so the policy should prefer to approach front-first. Back-
alignment still earns positive reward (not penalised) because awkward
geometry sometimes forces it (e.g., the robot ends up between ball
and own goal and must back-push to avoid scoring on itself). Setting
`back_weight=1.0` recovers the original symmetric behaviour.

Why this term ISN'T active during ball contact
----------------------------------------------
Once the ball is touching the robot, controlled steering emerges from
ball-at-side / ball-at-corner contacts that let the robot curve the
ball's trajectory by turning. Encouraging "face the ball" during
contact would suppress that emergent dribbling behaviour. We let
BallProgressReward (velocity-based) drive behaviour in the contact
regime, and the alignment reward fades to zero before contact begins
so there's no perverse "release the ball to grab the alignment bonus"
attractor at the contact boundary.

Reward shape (annular)
----------------------
A smooth (parabolic) annular gate, zero outside a narrow distance band
and peaking at 1.0 at the band midpoint:

    gate(d) = 0                                   if d <= inner
              0                                   if d >= outer
              4 (d - inner)(outer - d)/(outer-inner)²   otherwise

    cos_delta = cos(θ_robot - atan2(by - ry, bx - rx))
    alignment = max(0, cos_delta) + back_weight · max(0, -cos_delta)

    reward    = gate(d) * alignment

So with the default back_weight = 0.3:
    far from ball                     → 0     (other rewards do the work)
    in contact                        → 0     (don't disrupt dribble)
    mid-approach, front toward ball   → ~1.0
    mid-approach, back toward ball    → ~0.3
    mid-approach, perpendicular       → 0

Implementation note: the alignment is computed via dot product, not
atan2. With robot forward = (cos θ, sin θ) and ball-direction unit
vector = ((bx-rx)/d, (by-ry)/d), the dot product is exactly cos(Δθ).
Skipping atan2 avoids a transcendental in the inner reward loop.

Use with a POSITIVE weight (e.g. weight=+0.3). The peak per-step
contribution is `weight * 1.0` at the band midpoint with full FRONT
alignment, well below the magnitude of BallProgressReward in active
pushing scenarios.
"""

from __future__ import annotations

from AtomGym.rewards._base_reward import RewardContext, RewardTerm


class BallAlignmentReward(RewardTerm):
    name = "ball_alignment"

    def __init__(
        self,
        weight: float = 1.0,
        inner_radius: float = 0.044,
        outer_radius: float = 0.10,
        back_weight: float = 0.3,
    ) -> None:
        """
        Parameters
        ----------
        weight
            Positive magnitude. The term itself returns values in [0, 1],
            so the per-step contribution to total reward is bounded by
            `weight`.
        inner_radius
            Below this ball-to-robot distance (metres), the reward is
            silent. Default 0.044 ≈ chassis_half (0.030) + ball_radius
            (0.014) — i.e. just inside the contact range.
        outer_radius
            Above this distance, the reward is silent. Default 0.10 m,
            roughly 1.5-2 chassis widths from the ball — past this, the
            existing distance / progress shaping is sufficient on its
            own.
        back_weight
            Multiplier applied when the robot is back-aligned to the
            ball (cos_delta < 0). Default 0.3 ⟹ back-pushing earns 30%
            of the reward front-pushing earns. Range: [0, 1]. 0 = no
            reward for back alignment (still no penalty); 1 = original
            symmetric behaviour.
        """
        super().__init__(weight=weight)
        if inner_radius < 0.0:
            raise ValueError(
                f"inner_radius must be >= 0, got {inner_radius}"
            )
        if outer_radius <= inner_radius:
            raise ValueError(
                f"outer_radius ({outer_radius}) must exceed "
                f"inner_radius ({inner_radius})"
            )
        if not 0.0 <= back_weight <= 1.0:
            raise ValueError(
                f"back_weight must be in [0, 1], got {back_weight}"
            )
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.back_weight = float(back_weight)
        self._band_width_sq = (self.outer_radius - self.inner_radius) ** 2

    def __call__(self, ctx: RewardContext) -> float:
        v = ctx.obs_view
        # Denormalise positions to world-frame metres. Obs encodes x by
        # field_x_half and y by field_y_half independently — multiply
        # back by the matching scale.
        rx = v.self_px(ctx.obs) * ctx.field_x_half
        ry = v.self_py(ctx.obs) * ctx.field_y_half
        bx = v.ball_px(ctx.obs) * ctx.field_x_half
        by = v.ball_py(ctx.obs) * ctx.field_y_half

        dx = bx - rx
        dy = by - ry
        dist = (dx * dx + dy * dy) ** 0.5

        # Annular gate. Zero outside the band; parabolic peak at the
        # midpoint, normalised to 1.0.
        if dist <= self.inner_radius or dist >= self.outer_radius:
            return 0.0
        gate = (
            4.0
            * (dist - self.inner_radius)
            * (self.outer_radius - dist)
            / self._band_width_sq
        )

        # Body-axis alignment via dot product of robot forward unit
        # vector with the ball-direction unit vector. cos(Δθ) directly,
        # no atan2 needed. `dist > inner_radius >= 0` makes the divide
        # safe.
        cos_th = v.self_cos_th(ctx.obs)
        sin_th = v.self_sin_th(ctx.obs)
        cos_delta = (cos_th * dx + sin_th * dy) / dist
        # Asymmetric: front-aligned (cos_delta > 0) earns full reward,
        # back-aligned earns `back_weight` × that. Continuous in
        # cos_delta, with a kink at perpendicular (gradient changes from
        # +1 to -back_weight). PPO doesn't differentiate through reward,
        # so the kink is fine.
        if cos_delta >= 0.0:
            alignment = cos_delta
        else:
            alignment = self.back_weight * (-cos_delta)

        return gate * alignment
