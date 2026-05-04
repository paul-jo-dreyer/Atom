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

Distance gating (and why the default keeps it active in contact)
----------------------------------------------------------------
The annular gate has an `inner_radius` knob, but the default is 0 — the
term is active everywhere up to `outer_radius`, including in contact.

Earlier we masked contact (`inner_radius=0.044 ≈ chassis_half + ball_radius`)
on the theory that "face the ball" shaping would suppress the
emergent ball-at-corner dribbling that comes from contact dynamics.
Empirically we observed the opposite failure mode: a "tangential lock"
freeze at 40–60 mm — right at the inner-gate boundary — where the
robot ends up perpendicular to the ball, stationary, with no rotational
gradient. Distance is locally flat under perpendicular motion,
ball-progress is zero with the ball stationary, and alignment was
silenced by the gate. Three dense terms saying nothing.

With `inner_radius=0` the alignment signal stays alive into contact.
The "release the ball to chase alignment bonus" attractor we worried
about doesn't materialise in practice because BallProgressReward (which
fires on `|v_ball|·cos θ_to_goal`) dominates whenever the robot is
actively pushing — its peak magnitude is well above this term's at
typical pushing speeds. The `inner_radius` knob is preserved for
researchers who want to mask contact for a specific experiment.

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

So with the defaults (inner=0, outer=0.18, back_weight=0.3):
    far from ball (d > 0.18 m)        → 0     (other rewards do the work)
    in contact, front toward ball     → ~ gate(0.044)·1.0  (gate ≈ 0.79)
    band midpoint (d = 0.09 m), front → ~1.0
    band midpoint, back toward ball   → ~0.3
    band midpoint, perpendicular      → 0     (kink at α=±π/2)

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
        inner_radius: float = 0.0,
        outer_radius: float = 0.18,
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
            silent. Default 0 — the term is active in contact (see the
            "tangential lock" discussion in the module docstring). Set
            to ~0.044 (chassis_half + ball_radius) to recover the
            original "mask contact" behaviour.
        outer_radius
            Above this distance, the reward is silent. Default 0.18 m,
            wide enough to cover the freeze zone we observed at
            40-60 mm AND give a useful approach gradient up to ~3
            chassis widths from the ball.
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
