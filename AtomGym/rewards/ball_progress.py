"""BallProgressReward — reward proportional to ball velocity toward the
opponent's goal mouth.

Mathematically the reward is `|v_ball| · cos(θ)` where θ is the angle
between the ball's velocity vector and the unit direction from the ball
to the closest reachable point in the goal mouth. Equivalently:

    reward = v_ball ⋅ d̂ ,    d̂ = (target - ball_xy) / ||target - ball_xy||

where `target = (field_x_half, clip(ball_y, ±goal_y_half))`. Snapping the
target's y to the goal-mouth band gives the policy a smoother gradient
when the ball is above/below the mouth — the direction always points to
the nearest scorable point, not the literal mouth centre.

Signed: positive when the ball moves toward the goal, negative when it
moves away, zero when stationary or moving perpendicular. Units are m/s.
"""

from __future__ import annotations

from AtomGym.action_observation import V_BALL_MAX
from AtomGym.rewards._base_reward import RewardContext, RewardTerm


class BallProgressReward(RewardTerm):
    name = "ball_progress"

    def __call__(self, ctx: RewardContext) -> float:
        # De-normalize ball obs back to metric units.
        bx = ctx.obs_view.ball_px(ctx.obs) * ctx.field_x_half
        by = ctx.obs_view.ball_py(ctx.obs) * ctx.field_y_half
        bvx = ctx.obs_view.ball_vx(ctx.obs) * V_BALL_MAX
        bvy = ctx.obs_view.ball_vy(ctx.obs) * V_BALL_MAX

        # Target: closest scorable point inside the +x goal mouth.
        target_x = ctx.field_x_half
        if by > ctx.goal_y_half:
            target_y = ctx.goal_y_half
        elif by < -ctx.goal_y_half:
            target_y = -ctx.goal_y_half
        else:
            target_y = by

        dx = target_x - bx
        dy = target_y - by
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 1e-9:
            # Ball is on top of the target — direction is undefined. Reward
            # cannot meaningfully encode "progress" here, so return 0. (In
            # practice the env will have flagged a goal and terminated.)
            return 0.0

        # Project velocity onto the unit direction to target.
        return bvx * (dx / dist) + bvy * (dy / dist)
