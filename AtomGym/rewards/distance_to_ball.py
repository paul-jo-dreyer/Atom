"""DistanceToBallReward — L2 distance between robot xy and ball xy.

**Sign convention**: use a NEGATIVE weight (e.g. `weight=-0.5`). The
term returns POSITIVE L2 distance in metres (unsigned magnitude); the
negative weight turns "more distance" into "more negative reward",
i.e. closer-is-better. With a positive weight the agent would learn
to maximise distance from the ball — the opposite of what's almost
always intended.

The breakdown logged to TensorBoard is `weight × value`, so a negative
weight produces a negative breakdown entry — easy to spot at a glance.
"""

from __future__ import annotations

from AtomGym.rewards._base_reward import RewardContext, RewardTerm


class DistanceToBallReward(RewardTerm):
    name = "distance_to_ball"
    expected_weight_sign = -1

    def __call__(self, ctx: RewardContext) -> float:
        # De-normalize positions back to metric units before measuring
        # distance — the obs's x and y are normalized by DIFFERENT factors
        # (field_x_half vs field_y_half), so distance computed in normalized
        # coords would be anisotropically wrong.
        bx = ctx.obs_view.ball_px(ctx.obs) * ctx.field_x_half
        by = ctx.obs_view.ball_py(ctx.obs) * ctx.field_y_half
        rx = ctx.obs_view.self_px(ctx.obs) * ctx.field_x_half
        ry = ctx.obs_view.self_py(ctx.obs) * ctx.field_y_half
        dx = bx - rx
        dy = by - ry
        return (dx * dx + dy * dy) ** 0.5
