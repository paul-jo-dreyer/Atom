"""StallPenaltyReward — discourages near-zero actions ("do nothing").

PPO with a continuous Gaussian policy will often collapse to small-
magnitude actions early in training: random actions hurt cumulative
reward (the dense distance penalty accumulates), so the safest greedy
choice from a barely-trained value function is "stand still". Once the
std of the action distribution shrinks around zero, breaking out of that
attractor takes a long time.

This term puts a small constant pull AWAY from the (V, Ω) origin so the
policy keeps moving while it figures out what to do.

Formula
-------
    reward = max(0, 1 - sqrt(V² + Ω²))

So:
    (0, 0)        → 1.0    (maximum penalty when used with negative weight)
    (1, 0)        → 0.0    (full forward, no penalty)
    (0, 1)        → 0.0    (full spin, no penalty)
    (1, 1)        → 0.0    (clipped — magnitude exceeds 1)
    (0.5, 0)      → 0.5
    (0.7, 0.7)    → ~0.01

Use with a NEGATIVE weight (e.g. weight=-0.3) to turn this into a per-
step nudge away from stalling. Larger magnitude = stronger push.
"""

from __future__ import annotations

from AtomGym.rewards._base_reward import RewardContext, RewardTerm


class StallPenaltyReward(RewardTerm):
    name = "stall_penalty"

    def __call__(self, ctx: RewardContext) -> float:
        v = ctx.action_view.v(ctx.action)
        omega = ctx.action_view.omega(ctx.action)
        magnitude = (v * v + omega * omega) ** 0.5
        # Clamp at zero — actions with L2 > 1 (e.g. (0.8, 0.8)) shouldn't
        # earn negative-stall reward (which with a negative weight becomes
        # a positive bonus).
        return max(0.0, 1.0 - magnitude)
