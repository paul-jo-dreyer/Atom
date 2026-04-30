"""StallPenaltyReward — discourages near-zero actions ("do nothing").

PPO with a continuous Gaussian policy will often collapse to small-
magnitude actions early in training: random actions hurt cumulative
reward (the dense distance penalty accumulates), so the safest greedy
choice from a barely-trained value function is "stand still". Once the
std of the action distribution shrinks around zero, breaking out of
that attractor takes a long time.

This term puts a small constant pull AWAY from the (V, Ω) origin so the
policy keeps moving while it figures out what to do.

L∞ formulation (current)
------------------------
    reward = max(0, 1 - max(|V|, |Ω|))

The earlier L2 formulation `1 - sqrt(V² + Ω²)` had a radial gradient
that pushed the policy toward the (1, 1) corner of action space —
increasing either component or both reduced the penalty
proportionally, creating an attractor for "max throttle on both"
actions even when pure forward or pure rotation was the right move.

The L∞ version is symmetric in V and Ω: pure forward (V=1, Ω=0) and
pure rotation (V=0, Ω=1) give exactly the same zero penalty as full
mixed (V=1, Ω=1). The gradient flips sign along the V=Ω diagonal but
that's fine for stochastic-policy PPO — the kink is on a measure-zero
set in action space and noise-averaged updates handle it cleanly.

Properties
----------
    (0, 0)        → 1.0    (full penalty)
    (1, 0)        → 0.0    (full forward, no penalty)
    (0, 1)        → 0.0    (full rotation, no penalty — was identical
                            in L2, but mid-rotation now matches mid-V)
    (0.5, 0)      → 0.5
    (0, 0.5)      → 0.5    (matches; pure rotation is no worse than
                            pure thrust in the L∞ formulation)
    (0.5, 0.5)    → 0.5    (no benefit to adding Ω to V)
    (1, 1)        → 0.0

Use with a NEGATIVE weight (e.g. weight=-0.3) to turn this into a
per-step nudge away from stalling. Larger magnitude = stronger push.
If the policy still loiters in low-magnitude actions after this, the
basin can be tightened by clipping the input — see comments in the
`__call__` method.
"""

from __future__ import annotations

from AtomGym.rewards._base_reward import RewardContext, RewardTerm


class StallPenaltyReward(RewardTerm):
    name = "stall_penalty"

    def __call__(self, ctx: RewardContext) -> float:
        v = ctx.action_view.v(ctx.action)
        omega = ctx.action_view.omega(ctx.action)
        # L∞ norm: penalty depends on whichever component is largest in
        # magnitude. Treats V and Ω symmetrically; either at full
        # magnitude is sufficient to zero the penalty.
        magnitude = max(abs(v), abs(omega))
        # If you want to tighten the basin so penalty is zero whenever
        # either component exceeds some threshold < 1.0, change to:
        #   return max(0.0, 1.0 - magnitude / threshold)
        # but keep `weight` calibrated against the [0, 1] return range.
        return max(0.0, 1.0 - magnitude)
