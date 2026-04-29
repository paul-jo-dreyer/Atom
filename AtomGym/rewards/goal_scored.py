"""GoalScoredReward — sparse, large reward on goal events.

Reads the env's edge-detected info flags (`scored_for_us`,
`scored_against_us`) and emits ±1.0 on the rising edge of either. Use a
large weight to make these terminal events dominate the dense shaping
rewards (e.g. `GoalScoredReward(weight=20.0)` puts 20× the magnitude of a
unit shaping term).

Symmetric: scoring and conceding produce equal-magnitude opposite-sign
rewards. If you want asymmetric weighting, compose two instances and
filter each one on a single info key (subclass the term, or use distinct
named subclasses — TBD if we ever need it).
"""

from __future__ import annotations

from AtomGym.rewards._base_reward import RewardContext, RewardTerm


class GoalScoredReward(RewardTerm):
    name = "goal_scored"

    def __call__(self, ctx: RewardContext) -> float:
        if ctx.info.get("scored_for_us", False):
            return 1.0
        if ctx.info.get("scored_against_us", False):
            return -1.0
        return 0.0
