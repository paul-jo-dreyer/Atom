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

Gated by `info["ball_touched"]` — credit-hack guard
---------------------------------------------------
With random ball initial velocity (see `InitialStateRanges`), the ball
can fling into either goal in the first few steps after a reset before
ANY robot has had a chance to influence it. Crediting (or penalising)
the policy for these events is pure noise — the policy took whatever
random action it took, and the goal would have happened regardless.

The env tracks whether **any robot in the world** has touched the ball
at any point in the current episode and exposes the result as
`info["ball_touched"]`. This term suppresses BOTH the +1 and -1
branches when that flag is False — the episode still terminates (the
env owns that decision), but no sparse reward is delivered.

Self-play note: the gate covers the LEARNER's and the OPPONENT's
contacts symmetrically. If the opponent puts the ball into the
learner's goal, the learner SHOULD see the negative reward — that's a
defensive failure to learn from. Only goals where neither robot has
touched the ball are spurious. Solo envs only have the learner, so
"any robot touched" reduces trivially to "learner touched" there.

Default if the flag is missing from `info` is **False** — i.e. the
reward stays silent. That makes "forgot to plumb the flag" a loud
training failure (no goal signal at all), not a silent corruption
(spurious goals leaking back into training). The env always sets the
flag, so production is fully credited.
"""

from __future__ import annotations

from AtomGym.rewards._base_reward import RewardContext, RewardTerm


class GoalScoredReward(RewardTerm):
    name = "goal_scored"

    def __call__(self, ctx: RewardContext) -> float:
        if not ctx.info.get("ball_touched", False):
            return 0.0
        if ctx.info.get("scored_for_us", False):
            return 1.0
        if ctx.info.get("scored_against_us", False):
            return -1.0
        return 0.0
