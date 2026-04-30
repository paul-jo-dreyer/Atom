"""ObstacleContactPenalty — discourages time spent touching anything but
the ball.

Why this exists
---------------
Once StallPenaltyReward is in place, the policy is rewarded for keeping
some throttle on at all times. The pathological corner case: a robot that
drives chest-first into a wall now has a small per-step incentive to *keep
pushing*, because backing off briefly costs stall-penalty reward while the
robot is still pinned (no physical motion ⟹ no progress reward either).
The result is a robot that spends the rest of the episode wedged against
the wall.

The fix is a per-step penalty that scales with how much of the control
step the robot spent in contact with an obstacle (wall, goal-wall, or
another robot) — but specifically NOT the ball. Touching the ball is the
whole point.

Signal shape
------------
The env precomputes `info["obstacle_contact_frac"]`, the fraction of
physics substeps within the control step where the robot's body had at
least one active Box2D contact with an obstacle category. That value is
bounded in [0, 1]:

    0.0  — no contact at any substep
    1.0  — every substep had a contact (robot is fully pinned)

This term simply forwards that fraction. With a NEGATIVE weight `-k`, a
robot fully pinned for one second of sim time accumulates a penalty of
`k` (since 1 s × 1.0-frac = 1 unit of "contact-seconds").

Richer variants worth building once we have data
------------------------------------------------
The env also exposes `info["robot_contacts"]`, a flat list of every
`sim_py.RobotContactPoint` Box2D reported across all substeps. That data
supports several richer shapings — built as separate RewardTerms so we
can compose:

  * impulse-magnitude penalty: scales with how hard the wall is shoving
    back (`sum(c.normal_impulse) / dt`), so a glancing bump costs less
    than chest-pinning.
  * head-on penalty: dot the contact normal with the robot's heading;
    head-on collisions get a much bigger signal than sideswipes, so the
    gradient pushes the policy to *rotate away* before unsticking, which
    is the actual desired behaviour.
  * penetration-depth proxy: for soft signal even when impulse is small.

Start simple. If the fraction-frac signal alone fixes the wall-pinning
attractor, leave it. If it doesn't, layer the impulse-scaled term on top.
"""

from __future__ import annotations

from AtomGym.rewards._base_reward import RewardContext, RewardTerm


class ObstacleContactPenalty(RewardTerm):
    name = "obstacle_contact"

    def __call__(self, ctx: RewardContext) -> float:
        return float(ctx.info.get("obstacle_contact_frac", 0.0))
