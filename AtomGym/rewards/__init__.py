"""Reward terms for AtomGym envs.

The base machinery — `RewardContext`, `RewardTerm`, `RewardComposite` — lives
in `_base_reward.py`. Concrete reward terms live in their own files (one per
file for greppability) and are added here as they're written.
"""

from ._base_reward import RewardComposite, RewardContext, RewardTerm
from .ball_alignment import BallAlignmentReward
from .ball_progress import BallProgressReward
from .distance_to_ball import DistanceToBallReward
from .goal_scored import GoalScoredReward
from .obstacle_contact import ObstacleContactPenalty
from .stall_penalty import StallPenaltyReward
from .static_field_penalty import StaticFieldPenalty

__all__ = [
    "BallAlignmentReward",
    "BallProgressReward",
    "DistanceToBallReward",
    "GoalScoredReward",
    "ObstacleContactPenalty",
    "RewardComposite",
    "RewardContext",
    "RewardTerm",
    "StallPenaltyReward",
    "StaticFieldPenalty",
]
