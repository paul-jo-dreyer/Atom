"""Gymnasium environments for the soccer curriculum.

Each level lives in its own file:
    solo_env.py  — Level 1: single robot, ball, two goals, no opponent.
    team_env.py  — Level 2+ (1v1, 2v2): centralized team policy + opponent.
"""

from .initial_state import InitialStateRanges
from .solo_env import AtomSoloEnv

__all__ = ["AtomSoloEnv", "InitialStateRanges"]
