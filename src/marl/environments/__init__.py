"""
Multi-Agent Environments

This module contains various multi-agent environments for reinforcement learning.
"""

from .grid_world import MultiAgentGridWorld
from .cooperative_navigation import CooperativeNavigation
from .predator_prey import PredatorPrey
from .traffic import TrafficEnv
from .robotics import MultiAgentRobotNavigation

__all__ = [
    "MultiAgentGridWorld",
    "CooperativeNavigation",
    "PredatorPrey",
    "TrafficEnv",
    "MultiAgentRobotNavigation"
]