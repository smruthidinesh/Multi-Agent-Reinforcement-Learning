"""
Multi-Agent Reinforcement Learning Algorithms

This module contains various MARL algorithms for training multiple agents.
"""

from .independent_q_learning import IndependentQLearning
from .maddpg import MADDPG
from .mappo import MAPPO

__all__ = [
    'IndependentQLearning',
    'MADDPG',
    'MAPPO'
]