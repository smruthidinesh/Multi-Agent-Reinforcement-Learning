"""
Multi-Agent Reinforcement Learning Algorithms

This module contains various MARL algorithms for training multiple agents.
"""

from .base_algorithm import BaseAlgorithm
from .independent_q_learning import IndependentQLearning
from .maddpg import MADDPG
from .mappo import MAPPO
from .qmix import QMix

__all__ = [
    'BaseAlgorithm',
    'IndependentQLearning',
    'MADDPG',
    'MAPPO',
    'QMix'
]