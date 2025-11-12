"""
Utility functions for multi-agent reinforcement learning.

This module contains various utility functions for training, evaluation,
and analysis of multi-agent RL systems.

New Advanced Features:
- Attention mechanisms (TarMAC, multi-head attention)
- Graph Neural Networks (GAT, GCN, MPNN)
- Intrinsic Curiosity Module (ICM, RND)
"""

from .training_utils import *
from .evaluation_utils import *
from .config_utils import *
from .communication import CommunicationChannel

# Advanced modules (import on demand to avoid dependency issues)
try:
    from . import attention
    from . import graph_networks
    from . import curiosity
except ImportError:
    pass

__all__ = [
    'TrainingUtils',
    'EvaluationUtils',
    'ConfigUtils',
    'CommunicationChannel',
    'attention',
    'graph_networks',
    'curiosity'
]