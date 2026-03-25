"""
Multi-Agent Reinforcement Learning Agents

This module contains various agent implementations for multi-agent RL.

Available Agents:
- DQNAgent: Basic Deep Q-Network agent
- AttentionDQNAgent: DQN with attention-based communication (TarMAC)
- GNNDQNAgent: DQN with Graph Neural Networks for scalable coordination
- LSTMDQNAgent: DQN with LSTM for partial observability
- PolicyGradientAgent: REINFORCE agent
- ActorCriticAgent: Actor-Critic agent
"""

from .base_agent import BaseAgent
from .dqn_agent import DQNAgent
from .attention_dqn_agent import AttentionDQNAgent
from .gnn_dqn_agent import GNNDQNAgent
from .lstm_dqn_agent import LSTMDQNAgent
from .policy_gradient_agent import PolicyGradientAgent
from .actor_critic_agent import ActorCriticAgent

__all__ = [
    'BaseAgent',
    'DQNAgent',
    'AttentionDQNAgent',
    'GNNDQNAgent',
    'LSTMDQNAgent',
    'PolicyGradientAgent',
    'ActorCriticAgent'
]