"""
Base Agent Class

Abstract base class for all multi-agent RL agents.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class BaseAgent(ABC):
    """
    Abstract base class for all multi-agent RL agents.
    
    This class defines the interface that all agents must implement.
    """
    
    def __init__(
        self,
        agent_id: int,
        observation_space: Any,
        action_space: Any,
        learning_rate: float = 0.001,
        device: str = "cpu"
    ):
        self.agent_id = agent_id
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.device = device
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_steps = 0
        
    @abstractmethod
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select an action given an observation.
        
        Args:
            observation: Current observation
            training: Whether the agent is in training mode
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the agent's policy/value function.
        
        Args:
            batch: Training batch containing experiences
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the agent's model to file."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load the agent's model from file."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self.agent_id,
            "training_steps": self.training_steps,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "avg_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            "avg_length": np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0.0
        }
    
    def reset_episode(self) -> None:
        """Reset agent state for new episode."""
        pass
    
    def set_training_mode(self, training: bool) -> None:
        """Set training mode for the agent."""
        self.training = training