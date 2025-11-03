"""
Policy Gradient Agent

Implementation of a policy gradient agent for multi-agent reinforcement learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent


class PolicyNetwork(nn.Module):
    """Policy network for policy gradient methods."""
    
    def __init__(self, input_dim: int, output_dim: int, message_dim: int = 0, hidden_dims: list = [128, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim + message_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, messages: Optional[torch.Tensor] = None) -> torch.Tensor:
        if messages is not None:
            x = torch.cat([x, messages], dim=-1)
        logits = self.network(x)
        return F.softmax(logits, dim=-1)


class PolicyGradientAgent(BaseAgent):
    """
    Policy Gradient agent for multi-agent RL.
    
    Uses REINFORCE algorithm with baseline for variance reduction.
    """
    
    def __init__(
        self,
        agent_id: int,
        observation_space: Any,
        action_space: Any,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        device: str = "cpu",
        communication_channel: Optional[Any] = None,
        message_dim: int = 0
    ):
        super().__init__(agent_id, observation_space, action_space, learning_rate, device, communication_channel, message_dim)
        
        self.gamma = gamma
        
        # Network dimensions
        self.input_dim = observation_space.shape[0]
        self.output_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        # Policy network
        self.policy_network = PolicyNetwork(self.input_dim, self.output_dim, self.message_dim).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Episode storage
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Training mode
        self.training = True
        
    def get_action(self, observation: np.ndarray, messages: Optional[torch.Tensor] = None, training: bool = True) -> int:
        """Select action using policy network."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            if messages is not None:
                messages = messages.unsqueeze(0).to(self.device)
            action_probs = self.policy_network(obs_tensor, messages)
            
            if training and self.training:
                # Sample from policy
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
            else:
                # Greedy action
                action = action_probs.argmax().item()
        
        return action
    
    def store_experience(
        self, 
        observation: np.ndarray, 
        action: int, 
        reward: float
    ) -> None:
        """Store experience for episode."""
        self.episode_observations.append(observation)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def update(self, batch: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Update the policy network using REINFORCE."""
        if len(self.episode_observations) == 0:
            return {"loss": 0.0}
        
        # Convert to tensors
        observations = torch.FloatTensor(self.episode_observations).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        rewards = self.episode_rewards
        
        # Calculate discounted returns
        returns = self._calculate_returns(rewards)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        action_probs = self.policy_network(observations)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        
        # REINFORCE loss
        loss = -(log_probs * returns).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []
        
        self.training_steps += 1
        
        return {"loss": loss.item()}
    
    def _calculate_returns(self, rewards: List[float]) -> List[float]:
        """Calculate discounted returns."""
        returns = []
        running_return = 0
        
        for reward in reversed(rewards):
            running_return = reward + self.gamma * running_return
            returns.insert(0, running_return)
        
        return returns
    
    def save(self, filepath: str) -> None:
        """Save the agent's model."""
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load the agent's model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
    
    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Get action probabilities for all actions."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action_probs = self.policy_network(obs_tensor)
            return action_probs.cpu().numpy().flatten()
    
    def reset_episode(self) -> None:
        """Reset agent state for new episode."""
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []