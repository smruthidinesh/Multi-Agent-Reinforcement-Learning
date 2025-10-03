"""
Actor-Critic Agent

Implementation of an Actor-Critic agent for multi-agent reinforcement learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent


class ActorNetwork(nn.Module):
    """Actor network for policy."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [128, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        return F.softmax(logits, dim=-1)


class CriticNetwork(nn.Module):
    """Critic network for value function."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ActorCriticAgent(BaseAgent):
    """
    Actor-Critic agent for multi-agent RL.
    
    Uses separate networks for policy (actor) and value function (critic).
    """
    
    def __init__(
        self,
        agent_id: int,
        observation_space: Any,
        action_space: Any,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        device: str = "cpu"
    ):
        super().__init__(agent_id, observation_space, action_space, learning_rate, device)
        
        self.gamma = gamma
        
        # Network dimensions
        self.input_dim = observation_space.shape[0]
        self.output_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        # Networks
        self.actor_network = ActorNetwork(self.input_dim, self.output_dim).to(device)
        self.critic_network = CriticNetwork(self.input_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=learning_rate)
        
        # Episode storage
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        
        # Training mode
        self.training = True
        
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Select action using actor network."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action_probs = self.actor_network(obs_tensor)
            
            if training and self.training:
                # Sample from policy
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
            else:
                # Greedy action
                action = action_probs.argmax().item()
        
        return action
    
    def get_value(self, observation: np.ndarray) -> float:
        """Get value estimate for observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            value = self.critic_network(obs_tensor)
            return value.item()
    
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
        
        # Store value estimate
        value = self.get_value(observation)
        self.episode_values.append(value)
    
    def update(self, batch: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Update both actor and critic networks."""
        if len(self.episode_observations) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0}
        
        # Convert to tensors
        observations = torch.FloatTensor(self.episode_observations).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        rewards = self.episode_rewards
        values = torch.FloatTensor(self.episode_values).to(self.device)
        
        # Calculate advantages
        advantages = self._calculate_advantages(rewards, values)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update critic
        critic_loss = self._update_critic(observations, rewards, values)
        
        # Update actor
        actor_loss = self._update_actor(observations, actions, advantages)
        
        # Clear episode data
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        
        self.training_steps += 1
        
        return {"actor_loss": actor_loss, "critic_loss": critic_loss}
    
    def _calculate_advantages(self, rewards: List[float], values: torch.Tensor) -> List[float]:
        """Calculate advantages using GAE."""
        advantages = []
        running_advantage = 0
        
        for i in range(len(rewards) - 1, -1, -1):
            if i == len(rewards) - 1:
                # Last step
                running_advantage = rewards[i] - values[i].item()
            else:
                # Regular step
                running_advantage = rewards[i] + self.gamma * values[i + 1].item() - values[i].item()
                running_advantage += self.gamma * running_advantage
            
            advantages.insert(0, running_advantage)
        
        return advantages
    
    def _update_critic(self, observations: torch.Tensor, rewards: List[float], values: torch.Tensor) -> float:
        """Update critic network."""
        # Calculate target values
        target_values = []
        running_return = 0
        
        for reward in reversed(rewards):
            running_return = reward + self.gamma * running_return
            target_values.insert(0, running_return)
        
        target_values = torch.FloatTensor(target_values).to(self.device)
        
        # Calculate loss
        predicted_values = self.critic_network(observations).squeeze()
        critic_loss = F.mse_loss(predicted_values, target_values)
        
        # Optimize
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(self, observations: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor) -> float:
        """Update actor network."""
        # Calculate policy loss
        action_probs = self.actor_network(observations)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        
        # Actor loss
        actor_loss = -(log_probs * advantages).mean()
        
        # Optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def save(self, filepath: str) -> None:
        """Save the agent's model."""
        torch.save({
            'actor_network_state_dict': self.actor_network.state_dict(),
            'critic_network_state_dict': self.critic_network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_steps': self.training_steps
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load the agent's model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_network.load_state_dict(checkpoint['actor_network_state_dict'])
        self.critic_network.load_state_dict(checkpoint['critic_network_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
    
    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Get action probabilities for all actions."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action_probs = self.actor_network(obs_tensor)
            return action_probs.cpu().numpy().flatten()
    
    def reset_episode(self) -> None:
        """Reset agent state for new episode."""
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []