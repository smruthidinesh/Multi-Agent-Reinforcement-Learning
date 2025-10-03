"""
Deep Q-Network (DQN) Agent

Implementation of a DQN agent for multi-agent reinforcement learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import Dict, Any, Optional, Tuple
from .base_agent import BaseAgent


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture."""
    
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
        return self.network(x)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent for multi-agent RL.
    
    Uses experience replay and target network for stable learning.
    """
    
    def __init__(
        self,
        agent_id: int,
        observation_space: Any,
        action_space: Any,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        device: str = "cpu"
    ):
        super().__init__(agent_id, observation_space, action_space, learning_rate, device)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Network dimensions
        self.input_dim = observation_space.shape[0]
        self.output_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        # Networks
        self.q_network = DQNNetwork(self.input_dim, self.output_dim).to(device)
        self.target_network = DQNNetwork(self.input_dim, self.output_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay buffer
        self.buffer = deque(maxlen=buffer_size)
        
        # Training mode
        self.training = True
        
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and self.training and random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def store_experience(
        self, 
        observation: np.ndarray, 
        action: int, 
        reward: float, 
        next_observation: np.ndarray, 
        done: bool
    ) -> None:
        """Store experience in replay buffer."""
        self.buffer.append({
            'observation': observation,
            'action': action,
            'reward': reward,
            'next_observation': next_observation,
            'done': done
        })
    
    def update(self, batch: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Update the Q-network using experience replay."""
        if len(self.buffer) < self.batch_size:
            return {"loss": 0.0}
        
        # Sample batch from buffer
        if batch is None:
            batch = random.sample(self.buffer, self.batch_size)
        
        # Convert to tensors
        observations = torch.FloatTensor([exp['observation'] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        next_observations = torch.FloatTensor([exp['next_observation'] for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp['done'] for exp in batch]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(observations).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_observations).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.training and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return {"loss": loss.item(), "epsilon": self.epsilon}
    
    def save(self, filepath: str) -> None:
        """Save the agent's model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load the agent's model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
    
    def get_q_values(self, observation: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_tensor)
            return q_values.cpu().numpy().flatten()