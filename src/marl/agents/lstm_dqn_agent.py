"""
LSTM-based DQN Agent for Partial Observability

This agent uses LSTM networks to maintain memory of past observations,
enabling it to handle partially observable environments (POMDPs).

Key Features:
- LSTM memory for temporal reasoning
- Hidden state management across episodes
- Suitable for environments with incomplete information
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


class LSTMQNetwork(nn.Module):
    """
    Q-Network with LSTM for handling partial observability.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lstm_hidden_dim: int = 128,
        num_lstm_layers: int = 1
    ):
        """
        Args:
            input_dim: Observation dimension
            action_dim: Number of actions
            hidden_dim: Fully connected hidden dimension
            lstm_hidden_dim: LSTM hidden state dimension
            num_lstm_layers: Number of LSTM layers
        """
        super().__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(
        self,
        observations: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with LSTM.

        Args:
            observations: Observations [batch_size, seq_len, input_dim]
            hidden_state: Optional LSTM hidden state (h, c)

        Returns:
            q_values: Q-values [batch_size, seq_len, action_dim]
            hidden_state: Updated LSTM hidden state (h, c)
        """
        batch_size, seq_len, _ = observations.size()

        # Encode observations
        encoded_obs = self.obs_encoder(observations)  # [batch_size, seq_len, hidden_dim]

        # LSTM processing
        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(encoded_obs)
        else:
            lstm_out, hidden_state = self.lstm(encoded_obs, hidden_state)

        # Compute Q-values
        q_values = self.q_head(lstm_out)  # [batch_size, seq_len, action_dim]

        return q_values, hidden_state

    def init_hidden(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state.

        Args:
            batch_size: Batch size
            device: Device for tensors

        Returns:
            hidden_state: Initialized hidden state (h, c)
        """
        h = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_dim).to(device)
        c = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_dim).to(device)
        return (h, c)


class LSTMDQNAgent(BaseAgent):
    """
    DQN Agent with LSTM for partial observability.

    The LSTM allows the agent to remember past observations and make
    better decisions in partially observable environments.
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
        sequence_length: int = 8,
        target_update_freq: int = 100,
        hidden_dim: int = 128,
        lstm_hidden_dim: int = 128,
        num_lstm_layers: int = 1,
        device: str = "cpu"
    ):
        """
        Initialize LSTM-based DQN agent.

        Args:
            agent_id: Unique agent identifier
            observation_space: Observation space
            action_space: Action space
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Replay buffer size
            batch_size: Training batch size
            sequence_length: Length of observation sequences for training
            target_update_freq: Target network update frequency
            hidden_dim: Hidden layer dimension
            lstm_hidden_dim: LSTM hidden dimension
            num_lstm_layers: Number of LSTM layers
            device: Training device
        """
        super().__init__(
            agent_id=agent_id,
            observation_space=observation_space,
            action_space=action_space,
            learning_rate=learning_rate,
            device=device
        )

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_update_freq = target_update_freq

        # Network dimensions
        self.input_dim = observation_space.shape[0]
        self.action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]

        # Networks
        self.q_network = LSTMQNetwork(
            input_dim=self.input_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            num_lstm_layers=num_lstm_layers
        ).to(device)

        self.target_network = LSTMQNetwork(
            input_dim=self.input_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            num_lstm_layers=num_lstm_layers
        ).to(device)

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience replay buffer (stores sequences)
        self.buffer = deque(maxlen=buffer_size)

        # Episode buffer for collecting sequences
        self.episode_buffer = []

        # LSTM hidden state
        self.hidden_state = None
        self.training = True

    def reset_episode(self) -> None:
        """Reset episode-specific state."""
        self.hidden_state = None
        # Store episode buffer in replay buffer if long enough
        if len(self.episode_buffer) >= self.sequence_length:
            # Extract sequences from episode
            for i in range(len(self.episode_buffer) - self.sequence_length + 1):
                sequence = self.episode_buffer[i:i + self.sequence_length]
                self.buffer.append(sequence)
        self.episode_buffer = []

    def get_action(
        self,
        observation: np.ndarray,
        training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy with LSTM.

        Args:
            observation: Current observation
            training: Whether in training mode

        Returns:
            action: Selected action
        """
        # Epsilon-greedy exploration
        if training and self.training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            # Prepare observation
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0).to(self.device)

            # Forward pass through network
            q_values, self.hidden_state = self.q_network(obs_tensor, self.hidden_state)

            # Select greedy action
            action = q_values[0, -1].argmax().item()

        return action

    def store_experience(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ) -> None:
        """
        Store experience in episode buffer.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is done
        """
        self.episode_buffer.append({
            'observation': observation,
            'action': action,
            'reward': reward,
            'next_observation': next_observation,
            'done': done
        })

    def update(self, batch: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Update Q-network using experience replay with sequences.

        Returns:
            metrics: Training metrics
        """
        if len(self.buffer) < self.batch_size:
            return {"loss": 0.0, "epsilon": self.epsilon}

        # Sample sequences from buffer
        if batch is None:
            batch_sequences = random.sample(self.buffer, self.batch_size)
        else:
            batch_sequences = batch

        # Convert sequences to tensors
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for sequence in batch_sequences:
            seq_obs = [step['observation'] for step in sequence]
            seq_actions = [step['action'] for step in sequence]
            seq_rewards = [step['reward'] for step in sequence]
            seq_next_obs = [step['next_observation'] for step in sequence]
            seq_dones = [step['done'] for step in sequence]

            observations.append(seq_obs)
            actions.append(seq_actions)
            rewards.append(seq_rewards)
            next_observations.append(seq_next_obs)
            dones.append(seq_dones)

        observations = torch.FloatTensor(np.array(observations)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_observations = torch.FloatTensor(np.array(next_observations)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)

        # Current Q-values
        current_q_values, _ = self.q_network(observations, None)
        current_q_values = current_q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        # Next Q-values from target network
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_observations, None)
            next_q_values = next_q_values.max(dim=-1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        if self.training and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "q_value_mean": current_q_values.mean().item()
        }

    def save(self, filepath: str) -> None:
        """Save agent model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, filepath)

    def load(self, filepath: str) -> None:
        """Load agent model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
