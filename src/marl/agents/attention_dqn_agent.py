"""
Attention-based DQN Agent with TarMAC Communication

This agent uses attention mechanisms for both:
1. Communication with other agents (TarMAC)
2. Processing observations

This demonstrates state-of-the-art multi-agent communication.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import Dict, Any, Optional, Tuple, List
from .base_agent import BaseAgent
from ..utils.attention import TarMACModule, MultiHeadAttention, AttentionAggregator


class AttentionQNetwork(nn.Module):
    """
    Q-Network with attention-based communication and observation processing.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        message_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        use_communication: bool = True
    ):
        """
        Args:
            input_dim: Dimension of observations
            output_dim: Number of actions
            message_dim: Dimension of communication messages
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads
            use_communication: Whether to use inter-agent communication
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.message_dim = message_dim
        self.use_communication = use_communication

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Communication module (TarMAC)
        if use_communication:
            self.comm_module = TarMACModule(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                message_dim=message_dim,
                num_heads=num_heads
            )

        # Q-value head
        q_input_dim = hidden_dim + message_dim if use_communication else hidden_dim
        self.q_head = nn.Sequential(
            nn.Linear(q_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(
        self,
        observation: torch.Tensor,
        other_messages: Optional[torch.Tensor] = None,
        comm_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with optional communication.

        Args:
            observation: Agent observations [batch_size, input_dim]
            other_messages: Messages from other agents [batch_size, n_agents-1, message_dim]
            comm_mask: Communication mask [batch_size, 1, n_agents-1]

        Returns:
            q_values: Q-values for each action [batch_size, output_dim]
            own_message: Generated message [batch_size, message_dim]
            attention_weights: Attention over other agents' messages
        """
        # Encode observation
        encoded_obs = self.obs_encoder(observation)

        own_message = None
        attention_weights = None
        aggregated_message = None

        # Communication phase
        if self.use_communication:
            # Generate message from encoded observation
            own_message = self.comm_module.generate_message(encoded_obs)

            # If we received messages from other agents, process them
            if other_messages is not None and other_messages.size(1) > 0:
                aggregated_message, attention_weights = self.comm_module.process_messages(
                    own_message=own_message,
                    other_messages=other_messages,
                    mask=comm_mask
                )
            else:
                # No other agents, use own message
                aggregated_message = own_message

        # Combine encoded observation with communication
        if self.use_communication and aggregated_message is not None:
            combined_features = torch.cat([encoded_obs, aggregated_message], dim=-1)
        else:
            combined_features = encoded_obs

        # Compute Q-values
        q_values = self.q_head(combined_features)

        return q_values, own_message, attention_weights


class AttentionDQNAgent(BaseAgent):
    """
    DQN Agent with attention-based communication (TarMAC).

    Features:
    - Learned message generation
    - Attention-based message aggregation
    - Gated communication integration
    - Experience replay with communication
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
        message_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        use_communication: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize attention-based DQN agent.

        Args:
            agent_id: Unique agent identifier
            observation_space: Observation space
            action_space: Action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            message_dim: Dimension of communication messages
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads
            use_communication: Whether to enable communication
            device: Device for training (cpu/cuda)
        """
        super().__init__(
            agent_id=agent_id,
            observation_space=observation_space,
            action_space=action_space,
            learning_rate=learning_rate,
            device=device,
            message_dim=message_dim
        )

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_communication = use_communication

        # Network dimensions
        self.input_dim = observation_space.shape[0]
        self.output_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]

        # Networks
        self.q_network = AttentionQNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            message_dim=message_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            use_communication=use_communication
        ).to(device)

        self.target_network = AttentionQNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            message_dim=message_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            use_communication=use_communication
        ).to(device)

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience replay buffer
        self.buffer = deque(maxlen=buffer_size)

        # Statistics
        self.last_message = None
        self.last_attention_weights = None
        self.training = True

    def get_action(
        self,
        observation: np.ndarray,
        other_messages: Optional[torch.Tensor] = None,
        comm_mask: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Tuple[int, Optional[torch.Tensor]]:
        """
        Select action using epsilon-greedy policy.

        Args:
            observation: Current observation
            other_messages: Messages from other agents
            comm_mask: Communication mask
            training: Whether in training mode

        Returns:
            action: Selected action
            message: Generated message for other agents
        """
        # Epsilon-greedy exploration
        if training and self.training and random.random() < self.epsilon:
            action = random.randint(0, self.output_dim - 1)

            # Still generate message even during random action
            if self.use_communication:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                    _, message, _ = self.q_network(obs_tensor, None, None)
                    self.last_message = message
                    return action, message.squeeze(0) if message is not None else None
            else:
                return action, None

        # Greedy action selection
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

            # Forward pass through Q-network
            q_values, message, attention_weights = self.q_network(
                obs_tensor,
                other_messages.unsqueeze(0) if other_messages is not None else None,
                comm_mask.unsqueeze(0) if comm_mask is not None else None
            )

            action = q_values.argmax().item()

            # Store for analysis
            self.last_message = message
            self.last_attention_weights = attention_weights

        return action, message.squeeze(0) if message is not None else None

    def store_experience(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        other_messages: Optional[torch.Tensor] = None,
        next_other_messages: Optional[torch.Tensor] = None
    ) -> None:
        """
        Store experience in replay buffer.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is done
            other_messages: Messages from other agents at current step
            next_other_messages: Messages from other agents at next step
        """
        self.buffer.append({
            'observation': observation,
            'action': action,
            'reward': reward,
            'next_observation': next_observation,
            'done': done,
            'other_messages': other_messages.cpu() if other_messages is not None else None,
            'next_other_messages': next_other_messages.cpu() if next_other_messages is not None else None
        })

    def update(self, batch: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Update Q-network using experience replay.

        Returns:
            metrics: Dictionary of training metrics
        """
        if len(self.buffer) < self.batch_size:
            return {"loss": 0.0, "epsilon": self.epsilon}

        # Sample batch from buffer
        if batch is None:
            batch_data = random.sample(self.buffer, self.batch_size)
        else:
            batch_data = batch

        # Convert to tensors
        observations = torch.FloatTensor(
            np.array([exp['observation'] for exp in batch_data])
        ).to(self.device)

        actions = torch.LongTensor(
            np.array([exp['action'] for exp in batch_data])
        ).to(self.device)

        rewards = torch.FloatTensor(
            np.array([exp['reward'] for exp in batch_data])
        ).to(self.device)

        next_observations = torch.FloatTensor(
            np.array([exp['next_observation'] for exp in batch_data])
        ).to(self.device)

        dones = torch.BoolTensor(
            np.array([exp['done'] for exp in batch_data])
        ).to(self.device)

        # Handle communication messages
        other_messages = None
        next_other_messages = None

        if self.use_communication:
            # Collect messages (handle None values)
            messages_list = [exp['other_messages'] for exp in batch_data]
            if any(m is not None for m in messages_list):
                # Pad None messages with zeros
                max_agents = max(m.size(0) if m is not None else 0 for m in messages_list)
                if max_agents > 0:
                    padded_messages = []
                    for m in messages_list:
                        if m is None:
                            padded_messages.append(torch.zeros(max_agents, self.message_dim))
                        else:
                            padded_messages.append(m)
                    other_messages = torch.stack(padded_messages).to(self.device)

            next_messages_list = [exp['next_other_messages'] for exp in batch_data]
            if any(m is not None for m in next_messages_list):
                max_agents = max(m.size(0) if m is not None else 0 for m in next_messages_list)
                if max_agents > 0:
                    padded_next_messages = []
                    for m in next_messages_list:
                        if m is None:
                            padded_next_messages.append(torch.zeros(max_agents, self.message_dim))
                        else:
                            padded_next_messages.append(m)
                    next_other_messages = torch.stack(padded_next_messages).to(self.device)

        # Current Q-values
        current_q_values, _, _ = self.q_network(observations, other_messages, None)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))

        # Next Q-values from target network
        with torch.no_grad():
            next_q_values, _, _ = self.target_network(next_observations, next_other_messages, None)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

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

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get last attention weights for visualization."""
        return self.last_attention_weights

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
