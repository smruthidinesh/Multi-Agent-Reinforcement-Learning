"""
Graph Neural Network DQN Agent

This agent uses Graph Neural Networks to model relationships between agents
and enable scalable communication in large multi-agent systems.

Key Features:
- Dynamic graph construction based on agent positions
- Graph attention for selective communication
- Scalable to large numbers of agents
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
from ..utils.graph_networks import GNNEncoder, DynamicGraphConstructor


class GNNQNetwork(nn.Module):
    """
    Q-Network with Graph Neural Network for multi-agent coordination.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        gnn_hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        gnn_type: str = "gat",
        num_heads: int = 4,
        use_positions: bool = True
    ):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Number of actions
            hidden_dim: Hidden layer dimension
            gnn_hidden_dim: GNN hidden dimension
            num_gnn_layers: Number of GNN layers
            gnn_type: Type of GNN ("gat", "gcn", "mpnn")
            num_heads: Number of attention heads
            use_positions: Whether observations include positions for graph construction
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_positions = use_positions

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, gnn_hidden_dim),
            nn.ReLU()
        )

        # GNN for multi-agent communication
        self.gnn = GNNEncoder(
            input_dim=gnn_hidden_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_hidden_dim,
            num_layers=num_gnn_layers,
            gnn_type=gnn_type,
            num_heads=num_heads
        )

        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Graph constructor
        self.graph_constructor = DynamicGraphConstructor()

    def forward(
        self,
        observations: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with GNN.

        Args:
            observations: All agents' observations [batch_size, n_agents, obs_dim]
            adj: Optional adjacency matrix [batch_size, n_agents, n_agents]
            positions: Optional agent positions for graph construction [batch_size, n_agents, 2]

        Returns:
            q_values: Q-values for each agent [batch_size, n_agents, action_dim]
            adj: Adjacency matrix used [batch_size, n_agents, n_agents]
        """
        batch_size, n_agents, _ = observations.size()

        # Encode observations
        encoded_obs = self.obs_encoder(observations)  # [batch_size, n_agents, gnn_hidden_dim]

        # Construct adjacency matrix if not provided
        if adj is None:
            if positions is not None and self.use_positions:
                # Construct graph based on agent positions (k-NN)
                adj = self.graph_constructor.construct_knn_graph(positions, k=min(3, n_agents - 1))
            else:
                # Fully connected graph
                adj = torch.ones(batch_size, n_agents, n_agents, device=observations.device)
                adj = adj - torch.eye(n_agents, device=observations.device).unsqueeze(0)

        # Normalize adjacency for GCN
        if hasattr(self.gnn, 'gnn_type') and self.gnn.gnn_type == "gcn":
            adj = self.graph_constructor.normalize_adjacency(adj)

        # Apply GNN
        gnn_features = self.gnn(encoded_obs, adj)  # [batch_size, n_agents, gnn_hidden_dim]

        # Compute Q-values for each agent
        q_values = self.q_head(gnn_features)  # [batch_size, n_agents, action_dim]

        return q_values, adj


class GNNDQNAgent(BaseAgent):
    """
    DQN Agent with Graph Neural Networks for scalable multi-agent coordination.

    This agent can handle large numbers of agents by using dynamic graph
    construction and efficient message passing.
    """

    def __init__(
        self,
        agent_id: int,
        observation_space: Any,
        action_space: Any,
        n_agents: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        hidden_dim: int = 128,
        gnn_hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        gnn_type: str = "gat",
        num_heads: int = 4,
        use_positions: bool = True,
        k_neighbors: int = 3,
        device: str = "cpu"
    ):
        """
        Initialize GNN-based DQN agent.

        Args:
            agent_id: Unique agent identifier
            observation_space: Observation space
            action_space: Action space
            n_agents: Total number of agents
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Replay buffer size
            batch_size: Training batch size
            target_update_freq: Target network update frequency
            hidden_dim: Hidden layer dimension
            gnn_hidden_dim: GNN hidden dimension
            num_gnn_layers: Number of GNN layers
            gnn_type: GNN type ("gat", "gcn", "mpnn")
            num_heads: Number of attention heads
            use_positions: Whether to use positions for graph construction
            k_neighbors: Number of neighbors in k-NN graph
            device: Training device
        """
        super().__init__(
            agent_id=agent_id,
            observation_space=observation_space,
            action_space=action_space,
            learning_rate=learning_rate,
            device=device
        )

        self.n_agents = n_agents
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_positions = use_positions
        self.k_neighbors = k_neighbors

        # Network dimensions
        self.obs_dim = observation_space.shape[0]
        self.action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]

        # Shared Q-network (all agents use same network)
        self.q_network = GNNQNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gnn_layers=num_gnn_layers,
            gnn_type=gnn_type,
            num_heads=num_heads,
            use_positions=use_positions
        ).to(device)

        self.target_network = GNNQNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gnn_layers=num_gnn_layers,
            gnn_type=gnn_type,
            num_heads=num_heads,
            use_positions=use_positions
        ).to(device)

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience replay buffer
        self.buffer = deque(maxlen=buffer_size)

        # Graph constructor
        self.graph_constructor = DynamicGraphConstructor()

        # Statistics
        self.last_adj = None
        self.training = True

    def get_action(
        self,
        observation: np.ndarray,
        all_observations: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            observation: Own observation
            all_observations: All agents' observations [n_agents, obs_dim]
            positions: All agents' positions [n_agents, 2]
            training: Whether in training mode

        Returns:
            action: Selected action
        """
        # Epsilon-greedy exploration
        if training and self.training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            # Need all observations for GNN
            if all_observations is None:
                # Fallback: use single agent (no communication)
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0).to(self.device)
                pos_tensor = None
            else:
                obs_tensor = torch.FloatTensor(all_observations).unsqueeze(0).to(self.device)
                pos_tensor = torch.FloatTensor(positions).unsqueeze(0).to(self.device) if positions is not None else None

            # Forward pass
            q_values, adj = self.q_network(obs_tensor, None, pos_tensor)

            # Get Q-values for this agent
            agent_q_values = q_values[0, self.agent_id]
            action = agent_q_values.argmax().item()

            # Store adjacency for visualization
            self.last_adj = adj

        return action

    def store_experience(
        self,
        all_observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_all_observations: np.ndarray,
        dones: np.ndarray,
        positions: Optional[np.ndarray] = None,
        next_positions: Optional[np.ndarray] = None
    ) -> None:
        """
        Store experience in replay buffer.

        Note: For GNN agents, we store experiences for ALL agents together
        since the GNN requires all agents' states.

        Args:
            all_observations: All agents' observations [n_agents, obs_dim]
            actions: All agents' actions [n_agents]
            rewards: All agents' rewards [n_agents]
            next_all_observations: Next observations [n_agents, obs_dim]
            dones: Done flags [n_agents]
            positions: Agent positions [n_agents, 2]
            next_positions: Next positions [n_agents, 2]
        """
        self.buffer.append({
            'all_observations': all_observations,
            'actions': actions,
            'rewards': rewards,
            'next_all_observations': next_all_observations,
            'dones': dones,
            'positions': positions,
            'next_positions': next_positions
        })

    def update(self, batch: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Update Q-network using experience replay.

        Returns:
            metrics: Training metrics
        """
        if len(self.buffer) < self.batch_size:
            return {"loss": 0.0, "epsilon": self.epsilon}

        # Sample batch
        if batch is None:
            batch_data = random.sample(self.buffer, self.batch_size)
        else:
            batch_data = batch

        # Convert to tensors
        all_obs = torch.FloatTensor(
            np.array([exp['all_observations'] for exp in batch_data])
        ).to(self.device)  # [batch_size, n_agents, obs_dim]

        actions = torch.LongTensor(
            np.array([exp['actions'] for exp in batch_data])
        ).to(self.device)  # [batch_size, n_agents]

        rewards = torch.FloatTensor(
            np.array([exp['rewards'] for exp in batch_data])
        ).to(self.device)  # [batch_size, n_agents]

        next_all_obs = torch.FloatTensor(
            np.array([exp['next_all_observations'] for exp in batch_data])
        ).to(self.device)

        dones = torch.BoolTensor(
            np.array([exp['dones'] for exp in batch_data])
        ).to(self.device)

        # Handle positions
        positions = None
        next_positions = None
        if batch_data[0]['positions'] is not None:
            positions = torch.FloatTensor(
                np.array([exp['positions'] for exp in batch_data])
            ).to(self.device)
            next_positions = torch.FloatTensor(
                np.array([exp['next_positions'] for exp in batch_data])
            ).to(self.device)

        # Current Q-values
        current_q_values, _ = self.q_network(all_obs, None, positions)
        # Gather Q-values for taken actions
        current_q_values = current_q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        # Next Q-values from target network
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_all_obs, None, next_positions)
            next_q_values = next_q_values.max(dim=-1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss (average over all agents)
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

    def get_adjacency_matrix(self) -> Optional[torch.Tensor]:
        """Get last computed adjacency matrix for visualization."""
        return self.last_adj

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
