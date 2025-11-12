"""
Graph Neural Networks for Multi-Agent Coordination

This module implements various GNN architectures for multi-agent learning:
1. Graph Attention Networks (GAT)
2. Graph Convolutional Networks (GCN)
3. Message Passing Neural Networks (MPNN)
4. Dynamic graph construction based on agent proximity

References:
- "Graph Attention Networks" (Velickovic et al., 2018)
- "Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation" (You et al., 2018)
- "Learning to Communicate with Deep Multi-Agent Reinforcement Learning" (Foerster et al., 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).

    Allows agents to learn importance weights for their neighbors
    in the communication graph.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
            concat: Whether to concatenate or average multi-head outputs
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Learnable weight matrix
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention mechanism
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of graph attention layer.

        Args:
            h: Node features [batch_size, n_nodes, in_features]
            adj: Adjacency matrix [batch_size, n_nodes, n_nodes]

        Returns:
            h_prime: Updated node features [batch_size, n_nodes, out_features]
        """
        batch_size, n_nodes, _ = h.size()

        # Linear transformation
        Wh = torch.matmul(h, self.W)  # [batch_size, n_nodes, out_features]

        # Attention mechanism
        # Compute attention scores for all pairs
        Wh_repeated_1 = Wh.unsqueeze(2).repeat(1, 1, n_nodes, 1)  # [batch_size, n_nodes, n_nodes, out_features]
        Wh_repeated_2 = Wh.unsqueeze(1).repeat(1, n_nodes, 1, 1)  # [batch_size, n_nodes, n_nodes, out_features]

        # Concatenate for attention computation
        attention_input = torch.cat([Wh_repeated_1, Wh_repeated_2], dim=-1)  # [batch_size, n_nodes, n_nodes, 2*out_features]
        e = self.leakyrelu(torch.matmul(attention_input, self.a).squeeze(-1))  # [batch_size, n_nodes, n_nodes]

        # Mask attention for non-existing edges
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # Softmax normalization
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Aggregate neighbor features
        h_prime = torch.matmul(attention, Wh)  # [batch_size, n_nodes, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head Graph Attention Network.

    Applies multiple attention heads to learn different aspects
    of agent relationships.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension per head
            num_heads: Number of attention heads
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
        """
        super().__init__()

        self.num_heads = num_heads

        # Multiple attention heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(
                in_features,
                out_features,
                dropout=dropout,
                alpha=alpha,
                concat=True
            )
            for _ in range(num_heads)
        ])

        # Output projection
        self.out_proj = nn.Linear(num_heads * out_features, out_features)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of multi-head graph attention.

        Args:
            h: Node features [batch_size, n_nodes, in_features]
            adj: Adjacency matrix [batch_size, n_nodes, n_nodes]

        Returns:
            h_out: Updated node features [batch_size, n_nodes, out_features]
        """
        # Apply all attention heads
        head_outputs = [att(h, adj) for att in self.attentions]

        # Concatenate outputs from all heads
        h_concat = torch.cat(head_outputs, dim=-1)

        # Project to output dimension
        h_out = self.out_proj(h_concat)

        return h_out


class GraphConvolutionLayer(nn.Module):
    """
    Graph Convolutional Layer (GCN).

    Simple but effective layer for aggregating neighbor information.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of graph convolution.

        Args:
            h: Node features [batch_size, n_nodes, in_features]
            adj: Normalized adjacency matrix [batch_size, n_nodes, n_nodes]

        Returns:
            h_out: Updated node features [batch_size, n_nodes, out_features]
        """
        # Linear transformation
        support = torch.matmul(h, self.weight)

        # Graph convolution
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        return output


class MessagePassingLayer(nn.Module):
    """
    Message Passing Neural Network layer.

    Agents pass messages to neighbors and update their states
    based on aggregated messages.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        message_dim: int,
        hidden_dim: int = 128
    ):
        """
        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            message_dim: Message dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # Message generation network
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )

        # Node update network
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of message passing.

        Args:
            h: Node features [batch_size, n_nodes, node_dim]
            adj: Adjacency matrix [batch_size, n_nodes, n_nodes]
            edge_features: Optional edge features [batch_size, n_nodes, n_nodes, edge_dim]

        Returns:
            h_updated: Updated node features [batch_size, n_nodes, node_dim]
        """
        batch_size, n_nodes, node_dim = h.size()

        # Generate messages for all edges
        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)  # [batch_size, n_nodes, n_nodes, node_dim]
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)  # [batch_size, n_nodes, n_nodes, node_dim]

        if edge_features is not None:
            message_input = torch.cat([h_i, h_j, edge_features], dim=-1)
        else:
            # Use adjacency as simple edge feature
            edge_feat = adj.unsqueeze(-1)
            message_input = torch.cat([h_i, h_j, edge_feat], dim=-1)

        messages = self.message_net(message_input)  # [batch_size, n_nodes, n_nodes, message_dim]

        # Mask messages for non-existing edges
        adj_expanded = adj.unsqueeze(-1)
        messages = messages * adj_expanded

        # Aggregate messages
        aggregated_messages = messages.sum(dim=2)  # [batch_size, n_nodes, message_dim]

        # Update node features
        update_input = torch.cat([h, aggregated_messages], dim=-1)
        h_updated = self.update_net(update_input)

        return h_updated


class GNNEncoder(nn.Module):
    """
    Full GNN encoder with multiple layers.

    Stacks multiple graph neural network layers for deep feature learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        gnn_type: str = "gat",
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN layer ("gat", "gcn", "mpnn")
            num_heads: Number of attention heads (for GAT)
            dropout: Dropout probability
        """
        super().__init__()

        self.num_layers = num_layers
        self.gnn_type = gnn_type

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            if gnn_type == "gat":
                # For last layer, output to output_dim
                out_dim = output_dim if i == num_layers - 1 else hidden_dim
                self.gnn_layers.append(
                    MultiHeadGraphAttention(
                        hidden_dim,
                        out_dim // num_heads,
                        num_heads=num_heads,
                        dropout=dropout
                    )
                )
            elif gnn_type == "gcn":
                out_dim = output_dim if i == num_layers - 1 else hidden_dim
                self.gnn_layers.append(
                    GraphConvolutionLayer(hidden_dim, out_dim)
                )
            elif gnn_type == "mpnn":
                self.gnn_layers.append(
                    MessagePassingLayer(
                        node_dim=hidden_dim,
                        edge_dim=1,
                        message_dim=hidden_dim
                    )
                )

            self.layer_norms.append(nn.LayerNorm(hidden_dim if i < num_layers - 1 else output_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of GNN encoder.

        Args:
            h: Node features [batch_size, n_nodes, input_dim]
            adj: Adjacency matrix [batch_size, n_nodes, n_nodes]

        Returns:
            h_out: Encoded node features [batch_size, n_nodes, output_dim]
        """
        # Input projection
        h = self.input_proj(h)
        h = F.relu(h)

        # Apply GNN layers
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h_new = gnn_layer(h, adj)

            # Residual connection (except for last layer with different dimension)
            if i < self.num_layers - 1:
                h = layer_norm(h + self.dropout(h_new))
                h = F.relu(h)
            else:
                h = layer_norm(h_new)

        return h


class DynamicGraphConstructor:
    """
    Constructs dynamic communication graphs based on agent states.

    Useful for creating sparse communication patterns in large-scale
    multi-agent systems.
    """

    @staticmethod
    def construct_knn_graph(
        positions: torch.Tensor,
        k: int = 3
    ) -> torch.Tensor:
        """
        Construct k-nearest neighbor graph based on positions.

        Args:
            positions: Agent positions [batch_size, n_agents, position_dim]
            k: Number of nearest neighbors

        Returns:
            adj: Adjacency matrix [batch_size, n_agents, n_agents]
        """
        batch_size, n_agents, _ = positions.size()

        # Compute pairwise distances
        pos_i = positions.unsqueeze(2)  # [batch_size, n_agents, 1, position_dim]
        pos_j = positions.unsqueeze(1)  # [batch_size, 1, n_agents, position_dim]
        distances = torch.norm(pos_i - pos_j, dim=-1)  # [batch_size, n_agents, n_agents]

        # Find k nearest neighbors (excluding self)
        distances_no_self = distances + torch.eye(n_agents, device=distances.device).unsqueeze(0) * 1e10
        _, indices = torch.topk(distances_no_self, k, dim=-1, largest=False)

        # Create adjacency matrix
        adj = torch.zeros(batch_size, n_agents, n_agents, device=positions.device)
        batch_indices = torch.arange(batch_size, device=positions.device).view(-1, 1, 1).expand(-1, n_agents, k)
        agent_indices = torch.arange(n_agents, device=positions.device).view(1, -1, 1).expand(batch_size, -1, k)

        adj[batch_indices, agent_indices, indices] = 1.0

        # Make symmetric (undirected graph)
        adj = (adj + adj.transpose(-2, -1)).clamp(max=1.0)

        # Add self-connections
        adj = adj + torch.eye(n_agents, device=positions.device).unsqueeze(0)

        return adj

    @staticmethod
    def construct_distance_graph(
        positions: torch.Tensor,
        threshold: float = 5.0
    ) -> torch.Tensor:
        """
        Construct graph based on distance threshold.

        Args:
            positions: Agent positions [batch_size, n_agents, position_dim]
            threshold: Distance threshold for connectivity

        Returns:
            adj: Adjacency matrix [batch_size, n_agents, n_agents]
        """
        batch_size, n_agents, _ = positions.size()

        # Compute pairwise distances
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        distances = torch.norm(pos_i - pos_j, dim=-1)

        # Create adjacency based on threshold
        adj = (distances < threshold).float()

        # Ensure self-connections
        adj = adj + torch.eye(n_agents, device=positions.device).unsqueeze(0)
        adj = adj.clamp(max=1.0)

        return adj

    @staticmethod
    def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
        """
        Normalize adjacency matrix for GCN.

        Computes: D^(-1/2) * A * D^(-1/2)

        Args:
            adj: Adjacency matrix [batch_size, n_nodes, n_nodes]

        Returns:
            adj_norm: Normalized adjacency matrix
        """
        # Compute degree matrix
        degree = adj.sum(dim=-1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0

        # D^(-1/2) * A * D^(-1/2)
        degree_matrix = torch.diag_embed(degree_inv_sqrt)
        adj_norm = torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix)

        return adj_norm
