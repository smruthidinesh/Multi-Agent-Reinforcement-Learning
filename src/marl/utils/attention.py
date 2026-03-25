"""
Attention Mechanisms for Multi-Agent Communication

This module implements various attention mechanisms for multi-agent learning:
1. Multi-head attention for agent-agent communication
2. Targeted communication (TarMAC-style)
3. Self-attention for observation processing

References:
- "TarMAC: Targeted Multi-Agent Communication" (Das et al., 2019)
- "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for agent communication.

    This allows agents to selectively attend to messages from other agents,
    learning which agents' information is most relevant for decision-making.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor [batch_size, query_len, embed_dim]
            key: Key tensor [batch_size, key_len, embed_dim]
            value: Value tensor [batch_size, value_len, embed_dim]
            mask: Optional attention mask [batch_size, query_len, key_len]

        Returns:
            output: Attention output [batch_size, query_len, embed_dim]
            attention_weights: Attention weights [batch_size, num_heads, query_len, key_len]
        """
        batch_size = query.size(0)

        # Linear projections and reshape to [batch_size, num_heads, seq_len, head_dim]
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            # Expand mask for multiple heads
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, query_len, key_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output)

        return output, attention_weights


class TarMACModule(nn.Module):
    """
    Targeted Multi-Agent Communication (TarMAC) module.

    This implements the TarMAC architecture which allows agents to:
    1. Generate signatures (what to communicate)
    2. Use attention to selectively process messages from other agents
    3. Learn when and what to communicate

    Reference: "TarMAC: Targeted Multi-Agent Communication" (Das et al., 2019)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        message_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of agent observations
            hidden_dim: Hidden dimension for message generation
            message_dim: Dimension of messages
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim

        # Message generation network (signature generation)
        self.message_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, message_dim)
        )

        # Multi-head attention for processing received messages
        self.attention = MultiHeadAttention(
            embed_dim=message_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(message_dim)

        # Gate network to control message influence
        self.gate = nn.Sequential(
            nn.Linear(message_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim),
            nn.Sigmoid()
        )

    def generate_message(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Generate a message (signature) from observation.

        Args:
            observation: Agent's observation [batch_size, input_dim]

        Returns:
            message: Generated message [batch_size, message_dim]
        """
        return self.message_encoder(observation)

    def process_messages(
        self,
        own_message: torch.Tensor,
        other_messages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process messages from other agents using attention.

        Args:
            own_message: Own agent's message [batch_size, message_dim]
            other_messages: Messages from other agents [batch_size, n_agents, message_dim]
            mask: Optional mask for unavailable agents [batch_size, 1, n_agents]

        Returns:
            aggregated_message: Aggregated message [batch_size, message_dim]
            attention_weights: Attention weights [batch_size, num_heads, 1, n_agents]
        """
        batch_size = own_message.size(0)

        # Use own message as query
        query = own_message.unsqueeze(1)  # [batch_size, 1, message_dim]

        # Apply attention over other agents' messages
        attended_messages, attention_weights = self.attention(
            query=query,
            key=other_messages,
            value=other_messages,
            mask=mask
        )

        # Squeeze to remove sequence dimension
        attended_messages = attended_messages.squeeze(1)  # [batch_size, message_dim]

        # Gated integration of attended messages with own message
        gate_input = torch.cat([own_message, attended_messages], dim=-1)
        gate_values = self.gate(gate_input)

        # Combine own message with attended messages
        aggregated_message = gate_values * attended_messages + (1 - gate_values) * own_message

        # Layer normalization
        aggregated_message = self.layer_norm(aggregated_message)

        return aggregated_message, attention_weights


class CommNetModule(nn.Module):
    """
    Communication Network (CommNet) module.

    Simple averaging-based communication where each agent broadcasts
    its hidden state and receives the average of all other agents.

    Reference: "Learning Multiagent Communication with Backpropagation" (Sukhbaatar et al., 2016)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        message_dim: int
    ):
        """
        Args:
            input_dim: Dimension of agent observations
            hidden_dim: Hidden dimension
            message_dim: Dimension of messages
        """
        super().__init__()

        self.message_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )

        self.message_processor = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )

    def forward(
        self,
        observation: torch.Tensor,
        other_messages: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of CommNet.

        Args:
            observation: Agent's observation [batch_size, input_dim]
            other_messages: Messages from other agents [batch_size, n_agents, message_dim]

        Returns:
            output: Processed message [batch_size, message_dim]
        """
        # Generate own message
        own_message = self.message_encoder(observation)

        # Average other agents' messages
        avg_message = other_messages.mean(dim=1)

        # Process combined information
        combined = own_message + avg_message
        output = self.message_processor(combined)

        return output


class SelfAttentionEncoder(nn.Module):
    """
    Self-attention encoder for processing observations.

    Useful for agents with structured observations (e.g., multiple entities)
    where attention can help focus on relevant parts.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of attention layers
            dropout: Dropout probability
        """
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Stack of attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        self.feed_forward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of self-attention encoder.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask

        Returns:
            output: Encoded tensor [batch_size, seq_len, hidden_dim]
        """
        # Project input
        x = self.input_projection(x)

        # Apply attention layers
        for attention, layer_norm, ff in zip(
            self.attention_layers,
            self.layer_norms,
            self.feed_forward
        ):
            # Self-attention with residual connection
            attended, _ = attention(x, x, x, mask)
            x = layer_norm(x + self.dropout(attended))

            # Feed-forward with residual connection
            x = layer_norm(x + self.dropout(ff(x)))

        return x


class AttentionAggregator(nn.Module):
    """
    Simple attention-based aggregation of multiple inputs.

    Useful for aggregating information from multiple sources
    (e.g., multiple agents, multiple time steps).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Args:
            input_dim: Dimension of input vectors
            hidden_dim: Hidden dimension for attention computation
        """
        super().__init__()

        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate inputs using attention weights.

        Args:
            inputs: Input vectors [batch_size, n_items, input_dim]
            mask: Optional mask [batch_size, n_items]

        Returns:
            aggregated: Weighted sum [batch_size, input_dim]
            weights: Attention weights [batch_size, n_items]
        """
        # Compute attention scores
        scores = self.attention_net(inputs).squeeze(-1)  # [batch_size, n_items]

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        weights = F.softmax(scores, dim=-1)  # [batch_size, n_items]

        # Weighted sum
        aggregated = torch.sum(inputs * weights.unsqueeze(-1), dim=1)

        return aggregated, weights
