"""
Intrinsic Curiosity Module and Exploration Bonuses

This module implements various intrinsic motivation mechanisms for exploration:
1. Intrinsic Curiosity Module (ICM) - prediction error as curiosity
2. Random Network Distillation (RND) - novelty detection
3. Count-based exploration bonuses

References:
- "Curiosity-driven Exploration by Self-supervised Prediction" (Pathak et al., 2017)
- "Exploration by Random Network Distillation" (Burda et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np
from collections import defaultdict


class InverseDynamicsModel(nn.Module):
    """
    Inverse dynamics model that predicts action from state transitions.
    Used in ICM to learn meaningful state representations.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ):
        """
        Args:
            state_dim: State/observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict action from state transition.

        Args:
            state: Current state [batch_size, state_dim]
            next_state: Next state [batch_size, state_dim]

        Returns:
            action_logits: Predicted action logits [batch_size, action_dim]
        """
        concatenated = torch.cat([state, next_state], dim=-1)
        return self.network(concatenated)


class ForwardDynamicsModel(nn.Module):
    """
    Forward dynamics model that predicts next state from current state and action.
    Prediction error is used as intrinsic reward (curiosity).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ):
        """
        Args:
            state_dim: State/observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.action_dim = action_dim

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next state.

        Args:
            state: Current state [batch_size, state_dim]
            action: Action (one-hot or embedding) [batch_size, action_dim]

        Returns:
            next_state_pred: Predicted next state [batch_size, state_dim]
        """
        # If action is integer, convert to one-hot
        if action.dtype == torch.long:
            action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
        else:
            action_one_hot = action

        concatenated = torch.cat([state, action_one_hot], dim=-1)
        return self.network(concatenated)


class StateEncoder(nn.Module):
    """
    Encodes raw observations into feature representations.
    Used in ICM to learn task-relevant features.
    """

    def __init__(
        self,
        obs_dim: int,
        feature_dim: int,
        hidden_dim: int = 128
    ):
        """
        Args:
            obs_dim: Observation dimension
            feature_dim: Feature representation dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Encode observation to features.

        Args:
            observation: Raw observation [batch_size, obs_dim]

        Returns:
            features: Encoded features [batch_size, feature_dim]
        """
        return self.network(observation)


class IntrinsicCuriosityModule(nn.Module):
    """
    Intrinsic Curiosity Module (ICM).

    Generates intrinsic rewards based on prediction error of forward dynamics model.
    The idea is that the agent is curious about states where its predictions are poor.

    Reference: Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction" (2017)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        beta: float = 0.2,
        eta: float = 1.0
    ):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            feature_dim: Feature representation dimension
            hidden_dim: Hidden layer dimension
            beta: Weight for inverse model loss
            eta: Scaling factor for intrinsic reward
        """
        super().__init__()

        self.beta = beta
        self.eta = eta
        self.action_dim = action_dim

        # Feature encoder
        self.encoder = StateEncoder(obs_dim, feature_dim, hidden_dim)

        # Inverse dynamics model (predicts action)
        self.inverse_model = InverseDynamicsModel(feature_dim, action_dim, hidden_dim)

        # Forward dynamics model (predicts next state features)
        self.forward_model = ForwardDynamicsModel(feature_dim, action_dim, hidden_dim)

    def forward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute intrinsic reward and losses.

        Args:
            obs: Current observation [batch_size, obs_dim]
            next_obs: Next observation [batch_size, obs_dim]
            action: Action taken [batch_size]

        Returns:
            intrinsic_reward: Intrinsic reward [batch_size]
            losses: Dictionary of losses for training
        """
        # Encode observations
        state_features = self.encoder(obs)
        next_state_features = self.encoder(next_obs)

        # Inverse model: predict action
        action_logits = self.inverse_model(state_features, next_state_features)

        # Compute inverse model loss
        if action.dtype == torch.long:
            inverse_loss = F.cross_entropy(action_logits, action, reduction='none')
        else:
            inverse_loss = F.mse_loss(action_logits, action, reduction='none').mean(dim=-1)

        # Forward model: predict next state features
        next_state_pred = self.forward_model(state_features, action)

        # Compute forward model loss (prediction error)
        forward_loss = F.mse_loss(
            next_state_pred,
            next_state_features.detach(),
            reduction='none'
        ).mean(dim=-1)

        # Intrinsic reward is the prediction error
        intrinsic_reward = self.eta * forward_loss.detach()

        # Total ICM loss
        total_loss = (1 - self.beta) * forward_loss + self.beta * inverse_loss

        losses = {
            'icm_loss': total_loss.mean(),
            'forward_loss': forward_loss.mean(),
            'inverse_loss': inverse_loss.mean(),
            'intrinsic_reward_mean': intrinsic_reward.mean()
        }

        return intrinsic_reward, losses

    def get_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Get only intrinsic reward (for evaluation).

        Args:
            obs: Current observation [batch_size, obs_dim]
            next_obs: Next observation [batch_size, obs_dim]
            action: Action taken [batch_size]

        Returns:
            intrinsic_reward: Intrinsic reward [batch_size]
        """
        with torch.no_grad():
            intrinsic_reward, _ = self.forward(obs, next_obs, action)
        return intrinsic_reward


class RandomNetworkDistillation(nn.Module):
    """
    Random Network Distillation (RND) for exploration.

    Uses prediction error of a random target network as novelty measure.
    States that are novel (rarely visited) will have higher prediction error.

    Reference: Burda et al., "Exploration by Random Network Distillation" (2018)
    """

    def __init__(
        self,
        obs_dim: int,
        feature_dim: int = 64,
        hidden_dim: int = 128
    ):
        """
        Args:
            obs_dim: Observation dimension
            feature_dim: Feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # Target network (fixed, random)
        self.target_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        # Predictor network (trained)
        self.predictor_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        # Freeze target network
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Running statistics for normalization
        self.register_buffer('obs_mean', torch.zeros(obs_dim))
        self.register_buffer('obs_std', torch.ones(obs_dim))
        self.register_buffer('reward_mean', torch.zeros(1))
        self.register_buffer('reward_std', torch.ones(1))
        self.register_buffer('update_count', torch.zeros(1))

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations using running statistics."""
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)

    def update_stats(self, obs: torch.Tensor, rewards: torch.Tensor) -> None:
        """Update running statistics for normalization."""
        # Update observation statistics
        obs_mean = obs.mean(dim=0)
        obs_std = obs.std(dim=0)

        alpha = 0.01  # Moving average coefficient
        self.obs_mean = (1 - alpha) * self.obs_mean + alpha * obs_mean
        self.obs_std = (1 - alpha) * self.obs_std + alpha * obs_std

        # Update reward statistics
        reward_mean = rewards.mean()
        reward_std = rewards.std()

        self.reward_mean = (1 - alpha) * self.reward_mean + alpha * reward_mean
        self.reward_std = (1 - alpha) * self.reward_std + alpha * reward_std

        self.update_count += 1

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RND intrinsic reward and prediction loss.

        Args:
            obs: Observation [batch_size, obs_dim]

        Returns:
            intrinsic_reward: Intrinsic reward (prediction error)
            loss: Prediction loss for training
        """
        # Normalize observation
        obs_norm = self.normalize_obs(obs)

        # Target features (fixed)
        with torch.no_grad():
            target_features = self.target_network(obs_norm)

        # Predicted features
        predicted_features = self.predictor_network(obs_norm)

        # Prediction error
        loss = F.mse_loss(predicted_features, target_features, reduction='none').mean(dim=-1)

        # Intrinsic reward (normalized)
        intrinsic_reward = loss.detach()

        return intrinsic_reward, loss


class CountBasedExploration:
    """
    Count-based exploration bonus.

    Provides higher rewards for visiting less-frequent states.
    Uses hashing for continuous states.
    """

    def __init__(
        self,
        obs_dim: int,
        bonus_scale: float = 0.1,
        count_decay: float = 0.99
    ):
        """
        Args:
            obs_dim: Observation dimension
            bonus_scale: Scale of exploration bonus
            count_decay: Decay factor for counts (forgetting)
        """
        self.obs_dim = obs_dim
        self.bonus_scale = bonus_scale
        self.count_decay = count_decay

        # State visit counts
        self.state_counts = defaultdict(float)

    def _hash_state(self, state: np.ndarray) -> tuple:
        """Hash continuous state to discrete bin."""
        # Simple discretization: round to 1 decimal place
        return tuple(np.round(state, decimals=1))

    def get_bonus(self, state: np.ndarray) -> float:
        """
        Get exploration bonus for state.

        Args:
            state: State observation

        Returns:
            bonus: Exploration bonus (higher for less-visited states)
        """
        state_hash = self._hash_state(state)
        count = self.state_counts[state_hash]

        # Bonus inversely proportional to sqrt(count + 1)
        bonus = self.bonus_scale / np.sqrt(count + 1)

        # Update count
        self.state_counts[state_hash] += 1

        return bonus

    def decay_counts(self) -> None:
        """Decay all counts (forgetting)."""
        for key in self.state_counts:
            self.state_counts[key] *= self.count_decay


class CombinedCuriosityModule(nn.Module):
    """
    Combines multiple curiosity mechanisms.

    Can use ICM, RND, and count-based exploration together.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        use_icm: bool = True,
        use_rnd: bool = False,
        use_count: bool = False,
        icm_weight: float = 1.0,
        rnd_weight: float = 0.5,
        count_weight: float = 0.1
    ):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            use_icm: Whether to use ICM
            use_rnd: Whether to use RND
            use_count: Whether to use count-based exploration
            icm_weight: Weight for ICM rewards
            rnd_weight: Weight for RND rewards
            count_weight: Weight for count-based rewards
        """
        super().__init__()

        self.use_icm = use_icm
        self.use_rnd = use_rnd
        self.use_count = use_count

        self.icm_weight = icm_weight
        self.rnd_weight = rnd_weight
        self.count_weight = count_weight

        if use_icm:
            self.icm = IntrinsicCuriosityModule(obs_dim, action_dim)

        if use_rnd:
            self.rnd = RandomNetworkDistillation(obs_dim)

        if use_count:
            self.count_explorer = CountBasedExploration(obs_dim)

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined intrinsic reward.

        Args:
            obs: Current observation [batch_size, obs_dim]
            next_obs: Next observation [batch_size, obs_dim]
            action: Action taken [batch_size]

        Returns:
            total_reward: Combined intrinsic reward [batch_size]
            losses: Dictionary of losses and metrics
        """
        total_reward = torch.zeros(obs.size(0), device=obs.device)
        losses = {}

        # ICM reward
        if self.use_icm:
            icm_reward, icm_losses = self.icm(obs, next_obs, action)
            total_reward += self.icm_weight * icm_reward
            losses.update(icm_losses)

        # RND reward
        if self.use_rnd:
            rnd_reward, rnd_loss = self.rnd(next_obs)
            total_reward += self.rnd_weight * rnd_reward
            losses['rnd_loss'] = rnd_loss.mean()
            losses['rnd_reward_mean'] = rnd_reward.mean()

        # Count-based reward
        if self.use_count:
            count_rewards = []
            for i in range(obs.size(0)):
                state_np = obs[i].cpu().numpy()
                count_bonus = self.count_explorer.get_bonus(state_np)
                count_rewards.append(count_bonus)
            count_reward = torch.FloatTensor(count_rewards).to(obs.device)
            total_reward += self.count_weight * count_reward
            losses['count_reward_mean'] = count_reward.mean()

        losses['total_intrinsic_reward'] = total_reward.mean()

        return total_reward, losses
