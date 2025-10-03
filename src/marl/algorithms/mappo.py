"""
Multi-Agent Proximal Policy Optimization (MAPPO)

Implementation of MAPPO algorithm for multi-agent RL.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import random


class ActorNetwork(nn.Module):
    """Actor network for MAPPO."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        return F.softmax(logits, dim=-1)


class CriticNetwork(nn.Module):
    """Critic network for MAPPO."""
    
    def __init__(self, state_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
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


class MAPPOAgent:
    """MAPPO agent for multi-agent RL."""
    
    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cpu"
    ):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_log_probs = []
        
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """Select action using actor network."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
            if training:
                # Sample from policy
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            else:
                # Greedy action
                action = action_probs.argmax()
                log_prob = torch.log(action_probs[0, action] + 1e-8)
            
            return action.item(), log_prob.item(), value.item()
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float
    ) -> None:
        """Store experience for episode."""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_values.append(value)
        self.episode_log_probs.append(log_prob)
    
    def update(self) -> Dict[str, float]:
        """Update actor and critic networks using PPO."""
        if len(self.episode_states) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0}
        
        # Convert to tensors
        states = torch.FloatTensor(self.episode_states).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        rewards = self.episode_rewards
        values = torch.FloatTensor(self.episode_values).to(self.device)
        old_log_probs = torch.FloatTensor(self.episode_log_probs).to(self.device)
        
        # Calculate advantages and returns
        advantages, returns = self._calculate_gae(rewards, values)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update actor
        actor_loss = self._update_actor(states, actions, old_log_probs, advantages)
        
        # Update critic
        critic_loss = self._update_critic(states, returns)
        
        # Clear episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_log_probs = []
        
        return {"actor_loss": actor_loss, "critic_loss": critic_loss}
    
    def _calculate_gae(self, rewards: List[float], values: torch.Tensor) -> Tuple[List[float], List[float]]:
        """Calculate Generalized Advantage Estimation."""
        advantages = []
        returns = []
        
        # Add bootstrap value for last step
        rewards_with_bootstrap = rewards + [0.0]
        values_with_bootstrap = torch.cat([values, torch.zeros(1).to(self.device)])
        
        running_advantage = 0
        running_return = 0
        
        for i in range(len(rewards) - 1, -1, -1):
            # Calculate TD error
            td_error = rewards_with_bootstrap[i] + self.gamma * values_with_bootstrap[i + 1] - values_with_bootstrap[i]
            
            # Calculate advantage using GAE
            running_advantage = td_error + self.gamma * self.lambda_gae * running_advantage
            advantages.insert(0, running_advantage)
            
            # Calculate return
            running_return = rewards_with_bootstrap[i] + self.gamma * running_return
            returns.insert(0, running_return)
        
        return advantages, returns
    
    def _update_actor(self, states: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor) -> float:
        """Update actor network using PPO."""
        # Get current policy
        action_probs = self.actor(states)
        action_dist = torch.distributions.Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)
        
        # Calculate ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Calculate clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Add entropy bonus
        entropy = action_dist.entropy().mean()
        actor_loss -= self.entropy_coef * entropy
        
        # Optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_critic(self, states: torch.Tensor, returns: torch.Tensor) -> float:
        """Update critic network."""
        # Get current value estimates
        values = self.critic(states).squeeze()
        
        # Calculate loss
        critic_loss = F.mse_loss(values, returns)
        
        # Optimize
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()


class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization algorithm.
    
    Each agent learns using PPO with centralized value function estimation.
    """
    
    def __init__(
        self,
        env,
        n_agents: int,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cpu"
    ):
        self.env = env
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Create agents
        self.agents = {}
        for agent_id in range(n_agents):
            self.agents[agent_id] = MAPPOAgent(
                agent_id=agent_id,
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=learning_rate,
                gamma=gamma,
                lambda_gae=lambda_gae,
                clip_ratio=clip_ratio,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                device=device
            )
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = []
        
    def train(self, n_episodes: int, max_steps_per_episode: int = 100) -> Dict[str, List[float]]:
        """Train the agents."""
        print(f"Training MAPPO for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            episode_reward = 0
            episode_length = 0
            
            # Reset environment
            observations, _ = self.env.reset()
            
            for step in range(max_steps_per_episode):
                # Select actions
                actions = {}
                for agent_id, obs in observations.items():
                    action, log_prob, value = self.agents[agent_id].select_action(obs, training=True)
                    actions[agent_id] = action
                
                # Execute actions
                next_observations, rewards, terminated, truncated, _ = self.env.step(actions)
                
                # Store experiences
                for agent_id in range(self.n_agents):
                    self.agents[agent_id].store_experience(
                        observations[agent_id],
                        actions[agent_id],
                        rewards[agent_id],
                        self.agents[agent_id].episode_values[-1] if self.agents[agent_id].episode_values else 0.0,
                        self.agents[agent_id].episode_log_probs[-1] if self.agents[agent_id].episode_log_probs else 0.0
                    )
                
                # Update statistics
                episode_reward += sum(rewards.values())
                episode_length += 1
                
                # Check termination
                if all(terminated.values()) or all(truncated.values()):
                    break
                
                observations = next_observations
            
            # Update agents
            for agent_id in range(self.n_agents):
                metrics = self.agents[agent_id].update()
                if not self.training_metrics:
                    self.training_metrics.append({})
                self.training_metrics[-1][f"agent_{agent_id}_actor_loss"] = metrics["actor_loss"]
                self.training_metrics[-1][f"agent_{agent_id}_critic_loss"] = metrics["critic_loss"]
            
            # Store episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
        
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_metrics": self.training_metrics
        }
    
    def evaluate(self, n_episodes: int = 10, max_steps_per_episode: int = 100) -> Dict[str, float]:
        """Evaluate the trained agents."""
        print(f"Evaluating MAPPO for {n_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            episode_reward = 0
            episode_length = 0
            
            # Reset environment
            observations, _ = self.env.reset()
            
            for step in range(max_steps_per_episode):
                # Select actions (greedy)
                actions = {}
                for agent_id, obs in observations.items():
                    action, _, _ = self.agents[agent_id].select_action(obs, training=False)
                    actions[agent_id] = action
                
                # Execute actions
                next_observations, rewards, terminated, truncated, _ = self.env.step(actions)
                
                # Update statistics
                episode_reward += sum(rewards.values())
                episode_length += 1
                
                # Check termination
                if all(terminated.values()) or all(truncated.values()):
                    break
                
                observations = next_observations
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "avg_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths)
        }
    
    def save_models(self, filepath: str) -> None:
        """Save all agent models."""
        for agent_id, agent in self.agents.items():
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict()
            }, f"{filepath}_agent_{agent_id}.pth")
    
    def load_models(self, filepath: str) -> None:
        """Load all agent models."""
        for agent_id, agent in self.agents.items():
            checkpoint = torch.load(f"{filepath}_agent_{agent_id}.pth", map_location=self.device)
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])