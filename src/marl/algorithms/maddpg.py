"""
Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

Implementation of MADDPG algorithm for multi-agent RL with continuous actions.
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
    """Actor network for MADDPG."""
    
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
        layers.append(nn.Tanh())  # Bound actions to [-1, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CriticNetwork(nn.Module):
    """Critic network for MADDPG."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        # Input: state + action
        input_dim = state_dim + action_dim
        
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
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class MADDPGAgent:
    """MADDPG agent for multi-agent RL."""
    
    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu"
    ):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.target_actor = ActorNetwork(state_dim, action_dim).to(device)
        self.target_critic = CriticNetwork(state_dim, action_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Copy weights to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Noise for exploration
        self.noise_scale = 0.1
        
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using actor network."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy().flatten()
            
            if training:
                # Add noise for exploration
                noise = np.random.normal(0, self.noise_scale, size=action.shape)
                action = np.clip(action + noise, -1, 1)
            
            return action
    
    def update(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        rewards: torch.Tensor, 
        next_states: torch.Tensor, 
        dones: torch.Tensor,
        all_states: torch.Tensor,
        all_actions: torch.Tensor,
        all_next_states: torch.Tensor
    ) -> Dict[str, float]:
        """Update actor and critic networks."""
        batch_size = states.size(0)
        
        # Update critic
        with torch.no_grad():
            # Get next actions from target actors
            next_actions = self.target_actor(next_states)
            # Calculate target Q-values
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + (self.gamma * target_q * (1 - dones))
        
        # Current Q-values
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        # Get current actions
        current_actions = self.actor(states)
        # Calculate actor loss (negative Q-value)
        actor_loss = -self.critic(states, current_actions).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
    
    def _soft_update(self, target_network: nn.Module, source_network: nn.Module) -> None:
        """Soft update target network."""
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)


class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient algorithm.
    
    Each agent has its own actor-critic networks and learns using
    centralized training with decentralized execution.
    """
    
    def __init__(
        self,
        env,
        n_agents: int,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        self.env = env
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.device = device
        
        # Create agents
        self.agents = {}
        for agent_id in range(n_agents):
            self.agents[agent_id] = MADDPGAgent(
                agent_id=agent_id,
                state_dim=state_dim,
                action_dim=action_dim,
                n_agents=n_agents,
                learning_rate=learning_rate,
                gamma=gamma,
                tau=tau,
                device=device
            )
        
        # Experience replay buffer
        self.buffer = deque(maxlen=buffer_size)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = []
        
    def store_experience(
        self,
        states: Dict[int, np.ndarray],
        actions: Dict[int, np.ndarray],
        rewards: Dict[int, float],
        next_states: Dict[int, np.ndarray],
        dones: Dict[int, bool]
    ) -> None:
        """Store experience in replay buffer."""
        self.buffer.append({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        })
    
    def train(self, n_episodes: int, max_steps_per_episode: int = 100) -> Dict[str, List[float]]:
        """Train the agents."""
        print(f"Training MADDPG for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            episode_reward = 0
            episode_length = 0
            
            # Reset environment
            observations, _ = self.env.reset()
            
            for step in range(max_steps_per_episode):
                # Select actions
                actions = {}
                for agent_id, obs in observations.items():
                    action = self.agents[agent_id].select_action(obs, training=True)
                    actions[agent_id] = action
                
                # Execute actions
                next_observations, rewards, terminated, truncated, _ = self.env.step(actions)
                
                # Store experience
                self.store_experience(observations, actions, rewards, next_observations, terminated)
                
                # Update agents if buffer is large enough
                if len(self.buffer) >= self.batch_size:
                    self._update_agents()
                
                # Update statistics
                episode_reward += sum(rewards.values())
                episode_length += 1
                
                # Check termination
                if all(terminated.values()) or all(truncated.values()):
                    break
                
                observations = next_observations
            
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
    
    def _update_agents(self) -> None:
        """Update all agents using experience replay."""
        # Sample batch
        batch = random.sample(self.buffer, self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([exp['states'][agent_id] for exp in batch for agent_id in range(self.n_agents)]).to(self.device)
        actions = torch.FloatTensor([exp['actions'][agent_id] for exp in batch for agent_id in range(self.n_agents)]).to(self.device)
        rewards = torch.FloatTensor([exp['rewards'][agent_id] for exp in batch for agent_id in range(self.n_agents)]).to(self.device)
        next_states = torch.FloatTensor([exp['next_states'][agent_id] for exp in batch for agent_id in range(self.n_agents)]).to(self.device)
        dones = torch.BoolTensor([exp['dones'][agent_id] for exp in batch for agent_id in range(self.n_agents)]).to(self.device)
        
        # Update each agent
        for agent_id in range(self.n_agents):
            agent_states = states[agent_id::self.n_agents]
            agent_actions = actions[agent_id::self.n_agents]
            agent_rewards = rewards[agent_id::self.n_agents]
            agent_next_states = next_states[agent_id::self.n_agents]
            agent_dones = dones[agent_id::self.n_agents]
            
            # Get all states and actions for centralized training
            all_states = states.view(self.batch_size, self.n_agents, -1)
            all_actions = actions.view(self.batch_size, self.n_agents, -1)
            all_next_states = next_states.view(self.batch_size, self.n_agents, -1)
            
            metrics = self.agents[agent_id].update(
                agent_states, agent_actions, agent_rewards, 
                agent_next_states, agent_dones,
                all_states, all_actions, all_next_states
            )
            
            if not self.training_metrics:
                self.training_metrics.append({})
            self.training_metrics[-1][f"agent_{agent_id}_actor_loss"] = metrics["actor_loss"]
            self.training_metrics[-1][f"agent_{agent_id}_critic_loss"] = metrics["critic_loss"]
    
    def evaluate(self, n_episodes: int = 10, max_steps_per_episode: int = 100) -> Dict[str, float]:
        """Evaluate the trained agents."""
        print(f"Evaluating MADDPG for {n_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            episode_reward = 0
            episode_length = 0
            
            # Reset environment
            observations, _ = self.env.reset()
            
            for step in range(max_steps_per_episode):
                # Select actions (no noise)
                actions = {}
                for agent_id, obs in observations.items():
                    action = self.agents[agent_id].select_action(obs, training=False)
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
                'target_actor_state_dict': agent.target_actor.state_dict(),
                'target_critic_state_dict': agent.target_critic.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict()
            }, f"{filepath}_agent_{agent_id}.pth")
    
    def load_models(self, filepath: str) -> None:
        """Load all agent models."""
        for agent_id, agent in self.agents.items():
            checkpoint = torch.load(f"{filepath}_agent_{agent_id}.pth", map_location=self.device)
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            agent.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
            agent.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])