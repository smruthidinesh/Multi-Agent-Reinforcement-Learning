"""
Independent Q-Learning Algorithm

Each agent learns independently using DQN without considering other agents.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional
from ..agents.dqn_agent import DQNAgent
from ..environments.grid_world import MultiAgentGridWorld


class IndependentQLearning:
    """
    Independent Q-Learning algorithm.
    
    Each agent learns independently using DQN without considering other agents.
    This is a baseline approach for multi-agent RL.
    """
    
    def __init__(
        self,
        env: MultiAgentGridWorld,
        n_agents: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        device: str = "cpu",
        message_dim: int = 0
    ):
        self.env = env
        self.n_agents = n_agents
        self.device = device
        self.message_dim = message_dim
        
        # Create agents
        self.agents = {}
        for agent_id in range(n_agents):
            self.agents[agent_id] = DQNAgent(
                agent_id=agent_id,
                observation_space=env.observation_space,
                action_space=env.action_space,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon=epsilon,
                epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
                buffer_size=buffer_size,
                batch_size=batch_size,
                target_update_freq=target_update_freq,
                device=device,
                communication_channel=env.communication_channel if hasattr(env, 'communication_channel') else None,
                message_dim=message_dim
            )
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = []
        
    def train(self, n_episodes: int, max_steps_per_episode: int = 100) -> Dict[str, List[float]]:
        """Train the agents."""
        print(f"Training Independent Q-Learning for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            episode_reward = 0
            episode_length = 0
            
            # Reset environment
            observations, _ = self.env.reset()
            
            for step in range(max_steps_per_episode):
                # Get messages from other agents
                messages = {}
                if self.message_dim > 0:
                    for agent_id in range(self.n_agents):
                        messages[agent_id] = self.agents[agent_id].get_messages()

                # Select actions
                actions = {}
                for agent_id, obs in observations.items():
                    action = self.agents[agent_id].get_action(obs, messages.get(agent_id), training=True)
                    actions[agent_id] = action

                # Generate messages to send
                new_messages = {}
                if self.message_dim > 0:
                    for agent_id, obs in observations.items():
                        # For simplicity, agents send their observation as a message
                        # A more sophisticated approach would be to learn what to send
                        message = torch.FloatTensor(obs[:self.message_dim])
                        new_messages[agent_id] = message
                
                # Execute actions
                next_observations, rewards, terminated, truncated, _ = self.env.step(actions, new_messages)
                
                # Store experiences
                for agent_id in range(self.n_agents):
                    self.agents[agent_id].store_experience(
                        observations[agent_id],
                        actions[agent_id],
                        rewards[agent_id],
                        next_observations[agent_id],
                        terminated[agent_id] or truncated[agent_id]
                    )
                
                # Update agents
                for agent_id in range(self.n_agents):
                    metrics = self.agents[agent_id].update()
                    if episode == 0:
                        self.training_metrics.append({})
                    self.training_metrics[-1][f"agent_{agent_id}_loss"] = metrics.get("loss", 0.0)
                
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
    
    def evaluate(self, n_episodes: int = 10, max_steps_per_episode: int = 100) -> Dict[str, float]:
        """Evaluate the trained agents."""
        print(f"Evaluating Independent Q-Learning for {n_episodes} episodes...")
        
        # Set agents to evaluation mode
        for agent in self.agents.values():
            agent.set_training_mode(False)
        
        episode_rewards = []
        episode_lengths = []
        agent_rewards = {agent_id: [] for agent_id in range(self.n_agents)}
        
        successes = []
        for episode in range(n_episodes):
            episode_reward = 0
            episode_length = 0
            agent_episode_rewards = {agent_id: 0 for agent_id in range(self.n_agents)}
            
            # Reset environment
            observations, _ = self.env.reset()
            
            for step in range(max_steps_per_episode):
                # Get messages from other agents
                messages = {}
                if self.message_dim > 0:
                    for agent_id in range(self.n_agents):
                        messages[agent_id] = self.agents[agent_id].get_messages()

                # Select actions (greedy)
                actions = {}
                for agent_id, obs in observations.items():
                    action = self.agents[agent_id].get_action(obs, messages.get(agent_id), training=False)
                    actions[agent_id] = action

                # Generate messages to send
                new_messages = {}
                if self.message_dim > 0:
                    for agent_id, obs in observations.items():
                        message = torch.FloatTensor(obs[:self.message_dim])
                        new_messages[agent_id] = message
                
                # Execute actions
                next_observations, rewards, terminated, truncated, info = self.env.step(actions, new_messages)
                
                # Update statistics
                episode_reward += sum(rewards.values())
                episode_length += 1
                
                # Track individual agent rewards
                for agent_id, reward in rewards.items():
                    agent_episode_rewards[agent_id] += reward
                
                # Check termination
                if all(terminated.values()) or all(truncated.values()):
                    break
                
                observations = next_observations
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            successes.append(len(self.env.collected_targets) == self.env.n_targets)
            
            # Store individual agent rewards
            for agent_id, reward in agent_episode_rewards.items():
                agent_rewards[agent_id].append(reward)
        
        # Set agents back to training mode
        for agent in self.agents.values():
            agent.set_training_mode(True)
        
        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "agent_rewards": agent_rewards,
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "avg_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "success_rate": np.mean(successes)
        }
    
    def save_models(self, filepath: str) -> None:
        """Save all agent models."""
        import os
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        for agent_id, agent in self.agents.items():
            # Use the directory path + filename
            if dir_path:
                base_name = os.path.basename(filepath) or os.path.basename(dir_path)
                full_path = os.path.join(dir_path, f"{base_name}_agent_{agent_id}.pth")
            else:
                full_path = f"{filepath}_agent_{agent_id}.pth"
            agent.save(full_path)
    
    def load_models(self, filepath: str) -> None:
        """Load all agent models."""
        import os
        dir_path = os.path.dirname(filepath)
        for agent_id, agent in self.agents.items():
            # Use the same path logic as save_models
            if dir_path:
                base_name = os.path.basename(filepath) or os.path.basename(dir_path)
                full_path = os.path.join(dir_path, f"{base_name}_agent_{agent_id}.pth")
            else:
                full_path = f"{filepath}_agent_{agent_id}.pth"
            agent.load(full_path)
    
    def get_agent_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics for all agents."""
        return {agent_id: agent.get_stats() for agent_id, agent in self.agents.items()}