"""
Multi-Agent Grid World Environment

A customizable grid world environment where multiple agents can move around,
collect rewards, and interact with each other.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch


from marl.utils.communication import CommunicationChannel

class MultiAgentGridWorld(gym.Env):
    """
    Multi-agent grid world environment.
    
    Agents can move in 4 directions (up, down, left, right) and interact
    with the environment and each other.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        n_agents: int = 2,
        n_targets: int = 3,
        n_goals: int = 0,
        max_steps: int = 100,
        reward_scale: float = 1.0,
        collision_penalty: float = -0.1,
        target_reward: float = 10.0,
        goal_reward: float = 1.0,
        step_penalty: float = -0.01,
        render_mode: Optional[str] = None,
        message_dim: int = 0
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.n_goals = n_goals
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.collision_penalty = collision_penalty
        self.target_reward = target_reward
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.render_mode = render_mode
        self.message_dim = message_dim
        
        # Action space: 0=up, 1=down, 2=left, 3=right, 4=stay
        self.action_space = spaces.Discrete(5)
        
        # Observation space: agent position + target positions + other agent positions
        obs_dim = 2 + n_targets * 2 + (n_agents - 1) * 2 + (n_agents - 1) * message_dim
        self.observation_space = spaces.Box(
            low=0, high=max(grid_size), shape=(obs_dim,), dtype=np.float32
        )

        # Goal space
        if self.n_goals > 0:
            self.goal_space = spaces.Discrete(n_goals)
        else:
            self.goal_space = None
        
        # Communication channel
        if self.message_dim > 0:
            self.communication_channel = CommunicationChannel(n_agents, message_dim)
        else:
            self.communication_channel = None
        
        # Initialize state
        self.reset()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize agent positions randomly
        self.agent_positions = []
        for _ in range(self.n_agents):
            pos = self.np_random.integers(0, self.grid_size[0], size=2)
            self.agent_positions.append(pos)
        
        # Initialize target positions randomly
        self.target_positions = []
        for _ in range(self.n_targets):
            pos = self.np_random.integers(0, self.grid_size[0], size=2)
            self.target_positions.append(pos)

        # Initialize goal positions randomly
        if self.n_goals > 0:
            self.goal_positions = []
            for _ in range(self.n_goals):
                pos = self.np_random.integers(0, self.grid_size[0], size=2)
                self.goal_positions.append(pos)
        
        # Track collected targets
        self.collected_targets = set()
        self.step_count = 0

        # Reset communication channel
        if self.communication_channel:
            self.communication_channel.messages = torch.zeros((self.n_agents, self.message_dim))
        
        # Get initial observations
        observations = self._get_observations()
        info = {"step_count": self.step_count}
        
        return observations, info
    
    def step(self, actions: Dict[int, int], messages: Optional[Dict[int, torch.Tensor]] = None) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one step in the environment."""
        rewards = {}
        terminated = {}
        truncated = {}

        # Send messages
        if self.communication_channel and messages:
            for agent_id, message in messages.items():
                self.communication_channel.send_message(agent_id, message)
        
        # Move agents
        new_positions = []
        for agent_id, action in actions.items():
            new_pos = self._move_agent(agent_id, action)
            new_positions.append(new_pos)
        
        # Update agent positions
        self.agent_positions = new_positions
        
        # Calculate rewards
        for agent_id in range(self.n_agents):
            reward = self._calculate_reward(agent_id)
            rewards[agent_id] = reward
        
        # Check termination conditions
        for agent_id in range(self.n_agents):
            terminated[agent_id] = len(self.collected_targets) == self.n_targets
            truncated[agent_id] = self.step_count >= self.max_steps
        
        self.step_count += 1
        
        # Get observations
        observations = self._get_observations()
        info = {"step_count": self.step_count, "collected_targets": len(self.collected_targets)}
        
        return observations, rewards, terminated, truncated, info
    
    def _move_agent(self, agent_id: int, action: int) -> np.ndarray:
        """Move agent according to action."""
        current_pos = self.agent_positions[agent_id].copy()
        
        if action == 0:  # Up
            current_pos[0] = max(0, current_pos[0] - 1)
        elif action == 1:  # Down
            current_pos[0] = min(self.grid_size[0] - 1, current_pos[0] + 1)
        elif action == 2:  # Left
            current_pos[1] = max(0, current_pos[1] - 1)
        elif action == 3:  # Right
            current_pos[1] = min(self.grid_size[1] - 1, current_pos[1] + 1)
        # Action 4 is stay, no movement
        
        return current_pos
    
    def _calculate_reward(self, agent_id: int) -> float:
        """Calculate reward for agent."""
        reward = self.step_penalty
        
        # Check for target collection
        agent_pos = self.agent_positions[agent_id]
        for i, target_pos in enumerate(self.target_positions):
            if i not in self.collected_targets and np.array_equal(agent_pos, target_pos):
                self.collected_targets.add(i)
                reward += self.target_reward

        # Check for goal achievement
        if self.n_goals > 0:
            for goal_pos in self.goal_positions:
                if np.array_equal(agent_pos, goal_pos):
                    reward += self.goal_reward
        
        # Check for collisions with other agents
        for other_id, other_pos in enumerate(self.agent_positions):
            if other_id != agent_id and np.array_equal(agent_pos, other_pos):
                reward += self.collision_penalty
        
        return reward * self.reward_scale
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all agents."""
        observations = {}
        
        for agent_id in range(self.n_agents):
            obs = []
            
            # Agent's own position
            obs.extend(self.agent_positions[agent_id])
            
            # Target positions
            for target_pos in self.target_positions:
                obs.extend(target_pos)
            
            # Other agents' positions
            for other_id, other_pos in enumerate(self.agent_positions):
                if other_id != agent_id:
                    obs.extend(other_pos)

            # Messages from other agents
            if self.communication_channel:
                messages = self.communication_channel.get_messages(agent_id)
                obs.extend(messages.flatten().tolist())
            
            observations[agent_id] = np.array(obs, dtype=np.float32)
        
        return observations
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render environment for human viewing."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw grid
        for i in range(self.grid_size[0] + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
        for j in range(self.grid_size[1] + 1):
            ax.axvline(j - 0.5, color='black', linewidth=0.5)
        
        # Draw targets
        for i, target_pos in enumerate(self.target_positions):
            if i not in self.collected_targets:
                circle = patches.Circle(
                    (target_pos[1], target_pos[0]), 0.3, 
                    color='gold', alpha=0.8
                )
                ax.add_patch(circle)
        
        # Draw agents
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for agent_id, agent_pos in enumerate(self.agent_positions):
            color = colors[agent_id % len(colors)]
            circle = patches.Circle(
                (agent_pos[1], agent_pos[0]), 0.4, 
                color=color, alpha=0.8
            )
            ax.add_patch(circle)
            ax.text(agent_pos[1], agent_pos[0], str(agent_id), 
                   ha='center', va='center', fontweight='bold')
        
        ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
        ax.set_ylim(-0.5, self.grid_size[0] - 0.5)
        ax.set_aspect('equal')
        ax.set_title(f'Multi-Agent Grid World (Step {self.step_count})')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render environment as RGB array."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw grid
        for i in range(self.grid_size[0] + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
        for j in range(self.grid_size[1] + 1):
            ax.axvline(j - 0.5, color='black', linewidth=0.5)
        
        # Draw targets
        for i, target_pos in enumerate(self.target_positions):
            if i not in self.collected_targets:
                circle = patches.Circle(
                    (target_pos[1], target_pos[0]), 0.3, 
                    color='gold', alpha=0.8
                )
                ax.add_patch(circle)
        
        # Draw agents
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for agent_id, agent_pos in enumerate(self.agent_positions):
            color = colors[agent_id % len(colors)]
            circle = patches.Circle(
                (agent_pos[1], agent_pos[0]), 0.4, 
                color=color, alpha=0.8
            )
            ax.add_patch(circle)
            ax.text(agent_pos[1], agent_pos[0], str(agent_id), 
                   ha='center', va='center', fontweight='bold')
        
        ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
        ax.set_ylim(-0.5, self.grid_size[0] - 0.5)
        ax.set_aspect('equal')
        ax.set_title(f'Multi-Agent Grid World (Step {self.step_count})')
        ax.invert_yaxis()
        
        # Convert to RGB array
        fig.canvas.draw()
        rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return rgb_array
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the environment for visualization."""
        return {
            "grid_size": self.grid_size,
            "agents": [{"id": i, "pos": pos.tolist()} for i, pos in enumerate(self.agent_positions)],
            "targets": [{"id": i, "pos": pos.tolist()} for i, pos in enumerate(self.target_positions) if i not in self.collected_targets],
        }
    
    def close(self):
        """Close the environment."""
        pass