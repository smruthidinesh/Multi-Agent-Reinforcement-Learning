"""
Cooperative Navigation Environment

Multiple agents must navigate to landmarks while avoiding collisions.
This is a classic multi-agent coordination problem.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CooperativeNavigation(gym.Env):
    """
    Cooperative Navigation Environment
    
    Multiple agents must navigate to landmarks while avoiding collisions.
    The environment is continuous and agents can move in any direction.
    """
    
    def __init__(
        self,
        n_agents: int = 3,
        n_landmarks: int = 3,
        world_size: float = 2.0,
        max_steps: int = 100,
        collision_distance: float = 0.1,
        landmark_distance: float = 0.1,
        collision_penalty: float = -1.0,
        landmark_reward: float = 10.0,
        step_penalty: float = -0.01,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.world_size = world_size
        self.max_steps = max_steps
        self.collision_distance = collision_distance
        self.landmark_distance = landmark_distance
        self.collision_penalty = collision_penalty
        self.landmark_reward = landmark_reward
        self.step_penalty = step_penalty
        self.render_mode = render_mode
        
        # Action space: [x_velocity, y_velocity] for each agent
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation space: agent position + velocity + landmark positions + other agent positions
        obs_dim = 2 + 2 + n_landmarks * 2 + (n_agents - 1) * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize agent positions randomly
        self.agent_positions = []
        self.agent_velocities = []
        for _ in range(self.n_agents):
            pos = self.np_random.uniform(-self.world_size/2, self.world_size/2, size=2)
            vel = np.zeros(2)
            self.agent_positions.append(pos)
            self.agent_velocities.append(vel)
        
        # Initialize landmark positions randomly
        self.landmark_positions = []
        for _ in range(self.n_landmarks):
            pos = self.np_random.uniform(-self.world_size/2, self.world_size/2, size=2)
            self.landmark_positions.append(pos)
        
        # Track visited landmarks
        self.visited_landmarks = set()
        self.step_count = 0
        
        # Get initial observations
        observations = self._get_observations()
        info = {"step_count": self.step_count}
        
        return observations, info
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one step in the environment."""
        rewards = {}
        terminated = {}
        truncated = {}
        
        # Update agent velocities and positions
        for agent_id, action in actions.items():
            # Update velocity (with some damping)
            self.agent_velocities[agent_id] = 0.5 * self.agent_velocities[agent_id] + 0.5 * action
            
            # Update position
            new_pos = self.agent_positions[agent_id] + self.agent_velocities[agent_id] * 0.1
            
            # Keep agents within world bounds
            new_pos = np.clip(new_pos, -self.world_size/2, self.world_size/2)
            self.agent_positions[agent_id] = new_pos
        
        # Calculate rewards
        for agent_id in range(self.n_agents):
            reward = self._calculate_reward(agent_id)
            rewards[agent_id] = reward
        
        # Check termination conditions
        for agent_id in range(self.n_agents):
            terminated[agent_id] = len(self.visited_landmarks) == self.n_landmarks
            truncated[agent_id] = self.step_count >= self.max_steps
        
        self.step_count += 1
        
        # Get observations
        observations = self._get_observations()
        info = {"step_count": self.step_count, "visited_landmarks": len(self.visited_landmarks)}
        
        return observations, rewards, terminated, truncated, info
    
    def _calculate_reward(self, agent_id: int) -> float:
        """Calculate reward for agent."""
        reward = self.step_penalty
        
        # Check for landmark visits
        agent_pos = self.agent_positions[agent_id]
        for i, landmark_pos in enumerate(self.landmark_positions):
            if i not in self.visited_landmarks:
                distance = np.linalg.norm(agent_pos - landmark_pos)
                if distance < self.landmark_distance:
                    self.visited_landmarks.add(i)
                    reward += self.landmark_reward
        
        # Check for collisions with other agents
        for other_id, other_pos in enumerate(self.agent_positions):
            if other_id != agent_id:
                distance = np.linalg.norm(agent_pos - other_pos)
                if distance < self.collision_distance:
                    reward += self.collision_penalty
        
        return reward
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all agents."""
        observations = {}
        
        for agent_id in range(self.n_agents):
            obs = []
            
            # Agent's own position and velocity
            obs.extend(self.agent_positions[agent_id])
            obs.extend(self.agent_velocities[agent_id])
            
            # Landmark positions
            for landmark_pos in self.landmark_positions:
                obs.extend(landmark_pos)
            
            # Other agents' positions
            for other_id, other_pos in enumerate(self.agent_positions):
                if other_id != agent_id:
                    obs.extend(other_pos)
            
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
        
        # Draw world boundary
        ax.add_patch(patches.Rectangle(
            (-self.world_size/2, -self.world_size/2), 
            self.world_size, self.world_size,
            linewidth=2, edgecolor='black', facecolor='none'
        ))
        
        # Draw landmarks
        for i, landmark_pos in enumerate(self.landmark_positions):
            if i not in self.visited_landmarks:
                circle = patches.Circle(
                    landmark_pos, 0.1, 
                    color='gold', alpha=0.8
                )
                ax.add_patch(circle)
                ax.text(landmark_pos[0], landmark_pos[1] + 0.15, f'L{i}', 
                       ha='center', va='center', fontweight='bold')
        
        # Draw agents
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for agent_id, agent_pos in enumerate(self.agent_positions):
            color = colors[agent_id % len(colors)]
            circle = patches.Circle(
                agent_pos, 0.08, 
                color=color, alpha=0.8
            )
            ax.add_patch(circle)
            ax.text(agent_pos[0], agent_pos[1] + 0.12, f'A{agent_id}', 
                   ha='center', va='center', fontweight='bold')
            
            # Draw velocity vector
            if np.linalg.norm(self.agent_velocities[agent_id]) > 0.01:
                vel = self.agent_velocities[agent_id] * 0.5
                ax.arrow(agent_pos[0], agent_pos[1], vel[0], vel[1], 
                        head_width=0.05, head_length=0.05, fc=color, ec=color)
        
        ax.set_xlim(-self.world_size/2 - 0.5, self.world_size/2 + 0.5)
        ax.set_ylim(-self.world_size/2 - 0.5, self.world_size/2 + 0.5)
        ax.set_aspect('equal')
        ax.set_title(f'Cooperative Navigation (Step {self.step_count})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render environment as RGB array."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw world boundary
        ax.add_patch(patches.Rectangle(
            (-self.world_size/2, -self.world_size/2), 
            self.world_size, self.world_size,
            linewidth=2, edgecolor='black', facecolor='none'
        ))
        
        # Draw landmarks
        for i, landmark_pos in enumerate(self.landmark_positions):
            if i not in self.visited_landmarks:
                circle = patches.Circle(
                    landmark_pos, 0.1, 
                    color='gold', alpha=0.8
                )
                ax.add_patch(circle)
                ax.text(landmark_pos[0], landmark_pos[1] + 0.15, f'L{i}', 
                       ha='center', va='center', fontweight='bold')
        
        # Draw agents
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for agent_id, agent_pos in enumerate(self.agent_positions):
            color = colors[agent_id % len(colors)]
            circle = patches.Circle(
                agent_pos, 0.08, 
                color=color, alpha=0.8
            )
            ax.add_patch(circle)
            ax.text(agent_pos[0], agent_pos[1] + 0.12, f'A{agent_id}', 
                   ha='center', va='center', fontweight='bold')
            
            # Draw velocity vector
            if np.linalg.norm(self.agent_velocities[agent_id]) > 0.01:
                vel = self.agent_velocities[agent_id] * 0.5
                ax.arrow(agent_pos[0], agent_pos[1], vel[0], vel[1], 
                        head_width=0.05, head_length=0.05, fc=color, ec=color)
        
        ax.set_xlim(-self.world_size/2 - 0.5, self.world_size/2 + 0.5)
        ax.set_ylim(-self.world_size/2 - 0.5, self.world_size/2 + 0.5)
        ax.set_aspect('equal')
        ax.set_title(f'Cooperative Navigation (Step {self.step_count})')
        ax.grid(True, alpha=0.3)
        
        # Convert to RGB array
        fig.canvas.draw()
        rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return rgb_array
    
    def close(self):
        """Close the environment."""
        pass