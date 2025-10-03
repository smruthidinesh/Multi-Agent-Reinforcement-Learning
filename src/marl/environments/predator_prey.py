"""
Predator-Prey Environment

A classic multi-agent environment where predators must catch prey
while prey try to avoid being caught.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class PredatorPrey(gym.Env):
    """
    Predator-Prey Environment
    
    Multiple predators try to catch prey while prey try to avoid being caught.
    This creates a competitive multi-agent scenario.
    """
    
    def __init__(
        self,
        n_predators: int = 2,
        n_prey: int = 1,
        world_size: float = 2.0,
        max_steps: int = 100,
        capture_distance: float = 0.1,
        predator_speed: float = 0.1,
        prey_speed: float = 0.08,
        capture_reward: float = 10.0,
        escape_reward: float = 1.0,
        step_penalty: float = -0.01,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.n_predators = n_predators
        self.n_prey = n_prey
        self.world_size = world_size
        self.max_steps = max_steps
        self.capture_distance = capture_distance
        self.predator_speed = predator_speed
        self.prey_speed = prey_speed
        self.capture_reward = capture_reward
        self.escape_reward = escape_reward
        self.step_penalty = step_penalty
        self.render_mode = render_mode
        
        # Action space: [x_velocity, y_velocity] for each agent
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation space: agent position + velocity + other agent positions
        obs_dim = 2 + 2 + (n_predators + n_prey - 1) * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize predator positions randomly
        self.predator_positions = []
        self.predator_velocities = []
        for _ in range(self.n_predators):
            pos = self.np_random.uniform(-self.world_size/2, self.world_size/2, size=2)
            vel = np.zeros(2)
            self.predator_positions.append(pos)
            self.predator_velocities.append(vel)
        
        # Initialize prey positions randomly
        self.prey_positions = []
        self.prey_velocities = []
        for _ in range(self.n_prey):
            pos = self.np_random.uniform(-self.world_size/2, self.world_size/2, size=2)
            vel = np.zeros(2)
            self.prey_positions.append(pos)
            self.prey_velocities.append(vel)
        
        # Track captured prey
        self.captured_prey = set()
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
        
        # Update predator velocities and positions
        for predator_id, action in actions.items():
            if predator_id < self.n_predators:  # Predator action
                # Update velocity (with some damping)
                self.predator_velocities[predator_id] = 0.5 * self.predator_velocities[predator_id] + 0.5 * action
                
                # Update position
                new_pos = self.predator_positions[predator_id] + self.predator_velocities[predator_id] * self.predator_speed
                
                # Keep predators within world bounds
                new_pos = np.clip(new_pos, -self.world_size/2, self.world_size/2)
                self.predator_positions[predator_id] = new_pos
        
        # Update prey velocities and positions (simple escape behavior)
        for prey_id in range(self.n_prey):
            if prey_id not in self.captured_prey:
                # Simple escape behavior: move away from nearest predator
                nearest_predator_pos = self._find_nearest_predator(prey_id)
                escape_direction = self.prey_positions[prey_id] - nearest_predator_pos
                escape_direction = escape_direction / (np.linalg.norm(escape_direction) + 1e-8)
                
                # Update velocity
                self.prey_velocities[prey_id] = escape_direction * self.prey_speed
                
                # Update position
                new_pos = self.prey_positions[prey_id] + self.prey_velocities[prey_id]
                
                # Keep prey within world bounds
                new_pos = np.clip(new_pos, -self.world_size/2, self.world_size/2)
                self.prey_positions[prey_id] = new_pos
        
        # Calculate rewards
        for agent_id in range(self.n_predators + self.n_prey):
            reward = self._calculate_reward(agent_id)
            rewards[agent_id] = reward
        
        # Check termination conditions
        for agent_id in range(self.n_predators + self.n_prey):
            terminated[agent_id] = len(self.captured_prey) == self.n_prey
            truncated[agent_id] = self.step_count >= self.max_steps
        
        self.step_count += 1
        
        # Get observations
        observations = self._get_observations()
        info = {"step_count": self.step_count, "captured_prey": len(self.captured_prey)}
        
        return observations, rewards, terminated, truncated, info
    
    def _find_nearest_predator(self, prey_id: int) -> np.ndarray:
        """Find the nearest predator to a prey."""
        prey_pos = self.prey_positions[prey_id]
        min_distance = float('inf')
        nearest_predator_pos = None
        
        for predator_pos in self.predator_positions:
            distance = np.linalg.norm(prey_pos - predator_pos)
            if distance < min_distance:
                min_distance = distance
                nearest_predator_pos = predator_pos
        
        return nearest_predator_pos
    
    def _calculate_reward(self, agent_id: int) -> float:
        """Calculate reward for agent."""
        reward = self.step_penalty
        
        if agent_id < self.n_predators:  # Predator
            # Check for prey captures
            predator_pos = self.predator_positions[agent_id]
            for prey_id, prey_pos in enumerate(self.prey_positions):
                if prey_id not in self.captured_prey:
                    distance = np.linalg.norm(predator_pos - prey_pos)
                    if distance < self.capture_distance:
                        self.captured_prey.add(prey_id)
                        reward += self.capture_reward
        else:  # Prey
            prey_id = agent_id - self.n_predators
            if prey_id not in self.captured_prey:
                # Reward for surviving
                reward += self.escape_reward
        
        return reward
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all agents."""
        observations = {}
        
        # Predator observations
        for predator_id in range(self.n_predators):
            obs = []
            
            # Predator's own position and velocity
            obs.extend(self.predator_positions[predator_id])
            obs.extend(self.predator_velocities[predator_id])
            
            # Other predators' positions
            for other_id, other_pos in enumerate(self.predator_positions):
                if other_id != predator_id:
                    obs.extend(other_pos)
            
            # Prey positions
            for prey_pos in self.prey_positions:
                obs.extend(prey_pos)
            
            observations[predator_id] = np.array(obs, dtype=np.float32)
        
        # Prey observations
        for prey_id in range(self.n_prey):
            agent_id = self.n_predators + prey_id
            obs = []
            
            # Prey's own position and velocity
            obs.extend(self.prey_positions[prey_id])
            obs.extend(self.prey_velocities[prey_id])
            
            # Predator positions
            for predator_pos in self.predator_positions:
                obs.extend(predator_pos)
            
            # Other prey positions
            for other_id, other_pos in enumerate(self.prey_positions):
                if other_id != prey_id:
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
        
        # Draw predators
        for predator_id, predator_pos in enumerate(self.predator_positions):
            circle = patches.Circle(
                predator_pos, 0.08, 
                color='red', alpha=0.8
            )
            ax.add_patch(circle)
            ax.text(predator_pos[0], predator_pos[1] + 0.12, f'P{predator_id}', 
                   ha='center', va='center', fontweight='bold')
            
            # Draw velocity vector
            if np.linalg.norm(self.predator_velocities[predator_id]) > 0.01:
                vel = self.predator_velocities[predator_id] * 0.5
                ax.arrow(predator_pos[0], predator_pos[1], vel[0], vel[1], 
                        head_width=0.05, head_length=0.05, fc='red', ec='red')
        
        # Draw prey
        for prey_id, prey_pos in enumerate(self.prey_positions):
            if prey_id not in self.captured_prey:
                circle = patches.Circle(
                    prey_pos, 0.06, 
                    color='blue', alpha=0.8
                )
                ax.add_patch(circle)
                ax.text(prey_pos[0], prey_pos[1] + 0.12, f'Prey{prey_id}', 
                       ha='center', va='center', fontweight='bold')
                
                # Draw velocity vector
                if np.linalg.norm(self.prey_velocities[prey_id]) > 0.01:
                    vel = self.prey_velocities[prey_id] * 0.5
                    ax.arrow(prey_pos[0], prey_pos[1], vel[0], vel[1], 
                            head_width=0.05, head_length=0.05, fc='blue', ec='blue')
        
        ax.set_xlim(-self.world_size/2 - 0.5, self.world_size/2 + 0.5)
        ax.set_ylim(-self.world_size/2 - 0.5, self.world_size/2 + 0.5)
        ax.set_aspect('equal')
        ax.set_title(f'Predator-Prey (Step {self.step_count})')
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
        
        # Draw predators
        for predator_id, predator_pos in enumerate(self.predator_positions):
            circle = patches.Circle(
                predator_pos, 0.08, 
                color='red', alpha=0.8
            )
            ax.add_patch(circle)
            ax.text(predator_pos[0], predator_pos[1] + 0.12, f'P{predator_id}', 
                   ha='center', va='center', fontweight='bold')
            
            # Draw velocity vector
            if np.linalg.norm(self.predator_velocities[predator_id]) > 0.01:
                vel = self.predator_velocities[predator_id] * 0.5
                ax.arrow(predator_pos[0], predator_pos[1], vel[0], vel[1], 
                        head_width=0.05, head_length=0.05, fc='red', ec='red')
        
        # Draw prey
        for prey_id, prey_pos in enumerate(self.prey_positions):
            if prey_id not in self.captured_prey:
                circle = patches.Circle(
                    prey_pos, 0.06, 
                    color='blue', alpha=0.8
                )
                ax.add_patch(circle)
                ax.text(prey_pos[0], prey_pos[1] + 0.12, f'Prey{prey_id}', 
                       ha='center', va='center', fontweight='bold')
                
                # Draw velocity vector
                if np.linalg.norm(self.prey_velocities[prey_id]) > 0.01:
                    vel = self.prey_velocities[prey_id] * 0.5
                    ax.arrow(prey_pos[0], prey_pos[1], vel[0], vel[1], 
                            head_width=0.05, head_length=0.05, fc='blue', ec='blue')
        
        ax.set_xlim(-self.world_size/2 - 0.5, self.world_size/2 + 0.5)
        ax.set_ylim(-self.world_size/2 - 0.5, self.world_size/2 + 0.5)
        ax.set_aspect('equal')
        ax.set_title(f'Predator-Prey (Step {self.step_count})')
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