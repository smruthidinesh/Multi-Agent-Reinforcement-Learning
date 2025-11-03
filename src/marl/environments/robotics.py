import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional

class MultiAgentRobotNavigation(gym.Env):
    """
    A 2D environment where multiple robotic agents must navigate to target locations
    while avoiding obstacles and collisions with other robots.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 n_agents: int = 2,
                 grid_size: int = 10,
                 n_targets: int = 2,
                 n_obstacles: int = 3,
                 max_steps: int = 100,
                 target_reward: float = 10.0,
                 collision_penalty: float = -5.0,
                 step_penalty: float = -0.1,
                 render_mode: Optional[str] = None):
        super().__init__()

        self.n_agents = n_agents
        self.grid_size = grid_size
        self.n_targets = n_targets
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps
        self.current_step = 0

        self.target_reward = target_reward
        self.collision_penalty = collision_penalty
        self.step_penalty = step_penalty

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Define observation space for each agent
        # Each agent observes:
        #   - its own (x, y) position
        #   - its own (vx, vy) velocity
        #   - (x, y) positions of all other agents
        #   - (x, y) positions of all targets
        #   - (x, y) positions of all obstacles
        single_observation_space = spaces.Box(
            low=0, high=grid_size, shape=(2 + 2 + (n_agents - 1) * 2 + n_targets * 2 + n_obstacles * 2,), dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            f"agent_{i}": single_observation_space for i in range(self.n_agents)
        })

        # Define action space for each agent (linear_velocity, angular_velocity)
        # For simplicity, let's use discrete actions for now:
        # 0: move_forward, 1: turn_left, 2: turn_right, 3: stay
        single_action_space = spaces.Discrete(4)
        self.action_space = spaces.Dict({
            f"agent_{i}": single_action_space for i in range(self.n_agents)
        })

        self.agent_positions = None
        self.agent_velocities = None
        self.target_positions = None
        self.obstacle_positions = None
        self.collected_targets = None

    def _get_obs(self) -> Dict[str, np.ndarray]:
        observations = {}
        for i in range(self.n_agents):
            obs = []
            # Own position and velocity
            obs.extend(self.agent_positions[i])
            obs.extend(self.agent_velocities[i])

            # Other agents' positions
            for j in range(self.n_agents):
                if i != j:
                    obs.extend(self.agent_positions[j])

            # Target positions
            for target_pos in self.target_positions:
                obs.extend(target_pos)

            # Obstacle positions
            for obstacle_pos in self.obstacle_positions:
                obs.extend(obstacle_pos)

            observations[f"agent_{i}"] = np.array(obs, dtype=np.float32)
        return observations

    def _get_info(self) -> Dict[str, Dict]:
        return {f"agent_{i}": {} for i in range(self.n_agents)}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        super().reset(seed=seed)
        self.current_step = 0

        # Initialize agent positions randomly
        self.agent_positions = self.np_random.uniform(0, self.grid_size, size=(self.n_agents, 2))
        self.agent_velocities = np.zeros((self.n_agents, 2))

        # Initialize target positions randomly
        self.target_positions = self.np_random.uniform(0, self.grid_size, size=(self.n_targets, 2))
        self.collected_targets = [False] * self.n_targets

        # Initialize obstacle positions randomly
        self.obstacle_positions = self.np_random.uniform(0, self.grid_size, size=(self.n_obstacles, 2))

        observations = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observations, info

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict]]:
        self.current_step += 1
        rewards = {f"agent_{i}": self.step_penalty for i in range(self.n_agents)}
        terminated = {f"agent_{i}": False for i in range(self.n_agents)}
        truncated = {f"agent_{i}": False for i in range(self.n_agents)}

        new_agent_positions = np.copy(self.agent_positions)

        for i, agent_id in enumerate(actions):
            action = actions[agent_id]
            current_pos = self.agent_positions[i]
            current_vel = self.agent_velocities[i]

            # Simple movement model
            if action == 0: # Move forward
                new_agent_positions[i] += current_vel + np.array([0.5, 0]) # Assuming initial velocity is 0, move right
            elif action == 1: # Turn left (rotate velocity vector)
                # For simplicity, just change direction
                new_agent_positions[i] += np.array([-0.5, 0])
            elif action == 2: # Turn right
                new_agent_positions[i] += np.array([0.5, 0])
            # action == 3: Stay (no change to position)

            # Boundary conditions
            new_agent_positions[i] = np.clip(new_agent_positions[i], 0, self.grid_size - 1)

        self.agent_positions = new_agent_positions

        # Check for collisions and target collection
        for i in range(self.n_agents):
            agent_pos = self.agent_positions[i]

            # Agent-obstacle collision
            for obstacle_pos in self.obstacle_positions:
                if np.linalg.norm(agent_pos - obstacle_pos) < 0.5: # Simple collision detection
                    rewards[f"agent_{i}"] += self.collision_penalty
                    terminated[f"agent_{i}"] = True # Agent terminates on collision

            # Agent-agent collision
            for j in range(self.n_agents):
                if i != j and np.linalg.norm(agent_pos - self.agent_positions[j]) < 0.5:
                    rewards[f"agent_{i}"] += self.collision_penalty
                    terminated[f"agent_{i}"] = True

            # Target collection
            for t_idx in range(self.n_targets):
                if not self.collected_targets[t_idx] and np.linalg.norm(agent_pos - self.target_positions[t_idx]) < 0.5:
                    rewards[f"agent_{i}"] += self.target_reward
                    self.collected_targets[t_idx] = True

        # Check if all targets collected
        if all(self.collected_targets):
            for i in range(self.n_agents):
                terminated[f"agent_{i}"] = True

        # Check for max steps
        if self.current_step >= self.max_steps:
            for i in range(self.n_agents):
                truncated[f"agent_{i}"] = True

        observations = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        import pygame # Moved import here
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window_size = 500
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255)) # White background

        # Scale factor
        pix_square_size = self.window_size / self.grid_size

        # Draw targets
        for t_idx, target_pos in enumerate(self.target_positions):
            if not self.collected_targets[t_idx]:
                pygame.draw.circle(
                    canvas,
                    (0, 255, 0), # Green
                    (target_pos + 0.5) * pix_square_size,
                    pix_square_size / 3,
                )

        # Draw obstacles
        for obstacle_pos in self.obstacle_positions:
            pygame.draw.rect(
                canvas,
                (100, 100, 100), # Gray
                pygame.Rect(
                    obstacle_pos[0] * pix_square_size,
                    obstacle_pos[1] * pix_square_size,
                    pix_square_size,
                    pix_square_size,
                ),
            )

        # Draw agents
        for i, agent_pos in enumerate(self.agent_positions):
            pygame.draw.circle(
                canvas,
                (0, 0, 255) if i == 0 else (255, 0, 0), # Blue for agent 0, Red for others
                (agent_pos + 0.5) * pix_square_size,
                pix_square_size / 4,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else: # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
