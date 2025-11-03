
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TrafficEnv(gym.Env):
    """
    A simple multi-agent traffic environment.
    """
    def __init__(self, n_agents=2, grid_size=10):
        super(TrafficEnv, self).__init__()
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.agent_positions = [self._get_random_position() for _ in range(n_agents)]
        self.goal_positions = [self._get_random_position() for _ in range(n_agents)]

        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(self.n_agents, 2), dtype=np.int32)
        self.action_space = spaces.Discrete(5) # 0: up, 1: down, 2: left, 3: right, 4: stay

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_positions = [self._get_random_position() for _ in range(self.n_agents)]
        self.goal_positions = [self._get_random_position() for _ in range(self.n_agents)]
        return self._get_observation(), self._get_info()

    def step(self, actions):
        rewards = []
        for i in range(self.n_agents):
            self._move_agent(i, actions[i])
            reward = self._get_reward(i)
            rewards.append(reward)

        terminated = all(np.array_equal(self.agent_positions[i], self.goal_positions[i]) for i in range(self.n_agents))
        truncated = False 
        
        return self._get_observation(), rewards, terminated, truncated, self._get_info()

    def render(self, mode='human'):
        grid = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.n_agents):
            grid[self.agent_positions[i][0], self.agent_positions[i][1]] = 1
            grid[self.goal_positions[i][0], self.goal_positions[i][1]] = 2
        print(grid)

    def _get_random_position(self):
        return [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]

    def _get_observation(self):
        return np.array(self.agent_positions)

    def _get_info(self):
        return {}

    def _move_agent(self, agent_index, action):
        if action == 0:  # Up
            self.agent_positions[agent_index][0] = max(0, self.agent_positions[agent_index][0] - 1)
        elif action == 1:  # Down
            self.agent_positions[agent_index][0] = min(self.grid_size - 1, self.agent_positions[agent_index][0] + 1)
        elif action == 2:  # Left
            self.agent_positions[agent_index][1] = max(0, self.agent_positions[agent_index][1] - 1)
        elif action == 3:  # Right
            self.agent_positions[agent_index][1] = min(self.grid_size - 1, self.agent_positions[agent_index][1] + 1)

    def _get_reward(self, agent_index):
        if np.array_equal(self.agent_positions[agent_index], self.goal_positions[agent_index]):
            return 1.0
        return -0.1
