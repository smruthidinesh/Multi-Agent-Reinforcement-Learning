# Multi-Agent RL Tutorial

This tutorial will guide you through building and training multi-agent reinforcement learning systems using our framework.

## Table of Contents

1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
3. [Basic Multi-Agent RL](#basic-multi-agent-rl)
4. [Advanced Techniques](#advanced-techniques)
5. [Custom Environments](#custom-environments)
6. [Custom Agents](#custom-agents)
7. [Performance Optimization](#performance-optimization)
8. [Best Practices](#best-practices)

## Introduction

Multi-agent reinforcement learning (MARL) involves multiple agents learning to interact in a shared environment. Unlike single-agent RL, MARL introduces additional challenges:

- **Non-stationarity**: The environment changes as other agents learn
- **Coordination**: Agents must learn to cooperate or compete effectively
- **Scalability**: Learning becomes more difficult with more agents
- **Communication**: Agents may need to share information

## Environment Setup

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multi-agent-rl.git
cd multi-agent-rl

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import marl; print('Installation successful!')"
```

### Basic Imports

```python
import numpy as np
import torch
import matplotlib.pyplot as plt

from marl.environments import MultiAgentGridWorld
from marl.algorithms import IndependentQLearning
from marl.utils import TrainingUtils, EvaluationUtils
from marl.visualization import TrainingPlots
```

## Basic Multi-Agent RL

### Step 1: Create an Environment

Let's start with a simple grid world environment:

```python
# Create environment
env = MultiAgentGridWorld(
    grid_size=(8, 8),      # 8x8 grid
    n_agents=2,           # 2 agents
    n_targets=2,         # 2 targets to collect
    max_steps=50,         # Maximum 50 steps per episode
    target_reward=10.0,   # Reward for collecting target
    collision_penalty=-0.1,  # Penalty for collisions
    step_penalty=-0.01   # Small penalty per step
)

print(f"Environment created: {env}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
```

### Step 2: Create an Algorithm

We'll use Independent Q-Learning as our first algorithm:

```python
# Create algorithm
algorithm = IndependentQLearning(
    env=env,
    n_agents=2,
    learning_rate=0.001,   # Learning rate
    gamma=0.99,           # Discount factor
    epsilon=1.0,          # Initial exploration rate
    epsilon_min=0.01,     # Minimum exploration rate
    epsilon_decay=0.995,  # Exploration decay rate
    buffer_size=5000,     # Experience replay buffer size
    batch_size=32,        # Training batch size
    target_update_freq=50, # Target network update frequency
    device="cpu"          # Use CPU for computation
)
```

### Step 3: Train the Agents

```python
# Train for 500 episodes
print("Starting training...")
results = algorithm.train(n_episodes=500, max_steps_per_episode=50)

print(f"Training completed!")
print(f"Final average reward: {np.mean(results['episode_rewards'][-100:]):.2f}")
```

### Step 4: Visualize Training Progress

```python
# Plot training progress
TrainingPlots.plot_learning_curves(
    results['episode_rewards'],
    results['episode_lengths'],
    results['training_metrics']
)
```

### Step 5: Evaluate the Trained Agents

```python
# Evaluate the trained agents
print("Evaluating trained agents...")
eval_results = algorithm.evaluate(n_episodes=20, max_steps_per_episode=50)

print("Evaluation Results:")
print(f"Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
print(f"Average Length: {eval_results['avg_length']:.2f} ± {eval_results['std_length']:.2f}")
print(f"Success Rate: {eval_results['success_rate']:.2%}")
```

### Step 6: Visualize Evaluation Results

```python
# Plot evaluation results
EvaluationUtils.plot_evaluation_results(eval_results)
```

## Advanced Techniques

### Using Different Algorithms

#### MAPPO (Multi-Agent Proximal Policy Optimization)

```python
from marl.algorithms import MAPPO

# Create MAPPO algorithm
algorithm = MAPPO(
    env=env,
    n_agents=2,
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    learning_rate=0.001,
    gamma=0.99,
    lambda_gae=0.95,      # GAE parameter
    clip_ratio=0.2,      # PPO clip ratio
    value_coef=0.5,      # Value function coefficient
    entropy_coef=0.01,   # Entropy coefficient
    device="cpu"
)

# Train with MAPPO
results = algorithm.train(n_episodes=500, max_steps_per_episode=50)
```

#### MADDPG (Multi-Agent Deep Deterministic Policy Gradient)

```python
from marl.algorithms import MADDPG

# Note: MADDPG requires continuous action space
# For this example, we'll use a modified environment

# Create MADDPG algorithm
algorithm = MADDPG(
    env=env,
    n_agents=2,
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    learning_rate=0.001,
    gamma=0.99,
    tau=0.005,           # Soft update parameter
    buffer_size=100000,  # Larger buffer for MADDPG
    batch_size=64,       # Larger batch size
    device="cpu"
)

# Train with MADDPG
results = algorithm.train(n_episodes=500, max_steps_per_episode=50)
```

### Hyperparameter Tuning

```python
# Define hyperparameter search space
learning_rates = [0.0001, 0.001, 0.01]
gammas = [0.9, 0.95, 0.99]
epsilon_decays = [0.99, 0.995, 0.999]

best_reward = -float('inf')
best_params = None

for lr in learning_rates:
    for gamma in gammas:
        for eps_decay in epsilon_decays:
            print(f"Testing lr={lr}, gamma={gamma}, eps_decay={eps_decay}")
            
            # Create algorithm with current hyperparameters
            algorithm = IndependentQLearning(
                env=env,
                n_agents=2,
                learning_rate=lr,
                gamma=gamma,
                epsilon_decay=eps_decay,
                device="cpu"
            )
            
            # Train for a short time
            results = algorithm.train(n_episodes=100, max_steps_per_episode=50)
            
            # Evaluate
            eval_results = algorithm.evaluate(n_episodes=10, max_steps_per_episode=50)
            
            # Check if this is the best so far
            if eval_results['avg_reward'] > best_reward:
                best_reward = eval_results['avg_reward']
                best_params = {'lr': lr, 'gamma': gamma, 'eps_decay': eps_decay}
            
            print(f"Average reward: {eval_results['avg_reward']:.2f}")

print(f"Best parameters: {best_params}")
print(f"Best reward: {best_reward:.2f}")
```

### Algorithm Comparison

```python
# Compare different algorithms
algorithms = {
    'Independent Q-Learning': IndependentQLearning(env=env, n_agents=2, device="cpu"),
    'MAPPO': MAPPO(env=env, n_agents=2, state_dim=env.observation_space.shape[0], 
                   action_dim=env.action_space.n, device="cpu")
}

results_comparison = {}

for name, algorithm in algorithms.items():
    print(f"Training {name}...")
    results = algorithm.train(n_episodes=300, max_steps_per_episode=50)
    results_comparison[name] = results

# Plot comparison
TrainingPlots.plot_algorithm_comparison(results_comparison, 'episode_rewards')
```

## Custom Environments

### Creating a Custom Environment

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomMultiAgentEnv(gym.Env):
    def __init__(self, n_agents=2, world_size=10):
        super().__init__()
        
        self.n_agents = n_agents
        self.world_size = world_size
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.observation_space = spaces.Box(
            low=0, high=world_size, shape=(n_agents * 2,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize agent positions randomly
        self.agent_positions = []
        for _ in range(self.n_agents):
            pos = self.np_random.integers(0, self.world_size, size=2)
            self.agent_positions.append(pos)
        
        # Get initial observations
        observations = self._get_observations()
        info = {}
        
        return observations, info
    
    def step(self, actions):
        # Move agents
        for agent_id, action in actions.items():
            self._move_agent(agent_id, action)
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        # Check termination
        terminated = {agent_id: False for agent_id in range(self.n_agents)}
        truncated = {agent_id: False for agent_id in range(self.n_agents)}
        
        # Get observations
        observations = self._get_observations()
        info = {}
        
        return observations, rewards, terminated, truncated, info
    
    def _move_agent(self, agent_id, action):
        """Move agent according to action."""
        current_pos = self.agent_positions[agent_id].copy()
        
        if action == 0:  # Up
            current_pos[0] = max(0, current_pos[0] - 1)
        elif action == 1:  # Down
            current_pos[0] = min(self.world_size - 1, current_pos[0] + 1)
        elif action == 2:  # Left
            current_pos[1] = max(0, current_pos[1] - 1)
        elif action == 3:  # Right
            current_pos[1] = min(self.world_size - 1, current_pos[1] + 1)
        
        self.agent_positions[agent_id] = current_pos
    
    def _calculate_rewards(self):
        """Calculate rewards for all agents."""
        rewards = {}
        
        for agent_id in range(self.n_agents):
            # Simple reward: distance from center
            center = self.world_size // 2
            distance = np.linalg.norm(self.agent_positions[agent_id] - center)
            rewards[agent_id] = -distance  # Negative distance as reward
        
        return rewards
    
    def _get_observations(self):
        """Get observations for all agents."""
        observations = {}
        
        for agent_id in range(self.n_agents):
            # Simple observation: all agent positions
            obs = []
            for pos in self.agent_positions:
                obs.extend(pos)
            observations[agent_id] = np.array(obs, dtype=np.float32)
        
        return observations

# Use custom environment
custom_env = CustomMultiAgentEnv(n_agents=2, world_size=8)
algorithm = IndependentQLearning(env=custom_env, n_agents=2, device="cpu")
results = algorithm.train(n_episodes=200, max_steps_per_episode=30)
```

## Custom Agents

### Creating a Custom Agent

```python
from marl.agents import BaseAgent
import torch
import torch.nn as nn
import torch.optim as optim

class CustomAgent(BaseAgent):
    def __init__(self, agent_id, observation_space, action_space, learning_rate=0.001):
        super().__init__(agent_id, observation_space, action_space, learning_rate)
        
        # Define custom network
        self.network = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)
        )
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Custom parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def select_action(self, observation, training=True):
        """Select action using custom policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_space.n)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            q_values = self.network(obs_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def update(self, batch):
        """Update agent using custom learning rule."""
        if not batch:
            return {"loss": 0.0}
        
        # Custom update logic here
        # This is a simplified example
        loss = torch.tensor(0.0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return {"loss": loss.item(), "epsilon": self.epsilon}
    
    def save(self, filepath):
        """Save agent model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        """Load agent model."""
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

# Use custom agent
custom_agent = CustomAgent(0, env.observation_space, env.action_space)
```

## Performance Optimization

### Using GPU

```python
# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create algorithm with GPU
algorithm = IndependentQLearning(
    env=env,
    n_agents=2,
    device=device
)

# Train with GPU
results = algorithm.train(n_episodes=1000, max_steps_per_episode=50)
```

### Batch Processing

```python
# Increase batch size for better GPU utilization
algorithm = IndependentQLearning(
    env=env,
    n_agents=2,
    batch_size=64,  # Larger batch size
    buffer_size=50000,  # Larger buffer
    device="cuda"
)
```

### Parallel Training

```python
# Train multiple algorithms in parallel
import multiprocessing as mp

def train_algorithm(algorithm_config):
    algorithm = IndependentQLearning(**algorithm_config)
    return algorithm.train(n_episodes=500, max_steps_per_episode=50)

# Define multiple configurations
configs = [
    {'env': env, 'n_agents': 2, 'learning_rate': 0.001, 'device': 'cpu'},
    {'env': env, 'n_agents': 2, 'learning_rate': 0.01, 'device': 'cpu'},
    {'env': env, 'n_agents': 2, 'learning_rate': 0.0001, 'device': 'cpu'}
]

# Train in parallel
with mp.Pool(processes=3) as pool:
    results = pool.map(train_algorithm, configs)
```

## Best Practices

### 1. Start Simple

- Begin with simple environments and algorithms
- Use small numbers of agents initially
- Train for short episodes first

### 2. Monitor Training

```python
# Monitor training progress
def monitor_training(algorithm, env, n_episodes=1000):
    results = algorithm.train(n_episodes, max_steps_per_episode=50)
    
    # Plot progress every 100 episodes
    for i in range(0, n_episodes, 100):
        if i > 0:
            recent_rewards = results['episode_rewards'][i-100:i]
            print(f"Episodes {i-100}-{i}: Avg reward = {np.mean(recent_rewards):.2f}")
    
    return results
```

### 3. Hyperparameter Tuning

```python
# Systematic hyperparameter search
def hyperparameter_search(env, param_grid):
    best_score = -float('inf')
    best_params = None
    
    for params in param_grid:
        algorithm = IndependentQLearning(env=env, **params)
        results = algorithm.train(n_episodes=200, max_steps_per_episode=50)
        eval_results = algorithm.evaluate(n_episodes=20, max_steps_per_episode=50)
        
        score = eval_results['avg_reward']
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score
```

### 4. Save and Load Models

```python
# Save trained models
algorithm.save_models('trained_models/')

# Load trained models
algorithm.load_models('trained_models/')
```

### 5. Experiment Tracking

```python
# Track experiments
import json
from datetime import datetime

def track_experiment(algorithm, env, config, results):
    experiment_data = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'results': results,
        'environment': str(env),
        'algorithm': str(algorithm)
    }
    
    with open(f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(experiment_data, f, indent=2)
```

### 6. Error Handling

```python
# Robust training with error handling
def robust_train(algorithm, env, n_episodes=1000):
    try:
        results = algorithm.train(n_episodes, max_steps_per_episode=50)
        return results
    except Exception as e:
        print(f"Training failed: {e}")
        return None
```

## Conclusion

This tutorial has covered the basics of multi-agent reinforcement learning using our framework. Key takeaways:

1. **Start simple**: Begin with basic environments and algorithms
2. **Monitor progress**: Always visualize training progress
3. **Experiment**: Try different hyperparameters and algorithms
4. **Save results**: Keep track of your experiments
5. **Optimize**: Use GPU and batch processing for better performance

For more advanced topics, check out the examples directory and the full documentation.