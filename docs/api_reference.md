# API Reference

This document provides a comprehensive reference for the Multi-Agent RL framework API.

## Environments

### MultiAgentGridWorld

A discrete grid-based environment for multi-agent reinforcement learning.

```python
class MultiAgentGridWorld(gym.Env):
    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        n_agents: int = 2,
        n_targets: int = 3,
        max_steps: int = 100,
        reward_scale: float = 1.0,
        collision_penalty: float = -0.1,
        target_reward: float = 10.0,
        step_penalty: float = -0.01,
        render_mode: Optional[str] = None
    )
```

**Parameters:**
- `grid_size`: Dimensions of the grid (height, width)
- `n_agents`: Number of agents in the environment
- `n_targets`: Number of targets to collect
- `max_steps`: Maximum steps per episode
- `reward_scale`: Scaling factor for rewards
- `collision_penalty`: Penalty for agent collisions
- `target_reward`: Reward for collecting a target
- `step_penalty`: Penalty per step
- `render_mode`: Rendering mode ("human" or "rgb_array")

**Methods:**
- `reset(seed=None, options=None)`: Reset environment to initial state
- `step(actions)`: Execute actions and return next state
- `render()`: Render the environment
- `close()`: Close the environment

### CooperativeNavigation

A continuous environment for cooperative navigation tasks.

```python
class CooperativeNavigation(gym.Env):
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
    )
```

**Parameters:**
- `n_agents`: Number of agents
- `n_landmarks`: Number of landmarks to reach
- `world_size`: Size of the world
- `max_steps`: Maximum steps per episode
- `collision_distance`: Distance threshold for collisions
- `landmark_distance`: Distance threshold for reaching landmarks
- `collision_penalty`: Penalty for collisions
- `landmark_reward`: Reward for reaching landmarks
- `step_penalty`: Penalty per step
- `render_mode`: Rendering mode

### PredatorPrey

A competitive environment with predators and prey.

```python
class PredatorPrey(gym.Env):
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
    )
```

**Parameters:**
- `n_predators`: Number of predator agents
- `n_prey`: Number of prey agents
- `world_size`: Size of the world
- `max_steps`: Maximum steps per episode
- `capture_distance`: Distance threshold for capture
- `predator_speed`: Speed of predator agents
- `prey_speed`: Speed of prey agents
- `capture_reward`: Reward for capturing prey
- `escape_reward`: Reward for escaping
- `step_penalty`: Penalty per step
- `render_mode`: Rendering mode

## Agents

### BaseAgent

Abstract base class for all agents.

```python
class BaseAgent(ABC):
    def __init__(
        self,
        agent_id: int,
        observation_space: Any,
        action_space: Any,
        learning_rate: float = 0.001,
        device: str = "cpu"
    )
```

**Abstract Methods:**
- `select_action(observation, training=True)`: Select action given observation
- `update(batch)`: Update agent's policy/value function
- `save(filepath)`: Save agent's model
- `load(filepath)`: Load agent's model

### DQNAgent

Deep Q-Network agent with experience replay.

```python
class DQNAgent(BaseAgent):
    def __init__(
        self,
        agent_id: int,
        observation_space: Any,
        action_space: Any,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        device: str = "cpu"
    )
```

**Parameters:**
- `agent_id`: Unique identifier for the agent
- `observation_space`: Environment observation space
- `action_space`: Environment action space
- `learning_rate`: Learning rate for optimizer
- `gamma`: Discount factor
- `epsilon`: Initial exploration rate
- `epsilon_min`: Minimum exploration rate
- `epsilon_decay`: Exploration decay rate
- `buffer_size`: Size of experience replay buffer
- `batch_size`: Batch size for training
- `target_update_freq`: Frequency of target network updates
- `device`: Device to use for computation

### PolicyGradientAgent

REINFORCE agent with baseline.

```python
class PolicyGradientAgent(BaseAgent):
    def __init__(
        self,
        agent_id: int,
        observation_space: Any,
        action_space: Any,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        device: str = "cpu"
    )
```

### ActorCriticAgent

Actor-critic agent with separate networks.

```python
class ActorCriticAgent(BaseAgent):
    def __init__(
        self,
        agent_id: int,
        observation_space: Any,
        action_space: Any,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        device: str = "cpu"
    )
```

## Algorithms

### IndependentQLearning

Independent Q-Learning algorithm for multi-agent RL.

```python
class IndependentQLearning:
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
        device: str = "cpu"
    )
```

**Methods:**
- `train(n_episodes, max_steps_per_episode=100)`: Train all agents
- `evaluate(n_episodes=10, max_steps_per_episode=100)`: Evaluate trained agents
- `save_models(filepath)`: Save all agent models
- `load_models(filepath)`: Load all agent models
- `get_agent_stats()`: Get statistics for all agents

### MADDPG

Multi-Agent Deep Deterministic Policy Gradient.

```python
class MADDPG:
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
    )
```

**Parameters:**
- `env`: Environment instance
- `n_agents`: Number of agents
- `state_dim`: Dimension of state space
- `action_dim`: Dimension of action space
- `learning_rate`: Learning rate
- `gamma`: Discount factor
- `tau`: Soft update parameter
- `buffer_size`: Experience replay buffer size
- `batch_size`: Training batch size
- `device`: Computation device

### MAPPO

Multi-Agent Proximal Policy Optimization.

```python
class MAPPO:
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
    )
```

**Parameters:**
- `env`: Environment instance
- `n_agents`: Number of agents
- `state_dim`: Dimension of state space
- `action_dim`: Dimension of action space
- `learning_rate`: Learning rate
- `gamma`: Discount factor
- `lambda_gae`: GAE lambda parameter
- `clip_ratio`: PPO clip ratio
- `value_coef`: Value function coefficient
- `entropy_coef`: Entropy coefficient
- `device`: Computation device

## Utilities

### TrainingUtils

Utility functions for training.

```python
class TrainingUtils:
    @staticmethod
    def moving_average(data: List[float], window_size: int = 100) -> List[float]
    
    @staticmethod
    def plot_training_progress(
        episode_rewards: List[float],
        episode_lengths: List[float],
        training_metrics: List[Dict[str, float]],
        save_path: Optional[str] = None
    ) -> None
    
    @staticmethod
    def save_training_results(results: Dict[str, Any], filepath: str) -> None
    
    @staticmethod
    def load_training_results(filepath: str) -> Dict[str, Any]
    
    @staticmethod
    def compare_algorithms(
        results: Dict[str, Dict[str, List[float]]],
        metric: str = 'episode_rewards',
        window_size: int = 100,
        save_path: Optional[str] = None
    ) -> None
```

### EvaluationUtils

Utility functions for evaluation.

```python
class EvaluationUtils:
    @staticmethod
    def evaluate_episode(
        env,
        agents: Dict[int, Any],
        max_steps: int = 100,
        render: bool = False
    ) -> Dict[str, Any]
    
    @staticmethod
    def evaluate_agents(
        env,
        agents: Dict[int, Any],
        n_episodes: int = 100,
        max_steps: int = 100,
        render: bool = False
    ) -> Dict[str, Any]
    
    @staticmethod
    def plot_evaluation_results(
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None
    
    @staticmethod
    def compare_agent_performance(
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> None
```

### ConfigUtils

Utility functions for configuration management.

```python
class ConfigUtils:
    @staticmethod
    def load_config(filepath: str) -> ExperimentConfig
    
    @staticmethod
    def save_config(config: ExperimentConfig, filepath: str) -> None
    
    @staticmethod
    def create_default_config() -> ExperimentConfig
    
    @staticmethod
    def validate_config(config: ExperimentConfig) -> bool
    
    @staticmethod
    def setup_experiment_directory(config: ExperimentConfig) -> str
```

## Visualization

### TrainingPlots

Plotting functions for training visualization.

```python
class TrainingPlots:
    @staticmethod
    def plot_learning_curves(
        episode_rewards: List[float],
        episode_lengths: List[float],
        training_metrics: List[Dict[str, float]],
        window_size: int = 100,
        save_path: Optional[str] = None
    ) -> None
    
    @staticmethod
    def plot_algorithm_comparison(
        results: Dict[str, Dict[str, List[float]]],
        metric: str = 'episode_rewards',
        window_size: int = 100,
        save_path: Optional[str] = None
    ) -> None
    
    @staticmethod
    def plot_agent_performance(
        agent_rewards: Dict[int, List[float]],
        save_path: Optional[str] = None
    ) -> None
    
    @staticmethod
    def plot_learning_analysis(
        episode_rewards: List[float],
        window_size: int = 100,
        save_path: Optional[str] = None
    ) -> None
```

## Configuration Classes

### ExperimentConfig

Complete experiment configuration.

```python
@dataclass
class ExperimentConfig:
    environment: EnvironmentConfig
    agent: AgentConfig
    training: TrainingConfig
    algorithm: AlgorithmConfig
    experiment_name: str = "default_experiment"
    output_dir: str = "results"
```

### EnvironmentConfig

Environment configuration.

```python
@dataclass
class EnvironmentConfig:
    env_type: str = "grid_world"
    grid_size: tuple = (10, 10)
    n_agents: int = 2
    n_targets: int = 3
    max_steps: int = 100
    reward_scale: float = 1.0
    collision_penalty: float = -0.1
    target_reward: float = 10.0
    step_penalty: float = -0.01
```

### AgentConfig

Agent configuration.

```python
@dataclass
class AgentConfig:
    agent_type: str = "dqn"
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100
    hidden_dims: list = None
```

### TrainingConfig

Training configuration.

```python
@dataclass
class TrainingConfig:
    n_episodes: int = 1000
    max_steps_per_episode: int = 100
    eval_frequency: int = 100
    eval_episodes: int = 10
    save_frequency: int = 500
    device: str = "cpu"
    seed: int = 42
```

### AlgorithmConfig

Algorithm configuration.

```python
@dataclass
class AlgorithmConfig:
    algorithm_type: str = "independent_q_learning"
    tau: float = 0.005  # MADDPG specific
    lambda_gae: float = 0.95  # MAPPO specific
    clip_ratio: float = 0.2  # MAPPO specific
    value_coef: float = 0.5  # MAPPO specific
    entropy_coef: float = 0.01  # MAPPO specific
```

## Error Handling

### Common Exceptions

- `ValueError`: Invalid parameter values
- `RuntimeError`: Environment or algorithm errors
- `FileNotFoundError`: Missing model files
- `ImportError`: Missing dependencies

### Best Practices

1. Always validate inputs
2. Handle exceptions gracefully
3. Provide meaningful error messages
4. Log important events
5. Use type hints for better error detection