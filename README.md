# Multi-Agent Reinforcement Learning Framework

A comprehensive Python framework for implementing and experimenting with multi-agent reinforcement learning (MARL) algorithms and environments.

## Features

- **Multiple Environments**: Grid World, Cooperative Navigation, Predator-Prey
- **Various Algorithms**: Independent Q-Learning, MADDPG, MAPPO
- **Flexible Agent Types**: DQN, Policy Gradient, Actor-Critic
- **Comprehensive Visualization**: Training plots, evaluation metrics, agent behavior analysis
- **Easy-to-use Interface**: Command-line tools and Python API
- **Extensible Design**: Easy to add new environments, algorithms, and agents

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib, Seaborn
- Gymnasium (OpenAI Gym)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Package (Optional)

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from marl.environments import MultiAgentGridWorld
from marl.algorithms import IndependentQLearning

# Create environment
env = MultiAgentGridWorld(
    grid_size=(10, 10),
    n_agents=2,
    n_targets=3,
    max_steps=100
)

# Create algorithm
algorithm = IndependentQLearning(
    env=env,
    n_agents=2,
    learning_rate=0.001,
    gamma=0.99
)

# Train agents
results = algorithm.train(n_episodes=1000)

# Evaluate
eval_results = algorithm.evaluate(n_episodes=100)
```

### Command Line Interface

#### Training

```bash
python train.py --env grid_world --algorithm independent_q_learning --episodes 1000 --n-agents 2
```

#### Evaluation

```bash
python evaluate.py --model-path results/default/models/model --algorithm independent_q_learning --episodes 100
```

## Environments

### 1. Grid World

A discrete grid-based environment where multiple agents must collect targets while avoiding collisions.

```python
env = MultiAgentGridWorld(
    grid_size=(10, 10),      # Grid dimensions
    n_agents=2,             # Number of agents
    n_targets=3,            # Number of targets
    max_steps=100,          # Maximum steps per episode
    target_reward=10.0,     # Reward for collecting target
    collision_penalty=-0.1, # Penalty for collisions
    step_penalty=-0.01     # Penalty per step
)
```

**Features:**
- Discrete action space (up, down, left, right, stay)
- Partial observability
- Cooperative objectives
- Collision avoidance

### 2. Cooperative Navigation

A continuous environment where agents must navigate to landmarks while avoiding collisions.

```python
env = CooperativeNavigation(
    n_agents=3,             # Number of agents
    n_landmarks=3,           # Number of landmarks
    world_size=2.0,          # World size
    max_steps=100,           # Maximum steps per episode
    landmark_reward=10.0,    # Reward for reaching landmark
    collision_penalty=-1.0   # Penalty for collisions
)
```

**Features:**
- Continuous action space
- Continuous state space
- Cooperative objectives
- Collision avoidance

### 3. Predator-Prey

A competitive environment where predators try to catch prey while prey try to escape.

### 4. Traffic

A simple multi-agent traffic environment where agents must navigate to their goals while avoiding collisions.

```python
env = TrafficEnv(
    n_agents=2,             # Number of agents
    grid_size=10,           # Grid dimensions
)
```

**Features:**
- Discrete action space
- Cooperative objectives
- Collision avoidance

### 5. Robotics (Multi-Agent Robot Navigation)

A 2D environment where multiple robotic agents navigate to target locations while avoiding obstacles and collisions.

```python
from marl.environments import MultiAgentRobotNavigation

env = MultiAgentRobotNavigation(
    n_agents=2,             # Number of agents
    grid_size=10,           # Grid dimensions (e.g., 10x10)
    n_targets=2,            # Number of target locations
    n_obstacles=3,          # Number of static obstacles
    max_steps=100,          # Maximum steps per episode
    target_reward=10.0,     # Reward for reaching a target
    collision_penalty=-5.0, # Penalty for collisions with obstacles or other agents
    step_penalty=-0.1       # Penalty per step to encourage efficiency
)
```

**Features:**
- Continuous observation space (positions, velocities)
- Discrete action space (move forward, turn left, turn right, stay)
- Collision avoidance
- Cooperative objectives (all agents reach targets)

## Algorithms

### 1. Independent Q-Learning

Each agent learns independently using Deep Q-Networks (DQN) without considering other agents.

```python
algorithm = IndependentQLearning(
    env=env,
    n_agents=2,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    buffer_size=10000,
    batch_size=32
)
```

**Features:**
- Experience replay
- Target networks
- Epsilon-greedy exploration
- Independent learning

### 2. MADDPG (Multi-Agent Deep Deterministic Policy Gradient)

Centralized training with decentralized execution for continuous action spaces.

```python
algorithm = MADDPG(
    env=env,
    n_agents=2,
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=0.001,
    gamma=0.99,
    tau=0.005
)
```

**Features:**
- Actor-critic architecture
- Centralized training
- Decentralized execution
- Continuous actions

### 3. MAPPO (Multi-Agent Proximal Policy Optimization)

Proximal Policy Optimization adapted for multi-agent settings.

### 4. QMIX

QMIX is a value-based algorithm that learns a monotonic value function factorization.

```python
algorithm = QMix(
    env=env,
    n_agents=2,
    state_dim=state_dim,
    action_dim=action_dim,
    mixing_embed_dim=32
)
```

**Features:**
- Value function factorization
- Centralized training
- Decentralized execution

## Agents

### 1. DQN Agent

Deep Q-Network agent with experience replay and target networks.

```python
from marl.agents import DQNAgent

agent = DQNAgent(
    agent_id=0,
    observation_space=env.observation_space,
    action_space=env.action_space,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0
)
```

### 2. Policy Gradient Agent

REINFORCE agent with baseline for variance reduction.

```python
from marl.agents import PolicyGradientAgent

agent = PolicyGradientAgent(
    agent_id=0,
    observation_space=env.observation_space,
    action_space=env.action_space,
    learning_rate=0.001,
    gamma=0.99
)
```

### 3. Actor-Critic Agent

Actor-critic agent with separate networks for policy and value function.

## Hierarchical Multi-Agent Systems

The framework now supports hierarchical multi-agent systems, where a high-level policy can set goals for low-level policies.

```python
# Create a hierarchical agent
agent = HierarchicalAgent(
    agent_id=0,
    observation_space=env.observation_space,
    action_space=env.action_space,
    high_level_policy=high_level_policy,
    low_level_policy=low_level_policy
)
```

## Communication

Agents can communicate with each other by sending and receiving messages. The messages are added to the agents' observations.

```python
# Enable communication in the environment
env = MultiAgentGridWorld(
    n_agents=2,
    message_dim=4
)

# Agents can now send and receive messages
agent.send_message(torch.randn(4))
messages = agent.get_messages()
```

## Visualization

### Training Plots

```python
from marl.visualization import TrainingPlots

TrainingPlots.plot_learning_curves(
    episode_rewards,
    episode_lengths,
    training_metrics
)
```

### Evaluation Analysis

```python
from marl.utils import EvaluationUtils

EvaluationUtils.plot_evaluation_results(eval_results)
```

### Web-based Visualization

A web-based visualization interface is available for real-time monitoring of the environment.

```bash
python examples/web_visualization_demo.py
```

## Examples

### Basic Demo

```bash
python examples/basic_demo.py
```

### Algorithm Comparison

```bash
python examples/algorithm_comparison.py
```

### Environment Demo

```bash
python examples/environment_demo.py
```

## Configuration

### Using Configuration Files

```python
from marl.utils import ConfigUtils

# Load configuration
config = ConfigUtils.load_config('config.yaml')

# Create experiment
env = create_environment(config.environment)
algorithm = create_algorithm(config.algorithm, env)
```

### Configuration Example

```yaml
environment:
  env_type: "grid_world"
  grid_size: [10, 10]
  n_agents: 2
  n_targets: 3
  max_steps: 100

agent:
  agent_type: "dqn"
  learning_rate: 0.001
  gamma: 0.99
  epsilon: 1.0
  epsilon_min: 0.01

training:
  n_episodes: 1000
  max_steps_per_episode: 100
  device: "cpu"
  seed: 42

algorithm:
  algorithm_type: "independent_q_learning"
```

## Project Structure

```
multi_agent_rl/
├── src/marl/
│   ├── environments/          # Environment implementations
│   ├── agents/                # Agent implementations
│   ├── algorithms/            # MARL algorithm implementations
│   ├── utils/                 # Utility functions
│   └── visualization/         # Visualization tools
├── examples/                  # Example scripts
├── notebooks/                  # Jupyter notebooks
├── tests/                     # Unit tests
├── docs/                      # Documentation
├── train.py                   # Training script
├── evaluate.py               # Evaluation script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{multi_agent_rl_framework,
  title={Multi-Agent Reinforcement Learning Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/multi-agent-rl}
}
```

## Acknowledgments

- OpenAI Gym/Gymnasium for environment standards
- PyTorch for deep learning framework
- The multi-agent RL research community

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Slow training**: Check if GPU is being used correctly
3. **Import errors**: Ensure all dependencies are installed
4. **Environment rendering**: Install matplotlib and ensure display is available

### Getting Help

- Check the examples directory
- Read the documentation
- Open an issue on GitHub
- Check the troubleshooting section