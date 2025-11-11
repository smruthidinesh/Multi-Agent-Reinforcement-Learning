# Multi-Agent Reinforcement Learning Framework

A comprehensive Python framework for implementing and experimenting with multi-agent reinforcement learning (MARL) algorithms and environments.

## ✨ New Advanced Features

This framework now includes **state-of-the-art** multi-agent RL techniques from top-tier research:

### 🎯 Attention-based Communication (TarMAC)
- **Learned, targeted communication** between agents
- Multi-head attention for selective message processing
- Gated integration of communicated information
- **Reference**: Das et al., ICML 2019

### 🕸️ Graph Neural Networks (GNN)
- **Scalable coordination** for 100+ agents
- Dynamic graph construction based on agent proximity
- Graph Attention Networks (GAT), GCN, and MPNN implementations
- **Reference**: Veličković et al., ICLR 2018

### 🧠 Recurrent Policies (LSTM)
- **Memory-based agents** for partial observability
- Handle POMDPs and temporal dependencies
- Sequence-based training with experience replay
- **Reference**: Hausknecht & Stone, AAAI 2015

### 🔍 Intrinsic Curiosity Module (ICM)
- **Exploration bonuses** for sparse reward environments
- Prediction-based curiosity and novelty detection
- Random Network Distillation (RND) and count-based exploration
- **Reference**: Pathak et al., ICML 2017

## Features

- **Multiple Environments**: Grid World, Cooperative Navigation, Predator-Prey, Traffic, Robotics
- **Various Algorithms**: Independent Q-Learning, MADDPG, MAPPO, QMix
- **Flexible Agent Types**: DQN, Policy Gradient, Actor-Critic, **Attention DQN**, **GNN DQN**, **LSTM DQN**
- **Advanced Communication**: TarMAC, CommNet, Graph-based message passing
- **Exploration Mechanisms**: ICM, RND, count-based bonuses
- **Comprehensive Visualization**: Training plots, evaluation metrics, attention weights, graph structures
- **Easy-to-use Interface**: Command-line tools and Python API
- **Extensible Design**: Easy to add new environments, algorithms, and agents
- **Research-Ready**: Implements latest ICML/NeurIPS/ICLR papers

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

### 1. DQN Agent (Baseline)

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

### 2. Attention DQN Agent (TarMAC) ⭐ NEW

DQN with attention-based communication for coordinated decision-making.

```python
from marl.agents import AttentionDQNAgent

agent = AttentionDQNAgent(
    agent_id=0,
    observation_space=env.observation_space,
    action_space=env.action_space,
    message_dim=64,           # Communication message size
    num_heads=4,              # Number of attention heads
    use_communication=True,
    device="cuda"
)

# Get action and generate message
action, message = agent.get_action(
    observation=obs,
    other_messages=other_agent_messages,
    training=True
)
```

**Key Features**:
- Learned message generation from observations
- Multi-head attention over other agents' messages
- Gated integration of communication
- Suitable for 3-20 agents

### 3. GNN DQN Agent ⭐ NEW

DQN with Graph Neural Networks for scalable multi-agent coordination.

```python
from marl.agents import GNNDQNAgent

agent = GNNDQNAgent(
    agent_id=0,
    observation_space=env.observation_space,
    action_space=env.action_space,
    n_agents=100,             # Scales to many agents
    gnn_type="gat",           # GAT, GCN, or MPNN
    num_gnn_layers=3,
    num_heads=4,
    k_neighbors=5,            # k-NN graph construction
    device="cuda"
)

# Get action with graph-based coordination
action = agent.get_action(
    observation=obs,
    all_observations=all_agent_obs,
    positions=agent_positions,
    training=True
)
```

**Key Features**:
- Dynamic graph construction based on proximity
- Message passing on agent graphs
- Scales to 100+ agents efficiently
- Graph Attention, GCN, or MPNN architectures

### 4. LSTM DQN Agent ⭐ NEW

DQN with LSTM memory for partially observable environments.

```python
from marl.agents import LSTMDQNAgent

agent = LSTMDQNAgent(
    agent_id=0,
    observation_space=env.observation_space,
    action_space=env.action_space,
    lstm_hidden_dim=128,
    num_lstm_layers=1,
    sequence_length=8,        # Training sequence length
    device="cuda"
)

# Reset hidden state at episode start
agent.reset_episode()

# Get action (hidden state persists across steps)
action = agent.get_action(observation, training=True)
```

**Key Features**:
- LSTM memory for temporal reasoning
- Handles partial observability (POMDPs)
- Sequence-based training
- Remembers past observations

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

## Advanced Communication

### TarMAC (Targeted Multi-Agent Communication) ⭐ NEW

Learned, attention-based communication where agents decide what and when to communicate.

```python
from marl.agents import AttentionDQNAgent

# Create agents with TarMAC
agents = [
    AttentionDQNAgent(
        agent_id=i,
        observation_space=env.observation_space,
        action_space=env.action_space,
        message_dim=64,
        num_heads=4,
        use_communication=True
    )
    for i in range(n_agents)
]

# Training loop with communication
obs, _ = env.reset()
messages = {i: None for i in range(n_agents)}

for step in range(max_steps):
    actions = {}
    new_messages = {}

    for i, agent in enumerate(agents):
        # Collect messages from other agents
        other_msgs = [messages[j] for j in range(n_agents) if j != i]
        other_msgs_tensor = torch.stack(other_msgs) if other_msgs else None

        # Get action and generate message
        action, message = agent.get_action(obs[i], other_msgs_tensor)
        actions[i] = action
        new_messages[i] = message

    messages = new_messages
    obs, rewards, dones, _, _ = env.step(actions)
```

## Intrinsic Curiosity & Exploration ⭐ NEW

Enhance exploration in sparse reward environments with intrinsic motivation.

```python
from marl.utils.curiosity import IntrinsicCuriosityModule

# Create ICM module
icm = IntrinsicCuriosityModule(
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    feature_dim=64,
    beta=0.2,    # Inverse model loss weight
    eta=0.5      # Intrinsic reward scale
)

# During training
obs_tensor = torch.FloatTensor(obs)
next_obs_tensor = torch.FloatTensor(next_obs)
action_tensor = torch.LongTensor([action])

# Compute intrinsic reward
intrinsic_reward, losses = icm(obs_tensor, next_obs_tensor, action_tensor)

# Augment environment reward
total_reward = env_reward + intrinsic_reward

# Update ICM
icm_optimizer.zero_grad()
losses['icm_loss'].backward()
icm_optimizer.step()
```

**Available Curiosity Methods**:
- **ICM (Intrinsic Curiosity Module)**: Prediction error as curiosity
- **RND (Random Network Distillation)**: Novelty detection
- **Count-based**: Exploration bonus for rare states

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

## Advanced Features Quick Start

### 🚀 Try the Demo Notebook

```bash
jupyter notebook notebooks/advanced_features_demo.ipynb
```

The notebook includes:
- Interactive demos of all advanced features
- Visualization of attention weights and graph structures
- Performance comparisons
- Code examples ready to run

### 📚 Learn More

- **[Advanced Features Documentation](docs/advanced_features.md)** - Detailed technical guide
- **[Getting Started Guide](docs/getting_started.md)** - Beginner-friendly tutorial
- **[API Reference](docs/api_reference.md)** - Complete API documentation

### 🔬 Research Papers Implemented

This framework implements algorithms from top-tier research venues:

1. **Das, A., et al.** (2019). "TarMAC: Targeted Multi-Agent Communication." *ICML*.
2. **Veličković, P., et al.** (2018). "Graph Attention Networks." *ICLR*.
3. **Pathak, D., et al.** (2017). "Curiosity-driven Exploration by Self-supervised Prediction." *ICML*.
4. **Hausknecht, M., & Stone, P.** (2015). "Deep Recurrent Q-Learning for Partially Observable MDPs." *AAAI*.
5. **Burda, Y., et al.** (2019). "Exploration by Random Network Distillation." *ICLR*.
6. **Rashid, T., et al.** (2018). "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning." *ICML*.
7. **Lowe, R., et al.** (2017). "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." *NeurIPS*.

### 🎯 Use Cases

**Choose the right agent for your task:**

| Task Type | Recommended Agent | Why? |
|-----------|------------------|------|
| Coordination with communication | AttentionDQN | Learned selective communication |
| Large-scale systems (50+ agents) | GNNDQN | Graph-based scalability |
| Partial observability | LSTMDQN | Memory of past observations |
| Sparse rewards | Any agent + ICM | Exploration bonus |
| Real-time systems | DQN or GNN | Low latency |

## Project Structure

```
multi_agent_rl/
├── src/marl/
│   ├── environments/          # Environment implementations
│   ├── agents/                # Agent implementations (⭐ new agents)
│   │   ├── attention_dqn_agent.py  # TarMAC communication
│   │   ├── gnn_dqn_agent.py        # Graph Neural Networks
│   │   └── lstm_dqn_agent.py       # Recurrent policies
│   ├── algorithms/            # MARL algorithm implementations
│   ├── utils/                 # Utility functions (⭐ new modules)
│   │   ├── attention.py            # Attention mechanisms
│   │   ├── graph_networks.py       # GNN implementations
│   │   └── curiosity.py            # ICM, RND, exploration
│   └── visualization/         # Visualization tools
├── examples/                  # Example scripts
├── notebooks/                  # Jupyter notebooks
│   └── advanced_features_demo.ipynb  # ⭐ NEW: Interactive demo
├── tests/                     # Unit tests
├── docs/                      # Documentation
│   └── advanced_features.md   # ⭐ NEW: Technical documentation
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
-Add AI 