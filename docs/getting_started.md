# Getting Started with Multi-Agent RL

This guide will help you get started with the Multi-Agent Reinforcement Learning framework.

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/multi-agent-rl.git
cd multi-agent-rl
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```python
import marl
print("Multi-Agent RL framework installed successfully!")
```

## Your First Multi-Agent RL Experiment

### Step 1: Create an Environment

```python
from marl.environments import MultiAgentGridWorld

# Create a simple grid world environment
env = MultiAgentGridWorld(
    grid_size=(8, 8),    # 8x8 grid
    n_agents=2,         # 2 agents
    n_targets=2,       # 2 targets to collect
    max_steps=50        # Maximum 50 steps per episode
)
```

### Step 2: Create an Algorithm

```python
from marl.algorithms import IndependentQLearning

# Create Independent Q-Learning algorithm
algorithm = IndependentQLearning(
    env=env,
    n_agents=2,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995
)
```

### Step 3: Train the Agents

```python
# Train for 500 episodes
results = algorithm.train(n_episodes=500, max_steps_per_episode=50)

print(f"Training completed!")
print(f"Final average reward: {np.mean(results['episode_rewards'][-100:]):.2f}")
```

### Step 4: Evaluate the Trained Agents

```python
# Evaluate the trained agents
eval_results = algorithm.evaluate(n_episodes=20, max_steps_per_episode=50)

print("Evaluation Results:")
print(f"Average Reward: {eval_results['avg_reward']:.2f}")
print(f"Success Rate: {eval_results['success_rate']:.2%}")
```

### Step 5: Visualize Results

```python
from marl.visualization import TrainingPlots

# Plot training progress
TrainingPlots.plot_learning_curves(
    results['episode_rewards'],
    results['episode_lengths'],
    results['training_metrics']
)
```

## Understanding the Framework

### Environment Structure

All environments follow the Gymnasium interface:

```python
# Reset environment
observations, info = env.reset()

# Step through environment
actions = {0: 1, 1: 2}  # Actions for each agent
next_observations, rewards, terminated, truncated, info = env.step(actions)
```

### Agent Structure

Agents implement a common interface:

```python
# Select action
action = agent.select_action(observation, training=True)

# Update policy
metrics = agent.update(batch)

# Save/load model
agent.save('model.pth')
agent.load('model.pth')
```

### Algorithm Structure

Algorithms manage multiple agents:

```python
# Train all agents
results = algorithm.train(n_episodes=1000)

# Evaluate all agents
eval_results = algorithm.evaluate(n_episodes=100)

# Save/load all models
algorithm.save_models('models/')
algorithm.load_models('models/')
```

## Common Use Cases

### 1. Quick Experiment

```python
# Run a quick experiment
python train.py --env grid_world --algorithm independent_q_learning --episodes 500
```

### 2. Compare Algorithms

```python
# Compare different algorithms
python examples/algorithm_comparison.py
```

### 3. Custom Environment

```python
# Create your own environment
class MyEnvironment(gym.Env):
    def __init__(self):
        # Implement your environment
        pass
    
    def reset(self):
        # Reset environment
        pass
    
    def step(self, actions):
        # Execute actions
        pass
```

### 4. Custom Agent

```python
# Create your own agent
class MyAgent(BaseAgent):
    def __init__(self, agent_id, observation_space, action_space):
        super().__init__(agent_id, observation_space, action_space)
        # Initialize your agent
    
    def select_action(self, observation, training=True):
        # Implement action selection
        pass
    
    def update(self, batch):
        # Implement learning update
        pass
```

## Best Practices

### 1. Start Simple

- Begin with the grid world environment
- Use Independent Q-Learning algorithm
- Train for a small number of episodes first

### 2. Monitor Training

- Plot learning curves regularly
- Check for convergence
- Adjust hyperparameters if needed

### 3. Evaluate Thoroughly

- Use multiple evaluation episodes
- Compare with random baselines
- Analyze agent behavior

### 4. Save Results

- Save trained models
- Save training metrics
- Document your experiments

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the right directory
   cd multi_agent_rl
   python -c "import marl"
   ```

2. **CUDA Issues**
   ```python
   # Use CPU if CUDA is not available
   device = "cpu"
   ```

3. **Memory Issues**
   ```python
   # Reduce batch size or buffer size
   batch_size = 16
   buffer_size = 5000
   ```

4. **Slow Training**
   ```python
   # Check if GPU is being used
   print(f"Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
   ```

### Getting Help

- Check the examples directory
- Read the documentation
- Open an issue on GitHub
- Join the community discussions

## Next Steps

1. **Explore Examples**: Run the example scripts
2. **Read Documentation**: Check the full documentation
3. **Experiment**: Try different environments and algorithms
4. **Contribute**: Add new environments or algorithms
5. **Share**: Share your results with the community

## Resources

- [Multi-Agent RL Tutorial](tutorial.md)
- [API Reference](api_reference.md)
- [Examples Gallery](examples.md)
- [Research Papers](papers.md)
- [Community Forum](https://github.com/yourusername/multi-agent-rl/discussions)