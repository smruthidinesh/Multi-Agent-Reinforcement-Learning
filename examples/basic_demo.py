#!/usr/bin/env python3
"""
Basic demo script for multi-agent reinforcement learning.

This script demonstrates how to use the multi-agent RL framework
with a simple grid world environment.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from marl.environments import MultiAgentGridWorld
from marl.algorithms import IndependentQLearning
from marl.utils import TrainingUtils, EvaluationUtils
from marl.visualization import TrainingPlots


def main():
    print("Multi-Agent RL Basic Demo")
    print("=" * 40)
    
    # Create environment
    print("Creating grid world environment...")
    env = MultiAgentGridWorld(
        grid_size=(8, 8),
        n_agents=2,
        n_targets=2,
        max_steps=50,
        reward_scale=1.0,
        collision_penalty=-0.1,
        target_reward=10.0,
        step_penalty=-0.01
    )
    
    print(f"Environment created: {env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create algorithm
    print("\nCreating Independent Q-Learning algorithm...")
    algorithm = IndependentQLearning(
        env=env,
        n_agents=2,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=5000,
        batch_size=32,
        target_update_freq=50,
        device="cpu"
    )
    
    print(f"Algorithm created: {algorithm}")
    
    # Training
    print("\nStarting training...")
    n_episodes = 500
    
    results = algorithm.train(n_episodes, max_steps_per_episode=50)
    
    print(f"Training completed!")
    print(f"Final average reward: {np.mean(results['episode_rewards'][-100:]):.2f}")
    
    # Plot training progress
    print("\nGenerating training plots...")
    TrainingPlots.plot_learning_curves(
        results['episode_rewards'],
        results['episode_lengths'],
        results['training_metrics']
    )
    
    # Evaluation
    print("\nEvaluating trained agents...")
    eval_results = algorithm.evaluate(n_episodes=20, max_steps_per_episode=50)
    
    print("Evaluation Results:")
    print(f"Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Average Length: {eval_results['avg_length']:.2f} ± {eval_results['std_length']:.2f}")
    print(f"Success Rate: {eval_results['success_rate']:.2%}")
    
    # Plot evaluation results
    EvaluationUtils.plot_evaluation_results(eval_results)
    
    # Demo episode
    print("\nRunning demo episode...")
    env.render_mode = "human"
    
    observations, _ = env.reset()
    total_reward = 0
    
    for step in range(50):
        # Select actions
        actions = {}
        for agent_id, obs in observations.items():
            action = algorithm.agents[agent_id].get_action(obs, training=False)
            actions[agent_id] = action
        
        # Execute actions
        next_observations, rewards, terminated, truncated, _ = env.step(actions)
        
        total_reward += sum(rewards.values())
        
        # Render
        env.render()
        
        # Check termination
        if all(terminated.values()) or all(truncated.values()):
            break
        
        observations = next_observations
    
    print(f"Demo episode completed with total reward: {total_reward:.2f}")
    
    # Agent statistics
    print("\nAgent Statistics:")
    agent_stats = algorithm.get_agent_stats()
    for agent_id, stats in agent_stats.items():
        print(f"Agent {agent_id}:")
        print(f"  Training steps: {stats['training_steps']}")
        print(f"  Average reward: {stats['avg_reward']:.2f}")
        print(f"  Average length: {stats['avg_length']:.2f}")
    
    print("\nDemo completed successfully!")


if __name__ == '__main__':
    main()