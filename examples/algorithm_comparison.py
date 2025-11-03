#!/usr/bin/env python3
"""
Algorithm comparison demo for multi-agent reinforcement learning.

This script compares different MARL algorithms on the same environment
and generates comparison plots.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from marl.environments import MultiAgentGridWorld
from marl.algorithms import IndependentQLearning, MADDPG, MAPPO
from marl.utils import TrainingUtils, EvaluationUtils
from marl.visualization import TrainingPlots


def train_algorithm(algorithm_name, env, n_episodes=300):
    """Train a specific algorithm and return results."""
    print(f"\nTraining {algorithm_name}...")
    
    if algorithm_name == "Independent Q-Learning":
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
    elif algorithm_name == "MADDPG":
        # Note: MADDPG requires continuous actions, so we'll use a modified environment
        # For this demo, we'll skip MADDPG as it's designed for continuous action spaces
        print("Skipping MADDPG (requires continuous action space)")
        return None
    elif algorithm_name == "MAPPO":
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        algorithm = MAPPO(
            env=env,
            n_agents=2,
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=0.001,
            gamma=0.99,
            lambda_gae=0.95,
            clip_ratio=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            device="cpu"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    # Train
    results = algorithm.train(n_episodes, max_steps_per_episode=50)
    
    # Evaluate
    eval_results = algorithm.evaluate(n_episodes=20, max_steps_per_episode=50)
    
    return {
        'algorithm': algorithm,
        'results': results,
        'eval_results': eval_results
    }


def main():
    print("Multi-Agent RL Algorithm Comparison Demo")
    print("=" * 50)
    
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
    
    # Algorithms to compare
    algorithms = ["Independent Q-Learning", "MAPPO"]
    
    # Train all algorithms
    all_results = {}
    
    for algorithm_name in algorithms:
        result = train_algorithm(algorithm_name, env, n_episodes=300)
        if result is not None:
            all_results[algorithm_name] = result
    
    # Compare training results
    print("\nComparing training results...")
    
    # Extract episode rewards for comparison
    comparison_data = {}
    for algorithm_name, result in all_results.items():
        comparison_data[algorithm_name] = {
            'episode_rewards': result['results']['episode_rewards'],
            'episode_lengths': result['results']['episode_lengths']
        }
    
    # Plot comparison
    TrainingPlots.plot_algorithm_comparison(comparison_data, 'episode_rewards')
    
    # Compare evaluation results
    print("\nComparing evaluation results...")
    
    eval_comparison = {}
    for algorithm_name, result in all_results.items():
        eval_comparison[algorithm_name] = result['eval_results']
    
    EvaluationUtils.compare_agent_performance(eval_comparison)
    
    # Print summary
    print("\nAlgorithm Comparison Summary:")
    print("=" * 40)
    
    for algorithm_name, result in all_results.items():
        eval_results = result['eval_results']
        print(f"\n{algorithm_name}:")
        print(f"  Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"  Average Length: {eval_results['avg_length']:.2f} ± {eval_results['std_length']:.2f}")
        print(f"  Success Rate: {eval_results['success_rate']:.2%}")
    
    # Find best algorithm
    best_algorithm = max(all_results.items(), 
                        key=lambda x: x[1]['eval_results']['avg_reward'])
    
    print(f"\nBest performing algorithm: {best_algorithm[0]}")
    print(f"Best average reward: {best_algorithm[1]['eval_results']['avg_reward']:.2f}")
    
    # Demo with best algorithm
    print(f"\nRunning demo with best algorithm: {best_algorithm[0]}")
    
    env.render_mode = "human"
    best_algorithm_obj = best_algorithm[1]['algorithm']
    
    observations, _ = env.reset()
    total_reward = 0
    
    for step in range(50):
        # Select actions
        actions = {}
        for agent_id, obs in observations.items():
            if hasattr(best_algorithm_obj.agents[agent_id], 'get_action'):
                action = best_algorithm_obj.agents[agent_id].get_action(obs, training=False)
            else:
                # For MAPPO
                action, _, _ = best_algorithm_obj.agents[agent_id].select_action(obs, training=False)
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
    
    print("\nAlgorithm comparison completed successfully!")


if __name__ == '__main__':
    main()