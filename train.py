#!/usr/bin/env python3
"""
Training script for multi-agent reinforcement learning.

This script provides a command-line interface for training multi-agent RL agents
using different algorithms and environments.
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from marl.environments import MultiAgentGridWorld, CooperativeNavigation, PredatorPrey
from marl.algorithms import IndependentQLearning, MADDPG, MAPPO
from marl.utils import ConfigUtils, TrainingUtils
from marl.visualization import TrainingPlots


def create_environment(env_type: str, config: dict):
    """Create environment based on type and configuration."""
    if env_type == "grid_world":
        return MultiAgentGridWorld(
            grid_size=config.get('grid_size', (10, 10)),
            n_agents=config.get('n_agents', 2),
            n_targets=config.get('n_targets', 3),
            max_steps=config.get('max_steps', 100),
            reward_scale=config.get('reward_scale', 1.0),
            collision_penalty=config.get('collision_penalty', -0.1),
            target_reward=config.get('target_reward', 10.0),
            step_penalty=config.get('step_penalty', -0.01)
        )
    elif env_type == "cooperative_navigation":
        return CooperativeNavigation(
            n_agents=config.get('n_agents', 3),
            n_landmarks=config.get('n_landmarks', 3),
            world_size=config.get('world_size', 2.0),
            max_steps=config.get('max_steps', 100),
            collision_distance=config.get('collision_distance', 0.1),
            landmark_distance=config.get('landmark_distance', 0.1),
            collision_penalty=config.get('collision_penalty', -1.0),
            landmark_reward=config.get('landmark_reward', 10.0),
            step_penalty=config.get('step_penalty', -0.01)
        )
    elif env_type == "predator_prey":
        return PredatorPrey(
            n_predators=config.get('n_predators', 2),
            n_prey=config.get('n_prey', 1),
            world_size=config.get('world_size', 2.0),
            max_steps=config.get('max_steps', 100),
            capture_distance=config.get('capture_distance', 0.1),
            predator_speed=config.get('predator_speed', 0.1),
            prey_speed=config.get('prey_speed', 0.08),
            capture_reward=config.get('capture_reward', 10.0),
            escape_reward=config.get('escape_reward', 1.0),
            step_penalty=config.get('step_penalty', -0.01)
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def create_algorithm(algorithm_type: str, env, config: dict):
    """Create algorithm based on type and configuration."""
    if algorithm_type == "independent_q_learning":
        return IndependentQLearning(
            env=env,
            n_agents=config.get('n_agents', 2),
            learning_rate=config.get('learning_rate', 0.001),
            gamma=config.get('gamma', 0.99),
            epsilon=config.get('epsilon', 1.0),
            epsilon_min=config.get('epsilon_min', 0.01),
            epsilon_decay=config.get('epsilon_decay', 0.995),
            buffer_size=config.get('buffer_size', 10000),
            batch_size=config.get('batch_size', 32),
            target_update_freq=config.get('target_update_freq', 100),
            device=config.get('device', 'cpu')
        )
    elif algorithm_type == "maddpg":
        # For MADDPG, we need continuous action space
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
        return MADDPG(
            env=env,
            n_agents=config.get('n_agents', 2),
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=config.get('learning_rate', 0.001),
            gamma=config.get('gamma', 0.99),
            tau=config.get('tau', 0.005),
            buffer_size=config.get('buffer_size', 100000),
            batch_size=config.get('batch_size', 64),
            device=config.get('device', 'cpu')
        )
    elif algorithm_type == "mappo":
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
        return MAPPO(
            env=env,
            n_agents=config.get('n_agents', 2),
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=config.get('learning_rate', 0.001),
            gamma=config.get('gamma', 0.99),
            lambda_gae=config.get('lambda_gae', 0.95),
            clip_ratio=config.get('clip_ratio', 0.2),
            value_coef=config.get('value_coef', 0.5),
            entropy_coef=config.get('entropy_coef', 0.01),
            device=config.get('device', 'cpu')
        )
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")


def main():
    parser = argparse.ArgumentParser(description='Train multi-agent RL agents')
    
    # Environment arguments
    parser.add_argument('--env', type=str, default='grid_world',
                    choices=['grid_world', 'cooperative_navigation', 'predator_prey'],
                    help='Environment type')
    parser.add_argument('--n-agents', type=int, default=2, help='Number of agents')
    parser.add_argument('--grid-size', type=int, nargs=2, default=[10, 10],
                    help='Grid size for grid world environment')
    
    # Algorithm arguments
    parser.add_argument('--algorithm', type=str, default='independent_q_learning',
                    choices=['independent_q_learning', 'maddpg', 'mappo'],
                    help='Algorithm type')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial epsilon for exploration')
    parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum epsilon')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    
    # Training arguments
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--eval-frequency', type=int, default=100, help='Evaluation frequency')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--experiment-name', type=str, default='default', help='Experiment name')
    parser.add_argument('--save-models', action='store_true', help='Save trained models')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                    help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Config file
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training experiment: {args.experiment_name}")
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Number of agents: {args.n_agents}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    
    # Create environment
    env_config = {
        'n_agents': args.n_agents,
        'grid_size': tuple(args.grid_size),
        'max_steps': args.max_steps
    }
    
    env = create_environment(args.env, env_config)
    print(f"Created environment: {env}")
    
    # Create algorithm
    algorithm_config = {
        'n_agents': args.n_agents,
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'epsilon': args.epsilon,
        'epsilon_min': args.epsilon_min,
        'epsilon_decay': args.epsilon_decay,
        'device': device
    }
    
    algorithm = create_algorithm(args.algorithm, env, algorithm_config)
    print(f"Created algorithm: {algorithm}")
    
    # Training loop
    print(f"Starting training for {args.episodes} episodes...")
    
    results = algorithm.train(args.episodes, args.max_steps)
    
    # Save results
    results_file = output_dir / 'training_results.json'
    TrainingUtils.save_training_results(results, str(results_file))
    print(f"Saved training results to {results_file}")
    
    # Evaluation
    print("Evaluating trained agents...")
    eval_results = algorithm.evaluate(args.eval_episodes, args.max_steps)
    print(f"Evaluation results: {eval_results}")
    
    # Save evaluation results
    eval_file = output_dir / 'evaluation_results.json'
    TrainingUtils.save_training_results(eval_results, str(eval_file))
    
    # Save models
    if args.save_models:
        models_dir = output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        algorithm.save_models(str(models_dir / 'model'))
        print(f"Saved models to {models_dir}")
    
    # Generate plots
    if args.plot:
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Training progress plot
        TrainingPlots.plot_learning_curves(
            results['episode_rewards'],
            results['episode_lengths'],
            results['training_metrics'],
            save_path=str(plots_dir / 'training_progress.png')
        )
        
        # Learning analysis plot
        TrainingPlots.plot_learning_analysis(
            results['episode_rewards'],
            save_path=str(plots_dir / 'learning_analysis.png')
        )
        
        print(f"Generated plots in {plots_dir}")
    
    # Print summary
    summary = TrainingUtils.create_training_summary(results, args.algorithm, args.episodes)
    print(summary)
    
    # Save summary
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(f"Training completed! Results saved to {output_dir}")


if __name__ == '__main__':
    main()