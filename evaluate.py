#!/usr/bin/env python3
"""
Evaluation script for multi-agent reinforcement learning.

This script provides a command-line interface for evaluating trained
multi-agent RL agents and analyzing their performance.
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
from marl.utils import EvaluationUtils
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
            step_penalty=config.get('step_penalty', -0.01),
            render_mode="human"
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
            step_penalty=config.get('step_penalty', -0.01),
            render_mode="human"
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
            step_penalty=config.get('step_penalty', -0.01),
            render_mode="human"
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
    parser = argparse.ArgumentParser(description='Evaluate multi-agent RL agents')
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--algorithm', type=str, required=True,
                    choices=['independent_q_learning', 'maddpg', 'mappo'],
                    help='Algorithm type')
    
    # Environment arguments
    parser.add_argument('--env', type=str, default='grid_world',
                    choices=['grid_world', 'cooperative_navigation', 'predator_prey'],
                    help='Environment type')
    parser.add_argument('--n-agents', type=int, default=2, help='Number of agents')
    parser.add_argument('--grid-size', type=int, nargs=2, default=[10, 10],
                    help='Grid size for grid world environment')
    
    # Evaluation arguments
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--render', action='store_true', help='Render episodes')
    parser.add_argument('--save-episodes', action='store_true', help='Save episode data')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                    help='Device to use for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating model: {args.model_path}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Environment: {args.env}")
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
        'device': device
    }
    
    algorithm = create_algorithm(args.algorithm, env, algorithm_config)
    print(f"Created algorithm: {algorithm}")
    
    # Load trained model
    try:
        algorithm.load_models(args.model_path)
        print("Successfully loaded trained model")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Evaluation
    print(f"Starting evaluation for {args.episodes} episodes...")
    
    if args.render:
        print("Rendering episodes...")
        # Render a few episodes
        for episode in range(min(5, args.episodes)):
            print(f"Rendering episode {episode + 1}")
            result = EvaluationUtils.evaluate_episode(env, algorithm.agents, args.max_steps, render=True)
            print(f"Episode {episode + 1} - Reward: {result['episode_reward']:.2f}, Length: {result['episode_length']}")
    
    # Full evaluation
    eval_results = EvaluationUtils.evaluate_agents(
        env, algorithm.agents, args.episodes, args.max_steps, render=False
    )
    
    print("Evaluation Results:")
    print(f"Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Average Length: {eval_results['avg_length']:.2f} ± {eval_results['std_length']:.2f}")
    print(f"Success Rate: {eval_results['success_rate']:.2%}")
    
    # Save evaluation results
    results_file = output_dir / 'evaluation_results.json'
    # EvaluationUtils.save_evaluation_results(eval_results, str(results_file))
    print(f"Saved evaluation results to {results_file}")
    
    # Generate plots
    if args.plot:
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Evaluation results plot
        EvaluationUtils.plot_evaluation_results(
            eval_results,
            save_path=str(plots_dir / 'evaluation_results.png')
        )
        
        # Agent performance plot
        TrainingPlots.plot_agent_performance(
            eval_results['agent_rewards'],
            save_path=str(plots_dir / 'agent_performance.png')
        )
        
        print(f"Generated plots in {plots_dir}")
    
    # Create evaluation report
    report = EvaluationUtils.create_evaluation_report(
        eval_results, args.algorithm, args.episodes
    )
    print(report)
    
    # Save report
    report_file = output_dir / 'evaluation_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Evaluation completed! Results saved to {output_dir}")


if __name__ == '__main__':
    main()