"""
Evaluation utilities for multi-agent reinforcement learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd


class EvaluationUtils:
    """Utility class for evaluating multi-agent RL systems."""
    
    @staticmethod
    def evaluate_episode(
        env,
        agents: Dict[int, Any],
        max_steps: int = 100,
        render: bool = False
    ) -> Dict[str, Any]:
        """Evaluate a single episode."""
        episode_reward = 0
        episode_length = 0
        agent_rewards = {agent_id: 0 for agent_id in agents.keys()}
        
        # Reset environment
        observations, _ = env.reset()
        
        for step in range(max_steps):
            # Select actions
            actions = {}
            for agent_id, obs in observations.items():
                if hasattr(agents[agent_id], 'select_action'):
                    action = agents[agent_id].select_action(obs, training=False)
                else:
                    action = agents[agent_id](obs)
                actions[agent_id] = action
            
            # Execute actions
            next_observations, rewards, terminated, truncated, _ = env.step(actions)
            
            # Update statistics
            episode_reward += sum(rewards.values())
            episode_length += 1
            
            for agent_id, reward in rewards.items():
                agent_rewards[agent_id] += reward
            
            # Render if requested
            if render:
                env.render()
            
            # Check termination
            if all(terminated.values()) or all(truncated.values()):
                break
            
            observations = next_observations
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'agent_rewards': agent_rewards,
            'success': all(terminated.values())
        }
    
    @staticmethod
    def evaluate_agents(
        env,
        agents: Dict[int, Any],
        n_episodes: int = 100,
        max_steps: int = 100,
        render: bool = False
    ) -> Dict[str, Any]:
        """Evaluate agents over multiple episodes."""
        episode_rewards = []
        episode_lengths = []
        agent_rewards = {agent_id: [] for agent_id in agents.keys()}
        success_rate = 0
        
        for episode in range(n_episodes):
            result = EvaluationUtils.evaluate_episode(env, agents, max_steps, render)
            
            episode_rewards.append(result['episode_reward'])
            episode_lengths.append(result['episode_length'])
            
            for agent_id, reward in result['agent_rewards'].items():
                agent_rewards[agent_id].append(reward)
            
            if result['success']:
                success_rate += 1
        
        success_rate /= n_episodes
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'agent_rewards': agent_rewards,
            'success_rate': success_rate,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }
    
    @staticmethod
    def plot_evaluation_results(
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """Plot evaluation results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].hist(results['episode_rewards'], bins=30, alpha=0.7, color='blue')
        axes[0, 0].axvline(results['avg_reward'], color='red', linestyle='--', 
                          label=f'Mean: {results["avg_reward"]:.2f}')
        axes[0, 0].set_title('Episode Rewards Distribution')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[0, 1].hist(results['episode_lengths'], bins=30, alpha=0.7, color='green')
        axes[0, 1].axvline(results['avg_length'], color='red', linestyle='--',
                          label=f'Mean: {results["avg_length"]:.2f}')
        axes[0, 1].set_title('Episode Lengths Distribution')
        axes[0, 1].set_xlabel('Length')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Agent rewards comparison
        agent_data = []
        agent_labels = []
        for agent_id, rewards in results['agent_rewards'].items():
            agent_data.append(rewards)
            agent_labels.append(f'Agent {agent_id}')
        
        axes[1, 0].boxplot(agent_data, labels=agent_labels)
        axes[1, 0].set_title('Agent Rewards Comparison')
        axes[1, 0].set_xlabel('Agent')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Success rate
        axes[1, 1].bar(['Success', 'Failure'], 
                      [results['success_rate'], 1 - results['success_rate']],
                      color=['green', 'red'], alpha=0.7)
        axes[1, 1].set_title(f'Success Rate: {results["success_rate"]:.2%}')
        axes[1, 1].set_ylabel('Proportion')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def compare_agent_performance(
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> None:
        """Compare performance across different agent configurations."""
        algorithms = list(results.keys())
        avg_rewards = [results[alg]['avg_reward'] for alg in algorithms]
        success_rates = [results[alg]['success_rate'] for alg in algorithms]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Average rewards comparison
        bars1 = axes[0].bar(algorithms, avg_rewards, alpha=0.7, color='blue')
        axes[0].set_title('Average Episode Rewards')
        axes[0].set_ylabel('Reward')
        axes[0].set_xlabel('Algorithm')
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, avg_rewards):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
        
        # Success rates comparison
        bars2 = axes[1].bar(algorithms, success_rates, alpha=0.7, color='green')
        axes[1].set_title('Success Rates')
        axes[1].set_ylabel('Success Rate')
        axes[1].set_xlabel('Algorithm')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, success_rates):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def create_evaluation_report(
        results: Dict[str, Any],
        algorithm_name: str,
        n_episodes: int
    ) -> str:
        """Create a detailed evaluation report."""
        report = f"""
Evaluation Report for {algorithm_name}
=====================================

Evaluation Episodes: {n_episodes}
Average Episode Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}
Average Episode Length: {results['avg_length']:.2f} ± {results['std_length']:.2f}
Success Rate: {results['success_rate']:.2%}

Agent Performance:
"""
        
        for agent_id, rewards in results['agent_rewards'].items():
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            report += f"  Agent {agent_id}: {avg_reward:.2f} ± {std_reward:.2f}\n"
        
        report += f"""
Best Episode Reward: {max(results['episode_rewards']):.2f}
Worst Episode Reward: {min(results['episode_rewards']):.2f}
Longest Episode: {max(results['episode_lengths'])}
Shortest Episode: {min(results['episode_lengths'])}
        """
        
        return report.strip()
    
    @staticmethod
    def analyze_agent_cooperation(
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """Analyze cooperation patterns between agents."""
        agent_rewards = results['agent_rewards']
        n_agents = len(agent_rewards)
        
        # Calculate correlation matrix
        reward_matrix = np.array([agent_rewards[agent_id] for agent_id in range(n_agents)])
        correlation_matrix = np.corrcoef(reward_matrix)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   xticklabels=[f'Agent {i}' for i in range(n_agents)],
                   yticklabels=[f'Agent {i}' for i in range(n_agents)])
        plt.title('Agent Reward Correlation Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()