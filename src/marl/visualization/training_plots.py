"""
Training visualization plots for multi-agent reinforcement learning.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd


class TrainingPlots:
    """Class for creating training visualization plots."""
    
    @staticmethod
    def plot_learning_curves(
        episode_rewards: List[float],
        episode_lengths: List[float],
        training_metrics: List[Dict[str, float]],
        window_size: int = 100,
        save_path: Optional[str] = None
    ) -> None:
        """Plot learning curves for training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue', label='Raw')
        if len(episode_rewards) >= window_size:
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            axes[0, 0].plot(range(window_size-1, len(episode_rewards)), moving_avg, 
                           color='blue', linewidth=2, label=f'Moving Avg ({window_size})')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[0, 1].plot(episode_lengths, alpha=0.3, color='green', label='Raw')
        if len(episode_lengths) >= window_size:
            moving_avg = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
            axes[0, 1].plot(range(window_size-1, len(episode_lengths)), moving_avg,
                          color='green', linewidth=2, label=f'Moving Avg ({window_size})')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training metrics
        if training_metrics:
            # Extract and plot loss metrics
            loss_data = {}
            for metrics in training_metrics:
                for key, value in metrics.items():
                    if 'loss' in key.lower():
                        if key not in loss_data:
                            loss_data[key] = []
                        loss_data[key].append(value)
            
            for key, values in loss_data.items():
                if values:
                    axes[1, 0].plot(values, label=key, alpha=0.7)
            
            axes[1, 0].set_title('Training Losses')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Reward distribution
        axes[1, 1].hist(episode_rewards, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(episode_rewards):.2f}')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_algorithm_comparison(
        results: Dict[str, Dict[str, List[float]]],
        metric: str = 'episode_rewards',
        window_size: int = 100,
        save_path: Optional[str] = None
    ) -> None:
        """Plot comparison between different algorithms."""
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
        
        for i, (algorithm_name, algorithm_results) in enumerate(results.items()):
            if metric in algorithm_results:
                data = algorithm_results[metric]
                
                # Plot raw data with low alpha
                plt.plot(data, alpha=0.3, color=colors[i])
                
                # Plot moving average
                if len(data) >= window_size:
                    moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
                    plt.plot(range(window_size-1, len(data)), moving_avg, 
                            color=colors[i], linewidth=2, label=algorithm_name)
        
        plt.title(f'Algorithm Comparison - {metric.replace("_", " ").title()}')
        plt.xlabel('Episode')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_agent_performance(
        agent_rewards: Dict[int, List[float]],
        save_path: Optional[str] = None
    ) -> None:
        """Plot individual agent performance."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Individual agent rewards over time
        for agent_id, rewards in agent_rewards.items():
            axes[0].plot(rewards, label=f'Agent {agent_id}', alpha=0.7)
        
        axes[0].set_title('Agent Rewards Over Time')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Agent performance comparison
        agent_stats = []
        agent_labels = []
        
        for agent_id, rewards in agent_rewards.items():
            agent_stats.append({
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards)
            })
            agent_labels.append(f'Agent {agent_id}')
        
        # Create box plot
        data_for_boxplot = [rewards for rewards in agent_rewards.values()]
        axes[1].boxplot(data_for_boxplot, labels=agent_labels)
        axes[1].set_title('Agent Performance Distribution')
        axes[1].set_xlabel('Agent')
        axes[1].set_ylabel('Reward')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_training_metrics(
        training_metrics: List[Dict[str, float]],
        save_path: Optional[str] = None
    ) -> None:
        """Plot detailed training metrics."""
        if not training_metrics:
            print("No training metrics to plot")
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(training_metrics)
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("No numeric metrics to plot")
            return
        
        # Create subplots
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                axes[i].plot(df[col], alpha=0.7)
                axes[i].set_title(col.replace('_', ' ').title())
                axes[i].set_xlabel('Episode')
                axes[i].set_ylabel(col.replace('_', ' ').title())
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_learning_analysis(
        episode_rewards: List[float],
        window_size: int = 100,
        save_path: Optional[str] = None
    ) -> None:
        """Plot detailed learning analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Learning curve with trend
        axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue', label='Raw Rewards')
        
        if len(episode_rewards) >= window_size:
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            axes[0, 0].plot(range(window_size-1, len(episode_rewards)), moving_avg,
                          color='blue', linewidth=2, label=f'Moving Avg ({window_size})')
        
        # Add trend line
        x = np.arange(len(episode_rewards))
        z = np.polyfit(x, episode_rewards, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(x, p(x), "r--", alpha=0.8, label='Trend')
        
        axes[0, 0].set_title('Learning Curve with Trend')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning progress (difference between consecutive moving averages)
        if len(episode_rewards) >= window_size * 2:
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            progress = np.diff(moving_avg)
            axes[0, 1].plot(progress, color='green')
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Learning Progress')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Reward Improvement')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Reward stability (rolling standard deviation)
        if len(episode_rewards) >= window_size:
            rolling_std = []
            for i in range(window_size, len(episode_rewards) + 1):
                rolling_std.append(np.std(episode_rewards[i-window_size:i]))
            
            axes[1, 0].plot(range(window_size, len(episode_rewards) + 1), rolling_std, color='orange')
            axes[1, 0].set_title('Reward Stability (Rolling Std)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Standard Deviation')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Performance milestones
        milestones = [0.1, 0.25, 0.5, 0.75, 0.9]
        milestone_episodes = []
        
        for milestone in milestones:
            target_episode = int(len(episode_rewards) * milestone)
            if target_episode < len(episode_rewards):
                milestone_episodes.append(target_episode)
        
        if milestone_episodes:
            milestone_rewards = [episode_rewards[ep] for ep in milestone_episodes]
            axes[1, 1].scatter(milestone_episodes, milestone_rewards, 
                             c=['red', 'orange', 'yellow', 'lightgreen', 'green'][:len(milestone_episodes)],
                             s=100, alpha=0.7)
            axes[1, 1].plot(milestone_episodes, milestone_rewards, 'k--', alpha=0.5)
            axes[1, 1].set_title('Performance Milestones')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()