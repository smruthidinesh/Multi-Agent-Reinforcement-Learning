"""
Training utilities for multi-agent reinforcement learning.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime


class TrainingUtils:
    """Utility class for training multi-agent RL systems."""
    
    @staticmethod
    def moving_average(data: List[float], window_size: int = 100) -> List[float]:
        """Calculate moving average of data."""
        if len(data) < window_size:
            return data
        
        moving_avg = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(np.mean(data[start_idx:i + 1]))
        
        return moving_avg
    
    @staticmethod
    def plot_training_progress(
        episode_rewards: List[float],
        episode_lengths: List[float],
        training_metrics: List[Dict[str, float]],
        save_path: Optional[str] = None
    ) -> None:
        """Plot training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue')
        moving_avg_rewards = TrainingUtils.moving_average(episode_rewards)
        axes[0, 0].plot(moving_avg_rewards, color='blue', linewidth=2)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[0, 1].plot(episode_lengths, alpha=0.3, color='green')
        moving_avg_lengths = TrainingUtils.moving_average(episode_lengths)
        axes[0, 1].plot(moving_avg_lengths, color='green', linewidth=2)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training metrics
        if training_metrics:
            # Extract loss metrics
            loss_metrics = {}
            for metrics in training_metrics:
                for key, value in metrics.items():
                    if 'loss' in key.lower():
                        if key not in loss_metrics:
                            loss_metrics[key] = []
                        loss_metrics[key].append(value)
            
            # Plot loss metrics
            for key, values in loss_metrics.items():
                if values:
                    moving_avg = TrainingUtils.moving_average(values)
                    axes[1, 0].plot(moving_avg, label=key, alpha=0.7)
            
            axes[1, 0].set_title('Training Losses')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Reward distribution
        axes[1, 1].hist(episode_rewards, bins=50, alpha=0.7, color='purple')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def save_training_results(
        results: Dict[str, Any],
        filepath: str
    ) -> None:
        """Save training results to file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                serializable_results[key] = [v.tolist() for v in value]
            else:
                serializable_results[key] = value
        
        # Add metadata
        serializable_results['timestamp'] = datetime.now().isoformat()
        serializable_results['pytorch_version'] = torch.__version__
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    @staticmethod
    def load_training_results(filepath: str) -> Dict[str, Any]:
        """Load training results from file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def compare_algorithms(
        results: Dict[str, Dict[str, List[float]]],
        metric: str = 'episode_rewards',
        window_size: int = 100,
        save_path: Optional[str] = None
    ) -> None:
        """Compare training results across different algorithms."""
        plt.figure(figsize=(12, 8))
        
        for algorithm_name, algorithm_results in results.items():
            if metric in algorithm_results:
                data = algorithm_results[metric]
                moving_avg = TrainingUtils.moving_average(data, window_size)
                plt.plot(moving_avg, label=algorithm_name, linewidth=2)
        
        plt.title(f'Algorithm Comparison - {metric.replace("_", " ").title()}')
        plt.xlabel('Episode')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def create_training_summary(
        results: Dict[str, Any],
        algorithm_name: str,
        n_episodes: int
    ) -> str:
        """Create a summary of training results."""
        summary = f"""
Training Summary for {algorithm_name}
=====================================

Total Episodes: {n_episodes}
Final Average Reward: {np.mean(results['episode_rewards'][-100:]):.2f}
Final Average Length: {np.mean(results['episode_lengths'][-100:]):.2f}

Best Episode Reward: {max(results['episode_rewards']):.2f}
Worst Episode Reward: {min(results['episode_rewards']):.2f}

Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return summary.strip()
    
    @staticmethod
    def early_stopping(
        episode_rewards: List[float],
        patience: int = 100,
        min_improvement: float = 0.01
    ) -> bool:
        """Check if training should stop early."""
        if len(episode_rewards) < patience:
            return False
        
        recent_rewards = episode_rewards[-patience:]
        best_reward = max(recent_rewards)
        current_reward = recent_rewards[-1]
        
        return (best_reward - current_reward) < min_improvement