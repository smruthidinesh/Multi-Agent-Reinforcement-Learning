#!/usr/bin/env python3
"""
Environment demonstration script for multi-agent reinforcement learning.

This script demonstrates different multi-agent environments and their
characteristics.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from marl.environments import MultiAgentGridWorld, CooperativeNavigation, PredatorPrey


def demo_grid_world():
    """Demonstrate the grid world environment."""
    print("Grid World Environment Demo")
    print("=" * 30)
    
    env = MultiAgentGridWorld(
        grid_size=(8, 8),
        n_agents=2,
        n_targets=3,
        max_steps=50,
        render_mode="human"
    )
    
    print(f"Environment: {env}")
    print(f"Grid size: {env.grid_size}")
    print(f"Number of agents: {env.n_agents}")
    print(f"Number of targets: {env.n_targets}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run a few episodes
    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        observations, _ = env.reset()
        total_reward = 0
        
        for step in range(20):
            # Random actions
            actions = {}
            for agent_id in range(env.n_agents):
                actions[agent_id] = env.action_space.sample()
            
            next_observations, rewards, terminated, truncated, _ = env.step(actions)
            total_reward += sum(rewards.values())
            
            # Render
            env.render()
            
            if all(terminated.values()) or all(truncated.values()):
                break
            
            observations = next_observations
        
        print(f"Episode {episode + 1} completed with reward: {total_reward:.2f}")
    
    env.close()


def demo_cooperative_navigation():
    """Demonstrate the cooperative navigation environment."""
    print("\nCooperative Navigation Environment Demo")
    print("=" * 40)
    
    env = CooperativeNavigation(
        n_agents=3,
        n_landmarks=3,
        world_size=2.0,
        max_steps=50,
        render_mode="human"
    )
    
    print(f"Environment: {env}")
    print(f"Number of agents: {env.n_agents}")
    print(f"Number of landmarks: {env.n_landmarks}")
    print(f"World size: {env.world_size}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run a few episodes
    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        observations, _ = env.reset()
        total_reward = 0
        
        for step in range(20):
            # Random actions
            actions = {}
            for agent_id in range(env.n_agents):
                actions[agent_id] = env.action_space.sample()
            
            next_observations, rewards, terminated, truncated, _ = env.step(actions)
            total_reward += sum(rewards.values())
            
            # Render
            env.render()
            
            if all(terminated.values()) or all(truncated.values()):
                break
            
            observations = next_observations
        
        print(f"Episode {episode + 1} completed with reward: {total_reward:.2f}")
    
    env.close()


def demo_predator_prey():
    """Demonstrate the predator-prey environment."""
    print("\nPredator-Prey Environment Demo")
    print("=" * 35)
    
    env = PredatorPrey(
        n_predators=2,
        n_prey=1,
        world_size=2.0,
        max_steps=50,
        render_mode="human"
    )
    
    print(f"Environment: {env}")
    print(f"Number of predators: {env.n_predators}")
    print(f"Number of prey: {env.n_prey}")
    print(f"World size: {env.world_size}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run a few episodes
    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        observations, _ = env.reset()
        total_reward = 0
        
        for step in range(20):
            # Random actions
            actions = {}
            for agent_id in range(env.n_predators + env.n_prey):
                actions[agent_id] = env.action_space.sample()
            
            next_observations, rewards, terminated, truncated, _ = env.step(actions)
            total_reward += sum(rewards.values())
            
            # Render
            env.render()
            
            if all(terminated.values()) or all(truncated.values()):
                break
            
            observations = next_observations
        
        print(f"Episode {episode + 1} completed with reward: {total_reward:.2f}")
    
    env.close()


def main():
    print("Multi-Agent RL Environment Demonstration")
    print("=" * 45)
    
    # Demo each environment
    demo_grid_world()
    demo_cooperative_navigation()
    demo_predator_prey()
    
    print("\nEnvironment demonstration completed successfully!")
    print("\nEnvironment Summary:")
    print("1. Grid World: Discrete grid-based environment with targets to collect")
    print("2. Cooperative Navigation: Continuous environment with landmarks to reach")
    print("3. Predator-Prey: Competitive environment with predators and prey")


if __name__ == '__main__':
    main()