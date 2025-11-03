#!/usr/bin/env python3
"""
Basic tests for the multi-agent RL framework.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import unittest
import numpy as np
from marl.environments import MultiAgentGridWorld
from marl.algorithms import IndependentQLearning
from marl.agents import DQNAgent


class TestMultiAgentRL(unittest.TestCase):
    """Test cases for multi-agent RL framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MultiAgentGridWorld(
            grid_size=(5, 5),
            n_agents=2,
            n_targets=2,
            max_steps=20
        )
        
        self.algorithm = IndependentQLearning(
            env=self.env,
            n_agents=2,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            buffer_size=1000,
            batch_size=16,
            target_update_freq=10,
            device="cpu"
        )
    
    def test_environment_creation(self):
        """Test environment creation."""
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.n_agents, 2)
        self.assertEqual(self.env.n_targets, 2)
        self.assertEqual(self.env.grid_size, (5, 5))
    
    def test_environment_reset(self):
        """Test environment reset."""
        observations, info = self.env.reset()
        self.assertIsInstance(observations, dict)
        self.assertEqual(len(observations), 2)  # 2 agents
        self.assertIsInstance(info, dict)
    
    def test_environment_step(self):
        """Test environment step."""
        observations, _ = self.env.reset()
        
        # Random actions
        actions = {0: 0, 1: 1}
        next_observations, rewards, terminated, truncated, info = self.env.step(actions)
        
        self.assertIsInstance(next_observations, dict)
        self.assertIsInstance(rewards, dict)
        self.assertIsInstance(terminated, dict)
        self.assertIsInstance(truncated, dict)
        self.assertIsInstance(info, dict)
        
        self.assertEqual(len(next_observations), 2)
        self.assertEqual(len(rewards), 2)
        self.assertEqual(len(terminated), 2)
        self.assertEqual(len(truncated), 2)
    
    def test_algorithm_creation(self):
        """Test algorithm creation."""
        self.assertIsNotNone(self.algorithm)
        self.assertEqual(len(self.algorithm.agents), 2)
    
    def test_agent_action_selection(self):
        """Test agent action selection."""
        agent = self.algorithm.agents[0]
        observation = np.random.random(agent.observation_space.shape[0])
        
        action = agent.get_action(observation, training=True)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, agent.action_space.n)
    
    def test_short_training(self):
        """Test short training run."""
        # Train for just a few episodes
        results = self.algorithm.train(n_episodes=5, max_steps_per_episode=10)
        
        self.assertIsInstance(results, dict)
        self.assertIn('episode_rewards', results)
        self.assertIn('episode_lengths', results)
        self.assertIn('training_metrics', results)
        
        self.assertEqual(len(results['episode_rewards']), 5)
        self.assertEqual(len(results['episode_lengths']), 5)
    
    def test_evaluation(self):
        """Test evaluation."""
        # Train briefly first
        self.algorithm.train(n_episodes=3, max_steps_per_episode=10)
        
        # Evaluate
        eval_results = self.algorithm.evaluate(n_episodes=2, max_steps_per_episode=10)
        
        self.assertIsInstance(eval_results, dict)
        self.assertIn('avg_reward', eval_results)
        self.assertIn('avg_length', eval_results)
        self.assertIn('success_rate', eval_results)
    
    def test_agent_save_load(self):
        """Test agent save and load."""
        agent = self.algorithm.agents[0]
        
        # Save agent
        agent.save('test_agent.pth')
        self.assertTrue(os.path.exists('test_agent.pth'))
        
        # Load agent
        agent.load('test_agent.pth')
        
        # Clean up
        os.remove('test_agent.pth')
    
    def test_algorithm_save_load(self):
        """Test algorithm save and load."""
        # Train briefly
        self.algorithm.train(n_episodes=2, max_steps_per_episode=5)
        
        # Save algorithm
        self.algorithm.save_models('test_models/')
        self.assertTrue(os.path.exists('test_models/test_models_agent_0.pth'))
        self.assertTrue(os.path.exists('test_models/test_models_agent_1.pth'))
        
        # Load algorithm
        self.algorithm.load_models('test_models/')
        
        # Clean up
        os.remove('test_models/test_models_agent_0.pth')
        os.remove('test_models/test_models_agent_1.pth')
        os.rmdir('test_models')
    
    def test_observation_space(self):
        """Test observation space."""
        observations, _ = self.env.reset()
        
        for agent_id, obs in observations.items():
            self.assertEqual(obs.shape, self.env.observation_space.shape)
            self.assertEqual(obs.dtype, np.float32)
    
    def test_action_space(self):
        """Test action space."""
        self.assertEqual(self.env.action_space.n, 5)  # 5 actions: up, down, left, right, stay
    
    def tearDown(self):
        """Clean up after tests."""
        self.env.close()


if __name__ == '__main__':
    unittest.main()