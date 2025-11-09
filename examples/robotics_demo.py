import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import time
import numpy as np
from marl.environments.robotics import MultiAgentRobotNavigation

def main():
    env = MultiAgentRobotNavigation(
        n_agents=2,
        grid_size=10,
        n_targets=2,
        n_obstacles=3,
        max_steps=100,
        render_mode="human" # Set to "human" to visualize, or None for no rendering
    )

    obs, info = env.reset()
    done = False
    episode_reward = 0
    step_count = 0

    print("Starting Robotics Environment Demo...")

    while not done and step_count < env.max_steps:
        actions = {i: env.action_space[f"agent_{i}"].sample() for i in range(env.n_agents)}
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        current_rewards = sum(rewards.values())
        episode_reward += current_rewards
        
        done = all(terminated.values()) or all(truncated.values())
        step_count += 1

        if env.render_mode == "human":
            time.sleep(0.1) # Small delay for better visualization

    print(f"Demo Finished. Total Episode Reward: {episode_reward:.2f}, Steps: {step_count}")
    env.close()

if __name__ == "__main__":
    main()