import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
from marl.environments.traffic import TrafficEnv

def main():
    env = TrafficEnv(n_agents=2, grid_size=10)
    obs, info = env.reset()
    done = False
    step = 0

    while not done:
        env.render()
        actions = [env.action_space.sample() for _ in range(env.n_agents)]
        obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        step += 1
        print(f"Step: {step}, Actions: {actions}, Rewards: {rewards}")

if __name__ == "__main__":
    main()
