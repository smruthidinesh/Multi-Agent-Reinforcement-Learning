import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
from marl.environments.grid_world import MultiAgentGridWorld
from marl.algorithms.qmix import QMix
from marl.agents.dqn_agent import DQNAgent

def main():
    env = MultiAgentGridWorld(n_agents=2, grid_size=(5, 5))
    n_agents = env.n_agents
    obs_dim = int(np.prod(env.observation_space.shape))
    state_dim = obs_dim * n_agents
    action_dim = env.action_space.n

    agents = [DQNAgent(agent_id=i, observation_space=env.observation_space, action_space=env.action_space) for i in range(n_agents)]
    qmix = QMix(env, n_agents, state_dim, action_dim)

    obs, info = env.reset()
    done = False
    step = 0

    while not done:
        actions = [agents[i].get_action(obs[i]) for i in range(n_agents)]
        
        # The following is a simplified placeholder for a single step.
        # A full implementation would require a more complex training loop.
        # This is for demonstration purposes only.
        
        # Get Q-values for the chosen actions
        agent_qs = []
        for i in range(n_agents):
            q_values = agents[i].q_network(torch.FloatTensor(obs[i]).unsqueeze(0))
            agent_qs.append(q_values[0][actions[i]])
        agent_qs = torch.stack(agent_qs)

        # Simulate a step in the environment
        next_obs, rewards, terminated, truncated, info = env.step({i: actions[i] for i in range(n_agents)})
        done = all(terminated.values())

        # Train the mixer
        states = torch.FloatTensor(np.array(list(obs.values()))).view(1, -1)
        rewards_tensor = torch.FloatTensor(list(rewards.values())).unsqueeze(1)
        loss = qmix.train(agent_qs.unsqueeze(0), states, rewards_tensor)

        obs = next_obs
        step += 1
        print(f"Step: {step}, Loss: {loss}")

if __name__ == "__main__":
    main()
