import sys
sys.path.append("src")
from marl.environments.grid_world import MultiAgentGridWorld
from marl.algorithms.hierarchical_q_learning import HierarchicalQLearning

def main():
    env = MultiAgentGridWorld(n_agents=2, grid_size=(5, 5), n_goals=2)
    hql = HierarchicalQLearning(env, n_agents=2)
    hql.train(n_episodes=100)

if __name__ == "__main__":
    main()