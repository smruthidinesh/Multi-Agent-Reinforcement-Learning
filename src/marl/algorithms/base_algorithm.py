
from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    def __init__(self, env, n_agents, learning_rate, gamma):
        self.env = env
        self.n_agents = n_agents
        self.learning_rate = learning_rate
        self.gamma = gamma

    @abstractmethod
    def train(self, n_episodes):
        pass
