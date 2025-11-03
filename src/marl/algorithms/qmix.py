
import torch
import torch.nn as nn
import torch.optim as optim

class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, mixing_embed_dim):
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim

        self.hyper_w1 = nn.Linear(self.state_dim, self.mixing_embed_dim * self.n_agents)
        self.hyper_b1 = nn.Linear(self.state_dim, self.mixing_embed_dim)
        self.hyper_w2 = nn.Linear(self.state_dim, self.mixing_embed_dim)
        self.hyper_b2 = nn.Linear(self.state_dim, 1)

    def forward(self, agent_qs, states):
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.n_agents, self.mixing_embed_dim)
        b1 = b1.view(-1, 1, self.mixing_embed_dim)

        hidden = torch.relu(torch.bmm(agent_qs, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        w2 = w2.view(-1, self.mixing_embed_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(-1, 1)
        return q_total

class QMix:
    def __init__(self, env, n_agents, state_dim, action_dim, learning_rate=0.001, gamma=0.99, mixing_embed_dim=32):
        self.env = env
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.mixer = QMixer(n_agents, state_dim, mixing_embed_dim)
        self.optimizer = optim.Adam(self.mixer.parameters(), lr=self.learning_rate)

    def train(self, agent_qs, states, rewards):
        q_total = self.mixer(agent_qs, states)
        
        # This is a simplified version of the loss calculation
        # A full implementation would require a target network and more complex reward handling
        loss = torch.mean((rewards - q_total) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
