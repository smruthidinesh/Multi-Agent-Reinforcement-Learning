
import torch

class CommunicationChannel:
    def __init__(self, n_agents, message_dim):
        self.n_agents = n_agents
        self.message_dim = message_dim
        self.messages = torch.zeros((n_agents, message_dim))

    def send_message(self, agent_id, message):
        self.messages[agent_id] = message

    def get_messages(self, agent_id):
        # Agents receive messages from all other agents
        return torch.cat([self.messages[:agent_id], self.messages[agent_id+1:]])
