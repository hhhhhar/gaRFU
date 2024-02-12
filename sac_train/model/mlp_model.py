import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class policy_mlp_gaussian(nn.Module):
    def __init__(self, state, num_actions, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(policy_mlp_gaussian,self).__init__()
        self.state = state
        self.num_actions = num_actions
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.fc1 = nn.Linear(state, 256)
        self.fc2 = nn.Linear(256, 128)

        self.mu_head = nn.Linear(128, num_actions)
        self.log_std_head = nn.Linear(128, num_actions)

        self.log_std_head.weight.data.uniform_(-edge, edge)
        self.log_std_head.bias.data.uniform_(-edge, edge)
        self.mu_head.weight.data.uniform_(-edge, edge)
        self.mu_head.bias.data.uniform_(-edge, edge)

    def forward(self, x):
        # pdb.set_trace()
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std


class q_network(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(q_network, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.fc3.weight.data.uniform_(-edge, edge)
        self.fc3.bias.data.uniform_(-edge, edge)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DoubleQCritic_mlp(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim):
        super(DoubleQCritic_mlp, self).__init__()
        self.Q1 = q_network(obs_dim, action_dim)
        self.Q2 = q_network(obs_dim, action_dim)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)
        return q1, q2

