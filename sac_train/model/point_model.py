import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetfeat(nn.Module):
    def __init__(self, feat_dim):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(feat_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x


class PointNetFusion(nn.Module):

    def __init__(self, feat_dim):
        super(PointNetFusion, self).__init__()
        self.feat1 = PointNetfeat(feat_dim)
        self.feat2 = PointNetfeat(feat_dim)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.dp1 = nn.Dropout(p=0.3)
        self.dp2 = nn.Dropout(p=0.3)

    def forward(self, x1, x2):
        x1 = self.feat1(x1)
        x2 = self.feat2(x2)
        x = torch.cat([x1,x2], dim=-1)
        x = self.dp1(F.relu(self.fc1(x)))
        x = self.dp2(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return x


class policy_point_gaussian(nn.Module):
    def __init__(self, state_dim, state_robot_dim, action_dim, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(policy_point_gaussian, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.feat = PointNetFusion(state_dim)
        self.robot_ob1 = nn.Linear(state_robot_dim, 128)
        self.robot_ob2 = nn.Linear(128, 256)
        self.linear1 = nn.Linear(512 + 256, 256)
        self.linear2 = nn.Linear(256, 256)
        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)

        # self.mean_linear.weight.data.uniform_(-edge, edge)
        # self.mean_linear.bias.data.uniform_(-edge, edge)
        self.log_std_linear.weight.data.uniform_(-edge, edge)
        self.log_std_linear.bias.data.uniform_(-edge, edge)

    def forward(self, global_state, state, state_robot):
        x = self.feat(global_state, state)
        state_robot = self.robot_ob1(state_robot)
        state_robot = self.robot_ob2(state_robot)
        x = torch.cat([x, state_robot], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


# Soft Q Net
class q_network(nn.Module):
    def __init__(self, state_ob_dim, state_robot_dim, action_dim, edge=3e-3):
        super(q_network, self).__init__()
        self.feat = PointNetFusion(state_ob_dim)
        self.robot_ob1 = nn.Linear(state_robot_dim, 128)
        self.robot_ob2 = nn.Linear(128, 256)
        self.a1 = nn.Linear(action_dim, 128)
        self.a2 = nn.Linear(128, 256)
        self.linear1 = nn.Linear(512 + 256 + 256, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)
        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)

    def forward(self, global_state, state, state_robot, action):
        x = self.feat(global_state, state)
        action = self.a1(action)
        action = self.a2(action)
        state_robot = self.robot_ob1(state_robot)
        state_robot = self.robot_ob2(state_robot)
        x = torch.cat([x, action], 1)
        x = torch.cat([x, state_robot], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        # x = x.squeeze(0)
        return x


class DoubleQCritic_point(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, state_ob_dim, state_robot_dim, action_dim):
        super(DoubleQCritic_point, self).__init__()
        self.Q1 = q_network(state_ob_dim, state_robot_dim, action_dim)
        self.Q2 = q_network(state_ob_dim, state_robot_dim, action_dim)

    def forward(self, global_obs, obs, robot, action):
        assert obs.size(0) == action.size(0)
        q1 = self.Q1(global_obs, obs, robot, action)
        q2 = self.Q2(global_obs, obs, robot, action)
        return q1, q2

