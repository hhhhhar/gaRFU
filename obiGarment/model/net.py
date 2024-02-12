import sys
sys.path.append("..")
from typing import Any
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from torch.distributions import Normal

from .mlp import MLP_V2
from util.pool import Pool
from dataset.RLDataset import RLDataset
import random


class ModelAction(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hidden = MLP_V2([10, 128, 64], norm_type='layer', transpose_input=True)
        self.mu = torch.nn.Linear(64, 3)
        self.std = torch.nn.Linear(64, 3)

    def forward(self, obs):
        
        hidden = self.hidden(obs)
        mu = self.mu(hidden)
        std = F.softplus(self.std(hidden))
        dist = Normal(mu, std)
        normalSample = dist.rsample()
        log_prob = dist.log_prob(normalSample)
        action = torch.tanh(normalSample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        return action, log_prob



class SACNet(pl.LightningModule):
    def __init__(self, actor_lr, critic_lr, alpha_lr, \
                 target_entropy, tau, gamma, env=None) -> None:
        super(SACNet, self).__init__()
        self.dataPool = Pool()
        self.actor = ModelAction()
        self.critic_1 = MLP_V2([13, 128, 64, 3], norm_type='layer', transpose_input=True)
        self.target_critic_1 = MLP_V2([13, 128, 64, 3], norm_type='layer', transpose_input=True)
        self.critic_2 = MLP_V2([13, 128, 64, 3], norm_type='layer', transpose_input=True)
        self.target_critic_2 = MLP_V2([13, 128, 64, 3], norm_type='layer', transpose_input=True)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度

        self.env = env
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.automatic_optimization = False

    def setEnv(self, env):
        self.env = env

    #更新动作池
    def update(self):
        #每次更新不少于N条新数据
        old_len = len(self.dataPool.pool)
        while len(self.dataPool.pool) - old_len < 4000:
            self.dataPool.pool.extend(self.play()[0])

        #只保留最新的N条数据
        self.dataPool.pool = self.dataPool.pool[-2_0000:]

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).reshape([1, 1, -1])
        action = self.actor(state)[0]
        return action

    def on_train_epoch_start(self):
        self.update()

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        
        q1_value = self.target_critic_1(
            torch.cat([next_states, next_actions], dim=-1))
        q2_value = self.target_critic_2(
            torch.cat([next_states, next_actions], dim=-1))
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def train_dataloader(self):
        dataset = RLDataset(self.dataPool, self.device, sample_size=1)
        dataloader = DataLoader(dataset=dataset, batch_size=1)

        return dataloader

    def play(self):
        data = []
        reward_sum = 0

        obs_dict = self.env.reset()
        obs = torch.from_numpy(obs_dict["observation"]).to(self.device)
        done = False
        while not done:
            action = self.take_action(obs)
            nextObs, reward, done, info = self.env.step(action)

            data.append((obs, action, reward, nextObs["observation"], done))
            reward_sum += reward

            obs = torch.from_numpy(nextObs["observation"]).to(self.device)

        return data, reward_sum

    def configure_optimizers(self):
        actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.actor_lr)
        critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=self.critic_lr)
        critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=self.critic_lr)
        log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=self.alpha_lr)
        return actor_optimizer, critic_1_optimizer, \
                critic_2_optimizer, log_alpha_optimizer

    def training_step(self, batch, batch_idx):
        # TODO: retain_graph=True
        actor_opt, critic_1_opt, critic_2_opt, \
            log_alpha_opt = self.optimizers()
        
        print(batch)

        _obs = batch[0]
        _actions = batch[1]
        rewards = batch[2]
        next_states = batch[3]
        dones = batch[4]
        # 对奖励进行重塑以便训练
        rewards = (rewards + 8.0) / 8.0

        for _ in range(4000):
            # 更新策略网络
            obs = _obs.clone().detach().unsqueeze_(0).requires_grad_()
            new_actions, log_prob = self.actor(obs)
            entropy = -log_prob
            self.critic_1.eval()
            self.critic_2.eval()
            with torch.no_grad():
                q1_value = self.critic_1(torch.cat([obs, new_actions], dim=-1))
                q2_value = self.critic_2(torch.cat([obs, new_actions], dim=-1))
            actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                    torch.min(q1_value, q2_value))
            actor_opt.zero_grad()
            self.manual_backward(actor_loss)
            actor_opt.step()
            self.actor.requires_grad_(False)
            
            # 更新评估网络
            obs = _obs.clone().detach().requires_grad_()
            actions = _actions.clone().detach().requires_grad_()
            td_target = self.calc_target(rewards, next_states, dones)

            critic_1_loss = torch.mean(
                F.mse_loss(self.critic_1(torch.cat(
                    [obs, actions], dim=-1).unsqueeze_(0)), td_target.clone().detach()))
            critic_1_opt.zero_grad()
            self.manual_backward(critic_1_loss)
            critic_1_opt.step()
            
            obs = _obs.clone().detach().requires_grad_()
            actions = _actions.clone().detach().requires_grad_()
            critic_2_loss = torch.mean(
                F.mse_loss(self.critic_2(torch.cat(
                    [obs, actions], dim=-1).unsqueeze_(0)), td_target.clone().detach()))
            critic_2_opt.zero_grad()
            self.manual_backward(critic_2_loss)
            critic_2_opt.step()

            # 更新alpha
            alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
            log_alpha_opt.zero_grad()
            self.manual_backward(alpha_loss)
            log_alpha_opt.step()

            self.actor.requires_grad_()
            self.soft_update(self.critic_1, self.target_critic_1)
            self.soft_update(self.critic_2, self.target_critic_2)