import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model import policy_mlp_gaussian, DoubleQCritic_mlp
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from torch.optim.lr_scheduler import ExponentialLR
import pdb
import time
import random

class sac_agent(object):
    def __init__(self, arg, replay_buffer, device):
        self.args = arg
        self.replay_buffer = replay_buffer
        self.device = device
        # initialize networks
        self.state_dim = arg.state_dim
        self.action_dim = arg.action_dim
        self.critic = DoubleQCritic_mlp(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = DoubleQCritic_mlp(self.state_dim, self.action_dim).to(self.device)
        self.actor = policy_mlp_gaussian(self.state_dim, self.action_dim).to(self.device)
        self.memory = replay_buffer
        self.lr = arg.lr
        self.tau = arg.tau
        self.gamma = arg.gamma
        self.learn_step = 0
        self.critic_target_update_frequency = 2
        self.actor_update_frequency = 2
        self.learn_frequency = arg.learn_frequency
        self.log_alpha = torch.FloatTensor([0.0])
        self.log_alpha.requires_grad = True
        self.target_entropy = -self.action_dim
        self.total_learn_num = int((arg.num_train_step)*1.05/self.learn_frequency)

        # Load the target value network parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        self.optimizers = []
        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr,
                                                betas=[0.9, 0.999])

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.lr,
                                                 betas=[0.9, 0.999])

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=self.lr,
                                                    betas=[0.9, 0.999])
        self.optimizers.append(self.actor_optimizer)
        self.optimizers.append(self.critic_optimizer)
        self.optimizers.append(self.log_alpha_optimizer)
        self.critic_gradient_clipping_norm = 5
        self.actor_gradient_clipping_norm = 5
        folder_name = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        result_folder = os.path.join(arg.result_fold, folder_name)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        file_name = "optimize_sac"
        self.writer = SummaryWriter(log_dir=result_folder, comment=file_name)
        self.model_fold = os.path.join(arg.result_fold, "model_save")
        if not os.path.exists(self.model_fold):
            os.makedirs(self.model_fold)
        self.setup_seed(self.args.seed)


    def update_learning_rate(self):
        # change lr
        self._current_progress_remaining = 1.0 - float(self.learn_step) / float(self.total_learn_num)
        learning_rate = self.lr * self._current_progress_remaining
        self.writer.add_scalar('learning_rate/lr', torch.tensor(learning_rate), self.learn_step)
        for optimizer in self.optimizers:
            self._update_optimizer_learning_rate(optimizer, learning_rate)

    def _update_optimizer_learning_rate(self, optimizer, learning_rate):
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def action(self, obs, sample=True):
        state = torch.FloatTensor(obs).to(self.device)
        state = state.unsqueeze(0)
        mean, log_std = self.actor(state)
        if sample:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()  # rsample means it is sampled using reparameterisation trick
            action = torch.tanh(x_t).detach().cpu().numpy().flatten()
        else:
            action = torch.tanh(mean).detach().cpu().numpy().flatten()
        return action

    def evaluate(self, obs, device, epsilon=1e-6):
        self.actor.to(device)
        mean, log_std = self.actor(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)
        z = noise.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = normal.log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def write_train_process(self, reward, step, success=None):
        self.writer.add_scalar('train_reward/reward', torch.tensor(reward), step)
        if success:
            self.writer.add_scalar('train_reward/successs', torch.tensor(success), step)

    def write_evel_process(self, reward, step, success=None):
        self.writer.add_scalar('evel_reward/reward', torch.tensor(reward), step)
        if success:
            self.writer.add_scalar('evel_reward/success', torch.tensor(success), step)


    def update_critic_concate(self, batch_sample):
        self.alpha = self.log_alpha.exp().to(self.device)
        state = []
        action = []
        mask = []
        reward = []
        next_state = []
        for transition in batch_sample:
            state.append(transition.state)
            action.append(transition.action)
            mask.append(transition.mask)
            reward.append(transition.reward)
            next_state.append(transition.next_state)

        state_batch = torch.FloatTensor(state).to(self.device)
        action_batch = torch.FloatTensor(action).to(self.device)
        reward_batch = torch.FloatTensor(reward).to(self.device)
        mask_batch = torch.FloatTensor(mask).to(self.device)
        next_state_batch = torch.FloatTensor(next_state).to(self.device)
        mask_batch = mask_batch.unsqueeze(1)
        reward_batch = reward_batch.unsqueeze(1)
        # pdb.set_trace()
        next_action, log_prob = self.evaluate(next_state_batch, self.device)
        target_Q1, target_Q2 = self.critic_target(next_state_batch, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha * log_prob
        target_Q = reward_batch + (mask_batch * self.gamma * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(state_batch, action_batch)
        critic_loss1 = F.mse_loss(current_Q1, target_Q)
        critic_loss2 = F.mse_loss(current_Q2, target_Q)
        critic_loss = critic_loss1 + critic_loss2
        self.writer.add_scalar('train_critic/target_Q', target_Q.mean(), self.learn_step)
        self.writer.add_scalar('train_critic/current_Q1', current_Q1.mean(), self.learn_step)
        self.writer.add_scalar('train_critic/current_Q2', current_Q2.mean(), self.learn_step)
        self.writer.add_scalar('train_critic/critic_loss', critic_loss, self.learn_step)
        self.writer.add_scalar('train_critic/critic_loss1', critic_loss1, self.learn_step)
        self.writer.add_scalar('train_critic/critic_loss2', critic_loss2, self.learn_step)
        # Optimize the critic
        if self.learn_step % 2000 == 0:
            print("learn_step: ", self.learn_step, " critic_loss: ", critic_loss)
            print("current_Q1 :", current_Q1.shape, " target_Q: ", target_Q.shape, " current_Q2: ", current_Q2.shape)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_gradient_clipping_norm)
        self.critic_optimizer.step()

    def update_actor_and_alpha_concate(self, batch_sample):
        state = []
        for transition in batch_sample:
            state.append(transition.state)
        state_batch = torch.FloatTensor(state).to(self.device)
        action, log_prob = self.evaluate(state_batch, self.device)
        actor_Q1, actor_Q2 = self.critic(state_batch, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.to(self.device).detach() * log_prob - actor_Q).mean()
        alpha_loss = - (self.log_alpha.to(self.device) * (log_prob + self.target_entropy).detach()).mean()
        # optimize the actor
        self.writer.add_scalar('train_actor/actor_loss', actor_loss, self.learn_step)
        self.writer.add_scalar('train_actor/alpha_loss', alpha_loss, self.learn_step)
        self.writer.add_scalar('train_actor/q_value', actor_Q.mean(), self.learn_step)
        self.writer.add_scalar('train_actor/alpha', self.alpha, self.learn_step)
        self.writer.add_scalar('train_actor/target_entropy', self.target_entropy, self.learn_step)
        self.writer.add_scalar('train_actor/log_prob_total', log_prob.mean(), self.learn_step)
        if self.learn_step % 2000 == 0:
            print("learn_step: ", self.learn_step, " actor_loss: ", actor_loss, " alpha_loss: ", alpha_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_gradient_clipping_norm)
        self.actor_optimizer.step()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_critic_sample(self, batch_sample):
        critic_loss = 0
        number = len(batch_sample)
        self.alpha = self.log_alpha.exp().to(self.device)
        time_start = time.time()
        for transition in batch_sample:
            state = torch.FloatTensor(transition.state).to(self.device)
            action = torch.FloatTensor([transition.action]).to(self.device)
            mask = torch.FloatTensor([transition.mask]).to(self.device)
            reward = torch.FloatTensor([transition.reward]).to(self.device)
            next_state = torch.FloatTensor(transition.next_state).to(self.device)
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            next_action, log_prob = self.evaluate(next_state, self.device)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.to(self.device).detach() * log_prob
            target_Q = reward + (mask * self.gamma * target_V)
            target_Q = target_Q.detach()
            # get current Q estimation
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = critic_loss + F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q)
        if self.learn_step % 2000 == 0:
            print("learn_step: ", self.learn_step, " critic_loss: ", critic_loss)
            print("current_Q1 :", current_Q1.shape, " target_Q: ", target_Q.shape, " current_Q2: ",
                  current_Q2.shape)
            print("current_Q1 :", current_Q1, " target_Q: ", target_Q, " current_Q2: ",
                  current_Q2)
        time_end = time.time()
        self.writer.add_scalar('train_critic/critic_loss', critic_loss / number, self.learn_step)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_critic_batch(self, batch_sample):
        self.alpha = self.log_alpha.exp().to(self.device)
        state_batch = torch.FloatTensor(batch_sample.state).to(self.device)
        action_batch = torch.FloatTensor(batch_sample.action).to(self.device)
        reward_batch = torch.FloatTensor(batch_sample.reward).to(self.device)
        mask_batch = torch.FloatTensor(batch_sample.mask).to(self.device)
        next_state_batch = torch.FloatTensor(batch_sample.next_state).to(self.device)
        mask_batch = mask_batch.unsqueeze(1)
        reward_batch = reward_batch.unsqueeze(1)
        next_action, log_prob = self.evaluate(next_state_batch, self.device)
        target_Q1, target_Q2 = self.critic_target(next_state_batch, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha * log_prob
        target_Q = reward_batch + (mask_batch * self.gamma * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(state_batch, action_batch)
        critic_loss1 = F.mse_loss(current_Q1, target_Q)
        critic_loss2 = F.mse_loss(current_Q2, target_Q)
        critic_loss = critic_loss1 + critic_loss2
        self.writer.add_scalar('train_critic/target_Q', target_Q.mean(), self.learn_step)
        self.writer.add_scalar('train_critic/current_Q1', current_Q1.mean(), self.learn_step)
        self.writer.add_scalar('train_critic/current_Q2', current_Q2.mean(), self.learn_step)
        self.writer.add_scalar('train_critic/critic_loss', critic_loss, self.learn_step)
        self.writer.add_scalar('train_critic/critic_loss1', critic_loss1, self.learn_step)
        self.writer.add_scalar('train_critic/critic_loss2', critic_loss2, self.learn_step)
        # Optimize the critic
        if self.learn_step % 2000 == 0:
            print("learn_step: ",  self.learn_step, " critic_loss: ", critic_loss)
            print("current_Q1 :", current_Q1.shape, " target_Q: ", target_Q.shape,  " current_Q2: ", current_Q2.shape)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_gradient_clipping_norm)
        self.critic_optimizer.step()


    def update_actor_and_alpha_batch(self, batch_sample):
        state_batch = torch.FloatTensor(batch_sample.state).to(self.device)
        action, log_prob = self.evaluate(state_batch, self.device)
        actor_Q1, actor_Q2 = self.critic(state_batch, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.to(self.device).detach() * log_prob - actor_Q).mean()
        alpha_loss = - (self.log_alpha.to(self.device) * (log_prob + self.target_entropy).detach()).mean()
        # optimize the actor
        self.writer.add_scalar('train_actor/actor_loss', actor_loss, self.learn_step)
        self.writer.add_scalar('train_actor/alpha_loss', alpha_loss, self.learn_step)
        self.writer.add_scalar('train_actor/q_value', actor_Q.mean(), self.learn_step)
        self.writer.add_scalar('train_actor/alpha', self.alpha, self.learn_step)
        self.writer.add_scalar('train_actor/target_entropy', self.target_entropy, self.learn_step)
        self.writer.add_scalar('train_actor/log_prob_total', log_prob.mean(), self.learn_step)
        if self.learn_step % 2000 == 0:
            print("learn_step: ",  self.learn_step, " actor_loss: ", actor_loss, " alpha_loss: ", alpha_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_gradient_clipping_norm)
        self.actor_optimizer.step()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_actor_and_alpha(self, batch_sample):
        actor_loss = 0
        alpha_loss = 0
        log_prob_total = 0
        q_value = 0
        number = len(batch_sample)
        for transition in batch_sample:
            state = torch.FloatTensor(transition.state).to(self.device)
            state = state.unsqueeze(0)
            action, log_prob = self.evaluate(state, self.device)
            actor_Q1, actor_Q2 = self.critic(state, action)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            log_prob_total = log_prob_total + log_prob
            q_value = q_value + actor_Q
            actor_loss = actor_loss + (self.alpha.detach() * log_prob - actor_Q).mean()
            alpha_loss = alpha_loss - (self.alpha * (log_prob + self.target_entropy).detach()).mean()
        # optimize the actor
        self.writer.add_scalar('train_actor/actor_loss', actor_loss / number, self.learn_step)
        self.writer.add_scalar('train_actor/alpha_loss', alpha_loss / number, self.learn_step)
        self.writer.add_scalar('train_actor/q_value', q_value / number, self.learn_step)
        self.writer.add_scalar('train_actor/alpha', self.alpha, self.learn_step)
        self.writer.add_scalar('train_actor/target_entropy', self.target_entropy, self.learn_step)
        self.writer.add_scalar('train_actor/log_prob_total', log_prob_total / number, self.learn_step)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def learn_model_concate(self, batch_sample):
        time_start = time.time()
        self.update_critic_concate(batch_sample)
        if self.learn_step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha_concate(batch_sample)
        if self.learn_step % self.critic_target_update_frequency == 0:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        self.learn_step = self.learn_step + 1
        self.update_learning_rate()
        time_end = time.time()

    def learn_model_sample(self, batch_sample):
        time_start = time.time()
        self.update_critic(batch_sample)
        if self.learn_step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(batch_sample)
        if self.learn_step % self.critic_target_update_frequency == 0:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        self.learn_step = self.learn_step + 1
        time_end = time.time()


    def learn_model_batch(self, batch_sample):
        self.update_critic_batch(batch_sample)
        if self.learn_step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha_batch(batch_sample)
        if self.learn_step % self.critic_target_update_frequency == 0:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        self.learn_step = self.learn_step + 1
        self.update_learning_rate()
        self.save_parameter()


    def save(self, step):
        if not os.path.exists(self.model_fold):
            os.makedirs(self.model_fold)
        critic_save_path = os.path.join(self.model_fold, str(step) + '_critic_net.pth')
        print("save model to path:", critic_save_path)
        torch.save(self.critic.state_dict(), critic_save_path)

        critic_target_save_path = os.path.join(self.model_fold, str(step) + '_critic_target_net.pth')
        print("save model to path:", critic_target_save_path)
        torch.save(self.critic_target.state_dict(), critic_target_save_path)

        actor_save_path = os.path.join(self.model_fold, str(step) + "_actor_net.pth")
        print("save model to path:", actor_save_path)
        torch.save(self.actor.state_dict(), actor_save_path)

    def load(self, step):

        critic_path = os.path.join(self.model_fold, str(step) + '_critic_net.pth')
        if os.path.exists(critic_path):
            print("load model from path:", critic_path)
            self.critic.load_state_dict(torch.load(critic_path))

        critic_target_path = os.path.join(self.model_fold, str(step) + '_critic_target_net.pth')
        if os.path.exists(critic_target_path):
            print("load model from path:", critic_target_path)
            self.critic_target.load_state_dict(torch.load(critic_target_path))

        actor_path = os.path.join(self.model_fold, str(step) + '_actor_net.pth')
        if os.path.exists(actor_path):
            print("load model from path:", actor_path)
            self.actor.load_state_dict(torch.load(actor_path))


    def save_parameter(self):
        save_parameter = os.path.join(self.model_fold, "train_parameter.npz")
        np.savez(
            save_parameter,
            learning_step=self.learn_step,
            learning_rate=self.lr,
            log_alpha=self.log_alpha.cpu().detach().numpy()
        )

    def load_parameter(self):
        save_parameter = os.path.join(self.model_fold, "train_parameter.npz")
        record_result = np.load(save_parameter)
        self.learn_step = record_result["learning_step"]
        self.lr = record_result["learning_rate"]
        self.log_alpha = torch.tensor(record_result["log_alpha"], requires_grad=True).to(self.device)

