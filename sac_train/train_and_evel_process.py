import os
import numpy as np
from numpy import *
import pdb


class train_process_record(object):
    def __init__(self, result_fold, if_continue=False):
        self.model_fold = os.path.join(result_fold, "train_process_save")
        if not os.path.exists(self.model_fold):
            os.makedirs(self.model_fold)
        self.log_path = os.path.join(self.model_fold, "train_process.npz")
        if if_continue:
            record_result = np.load(self.log_path)
            self.train_timesteps = record_result["timesteps"].tolist()
            self.train_result = record_result["results"].tolist()
            self.train_lengths = record_result["ep_lengths"].tolist()
            self.is_successes = record_result["success"].tolist()

        else:
            self.train_timesteps = []
            self.train_results = []
            self.train_lengths = []
            self.is_successes = []

    def save_data(self, time_steps, reward, episode_length, is_cuccess=0):
        self.train_timesteps.append(time_steps)
        self.train_results.append(reward)
        self.train_lengths.append(episode_length)
        self.is_successes.append(is_cuccess)
        np.savez(
            self.log_path,
            timesteps=self.train_timesteps,
            results=self.train_results,
            ep_lengths=self.train_lengths,
            success=self.is_successes
        )


class evel_process_record(object):
    def __init__(self, result_fold, evel_frequency, evel_number, if_continue=False):
        self.model_fold = os.path.join(result_fold, "evel_process_save")
        if not os.path.exists(self.model_fold):
            os.makedirs(self.model_fold)
        self.log_path = os.path.join(self.model_fold, "evel_process.npz")
        if if_continue:
            record_result = np.load(self.log_path)
            self.evel_timesteps = record_result["timesteps"].tolist()
            self.evel_results = record_result["results"].tolist()
            self.evel_lengths = record_result["ep_lengths"].tolist()
            self.evel_successes = record_result["success"].tolist()

        else:
            self.evel_timesteps = []
            self.evel_results = []
            self.evel_lengths = []
            self.evel_successes = []
        self.best_reward = -1000
        self.best_success = 0
        self.evel_frequency = evel_frequency
        self.evel_number = evel_number


    def evel_train_model(self, agent, evel_env, time_steps):
        print("*"*100)
        print("evel model: ", time_steps)
        episode_rewards = []
        episode_steps = []
        # episode_successes = []
        for _ in range(self.evel_number):
            episode_reward = 0
            episode_step = 0
            state = evel_env.reset()
            while True:
                action = agent.action(state, sample=False)
                # print("action :", action)
                action = action.clip(evel_env.action_space.low, evel_env.action_space.high)
                state_new, reward, done, info = evel_env.step(action)
                state = state_new
                episode_reward += reward
                episode_step += 1
                if done:
                    episode_rewards.append(episode_reward)
                    episode_steps.append(episode_step)
                    # episode_successes.append(info["success"])
                    self.save_data(time_steps, episode_reward, episode_step)
                    break

        mean_reward = mean(episode_rewards)
        agent.write_evel_process(mean_reward, time_steps)
        print("test Total T:{} Mean Reward: \t{:0.2f}".format(time_steps, mean_reward))
        # pdb.set_trace()
        if self.best_reward < mean_reward:
            self.best_reward = mean_reward
            agent.save("best")
        # self.best_success = mean(episode_successes)
    def save_data(self, time_steps, reward, episode_length):
        # print("*" * 100)
        # print("evel model: ", time_steps)
        self.evel_timesteps.append(time_steps)
        self.evel_results.append(reward)
        self.evel_lengths.append(episode_length)
        # self.evel_successes.append(is_success)
        np.savez(
            self.log_path,
            timesteps=self.evel_timesteps,
            results=self.evel_results,
            ep_lengths=self.evel_lengths,
            success=self.evel_successes
        )


class evel_point_process_record(object):
    def __init__(self, result_fold, evel_frequency, evel_number, if_continue):
        self.model_fold = os.path.join(result_fold, "evel_process_save")
        if not os.path.exists(self.model_fold):
            os.makedirs(self.model_fold)
        self.log_path = os.path.join(self.model_fold, "evel_process.npz")
        if if_continue:
            record_result = np.load(self.log_path)
            self.evel_timesteps = record_result["timesteps"].tolist()
            self.evel_results = record_result["results"].tolist()
            self.evel_lengths = record_result["ep_lengths"].tolist()
            self.evel_successes = record_result["success"].tolist()

        else:
            self.evel_timesteps = []
            self.evel_results = []
            self.evel_lengths = []
            self.evel_successes = []
        self.evel_timesteps = []
        self.evel_results = []
        self.evel_lengths = []
        self.evel_successes = []
        self.best_reward = -1000
        self.best_success = 0
        self.evel_frequency = evel_frequency
        self.evel_number = evel_number


    def evel_train_model(self, agent, evel_env, time_steps):
        print("*"*100)
        print("evel model: ", time_steps)
        episode_rewards = []
        eposide_success = []
        episode_steps = []
        # episode_successes = []
        for _ in range(self.evel_number):
            episode_reward = 0
            episode_step = 0
            state = evel_env.reset()
            global_state = state[0]["point_cloud"]
            local_state = state[1]["point_cloud"]
            robot_state = state[1]["robot_state"]
            while True:
                action = agent.action(global_state, local_state, robot_state, sample=False)
                # print("action :", action)
                action = action.clip(evel_env.action_space.low, evel_env.action_space.high)
                state_new, reward, done, info = evel_env.step(action)
                local_state = state_new["point_cloud"]
                robot_state = state_new["robot_state"]
                episode_reward += reward
                episode_step += 1
                if done:
                    episode_rewards.append(episode_reward)
                    episode_steps.append(episode_step)
                    eposide_success.append(info["success"])
                    # episode_successes.append(info["success"])
                    self.save_data(time_steps, episode_reward, episode_step)
                    break

        mean_reward = mean(episode_rewards)
        mean_succcess = mean(eposide_success)
        agent.write_evel_process(mean_reward, mean_succcess, time_steps)
        print("test Total T:{} Mean Reward: \t{:0.2f}".format(time_steps, mean_reward))
        # pdb.set_trace()
        if self.best_reward < mean_reward:
            self.best_reward = mean_reward
            agent.save("best")
        # self.best_success = mean(episode_successes)
    def save_data(self, time_steps, reward, episode_length):
        # print("*" * 100)
        # print("evel model: ", time_steps)
        self.evel_timesteps.append(time_steps)
        self.evel_results.append(reward)
        self.evel_lengths.append(episode_length)
        # self.evel_successes.append(is_success)
        np.savez(
            self.log_path,
            timesteps=self.evel_timesteps,
            results=self.evel_results,
            ep_lengths=self.evel_lengths,
            success=self.evel_successes
        )



class evel_abs_process_record(object):
    def __init__(self, result_fold, evel_frequency, evel_number, if_continue):
        self.model_fold = os.path.join(result_fold, "evel_process_save")
        if not os.path.exists(self.model_fold):
            os.makedirs(self.model_fold)
        self.log_path = os.path.join(self.model_fold, "evel_process.npz")
        if if_continue:
            record_result = np.load(self.log_path)
            self.evel_timesteps = record_result["timesteps"].tolist()
            self.evel_results = record_result["results"].tolist()
            self.evel_lengths = record_result["ep_lengths"].tolist()
            self.evel_successes = record_result["success"].tolist()

        else:
            self.evel_timesteps = []
            self.evel_results = []
            self.evel_lengths = []
            self.evel_successes = []
        self.best_reward = -1000
        self.best_success = 0
        self.evel_frequency = evel_frequency
        self.evel_number = evel_number


    def evel_train_model(self, agent, evel_env, time_steps):
        print("*"*100)
        print("evel model: ", time_steps)
        episode_rewards = []
        episode_steps = []
        episode_successes = []
        for _ in range(self.evel_number):
            episode_reward = 0
            episode_step = 0
            state = evel_env.reset()
            # global_state = state[0]
            # local_state = state[2]
            abs_state = state["abs"]
            while True:
                action = agent.action(abs_state, sample=False)
                # print("action :", action)
                action = action.clip(evel_env.action_space.low, evel_env.action_space.high)
                state_new, reward, done, info = evel_env.step(action)
                # local_state = state_new[0]
                abs_state = state_new["abs"]
                episode_reward += reward
                episode_step += 1
                if done:
                    episode_rewards.append(episode_reward)
                    episode_steps.append(episode_step)
                    episode_successes.append(info["success"])
                    self.save_data(time_steps, episode_reward, episode_step)
                    break

        mean_reward = mean(episode_rewards)
        mean_success = mean(episode_successes)
        agent.write_evel_process(mean_reward, mean_success, time_steps)
        print("test Total T:{} Mean Reward: \t{:0.2f}".format(time_steps, mean_reward))
        # pdb.set_trace()
        if self.best_reward < mean_reward:
            self.best_reward = mean_reward
            agent.save("best")
        # self.best_success = mean(episode_successes)
    def save_data(self, time_steps, reward, episode_length):
        # print("*" * 100)
        # print("evel model: ", time_steps)
        self.evel_timesteps.append(time_steps)
        self.evel_results.append(reward)
        self.evel_lengths.append(episode_length)
        # self.evel_successes.append(is_success)
        np.savez(
            self.log_path,
            timesteps=self.evel_timesteps,
            results=self.evel_results,
            ep_lengths=self.evel_lengths,
            success=self.evel_successes
        )

class evel_dqn_sac_process_record(object):
    def __init__(self, result_fold, evel_frequency, evel_number, if_continue):
        self.model_fold = os.path.join(result_fold, "evel_process_save")
        if not os.path.exists(self.model_fold):
            os.makedirs(self.model_fold)
        self.log_path = os.path.join(self.model_fold, "evel_process.npz")
        if if_continue:
            record_result = np.load(self.log_path)
            self.evel_timesteps = record_result["timesteps"].tolist()
            self.evel_results = record_result["results"].tolist()
            self.evel_lengths = record_result["ep_lengths"].tolist()
            self.evel_successes = record_result["success"].tolist()

        else:
            self.evel_timesteps = []
            self.evel_results = []
            self.evel_lengths = []
            self.evel_successes = []
        self.best_reward = -1000
        self.best_success = 0
        self.evel_frequency = evel_frequency
        self.evel_number = evel_number


    def evel_train_model(self, agent, evel_env, time_steps):
        print("*"*100)
        print("evel model: ", time_steps)
        episode_rewards = []
        episode_steps = []
        episode_successes = []
        for _ in range(self.evel_number):
            episode_reward = 0
            episode_step = 0
            state = evel_env.reset_grasp_pose()
            global_state = state["point_cloud"]
            init_local_state = state["point_cloud"]
            grasp_pose = state["grasp_pose"]
            init_robot_state = state["robot_state"]
            grasp_action = agent.choose_grasp_action(global_state, init_local_state, grasp_pose, init_robot_state, sample=False)
            evel_env.selected_best_grasp_pose(grasp_action)
            state = evel_env.reset()
            local_state = state["point_cloud"]
            robot_state = state["robot_state"]
            while True:
                action = agent.action(global_state, local_state, robot_state, sample=False)
                # print("action :", action)
                action = action.clip(evel_env.action_space.low, evel_env.action_space.high)
                state_new, reward, done, info = evel_env.step(action)
                # local_state = state_new[0]
                local_state = state_new["point_cloud"]
                robot_state = state_new["robot_state"]
                episode_reward += reward
                episode_step += 1
                if done:
                    episode_rewards.append(episode_reward)
                    episode_steps.append(episode_step)
                    episode_successes.append(info["success"])
                    self.save_data(time_steps, episode_reward, episode_step)
                    break

        mean_reward = mean(episode_rewards)
        mean_success = mean(episode_successes)
        agent.write_evel_process(mean_reward, mean_success, time_steps)
        print("test Total T:{} Mean Reward: \t{:0.2f}".format(time_steps, mean_reward))
        # pdb.set_trace()
        if self.best_reward < mean_reward:
            self.best_reward = mean_reward
            agent.save("best")
        # self.best_success = mean(episode_successes)
    def save_data(self, time_steps, reward, episode_length):
        # print("*" * 100)
        # print("evel model: ", time_steps)
        self.evel_timesteps.append(time_steps)
        self.evel_results.append(reward)
        self.evel_lengths.append(episode_length)
        # self.evel_successes.append(is_success)
        np.savez(
            self.log_path,
            timesteps=self.evel_timesteps,
            results=self.evel_results,
            ep_lengths=self.evel_lengths,
            success=self.evel_successes
        )


class evel_dqn_sac_abs_process_record(object):
    def __init__(self, result_fold, evel_frequency, evel_number, if_continue):
        self.model_fold = os.path.join(result_fold, "evel_process_save")
        if not os.path.exists(self.model_fold):
            os.makedirs(self.model_fold)
        self.log_path = os.path.join(self.model_fold, "evel_process.npz")
        if if_continue:
            record_result = np.load(self.log_path)
            self.evel_timesteps = record_result["timesteps"].tolist()
            self.evel_results = record_result["results"].tolist()
            self.evel_lengths = record_result["ep_lengths"].tolist()
            self.evel_successes = record_result["success"].tolist()

        else:
            self.evel_timesteps = []
            self.evel_results = []
            self.evel_lengths = []
            self.evel_successes = []
        self.best_reward = -1000
        self.best_success = 0
        self.evel_frequency = evel_frequency
        self.evel_number = evel_number


    def evel_train_model(self, agent, evel_env, time_steps):
        print("*"*100)
        print("evel model: ", time_steps)
        episode_rewards = []
        episode_steps = []
        episode_successes = []
        for _ in range(self.evel_number):
            episode_reward = 0
            episode_step = 0
            state = evel_env.reset_grasp_pose()
            global_state = state["point_cloud"]
            init_local_state = state["point_cloud"]
            grasp_pose = state["grasp_pose"]
            init_robot_state = state["robot_state"]
            grasp_action = agent.choose_grasp_action(global_state, init_local_state, grasp_pose, init_robot_state, sample=False)
            evel_env.selected_best_grasp_pose(grasp_action)
            state = evel_env.reset()
            # local_state = state["point_cloud"]
            robot_state = state["robot_state"]
            while True:
                # print("robot_state: ", robot_state)
                action = agent.action(robot_state, sample=False)
                # print("action :", action)
                action = action.clip(evel_env.action_space.low, evel_env.action_space.high)
                state_new, reward, done, info = evel_env.step(action)
                robot_state = state_new["robot_state"]
                episode_reward += reward
                episode_step += 1
                if done:
                    episode_rewards.append(episode_reward)
                    episode_steps.append(episode_step)
                    episode_successes.append(info["success"])
                    self.save_data(time_steps, episode_reward, episode_step)
                    break

        mean_reward = mean(episode_rewards)
        mean_success = mean(episode_successes)
        agent.write_evel_process(mean_reward, mean_success, time_steps)
        print("test Total T:{} Mean Reward: \t{:0.2f}".format(time_steps, mean_reward))
        # pdb.set_trace()
        if self.best_reward < mean_reward:
            self.best_reward = mean_reward
            agent.save("best")
        # self.best_success = mean(episode_successes)
    def save_data(self, time_steps, reward, episode_length):
        # print("*" * 100)
        # print("evel model: ", time_steps)
        self.evel_timesteps.append(time_steps)
        self.evel_results.append(reward)
        self.evel_lengths.append(episode_length)
        # self.evel_successes.append(is_success)
        np.savez(
            self.log_path,
            timesteps=self.evel_timesteps,
            results=self.evel_results,
            ep_lengths=self.evel_lengths,
            success=self.evel_successes
        )


class evel_robot_process_record(object):
    def __init__(self, result_fold, evel_frequency, evel_number, if_continue):
        self.model_fold = os.path.join(result_fold, "evel_process_save")
        if not os.path.exists(self.model_fold):
            os.makedirs(self.model_fold)
        self.log_path = os.path.join(self.model_fold, "evel_process.npz")
        if if_continue:
            record_result = np.load(self.log_path)
            self.evel_timesteps = record_result["timesteps"].tolist()
            self.evel_results = record_result["results"].tolist()
            self.evel_lengths = record_result["ep_lengths"].tolist()
            self.evel_successes = record_result["success"].tolist()

        else:
            self.evel_timesteps = []
            self.evel_results = []
            self.evel_lengths = []
            self.evel_successes = []
        self.best_reward = -1000
        self.best_success = 0
        self.evel_frequency = evel_frequency
        self.evel_number = evel_number


    def evel_train_model(self, agent, evel_env, time_steps):
        print("*"*100)
        print("evel model: ", time_steps)
        episode_rewards = []
        episode_steps = []
        episode_successes = []
        for _ in range(self.evel_number):
            episode_reward = 0
            episode_step = 0
            state = evel_env.reset()
            # global_state = state[0]
            # local_state = state[2]
            abs_state = state["robot_state"]
            while True:
                action = agent.action(abs_state, sample=False)
                # print("action :", action)
                action = action.clip(evel_env.action_space.low, evel_env.action_space.high)
                state_new, reward, done, info = evel_env.step(action)
                # local_state = state_new[0]
                abs_state = state_new["robot_state"]
                episode_reward += reward
                episode_step += 1
                if done:
                    episode_rewards.append(episode_reward)
                    episode_steps.append(episode_step)
                    episode_successes.append(info["success"])
                    self.save_data(time_steps, episode_reward, episode_step)
                    break

        mean_reward = mean(episode_rewards)
        mean_success = mean(episode_successes)
        agent.write_evel_process(mean_reward, mean_success, time_steps)
        print("test Total T:{} Mean Reward: \t{:0.2f}".format(time_steps, mean_reward))
        # pdb.set_trace()
        if self.best_reward < mean_reward:
            self.best_reward = mean_reward
            agent.save("best")
        # self.best_success = mean(episode_successes)
    def save_data(self, time_steps, reward, episode_length):
        # print("*" * 100)
        # print("evel model: ", time_steps)
        self.evel_timesteps.append(time_steps)
        self.evel_results.append(reward)
        self.evel_lengths.append(episode_length)
        # self.evel_successes.append(is_success)
        np.savez(
            self.log_path,
            timesteps=self.evel_timesteps,
            results=self.evel_results,
            ep_lengths=self.evel_lengths,
            success=self.evel_successes
        )