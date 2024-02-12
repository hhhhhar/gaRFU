import argparse
import torch
import gym
from buffer.memory_buffer import Replay_Buffer
from sac_mlp_agent import sac_agent
from collections import namedtuple
from train_and_evel_process import train_process_record, evel_process_record
import random
import numpy as np
from env.franka_garment_env import FrankaGarmentEnv
import sys
sys.path.append(".")  

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_env, evel_env, train_agent):

    train_model = train_process_record(args.result_fold)
    evel_model = evel_process_record(
        args.result_fold, args.evel_frequency, args.evel_number)
    train_step = int(args.num_train_step * 1.05)
    obs_dict = train_env.reset()
    state = obs_dict['observation']
    episode_reward = 0
    episode_step = 0
    for step_i in range(train_step):
        if step_i < args.learn_start:
            action = train_env.action_space.sample()
        else:
            action = train_agent.action(state)
        action = action.clip(train_env.action_space.low,
                             train_env.action_space.high).reshape(1, 1, 3)
        print("action :", action)
        
        state_new, reward, done, info = train_env.step(action)
        train_agent.memory.store_transition(
            state, action, reward, state_new)
        state = state_new
        episode_reward += reward
        episode_step += 1
        if step_i > args.learn_start and step_i % args.learn_frequency == 0:
            if args.learn_batch:
                batch_sample = train_agent.memory.sample_batch()
                if len(batch_sample) > 0:
                    train_agent.learn_model_batch(batch_sample)
                    # train_agent.update_learning_rate(step_i)
                else:
                    print("batch size is not enough")
                # print(" learn batch")
            else:
                batch_sample = train_agent.memory.sample()
                if len(batch_sample) > 0:
                    train_agent.learn_model_concate(batch_sample)
                else:
                    print("batch size is not enough")
                # print(" learn sample")
        if step_i % args.evel_frequency == 0 and step_i > 0:
            evel_model.evel_train_model(train_agent, evel_env, step_i)
        if done:
            print("train Total T:{} Mean Reward: \t{:0.2f}".format(
                step_i, episode_reward))
            train_model.save_data(step_i, episode_reward, episode_step)
            train_agent.write_train_process(episode_reward, step_i)
            episode_reward = 0
            episode_step = 0
            state = train_env.reset()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--seed', type=int, default=0)
    # env
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v3')
    parser.add_argument("--action_dim", default=3, type=int)
    parser.add_argument("--state_dim", default=7, type=int)
    #
    parser.add_argument('--num_train_step', type=int, default=1e7, metavar='N',
                        help='number of steps (default: 1e7)')
    # replay
    parser.add_argument('--buffer_size', type=int, default=1000000, metavar='N',
                        help='replay buffer ')

    parser.add_argument("--batch_size", default=256, type=int)

    parser.add_argument('--result_fold', type=str, default='train_process')
    # learning
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='G',
                        help='learning rate 0.001)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='parameter update')

    parser.add_argument("--learn_start", default=10000, type=int)
    parser.add_argument("--learn_frequency", default=5, type=int)
    parser.add_argument("--evel_frequency", default=10000, type=int)
    parser.add_argument("--evel_number", default=10, type=int)
    parser.add_argument("--learn_batch", default=True, type=bool)
    # save
    args = parser.parse_args()
    # 设置随机数种子
    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print('using the GPU...')
        torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子
    else:
        print('using the CPU...')
        torch.manual_seed(args.seed)

    train_env = FrankaGarmentEnv(
        executable_file='@editor',
        # scene_file='FrankaRobotics.json',
        max_episode_length=500,
        reward_type='',
        seed=None,
        tolerance=0.05,
        load_object=True,
        target_in_air=True,
        block_gripper=False,
        target_xz_range=0.15,
        target_y_range=0.6,
        object_xz_range=0.15,
        asset_bundle_file=None
    )
    
    # pdb.set_trace()
    args.action_dim = train_env.action_space.shape[0]
    args.state_dim = train_env.observation_space['observation'].shape[0]
    args.result_fold = args.env_name + "_" + \
        args.result_fold + "_" + str(args.seed)

    
    transition = namedtuple(
        'Transition', ('state', 'action', 'reward', 'next_state'))

    replay_buffer = Replay_Buffer(
        args.buffer_size, transition, args.batch_size)
    train_agent = sac_agent(args, replay_buffer, device)
    train(args, train_env, train_env, train_agent)
