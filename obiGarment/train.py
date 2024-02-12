import torch
import sys
sys.path.append("..")
from env.franka_garment_env import FrankaGarmentEnv
from model import net

import pytorch_lightning as pl

env = FrankaGarmentEnv(
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

actor_lr = 3e-4
critic_lr = 3e-3
alpha_lr = 3e-4
target_entropy = -env.action_space.shape[0]
gamma = 0.99
tau = 0.005  # 软更新参数

RLModel = net.SACNet(env, actor_lr, critic_lr, alpha_lr, \
                     target_entropy, tau, gamma)

trainer = pl.Trainer(  
        devices='1',  
        max_epochs=200,  
        val_check_interval=100  
    ) 

trainer.fit(RLModel)

net.SACNet.load_from_checkpoint()
