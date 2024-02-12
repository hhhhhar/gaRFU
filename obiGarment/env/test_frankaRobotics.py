import sys
sys.path.append("..")
from env.franka_garment_env import FrankaGarmentEnv
import random

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
env.reset()

while 1:
    obs, reward, done, info = env.step(env.action_space.sample())
    if done:
        env.reset()