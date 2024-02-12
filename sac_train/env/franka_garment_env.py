from pyrfuniverse.envs.gym_goal_wrapper_env import RFUniverseGymGoalWrapper
import numpy as np
import quaternion
from gym import spaces
from gym.utils import seeding
import copy
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from pyrfuniverse.side_channel import (
    IncomingMessage,
    OutgoingMessage,
)
import pyrfuniverse.attributes as attr
import time
import pyrfuniverse.utils.depth_processor as dp


class FrankaGarmentEnv(RFUniverseGymGoalWrapper):
    metadata = {"render.modes": ["human"]}
    # height_offset is just the object width / 2, which is the object's stable height value.
    height_offset = 0.025

    def __init__(
        self,
        max_episode_length,
        reward_type,
        tolerance,
        load_object,
        target_in_air,
        block_gripper,
        target_xz_range,
        target_y_range,
        object_xz_range,
        seed=1234,
        executable_file=None,
        scene_file=None,
        asset_bundle_file=None,
        assets: list = [],
    ):
        super().__init__(executable_file, scene_file, assets=assets)
        self.max_steps = max_episode_length
        self.reward_type = reward_type
        self.tolerance = tolerance
        self.load_object = load_object
        self.target_in_air = target_in_air
        self.block_gripper = block_gripper
        self.goal_range_low = np.array(
            [-target_xz_range, self.height_offset, -target_xz_range]
        )
        self.goal_range_high = np.array(
            [target_xz_range, target_y_range, target_xz_range]
        )
        self.object_range_low = np.array(
            [-object_xz_range, self.height_offset, -object_xz_range]
        )
        self.object_range_high = np.array(
            [object_xz_range, self.height_offset, object_xz_range]
        )
        self.asset_bundle_file = asset_bundle_file

        self.seed(seed) 
        self._env_setup()
        self.init_pos = [0.15, 0.75, 0]
        self.t = 0
        self.action_space = spaces.Box(low=-0.8, high=0.8, shape=(3,), dtype=np.float32)
        self.midPt = [0., 0., 0.]

        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32
                ),
                "desired_goal": spaces.Box(
                    -np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32
                ),
                "achieved_goal": spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32
                ),
            }
        )
        
        self.resetTurn = False

    def step(self, action: np.ndarray):
        """
        Params:
            action: 4-d numpy array.
        """
        # print(action)
        pos_ctrl = action[0, 0, :3] * 0.008
        # euler = action[3:]
        # euler = np.quaternion(euler[0], euler[1], euler[2], euler[3])

        curr_pos = np.array(self.attrs[9658740].data["positions"][3])
        # print(curr_pos)
        pos_ctrl = curr_pos + pos_ctrl

        # print(pos_ctrl)
        # self.instance_channel.set_action(
        #     "IKTargetDoMove",
        #     id=965874,
        #     position=[pos_ctrl[0], pos_ctrl[1], pos_ctrl[2]],
        #     duration=0.1,
        #     speed_based=True
        # )
        self.attrs[965874].IKTargetDoMove(
            position=[pos_ctrl[0], pos_ctrl[1], pos_ctrl[2]],
            duration=0.1,
            speed_based=True,
        )

        # self._set_franka_joints(np.array(joint_positions))
        self._step()
        self.t += 1

        obs = self._get_obs()
        done = False
        info = {"is_success": self._check_success(obs)}
        if self._check_success(obs):
            done = True
            self._set_gripper_width(0.02)

        reward = self.compute_reward(obs["achieved_goal"], info)

        if self.t == self.max_steps:
            done = True

        return obs, reward, done, info

    def reset(self):
        super().reset()
        
        if self.load_object and self.resetTurn:
            self.delCloth()
            self._step()
            self.testLoadPre(1)
            self._step()
            self.attrs[114514] = self.GetAttr(114514)
            self._step()
        if(not self.resetTurn):
           self.resetTurn = True
        self.t = 0
        object_pos = None

        # self.instance_channel.set_action(
        #     'SetTransform',
        #     id=0,
        #     position=list(self.goal)
        # )
        # self._step()

        # self.ik_controller.reset()

        if self.target_in_air:
            self.attrs[965874].IKTargetDoMove(
                position=[self.init_pos[0], self.init_pos[1], self.init_pos[2]],
                duration=0,
                speed_based=False,
            )
            self._step()
            self.attrs[9658740].SetJointPositionDirectly(joint_positions=[0.04, 0.04])
            self._step()

        self._step()

        return self._get_obs()

    def seed(self, seed=1234):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        self._step()

    def compute_reward(self, achieved_goal, info):
        self.getCollarMid()
        self._step()
        collarDis = np.linalg.norm(achieved_goal - self.midPt)
        if self.reward_type == "sparse":
            return -(np.array(collarDis > self.tolerance, dtype=np.float32))
        else:
            return -collarDis

    def _get_obs(self) -> dict:
        gripper_position = np.array(self.attrs[9658740].data["positions"][3])
        gripper_velocity = np.array(self.attrs[9658740].data["velocities"][3])
        gripper_quaternion = np.array(self.attrs[9658740].data["quaternions"][3])
        gripper_width = self._get_gripper_width()
        # gripper_joint_position = np.array(self.articulation_channel.data[1]['joint_positions'])
        # gripper_joint_velocity = np.array(self.articulation_channel.data[1]['joint_velocities'])
        achieved_goal = gripper_position.copy()
        panda_obs = np.concatenate(
            (gripper_position, gripper_velocity, [gripper_width])
        )

        self.getCollarMid()
        self._step()
        collarMid = np.array(self.midPt)
        object_obs = np.concatenate((panda_obs, collarMid))

        return {
            "observation": object_obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": collarMid,
        }

    def _env_setup(self):
        if self.load_object:
            self.testLoadPre(1)
            self._step()
            self.attrs[114514] = self.GetAttr(114514)
        
        self.camera = self.GetAttr(8864)
        self._step()

        self.attrs[965874].SetIKTargetOffset(position=[0, 0.105, 0])
        self.attrs[965874].IKTargetDoRotate(
            rotation=[0, 45, 180], duration=0, speed_based=False
        )
        self._step()

    def _generate_random_float(self, min: float, max: float) -> float:
        assert min < max, "Min value is {}, while max value is {}.".format(min, max)
        random_float = np.random.rand()
        random_float = random_float * (max - min) + min

        return random_float

    def _set_gripper_width(self, w: float):
        w = w / 2
        self.attrs[9658740].SetJointPosition(joint_positions=[w, w])

    def _get_gripper_width(self) -> float:
        gripper_joint_positions = copy.deepcopy(
            self.attrs[9658740].data["joint_positions"]
        )
        return -1 * (gripper_joint_positions[0] + gripper_joint_positions[1])

    def _check_success(self, obs):
        achieved_goal = obs["achieved_goal"]
        self.grasp2dis(achieved_goal)
        self._step()
        return np.array(self.collarDis < self.tolerance, dtype=np.float32)

    def _compute_goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def rotateObject(self):
        rotate = np.random.rand(3)
        rotate = (rotate * 45).tolist()
        self.attrs[114514].SetTransform(rotation=rotate)
        self._step()
    
    def testListen(self, index:int)->None:
        self.SendMessage('testLog', index)

    def testLoadPre(self, index:int)->None:
        self.SendMessage('loadTest', index)
        self.AddListener('loadRes', self.getLoadRes)
    
    def getLoadRes(self, msg:IncomingMessage):
        res = msg.read_bool()
        print(f'Load is {res}')

    def delCloth(self)->None:
        self.SendMessage('delCloth')

    def grasp2dis(self, grasp):
        self.SendMessage('getGrasp', grasp[0], grasp[1], grasp[2])
        self.AddListener("collarDis", self.getCollarDis)

    def getCollarDis(self, msg:IncomingMessage):
        self.collarDis = msg.read_float32()

    def getCollarMid(self):
        self.SendMessage('getCollarMid')
        self.AddListener("collarMid", self.recCollarMid)

    def recCollarMid(self, msg:IncomingMessage):
        self.midPt = msg.read_float32_list()
