import datetime, uuid
from copy import deepcopy
import numpy as np
import os
import robocasa  # we need this to register environments  # noqa: F401
import robosuite

# from robocasa.environments.tabletop.tabletop import Tabletop
# from robocasa.models.robots import (
#     # GROOT_ROBOCASA_ENVS_GR1_ARMS_ONLY,
#     # GROOT_ROBOCASA_ENVS_GR1_ARMS_AND_WAIST,
#     # GROOT_ROBOCASA_ENVS_GR1_FIXED_LOWER_BODY,
#     # gather_robot_observations,
#     # make_key_converter,
# )

import gymnasium as gym
from gymnasium import spaces

from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from robosuite.controllers.composite.composite_controller import HybridMobileBase
from robosuite.environments.base import REGISTERED_ENVS


ALLOWED_LANGUAGE_CHARSET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\n\t[]{}()!?'_:"
)


from robocasa.utils.env_utils import create_env


# def gather_robot_observations(env, verbose=False):
#     observations = {}

#     for robot_id, robot in enumerate(env.robots):
#         sim = robot.sim
#         gripper_names = {
#             robot.get_gripper_name(arm): robot.gripper[arm] for arm in robot.arms
#         }
#         for part_name, indexes in robot._ref_joints_indexes_dict.items():
#             qpos_values = []
#             for joint_id in indexes:
#                 qpos_addr = sim.model.jnt_qposadr[joint_id]
#                 # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtjoint
#                 joint_type = sim.model.jnt_type[joint_id]
#                 if joint_type == mujoco.mjtJoint.mjJNT_FREE:
#                     qpos_size = 7  # Free joint has 7 DOFs
#                 elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
#                     qpos_size = 4  # Ball joint has 4 DOFs (quaternion)
#                 else:
#                     qpos_size = 1  # Revolute or prismatic joint has 1 DOF
#                 qpos_values = np.append(
#                     qpos_values, sim.data.qpos[qpos_addr : qpos_addr + qpos_size]
#                 )
#             if part_name in gripper_names.keys():
#                 gripper = gripper_names[part_name]
#                 # Reverse the order to match the real robot
#                 qpos_values = unformat_gripper_space(gripper, qpos_values)[::-1]
#             if len(qpos_values) > 0:
#                 observations[f"robot{robot_id}_{part_name}"] = qpos_values

#     if verbose:
#         print("States:", [(k, len(observations[k])) for k in observations])

#     return observations


class RobotKeyConverter:
    @classmethod
    def get_camera_config(cls):
        raise NotImplementedError

    @classmethod
    def map_obs(cls, input_obs):
        raise NotImplementedError

    @classmethod
    def map_action(cls, input_action):
        raise NotImplementedError

    @classmethod
    def unmap_action(cls, input_action):
        raise NotImplementedError

    @classmethod
    def get_metadata(cls, name):
        raise NotImplementedError

    @classmethod
    def map_obs_in_eval(cls, input_obs):
        output_obs = {}
        mapped_obs = cls.map_obs(input_obs)
        for k, v in mapped_obs.items():
            assert k.startswith("hand.") or k.startswith("body.")
            output_obs["state." + k[5:]] = v
        return output_obs

    @classmethod
    def get_missing_keys_in_dumping_dataset(cls):
        return {}

    @classmethod
    def convert_to_float64(cls, input):
        for k, v in input.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float32:
                input[k] = v.astype(np.float64)
        return input

    @classmethod
    def deduce_observation_space(cls, env):
        obs = (
            env.viewer._get_observations(force_update=True)
            if env.viewer_get_obs
            else env._get_observations(force_update=True)
        )
        # obs.update(gather_robot_observations(env))
        obs = cls.map_obs(obs)
        observation_space = spaces.Dict()

        for k, v in obs.items():
            if k.startswith("hand.") or k.startswith("body."):
                observation_space["state." + k[5:]] = spaces.Box(
                    low=-1, high=1, shape=(len(v),), dtype=np.float32
                )
            else:
                raise ValueError(f"Unknown key: {k}")

        return observation_space

    @classmethod
    def deduce_action_space(cls, env):
        # action = cls.map_action(reconstruct_latest_actions(env))
        action = {
            "hand.gripper_close": np.zeros(1),
            "body.end_effector_position": np.zeros(3),
            "body.end_effector_rotation": np.zeros(3),
            "body.base_motion": np.zeros(4),
            "body.control_mode": np.zeros(1),
        }
        action_space = spaces.Dict()
        for k, v in action.items():
            if isinstance(v, np.int64):
                action_space["action." + k[5:]] = spaces.Discrete(2)
            elif isinstance(v, np.ndarray):
                action_space["action." + k[5:]] = spaces.Box(
                    low=-1, high=1, shape=(len(v),), dtype=np.float32
                )
            else:
                raise ValueError(f"Unknown type: {type(v)}")
        return action_space


class PandaOmronKeyConverter(RobotKeyConverter):
    @classmethod
    def get_camera_config(cls):
        mapped_names = [
            "video.robot0_agentview_left",
            "video.robot0_agentview_right",
            "video.robot0_eye_in_hand",
        ]
        camera_names = [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ]
        camera_widths, camera_heights = 128, 128
        return mapped_names, camera_names, camera_widths, camera_heights

    @classmethod
    def map_obs(cls, input_obs):
        output_obs = type(input_obs)()
        output_obs = {
            "hand.gripper_qpos": input_obs["robot0_gripper_qpos"],
            "body.base_position": input_obs["robot0_base_pos"],
            "body.base_rotation": input_obs["robot0_base_quat"],
            "body.end_effector_position_relative": input_obs["robot0_base_to_eef_pos"],
            "body.end_effector_rotation_relative": input_obs["robot0_base_to_eef_quat"],
        }
        return output_obs

    @classmethod
    def map_action(cls, input_action):
        output_action = type(input_action)()
        output_action = {
            "hand.gripper_close": (
                np.int64(0) if input_action["robot0_right_gripper"] < 0 else np.int64(1)
            ),
            # "hand.gripper_close": input_action["robot0_right_gripper"],
            "body.end_effector_position": input_action["robot0_right"][..., 0:3],
            "body.end_effector_rotation": input_action["robot0_right"][..., 3:6],
            "body.base_motion": np.concatenate(
                (
                    input_action["robot0_base"],
                    input_action["robot0_torso"],
                ),
                axis=-1,
            ),
            "body.control_mode": (
                np.int64(0) if input_action["robot0_base_mode"] < 0 else np.int64(1)
            ),
            # "body.control_mode": input_action["robot0_base_mode"],
        }
        return output_action

    @classmethod
    def unmap_action(cls, input_action):
        output_action = type(input_action)()
        output_action = {
            "robot0_right_gripper": (
                -1.0 if input_action["action.gripper_close"] < 0.5 else 1.0
            ),
            "robot0_right_gripper": input_action["action.gripper_close"],
            "robot0_right": np.concatenate(
                (
                    input_action["action.end_effector_position"],
                    input_action["action.end_effector_rotation"],
                ),
                axis=-1,
            ),
            "robot0_base": input_action["action.base_motion"][..., 0:3],
            "robot0_torso": input_action["action.base_motion"][..., 3:4],
            "robot0_base_mode": input_action["action.control_mode"],
            "robot0_base_mode": (
                -1.0 if input_action["action.control_mode"] < 0.5 else 1.0
            ),
        }
        return output_action

    @classmethod
    def get_metadata(cls, name):
        from gr00t.data.schema import RotationType

        if name in [
            "body.base_position",
            "body.end_effector_position_relative",
            "body.end_effector_position",
        ]:
            return {
                "absolute": False,
                "rotation_type": None,
            }
        elif name in ["body.base_rotation", "body.end_effector_rotation_relative"]:
            return {
                "absolute": False,
                "rotation_type": RotationType.QUATERNION,
            }
        elif name in ["body.end_effector_rotation"]:
            return {
                "absolute": False,
                "rotation_type": RotationType.AXIS_ANGLE,
            }
        else:
            return {
                "absolute": True,
                "rotation_type": None,
            }


class RoboCasaEnv(gym.Env):
    def __init__(
        self,
        env_name=None,
        robots_name=None,
        camera_names=None,
        camera_widths=None,
        camera_heights=None,
        enable_render=True,
        dump_rollout_dataset_dir=None,
        split="train",
        **kwargs,  # Accept additional kwargs
    ):
        self.key_converter = PandaOmronKeyConverter
        (
            _,
            camera_names,
            default_camera_widths,
            default_camera_heights,
        ) = self.key_converter.get_camera_config()

        if camera_widths is None:
            camera_widths = default_camera_widths
        if camera_heights is None:
            camera_heights = default_camera_heights

        # controller_configs = load_composite_controller_config(
        #     controller=None,
        #     robot=robots_name.split("_")[0],
        # )
        # if (
        #     robots_name in GROOT_ROBOCASA_ENVS_GR1_ARMS_ONLY
        #     or robots_name in GROOT_ROBOCASA_ENVS_GR1_ARMS_AND_WAIST
        #     or robots_name in GROOT_ROBOCASA_ENVS_GR1_FIXED_LOWER_BODY
        # ):
        #     controller_configs["type"] = "BASIC"
        #     controller_configs["composite_controller_specific_configs"] = {}
        #     controller_configs["control_delta"] = False

        # self.env, self.env_kwargs = create_env_robosuite(
        #     env_name=env_name,
        #     robots=robots_name.split("_"),
        #     controller_configs=controller_configs,
        #     camera_names=camera_names,
        #     camera_widths=camera_widths,
        #     camera_heights=camera_heights,
        #     enable_render=enable_render,
        #     **kwargs,  # Forward kwargs to create_env_robosuite
        # )

        self.env_name = env_name
        print(f"Creating {env_name} with split={split}")
        self.env = create_env(
            env_name=env_name,
            render_onscreen=False,
            # seed=0, # set seed=None to run unseeded
            split=split,
        )
        self.env.reset()

        # TODO: the following info should be output by grootrobocasa
        self.camera_names = camera_names
        self.camera_widths = camera_widths
        self.camera_heights = camera_heights
        self.enable_render = enable_render
        self.render_obs_key = f"{camera_names[0]}_image"
        self.render_cache = None

        # setup spaces
        action_space = spaces.Dict()
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            for part_name, controller in cc.part_controllers.items():
                min_value, max_value = -1, 1
                start_idx, end_idx = cc._action_split_indexes[part_name]
                shape = [end_idx - start_idx]
                this_space = spaces.Box(
                    low=min_value, high=max_value, shape=shape, dtype=np.float32
                )
                action_space[f"{pf}{part_name}"] = this_space
            if isinstance(cc, HybridMobileBase):
                this_space = spaces.Discrete(2)
                action_space[f"{pf}base_mode"] = this_space

            action_space = spaces.Dict(action_space)
            self.action_space = action_space

        obs = (
            self.env.viewer._get_observations(force_update=True)
            if self.env.viewer_get_obs
            else self.env._get_observations(force_update=True)
        )
        # obs.update(gather_robot_observations(self.env))
        observation_space = spaces.Dict()
        for obs_name, obs_value in obs.items():
            shape = list(obs_value.shape)
            if obs_name.endswith("_image"):
                continue
            min_value, max_value = -1, 1
            this_space = spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=np.float32
            )
            observation_space[obs_name] = this_space

        for camera_name in camera_names:
            shape = [camera_heights, camera_widths, 3]
            this_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
            observation_space[f"{camera_name}_image"] = this_space

        observation_space["language"] = spaces.Text(
            max_length=256, charset=ALLOWED_LANGUAGE_CHARSET
        )

        self.observation_space = observation_space

        self.dump_rollout_dataset_dir = dump_rollout_dataset_dir
        self.groot_exporter = None
        self.np_exporter = None

    def get_basic_observation(self, raw_obs):
        # raw_obs.update(gather_robot_observations(self.env))

        # Image are in (H, W, C), flip it upside down
        def process_img(img):
            return np.copy(img[::-1, :, :])

        for obs_name, obs_value in raw_obs.items():
            if obs_name.endswith("_image"):
                # image observations
                raw_obs[obs_name] = process_img(obs_value)
            else:
                # non-image observations
                raw_obs[obs_name] = obs_value.astype(np.float32)

        # Return black image if rendering is disabled
        if not self.enable_render:
            for name in self.camera_names:
                raw_obs[f"{name}_image"] = np.zeros(
                    (self.camera_heights, self.camera_widths, 3), dtype=np.uint8
                )

        self.render_cache = raw_obs[self.render_obs_key]
        raw_obs["language"] = self.env.get_ep_meta().get("lang", "")

        return raw_obs

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        raw_obs = self.env.reset()
        # return obs
        obs = self.get_basic_observation(raw_obs)

        info = {}
        info["success"] = False
        info["grasp_distractor_obj"] = False

        return obs, info

    def step(self, action_dict):
        env_action = []
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            action = np.zeros(cc.action_limits[0].shape)
            for part_name, controller in cc.part_controllers.items():
                start_idx, end_idx = cc._action_split_indexes[part_name]
                act = action_dict.pop(f"{pf}{part_name}")
                action[start_idx:end_idx] = act
            if isinstance(cc, HybridMobileBase):
                action[-1] = action_dict.pop(f"{pf}base_mode")
            env_action.append(action)

        assert len(action_dict) == 0, f"Unprocessed actions: {action_dict}"
        env_action = np.concatenate(env_action)

        raw_obs, reward, done, info = self.env.step(env_action)
        # sparse reward
        is_success = self.env._check_success()
        reward = 1.0 if is_success else 0.0

        obs = self.get_basic_observation(raw_obs)

        truncated = False

        info["success"] = reward > 0
        info["grasp_distractor_obj"] = False
        if hasattr(self, "_check_grasp_distractor_obj"):
            info["grasp_distractor_obj"] = self._check_grasp_distractor_obj()

        return obs, reward, done, truncated, info

    def render(self):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        return self.render_cache

    def close(self):
        self.env.close()
