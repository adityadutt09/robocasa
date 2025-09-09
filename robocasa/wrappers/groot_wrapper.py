import sys
from typing import Any, Dict

import cv2
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

from .gym_wrapper import (
    REGISTERED_ENVS,
    RoboCasaEnv,
)

ALLOWED_LANGUAGE_CHARSET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\n\t[]{}()!?'_:"
)
FINAL_IMAGE_RESOLUTION = (256, 256)


class GrootRoboCasaEnv(RoboCasaEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = self.key_converter.deduce_observation_space(self.env)
        mapped_names, _, _, _ = self.key_converter.get_camera_config()
        for mapped_name in mapped_names:
            self.observation_space[mapped_name] = spaces.Box(
                low=0, high=255, shape=(*FINAL_IMAGE_RESOLUTION, 3), dtype=np.uint8
            )

        self.observation_space["annotation.human.task_description"] = spaces.Text(
            max_length=256, charset=ALLOWED_LANGUAGE_CHARSET
        )
        self.action_space = self.key_converter.deduce_action_space(self.env)

        self.verbose = False
        for k, v in self.observation_space.items():
            self.verbose and print("{OBS}", k, v)
        for k, v in self.action_space.items():
            self.verbose and print("{ACTION}", k, v)

    @staticmethod
    def process_img(img):
        h, w, _ = img.shape
        if h != w:
            dim = max(h, w)
            y_offset = (dim - h) // 2
            x_offset = (dim - w) // 2
            img = np.pad(img, ((y_offset, y_offset), (x_offset, x_offset), (0, 0)))
            h, w = dim, dim
        if (h, w) != FINAL_IMAGE_RESOLUTION:
            img = cv2.resize(img, FINAL_IMAGE_RESOLUTION, cv2.INTER_AREA)
        return np.copy(img)

    def get_groot_observation(self, raw_obs):
        obs = {}
        temp_obs = self.key_converter.map_obs(raw_obs)
        for k, v in temp_obs.items():
            if k.startswith("hand.") or k.startswith("body."):
                obs["state." + k[5:]] = v
            else:
                raise ValueError(f"Unknown key: {k}")
        mapped_names, camera_names, _, _ = self.key_converter.get_camera_config()
        for mapped_name, camera_name in zip(mapped_names, camera_names):
            obs[mapped_name] = GrootRoboCasaEnv.process_img(
                raw_obs[camera_name + "_image"]
            )

        obs["annotation.human.task_description"] = raw_obs["language"]

        return obs

    def reset(self, seed=None, options=None):
        raw_obs, info = super().reset(seed=seed, options=options)
        obs = self.get_groot_observation(raw_obs)
        return obs, info

    def step(self, action):
        for k, v in action.items():
            self.verbose and print("<ACTION>", k, v)

        action = self.key_converter.unmap_action(action)
        raw_obs, reward, terminated, truncated, info = super().step(action)
        obs = self.get_groot_observation(raw_obs)

        for k, v in obs.items():
            self.verbose and print("<OBS>", k, v.shape if k.startswith("video.") else v)
        self.verbose = False

        return obs, reward, terminated, truncated, info


def create_grootrobocasa_env_class(env):
    class_name = f"{env}"
    id_name = f"robocasa/{class_name}"

    env_class_type = type(
        class_name,
        (GrootRoboCasaEnv,),
        {
            "__init__": lambda self, **kwargs: super(self.__class__, self).__init__(
                env_name=env,
                **kwargs,
            )
        },
    )

    current_module = sys.modules["robocasa.wrappers.groot_wrapper"]
    setattr(current_module, class_name, env_class_type)
    register(
        id=id_name,  # Unique ID for the environment
        entry_point=f"robocasa.wrappers.groot_wrapper:{class_name}",  # Path to your environment class
    )


for ENV in REGISTERED_ENVS:
    create_grootrobocasa_env_class(ENV)
