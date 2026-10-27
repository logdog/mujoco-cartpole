
import gymnasium as gym
import platform
import distutils.util
import os
import subprocess
import mujoco
import time
import itertools
import numpy as np

# Graphics and plotting.
# import mediapy as media
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from collections import deque

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from env_balance import CartPoleBalanceEnv

if __name__ == "__main__":
    gym.register(
        id="myenv/CartPoleBalance",
        entry_point=CartPoleBalanceEnv,
        max_episode_steps=10000
    )
    
    # Use GPU rendering
    import os
    os.environ["MUJOCO_GL"] = "egl"

    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    vec_env = make_vec_env("myenv/CartPoleBalance", n_envs=4)
    # agent = PPO.load("./logs/ppo_cartpole_balance_100000_steps.zip", env=vec_env, device="cpu")
    agent = PPO.load("./ppo_cartpole_balance", env=vec_env, device="cpu")
    
    obs = vec_env.reset()
    i = 0
    while True:
        i += 1
        action, _states = agent.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        if i % 3 == 0:
            vec_env.render("human")