
import gymnasium as gym
import platform
import os
import numpy as np

# Graphics and plotting.
# import mediapy as media
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env_swingup import CartPoleSwingupEnv

if __name__ == "__main__":
    gym.register(
        id="myenv/CartPoleSwingup",
        entry_point=CartPoleSwingupEnv,
        max_episode_steps=3000
    )
    
    # Use GPU rendering
    import os
    os.environ["MUJOCO_GL"] = "egl"

    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    vec_env = make_vec_env("myenv/CartPoleSwingup", n_envs=4)
    # agent = PPO.load("./logs/ppo_cartpole_swingup_1800000_steps.zip", env=vec_env, device="cpu")
    agent = PPO.load("./ppo_cartpole_swingup", env=vec_env, device="cpu")
    
    obs = vec_env.reset()
    i = 0
    while True:
        i += 1
        action, _states = agent.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        if i % 3 == 0:
            vec_env.render("human")