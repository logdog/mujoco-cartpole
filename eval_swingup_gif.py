
import os
import time

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO

from env_swingup import CartPoleSwingupEnv

if __name__ == "__main__":
    gym.register(
        id="myenv/CartPoleSwingup",
        entry_point=CartPoleSwingupEnv,
        max_episode_steps=2048,
    )

    # Use GPU rendering.
    os.environ["MUJOCO_GL"] = "egl"

    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    env = gym.make("myenv/CartPoleSwingup", render_mode="rgb_array")
    agent = PPO.load("./ppo_cartpole_swingup", env=env, device="cpu")

    obs, _ = env.reset()
    frames = []
    total_steps = 0
    max_steps = 500

    while total_steps < max_steps:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        total_steps += 1
        # time.sleep(1.0 / 100.0)

        if terminated or truncated:
            obs, _ = env.reset()

    frames = frames[::5]
    if frames:
        imageio.mimsave("eval_swingup.gif", frames, fps=20, loop=0)
        print(f"Saved {len(frames)} frames to eval_swingup.gif")
    else:
        print("No frames captured; GIF not saved.")

    env.close()