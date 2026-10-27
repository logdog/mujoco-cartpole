# Mujoco Cart Pole

Goals:

1. Create a sim environment for a custom cart-pole in Mujoco. (`cart-pole.xml`)
2. Design a stabilization controller with PPO. (`env_balance.py` and `eval_balance.py`)
3. Do swing-up control of the cart-pole with PPO. (`env_swingup.py` and `eval_swingup.py`)

## Running this locally

You will need `Mujoco`, `gymnasium`, and `StableBaselines3`. If you are using `conda`, you can run
```
conda create --file environment.yml
```
and then run `conda activate mujoco-rl`. 

## Files

The `cart-pole.xml` describes the cart pole setup for Mujoco.

`cartpole_v1.py` is from the StableBaselines3 tutorial to verify that PPO works on my machine. It does not use Mujoco.

`env_balance.py` creates the training environment and trains a PPO agent for the cart pole system with the pendulum already in the up position. You can run the agent with `eval_balance.py`.

`env_swingup.py` trains a PPO agent for swing-up and balance of the cart-pole system. Pay attention to the reward function, it is non-trivial! You can evaluate it using `eval_swingup.py`.

The `manual_control.py` and `open_loop.py` are exactly what you would expect.

