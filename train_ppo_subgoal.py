# from gym.wrappers import Monitor
from typing import Tuple

from stable_baselines3 import PPO
from SubGoalEnv import SubGoalEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from RL_PPA_monitor import RLPPAMonitor


def train():
    models_dir = f"models/PPO"
    logdir = "logs"
    TIMESTEPS = 256
    env = SubGoalEnv("pick-place-v2", rew_type="rew1")
    env_vec = SubprocVecEnv([lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env, lambda: env,
                             # lambda: env, lambda: env, lambda: env,  # lambda: env,
                             ])
    env_vec = RLPPAMonitor(env_vec, "logs/PPO_0", )
    # right batch_size: https://github.com/llSourcell/Unity_ML_Agents/blob/master/docs/best-practices-ppo.md
    # TODO what are right paramters
    model = PPO('MlpPolicy', env_vec, verbose=1, tensorboard_log=logdir, n_steps=TIMESTEPS,
                batch_size=4096, )
    # model = ALGO.load("models/PPO3/15360000.zip", env=env_vec,tensorboard_log=logdir)
    iters = 0
    while True:
        print(iters)
        iters += 1
        model = model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,
                            tb_log_name="PPO_0", )
        model.save(f"{models_dir}/{TIMESTEPS * iters * 24}")


if __name__ == '__main__':
    train()
