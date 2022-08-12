# from helper import pretty_obs_subgoal
import time

import gym
from stable_baselines3 import PPO
import metaworld
from SubGoalEnv import SubGoalEnv, pretty_obs
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecVideoRecorder


def execute():
    print(get_device())
    number_envs_per_task = [1] * 10
    mt10 = metaworld.MT10()
    env_array = []

    for i, (name, _) in enumerate(mt10.train_classes.items()):
        print(i, name)
        for _ in range(number_envs_per_task[i]):
            env_array.append(make_env(name, "rew1", 10, i))

    env_vec = SubprocVecEnv(env_array)
    env = make_env("pick-place-v2", "rew1", 10, 2)()
    env = VecVideoRecorder(env_vec, './video', record_video_trigger=lambda x: x == 0, name_prefix="bla", )

    models_dir = "models/PPO"
    model_path = f"{models_dir}/846692352.zip"
    model = PPO.load(model_path, env=env)

    obs = [0.00615235, 0.6001898, 0.19430117, 1., 0.06607989, 0.61659635,
           0.03, 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0.,
           0.00615235, 0.6001898, 0.19430117, 1., 0.06607989, 0.61659635,
           0.03, 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0.,
           -0.31399141, 0.59422512, 0.12919564, 0., 0., 0.,
           0., 0., 0., 0., 1., 0.,
           0.]
    action, _states = model.predict(obs, deterministic=True)
    print(action)


def make_env(name, rew_type, number_of_one_hot_tasks, one_hot_task_index):
    def _init():
        return SubGoalEnv(env=name, rew_type=rew_type, number_of_one_hot_tasks=number_of_one_hot_tasks,
                          one_hot_task_index=one_hot_task_index, render_subactions=False)

    return _init


if __name__ == '__main__':
    execute()
