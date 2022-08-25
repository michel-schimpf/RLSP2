# from helper import pretty_obs_subgoal
import time

import gym
from stable_baselines3 import PPO
import metaworld
from SubGoalEnv import SubGoalEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecVideoRecorder

def execute():
    print(get_device())
    # number_envs_per_task = [1]*10
    # mt10 = metaworld.MT10()
    # env_array = []
    # for i, (name, _) in enumerate(mt10.train_classes.items()):
    #     print(i,name)
    #     for _ in range(number_envs_per_task[i]):
    #         env_array.append(make_env(name, "rew1", 10, i))
    # env_vec = SubprocVecEnv(env_array)

    # env = make_env("pick-place-v2","rew1",10,2)()
    env = SubGoalEnv("pick-place-v2", render_subactions=False, rew_type="rew1")
    # env = SubGoalEnv("pick-place-v2", rew_type="rew1")
    # 0
    # reach - v2
    # 1
    # push - v2
    # 2
    # pick - place - v2
    # 3
    # door - open - v2
    # 4
    # drawer - open - v2
    # 5
    # drawer - close - v2
    # 6
    # button - press - topdown - v2
    # 7
    # peg - insert - side - v2
    # 8
    # window - open - v2
    # 9
    # window - close - v2

    # env = VecVideoRecorder(env_vec, './video', record_video_trigger=lambda x: x == 0, name_prefix="bla",)

    models_dir = "models"
    model_path = f"{models_dir}/56899584"
    model = PPO.load(model_path, env=env)

    episodes = 100
    mean_rew_all_tasks = 0
    num_success = 0
    mean_steps = 0
    for ep in range(episodes):
        print("\n---------\nepisode:", ep)
        obs = env.reset()

        done = False
        steps = 0
        total_reward = 0
        success = False
        while not done:
            # env.render()
            action, _states = model.predict(obs,)
            # print("obs", pretty_obs(obs))
            # print("action:", action)
            # print("intended subgoal:", env.scale_action_to_env_pos(action))
            obs, reward, done, info = env.step(action)
            # print("ob", pretty_obs(obs))
            # print("goal:", pretty_obs(obs)["goal"])
            # obj = pretty_obs(obs)['first_obj']
            # distance_to_subgoal = np.linalg.norm(obs[:3] - obj[:3])
            # print("distance to object:", distance_to_subgoal)
            # print("info",info)
            # print("reward:", reward)
            steps += 1
            total_reward += reward
            if info['success']:
                success = True
            # print()
            if done and success:
                num_success += 1
        # print("total reward:",total_reward)
        # print("mean reward:",total_reward/steps)
        # print("finished after: ", steps, " steps \n")
        mean_rew_all_tasks += total_reward
        mean_steps += steps
    print("mean_tot_rew:",mean_rew_all_tasks/episodes)
    print("mean_steps:", mean_steps/episodes)
    print("success rate:",num_success/episodes)


# def make_env(name,rew_type,number_of_one_hot_tasks, one_hot_task_index):
#
#     def _init():
#         return SubGoalEnv(env=name, rew_type=rew_type, number_of_one_hot_tasks=number_of_one_hot_tasks,
#                           one_hot_task_index=one_hot_task_index,render_subactions=True)
#     return _init


if __name__ == '__main__':
    execute()