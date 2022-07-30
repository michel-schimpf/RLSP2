import numpy as np

from SubGoalEnv import SubGoalEnv, scale_env_pos_to_action, pretty_obs

env = SubGoalEnv("pick-place-v2", render_subactions=True)
obs = env.reset()
print("----------------------\nTest action to env point\n----------------------")
actions = [[-1, -1, -0.9, 0], [1, -1, -0.1, 0], [-1, 1, -0.1, 0], [1, 1, -0.1, 0],
           [-1, -1, 1, 0], [1, -1, 1, 0], [-1, 1, 1, 0], [1, 1, 1, 0], ]
total_reach = 0

#
# for a in actions:
#     obs = env.reset()
#     print("o:", obs[:4])
#     print("a:", a)
#     print("pos:", scale_action_to_env_pos(a))
#     obs, r, d, i1 = env.step(a)
#
#     goal = pretty_obs(obs)['goal']
#     distance_to_subgoal = np.linalg.norm(obs[:3] - goal[:3])
#     print("distance to goal:", distance_to_subgoal)
#
#
#     # -----
#     print("---")
#     print("o:", obs[:4])
#     first_obj = pretty_obs(obs)['first_obj']
#     print("first_obj:", first_obj)
#     action_to_reach_goal = scale_env_pos_to_action(first_obj)
#     action_to_reach_goal.append(1)
#     print("action:", action_to_reach_goal)
#     print("action in env pos:", scale_action_to_env_pos(action_to_reach_goal))
#     obs, r, d, i1 = env.step(action_to_reach_goal)
#     print("reward:", r)
#     print("info", i1)
#     #  -----
#     print("---")
#     print(pretty_obs(obs))
#     goal = pretty_obs(obs)['goal']
#     print("goal:", goal)
#     action_to_reach_goal = scale_env_pos_to_action(goal)
#     action_to_reach_goal.append(-1)
#     print("action:", action_to_reach_goal)
#     print("action as env pos:", scale_action_to_env_pos(action_to_reach_goal))
#     obs, r, d, i2 = env.step(action_to_reach_goal)
#     print("reward:", r)
#     print(i2)
#     if i2['success']:
#         print("reached with:", action_to_reach_goal)
#         total_reach += 1
#     else:
#         print("not reached with:", action_to_reach_goal)
#
#     print("--------------------------------------------")

total_reward = 0
it= 200
for i in range(it):
    print(i)
    obs = env.reset()
    # print("o:", obs[:4])
    a = env.action_space.sample()
    # print("a:", a)
    # print("pos:", scale_action_to_env_pos(a))
    obs, r, d, i1 = env.step(a)
    # total_reward += r
    # print("reward:", r)
    # print("---")
    # print("o:", obs[:4])
    goal = pretty_obs(obs)['first_obj']
    distance_to_subgoal = np.linalg.norm(obs[:3] - goal[:3])
    # print("distance to object:", distance_to_subgoal)
    # print("goal:", goal)
    action_to_reach_goal = scale_env_pos_to_action(goal)
    action_to_reach_goal.append(1)
    # print("action:", action_to_reach_goal)
    obs, r, d, i1 = env.step(action_to_reach_goal)
    total_reward += r
    # print("reward:", r)
    # print("info", i1)
    # print("---")
    # print(pretty_obs(obs))
    goal = pretty_obs(obs)['goal']
    # print("goal:", goal)
    action_to_reach_goal = scale_env_pos_to_action(goal)
    action_to_reach_goal.append(-1)
    # print("action:", action_to_reach_goal)
    obs, r, d, i2 = env.step(action_to_reach_goal)
    total_reward += r
    # print("reward:", r)
    # print(i2)
    if i2['success']:
        print("reached with:", action_to_reach_goal)
        total_reach += 1
    else:
        print("not reached with:", action_to_reach_goal)

    print("--------------------------------------------")
#
print(f"total_rach of {it}:", total_reach)
# print("total mean rew:",total_reward/it)



