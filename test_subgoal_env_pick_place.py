import time

from SubGoalEnv import SubGoalEnv
env = SubGoalEnv("pick-place-v2", render_subactions=True, rew_type="meta_world_rew")

total_reach = 0
for i in range(50):
    obs = env.reset()
    total_reward = 0
    print("----------------------\nTest pick object actions:\n----------------------")
    # print(obs)
    goal = env.pretty_obs(obs)['first_obj']
    print("obs:", env.pretty_obs(obs))
    action_to_reach_goal = env.scale_env_pos_to_action(goal)
    action_to_reach_goal.append(1)
    action_to_reach_goal[0] -= 0.07
    print("action:", action_to_reach_goal)
    print("pos to action",env.scale_action_to_env_pos(action_to_reach_goal))
    obs, r, d, i1 = env.step(action_to_reach_goal)
    print("info", i1)
    print("reward:", r)
    total_reward += r
    print("----------------------\nTest hold actions:\n----------------------")
    # print(pretty_obs(obs))
    goal = env.pretty_obs(obs)['goal']
    # print("goal:", goal)
    # goal[0] = 0.1
    action_to_reach_goal = env.scale_env_pos_to_action(goal)
    action_to_reach_goal.append(-0.1)
    # print("action:", action_to_reach_goal)
    obs, r, d, i2 = env.step(action_to_reach_goal)
    total_reward += r
    print(i2)
    print("reward:", r)
    # print(i2)
    print(total_reward)
    if i2['success']:
        print("reached with:", action_to_reach_goal)
        total_reach += 1
        print()
        continue
    else:
        print("not reached with:", action_to_reach_goal)
    print()
print(f"\n\n--------------- \nreached:{total_reach}")