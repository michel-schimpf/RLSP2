import time
from ObstacleEnviroment.fetch.pick_dyn_obstacles2 import pretty_obs
from SubGoalEnv import SubGoalEnv

env = SubGoalEnv(env="obstacle_env", render_subactions=False, rew_type="rew1")
obs = env.reset()
num_suc = 0
# print("----------------------\nTest 5 random actions:\n----------------------")
# for _ in range(5):
#     obs = env.reset()
#     action = env.action_space.sample()
#     obs, r, d, i = env.step(action)
#
#     goal_position = env.pretty_obs(obs)["goal"]
#     print("Goal_position:", goal_position)
#     action = env.scale_env_pos_to_action(goal_position)
#     action.append(1)
#     obs, r, d, i = env.step(action)
#     print(r, d)
#     print(i)


#
for _ in range(20):
    print("---------------------------------------")
    obs = env.reset()
    print(obs)
    print(pretty_obs(obs))
    goal_position = env.pretty_obs(obs)["goal"]
    print("Goal_position:", goal_position)
    action = env.scale_env_pos_to_action(goal_position)
    gripper_pos = env.pretty_obs(obs)['gripper_pos'][:3]
    print("gripper_pos:", gripper_pos)
    # gripper_pos[0] -= 0.1
    # gripper_pos[1] += 0.1
    # action = env.scale_env_pos_to_action(gripper_pos)
    action.append(1)
    # action = env.action_space.sample()
    # action = [-1, -1,1,1]
    print("action:", action)
    print("env_pos for action:", env.scale_action_to_env_pos(action))
    obs, r, d, i = env.step(action)
    # print("reward:",r)
    # print("pretty obs:", pretty_obs(obs))
    print("info:",i)
    if i["success"]:
        # print("done!")
        num_suc += 1
    else:
        time.sleep(2)
        print("DOOOOOD")
    # print()
print("Number of successes:",num_suc)








