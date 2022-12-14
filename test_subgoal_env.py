import time
from SubGoalEnv import SubGoalEnv
env = SubGoalEnv("reach-v2", render_subactions=True)
obs = env.reset()

print("\ntest  20 times with different taks:")
num_suc = 0
for i in range(20):
    obs = env.reset()
    print(obs)
    goal = env.pretty_obs(obs)['goal']
    print("goal:",goal)
    # print("gripper_pos:",obs[:3])
    action_to_reach_goal = env.scale_env_pos_to_action(goal)
    print("action:",action_to_reach_goal)

    action_to_reach_goal.append(0)
    obs, r, d, i = env.step(action_to_reach_goal)
    print("reward:",r)
    print(i)
    if i['success']:
        num_suc += 1
        print("reached with:",action_to_reach_goal)
    else:
        print("not reached with:",action_to_reach_goal)
    print()
    time.sleep(1)
print("Was: ", num_suc, "times  successfull")



# print("----------------------\nTest 5 random actions:\n----------------------")
# for i in range(5):
#     action = env.action_space.sample()
#     print("action:", action)
#     e_pos = env.scale_action_to_env_pos(action)
#     print("expected env pos", env.scale_action_to_env_pos(action))
#     o, r, d, i = env.step(action)
#     print("observation:", pretty_obs(o))
#     o_g_pos = pretty_obs(o)["gripper_pos"]
#     print("diffrence: x:", abs(o_g_pos[0] - e_pos[0]), " y: ", abs(o_g_pos[1] - e_pos[1]), " y: ",
#           abs(o_g_pos[2] - e_pos[2]),)
#     print("info:",i,"\n")
# print("----------------------------------\nTest if action reach reaches goal:\n----------------------------------")






