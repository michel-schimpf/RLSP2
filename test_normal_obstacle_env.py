import time
from ObstacleEnviroment.fetch.pick_dyn_obstacles2 import FetchPickDynObstaclesEnv2

env = FetchPickDynObstaclesEnv2()
obs = env.reset()
num_suc = 0
print("----------------------\nTest 5 random actions:\n----------------------")
for _ in range(10000):
    env.render()
    action = env.action_space.sample()
    print(action)
    obs, r, d, i = env.step(action)
    print(obs)


