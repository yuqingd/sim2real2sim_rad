from environments.kitchen.adept_envs.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1
from matplotlib import pyplot as plt
import time

env = KitchenTaskRelaxV1()
env.reset()
env.render('human')

done = False
while not done:
  o = env.render('rgb_array')
  plt.imshow(o)
  plt.show()
  time.sleep(0.1)
  print("action", env.action_space.sample())
  state, done, _, _ = env.step(env.action_space.sample())
  print("state", state)


print("x")