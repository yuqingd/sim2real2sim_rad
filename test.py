from env_wrapper import make
import matplotlib.pyplot as plt
import time
import numpy as np

# ENV = 'FetchPickAndPlace-v1'
ENV = 'FetchReach-v1'


def show(i):
    img = env.render(mode='rgb_array', camera_id=i)
    plt.imshow(img.transpose(1, 2, 0))
    plt.title(ENV)
    plt.show()


def save(i):
    img = env.render(mode='rgb_array', camera_id=i)
    plt.imsave(ENV + '.png', img.transpose(1, 2, 0))


env = make(ENV, None, np.random.randint(100000), False, 100, 100, [1], change_model=True)
# env.set_special_reset('grip')
# env.reset()

# env.reset(save_special_steps=True)
# while True:
#     env.render()

# while True:
#     start = time.time()
#     env.render(mode='rgb_array')
#     end = time.time()
#     print(end - start)
# show(0)
show(10)
# show(2)
