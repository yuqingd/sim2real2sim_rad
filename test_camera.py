import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from kitchen_env import Kitchen

env = Kitchen('rope')
img = env.render()
plt.imshow(img.transpose(1,2,0))
plt.show()
