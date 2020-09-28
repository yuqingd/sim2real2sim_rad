import argparse
import numpy as np
import matplotlib.pyplot as plt

"""
takes in any amount of .npy files, and plots the wanted key
"""

# SKETCHY!!

# https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa/56062555
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


parser = argparse.ArgumentParser()
parser.add_argument("--save_name", type=str, default="plot.png")
parser.add_argument("--logdir", nargs="+", type=str)
parser.add_argument("--key", type=str, default='mean_ep_reward')
args = parser.parse_args()

stuff = []

plt.figure()
plt.xlabel(args.key)
plt.ylabel("timesteps")
for logdir in args.logdir:
	name = logdir.split("/")[-1].strip('--eval_scores.npy')
	print(name)
	data = np.load(logdir)
	outer_dict = data.reshape(1,)[0]
	key = list(outer_dict.keys())[0] #TODO: change for multiple keys?
	dicts = outer_dict[key]
	# logged_vals = dicts[0].keys()
	timesteps = list(dicts.keys())
	mean_rews = [d[args.key] for d in dicts.values()]
	plt.plot(timesteps, mean_rews, label=name)
plt.legend()
plt.savefig(args.save_name)
plt.show()
