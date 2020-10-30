from env_wrapper import make
from dr import config_dr
import numpy as np
import moviepy.editor as mpy
import os

# ====== ADJUST THESE ==========
save_dir = 'videos_cartpole'
domain_name = 'dmc_cartpole'
task_name = 'balance'
dr_option = 'simple_dr'
action_repeat = 4
time_limit = 8
mean_scale = 2
range_scale = 1
seed = 7
prop_range_scale = False
# ========= END ==========

# Source: https://goodcode.io/articles/python-dict-object/
class config_maker(object):
    def __init__(self, d):
        self.__dict__ = d


def get_traj(env, dr=None):
    obs = env.reset()
    if dr is not None:
        env.set_dr(dr)
        env.apply_dr()
    if type(obs) is dict:
        obs = obs['image']
    a = env.action_space.high
    frames = [obs]
    for _ in range(time_limit):
        obs, _, _, _ = env.step(a)
        if type(obs) is dict:
            obs = obs['image']
        frames.append(obs)
    frames = [np.transpose(obs, (1, 2, 0)) for obs in frames]
    return frames

def compute_diff_traj(dr):
    env1 = make(domain_name, task_name, seed, 480, 480, frame_skip=action_repeat, real_world=False, prop_range_scale=False,
                mean_only=True, dr=None, real_dr_params=config.real_dr_params, dr_list=config.real_dr_list, range_scale=0)
    env2 = make(domain_name, task_name, seed, 480, 480, frame_skip=action_repeat, real_world=True, prop_range_scale=False,
                mean_only=True, real_dr_params=config.real_dr_params, dr_list=config.real_dr_list, range_scale=0)

    original_traj = get_traj(env1, dr)
    modified_traj = get_traj(env2)
    diff_traj = [((x - y + 255) / 2).astype(np.int32) for x, y in zip(original_traj, modified_traj)]
    diff_arr = np.stack(diff_traj)
    return diff_arr

try:
    os.mkdir(save_dir)
except:
    print("save_dir", save_dir, "already exists")

config = {
    'domain_name': domain_name,
    'dr_option': dr_option,
    'simple_randomization': False,
    'mean_scale': 1,
    'range_scale': 0,
    'scale_large_and_small': False,
    'mean_only': True,
}
config = config_maker(config)
config = config_dr(config)

# Only randomize a single param at a time
for param in config.real_dr_list:
    r1 = config.real_dr_params[param]
    r2 = mean_scale
    r3 = (1 - range_scale)
    if prop_range_scale:
        low_val = np.clip(config.real_dr_params[param] * (1 / mean_scale) * (1 - range_scale), 1e-2, float('inf'))
        high_val = np.clip(config.real_dr_params[param] * mean_scale * (1 + range_scale), 1e-2, float('inf'))
    else:
        low_val = np.clip(config.real_dr_params[param] * (1 / mean_scale) - range_scale, 1e-2, float('inf'))
        high_val = np.clip(config.real_dr_params[param] * mean_scale + range_scale, 1e-2, float('inf'))
    dr_low = {param: low_val}
    dr_high = {param: high_val}
    diff_traj_low = compute_diff_traj(dr_low)
    diff_traj_high = compute_diff_traj(dr_high)

    diff_traj = [np.concatenate([low, high], axis=1) for low, high in zip(diff_traj_low, diff_traj_high)]

    fps = 2
    clip = mpy.ImageSequenceClip(diff_traj, fps=fps)
    clip.write_gif(os.path.join(save_dir, 'DIFF_IN_' + param + '.gif'), fps=fps)