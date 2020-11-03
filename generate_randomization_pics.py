from matplotlib import pyplot as plt
import numpy as np
from dr import config_dr


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['EGL_DEVICE_ID'] = "1"
os.environ['MUJOCO_GL'] = 'egl'
import env_wrapper
envs = [
    ('dmc_ball_in_cup', 'catch'),
    ('dmc_walker', 'walk'),
    ('dmc_cheetah', 'run'),
    ('dmc_finger', 'spin'),
    ('dmc_cartpole', 'balance'),
    ('kitchen', 'rope'),
    ('kitchen', 'real_c')
]

class config_maker(object):
    def __init__(self, d):
        self.__dict__ = d

for domain_name, task_name in envs:
    print("Generating videos for", domain_name, task_name)
    config = {
        'task_name': task_name,
        'domain_name': domain_name,
        'dr_option': 'nonconflicting_dr' if domain_name == 'kitchen' else 'simple_dr',
        'simple_randomization': False,
        'mean_scale': 1,
        'range_scale': 0,
        'scale_large_and_small': False,
        'mean_only': True,
    }
    config = config_maker(config)
    config = config_dr(config)
    real_dr_list = config.real_dr_list
    real_dr_params = config.real_dr_params
    dr = config.dr

    sim_env = env_wrapper.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=0,
        height=240,  # args.pre_transform_image_size,
        width=240,  # args.pre_transform_image_size,
        dr_list=real_dr_list,
        real_world=False,
        dr=dr,
        range_scale=1,
        prop_range_scale=True,
        real_dr_params=None,
        frame_skip=1,
        mean_only=True,
    )
    real_env = env_wrapper.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=0,
        height=480,
        width=480,
        dr_list=real_dr_list,
        real_world=True,
        range_scale="NONE",
        prop_range_scale=True,
        real_dr_params=real_dr_params,
        frame_skip=1,
        mean_only=True,
    )
    sim_env.reset()
    real_env.reset()
    real_img = real_env.render('rgb_array')
    real_img = np.transpose(real_img, (1,2,0))
    plt.imshow(real_img)
    plt.savefig(f'ENVS/{task_name}_oracle.png')
    plt.show()
    np.save(f'ENVS/{task_name}_oracle.npy', real_img)
    # plt.show()

    for i in range(20):
        sim_env.reset()
        sim_img = sim_env.render('rgb_array')
        sim_img = np.transpose(sim_img, (1, 2, 0))
        plt.imshow(sim_img)
        plt.savefig(f'ENVS/{task_name}_sim_{i}.png')
        np.save(f'ENVS/{task_name}_sim_{i}.npy', sim_img)


raise ValueError("Done with first part of script; Now go check the videos, choose 4 representative samples for "
                 "each, add them to the dictionary below, and then comment out this and above.")


envs = [
    ('dmc_ball_in_cup', 'catch', [0, 2, 3, 9]),
    ('dmc_walker', 'walk', [0, 1, 2, 9]),
    ('dmc_cheetah', 'run', [0, 2, 3, 5]),
    ('dmc_finger', 'spin', [0, 1, 2, 3]),
    # ('dmc_cartpole', 'balance', [15,1,3,11]),
    ('kitchen', 'rope', [0, 1, 7, 11]),
    ('kitchen', 'real_c', [0, 4, 10, 16])
]

full_list = []
for domain_name, task_name, ids in envs:
    oracle_img = np.load(f'ENVS/{task_name}_oracle.npy')
    sim_env_0 = np.load(f'ENVS/{task_name}_sim_0.npy')
    sim_env_1 = np.load(f'ENVS/{task_name}_sim_1.npy')
    sim_env_2 = np.load(f'ENVS/{task_name}_sim_2.npy')
    sim_env_3 = np.load(f'ENVS/{task_name}_sim_3.npy')
    top = np.concatenate([sim_env_0, sim_env_1], axis=1)
    bottom = np.concatenate([sim_env_2, sim_env_3], axis=1)
    full_sim = np.concatenate([top, bottom], axis=0)
    full_sim[0] = 255
    full = np.concatenate([oracle_img, full_sim])
    full[:, 0] = 255

    # Save
    import imageio
    imageio.imwrite(f'full_{task_name}.png', full)
    imageio.imwrite(f'full_sim_{task_name}.png', full_sim)
    full_list.append(full)

all_envs = np.concatenate(full_list, axis=1)
imageio.imwrite(f'all_envs.png', all_envs)