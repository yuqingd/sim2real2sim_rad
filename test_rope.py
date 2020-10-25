from env_wrapper import make
from video import VideoRecorder
import os
import utils
from train import parse_args, make_agent
import torch
from train import parse_args
# ENV = 'FetchPickAndPlace-v1'
#

# from mujoco_py import GlfwContext
import mujoco_py
# GlfwContext(offscreen=True)

args = parse_args()

env = make('kitchen', real_world=False,
        task_name='rope',
        seed=0,
        height=512,
        width=512,
        dr_list=args.real_dr_list,
        dr_shape=args.sim_params_size,
        dr=args.dr,
        mean_only=args.mean_only,
    )
# env.set_special_reset('grip')
env.reset()
env.apply_dr()
num_episodes = 1
time_limit = 60
image_size = 512
real_video_dir = utils.make_dir(os.path.join('./logdir', 'debug_video'))

video = VideoRecorder(real_video_dir, camera_id=0)

cpf = 3
obs_shape = (cpf * 3, image_size, image_size)
action_shape = env.action_space.shape
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_eval_loop(sample_stochastically=True):
    for i in range(num_episodes):
        obs_dict = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        obs_traj = []
        step = 0
        while not done and step < time_limit:
            obs = obs_dict['image']
            # center crop image
            obs = utils.center_crop_image(obs, image_size)
            if step < 20:
                action = [0, .3, .3]
            elif step < 40:
                action = [.1, .1, 0]
            else:
                action = [0.00, 0, -.3]
            step+=1
            obs_traj.append(obs)
            obs_dict, reward, done, info = env.step(action)
            video.record(env)
            episode_reward += reward
            print(reward)
            if info["success"]:
                print("Done step", step)
                break
        video.save('sim_rope_damping10.mp4')

run_eval_loop()


# ######## TEST CABINET
# env = make('kitchen', real_world=False,
#         task_name='real_c',
#         seed=0,
#         height=512,
#         width=512,
#         dr_list=[],
#         dr_shape=args.sim_params_size,
#         dr={},
#         mean_only=True,
#         range_scale=1,
#     )
# # env.set_special_reset('grip')
# env.reset()
# env.apply_dr()
# num_episodes = 4
# time_limit = 100
# image_size = 512
# real_video_dir = utils.make_dir(os.path.join('./logdir', 'debug_video'))
#
# video = VideoRecorder(real_video_dir, camera_id=0, height=512, width=512)
#
#
# model = env._env._env.sim.model
# geom_dict = model._geom_name2id
# body_dict = model._body_name2id
#
#
# cpf = 3
# cabinet_joint = model.joint_name2id('slidedoor_joint')
# # model.jnt_range[cabinet_joint:cabinet_joint+1, 1] = .05
# cabinet_collision_indices = [geom_dict[name] for name in geom_dict.keys() if "cabinet_collision" in name]
# for p in cabinet_collision_indices:
#     model.geom_friction[p, 0:1] = 5
#
#
#
# obs_shape = (cpf * 3, image_size, image_size)
# action_shape = env.action_space.shape
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# def run_eval_loop(sample_stochastically=True):
#     video.init(enabled=True)
#     for i in range(num_episodes):
#         obs_dict = env.reset()
#         video.record(env)
#         done = False
#         episode_reward = 0
#         obs_traj = []
#         step = 0
#         while not done and step < time_limit:
#             obs = obs_dict['image']
#             # center crop image
#             obs = utils.center_crop_image(obs, image_size)
#
#             if i % 4 == 0:
#                 # Succeeds in the task
#                 if step < 20:
#                     action = [.3, .3, .005]
#                 elif step < 40:
#                     action = [-.3, .1, -.2]
#                 else:
#                     action = [-.3, 0, -.1]
#
#             if i % 4  == 1:
#                 # hits the cabinet from the side
#                 if step < 20:
#                     action = [.3, .3, -.2]
#                 elif step < 40:
#                     action = [-.3, .1, -.2]
#                 else:
#                     action = [-.3, 0, -.1]
#                 # action = [0, 0, 0]
#
#             if i % 4  == 2:
#                 # Punches down the lid of the cabinet
#                 if step < 20:
#                     action = [-.1, .3, .01]
#                 else:
#                     action = [0, 0, -1]
#
#             if i  % 4 == 3:
#                 # pulls the handle, but releases halfway (to see if it pops back)
#                 if step < 20:
#                     action = [.3, .3, .005]
#                 elif step < 70:
#                     action = [-.3, .1, -.2]
#                 else:
#                     action = [0, 0, 1]
#
#             step+=1
#             obs_traj.append(obs)
#             obs_dict, reward, done, info = env.step(action)
#             video.record(env)
#             episode_reward += reward
#             print(step, info['success'], reward)
#     video.save('sim_cabinet.mp4')
#
# run_eval_loop()

# ######## TEST PUSH
# image_size = 480
# env = make('kitchen', real_world=False,
#         task_name='real_p',
#         seed=0,
#         height=image_size,
#         width=image_size,
#         dr_list=args.real_dr_list,
#         dr_shape=args.sim_params_size,
#         dr=args.dr,
#         mean_only=args.mean_only,
#     )
# # env.set_special_reset('grip')
# env.reset()
# env.apply_dr()
# print(env.goal)
# num_episodes = 1
# time_limit = 60
# real_video_dir = utils.make_dir(os.path.join('./logdir', 'debug_video'))
#
# video = VideoRecorder(real_video_dir, camera_id=0)
#
# cpf = 3
# obs_shape = (cpf * 3, image_size, image_size)
# action_shape = env.action_space.shape
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# def run_eval_loop(sample_stochastically=True):
#     for i in range(num_episodes):
#         obs_dict = env.reset()
#         video.init(enabled=(i == 0))
#         done = False
#         episode_reward = 0
#         obs_traj = []
#         step = 0
#         while not done and step < time_limit:
#             obs = obs_dict['image']
#             # center crop image
#             obs = utils.center_crop_image(obs, image_size)
#             if step < 10:
#                 action = [.17, 0., -1]
#             else:
#                 action = [-.325, 1, 0]
#
#             step+=1
#             obs_traj.append(obs)
#             obs_dict, reward, done, info = env.step(action)
#             video.record(env)
#             episode_reward += reward
#         video.save('sim_push.mp4')
#
# run_eval_loop()
