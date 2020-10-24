from env_wrapper import make
from video import VideoRecorder
import os
import utils
from train import parse_args, make_agent
import torch
from train import parse_args
# ENV = 'FetchPickAndPlace-v1'
#
args = parse_args()
#
# env = make('kitchen', real_world=False,
#         task_name='rope',
#         seed=0,
#         height=84,
#         width=84,
#         dr_list=args.real_dr_list,
#         dr_shape=args.sim_params_size,
#         dr=args.dr,
#         mean_only=args.mean_only,
#     )
# # env.set_special_reset('grip')
# env.reset()
# env.apply_dr()
# num_episodes = 1
# time_limit = 60
# image_size = 84
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
#             if step < 20:
#                 action = [0, .3, .6]
#             elif step < 40:
#                 action = [.15, .1, 0]
#             else:
#                 action = [0.00, 0, -.2]
#             step+=1
#             obs_traj.append(obs)
#             obs_dict, reward, done, info = env.step(action)
#             video.record(env)
#             episode_reward += reward
#             print(reward)
#         video.save('sim_rope_damping10.mp4')
#
# run_eval_loop()


######## TEST CABINET

env = make('kitchen', real_world=False,
        task_name='real_c',
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

video = VideoRecorder(real_video_dir, camera_id=0, height=512, width=512)

cpf = 3
obs_shape = (cpf * 3, image_size, image_size)
action_shape = env.action_space.shape
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_eval_loop(sample_stochastically=True):
    for i in range(num_episodes):
        obs_dict = env.reset()
        video.init(enabled=(i == 0))
        video.record(env)
        done = False
        episode_reward = 0
        obs_traj = []
        step = 0
        while not done and step < time_limit:
            obs = obs_dict['image']
            # center crop image
            obs = utils.center_crop_image(obs, image_size)
            if step < 20:
                action = [.3, .3, .6]
            elif step < 40:
                action = [-.3, .1, -.2]
            else:
                action = [-.3, 0, -.1]
            action = [0, 0, 0]
            step+=1
            obs_traj.append(obs)
            obs_dict, reward, done, info = env.step(action)
            video.record(env)
            episode_reward += reward
            print(reward)
        video.save('sim_cabinet.mp4')

run_eval_loop()

# ######## TEST PUSH
#
# env = make('kitchen', real_world=False,
#         task_name='real_p',
#         seed=0,
#         height=84,
#         width=84,
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
# time_limit = 1
# image_size = 84
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
#             action = [0., 1, 0 ]
#             step+=1
#             obs_traj.append(obs)
#             obs_dict, reward, done, info = env.step(action)
#             video.record(env)
#             episode_reward += reward
#             print(reward)
#         video.save('sim_push.mp4')
#
# run_eval_loop()
