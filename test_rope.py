from env_wrapper import make
from video import VideoRecorder
import os
import utils
from train import parse_args, make_agent
import torch
from train import parse_args
# ENV = 'FetchPickAndPlace-v1'
from mujoco_py import GlfwContext
import mujoco_py
import cv2
import numpy as np
args = parse_args()
GlfwContext(offscreen=True)
os.environ['MUJOCO_GL'] = 'glfw'
import time
utils.set_seed_everywhere(1)

env_real = make('kitchen', real_world=True,
        task_name='real_c',
        seed=1,
        height=512,
        width=512,
        frame_skip=0,
        state_type="robot",
        time_limit=60,
    )

env_sim = make('kitchen', real_world=False,
        task_name='real_c',
        seed=0,
        height=512,
        width=512,
        delay_steps=2
               )
env_sim = utils.FrameStack(env_sim, k=args.frame_stack)
env_real = utils.FrameStack(env_real, k=args.frame_stack)

env_real.reset()
env_sim.reset()
num_episodes = 3

time_limit = 60
image_size = 84
real_video_dir = utils.make_dir(os.path.join('./logdir', 'eval_video'))

video = VideoRecorder(real_video_dir, camera_id=0)

cpf = 3
obs_shape = (cpf * 3, image_size, image_size)
action_shape = env_sim.action_space.shape
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_shape = env_sim.reset()['state'].shape[0]

agent = make_agent(
    obs_shape=obs_shape,
    state_shape=state_shape,
    action_shape=action_shape,
    args=args,
    device=device
)

model_dir = args.work_dir + '/baseline/cabinet/'
agent.load(model_dir, 405000)
agent.load_curl(model_dir, 405000)

# agent.load(model_dir, 350040)
# agent.load_curl(model_dir, 350040)


def run_eval_loop(env, name):
    for i in range(num_episodes):
        obs_dict = env.reset()
        video.init()
        video.record(env)

        done = False
        episode_reward = 0
        obs_traj = []
        step = 0
        while not done and step < time_limit:
            obs = obs_dict['image']
            obs = np.transpose(obs, [1, 2, 0])
            obs =  cv2.resize(obs, dsize=(image_size, image_size))

            obs = np.transpose(obs, [2, 0, 1])

            # center crop image
            obs = utils.center_crop_image(obs, image_size)
            # ##CABINET
            # if step < 20:
            #     action = [-.2, .3, -.2]
            # else:
            #     action = [-.15, 0, 0]
            # ##ROPE
            # if step < 20:
            #     action = [.2, .5, .3]
            # else:
            #     action = [.15, -.1, -.2]
            with utils.eval_mode(agent):
                action = agent.sample_action(obs.copy())
            step+=1
            obs_traj.append(obs)
            obs_dict, reward, done, info = env.step(action)
            video.record(env)

        video.save('conv405000_replay_cab_{}_{}.mp4'.format(name, i))


#run_eval_loop(env_sim, 'sim')
run_eval_loop(env_real, 'real')