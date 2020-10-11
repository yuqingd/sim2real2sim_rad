from env_wrapper import make
import matplotlib.pyplot as plt
import time
import numpy as np
from video import VideoRecorder
import os
import utils
from train import parse_args, make_agent
import torch
import re
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
env.reset()
num_episodes = 10
time_limit = 200
image_size = 84
real_video_dir = utils.make_dir(os.path.join('./logdir', 'real_video'))

video = VideoRecorder(real_video_dir, camera_id=0)

args = parse_args()
cpf = 3 * len(args.cameras)
obs_shape = (cpf * args.frame_stack, image_size, image_size)
action_shape = env.action_space.shape
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

agent = make_agent(
    obs_shape=obs_shape,
    action_shape=action_shape,
    args=args,
    device=device
)
agent_checkpoint = -1

model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
if os.path.exists(os.path.join(args.work_dir, 'model')):
    print("Loading checkpoint...")
    load_model = True
    checkpoints = os.listdir(os.path.join(args.work_dir, 'model'))
    buffer = os.listdir(os.path.join(args.work_dir, 'buffer'))
    if len(checkpoints) == 0 or len(buffer) == 0:
        print("No checkpoints found")
        load_model = False
    else:
        agent_checkpoint = [f for f in checkpoints if 'curl' in f]
        if args.outer_loop_version in [1, 3]:
            sim_param_checkpoint = [f for f in checkpoints if 'sim_param' in f]

agent_step = 0
if agent_checkpoint < 0:
    if args.outer_loop_version in [1, 3]:
        sim_param_checkpoint = [f for f in checkpoints if 'sim_param' in f]
    for checkpoint in agent_checkpoint:
        agent_step = max(agent_step, [int(x) for x in re.findall('\d+', checkpoint)][-1])
        if args.outer_loop_version in [1, 3]:
            sim_param_checkpoint = [f for f in checkpoints if 'sim_param' in f]
else:
    agent_step = agent_checkpoint

agent.load_curl(model_dir, agent_step)


def run_eval_loop(sample_stochastically=True):
    for i in range(num_episodes):
        obs_dict = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        obs_traj = []
        while not done and len(obs_traj) < time_limit:
            obs = obs_dict['image']
            # center crop image
            obs = utils.center_crop_image(obs, image_size)

            with utils.eval_mode(agent):
                if sample_stochastically:
                    action = agent.sample_action(obs)
                else:
                    action = agent.select_action(obs)
            obs_traj.append(obs)
            obs_dict, reward, done, _ = env.step(action)
            video.record(env)
            episode_reward += reward

        video.save('real_%d.mp4')


run_eval_loop()