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

args = parse_args()

env = make('kitchen', real_world=True,
        task_name='rope',
        seed=args.seed,
        visualize_reward=False,
        from_pixels=True,
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        frame_skip=1,
        mean_only=args.mean_only,
        dr_list=args.real_dr_list,
        simple_randomization=args.dr_option == 'simple',
        dr_shape=args.sim_params_size,
        dr=args.dr,
        use_state=args.use_state,
        use_img=args.use_img,
        grayscale=args.grayscale,
        delay_steps=args.delay_steps,
        range_scale=args.range_scale,
        prop_range_scale=args.prop_range_scale,
        state_concat=args.state_concat,
        real_dr_params=None,
    )
# env.set_special_reset('grip')
env.reset()
num_episodes = 10
time_limit = 200
image_size = 84
real_video_dir = utils.make_dir(os.path.join('./logdir', 'real_video'))

video = VideoRecorder(real_video_dir, camera_id=0)

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