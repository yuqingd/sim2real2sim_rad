from env_wrapper import make
from video import VideoRecorder
import os
import utils
from train import parse_args, make_agent
import torch
# ENV = 'FetchPickAndPlace-v1'
env = make('kitchen', real_world=True,
        task_name='rope',
        seed=0,
        from_pixels=True,
        height=84,
        width=84
    )
# env.set_special_reset('grip')
env.reset()
num_episodes = 1
time_limit = 60
image_size = 84
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
                action = [0, .5, .6]
            elif step < 40:
                action = [-.1, .5, 0]
            else:
                action = [-.4, .5, -.3]
            step+=1
            obs_traj.append(obs)
            obs_dict, reward, done, info = env.step(action)
            video.record(env)
            episode_reward += reward
            print(reward)
        video.save('real.mp4')

run_eval_loop()