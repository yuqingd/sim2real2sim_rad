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
args = parse_args()
GlfwContext(offscreen=True)
os.environ['MUJOCO_GL'] = 'glfw'
env_real = make('kitchen', real_world=True,
        task_name='rope',
        seed=0,
        height=256,
        width=256,
        state_type="robot"
    )

env_sim = make('kitchen', real_world=False,
        task_name='rope',
        seed=0,
        height=84,
        width=84,
        state_type="robot",
        delay_steps=3
               )
env_sim = utils.FrameStack(env_sim, k=args.frame_stack)
env_real = utils.FrameStack(env_real, k=args.frame_stack)

env_real.reset()
env_sim.reset()
num_episodes = 1
time_limit = 60
image_size = 84
real_video_dir = utils.make_dir(os.path.join('./logdir', 'debug_video'))

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

args.work_dir = args.work_dir + '/ol31006_kitchen-rope-im84-b128-s1-curl_sac-pixel-crop'
model_dir =  os.path.join(args.work_dir, 'model')
agent.load(model_dir, 130000)
agent.load_curl(model_dir, 130000)

def run_eval_loop(env, name):
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
            ##CABINET
            if step < 20:
                action = [-.2, .3, -.2]
            else:
                action = [-.15, 0, 0]
            ##ROPE
            if step < 20:
                action = [.2, .5, .3]
            else:
                action = [.15, -.1, -.2]
            with utils.eval_mode(agent):
                action = agent.sample_action(obs.copy())
            step+=1
            obs_traj.append(obs)
            obs_dict, reward, done, info = env.step(action)
            video.record(env)
            episode_reward += reward
            print(reward)
        video.save('replay_rope_{}.mp4'.format(name))


run_eval_loop(env_sim, 'sim')
run_eval_loop(env_real, 'real')