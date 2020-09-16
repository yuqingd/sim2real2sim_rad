import os
os.environ['MUJOCO_GL'] = 'osmesa'

import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import copy

import utils
from logger import Logger
from video import VideoRecorder

from curl_sac import CurlSacAgent, RadSacAgent
from sim_param_model import SimParamModel
from torchvision import transforms

import env_wrapper


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default=None)
    parser.add_argument('--pre_transform_image_size', default=100, type=int)
    parser.add_argument('--cameras', nargs='+', default=[0], type=int)
    parser.add_argument('--observation_type', default='pixel')

    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='rad_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)  # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2,
                        type=int)  # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='./logdir', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')

    parser.add_argument('--data_augs', default='crop', type=str)

    parser.add_argument('--log_interval', default=100, type=int)

    #S2R2S params
    parser.add_argument('--mean_only', default=True, action='store_true')
    parser.add_argument('--use_state', default=False, action='store_true')
    parser.add_argument('--use_img', default=True, action='store_true')
    parser.add_argument('--grayscale', default=False, action='store_true')
    parser.add_argument('--dr', action='store_true')
    parser.add_argument('--dr_option', default=None, type=str)
    parser.add_argument('--simple_randomization', default=False, type=bool)
    parser.add_argument('--mass_mean', default=.2, type=float)
    parser.add_argument('--mass_range', default=.01, type=float)
    parser.add_argument('--mean_scale', default=.67, type=float)
    parser.add_argument('--range_scale', default=.33, type=float)
    parser.add_argument('--range_only', default=False, type=bool)
    parser.add_argument('--sim_param_lr', default=1e-3, type=float)
    parser.add_argument('--sim_param_beta', default=0.9, type=float)
    parser.add_argument('--sim_param_layers', default=2, type=float)
    parser.add_argument('--sim_param_units', default=400, type=float)

    # Outer loop options
    parser.add_argument('--sample_real_every', default=2, type=int)
    parser.add_argument('--num_real_world', default=1, type=int)
    parser.add_argument('--anneal_range_scale', default=1.0, type=float)
    parser.add_argument('--predict_val', default=True, type=bool)
    parser.add_argument('--outer_loop_version', default=0, type=int, choices=[0, 1])
    parser.add_argument('--alpha', default=.3, type=float)
    parser.add_argument('--sim_params_size', default=0, type=int)
    parser.add_argument('--ol1_episodes', default=10, type=int)
    parser.add_argument('--binary_prediction', default=False, type=bool)

    # MISC
    parser.add_argument('--offscreen', action='store_true')
    parser.add_argument('--id', default='debug', type=str)


    args = parser.parse_args()
    if args.dr:
        args = config_dr(args)
    else:
        args.real_dr_list = []
        args.dr = None

    return args


def config_dr(config):
  if config.dr_option == 'simple':
      if 'basketball' in config.task_name:
        config.real_dr_params = {
          "object_mass": .01
        }
        config.dr = {  # (mean, range)
          "object_mass": (config.mass_mean, config.mass_range)
        }
        config.real_dr_list = ["object_mass"]
        config.sim_params_size = 1
      elif 'stick' in config.task_name or 'basketball' in config.task_name:
          config.real_dr_params = {
            "object_mass": .128  # TODO: ???
          }
          config.dr = {  # (mean, range)
            "object_mass": (config.mass_mean, config.mass_range)
          }
          config.real_dr_list = ["object_mass"]
          config.sim_params_size = 1
  elif config.dr_option == 'all_dr':
      real_dr_joint = {
        "table_friction": 2.,
        "table_r": .6,
        "table_g": .6,
        "table_b": .5,
        "robot_friction": 1.,
        "robot_r": .5,
        "robot_g": .1,
        "robot_b": .1,
      }
      if 'basketball' in config.task_name:
        config.real_dr_params = {
          "basket_friction": .5,
          "basket_goal_r": .5,
          "basket_goal_g": .5,
          "basket_goal_b": .5,
          "backboard_r": .5,
          "backboard_g": .5,
          "backboard_b": .5,
          "object_mass": .01,
          "object_friction": 1.,
          "object_r": 0.,
          "object_g": 0.,
          "object_b": 0.,
        }
        config.real_dr_params.update(real_dr_joint)
        config.real_dr_list = list(config.real_dr_params.keys())
      elif 'stick' in config.task_name:
        config.real_dr_params = {
          "stick_mass": 1.,
          "stick_friction": 1.,
          "stick_r": 1.,
          "stick_g": .3,
          "stick_b": .3,
          "object_mass": .128,
          "object_friction": 1.,
          "object_body_r": 0.,
          "object_body_g": 0.,
          "object_body_b": 1.,
          "object_handle_r": 0,
          "object_handle_g": 0,
          "object_handle_b": 0,
        }
        config.real_dr_params.update(real_dr_joint)
        config.real_dr_list = list(config.real_dr_params.keys())
      config.sim_params_size = len(config.real_dr_list)
      mean_scale = config.mean_scale
      range_scale = config.range_scale
      config.dr = {}  # (mean, range)
      for key, real_val in config.real_dr_params.items():
        if real_val == 0:
          real_val = 5e-2
        if config.mean_only:
          config.dr[key] = real_val * mean_scale
        else:
          config.dr[key] = (real_val * mean_scale, real_val * range_scale)
  else:
      config.dr = {}
      config.real_dr_params = {}
      config.dr_list = []

  for k, v in config.dr.items():
    print(k)
    print(v)

  if config.mean_only:
    config.initial_dr_mean = np.array([config.dr[param] for param in config.real_dr_list])
  else:
    config.initial_dr_mean = np.array([config.dr[param][0] for param in config.real_dr_list])
    config.initial_dr_range = np.array([config.dr[param][1] for param in config.real_dr_list])

  return config

def update_sim_params(sim_param_model, sim_env, args, obs, step, L):
    with torch.no_grad():
        pred_sim_params = sim_param_model.forward(obs).mean[0].cpu().numpy()

    for i, param in enumerate(args.real_dr_list):
        prev_mean = sim_env.dr[param]

        try:
            pred_mean = pred_sim_params[i]
        except:
            pred_mean = pred_sim_params
        alpha = args.alpha

        new_mean = prev_mean * (1 - alpha) + alpha * pred_mean
        sim_env.dr[param] = new_mean

        print("NEW MEAN", param, new_mean, step, pred_mean, "!" * 30)
        L.log(f'eval/agent-sim_param/{param}/mean', new_mean, step)
        L.log(f'eval/agent-sim_param/{param}/pred_mean', pred_mean, step)
        if args.anneal_range_scale > 0:
            L.log(f'eval/agent-sim_param/{param}/range', args.anneal_range_scale * (1 - float(step / args.num_train_steps)), step)

        real_dr_param = args.real_dr_params[param]
        if not np.mean(real_dr_param) == 0:
            L.log(f'eval/agent-sim_param/{param}/sim_param_error', (new_mean - real_dr_param) / real_dr_param, step)
        else:
            L.log(f'eval/agent-sim_param/{param}/sim_param_error', (new_mean - real_dr_param), step)


def evaluate(real_env, sim_env, agent, sim_param_model, video, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs_dict = real_env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            obs_traj = []
            while not done:
                obs = obs_dict['image']
                # center crop image
                if (args.agent == 'curl_sac' and args.encoder_type == 'pixel') or (args.agent == 'rad_sac' and (args.encoder_type == 'pixel' or 'crop' in args.data_augs)):
                    obs = utils.center_crop_image(obs, args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs_traj.append(obs)
                obs_dict, reward, done, _ = real_env.step(action)
                video.record(real_env)
                episode_reward += reward

            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)

        if args.outer_loop_version == 1:
            update_sim_params(sim_param_model, sim_env, args, obs_traj, step, L)

        L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        std_ep_reward = np.std(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

        filename = args.work_dir + '/eval_scores.npy'
        key = args.domain_name + '-' + str(args.task_name) + '-' + args.data_augs
        try:
            log_data = np.load(filename, allow_pickle=True)
            log_data = log_data.item()
        except FileNotFoundError:
            log_data = {}

        if key not in log_data:
            log_data[key] = {}

        log_data[key][step] = {}
        log_data[key][step]['step'] = step
        log_data[key][step]['mean_ep_reward'] = mean_ep_reward
        log_data[key][step]['max_ep_reward'] = best_ep_reward
        log_data[key][step]['std_ep_reward'] = std_ep_reward
        log_data[key][step]['env_step'] = step * args.action_repeat

        np.save(filename, log_data)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'curl_sac':
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            latent_dim=args.latent_dim
        )
    elif args.agent == 'rad_sac':
        return RadSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            latent_dim=args.latent_dim,
            data_augs=args.data_augs
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def main():
    args = parse_args()
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    sim_env = env_wrapper.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.observation_type == 'pixel'),
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        frame_skip=args.action_repeat,
        mean_only=args.mean_only,
        dr_list=args.real_dr_list,
        simple_randomization=args.dr_option == 'simple',
        dr_shape=args.sim_params_size,
        real_world=False,
        dr=args.dr,
        use_state=args.use_state,
        use_img=args.use_img,
        grayscale=args.grayscale,
        offscreen=args.offscreen,
    )


    real_env = env_wrapper.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.observation_type == 'pixel'),
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        frame_skip=args.action_repeat,
        dr_list=args.real_dr_list,
        dr_shape=args.sim_params_size,
        mean_only=args.mean_only,
        real_world=True,
        use_state=args.use_state,
        use_img=args.use_img,
        grayscale=args.grayscale,
        offscreen=args.offscreen,
    )

    sim_env.seed(args.seed)
    real_env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        sim_env = utils.FrameStack(sim_env, k=args.frame_stack)
        real_env = utils.FrameStack(real_env, k=args.frame_stack)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    if args.task_name is None:
        env_name = args.domain_name
    else:
        env_name = args.domain_name + '-' + args.task_name
    exp_name = env_name + '-' + ts + '-im' + str(args.image_size) + '-b' + str(args.batch_size)
    exp_name += '-s' + str(args.seed) + '-' + args.agent + '-' + args.encoder_type + '-' + args.data_augs
    args.work_dir = args.work_dir + '/' + args.id + '_' + exp_name

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None, camera_id=args.cameras[0])

    # with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
    #     json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = sim_env.action_space.shape

    if args.encoder_type == 'pixel':
        cpf = 3 * len(args.cameras)
        obs_shape = (cpf * args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (cpf * args.frame_stack, args.pre_transform_image_size, args.pre_transform_image_size)
    else:
        obs_shape = sim_env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        example_obs=sim_env.reset(),
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )
    if args.outer_loop_version == 1:
        sim_param_model = SimParamModel(
            shape=args.sim_params_size,
            layers=args.sim_param_layers,
            units=args.sim_param_units,
            device=device,
            obs_shape=obs_shape,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_num_layers=args.num_layers,
            encoder_num_filters=args.num_filters,
            agent=agent,
            sim_param_lr=args.sim_param_lr,
            sim_param_beta=args.sim_param_beta,
        ).to(device)
    else:
        sim_param_model = None

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(args.num_train_steps):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(real_env, sim_env, agent, sim_param_model, video, args.num_eval_episodes, L, step, args)
            if args.save_model:
                agent.save_curl(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = sim_env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)
        episode_step = 0

        # sample action for data collection
        if step < args.init_steps:
            action = sim_env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs['image'])

        # run training update
        if step >= args.init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)
            if step % 50 == 0 and args.outer_loop_version == 1:  # TODO: update?
                sim_param_model.update(replay_buffer, L, step, True) #TODO: change update freq if needed


        next_obs, reward, done, _ = sim_env.step(action)

        # allow infinite bootstrap
        done_bool = 0 if episode_step + 1 == sim_env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    main()
