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

from dr import config_dr
import re


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='metaworld')
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
    parser.add_argument('--agent', default='curl_sac', type=str)
    parser.add_argument('--init_steps_policy', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=2000, type=int)
    parser.add_argument('--num_eval_episodes', default=3, type=int)
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

    # S2R2S params
    parser.add_argument('--mean_only', default=True, action='store_true')
    parser.add_argument('--state_type', default='robot', choices=['none', 'robot', 'all'])
    parser.add_argument('--use_img', default=False, action='store_true')
    parser.add_argument('--grayscale', default=False, action='store_true')
    parser.add_argument('--dr', action='store_true')
    parser.add_argument('--dr_option', default=None, type=str)
    parser.add_argument('--simple_randomization', default=False, type=bool)
    parser.add_argument('--mass_mean', default=.2, type=float)
    parser.add_argument('--mass_range', default=.01, type=float)
    parser.add_argument('--mean_scale', default=.67, type=float)
    parser.add_argument('--range_scale', default=.1, type=float)
    parser.add_argument('--range_only', default=False, type=bool)
    parser.add_argument('--sim_param_lr', default=1e-3, type=float)
    parser.add_argument('--sim_param_beta', default=0.9, type=float)
    parser.add_argument('--sim_param_layers', default=2, type=int)
    parser.add_argument('--sim_param_units', default=400, type=int)
    parser.add_argument('--separate_trunks', default=False, action='store_true')
    parser.add_argument('--train_range_scale', default=1, type=float)
    parser.add_argument('--scale_large_and_small', default=False, action='store_true')
    parser.add_argument('--num_sim_param_updates', default=1, type=int)
    parser.add_argument('--state_concat', default=False, action='store_true')
    parser.add_argument('--prop_initial_range', default=False, action='store_true')
    parser.add_argument('--prop_range_scale', default=False, action='store_true')
    parser.add_argument('--prop_train_range_scale', default=False, action='store_true')
    parser.add_argument('--prop_alpha', default=False, action='store_true')
    parser.add_argument('--clip_positive', default=False, action='store_true')
    parser.add_argument('--update_sim_param_from', choices=['latest', 'buffer', 'both'], type=str.lower,
                        default='latest')

    # Outer loop options
    parser.add_argument('--sample_real_every', default=2, type=int)
    parser.add_argument('--num_real_world', default=1, type=int)
    parser.add_argument('--anneal_range', default=False, action='store_true')
    parser.add_argument('--anneal_range_scale_start', default=1., type=float)
    parser.add_argument('--anneal_range_scale_end', default=.1, type=float)
    parser.add_argument('--predict_val', default=True, type=bool)
    parser.add_argument('--outer_loop_version', default=0, type=int, choices=[0, 1, 3])
    parser.add_argument('--alpha', default=.1, type=float)
    parser.add_argument('--sim_params_size', default=0, type=int)
    parser.add_argument('--ol1_episodes', default=10, type=int)
    parser.add_argument('--binary_prediction', default=False, type=bool)
    parser.add_argument('--train_sim_param_every', default=1, type=int)
    parser.add_argument('--momentum', default=0, type=float)
    parser.add_argument('--round_predictions', default=True, action='store_true')
    parser.add_argument('--single_window', default=False, action='store_true')
    parser.add_argument('--no_train_policy', default=False, action='store_true')
    parser.add_argument('--share_encoder', default=False, action='store_true',
                        help="use agent encoder for sim param model")
    parser.add_argument('--normalize_features', default=False, action='store_true')
    parser.add_argument('--use_layer_norm', default=False, action='store_true')
    parser.add_argument('--weight_init', default=False, action='store_true')
    parser.add_argument('--frame_skip', default=1, type=int)
    parser.add_argument('--spatial_softmax_agent', default=False, action='store_true')
    parser.add_argument('--spatial_softmax_sp', default=False, action='store_true')
    parser.add_argument('--num_frames', default=10, type=int)
    parser.add_argument('--alternate_training', default=False, action='store_true')
    parser.add_argument('--sp_itrs', default=10000, type=int, help='only used with alternate_training')
    parser.add_argument('--policy_itrs', default=40000, type=int, help='only used with alternate_training')
    parser.add_argument('--range_scale_sp', default=1., type=float)
    parser.add_argument('--update_sp_itrs', default=80000, type=int, help='only used with alternate_training')
    parser.add_argument('--collect_sp_itrs', default=50000, type=int, help='only used with alternate_training')
    parser.add_argument('--pretrain_sp_itrs', default=50000, type=int, help='only used with alternate_training')

    # MISC
    parser.add_argument('--id', default='debug', type=str)
    parser.add_argument('--gpudevice', type=str, required=True, help='cuda visible devices')
    parser.add_argument('--time_limit', default=200, type=int)
    parser.add_argument('--delay_steps', default=0, type=int)
    parser.add_argument('--full_screen_square', default=False, action='store_true')
    parser.add_argument('--train_offline_dir', default=None, type=str)
    parser.add_argument('--val_split', default=.2, type=float,
                        help='validation split; currently only for offline training but we should fix this.')
    parser.add_argument('--continue_train', default=False, action='store_true')

    args = parser.parse_args()
    if args.dr:
        args = config_dr(args)
    else:
        args.real_dr_list = []
        args.real_dr_params = {}
        args.dr = None
    args.update = [0] * len(args.real_dr_list)
    if args.alternate_training:
        # Init_steps is where the policy starts updating (after the pretraining phase)
        args.init_steps_policy = args.init_steps_policy + args.collect_sp_itrs + args.pretrain_sp_itrs + args.update_sp_itrs

    return args


def train_offline(args, L, real_env, sim_env, agent, sim_param_model, video_real, video_sim, replay_buffer):
    train_sim_model_time = 0
    eval_time = 0
    start_step = 0
    for step in range(start_step, args.num_train_steps, 200):
        should_log = step % args.eval_freq == 0
        if should_log:
            total_time = train_sim_model_time + eval_time
            if total_time > 0:
                L.log('eval_time/train_sim_model', train_sim_model_time / total_time, step)
                L.log('eval_time/eval', eval_time / total_time, step)

            start_eval = time.time()
            evaluate(real_env, sim_env, agent, sim_param_model, video_real, video_sim,
                     args.num_eval_episodes, L, step, args, True, 'both')

            sim_param_model.update(None, None, sim_env.distribution_mean,
                                   L, step, should_log, replay_buffer, val=True, tag="train_val")
        eval_time += time.time() - start_eval
        L.dump(step)
    start_sim_model = time.time()
    sim_param_model.update(None, None, sim_env.distribution_mean,
                           L, step, should_log, replay_buffer)
    train_sim_model_time += time.time() - start_sim_model


def evaluate_sim_params(sim_param_model, args, obs, step, L, prefix, real_dr_params, current_sim_params):
    if len(real_dr_params) == 0:
        return

    with torch.no_grad():
        if args.outer_loop_version == 1:
            pred_sim_params = sim_param_model.forward(obs).mean[0].cpu().numpy()
        elif args.outer_loop_version == 3:
            pred_sim_params = []
            if len(obs) > 1:
                for ob in obs:
                    pred_sim_params.append(predict_sim_params(sim_param_model, ob, current_sim_params, args))
            else:
                pred_sim_params.append(predict_sim_params(sim_param_model, obs[0], current_sim_params, args))
            pred_sim_params = np.mean(pred_sim_params, axis=0)

            real_dr_params = (current_sim_params[0].cpu().numpy() > real_dr_params).astype(np.int32)

        for i, param in enumerate(args.real_dr_list):
            filename = args.work_dir + f'/{prefix}_{param}_error.npy'
            key = args.domain_name + '-' + str(args.task_name) + '-' + args.data_augs
            try:
                log_data = np.load(filename, allow_pickle=True)
                log_data = log_data.item()
            except FileNotFoundError:
                log_data = {}
            if key not in log_data:
                log_data[key] = {}
            log_data[key][step] = {}

            if args.outer_loop_version == 1:
                try:
                    pred_mean = pred_sim_params[i]
                except:
                    pred_mean = pred_sim_params
                real_dr_param = real_dr_params[i]

                if not np.mean(real_dr_param) == 0:
                    error = (pred_mean - real_dr_param) / real_dr_param
                else:
                    error = pred_mean - real_dr_param
            elif args.outer_loop_version == 3:
                try:
                    pred_mean = pred_sim_params[i]
                except:
                    pred_mean = pred_sim_params
                real_dr_param = real_dr_params[i]
                try:
                    assert real_dr_param >= 0
                    assert real_dr_param <= 1
                    assert pred_mean >= 0
                    assert pred_mean <= 1
                except:
                    print("PRED MEAN", pred_mean)
                    print("REAL DR", real_dr_param)
                    import IPython
                    IPython.embed()
                error = np.mean(pred_mean - real_dr_param)
            accuracy = np.round(pred_mean) == real_dr_param
            loss = torch.nn.BCELoss()(torch.FloatTensor([pred_mean]), torch.FloatTensor([real_dr_param]))

            L.log(f'eval/{prefix}/{param}/error', error, step)
            L.log(f'eval/{prefix}/{param}/accuracy', accuracy, step)
            L.log(f'eval/{prefix}/{param}/loss', loss, step)
            log_data[key][step]['error'] = error
            log_data[key][step]['accuracy'] = accuracy
            log_data[key][step]['loss'] = loss
            np.save(filename, log_data)


def predict_sim_params(sim_param_model, traj, current_sim_params, args, step=10, confidence_level=.3):
    segment_length = sim_param_model.num_frames
    windows = []
    index = 0
    while index <= len(traj) - segment_length * args.frame_skip:
        windows.append(traj[index: index + segment_length * args.frame_skip])
        index += step
    if args.single_window:
        windows = [windows[0]]

    if args.round_predictions:
        with torch.no_grad():
            preds = sim_param_model.forward_classifier(windows, current_sim_params).cpu().numpy()
            mask = (preds > confidence_level) & (preds < 1 - confidence_level)
            preds = np.round(preds)
            preds[mask] = 0.5

    # Round to the nearest integer so each prediction is voting up or down
    # Alternatively, we could just take a mean of their probabilities
    # The only difference is whether we want to give each confident segment equal weight or not
    # And whether we want to be confident (e.g. if all windows predict .6, do we predict .6 or 1?
    confident_preds = np.mean(preds, axis=0)
    return confident_preds


def update_sim_params(sim_param_model, sim_env, args, obs, step, L):
    with torch.no_grad():
        if args.outer_loop_version == 1:
            pred_sim_params = sim_param_model.forward(obs).mean
            pred_sim_params = pred_sim_params[0].cpu().numpy()
        elif args.outer_loop_version == 3:
            current_sim_params = torch.FloatTensor(sim_env.distribution_mean).unsqueeze(0)
            pred_sim_params = []
            if len(obs) > 1:
                for ob in obs:
                    pred_sim_params.append(predict_sim_params(sim_param_model, ob, current_sim_params, args))
            else:
                pred_sim_params.append(predict_sim_params(sim_param_model, obs[0], current_sim_params, args))
            pred_sim_params = np.mean(pred_sim_params, axis=0)

    updates = []
    for i, param in enumerate(args.real_dr_list):
        prev_mean = sim_env.dr[param]

        try:
            pred_mean = pred_sim_params[i]
        except:
            pred_mean = pred_sim_params
        alpha = args.alpha

        if args.outer_loop_version == 1:
            new_mean = prev_mean * (1 - alpha) + alpha * pred_mean
        elif args.outer_loop_version == 3:
            if args.prop_alpha:
                scale_factor = max(prev_mean, args.alpha)
                new_update = - alpha * (np.mean(pred_mean) - 0.5) * scale_factor
            else:
                new_update = - alpha * (np.mean(pred_mean) - 0.5)
            curr_update = args.momentum * args.update[i] + (1 - args.momentum) * new_update
            new_mean = prev_mean + curr_update
            updates.append(curr_update)

        new_mean = max(new_mean, 1e-3)
        sim_env.dr[param] = new_mean

        filename = args.work_dir + f'/agent-sim-params_{param}.npy'
        key = args.domain_name + '-' + str(args.task_name) + '-' + args.data_augs
        try:
            log_data = np.load(filename, allow_pickle=True)
            log_data = log_data.item()
        except FileNotFoundError:
            log_data = {}
        if key not in log_data:
            log_data[key] = {}
        log_data[key][step] = {}

        print("NEW MEAN", param, new_mean, step, pred_mean, "!" * 30)
        L.log(f'eval/agent-sim_param/{param}/mean', new_mean, step)
        L.log(f'eval/agent-sim_param/{param}/pred_mean', pred_mean, step)
        log_data[key][step]['mean'] = new_mean
        log_data[key][step]['pred_mean'] = pred_mean

        real_dr_param = args.real_dr_params[param]
        if not np.mean(real_dr_param) == 0:
            sim_param_error = (new_mean - real_dr_param) / real_dr_param
        else:
            sim_param_error = new_mean - real_dr_param
        L.log(f'eval/agent-sim_param/{param}/sim_param_error', sim_param_error, step)
        log_data[key][step]['sim_param_error'] = sim_param_error
        np.save(filename, log_data)

        # Update range scale, if necessary
        if args.anneal_range:
            proportion_through_training = float(step / args.num_train_steps)
            start = args.anneal_range_scale_start
            end = args.anneal_range_scale_end
            new_range_scale = start * (1 - proportion_through_training) + end * proportion_through_training
            print("Updating range scale to", new_range_scale)
            sim_env.set_range_scale(new_range_scale)
            sim_param_model.train_range_scale = new_range_scale

    args.updates = updates


def evaluate(real_env, sim_env, agent, sim_param_model, video_real, video_sim, num_episodes, L, step, args, use_policy,
             update_distribution, training_phase):
    log_reward = training_phase in ['both', 'policy']
    all_ep_rewards = []
    all_ep_success = []

    def run_eval_loop(sample_stochastically=False):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        obs_batch = []
        real_sim_params = real_env.reset()['sim_params']
        for i in range(num_episodes):
            obs_dict = real_env.reset()
            video_real.init(enabled=(i == 0))
            video_real.record(real_env)
            done = False
            episode_reward = 0
            obs_traj = []
            while not done and len(obs_traj) < args.time_limit:
                obs_img = obs_dict['image']
                # center crop image
                if (args.agent == 'curl_sac' and args.encoder_type == 'pixel') or (
                    args.agent == 'rad_sac' and (args.encoder_type == 'pixel' or 'crop' in args.data_augs)):
                    obs_img = utils.center_crop_image(obs_img, args.image_size)
                obs_state = obs_dict['state']
                with utils.eval_mode(agent):
                    if not use_policy:
                        action = sim_env.action_space.sample()
                    elif sample_stochastically:
                        action = agent.sample_action(obs_img)
                    else:
                        action = agent.select_action(obs_img)
                obs_traj.append((obs_img, obs_state, action))
                obs_dict, reward, done, _ = real_env.step(action)
                video_real.record(real_env)
                episode_reward += reward

            video_real.save(f'real_{training_phase}_{step}.mp4')
            if log_reward:
                L.log('eval/' + prefix + f'episode_reward', episode_reward, step)
            if 'success' in obs_dict.keys():
                if log_reward:
                    L.log('eval/' + prefix + 'episode_success', obs_dict['success'], step)
                all_ep_success.append(obs_dict['success'])
            all_ep_rewards.append(episode_reward)
            obs_batch.append(obs_traj)
        if (not args.outer_loop_version == 0) and update_distribution:
            current_sim_params = torch.FloatTensor([sim_env.distribution_mean])
            evaluate_sim_params(sim_param_model, args, obs_batch, step, L, "test", real_sim_params,
                                current_sim_params)
            update_sim_params(sim_param_model, sim_env, args, obs_batch, step, L)

        if log_reward:
            L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
            mean_ep_reward = np.mean(all_ep_rewards)
            best_ep_reward = np.max(all_ep_rewards)
            std_ep_reward = np.std(all_ep_rewards)
            L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
            L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)
            if len(all_ep_success) > 0:
                mean_ep_success = np.mean(all_ep_success)
                L.log('eval/' + prefix + 'mean_episode_success', mean_ep_success, step)

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
        if log_reward:
            log_data[key][step]['mean_ep_reward'] = mean_ep_reward
            log_data[key][step]['max_ep_reward'] = best_ep_reward
            log_data[key][step]['std_ep_reward'] = std_ep_reward
        log_data[key][step]['env_step'] = step * args.action_repeat
        if log_reward and len(all_ep_success) > 0:
            log_data[key][step]['mean_ep_success'] = mean_ep_success

        np.save(filename, log_data)

        obs_dict = sim_env.reset()
        done = False
        obs_traj_sim = []
        video_sim.init(enabled=True)
        video_sim.record(sim_env)
        while not done and len(obs_traj_sim) < args.time_limit:
            obs_img = obs_dict['image']
            # center crop image
            if (args.agent == 'curl_sac' and args.encoder_type == 'pixel') or (
                args.agent == 'rad_sac' and (args.encoder_type == 'pixel' or 'crop' in args.data_augs)):
                obs_img = utils.center_crop_image(obs_img, args.image_size)
            obs_state = obs_dict['state']
            with utils.eval_mode(agent):
                if not use_policy:
                    action = sim_env.action_space.sample()
                elif sample_stochastically:
                    action = agent.sample_action(obs_img)
                else:
                    action = agent.select_action(obs_img)
            obs_traj_sim.append((obs_img, obs_state, action))
            obs_dict, reward, done, _ = sim_env.step(action)

            video_sim.record(sim_env)
            sim_params = obs_dict['sim_params']
        if update_distribution and sim_param_model is not None:
            current_sim_params = torch.FloatTensor([sim_env.distribution_mean])
            evaluate_sim_params(sim_param_model, args, [obs_traj_sim], step, L, "val", sim_params, current_sim_params)

        video_sim.save(f'sim_{training_phase}_%d.mp4' % step)

    run_eval_loop(sample_stochastically=True)
    L.dump(step)


def make_agent(obs_shape, state_shape, action_shape, args, device):
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
            latent_dim=args.latent_dim,
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
            data_augs=args.data_augs,
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def is_training_phase(timestep, pretrain_steps, sp_itrs, policy_itrs):
    target = pretrain_steps
    policy_steps = 0
    # Loop through timesteps, alternating between phases
    if timestep == 0:
        return False, target, 0
    t = 0
    is_policy_phase = False
    while t < timestep:
        if t == 0:
            t += pretrain_steps
        else:
            if is_policy_phase:
                policy_steps += min(policy_itrs, timestep - t)
                t += policy_itrs
                target += policy_itrs
            else:
                t += sp_itrs
                target += sp_itrs
        is_policy_phase = not is_policy_phase
    return not is_policy_phase, target, policy_steps


def main():
    args = parse_args()
    os.environ['EGL_DEVICE_ID'] = args.gpudevice
    if 'dmc' or 'kitchen' in args.domain_name:
        os.environ['MUJOCO_GL'] = 'egl'
    else:
        os.environ['MUJOCO_GL'] = 'osmesa'
    import env_wrapper

    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    sim_env = env_wrapper.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        frame_skip=args.action_repeat,
        mean_only=args.mean_only,
        dr_list=args.real_dr_list,
        simple_randomization=args.dr_option == 'simple',
        dr_shape=args.sim_params_size,
        real_world=False,
        dr=args.dr,
        state_type=args.state_type,
        grayscale=args.grayscale,
        delay_steps=args.delay_steps,
        range_scale=args.range_scale,
        prop_range_scale=args.prop_range_scale,
        prop_initial_range=args.prop_initial_range,
        state_concat=args.state_concat,
        real_dr_params=None,
        time_limit=args.time_limit,
        full_screen_square=args.full_screen_square,
    )

    real_env = env_wrapper.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        frame_skip=args.action_repeat,
        dr_list=args.real_dr_list,
        dr_shape=args.sim_params_size,
        mean_only=args.mean_only,
        real_world=True,
        state_type=args.state_type,
        grayscale=args.grayscale,
        delay_steps=args.delay_steps,
        range_scale="NONE",
        prop_range_scale=args.prop_range_scale,
        state_concat=args.state_concat,
        real_dr_params=args.real_dr_params,
        time_limit=args.time_limit,
        full_screen_square=args.full_screen_square,
    )

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
    exp_name = env_name + '-im' + str(args.image_size) + '-b' + str(args.batch_size)
    exp_name += '-s' + str(args.seed) + '-' + args.agent + '-' + args.encoder_type + '-' + args.data_augs
    args.work_dir = args.work_dir + '/' + args.id + '_' + exp_name

    load_model = False
    if os.path.exists(os.path.join(args.work_dir, 'model')):
        print("Loading checkpoint...")
        load_model = True
        checkpoints = os.listdir(os.path.join(args.work_dir, 'model'))
        if args.alternate_training:
            buffer_sp = os.listdir(os.path.join(args.work_dir, 'buffer_sp'))
            buffer = os.listdir(os.path.join(args.work_dir, 'buffer_policy'))
        else:
            buffer = os.listdir(os.path.join(args.work_dir, 'buffer'))
        if len(checkpoints) == 0:  # Always load checkpoints.  It's now on the user to make sure any needed buffers are there
            print("No checkpoints found")
            load_model = False  # if we're continuing training, we can load model even w/o buffer
        else:
            agent_checkpoint = [f for f in checkpoints if 'curl' in f]
            if args.outer_loop_version in [1, 3]:
                sim_param_checkpoint = [f for f in checkpoints if 'sim_param' in f]

    utils.make_dir(args.work_dir)
    sim_video_dir = utils.make_dir(os.path.join(args.work_dir, 'sim_video'))
    real_video_dir = utils.make_dir(os.path.join(args.work_dir, 'real_video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    sim_env_dir = os.path.join(args.work_dir, "sim_env_data.pkl")
    real_env_dir = os.path.join(args.work_dir, "real_env_data.pkl")
    if args.alternate_training:
        buffer_dir_sp = utils.make_dir(os.path.join(args.work_dir, 'buffer_sp'))
        buffer_dir_policy = utils.make_dir(os.path.join(args.work_dir, 'buffer_policy'))
    else:
        if args.train_offline_dir is None:
            buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))
        else:
            buffer_dir = utils.make_dir(os.path.join(args.train_offline_dir, 'buffer'))

    video_real = VideoRecorder(real_video_dir if args.save_video else None, camera_id=args.cameras[0])
    video_sim = VideoRecorder(sim_video_dir if args.save_video else None, camera_id=args.cameras[0])

    # with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
    #     json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = sim_env.action_space.shape

    if args.encoder_type == 'pixel':
        cpf = 3 * len(args.cameras)
        obs_shape = (cpf * args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (cpf * args.frame_stack, args.pre_transform_image_size, args.pre_transform_image_size)
        state_shape = sim_env.reset()['state'].shape[0]
    else:
        obs_shape = sim_env.reset()['state'].shape
        pre_aug_obs_shape = obs_shape
        state_shape = 1  # null value

    if args.alternate_training:
        replay_buffer_sp = utils.ReplayBuffer(
            example_obs=sim_env.reset(),
            action_shape=action_shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device,
            image_size=args.image_size,
            max_traj_length=args.time_limit,
            val_split=args.val_split if (args.train_offline_dir is not None) else None,
        )
        replay_buffer_policy = utils.ReplayBuffer(
            example_obs=sim_env.reset(),
            action_shape=action_shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device,
            image_size=args.image_size,
            max_traj_length=args.time_limit,
            val_split=args.val_split if (args.train_offline_dir is not None) else None,
        )
    else:
        replay_buffer = utils.ReplayBuffer(
            example_obs=sim_env.reset(),
            action_shape=action_shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device,
            image_size=args.image_size,
            max_traj_length=args.time_limit,
            val_split=args.val_split if (args.train_offline_dir is not None) else None,
        )

    agent = make_agent(
        obs_shape=obs_shape,
        state_shape=state_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )
    if args.outer_loop_version in [1, 3]:
        dist = 'binary' if args.outer_loop_version == 3 else 'normal'
        if args.prop_initial_range:
            initial_range = sim_env.get_dr() * args.train_range_scale
        else:
            initial_range = None
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
            batch_size=args.batch_size,
            sim_param_lr=args.sim_param_lr,
            sim_param_beta=args.sim_param_beta,
            dist=dist,
            traj_length=args.time_limit,
            use_img=args.use_img,
            state_dim=state_shape,
            separate_trunks=args.separate_trunks,
            param_names=args.real_dr_list,
            train_range_scale=args.train_range_scale,
            prop_train_range_scale=args.prop_train_range_scale,
            initial_range=initial_range,
            clip_positive=args.clip_positive,
            action_space=sim_env.action_space,
            single_window=args.single_window,
            share_encoder=args.share_encoder,
            normalize_features=args.normalize_features,
            use_layer_norm=args.use_layer_norm,
            use_weight_init=args.weight_init,
            frame_skip=args.frame_skip,
            spatial_softmax=args.spatial_softmax_sp,
            num_frames=args.num_frames,
        ).to(device)
        # Use the same encoder for the agent and the sim param model
        if args.share_encoder:
            sim_param_model.encoder.copy_conv_weights_from(agent.critic.encoder)
    else:
        sim_param_model = None

    start_step = 0
    start_policy_step = 0
    if args.alternate_training:
        training_phase = 'sp'
        replay_buffer = replay_buffer_sp
        sim_env.set_range_scale(args.range_scale_sp)
        target_step = args.collect_sp_itrs + args.pretrain_sp_itrs + args.update_sp_itrs
    elif args.no_train_policy:
        training_phase = 'sp'
        args.init_steps_policy = args.collect_sp_itrs + args.pretrain_sp_itrs
    else:
        training_phase = 'both'

    # If we're continuing training, load the envs regardless of whether we load a model
    if args.continue_train:
        print("loading envs!")
        sim_env.load(sim_env_dir)
        real_env.load(real_env_dir)

    if load_model:
        agent_step = 0
        for checkpoint in agent_checkpoint:
            agent_step = max(agent_step, [int(x) for x in re.findall('\d+', checkpoint)][-1])
        agent.load(model_dir, agent_step)
        agent.load_curl(model_dir, agent_step)
        print("LOADING MODEL!")
    try:
        sim_env.load(sim_env_dir)
        real_env.load(real_env_dir)
    except Exception as e:
        print("No envs to load")

    try:
        start_step = agent_step
        if sim_param_model is not None:
            sim_param_step = 0
            for checkpoint in sim_param_checkpoint:
                sim_param_step = max(sim_param_step, [int(x) for x in re.findall('\d+', checkpoint)][-1])
            sim_param_model.load(model_dir, sim_param_step)
            if args.continue_train:
                start_step = 0
            else:
                start_step = min(start_step, sim_param_step)  # TODO: do we have to save optimizer?
        if args.alternate_training:
            # Find out how many of the steps we've been through are policy steps
            pretrain_steps = args.collect_sp_itrs + args.pretrain_sp_itrs + args.update_sp_itrs
            is_policy_phase, target_step, start_policy_step = is_training_phase(start_step, pretrain_steps,
                                                                                args.sp_itrs, args.policy_itrs)
            if is_policy_phase:
                training_phase = 'policy'
                sim_env.set_range_scale(args.range_scale)
                replay_buffer = replay_buffer_policy
            else:
                training_phase = 'sp'
                sim_env.set_range_scale(args.range_scale_sp)
                replay_buffer = replay_buffer_sp
        else:
            start_policy_step = start_step
        print(f"Loaded sp model! We're at {start_step} steps and {start_policy_step} policy steps.")
    except Exception as e:
        print("NO SP model to load")

    try:
        print("Loading replay buffer")
        if args.alternate_training:
            replay_buffer_sp.load(buffer_dir_sp)
            replay_buffer_policy.load(buffer_dir_policy)
            print(f"finished loading! We have {replay_buffer_sp.idx} steps in the sp buffer and {replay_buffer_policy.idx} in the policy buffer")
        else:
            replay_buffer.load(buffer_dir)
            print("finished loading! Number of steps: ", replay_buffer.idx)
    except:
        print("No replay buffer to load")

    L = Logger(args.work_dir, use_tb=args.save_tb)

    if args.train_offline_dir is not None:
        train_offline(args, L, real_env, sim_env, agent, sim_param_model, video_real, video_sim, replay_buffer)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    success = None
    obs_traj = None

    train_policy_time = 0
    train_sim_model_time = 0
    eval_time = 0
    collect_data_time = 0

    # Training phase specifies whether we train the agent, the policy, or both
    num_train_policy_steps = args.num_train_steps
    step = start_step
    policy_step = start_policy_step

    eval_target_step = step - (step % args.eval_freq) + args.eval_freq
    print("Starting step: ", step)
    print("Starting policy step: ", policy_step)
    print("Starting eval target step", eval_target_step)
    print("Starting training phase", training_phase)
    print("Starting target step", target_step)

    while policy_step < num_train_policy_steps:

        if args.alternate_training:
            if step == 0:
                print(step, "============================= PHASE 0 - collect only ===============================")
            if step == args.collect_sp_itrs:
                print(step, "============================= PHASE 1 - collect and train sp only ===============================")
            if step == args.collect_sp_itrs + args.pretrain_sp_itrs:
                print(step, "============================= PHASE 2 - collect, train sp, and update dist only ===============================")
            if step == args.collect_sp_itrs + args.pretrain_sp_itrs + args.update_sp_itrs:
                print(step, "============================= PHASE 3 - alternate policy collect and sp collect+train+update ===============================")
            if step == args.init_steps_policy:
                print(step, "============================= PHASE 4 - alternate policy collect+train and sp collect+train+update ===============================")

        # evaluate agent periodically
        if step >= eval_target_step and done:
            eval_target_step += args.eval_freq
            total_time = train_policy_time + train_sim_model_time + eval_time + collect_data_time
            if total_time > 0:
                L.log('eval_time/train_policy', train_policy_time / total_time, step)
                L.log('eval_time/train_sim_model', train_sim_model_time / total_time, step)
                L.log('eval_time/eval', eval_time / total_time, step)
                L.log('eval_time/collect_data', collect_data_time / total_time, step)

            L.log('eval/episode', episode, step)

            start_eval = time.time()
            use_policy = step >= args.init_steps_policy
            if args.alternate_training:
                update_distribution = training_phase == 'sp' and (step >= args.collect_sp_itrs + args.pretrain_sp_itrs)
            else:
                update_distribution = step >= args.init_steps_policy
            evaluate(real_env, sim_env, agent, sim_param_model, video_real, video_sim,
                     args.num_eval_episodes, L, step, args, use_policy, update_distribution, training_phase)
            eval_time += time.time() - start_eval
            if args.save_model:
                print("SAVING MODEL!")
                agent.save(model_dir, step)
                agent.save_curl(model_dir, step)
                sim_env.save(sim_env_dir)
                real_env.save(real_env_dir)
                if sim_param_model is not None:
                    sim_param_model.save(model_dir, step)
            if args.save_buffer:
                if args.alternate_training:
                    replay_buffer_sp.save(buffer_dir_sp)
                    replay_buffer_policy.save(buffer_dir_policy)
                else:
                    replay_buffer.save(buffer_dir)

        # Alternate training phases
        if args.alternate_training:
            L.log('train/in_policy_phase', int(training_phase == 'policy'), step)
            if step == target_step:
                if training_phase == 'sp':
                    training_phase = 'policy'
                    replay_buffer = replay_buffer_policy
                    sim_env.set_range_scale(args.range_scale)
                    target_step += args.policy_itrs
                elif training_phase == 'policy':
                    training_phase = 'sp'
                    replay_buffer = replay_buffer_sp
                    sim_env.set_range_scale(args.range_scale_sp)
                    target_step += args.sp_itrs

        if done:
            if step > 0:
                should_update_sp = args.outer_loop_version != 0
                should_update_sp = should_update_sp and training_phase in ['both', 'sp']
                if args.alternate_training:
                    should_update_sp = should_update_sp and step >= args.collect_sp_itrs
                else:
                    should_update_sp = should_update_sp and step >= args.init_steps_policy
                if should_update_sp and obs_traj is not None:
                    should_log = step % (10 * args.eval_freq) == 0
                    start_sim_model = time.time()
                    for i in range(args.num_sim_param_updates):
                        should_log_i = should_log and i == 0
                        if args.update_sim_param_from in ['latest', 'both']:
                            sim_param_model.update(obs_traj, sim_params, sim_env.distribution_mean,
                                                   L, step, should_log_i)
                        if args.update_sim_param_from in ['buffer', 'both']:
                            sim_param_model.update(obs_traj, sim_params, sim_env.distribution_mean,
                                                   L, step, should_log_i, replay_buffer)
                    train_sim_model_time += time.time() - start_sim_model

                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                filename = args.work_dir + f'/train_reward.npy'
                key = args.domain_name + '-' + str(args.task_name) + '-' + args.data_augs
                try:
                    log_data = np.load(filename, allow_pickle=True)
                    log_data = log_data.item()
                except FileNotFoundError:
                    log_data = {}
                if key not in log_data:
                    log_data[key] = {}
                log_data[key][step] = {}

                if training_phase in ['policy', 'both']:
                    L.log('train/episode_reward', episode_reward, step)
                    log_data[key][step]['episode_reward'] = episode_reward
                    if success is not None:
                        L.log('train/episode_success', success, step)
                        log_data[key][step]['episode_success'] = success

                np.save(filename, log_data)

            collect_data_start = time.time()
            obs = sim_env.reset()
            sim_params = obs['sim_params']
            obs_img = obs['image']
            if (args.agent == 'curl_sac' and args.encoder_type == 'pixel') or (
                args.agent == 'rad_sac' and (args.encoder_type == 'pixel' or 'crop' in args.data_augs)):
                obs_img = utils.center_crop_image(obs_img, args.image_size)
            obs_state = obs['state']
            obs_traj = []
            success = 0.0 if 'success' in obs.keys() else None
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)
            collect_data_time += time.time() - collect_data_start

        # sample action for data collection
        train_policy_start = time.time()
        if args.no_train_policy or step < args.init_steps_policy:
            action = sim_env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs_img)

        # run training update
        if training_phase in ['both', 'policy'] and step >= args.init_steps_policy:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)
        train_policy_time += time.time() - train_policy_start

        collect_data_start = time.time()
        next_obs, reward, done, _ = sim_env.step(action)

        # allow infinite bootstrap
        done = True if episode_step >= args.time_limit - 1 else done
        done_bool = float(done)
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        obs_traj.append((obs_img, obs_state, action))

        if 'success' in obs.keys():
            success = obs['success']

        obs = next_obs
        obs_img = obs['image']
        obs_state = obs['state']

        if (args.agent == 'curl_sac' and args.encoder_type == 'pixel') or (
            args.agent == 'rad_sac' and (args.encoder_type == 'pixel' or 'crop' in args.data_augs)):
            obs_img = utils.center_crop_image(obs_img, args.image_size)

        episode_step += 1
        collect_data_time += time.time() - collect_data_start
        step += 1
        if training_phase in ['policy', 'both']:
            policy_step += 1


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    main()
