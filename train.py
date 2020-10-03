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
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int)
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
    parser.add_argument('--save_model', default=True, action='store_true')
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
    parser.add_argument('--range_scale', default=.1, type=float)
    parser.add_argument('--range_only', default=False, type=bool)
    parser.add_argument('--sim_param_lr', default=1e-3, type=float)
    parser.add_argument('--sim_param_beta', default=0.9, type=float)
    parser.add_argument('--sim_param_layers', default=2, type=float)
    parser.add_argument('--sim_param_units', default=400, type=float)


    # Outer loop options
    parser.add_argument('--sample_real_every', default=2, type=int)
    parser.add_argument('--num_real_world', default=1, type=int)
    parser.add_argument('--anneal_range_scale', default=0, type=float)
    parser.add_argument('--predict_val', default=True, type=bool)
    parser.add_argument('--outer_loop_version', default=0, type=int, choices=[0, 1, 3])
    parser.add_argument('--alpha', default=.1, type=float)
    parser.add_argument('--sim_params_size', default=0, type=int)
    parser.add_argument('--ol1_episodes', default=10, type=int)
    parser.add_argument('--binary_prediction', default=False, type=bool)
    parser.add_argument('--start_outer_loop', default=0, type=int)
    parser.add_argument('--train_sim_param_every', default=50, type=int)


    # MISC
    parser.add_argument('--id', default='debug', type=str)
    parser.add_argument('--gpudevice', type=str, required=True, help='cuda visible devices')
    parser.add_argument('--time_limit', default=200, type=float)
    parser.add_argument('--delay_steps', default=0, type=int)

    args = parser.parse_args()
    if args.dr:
        args = config_dr(args)
    else:
        args.real_dr_list = []
        args.dr = None

    return args


def evaluate_sim_params(sim_param_model, args, obs, step, L, prefix, real_dr_params, current_sim_params):
    with torch.no_grad():
        if args.outer_loop_version == 1:
            pred_sim_params = sim_param_model.forward(obs).mean[0].cpu().numpy()
        elif args.outer_loop_version == 3:
            pred_sim_params = []
            if len(obs) > 1:
                for ob in obs:
                    pred_sim_params.append(predict_sim_params(sim_param_model, ob, current_sim_params))
            else:
                pred_sim_params.append(predict_sim_params(sim_param_model, obs[0], current_sim_params))
            pred_sim_params = np.mean(pred_sim_params, axis=0)

            real_dr_params = (current_sim_params[0].cpu().numpy() > real_dr_params).astype(np.int32)
            assert len(pred_sim_params.shape) == len(real_dr_params.shape) == 1

        preds = []

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

            L.log(f'eval/{prefix}/{param}/error', error, step)
            log_data[key][step]['error'] = error
            np.save(filename, log_data)
            preds.append(error)
        return preds

def predict_sim_params(sim_param_model, traj, current_sim_params, step=5, confidence_level=.3):
    segment_length = sim_param_model.num_frames
    windows = []
    index = 0
    while index < len(traj) - segment_length:
        windows.append(traj[index: index + segment_length])
        index += step
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


def predict_sim_params_range(sim_param_model, traj_list, current_sim_params, L, step, prefix, real_sim_params, alpha,
                             num_steps=10, confidence_level=.3, randomization_range=1):  # TODO: real_sim_params
    pred_list = []
    sim_param_value = current_sim_params[0].item()  # TODO: currently hard-coded for a single param.  If this works, generalize for arbitrary vectors
    min_val = max(1e-3, sim_param_value - randomization_range * sim_param_value)
    max_val = sim_param_value + randomization_range * sim_param_value
    sim_param_range = torch.FloatTensor(np.linspace(min_val, max_val, num_steps)).unsqueeze(1)
    for traj in traj_list:
        preds = sim_param_model.forward_classifier([traj], sim_param_range).cpu().numpy()
        pred_list.append(preds)

    def to_numpy(arr):
        return arr.detach().cpu().numpy()

    # For every index, log what percent predicted higher:
    pred_arr = np.stack(pred_list, axis=0)[:, :, 0]  # num_trajs by step
    for i in range(num_steps):
        L.log(f'eval/update_{prefix}/step{i}/mean_prob', np.mean(pred_arr, axis=0)[i], step)
        L.log(f'eval/update_{prefix}/step{i}/mean_std', np.std(pred_arr, axis=0)[i], step)
        L.log(f'eval/update_{prefix}/step{i}/mean_higher', np.mean(np.round(pred_arr), axis=0)[i], step)
        L.log(f'eval/update_{prefix}/step{i}/mean_higher_real', sim_param_range[i].item() > real_sim_params.item(), step)

        # Log the first one separately
        L.log(f'eval/update/step{i}/first_prob', pred_arr[0, i].item(), step)
        L.log(f'eval/update/step{i}/first_higher', np.round(pred_arr[0, i]).item(), step)

    all_same = []
    increasing_pred = []
    increasing_pred_rounded = []
    nondecreasing_pred_rounded = []
    split = []
    correct_split = []
    correct_at_current_sim = []
    correct_at_real_below = []
    correct_at_real_above = []
    totally_correct = []
    prediction_at_current_sim = []
    interpolated = []

    predictions = []


    # TODO: if this seems promising, batch it!
    sim_param_range = to_numpy(sim_param_range)[:, 0]
    for i, pred in enumerate(pred_list):
        pred = pred[:, 0]
        pred_rounded = np.round(pred)
        all_same.append( np.min(pred_rounded) == np.max(pred_rounded))
        increasing_pred.append(np.min(np.diff(pred)) > 0)
        increasing_pred_rounded.append(np.min(np.diff(pred_rounded)) > 0)
        nondecreasing_rounded = np.min(np.diff(pred_rounded)) >= 0
        nondecreasing_pred_rounded.append(nondecreasing_rounded)
        if nondecreasing_rounded:
            split_index = np.argmax(pred_rounded)
            split.append(split_index)
        desired_split = np.argmax(sim_param_range > real_sim_params.item())
        correct_split.append(desired_split)

        # check if we're actually taking that desired split
        chose_correct_split = split_index == desired_split
        totally_correct.append(chose_correct_split)

        # Is our prediction accurate around the current sim_param mean?
        current_sim_index = np.argmin(sim_param_range - sim_param_value)
        current_correct_at_sim = pred_rounded[current_sim_index] == (sim_param_value > real_sim_params.item())
        correct_at_current_sim.append(current_correct_at_sim)
        prediction_at_current_sim.append(pred_rounded[current_sim_index])

        # Is our prediction accurate around the real sim param mean?
        above = sim_param_range > real_sim_params.item()
        first_above = np.argmax(above)
        last_below = first_above - 1
        if np.min(above) == 1:  # the entire range is above
            correct_at_real_below.append(-1)
            assert first_above == 0
            correct_at_real_above.append(pred[first_above] == 1)
            interpolated.append(0)
        elif np.max(above) == 0:  # the entire range is below
            correct_at_real_below.append(pred[-1] == 0)
            correct_at_real_above.append(-1)
            interpolated.append(0)
        else:
            correct_at_real_below.append(pred_rounded[last_below] == 0)
            correct_at_real_above.append(pred_rounded[first_above] == 1)
            interpolated.append(1)

        if nondecreasing_rounded: # If there is a correct split, go there
            predictions.append(sim_param_range[split_index])

    # If we have any runs with a clean split, update based on that.  Otherwise, update using the original formula
    if len(predictions) > 0:
        update = np.mean(predictions)
        update_fancy = 0
    else:
        update_direction = np.mean(prediction_at_current_sim) - .5
        new_mean = current_sim_params - alpha * (update_direction - 0.5)
        update = max(new_mean, 1e-3)
        update_fancy = 1

    L.log(f'eval/update_{prefix}/update_fancy', update_fancy, step)
    L.log(f'eval/update_{prefix}/update', update, step)
    L.log(f'eval/update_{prefix}/all_same', np.mean(all_same), step)
    L.log(f'eval/update_{prefix}/increasing_pred', np.mean(increasing_pred), step)
    L.log(f'eval/update_{prefix}/increasing_pred_rounded', np.mean(increasing_pred_rounded), step)
    L.log(f'eval/update_{prefix}/split', np.mean(split), step)
    L.log(f'eval/update_{prefix}/correct_split', np.mean(correct_split), step)
    L.log(f'eval/update_{prefix}/correct_at_current_sim', np.mean(correct_at_current_sim), step)
    L.log(f'eval/update_{prefix}/correct_at_real_below', np.mean(correct_at_real_below), step)
    L.log(f'eval/update_{prefix}/correct_at_real_above', np.mean(correct_at_real_above), step)
    L.log(f'eval/update_{prefix}/totally_correct', np.mean(totally_correct), step)
    L.log(f'eval/update_{prefix}/prediction_at_current_sim', np.mean(prediction_at_current_sim), step)
    L.log(f'eval/update_{prefix}/interpolated', np.mean(interpolated), step)

    return update


def update_sim_params_exact(sim_param_model, sim_env, args, obs, step, L, real_sim_params):
    assert args.outer_loop_version == 3, "this version of updating is only defined for OL3"
    with torch.no_grad():
        current_sim_params = torch.FloatTensor(sim_env.distribution_mean).unsqueeze(0)
        pred_sim_params = predict_sim_params_range(sim_param_model, obs, current_sim_params, L, step, "test",
                                                   real_sim_params, args.alpha)

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
            scale_factor = max(prev_mean, .1)
            new_mean = pred_mean
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
        if args.anneal_range_scale > 0:
            range_value = args.anneal_range_scale * (1 - float(step / args.num_train_steps))
            L.log(f'eval/agent-sim_param/{param}/range', range_value, step)
            log_data[key][step]['range'] = range_value

        real_dr_param = args.real_dr_params[param]
        if not np.mean(real_dr_param) == 0:
            sim_param_error = (new_mean - real_dr_param) / real_dr_param
        else:
            sim_param_error = new_mean - real_dr_param
        L.log(f'eval/agent-sim_param/{param}/sim_param_error', sim_param_error, step)
        log_data[key][step]['sim_param_error'] = sim_param_error
        np.save(filename, log_data)


def update_sim_params_step(sim_param_model, sim_env, args, obs, step, L):
    with torch.no_grad():
        if args.outer_loop_version == 1:
            pred_sim_params = sim_param_model.forward(obs).mean
            pred_sim_params = pred_sim_params[0].cpu().numpy()
        elif args.outer_loop_version == 3:
            current_sim_params = torch.FloatTensor(sim_env.distribution_mean).unsqueeze(0)
            pred_sim_params = []
            if len(obs) > 1:
                for ob in obs:
                    pred_sim_params.append(predict_sim_params(sim_param_model, ob, current_sim_params))
            else:
                pred_sim_params.append(predict_sim_params(sim_param_model, obs[0], current_sim_params))
            pred_sim_params = np.mean(pred_sim_params, axis=0)

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
            scale_factor = max(prev_mean, .1)
            new_mean = prev_mean - alpha * (np.mean(pred_mean) - 0.5) #* scale_factor
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
        if args.anneal_range_scale > 0:
            range_value = args.anneal_range_scale * (1 - float(step / args.num_train_steps))
            L.log(f'eval/agent-sim_param/{param}/range', range_value, step)
            log_data[key][step]['range'] = range_value

        real_dr_param = args.real_dr_params[param]
        if not np.mean(real_dr_param) == 0:
            sim_param_error = (new_mean - real_dr_param) / real_dr_param
        else:
            sim_param_error = new_mean - real_dr_param
        L.log(f'eval/agent-sim_param/{param}/sim_param_error', sim_param_error, step)
        log_data[key][step]['sim_param_error'] = sim_param_error
        np.save(filename, log_data)


def evaluate(real_env, sim_env, agent, sim_param_model, video_real, video_sim, num_episodes, L, step, args):
    all_ep_rewards = []
    all_ep_success = []
    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        obs_batch = []
        real_sim_params = real_env.reset()['sim_params']
        for i in range(num_episodes):
            obs_dict = real_env.reset()
            video_real.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            obs_traj = []
            while not done and len(obs_traj) < args.time_limit:
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
                video_real.record(real_env)
                episode_reward += reward

            video_real.save('real_%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            if 'success' in obs_dict.keys():
                L.log('eval/' + prefix + 'episode_success', obs_dict['success'], step)
                all_ep_success.append(obs_dict['success'])
            all_ep_rewards.append(episode_reward)
            obs_batch.append(obs_traj)
        if not args.outer_loop_version == 0 and step > args.start_outer_loop:
            current_sim_params = torch.FloatTensor([sim_env.distribution_mean])
            test_preds_before = evaluate_sim_params(sim_param_model, args, obs_batch, step, L, "test", real_sim_params, current_sim_params)

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
        log_data[key][step]['mean_ep_reward'] = mean_ep_reward
        log_data[key][step]['max_ep_reward'] = best_ep_reward
        log_data[key][step]['std_ep_reward'] = std_ep_reward
        log_data[key][step]['env_step'] = step * args.action_repeat
        if len(all_ep_success) > 0:
            log_data[key][step]['mean_ep_success'] = mean_ep_success

        np.save(filename, log_data)

        obs_dict = sim_env.reset()
        done = False
        obs_traj_sim = []
        video_sim.init(enabled=True)
        while not done and len(obs_traj_sim) < args.time_limit:
            obs = obs_dict['image']
            # center crop image
            if (args.agent == 'curl_sac' and args.encoder_type == 'pixel') or (
                args.agent == 'rad_sac' and (args.encoder_type == 'pixel' or 'crop' in args.data_augs)):
                obs = utils.center_crop_image(obs, args.image_size)
            with utils.eval_mode(agent):
                if sample_stochastically:
                    action = agent.sample_action(obs)
                else:
                    action = agent.select_action(obs)
            obs_traj_sim.append(obs)
            obs_dict, reward, done, _ = sim_env.step(action)
            video_sim.record(sim_env)
            sim_params = obs_dict['sim_params']
        if sim_param_model is not None:
            dist_mean = obs_dict['distribution_mean']
            current_sim_params = torch.FloatTensor([sim_env.distribution_mean])
            train_preds_before = evaluate_sim_params(sim_param_model, args, [obs_traj_sim], step, L, "train", sim_params, current_sim_params)
            if args.outer_loop_version == 3:
                sim_param_model.train_classifier(obs_traj_sim, sim_params, dist_mean, L, step, True)
        video_sim.save('sim_%d.mp4' % step)

        if not args.outer_loop_version == 0 and step > args.start_outer_loop:
            update_sim_params_exact(sim_param_model, sim_env, args, obs_batch, step, L, real_sim_params)  # TODO: change this back, or make an option
            current_sim_params = torch.FloatTensor([sim_env.distribution_mean])
            test_preds_after = evaluate_sim_params(sim_param_model, args, obs_batch, step, L, "test_after_update", real_sim_params, current_sim_params)
            train_preds_after = evaluate_sim_params(sim_param_model, args, [obs_traj_sim], step, L, "train_after_update", sim_params, current_sim_params)
            for i, param in enumerate(args.real_dr_list):
                L.log(f'eval/train/{param}/train_diff', train_preds_after[i] - train_preds_before[i], step)
                L.log(f'eval/test/{param}/test_diff', test_preds_after[i] - test_preds_before[i], step)
                L.log(f'eval/train/{param}/train_test_diff', train_preds_before[i] - test_preds_before[i], step)

            obs_dict = sim_env.reset()
            done = False
            obs_traj_sim = []
            while not done and len(obs_traj_sim) < args.time_limit:
                obs = obs_dict['image']
                # center crop image
                if (args.agent == 'curl_sac' and args.encoder_type == 'pixel') or (
                    args.agent == 'rad_sac' and (args.encoder_type == 'pixel' or 'crop' in args.data_augs)):
                    obs = utils.center_crop_image(obs, args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs_traj_sim.append(obs)
                obs_dict, reward, done, _ = sim_env.step(action)
                sim_params = obs_dict['sim_params']
            if sim_param_model is not None:
                current_sim_params = torch.FloatTensor([sim_env.distribution_mean])
                train_preds_before = evaluate_sim_params(sim_param_model, args, [obs_traj_sim], step, L, "train_final",
                                                         sim_params, current_sim_params)

        else:
            print("skipping outer loop on", step)

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
        delay_steps=args.delay_steps,
        range_scale=args.range_scale,
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
        delay_steps=args.delay_steps,
        range_scale=args.range_scale,
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
        buffer = os.listdir(os.path.join(args.work_dir, 'buffer'))
        if len(checkpoints) == 0 or len(buffer) == 0:
            print("No checkpoints found")
            load_model = False
        else:
            agent_checkpoint = [f for f in checkpoints if 'curl' in f]
            if args.outer_loop_version in [1,3]:
                sim_param_checkpoint = [f for f in checkpoints if 'sim_param' in f]




    utils.make_dir(args.work_dir)
    sim_video_dir = utils.make_dir(os.path.join(args.work_dir, 'sim_video'))
    real_video_dir = utils.make_dir(os.path.join(args.work_dir, 'real_video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

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
        max_traj_length=args.time_limit
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    if args.outer_loop_version in [1, 3]:
        dist = 'binary' if args.outer_loop_version == 3 else 'normal'
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
            dist=dist,
            traj_length=args.time_limit,
        ).to(device)
    else:
        sim_param_model = None

    start_step = 0
    if load_model:
        agent_step = 0
        for checkpoint in agent_checkpoint:
            agent_step = max(agent_step, [int(x) for x in re.findall('\d+', checkpoint)][-1])
        agent.load_curl(model_dir, agent_step)
        start_step = agent_step
        if sim_param_model is not None:
            sim_param_step = 0
            for checkpoint in sim_param_checkpoint:
                sim_param_step = max(sim_param_step,[int(x) for x in re.findall('\d+', checkpoint)][-1])
            sim_param_model.load(model_dir, sim_param_step)
            start_step = min(start_step, sim_param_step)
        replay_buffer.load(buffer_dir)  # TODO: do we have to save optimizer?

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    success = None

    for step in range(start_step, args.num_train_steps):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(real_env, sim_env, agent, sim_param_model, video_real, video_sim, args.num_eval_episodes, L, step, args)
            if args.save_model:
                agent.save_curl(model_dir, step)
                if sim_param_model is not None:
                    sim_param_model.save(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
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

                L.log('train/episode_reward', episode_reward, step)
                log_data[key][step]['episode_reward'] = episode_reward
                if success is not None:
                    L.log('train/episode_success', success, step)
                    log_data[key][step]['episode_success'] = success

                np.save(filename, log_data)

            obs = sim_env.reset()
            success = 0.0 if 'success' in obs.keys() else None
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

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
                if step % args.train_sim_param_every == 0 and args.outer_loop_version != 0:
                    sim_param_model.update(replay_buffer, L, step, True)


        next_obs, reward, done, _ = sim_env.step(action)

        # allow infinite bootstrap
        done = True if episode_step >= args.time_limit - 1 else done
        done_bool = float(done)
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        if 'success' in obs.keys():
            success = obs['success']

        obs = next_obs
        episode_step += 1



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    main()
