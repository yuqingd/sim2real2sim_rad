import torch
import numpy as np
import torch.nn as nn
import gym
import os
import copy
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import time
from skimage.util.shape import view_as_windows


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, example_obs, action_shape, capacity, batch_size, device, image_size=84, transform=None,
                 max_traj_length=200, val_split=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        self.max_traj_length = int(max_traj_length)  # Generalize!

        self.obses = {}
        self.next_obses = {}
        for key in example_obs.keys():
            val = example_obs[key]
            try:
                self.obses[key] = np.empty((capacity, *val.shape), dtype=val.dtype)
                self.next_obses[key] = np.empty((capacity, *val.shape), dtype=val.dtype)
            except Exception as e:
                self.obses[key] = np.empty((capacity, 1), dtype=type(val))
                self.next_obses[key] = np.empty((capacity, 1), dtype=type(val))
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.traj_ids = np.zeros((capacity, 1), dtype=np.int32) - 1
        self.traj_id = 0

        self.idx = 0
        self.done_idx = 0
        self.last_save = 0
        self.full = False
        self.val_split = val_split

    def add(self, obs, action, reward, next_obs, done):
        for k, v in obs.items():
            np.copyto(self.obses[k][self.idx], v)
        for k, v in next_obs.items():
            np.copyto(self.next_obses[k][self.idx], v)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.traj_ids[self.idx], self.traj_id)
        if done:
            self.traj_id += 1
            self.done_idx = (self.idx + 1) % self.capacity

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def get_idxs(self, val=False):
        total_num_samples = self.capacity if self.full else self.done_idx

        if self.val_split is not None:
            cutoff = int(total_num_samples * self.val_split)
            if val:
                start_idx = 0
                end_idx = cutoff
            else:
                start_idx = cutoff
                end_idx = total_num_samples
        else:
            start_idx = 0
            end_idx = total_num_samples

        idxs = np.random.randint(start_idx, end_idx, size=self.batch_size)
        return idxs

    def get_idxs_traj(self, num_trajs, val=False):
        total_num_samples = self.capacity if self.full else self.done_idx
        if self.val_split is not None:
            cutoff = int(total_num_samples * self.val_split)
            if val:
                start_idx = 0
                end_idx = cutoff
            else:
                start_idx = cutoff
                end_idx = total_num_samples
        else:
            start_idx = 0
            end_idx = total_num_samples
        unique_trajs = np.unique(self.traj_ids[start_idx: end_idx])
        # The way we split trajectories will result in one trajectory being in train and val. We remove it from val.
        if val:
            unique_trajs = unique_trajs[:-1]
        traj_ids = np.random.choice(unique_trajs, size=num_trajs, replace=True)
        return traj_ids

    def sample_proprio(self, use_img=False, val=False):
        idxs = self.get_idxs(val)
        results = self._sample_proprio(idxs, obs_vec_only=True)
        obses_img, obses_state, actions, rewards, next_obses_img, next_obses_state, not_dones = results
        return obses_img, actions, rewards, next_obses_img, not_dones

    def _sample_proprio(self, idxs, obs_vec_only=True):
        if obs_vec_only:
            obses_img = torch.as_tensor(self.obses['image'][idxs], device=self.device).float()
            next_obses_img = torch.as_tensor(self.next_obses['image'][idxs], device=self.device).float()
            obses_state = torch.as_tensor(self.obses['state'][idxs], device=self.device).float()
            next_obses_state = torch.as_tensor(self.next_obses['state'][idxs], device=self.device).float()
        else:
            obses = {}
            next_obses = {}
            for k in self.obses.keys():
                obses[k] = torch.as_tensor(self.obses[k][idxs], device=self.device).float()
            for k in self.next_obses.keys():
                next_obses[k] = torch.as_tensor(self.next_obses[k][idxs], device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        if not obs_vec_only:
            obses_dict = {}
            next_obses_dict = {}
            for k, v in self.obses.items():
                obses_dict[k] = torch.as_tensor(v[idxs], device=self.device).float()
            for k, v in self.next_obses.items():
                next_obses_dict[k] = torch.as_tensor(v[idxs], device=self.device).float()
            obses_img = obses_dict
            next_obses_img = next_obses_dict
            # State is empty, since state is already contained within the dict
            obses_state = None
            next_obses_state = None

        return obses_img, obses_state, actions, rewards, next_obses_img, next_obses_state, not_dones

    def sample_proprio_traj(self, num_trajs, val=False):
        # Select trajectory indices
        traj_ids = self.get_idxs_traj(num_trajs, val)

        # Obtain indices for each trajectory
        obs_list = []
        actions_list = []
        rewards_list = []
        next_obses_list = []
        not_dones_list = []
        for traj in traj_ids:
            idxs = np.where(self.traj_ids == traj)[0]
            # We should never have to use this if case unless we're using an environment with early termination.
            if len(idxs) < self.max_traj_length:
                last = idxs[-1]
                last_repeat = np.zeros(self.max_traj_length - len(idxs), dtype=np.int32) + last
                idxs = np.concatenate([idxs, last_repeat], 0)
            obs, _, actions, rewards, next_obses, _, not_dones = self._sample_proprio(idxs, obs_vec_only=False)
            obs_list.append(obs)
            actions_list.append(actions)
            rewards_list.append(rewards)
            next_obses_list.append(next_obses)
            not_dones_list.append(not_dones)
        return obs_list, actions_list, rewards_list, next_obses_list, not_dones_list

    def sample_cpc(self, val=False):
        idxs = self.get_idxs(val)
        results = self._sample_cpc(idxs, obs_vec_only=True)
        obses_img, obses_state, actions, rewards, next_obses_img, next_obses_state, not_dones, cpc_kwargs = results
        return obses_img, actions, rewards, next_obses_img, not_dones, cpc_kwargs

    def _sample_cpc(self, idxs, obs_vec_only=True, random_crop_img=True):

        obses_img = self.obses['image'][idxs]
        next_obses_img = self.next_obses['image'][idxs]
        obses_state = self.obses['state'][idxs]
        next_obses_state = self.next_obses['state'][idxs]

        pos = obses_img.copy()

        if random_crop_img:
            obses_img = random_crop(obses_img, self.image_size)
            next_obses_img = random_crop(next_obses_img, self.image_size)
            pos = random_crop(pos, self.image_size)
        else:
            obses_img = center_crop_image(obses_img, self.image_size)
            next_obses_img = center_crop_image(next_obses_img, self.image_size)
            pos = center_crop_image(pos, self.image_size)

        obses_img = torch.as_tensor(obses_img, device=self.device).float()
        next_obses_img = torch.as_tensor(next_obses_img, device=self.device).float()
        obses_state = torch.as_tensor(obses_state, device=self.device).float()
        next_obses_state = torch.as_tensor(next_obses_state, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses_img, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        if not obs_vec_only:
            obses_dict = {}
            next_obses_dict = {}
            for k, v in self.obses.items():
                obses_dict[k] = torch.as_tensor(v[idxs], device=self.device).float()
            for k, v in self.next_obses.items():
                next_obses_dict[k] = torch.as_tensor(v[idxs], device=self.device).float()
            obses_dict['image'] = obses_img
            next_obses_dict['image'] = next_obses_img
            obses_img = obses_dict
            next_obses_img = next_obses_dict
            # State is empty, since state is already contained within the dict
            obses_state = None
            next_obses_state = None

        return obses_img, obses_state, actions, rewards, next_obses_img, next_obses_state, not_dones, cpc_kwargs

    def sample_cpc_traj(self, num_trajs, val=False):
        # Select trajectory indices
        traj_ids = self.get_idxs_traj(num_trajs, val)

        # Obtain indices for each trajectory
        obs_list = []
        state_list = []
        actions_list = []
        rewards_list = []
        next_obses_list = []
        not_dones_list = []
        cpc_kwargs_list = []
        for traj_id in traj_ids:
            idxs = np.where(self.traj_ids[:self.capacity if self.full else self.done_idx] == traj_id)[0]
            if len(idxs) < self.max_traj_length:
                last = idxs[-1]
                last_repeat = np.zeros(self.max_traj_length - len(idxs), dtype=np.int32) + last
                idxs = np.concatenate([idxs, last_repeat], 0)
            obs, _, actions, rewards, next_obses, _, not_dones, cpc_kwargs = self._sample_cpc(idxs, obs_vec_only=False,
                                                                                        random_crop_img=False)
            obs_list.append(obs)
            actions_list.append(actions)
            rewards_list.append(rewards)
            next_obses_list.append(next_obses)
            not_dones_list.append(not_dones)
            cpc_kwargs_list.append(cpc_kwargs)
        return obs_list, actions_list, rewards_list, next_obses_list, not_dones_list, cpc_kwargs_list

    def sample_rad(self, aug_funcs, val=False):
        idxs = self.get_idxs(val)
        results = self._sample_rad(aug_funcs, idxs, obs_vec_only=True)
        obses_img, obses_state, actions, rewards, next_obses_img, next_obses_state, not_dones = results
        return obses_img, actions, rewards, next_obses_img, not_dones

    def _sample_rad(self, aug_funcs, idxs, obs_vec_only=True):
        # augs specified as flags
        # curl_sac organizes flags into aug funcs
        # passes aug funcs into sampler

        obses_img = self.obses['image'][idxs]
        next_obses_img = self.next_obses['image'][idxs]
        obses_state = self.obses['state'][idxs]
        next_obses_state = self.next_obses['state'][idxs]

        if aug_funcs:
            for aug, func in aug_funcs.items():
                # apply crop and cutout first
                if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                    obses_img = func(obses_img)
                    next_obses_img = func(next_obses_img)

        obses_img = torch.as_tensor(obses_img, device=self.device).float()
        next_obses_img = torch.as_tensor(next_obses_img, device=self.device).float()
        obses_state = torch.as_tensor(obses_img, device=self.device).float()
        next_obses_state = torch.as_tensor(next_obses_img, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        obses_img = obses_img / 255.
        next_obses_img = next_obses_img / 255.

        # augmentations go here
        if aug_funcs:
            for aug, func in aug_funcs.items():
                # skip crop and cutout augs
                if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                    continue
                obses_img = func(obses_img)
                next_obses_img = func(next_obses_img)

        if not obs_vec_only:
            obses_dict = {}
            next_obses_dict = {}
            for k, v in self.obses.items():
                obses_dict[k] = torch.as_tensor(v[idxs], device=self.device).float()
            for k, v in self.next_obses.items():
                next_obses_dict[k] = torch.as_tensor(v[idxs], device=self.device).float()
            obses_dict['image'] = obses_img
            next_obses_dict['image'] = next_obses_img
            obses_img = obses_dict
            next_obses_img = next_obses_dict
            # State is empty, since state is already contained within the dict
            obses_state = None
            next_obses_state = None

        return obses_img, obses_state, actions, rewards, next_obses_img, next_obses_state, not_dones

    def sample_rad_traj(self, aug_funcs, num_trajs, val=False):
        # Select trajectory indices
        traj_ids = self.get_idxs_traj(num_trajs, val)

        # Obtain indices for each trajectory
        obs_list = []
        actions_list = []
        rewards_list = []
        next_obses_list = []
        not_dones_list = []
        for traj in traj_ids:
            idxs = np.where(self.traj_ids == traj)[0]
            if len(idxs) < self.max_traj_length:
                last = idxs[-1]
                last_repeat = np.zeros(self.max_traj_length - len(idxs), dtype=np.int32) + last
                idxs = np.concatenate([idxs, last_repeat], 0)
            obs, _, actions, rewards, next_obses, _, not_dones = self._sample_rad(aug_funcs, idxs, obs_vec_only=False)
            obs_list.append(obs)
            actions_list.append(actions)
            rewards_list.append(rewards)
            next_obses_list.append(next_obses)
            not_dones_list.append(not_dones)
        return obs_list, actions_list, rewards_list, next_obses_list, not_dones_list
        
    def save(self, save_dir):
        if self.idx <= self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))

        sliced_obses = {k:v[self.last_save:self.idx] for k, v in self.obses.items()}
        sliced_next_obses = {k:v[self.last_save:self.idx] for k, v in self.next_obses.items()}

        payload = [
            sliced_obses,
            sliced_next_obses,
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx],
            self.traj_ids[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            if self.idx != start or end <= start:
                continue
            try:
                payload = torch.load(path)
            except:
                print("Unable to load ", str(path))
                continue
            for k,v in payload[0].items():
                self.obses[k][start:end] = v
            for k,v in payload[1].items():
                self.next_obses[k][start:end] = v
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.traj_ids[start:end] = payload[5]
            self.idx = end
            self.done_idx = end - (end % self.max_traj_length)
            self.last_save = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity 


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space['image'].shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space['image'].dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        img = np.concatenate(list([f['image'] for f in self._frames]), axis=0)
        d = copy.deepcopy(self._frames[-1])
        d['image'] = img
        return d


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs


def center_crop_image(image, output_size):
    if len(image.shape) == 3:  # No batch
        h, w = image.shape[1:]
        new_h, new_w = output_size, output_size

        top = (h - new_h)//2
        left = (w - new_w)//2

        image = image[:, top:top + new_h, left:left + new_w]
    else:
        b, c, h, w = image.shape
        new_h, new_w = output_size, output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        image = image[:, :, top:top + new_h, left:left + new_w]
    return image
