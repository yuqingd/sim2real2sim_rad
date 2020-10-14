import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
from encoder import make_encoder, PixelEncoder
import data_augs as rad
from curl_sac import weight_init
from positional_encoding import get_embedder

class SimParamModel(nn.Module):
    def __init__(self, shape, action_space, layers, units, device, obs_shape, encoder_type,
        encoder_feature_dim, encoder_num_layers, encoder_num_filters, agent, sim_param_lr=1e-3, sim_param_beta=0.9,
                 dist='normal', act=nn.ELU, batch_size=32, traj_length=200, num_frames=10,
                 embedding_multires=10, use_img=True, state_dim=0, separate_trunks=False, param_names=[],
                 train_range_scale=1, prop_train_range_scale=False, clip_positive=False, dropout=0.5,
                 initial_range=None, single_window=False):
        super(SimParamModel, self).__init__()
        self._shape = shape
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act
        self.device = device
        self.encoder_type = encoder_type
        self.batch = batch_size
        self.traj_length = traj_length
        self.encoder_feature_dim = encoder_feature_dim
        self.num_frames = num_frames
        self.use_img = use_img
        self.state_dim = state_dim
        self.separate_trunks = separate_trunks
        positional_encoding, embedding_dim = get_embedder(embedding_multires, shape, i=0)
        self.positional_encoding = positional_encoding
        additional = 0 if dist == 'normal' else embedding_dim
        self.param_names = param_names
        self.train_range_scale = train_range_scale
        self.prop_train_range_scale = prop_train_range_scale
        self.clip_positive = clip_positive
        self.initial_range = initial_range
        self.single_window = single_window
        action_space_dim = np.prod(action_space.shape)

        if self.use_img:
            trunk_input_dim = encoder_feature_dim + additional + action_space_dim * num_frames
        else:
            trunk_input_dim = state_dim[-1] * self.num_frames + additional + action_space_dim * num_frames

        # If each sim param has its own trunk, create a separate trunk for each
        num_sim_params = shape
        if separate_trunks:
            trunk_list = []
            for _ in range(num_sim_params):
                trunk = []
                trunk.append(nn.Linear(trunk_input_dim, self._units))
                trunk.append(self._act())
                for index in range(self._layers - 1):
                    trunk.append(nn.Linear(self._units, self._units))
                    trunk.append(self._act())
                    trunk.append(nn.Dropout(p=dropout))
                trunk.append(nn.Linear(self._units, 1))
                trunk_list.append(nn.Sequential(*trunk).to(self.device))
            self.trunk = torch.nn.ModuleList(trunk_list)
        else:
            trunk = []
            trunk.append(nn.Linear(trunk_input_dim, self._units))
            trunk.append(self._act())
            for index in range(self._layers - 1):
                trunk.append(nn.Linear(self._units, self._units))
                trunk.append(self._act())
                trunk.append(nn.Dropout(p=dropout))
            trunk.append(nn.Linear(self._units, num_sim_params))
            self.trunk = nn.Sequential(*trunk).to(self.device)

        if self.use_img:
            # change obs_shape to account for the trajectory length, since images are stacked channel-wise
            c, h, w = obs_shape
            obs_shape = (3 * self.num_frames, h, w)
            self.encoder = make_encoder(
                encoder_type, obs_shape, encoder_feature_dim, encoder_num_layers,
                encoder_num_filters, output_logits=True
            )

        self.apply(weight_init)

        parameters = list(self.trunk.parameters())
        if self.use_img:
            parameters += list(self.encoder.parameters())
        self.sim_param_optimizer = torch.optim.Adam(
            parameters, lr=sim_param_lr, betas=(sim_param_beta, 0.999)
        )

    def get_features(self, obs_traj):
        # detach_encoder allows to stop gradient propagation to encoder

        with torch.no_grad():
            if self.use_img:
                if type(obs_traj[0][0]) is np.ndarray:
                    input = torch.FloatTensor(obs_traj).to(self.device)
                    B, num_frames, C, H, W = input.shape
                    input = input.view(B, num_frames * C, H, W)
                elif type(obs_traj[0][0]) is torch.Tensor:
                    input = torch.stack([torch.cat([o for o in traj], dim=0) for traj in obs_traj], dim=0)
                else:
                    raise NotImplementedError(type(obs_traj[0][0]))

                features = self.encoder(input, detach=True)
                features = features #/ torch.norm(features)
            else:
                if type(obs_traj[0][0]) is torch.Tensor:
                    features = torch.stack([torch.cat([o for o in traj], dim=0) for traj in obs_traj], dim=0)
                elif type(obs_traj[0][0]) is np.ndarray:
                    features = torch.FloatTensor([np.concatenate([o for o in traj], axis=0) for traj in obs_traj]).to(self.device)
                else:
                    raise NotImplementedError(type(obs_traj[0][0]))

        return features

    def forward(self, obs_traj):
        if self.use_img:
            features = self.get_features(obs_traj)
        else:
            features = torch.stack(obs_traj) # TODO: actually do this right, possibly with diff if cases for ndarray vs torch like above
        x = features.view(1, -1)
        if self.separate_trunks:
            x = torch.cat([trunk(x) for trunk in self.trunk], dim=-1)
        else:
            x = self.trunk(x)
        if self._dist == 'normal':
            return torch.distributions.normal.Normal(x, 1)
        if self._dist == 'binary':
            return torch.distributions.bernoulli.Bernoulli(x)
        raise NotImplementedError(self._dist)

    def forward_classifier(self, full_traj, pred_labels, step=5):
        """ obs traj list of lists, pred labels is array [B, num_sim_params] """
        full_obs_traj = []
        full_action_traj = []
        for traj in full_traj:
            # TODO: previously, we were always taking the first window.  Now, we always take a random one.
            #   We could consider choosing multiple, or choosing a separate segmentation for each batch element.
            if self.single_window:
                index = 0
            else:
                index = np.random.choice(len(traj) - self.num_frames + 1)
            traj = traj[index: index + self.num_frames]
            obs_traj, action_traj = zip(*traj)
            if type(action_traj[0]) is np.ndarray:
                full_action_traj.append(torch.FloatTensor(np.concatenate(action_traj)).to(self.device))
            elif type(action_traj[0]) is torch.Tensor:
                full_action_traj.append(torch.cat(action_traj))
            else:
                raise NotImplementedError
            # If we're using images, only use the first of the stacked frames
            if self.use_img:
                full_obs_traj.append([o[:3] for o in obs_traj])
            else:
                full_obs_traj.append(obs_traj)


        # normalize [-1, 1]
        pred_labels = pred_labels.to(self.device)

        encoded_pred_labels = self.positional_encoding(pred_labels)
        #encoded_pred_labels = pred_labels.repeat(1, 21)
        full_action_traj = torch.stack(full_action_traj)

        feat = self.get_features(full_obs_traj)
        B_label = len(pred_labels)
        B_traj = len(full_traj)
        fake_pred = torch.cat([encoded_pred_labels.repeat(B_traj, 1), feat.repeat(B_label, 1), full_action_traj.repeat(B_label, 1)], dim=-1)

        if self.separate_trunks:
            x = torch.cat([trunk(fake_pred) for trunk in self.trunk], dim=-1)
        else:
            x = self.trunk(fake_pred)
        pred_class = torch.distributions.bernoulli.Bernoulli(logits=x)
        pred_class = pred_class.mean
        return pred_class


    def train_classifier(self, obs_traj, sim_params, distribution_mean,  L, step, should_log):
        if self.initial_range is not None:
            dist_range = self.initial_range
        elif self.prop_train_range_scale:
            dist_range = self.train_range_scale * torch.FloatTensor(distribution_mean)
        else:
            dist_range = self.train_range_scale
        sim_params = torch.FloatTensor(sim_params) # 1 - dimensional
        eps = 1e-3
        distribution_mean = np.array(distribution_mean)
        if self.clip_positive:
            low_val = torch.clamp(distribution_mean - dist_range, eps, float('inf'))
        else:
            low_val = distribution_mean - dist_range

        num_low = np.random.randint(0, self.batch)
        #np.random.seed(0)
        #torch.manual_seed(0)
        low = torch.FloatTensor(
            np.random.uniform(size=(num_low, len(sim_params)), low=low_val,
                              high=distribution_mean)).to(self.device)

        high = torch.FloatTensor(
            np.random.uniform(size=(self.batch * 4 - num_low, len(sim_params)),
                              low=distribution_mean,
                              high=distribution_mean + dist_range)).to(self.device)
        dist_mean = torch.FloatTensor(distribution_mean).unsqueeze(0).to(self.device)
        fake_pred = torch.cat([low, high, dist_mean], dim=0)
        labels = (fake_pred > sim_params.unsqueeze(0).to(self.device)).long()

        # Shuffle all params but the last, which is distribution mean
        shuffled_indices = torch.stack([torch.randperm(len(fake_pred) - 1) for _ in range(len(sim_params))], dim=1).to(
            self.device)
        dist_mean_indices = torch.zeros(1, len(distribution_mean)).to(self.device).int() + len(shuffled_indices)
        shuffled_indices = torch.cat([shuffled_indices, dist_mean_indices])

        labels = torch.gather(labels, 0, shuffled_indices)
        fake_pred = torch.gather(fake_pred, 0, shuffled_indices)
        assert obs_traj[0][0].shape == (9, 84, 84), obs_traj[0][0].shape
        try:
            dr_param = np.max(obs_traj[0][0][0]) / 255 # red channel of first
        except:
            dr_param = torch.max(obs_traj[0][0][0]) / 255
        pred_labels = fake_pred > dr_param
        hardcoded_accuracy = (torch.sum(labels == pred_labels) * 1.0) / np.prod(labels.shape)

        pred_class = self.forward_classifier([obs_traj], fake_pred)
        pred_class_flat = pred_class.flatten().unsqueeze(0).float()
        labels_flat = labels.flatten().unsqueeze(0).float()
        loss = nn.BCELoss()(pred_class_flat, labels_flat)
        full_loss = nn.BCELoss(reduction='none')(pred_class.float(), labels.float()).detach().cpu().numpy()
        individual_loss = np.mean(full_loss, axis=0)
        accuracy = torch.round(pred_class) == labels
        individual_accuracy = torch.mean(accuracy.float(), dim=0).detach().cpu().numpy()
        accuracy_mean = torch.mean(accuracy.float()).detach().cpu().numpy()
        error = pred_class - labels
        individual_error = torch.mean(error.float(), dim=0).detach().cpu().numpy()
        error_mean = torch.mean(error.float()).detach().cpu().numpy()

        dist_error_mean = torch.mean(error[-1].float()).detach().cpu().numpy()
        dist_error_individual = error[-1].detach().cpu().numpy()
        dist_accuracy_mean = torch.mean(accuracy[-1].float()).detach().cpu().numpy()
        dist_accuracy_individual = accuracy[-1].float().detach().cpu().numpy()
        dist_loss_mean = np.mean(full_loss[-1])
        dist_loss_individual = full_loss[-1]

        if should_log:
            print("hardcoded accuracy", hardcoded_accuracy.item())
            L.log('train_hardcoded_accuracy', hardcoded_accuracy, step)
            L.log('train_sim_params/loss', loss, step)
            L.log('train_sim_params/accuracy', accuracy_mean, step)
            L.log('train_sim_params/error', error_mean, step)
            L.log('train_sim_params/dist_mean_loss', dist_loss_mean, step)
            L.log('train_sim_params/dist_mean_accuracy', dist_accuracy_mean, step)
            L.log('train_sim_params/dist_mean_error', dist_error_mean, step)
            for i, param in enumerate(self.param_names):
                L.log(f'train_sim_params/{param}/loss', individual_loss[i], step)
                L.log(f'train_sim_params/{param}/accuracy', individual_accuracy[i], step)
                L.log(f'train_sim_params/{param}/error', individual_error[i], step)
                L.log(f'train_sim_params/{param}/dist_mean_loss', dist_loss_individual[i], step)
                L.log(f'train_sim_params/{param}/dist_mean_accuracy', dist_accuracy_individual[i], step)
                L.log(f'train_sim_params/{param}/dist_mean_error', dist_error_individual[i], step)

        # Optimize the critic
        self.sim_param_optimizer.zero_grad()
        loss.backward()
        self.sim_param_optimizer.step()

    def update(self, obs_list, sim_params, dist_mean, L, step, should_log, replay_buffer=None):
        if replay_buffer is not None:
            if self.encoder_type == 'pixel':
                obs_list, actions_list, rewards_list, next_obses_list, not_dones_list, cpc_kwargs_list = replay_buffer.sample_cpc_traj(1)
            else:
                obs_list, actions_list, rewards_list, next_obses_list, not_dones_list = replay_buffer.sample_proprio_traj(16)

        if self._dist == 'normal':
            pred_sim_params = []
            actual_params = []
            for traj in obs_list:
                pred_sim_params.append(self.forward(traj['image']).mean[0])
                actual_params.append(traj['sim_params'][-1]) #take last obs

            loss = F.mse_loss(torch.stack(pred_sim_params), torch.stack(actual_params))

            if should_log:
                L.log('train_sim_params/loss', loss, step)

            # Optimize the critic
            self.sim_param_optimizer.zero_grad()
            loss.backward()
            self.sim_param_optimizer.step()
        else:
            def to_numpy(x):
                return x.detach().cpu().numpy()

            if replay_buffer is None:
                self.train_classifier(obs_list, np.array(sim_params), np.array(dist_mean),
                                      L, step, should_log)
            else:
                for obs_traj, action_traj in zip(obs_list, actions_list):
                    action_traj = [to_numpy(a) for a in action_traj]
                    if self.encoder_type == 'pixel':
                        obs_traj_image = [to_numpy(i) for i in obs_traj['image']]
                        self.train_classifier(list(zip(obs_traj_image, action_traj)),
                                              to_numpy(obs_traj['sim_params'][-1]),
                                              to_numpy(obs_traj['distribution_mean'][-1]), L, step, should_log)
                    else:
                        obs_traj_state = [to_numpy(i) for i in obs_traj['state']]
                        self.train_classifier(list(zip(obs_traj_state, action_traj)),
                                              to_numpy(obs_traj['sim_params'][-1]),
                                              to_numpy(obs_traj['distribution_mean'][-1]), L, step, should_log)


    def save(self, model_dir, step):
        torch.save(
            self.state_dict(), '%s/sim_param_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.load_state_dict(
            torch.load('%s/sim_param_%s.pt' % (model_dir, step))
        )

