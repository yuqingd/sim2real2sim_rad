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
                 initial_range=None, single_window=False, share_encoder=False, normalize_features=False,
                 use_downsampling=True, use_encoder=True, downsample_size=32, use_layer_norm=False,
                 use_weight_init=False):
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
        self.share_encoder = share_encoder
        self.use_downsampling = use_downsampling
        self.use_encoder = use_encoder
        self.normalize_features = normalize_features
        self.downsample_size = downsample_size
        self.use_layer_norm = use_layer_norm
        self.use_weight_init = use_weight_init
        self.feature_norm = None
        action_space_dim = np.prod(action_space.shape)

        if self.use_img:
            assert use_encoder or use_downsampling, "we must use at least one encoding scheme"
            if self.use_encoder:
                if self.share_encoder:
                    encoder_dims = encoder_feature_dim * num_frames
                else:
                    encoder_dims = encoder_feature_dim
            else:
                encoder_dims = 0
            if self.use_downsampling:
                downsample_dims = downsample_size * downsample_size * 3
            else:
                downsample_dims = 0
            trunk_input_dim = encoder_dims + downsample_dims + additional + action_space_dim * num_frames
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
            if self.share_encoder:
                obs_shape = (3, h, w)
                self.encoder = make_encoder(
                    encoder_type, obs_shape, encoder_feature_dim, encoder_num_layers,
                    encoder_num_filters, output_logits=True, use_layer_norm=self.use_layer_norm
                )
            else:
                obs_shape = (3 * self.num_frames, h, w)
                self.encoder = make_encoder(
                    encoder_type, obs_shape, encoder_feature_dim, encoder_num_layers,
                    encoder_num_filters, output_logits=True, use_layer_norm=self.use_layer_norm
                )

        if self.use_weight_init:
            self.apply(weight_init)

        parameters = list(self.trunk.parameters())
        if self.use_img:
            parameters += list(self.encoder.parameters())
        self.sim_param_optimizer = torch.optim.Adam(
            parameters, lr=sim_param_lr, betas=(sim_param_beta, 0.999)
        )

    def get_features(self, obs_traj):
        # detach_encoder allows to stop gradient propagation to encoder

        if self.use_img:
            # Share encoder case
            if obs_traj[0][0].shape[0] == 9:
                input = []
                for i in range(len(obs_traj[0])):
                    if type(obs_traj[0][0]) is torch.Tensor:
                        input.append(torch.FloatTensor([traj[i].detach().cpu().numpy() for traj in obs_traj]).to(self.device))
                    else:
                        input.append(torch.FloatTensor([traj[i] for traj in obs_traj]).to(self.device))
            elif type(obs_traj[0][0]) is np.ndarray:
                input = torch.FloatTensor(obs_traj).to(self.device)
                B, num_frames, C, H, W = input.shape
                input = input.view(B, num_frames * C, H, W)
            elif type(obs_traj[0][0]) is torch.Tensor:
                input = torch.stack([torch.cat([o for o in traj], dim=0) for traj in obs_traj], dim=0)
            else:
                raise NotImplementedError(type(obs_traj[0][0]))

            if self.use_encoder:
                if self.share_encoder:
                    features = [self.encoder(img, detach=True) for img in input]
                    features = torch.cat(features, dim=1)
                else:
                    # Don't update the conv layers if we're sharing, otherwise to
                    features = self.encoder(input, detach=self.share_encoder)
                self.feature_norm = torch.norm(features).detach()
                if self.normalize_features:
                    features = features / torch.norm(features).detach()
            if self.use_downsampling:
                # input is multiple frames stacked, but we only need the first
                if type(input) is list:
                    first_input = input[0][:, :3]
                else:
                    first_input = input[:, :3]

                input_size = first_input.shape[-1]
                output_size = self.downsample_size
                bin_size = input_size // output_size

                if torch.max(first_input).item() > 1:
                    first_input = first_input / 255
                b, c, h, w = first_input.shape
                downsample_features = first_input.reshape(
                    (b, c, output_size, bin_size, output_size, bin_size)).mean(5).mean(3).reshape(b, -1)
                if self.use_encoder:
                    features = torch.cat([features, downsample_features], dim=1)
                else:
                    features = downsample_features
            temp = 3

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
            if self.use_img and not self.share_encoder:
                full_obs_traj.append([o[:3] for o in obs_traj])
            else:
                full_obs_traj.append(obs_traj)


        # normalize [-1, 1]
        pred_labels = pred_labels.to(self.device)

        encoded_pred_labels = self.positional_encoding(pred_labels)
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
        if self.clip_positive:
            low_val = torch.clamp(sim_params - dist_range, eps, float('inf'))
        else:
            low_val = sim_params - dist_range

        num_low = np.random.randint(0, self.batch * 4)  # TODO: figure out why x4
        low = torch.FloatTensor(
            np.random.uniform(size=(num_low, len(sim_params)), low=low_val,
                              high=sim_params)).to(self.device)

        high = torch.FloatTensor(
            np.random.uniform(size=(self.batch * 4 - num_low, len(sim_params)),
                              low=sim_params,
                              high=sim_params + dist_range)).to(self.device)
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
            L.log('train_sim_params/loss', loss, step)
            L.log('train_sim_params/accuracy', accuracy_mean, step)
            L.log('train_sim_params/error', error_mean, step)
            L.log('train_sim_params/dist_mean_loss', dist_loss_mean, step)
            L.log('train_sim_params/dist_mean_accuracy', dist_accuracy_mean, step)
            L.log('train_sim_params/dist_mean_error', dist_error_mean, step)
            if self.feature_norm is not None:
                L.log('train_sim_params/feature_norm', self.feature_norm, step)
            for i, param in enumerate(self.param_names):
                L.log(f'train_sim_params/{param}/loss', individual_loss[i], step)
                L.log(f'train_sim_params/{param}/accuracy', individual_accuracy[i], step)
                L.log(f'train_sim_params/{param}/error', individual_error[i], step)
                L.log(f'train_sim_params/{param}/dist_mean_loss', dist_loss_individual[i], step)
                L.log(f'train_sim_params/{param}/dist_mean_accuracy', dist_accuracy_individual[i], step)
                L.log(f'train_sim_params/{param}/dist_mean_error', dist_error_individual[i], step)

        # Optimize the critic
        return loss
        # self.sim_param_optimizer.zero_grad()
        # loss.backward()
        # self.sim_param_optimizer.step()

    def update(self, obs_list, sim_params, dist_mean, L, step, should_log, replay_buffer=None):
        total_num_trajs = 16
        if replay_buffer is not None:
            if self.encoder_type == 'pixel':
                obs_list, actions_list, rewards_list, next_obses_list, not_dones_list, cpc_kwargs_list = replay_buffer.sample_cpc_traj(total_num_trajs)
            else:
                obs_list, actions_list, rewards_list, next_obses_list, not_dones_list = replay_buffer.sample_proprio_traj(total_num_trajs)

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
            if replay_buffer is None:
                loss = self.train_classifier(obs_list, sim_params, dist_mean,
                                      L, step, should_log)
            else:
                losses = []
                for obs_traj, action_traj in zip(obs_list, actions_list):
                    if self.encoder_type == 'pixel':
                        losses.append (self.train_classifier(list(zip(obs_traj['image'], action_traj)),
                                              obs_traj['sim_params'][-1].to('cpu'),
                                              obs_traj['distribution_mean'][-1].to('cpu'), L, step, should_log))
                    else:
                        losses.append(self.train_classifier(list(zip(obs_traj['state'], action_traj)), obs_traj['sim_params'][-1].to('cpu'),
                                              obs_traj['distribution_mean'][-1].to('cpu'), L, step, should_log))
                loss = sum(losses)
            self.sim_param_optimizer.zero_grad()
            loss.backward()
            self.sim_param_optimizer.step()


    def save(self, model_dir, step):
        torch.save(
            self.state_dict(), '%s/sim_param_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.load_state_dict(
            torch.load('%s/sim_param_%s.pt' % (model_dir, step))
        )

