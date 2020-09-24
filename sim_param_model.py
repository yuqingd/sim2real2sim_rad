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

class SimParamModel(nn.Module):
    def __init__(self, shape, layers, units, device, obs_shape, encoder_type,
        encoder_feature_dim, encoder_num_layers, encoder_num_filters, agent, sim_param_lr=1e-3, sim_param_beta=0.9,
                 dist='normal', act=nn.ELU, batch_size=32, traj_length=200, use_gru=True):
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
        self.use_gru = use_gru
        self.encoder_feature_dim = encoder_feature_dim
        additional = 0 if dist == 'normal' else shape

        trunk = []
        if self.use_gru:
            trunk.append(nn.Linear(encoder_feature_dim + additional, self._units))
        else:
            trunk.append(nn.Linear(traj_length * encoder_feature_dim + additional, self._units))
        trunk.append(self._act())
        for index in range(self._layers - 1):
            trunk.append(nn.Linear(self._units, self._units))
            trunk.append(self._act())
        trunk.append(nn.Linear(self._units, np.prod(self._shape)))
        self.trunk = nn.Sequential(*trunk).to(self.device)

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, encoder_num_layers,
            encoder_num_filters, output_logits=True
        )

        if self.use_gru:
            self.gru = nn.GRU(encoder_feature_dim + additional, encoder_feature_dim + additional)

        self.apply(weight_init)
        self.encoder.copy_conv_weights_from(agent.critic.encoder)

        if self.use_gru:
            self.sim_param_optimizer = torch.optim.Adam(
                list(self.encoder.parameters())+ list(self.trunk.parameters()) + list(self.gru.parameters()), lr=sim_param_lr, betas=(sim_param_beta, 0.999)
            )
        else:
            self.sim_param_optimizer = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.trunk.parameters()), lr=sim_param_lr, betas=(sim_param_beta, 0.999)
            )

    def get_features(self, obs_traj):
        # detach_encoder allows to stop gradient propagation to encoder

        with torch.no_grad():
            if type(obs_traj[0]) is np.ndarray:
                obs = np.stack(obs_traj)
                input = torch.FloatTensor(obs).to(self.device)
            elif type(obs_traj[0]) is torch.Tensor:
                input = obs_traj
            else:
                raise NotImplementedError(type(obs_traj[0]))


            if len(input) < self.traj_length:  # TODO: generalize!
                    last = input[-1]
                    last_arr = torch.stack([copy.deepcopy(last) for _ in range(self.traj_length - len(obs_traj))]).to(self.device)
                    input = torch.cat([input, last_arr])

            features = self.encoder(input, detach=True)

        return features

    def forward(self, obs_traj):
        features = self.get_features(obs_traj)
        x = features.view(1, -1)
        x = self.trunk(x)
        if self._dist == 'normal':
            return torch.distributions.normal.Normal(x, 1)
        if self._dist == 'binary':
            return torch.distributions.bernoulli.Bernoulli(x)
        raise NotImplementedError(self._dist)

    def forward_classifier(self, obs_traj, pred_labels):
        pred_labels = pred_labels.to(self.device)
        feat = self.get_features(obs_traj)
        B = len(pred_labels)
        num_traj = len(obs_traj)
        #feat = feat.flatten()
        feat_tiled = feat.unsqueeze(1).repeat(1, B, 1)
        pred_tiled = pred_labels.unsqueeze(0).repeat(num_traj, 1, 1)
        fake_pred = torch.cat([pred_tiled, feat_tiled], dim=-1).view(-1, B, self.encoder_feature_dim + self._shape)
    
        if self.use_gru:
            hidden = torch.zeros(1, B, self.encoder_feature_dim + self._shape, device=self.device)
            fake_pred, hidden = self.gru(fake_pred, hidden)
            fake_pred = fake_pred[-1] #only look at end of traj

        x = self.trunk(fake_pred)
        pred_class = torch.distributions.bernoulli.Bernoulli(logits=x)
        pred_class = pred_class.mean
        return pred_class


    def train_classifier(self, obs_traj, sim_params, distribution_mean,  L, step, should_log):

        dist_range = 10 * torch.FloatTensor(distribution_mean)
        sim_params = torch.FloatTensor(sim_params) # 1 - dimensional
        eps = 1e-3
        low = torch.FloatTensor(
            np.random.uniform(size=(self.batch, len(sim_params)), low=torch.clamp(sim_params - dist_range, eps, float('inf')),
                              high=sim_params)).to(self.device)

        high = torch.FloatTensor(
            np.random.uniform(size=(self.batch, len(sim_params)),
                              low=sim_params,
                              high=sim_params + dist_range)).to(self.device)
        fake_pred = torch.cat([low, high], dim=0)
        labels = (fake_pred > sim_params.unsqueeze(0).to(self.device)).long()
        shuffled_indices = torch.randperm(labels.size()[0]).to(self.device)
        labels = labels[shuffled_indices]
        fake_pred = fake_pred[shuffled_indices]
        #shuffled_indices = torch.randperm(len(labels)).to(self.device)
        #labels = torch.gather(labels, 0, shuffled_indices)
        #fake_pred = torch.gather(fake_pred, 0, shuffled_indices)
        #labels = (fake_pred > sim_params.unsqueeze(0).to(self.device)).long()
        pred_class = self.forward_classifier(obs_traj, fake_pred)
        pred_class = pred_class.flatten().unsqueeze(0).float()
        labels = labels.flatten().unsqueeze(0).float()
        loss = -nn.BCELoss()(pred_class, labels)

        if should_log:
            L.log('train_sim_params/loss', loss, step)

        # Optimize the critic
        self.sim_param_optimizer.zero_grad()
        loss.backward()
        self.sim_param_optimizer.step()

    def update(self, replay_buffer, L, step, should_log):
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
            for traj in obs_list:
                self.train_classifier(traj['image'], traj['sim_params'][-1].to('cpu'), traj['distribution_mean'][-1].to('cpu'),
                                      L, step, should_log)


    def save(self, model_dir, step):
        torch.save(
            self.state_dict(), '%s/sim_param_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.load_state_dict(
            torch.load('%s/sim_param_%s.pt' % (model_dir, step))
        )

