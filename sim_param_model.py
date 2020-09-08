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
        encoder_feature_dim, encoder_num_layers, encoder_num_filters, agent, sim_param_lr=1e-3, sim_param_beta=0.9, dist='normal', act=nn.elu):
        self._shape = shape
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act
        self.device = device

        trunk = []
        for index in range(self._layers):
            trunk.append(nn.Linear(self._units))
            trunk.append(self._act)
        trunk.append(nn.Linear(np.prod(self._shape)))

        self.trunk = nn.Sequential(*trunk).to(self.device)

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, encoder_num_layers,
            encoder_num_filters, output_logits=True
        )

        self.apply(weight_init)
        self.encoder.copy_conv_weights_from(agent.critic.encoder)

        self.sim_param_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=sim_param_lr, betas=(sim_param_beta, 0.999)
        )


    def forward(
        self, features,
    ):
    # detach_encoder allows to stop gradient propagation to encoder


        x = features
        x = self.trunk(x)
        x = x.view(torch.cat([features.size()[:-1], self._shape], 0))

        if self._dist == 'normal':
            return torch.distributions.independent.Independent(torch.distributions.normal.Normal(x, 1), len(self._shape))
        if self._dist == 'binary':
            return torch.distributions.independent.Independent(torch.distributions.bernoulli.Bernoulli(x), len(self._shape))
        raise NotImplementedError(self._dist)

    def update(self, replay_buffer, L, step):
        if self.encoder_type == 'pixel':
            obs_list, actions_list, rewards_list, next_obses_list, not_dones_list, cpc_kwargs_list = replay_buffer.sample_cpc_traj()
        else:
            obs_list, actions_list, rewards_list, next_obses_list, not_dones_list = replay_buffer.sample_proprio_traj()


        feat = []
        pred_sim_params = []
        actual_params = []
        for traj in obs_list:
            for obs in traj:
                feat.append(self.encoder(obs['image'], detach=True))
            pred_sim_params.append(self.forward(feat))
            actual_params.append(obs['sim_params']) #take last obs


        loss = F.mse_loss(pred_sim_params, actual_params)
        if step % self.log_interval == 0:
            L.log('train_sim_params/loss', loss, step)

        # Optimize the critic
        self.sim_param_optimizer.zero_grad()
        loss.backward()
        self.sim_param_optimizer.step()

