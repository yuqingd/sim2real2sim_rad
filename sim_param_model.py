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
                 dist='normal', act=nn.ELU):
        super(SimParamModel, self).__init__()
        self._shape = shape
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act
        self.device = device
        self.encoder_type = encoder_type

        trunk = []
        trunk.append(nn.Linear(200 * 50, self._units))
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

        self.apply(weight_init)
        self.encoder.copy_conv_weights_from(agent.critic.encoder)

        self.sim_param_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=sim_param_lr, betas=(sim_param_beta, 0.999)
        )


    def forward(self, obs_traj):
        # detach_encoder allows to stop gradient propagation to encoder
        with torch.no_grad():
            if type(obs_traj[0]) is np.ndarray:
                obs = np.stack(obs_traj)
                input = torch.FloatTensor(obs).to(self.device)
            elif type(obs_traj[0]) is torch.Tensor:
                input = obs_traj
            else:
                raise NotImplementedError(type(obs_traj[0]))

        features = self.encoder(input, detach=True)

        x = features.view(1, -1)
        x = self.trunk(x)
        if self._dist == 'normal':
            return torch.distributions.normal.Normal(x, 1)
        if self._dist == 'binary':
            return torch.distributions.bernoulli.Bernoulli(x)
        raise NotImplementedError(self._dist)

    def update(self, replay_buffer, L, step):
        if self.encoder_type == 'pixel':
            obs_list, actions_list, rewards_list, next_obses_list, not_dones_list, cpc_kwargs_list = replay_buffer.sample_cpc_traj(1)
        else:
            obs_list, actions_list, rewards_list, next_obses_list, not_dones_list = replay_buffer.sample_proprio_traj(16)

        pred_sim_params = []
        actual_params = []
        for traj in obs_list:
            pred_sim_params.append(self.forward(traj['image']).mean[0])
            actual_params.append(traj['sim_params'][-1]) #take last obs

        loss = F.mse_loss(torch.stack(pred_sim_params), torch.stack(actual_params))
        L.log('train_sim_params/loss', loss, step)

        # Optimize the critic
        self.sim_param_optimizer.zero_grad()
        loss.backward()
        self.sim_param_optimizer.step()

