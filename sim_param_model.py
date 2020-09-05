import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
from encoder import make_encoder, PixelEncoder
import data_augs as rad

class SimParamModel(nn.Module):
    def __init__(self, shape, layers, units, dist='normal', act=nn.elu):
        self._shape = shape
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act

        trunk = []
        for index in range(self._layers):
            trunk.append(nn.Linear(self._units))
            trunk.append(self._act)
        trunk.append(nn.Linear(np.prod(self._shape)))

        self.trunk = nn.Sequential(*trunk)


    def forward(
        self, features
    ):
        x = features
        x = self.trunk(x)
        x = x.view(torch.cat([features.size()[:-1], self._shape], 0))

        if self._dist == 'normal':
            return torch.distributions.independent.Independent(torch.distributions.normal.Normal(x, 1), len(self._shape))
        if self._dist == 'binary':
            return torch.distributions.independent.Independent(torch.distributions.bernoulli.Bernoulli(x), len(self._shape))
        raise NotImplementedError(self._dist)
