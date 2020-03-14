import torch
from torch import nn
from copy import deepcopy

from tianshou.data import Batch
from tianshou.policy import BasePolicy


class DQNPolicy(BasePolicy, nn.Module):
    """docstring for DQNPolicy"""

    def __init__(self, model, discount_factor=0.99, estimation_step=1,
                 use_target_network=True):
        super().__init__()
        self.model = model
        self._gamma = discount_factor
        self._n_step = estimation_step
        self._target = use_target_network
        if use_target_network:
            self.model_old = deepcopy(self.model)

    def act(self, batch, hidden_state=None):
        batch_result = Batch()
        return batch_result

    def sync_weights(self):
        if self._use_target_network:
            for old, new in zip(
                    self.model_old.parameters(), self.model.parameters()):
                old.data.copy_(new.data)

    def process_fn(self, batch, buffer, indice):
        return batch
