from torch import nn
from abc import ABC, abstractmethod


class BasePolicy(ABC, nn.Module):
    """docstring for BasePolicy"""

    def __init__(self):
        super().__init__()

    def process_fn(self, batch, buffer, indice):
        return batch

    @abstractmethod
    def __call__(self, batch, state=None):
        # return Batch(logits=..., act=..., state=None, ...)
        pass

    @abstractmethod
    def learn(self, batch, batch_size=None):
        # return a dict which includes loss and its name
        pass

    def sync_weight(self):
        pass
