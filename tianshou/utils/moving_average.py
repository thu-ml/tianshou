import torch
import numpy as np


class MovAvg(object):
    def __init__(self, size=100):
        super().__init__()
        self.size = size
        self.cache = []

    def add(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(x, list):
            for _ in x:
                if _ != np.inf:
                    self.cache.append(_)
        elif x != np.inf:
            self.cache.append(x)
        if self.size > 0 and len(self.cache) > self.size:
            self.cache = self.cache[-self.size:]
        return self.get()

    def get(self):
        if len(self.cache) == 0:
            return 0
        return np.mean(self.cache)

    def mean(self):
        return self.get()

    def std(self):
        if len(self.cache) == 0:
            return 0
        return np.std(self.cache)
