import numpy as np


class MovAvg(object):
    def __init__(self, size=100):
        super().__init__()
        self.size = size
        self.cache = []

    def add(self, x):
        if hasattr(x, 'detach'):
            # which means x is torch.Tensor (?)
            x = x.detach().cpu().numpy()
        if x != np.inf:
            self.cache.append(x)
        if self.size > 0 and len(self.cache) > self.size:
            self.cache = self.cache[-self.size:]
        return self.get()

    def get(self):
        if len(self.cache) == 0:
            return 0
        return np.mean(self.cache)
