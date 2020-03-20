import torch
import numpy as np


class Batch(object):
    """Suggested keys: [obs, act, rew, done, obs_next, info]"""

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def __getitem__(self, index):
        b = Batch()
        for k in self.__dict__.keys():
            if self.__dict__[k] is not None:
                b.update(**{k: self.__dict__[k][index]})
        return b

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def append(self, batch):
        assert isinstance(batch, Batch), 'Only append Batch is allowed!'
        for k in batch.__dict__.keys():
            if batch.__dict__[k] is None:
                continue
            if not hasattr(self, k) or self.__dict__[k] is None:
                self.__dict__[k] = batch.__dict__[k]
            elif isinstance(batch.__dict__[k], np.ndarray):
                self.__dict__[k] = np.concatenate([
                    self.__dict__[k], batch.__dict__[k]])
            elif isinstance(batch.__dict__[k], torch.Tensor):
                self.__dict__[k] = torch.cat([
                    self.__dict__[k], batch.__dict__[k]])
            elif isinstance(batch.__dict__[k], list):
                self.__dict__[k] += batch.__dict__[k]
            else:
                raise TypeError(
                    'No support for append with type {} in class Batch.'
                    .format(type(batch.__dict__[k])))

    def split(self, size=None, permute=True):
        length = min([
            len(self.__dict__[k]) for k in self.__dict__.keys()
            if self.__dict__[k] is not None])
        if size is None:
            size = length
        temp = 0
        if permute:
            index = np.random.permutation(length)
        else:
            index = np.arange(length)
        while temp < length:
            yield self[index[temp:temp + size]]
            temp += size
