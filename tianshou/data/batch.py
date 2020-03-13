import numpy as np


class Batch(object):
    """Suggested keys: [obs, act, rew, done, obs_next, info]"""

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

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
            elif isinstance(batch.__dict__[k], list):
                self.__dict__[k] += batch.__dict__[k]
            else:
                raise TypeError(
                    'No support append method with {} in class Batch.'
                    .format(type(batch.__dict__[k])))
