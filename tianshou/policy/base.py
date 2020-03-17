from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """docstring for BasePolicy"""

    def __init__(self):
        super().__init__()
        self.model = None

    def process_fn(self, batch, buffer, indice):
        return batch

    @abstractmethod
    def __call__(self, batch, state=None):
        # return Batch(logits=..., act=np.array(), state=None, ...)
        pass

    @abstractmethod
    def learn(self, batch, batch_size=None):
        pass

    def sync_weight(self):
        pass
