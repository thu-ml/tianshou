from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """docstring for BasePolicy"""

    def __init__(self):
        super().__init__()
        self.model = None

    @abstractmethod
    def __call__(self, batch, hidden_state=None):
        # return Batch(act=np.array(), state=None, ...)
        pass

    @abstractmethod
    def learn(self, batch):
        pass

    def process_fn(self, batch, buffer, indice):
        return batch

    def sync_weight(self):
        pass
