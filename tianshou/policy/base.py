from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """docstring for BasePolicy"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def act(self, batch, hidden_state=None):
        # return Batch(policy, action, hidden)
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def reset(self):
        pass

    def process_fn(self, batch, buffer, indice):
        return batch

    def sync_weights(self):
        pass

    def exploration(self):
        pass
