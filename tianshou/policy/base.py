from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """docstring for BasePolicy"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def act(self, batch, hidden_state=None):
        # return {policy, action, hidden}
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def reset(self):
        pass

    @staticmethod
    def process_fn(batch, buffer, indice):
        return batch

    def exploration(self):
        pass
