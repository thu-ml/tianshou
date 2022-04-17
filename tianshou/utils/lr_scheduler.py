from typing import Dict, List

import torch


class MultipleLRSchedulers:
    """A wrapper for multiple learning rate schedulers.

    Every time :meth:`~tianshou.utils.MultipleLRSchedulers.step()` is called, \
    it calls the step() method of each of the schedulers that it contains.
    """

    def __init__(self, *args: torch.optim.lr_scheduler.LambdaLR):
        self.schedulers = args

    def step(self) -> None:
        """Take a step in each of the learning rate schedulers."""
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> List[Dict]:
        """Get state dictionaries for each of the learning rate schedulers.

        :return: A list of state dictionaries of learning rate schedulers
        """
        return [s.state_dict() for s in self.schedulers]

    def load_state_dict(self, state_dict: List[Dict]) -> None:
        """Load states from  dictionaries.

        :param List[Dict] state_dict: A list of learning rate scheduler
            state dictionaries, in the same order as the schedulers.
        """
        for (s, sd) in zip(self.schedulers, state_dict):
            s.__dict__.update(sd)
