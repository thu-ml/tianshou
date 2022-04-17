from typing import List, Dict

import torch


class MultipleLRSchedulers:

    def __init__(self, *args: torch.optim.lr_scheduler.LambdaLR):
        self.schedulers = args

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> List[Dict]:
        return [s.state_dict() for s in self.schedulers]

    def load_state_dict(self, state_dict: List[Dict]) -> None:
        for (s, sd) in zip(self.schedulers, state_dict):
            s.__dict__.update(sd)
