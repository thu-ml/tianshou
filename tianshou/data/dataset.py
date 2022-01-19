from typing import Tuple

import torch

from tianshou.data import Batch


class TransitionDataset(torch.utils.data.Dataset):
    """Construct transition dataset."""

    def __init__(
        self,
        batch: Batch,
    ) -> None:
        self.size = len(batch)
        observation = batch.obs
        action = batch.act
        reward = batch.rew[:, None]
        next_observation = batch.obs_next
        delta_observation = next_observation - observation

        self.inputs = torch.cat((observation, action), dim=-1)
        self.outputs = torch.cat((delta_observation, reward), dim=-1)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.outputs[idx]

    def __len__(self) -> int:
        return self.size


class TransitionEnsembleDataset(TransitionDataset):
    """Construct transition dataset with data randomly shuffled."""

    def __init__(
        self,
        batch: Batch,
        ensemble_size: int = 1,
    ) -> None:
        super().__init__(batch=batch, )

        indices = torch.randint(self.size, (ensemble_size, self.size))
        self.inputs = self.inputs[indices]
        self.outputs = self.outputs[indices]

    def __getitem__(self, idx):
        return self.inputs[:, idx, :], self.outputs[:, idx, :]
