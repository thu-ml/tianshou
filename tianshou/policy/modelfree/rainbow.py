from dataclasses import dataclass
from typing import TypeVar

from torch import nn

from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import C51
from tianshou.policy.modelfree.c51 import C51TrainingStats
from tianshou.utils.net.discrete import NoisyLinear


@dataclass(kw_only=True)
class RainbowTrainingStats(C51TrainingStats):
    loss: float


TRainbowTrainingStats = TypeVar("TRainbowTrainingStats", bound=RainbowTrainingStats)


class RainbowDQN(C51[TRainbowTrainingStats]):
    """Implementation of Rainbow DQN. arXiv:1710.02298.

    .. seealso::

        Please refer to :class:`~tianshou.policy.C51Policy` for more detailed
        explanation.
    """

    @staticmethod
    def _sample_noise(model: nn.Module) -> bool:
        """Sample the random noises of NoisyLinear modules in the model.

        Returns True if at least one NoisyLinear submodule was found.

        :param model: a PyTorch module which may have NoisyLinear submodules.
        :returns: True if model has at least one NoisyLinear submodule;
            otherwise, False.
        """
        sampled_any_noise = False
        for m in model.modules():
            if isinstance(m, NoisyLinear):
                m.sample()
                sampled_any_noise = True
        return sampled_any_noise

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> TRainbowTrainingStats:
        self._sample_noise(self.policy.model)
        if self.use_target_network and self._sample_noise(self.model_old):  # type: ignore
            assert self.model_old is not None
            self.model_old.train()  # so that NoisyLinear takes effect
        return super()._update_with_batch(batch)
