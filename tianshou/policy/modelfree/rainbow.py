from dataclasses import dataclass
from typing import Any, TypeVar

from torch import nn

from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import C51Policy
from tianshou.policy.modelfree.c51 import C51TrainingStats
from tianshou.utils.net.discrete import NoisyLinear


# TODO: this is a hacky thing interviewing side-effects and a return. Should improve.
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


@dataclass(kw_only=True)
class RainbowTrainingStats(C51TrainingStats):
    loss: float


TRainbowTrainingStats = TypeVar("TRainbowTrainingStats", bound=RainbowTrainingStats)


# TODO: is this class worth keeping? It barely does anything
class RainbowPolicy(C51Policy[TRainbowTrainingStats]):
    """Implementation of Rainbow DQN. arXiv:1710.02298.

    Same parameters as :class:`~tianshou.policy.C51Policy`.

    .. seealso::

        Please refer to :class:`~tianshou.policy.C51Policy` for more detailed
        explanation.
    """

    def learn(
        self,
        batch: RolloutBatchProtocol,
        *args: Any,
        **kwargs: Any,
    ) -> TRainbowTrainingStats:
        _sample_noise(self.model)
        if self._target and _sample_noise(self.model_old):
            self.model_old.train()  # so that NoisyLinear takes effect
        return super().learn(batch, **kwargs)
