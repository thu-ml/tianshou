from dataclasses import dataclass

from torch import nn

from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy.modelfree.c51 import C51, C51Policy
from tianshou.policy.modelfree.pg import LossSequenceTrainingStats
from tianshou.policy.optim import OptimizerFactory
from tianshou.utils.lagged_network import EvalModeModuleWrapper
from tianshou.utils.net.discrete import NoisyLinear


@dataclass(kw_only=True)
class RainbowTrainingStats:
    loss: float


class RainbowDQN(C51):
    """Implementation of Rainbow DQN. arXiv:1710.02298."""

    def __init__(
        self,
        *,
        policy: C51Policy,
        optim: OptimizerFactory,
        gamma: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
    ) -> None:
        """
        :param policy: a policy following the rules (s -> action_values_BA)
        :param optim: the optimizer factory for the policy's model.
        :param gamma: the discount factor in [0, 1] for future rewards.
            This determines how much future rewards are valued compared to immediate ones.
            Lower values (closer to 0) make the agent focus on immediate rewards, creating "myopic"
            behavior. Higher values (closer to 1) make the agent value long-term rewards more,
            potentially improving performance in tasks where delayed rewards are important but
            increasing training variance by incorporating more environmental stochasticity.
            Typically set between 0.9 and 0.99 for most reinforcement learning tasks
        :param estimation_step: the number of future steps (> 0) to consider when computing temporal
            difference (TD) targets. Controls the balance between TD learning and Monte Carlo methods:
            higher values reduce bias (by relying less on potentially inaccurate value estimates)
            but increase variance (by incorporating more environmental stochasticity and reducing
            the averaging effect). A value of 1 corresponds to standard TD learning with immediate
            bootstrapping, while very large values approach Monte Carlo-like estimation that uses
            complete episode returns.
        :param target_update_freq: the number of training iterations between each complete update of
            the target network.
            Controls how frequently the target Q-network parameters are updated with the current
            Q-network values.
            A value of 0 disables the target network entirely, using only a single network for both
            action selection and bootstrap targets.
            Higher values provide more stable learning targets but slow down the propagation of new
            value estimates. Lower positive values allow faster learning but may lead to instability
            due to rapidly changing targets.
            Typically set between 100-10000 for DQN variants, with exact values depending on environment
            complexity.
        """
        super().__init__(
            policy=policy,
            optim=optim,
            gamma=gamma,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
        )

        # Remove the wrapper that forces eval mode for the target network,
        # because Rainbow requires it to be set to train mode for sampling noise
        # in NoisyLinear layers to take effect.
        if self.use_target_network:
            assert isinstance(self.model_old, EvalModeModuleWrapper)
            self.model_old = self.model_old.module

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
    ) -> LossSequenceTrainingStats:
        self._sample_noise(self.policy.model)
        if self.use_target_network:
            self._sample_noise(self.model_old)
        return super()._update_with_batch(batch)
