import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import Algorithm
from tianshou.policy.base import (
    OffPolicyAlgorithm,
    OffPolicyWrapperAlgorithm,
    OnPolicyAlgorithm,
    OnPolicyWrapperAlgorithm,
    TPolicy,
    TrainingStats,
    TrainingStatsWrapper,
)
from tianshou.policy.optim import OptimizerFactory
from tianshou.utils.net.discrete import IntrinsicCuriosityModule


class ICMTrainingStats(TrainingStatsWrapper):
    def __init__(
        self,
        wrapped_stats: TrainingStats,
        *,
        icm_loss: float,
        icm_forward_loss: float,
        icm_inverse_loss: float,
    ) -> None:
        self.icm_loss = icm_loss
        self.icm_forward_loss = icm_forward_loss
        self.icm_inverse_loss = icm_inverse_loss
        super().__init__(wrapped_stats)


class _ICMMixin:
    """Implementation of the Intrinsic Curiosity Module (ICM) algorithm. arXiv:1705.05363."""

    def __init__(
        self,
        *,
        model: IntrinsicCuriosityModule,
        optim: Algorithm.Optimizer,
        lr_scale: float,
        reward_scale: float,
        forward_loss_weight: float,
    ) -> None:
        """
        :param model: the ICM model.
        :param optim: the optimizer factory.
        :param lr_scale: the scaling factor for ICM learning.
        :param forward_loss_weight: the weight for forward model loss.
        """
        self.model = model
        self.optim = optim
        self.lr_scale = lr_scale
        self.reward_scale = reward_scale
        self.forward_loss_weight = forward_loss_weight

    def _icm_preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> None:
        mse_loss, act_hat = self.model(batch.obs, batch.act, batch.obs_next)
        batch.policy = Batch(orig_rew=batch.rew, act_hat=act_hat, mse_loss=mse_loss)
        batch.rew += to_numpy(mse_loss * self.reward_scale)

    @staticmethod
    def _icm_postprocess_batch(batch: BatchProtocol) -> None:
        # restore original reward
        batch.rew = batch.policy.orig_rew

    def _icm_update(
        self,
        batch: RolloutBatchProtocol,
        original_stats: TrainingStats,
    ) -> ICMTrainingStats:
        act_hat = batch.policy.act_hat
        act = to_torch(batch.act, dtype=torch.long, device=act_hat.device)
        inverse_loss = F.cross_entropy(act_hat, act).mean()
        forward_loss = batch.policy.mse_loss.mean()
        loss = (
            (1 - self.forward_loss_weight) * inverse_loss + self.forward_loss_weight * forward_loss
        ) * self.lr_scale
        self.optim.step(loss)

        return ICMTrainingStats(
            original_stats,
            icm_loss=loss.item(),
            icm_forward_loss=forward_loss.item(),
            icm_inverse_loss=inverse_loss.item(),
        )


class ICMOffPolicyWrapper(OffPolicyWrapperAlgorithm[TPolicy], _ICMMixin):
    """Implementation of the Intrinsic Curiosity Module (ICM) algorithm for off-policy learning. arXiv:1705.05363."""

    def __init__(
        self,
        *,
        wrapped_algorithm: OffPolicyAlgorithm[TPolicy],
        model: IntrinsicCuriosityModule,
        optim: OptimizerFactory,
        lr_scale: float,
        reward_scale: float,
        forward_loss_weight: float,
    ) -> None:
        """
        :param wrapped_algorithm: the base algorithm to which we want to add the ICM.
        :param model: the ICM model.
        :param optim: the optimizer factory for the ICM model.
        :param lr_scale: the scaling factor for ICM learning.
        :param forward_loss_weight: the weight for forward model loss.
        """
        OffPolicyWrapperAlgorithm.__init__(
            self,
            wrapped_algorithm=wrapped_algorithm,
        )
        _ICMMixin.__init__(
            self,
            model=model,
            optim=self._create_optimizer(model, optim),
            lr_scale=lr_scale,
            reward_scale=reward_scale,
            forward_loss_weight=forward_loss_weight,
        )

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        self._icm_preprocess_batch(batch)
        return super()._preprocess_batch(batch, buffer, indices)

    def _postprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> None:
        super()._postprocess_batch(batch, buffer, indices)
        self._icm_postprocess_batch(batch)

    def _wrapper_update_with_batch(
        self,
        batch: RolloutBatchProtocol,
        original_stats: TrainingStats,
    ) -> ICMTrainingStats:
        return self._icm_update(batch, original_stats)


class ICMOnPolicyWrapper(OnPolicyWrapperAlgorithm[TPolicy], _ICMMixin):
    """Implementation of the Intrinsic Curiosity Module (ICM) algorithm for on-policy learning. arXiv:1705.05363."""

    def __init__(
        self,
        *,
        wrapped_algorithm: OnPolicyAlgorithm[TPolicy],
        model: IntrinsicCuriosityModule,
        optim: OptimizerFactory,
        lr_scale: float,
        reward_scale: float,
        forward_loss_weight: float,
    ) -> None:
        """
        :param wrapped_algorithm: the base algorithm to which we want to add the ICM.
        :param model: the ICM model.
        :param optim: the optimizer factory for the ICM model.
        :param lr_scale: the scaling factor for ICM learning.
        :param forward_loss_weight: the weight for forward model loss.
        """
        OnPolicyWrapperAlgorithm.__init__(
            self,
            wrapped_algorithm=wrapped_algorithm,
        )
        _ICMMixin.__init__(
            self,
            model=model,
            optim=self._create_optimizer(model, optim),
            lr_scale=lr_scale,
            reward_scale=reward_scale,
            forward_loss_weight=forward_loss_weight,
        )

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        self._icm_preprocess_batch(batch)
        return super()._preprocess_batch(batch, buffer, indices)

    def _postprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> None:
        super()._postprocess_batch(batch, buffer, indices)
        self._icm_postprocess_batch(batch)

    def _wrapper_update_with_batch(
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
        original_stats: TrainingStats,
    ) -> ICMTrainingStats:
        return self._icm_update(batch, original_stats)
