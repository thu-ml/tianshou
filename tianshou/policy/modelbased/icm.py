import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy.base import (
    OffPolicyAlgorithm,
    OffPolicyWrapperAlgorithm,
    OnPolicyAlgorithm,
    OnPolicyWrapperAlgorithm,
    TLearningRateScheduler,
    TPolicy,
    TrainingStats,
    TrainingStatsWrapper,
    TTrainingStats,
)
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
        optim: torch.optim.Optimizer,
        lr_scale: float,
        reward_scale: float,
        forward_loss_weight: float,
    ) -> None:
        """
        :param model: the ICM model.
        :param optim: the optimizer for parameter `model`.
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
        self.optim.zero_grad()
        act_hat = batch.policy.act_hat
        act = to_torch(batch.act, dtype=torch.long, device=act_hat.device)
        inverse_loss = F.cross_entropy(act_hat, act).mean()
        forward_loss = batch.policy.mse_loss.mean()
        loss = (
            (1 - self.forward_loss_weight) * inverse_loss + self.forward_loss_weight * forward_loss
        ) * self.lr_scale
        loss.backward()
        self.optim.step()

        return ICMTrainingStats(
            original_stats,
            icm_loss=loss.item(),
            icm_forward_loss=forward_loss.item(),
            icm_inverse_loss=inverse_loss.item(),
        )


class ICMOffPolicyWrapper(
    OffPolicyWrapperAlgorithm[TPolicy, ICMTrainingStats, TTrainingStats], _ICMMixin
):
    """Implementation of the Intrinsic Curiosity Module (ICM) algorithm for off-policy learning. arXiv:1705.05363."""

    def __init__(
        self,
        *,
        wrapped_algorithm: OffPolicyAlgorithm[TPolicy, TTrainingStats],
        model: IntrinsicCuriosityModule,
        optim: torch.optim.Optimizer,
        lr_scale: float,
        reward_scale: float,
        forward_loss_weight: float,
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        """
        :param wrapped_algorithm: the base algorithm to which we want to add the ICM.
        :param model: the ICM model.
        :param optim: the optimizer for parameter `model`.
        :param lr_scale: the scaling factor for ICM learning.
        :param forward_loss_weight: the weight for forward model loss.
        :param lr_scheduler: if not None, will be called in `policy.update()`.
        """
        OffPolicyWrapperAlgorithm.__init__(
            self,
            wrapped_algorithm=wrapped_algorithm,
            lr_scheduler=lr_scheduler,
        )
        _ICMMixin.__init__(
            self,
            model=model,
            optim=optim,
            lr_scale=lr_scale,
            reward_scale=reward_scale,
            forward_loss_weight=forward_loss_weight,
        )

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        self._icm_preprocess_batch(batch)
        return super().process_fn(batch, buffer, indices)

    def post_process_fn(
        self,
        batch: BatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> None:
        super().post_process_fn(batch, buffer, indices)
        self._icm_postprocess_batch(batch)

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> ICMTrainingStats:
        wrapped_stats = super()._update_with_batch(batch)
        return self._icm_update(batch, wrapped_stats)


class ICMOnPolicyWrapper(
    OnPolicyWrapperAlgorithm[TPolicy, ICMTrainingStats, TTrainingStats], _ICMMixin
):
    """Implementation of the Intrinsic Curiosity Module (ICM) algorithm for on-policy learning. arXiv:1705.05363."""

    def __init__(
        self,
        *,
        wrapped_algorithm: OnPolicyAlgorithm[TPolicy, TTrainingStats],
        model: IntrinsicCuriosityModule,
        optim: torch.optim.Optimizer,
        lr_scale: float,
        reward_scale: float,
        forward_loss_weight: float,
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        """
        :param wrapped_algorithm: the base algorithm to which we want to add the ICM.
        :param model: the ICM model.
        :param optim: the optimizer for parameter `model`.
        :param lr_scale: the scaling factor for ICM learning.
        :param forward_loss_weight: the weight for forward model loss.
        :param lr_scheduler: if not None, will be called in `policy.update()`.
        """
        OnPolicyWrapperAlgorithm.__init__(
            self,
            wrapped_algorithm=wrapped_algorithm,
            lr_scheduler=lr_scheduler,
        )
        _ICMMixin.__init__(
            self,
            model=model,
            optim=optim,
            lr_scale=lr_scale,
            reward_scale=reward_scale,
            forward_loss_weight=forward_loss_weight,
        )

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        self._icm_preprocess_batch(batch)
        return super().process_fn(batch, buffer, indices)

    def post_process_fn(
        self,
        batch: BatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> None:
        super().post_process_fn(batch, buffer, indices)
        self._icm_postprocess_batch(batch)

    def _update_with_batch(
        self, batch: RolloutBatchProtocol, batch_size: int | None, repeat: int
    ) -> ICMTrainingStats:
        wrapped_stats = super()._update_with_batch(batch, batch_size=batch_size, repeat=repeat)
        return self._icm_update(batch, wrapped_stats)
