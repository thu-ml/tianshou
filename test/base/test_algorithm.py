from typing import cast

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.optim.lr_scheduler import LRScheduler

from tianshou.algorithm.algorithm_base import (
    OffPolicyAlgorithm,
    OffPolicyWrapperAlgorithm,
    OnPolicyAlgorithm,
    OnPolicyWrapperAlgorithm,
    Policy,
    TrainingStats,
)
from tianshou.algorithm.optim import AdamOptimizerFactory, LRSchedulerFactoryLinear
from tianshou.data import Batch, ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol
from tianshou.utils.torch_utils import policy_within_training_step


def create_scheduled_optimizer(
    algorithm: OnPolicyAlgorithm | OffPolicyAlgorithm,
    model: torch.nn.Module,
) -> OnPolicyAlgorithm.Optimizer:
    optim_factory = AdamOptimizerFactory(lr=1.0).with_lr_scheduler_factory(
        LRSchedulerFactoryLinear(
            max_epochs=3,
            epoch_num_steps=1,
            collection_step_num_env_steps=1,
        ),
    )
    return algorithm._create_optimizer(model, optim_factory)


def create_replay_buffer() -> ReplayBuffer:
    buffer = ReplayBuffer(size=1)
    buffer.add(
        cast(
            RolloutBatchProtocol,
            Batch(
                obs=np.array([0.0]),
                act=0,
                rew=0.0,
                terminated=False,
                truncated=False,
                obs_next=np.array([0.0]),
                info={},
            ),
        ),
    )
    return buffer


class ScheduledOnPolicyAlgorithm(OnPolicyAlgorithm[Policy]):
    def __init__(self) -> None:
        super().__init__(policy=Policy(action_space=gym.spaces.Discrete(2)))
        self.model = torch.nn.Linear(1, 1)
        self.optim = create_scheduled_optimizer(self, self.model)

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
    ) -> TrainingStats:
        loss = self.model(torch.ones(1, 1)).sum()
        self.optim.step(loss)
        return TrainingStats()


class ScheduledOnPolicyWrapper(OnPolicyWrapperAlgorithm[Policy]):
    def __init__(self, wrapped_algorithm: OnPolicyAlgorithm[Policy]) -> None:
        super().__init__(wrapped_algorithm)
        self.model = torch.nn.Linear(1, 1)
        self.optim = create_scheduled_optimizer(self, self.model)

    def _wrapper_update_with_batch(
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
        original_stats: TrainingStats,
    ) -> TrainingStats:
        loss = self.model(torch.ones(1, 1)).sum()
        self.optim.step(loss)
        return original_stats


class ScheduledOffPolicyAlgorithm(OffPolicyAlgorithm[Policy]):
    def __init__(self) -> None:
        super().__init__(policy=Policy(action_space=gym.spaces.Discrete(2)))
        self.model = torch.nn.Linear(1, 1)
        self.optim = create_scheduled_optimizer(self, self.model)

    def _update_with_batch(self, batch: RolloutBatchProtocol) -> TrainingStats:
        loss = self.model(torch.ones(1, 1)).sum()
        self.optim.step(loss)
        return TrainingStats()


class ScheduledOffPolicyWrapper(OffPolicyWrapperAlgorithm[Policy]):
    def __init__(self, wrapped_algorithm: OffPolicyAlgorithm[Policy]) -> None:
        super().__init__(wrapped_algorithm)
        self.model = torch.nn.Linear(1, 1)
        self.optim = create_scheduled_optimizer(self, self.model)

    def _wrapper_update_with_batch(
        self,
        batch: RolloutBatchProtocol,
        original_stats: TrainingStats,
    ) -> TrainingStats:
        loss = self.model(torch.ones(1, 1)).sum()
        self.optim.step(loss)
        return original_stats


def assert_schedulers_stepped_once(
    schedulers: list[LRScheduler],
    initial_epochs: list[int],
) -> None:
    assert [scheduler.last_epoch for scheduler in schedulers] == [
        epoch + 1 for epoch in initial_epochs
    ]
    for scheduler in schedulers:
        assert scheduler.get_last_lr() == pytest.approx([2.0 / 3.0])


def test_nested_on_policy_wrappers_step_all_lr_schedulers_once() -> None:
    wrapped_algorithm = ScheduledOnPolicyAlgorithm()
    inner_wrapper = ScheduledOnPolicyWrapper(wrapped_algorithm)
    outer_wrapper = ScheduledOnPolicyWrapper(inner_wrapper)
    schedulers = [
        wrapped_algorithm.lr_schedulers[0],
        inner_wrapper.lr_schedulers[0],
        outer_wrapper.lr_schedulers[0],
    ]
    initial_epochs = [scheduler.last_epoch for scheduler in schedulers]

    with policy_within_training_step(outer_wrapper.policy):
        outer_wrapper.update(create_replay_buffer(), batch_size=1, repeat=1)

    assert_schedulers_stepped_once(schedulers, initial_epochs)


def test_nested_off_policy_wrappers_step_all_lr_schedulers_once() -> None:
    wrapped_algorithm = ScheduledOffPolicyAlgorithm()
    inner_wrapper = ScheduledOffPolicyWrapper(wrapped_algorithm)
    outer_wrapper = ScheduledOffPolicyWrapper(inner_wrapper)
    schedulers = [
        wrapped_algorithm.lr_schedulers[0],
        inner_wrapper.lr_schedulers[0],
        outer_wrapper.lr_schedulers[0],
    ]
    initial_epochs = [scheduler.last_epoch for scheduler in schedulers]

    with policy_within_training_step(outer_wrapper.policy):
        outer_wrapper.update(create_replay_buffer(), sample_size=1)

    assert_schedulers_stepped_once(schedulers, initial_epochs)
