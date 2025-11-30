from typing import cast

import numpy as np
import pytest
import torch
from torch.distributions import Categorical, Normal

from tianshou.algorithm.algorithm_base import TrainingStats, TrainingStatsWrapper
from tianshou.data import Batch, CollectStats
from tianshou.data.collector import CollectStepBatchProtocol, get_stddev_from_dist


class DummyTrainingStatsWrapper(TrainingStatsWrapper):
    def __init__(self, wrapped_stats: TrainingStats, *, dummy_field: int) -> None:
        self.dummy_field = dummy_field
        super().__init__(wrapped_stats)


class TestStats:
    @staticmethod
    def test_training_stats_wrapper() -> None:
        train_stats = TrainingStats(train_time=1.0)

        setattr(train_stats, "loss_field", 12)  # noqa: B010

        wrapped_train_stats = DummyTrainingStatsWrapper(train_stats, dummy_field=42)

        # basic readout
        assert wrapped_train_stats.train_time == 1.0
        assert wrapped_train_stats.loss_field == 12

        # mutation of TrainingStats fields
        wrapped_train_stats.train_time = 2.0
        wrapped_train_stats.smoothed_loss["foo"] = 50
        assert wrapped_train_stats.train_time == 2.0
        assert wrapped_train_stats.smoothed_loss["foo"] == 50

        # loss stats dict
        assert wrapped_train_stats.get_loss_stats_dict() == {
            "loss_field": 12,
            "dummy_field": 42,
        }

        # new fields can't be added
        with pytest.raises(AttributeError):
            wrapped_train_stats.new_loss_field = 90

        # existing fields, wrapped and not-wrapped, can be mutated
        wrapped_train_stats.loss_field = 13
        wrapped_train_stats.dummy_field = 43
        assert hasattr(
            wrapped_train_stats.wrapped_stats,
            "loss_field",
        ), "Attribute `loss_field` not found in `wrapped_train_stats.wrapped_stats`."
        assert hasattr(
            wrapped_train_stats,
            "loss_field",
        ), "Attribute `loss_field` not found in `wrapped_train_stats`."
        assert wrapped_train_stats.wrapped_stats.loss_field == wrapped_train_stats.loss_field == 13

    @staticmethod
    @pytest.mark.parametrize(
        "act,dist",
        (
            (np.array(1), Categorical(probs=torch.tensor([0.5, 0.5]))),
            (np.array([1, 2, 3]), Normal(torch.zeros(3), torch.ones(3))),
        ),
    )
    def test_collect_stats_update_at_step(
        act: np.ndarray,
        dist: torch.distributions.Distribution,
    ) -> None:
        step_batch = cast(
            CollectStepBatchProtocol,
            Batch(
                info={},
                obs=np.array([1, 2, 3]),
                obs_next=np.array([4, 5, 6]),
                act=act,
                rew=np.array(1.0),
                done=np.array(False),
                terminated=np.array(False),
                dist=dist,
            ).to_at_least_2d(),
        )
        stats = CollectStats()
        for _ in range(10):
            stats.update_at_step_batch(step_batch)
        stats.refresh_all_sequence_stats()
        assert stats.n_collected_steps == 10
        assert stats.pred_dist_std_array is not None
        assert np.allclose(stats.pred_dist_std_array, get_stddev_from_dist(dist))
        assert stats.pred_dist_std_array_stat is not None
        assert stats.pred_dist_std_array_stat[0].mean == get_stddev_from_dist(dist)[0].item()
