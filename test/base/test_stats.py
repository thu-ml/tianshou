import pytest

from tianshou.policy.base import TrainingStats, TrainingStatsWrapper


class DummyTrainingStatsWrapper(TrainingStatsWrapper):
    def __init__(self, wrapped_stats: TrainingStats, *, dummy_field: int):
        self.dummy_field = dummy_field
        super().__init__(wrapped_stats)


class TestStats:
    @staticmethod
    def test_training_stats_wrapper():
        train_stats = TrainingStats(train_time=1.0)
        train_stats.loss_field = 12

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
        assert wrapped_train_stats.get_loss_stats_dict() == {"loss_field": 12, "dummy_field": 42}

        # new fields can't be added
        with pytest.raises(AttributeError):
            wrapped_train_stats.new_loss_field = 90

        # existing fields, wrapped and not-wrapped, can be mutated
        wrapped_train_stats.loss_field = 13
        wrapped_train_stats.dummy_field = 43
        assert wrapped_train_stats.wrapped_stats.loss_field == wrapped_train_stats.loss_field == 13
