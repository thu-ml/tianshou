from test.highlevel.env_factory import DiscreteTestEnvFactory

import pytest

from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.experiment import (
    A2CExperimentBuilder,
    DiscreteSACExperimentBuilder,
    DQNExperimentBuilder,
    ExperimentConfig,
    IQNExperimentBuilder,
    PPOExperimentBuilder,
)


@pytest.mark.parametrize(
    "builder_cls",
    [
        PPOExperimentBuilder,
        A2CExperimentBuilder,
        DQNExperimentBuilder,
        DiscreteSACExperimentBuilder,
        IQNExperimentBuilder,
    ],
)
def test_experiment_builder_discrete_default_params(builder_cls):
    env_factory = DiscreteTestEnvFactory()
    sampling_config = SamplingConfig(num_epochs=1, step_per_epoch=100)
    builder = builder_cls(
        experiment_config=ExperimentConfig(persistence_enabled=False),
        env_factory=env_factory,
        sampling_config=sampling_config,
    )
    experiment = builder.build()
    experiment.run("test")
    print(experiment)
