from test.highlevel.env_factory import ContinuousTestEnvFactory

import pytest

from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.experiment import (
    A2CExperimentBuilder,
    DDPGExperimentBuilder,
    ExperimentConfig,
    PGExperimentBuilder,
    PPOExperimentBuilder,
    REDQExperimentBuilder,
    SACExperimentBuilder,
    TD3ExperimentBuilder,
    TRPOExperimentBuilder,
)


@pytest.mark.parametrize(
    "builder_cls",
    [
        PPOExperimentBuilder,
        A2CExperimentBuilder,
        SACExperimentBuilder,
        DDPGExperimentBuilder,
        TD3ExperimentBuilder,
        # NPGExperimentBuilder,  # TODO test fails non-deterministically
        REDQExperimentBuilder,
        TRPOExperimentBuilder,
        PGExperimentBuilder,
    ],
)
def test_experiment_builder_continuous_default_params(builder_cls):
    env_factory = ContinuousTestEnvFactory()
    sampling_config = SamplingConfig(num_epochs=1, step_per_epoch=100)
    experiment_config = ExperimentConfig()
    builder = builder_cls(
        experiment_config=experiment_config,
        env_factory=env_factory,
        sampling_config=sampling_config,
    )
    experiment = builder.build()
    experiment.run("test")
    print(experiment)
