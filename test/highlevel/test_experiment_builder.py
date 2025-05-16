from test.highlevel.env_factory import ContinuousTestEnvFactory, DiscreteTestEnvFactory

import pytest

from tianshou.highlevel.config import (
    OffPolicyTrainingConfig,
    OnPolicyTrainingConfig,
)
from tianshou.highlevel.experiment import (
    A2CExperimentBuilder,
    DDPGExperimentBuilder,
    DiscreteSACExperimentBuilder,
    DQNExperimentBuilder,
    ExperimentBuilder,
    ExperimentConfig,
    IQNExperimentBuilder,
    OffPolicyExperimentBuilder,
    OnPolicyExperimentBuilder,
    PPOExperimentBuilder,
    REDQExperimentBuilder,
    ReinforceExperimentBuilder,
    SACExperimentBuilder,
    TD3ExperimentBuilder,
    TRPOExperimentBuilder,
)


def create_training_config(
    builder_cls: type[ExperimentBuilder],
    num_epochs: int = 1,
    epoch_num_steps: int = 100,
    num_train_envs: int = 2,
    num_test_envs: int = 2,
) -> OffPolicyTrainingConfig | OnPolicyTrainingConfig:
    if issubclass(builder_cls, OffPolicyExperimentBuilder):
        return OffPolicyTrainingConfig(
            max_epochs=num_epochs,
            epoch_num_steps=epoch_num_steps,
            num_train_envs=num_train_envs,
            num_test_envs=num_test_envs,
        )
    elif issubclass(builder_cls, OnPolicyExperimentBuilder):
        return OnPolicyTrainingConfig(
            max_epochs=num_epochs,
            epoch_num_steps=epoch_num_steps,
            num_train_envs=num_train_envs,
            num_test_envs=num_test_envs,
        )
    else:
        raise ValueError


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
        ReinforceExperimentBuilder,
    ],
)
def test_experiment_builder_continuous_default_params(builder_cls: type[ExperimentBuilder]) -> None:
    env_factory = ContinuousTestEnvFactory()
    training_config = create_training_config(
        builder_cls,
        num_epochs=1,
        epoch_num_steps=100,
        num_train_envs=2,
        num_test_envs=2,
    )
    experiment_config = ExperimentConfig(persistence_enabled=False)
    builder = builder_cls(
        experiment_config=experiment_config,
        env_factory=env_factory,
        training_config=training_config,
    )
    experiment = builder.build()
    experiment.run(run_name="test")
    print(experiment)


@pytest.mark.parametrize(
    "builder_cls",
    [
        ReinforceExperimentBuilder,
        PPOExperimentBuilder,
        A2CExperimentBuilder,
        DQNExperimentBuilder,
        DiscreteSACExperimentBuilder,
        IQNExperimentBuilder,
    ],
)
def test_experiment_builder_discrete_default_params(builder_cls: type[ExperimentBuilder]) -> None:
    env_factory = DiscreteTestEnvFactory()
    training_config = create_training_config(
        builder_cls,
        num_epochs=1,
        epoch_num_steps=100,
        num_train_envs=2,
        num_test_envs=2,
    )
    builder = builder_cls(
        experiment_config=ExperimentConfig(persistence_enabled=False),
        env_factory=env_factory,
        training_config=training_config,
    )
    experiment = builder.build()
    experiment.run(run_name="test")
    print(experiment)
