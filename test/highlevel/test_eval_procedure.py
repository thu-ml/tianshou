from test.highlevel.env_factory import ContinuousTestEnvFactory, DiscreteTestEnvFactory

import pytest

from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.experiment import (
    A2CExperimentBuilder,
    DDPGExperimentBuilder,
    DiscreteSACExperimentBuilder,
    DQNExperimentBuilder,
    IQNExperimentBuilder,
    PGExperimentBuilder,
    PPOExperimentBuilder,
    REDQExperimentBuilder,
    SACExperimentBuilder,
    TD3ExperimentBuilder,
    TRPOExperimentBuilder,
    ExperimentConfig,
    EvaluationProtocalExperimentConfig
)
from examples.mujoco.mujoco_env import MujocoEnvFactory


def test_standard_experiment_config():
    experiment_config = ExperimentConfig
    sampling_config = SamplingConfig(
        num_epochs=1,
        step_per_epoch=100,
        num_train_envs=2,
        num_test_envs=2,
    )
    env_factory = MujocoEnvFactory(task="Ant-v4",
                                   seed=experiment_config.seed,
                                   obs_norm=True,)

    ppo = PPOExperimentBuilder(
        experiment_config=experiment_config,
        env_factory=env_factory,
        sampling_config=sampling_config,
    )
    experiment = ppo.build()
    experiment.run("test")
    print(experiment)


@pytest.mark.parametrize("experiment_config",
    [
        EvaluationProtocalExperimentConfig(
             persistence_enabled=False,
             train_seed_mechanism="consecutive",
             test_seeds=(2,3)),
         EvaluationProtocalExperimentConfig(persistence_enabled=False, train_seed_mechanism="repeat",
                                             test_seeds=(2,3)),
        EvaluationProtocalExperimentConfig(persistence_enabled=False, train_seed_mechanism=None,
                                           test_seeds=(2,3)),

    ],
)
def test_experiment_builder_continuous_default_params(experiment_config):
    sampling_config = SamplingConfig(
        num_epochs=1,
        step_per_epoch=100,
        num_train_envs=2,
        num_test_envs=2,
    )
    env_factory = MujocoEnvFactory(task="Ant-v4",
                                   seed=experiment_config.seed,
                                   obs_norm=True,
                                   train_seed_mechanism=experiment_config.train_seed_mechanism,
                                   test_seeds=experiment_config.test_seeds)

    ppo = PPOExperimentBuilder(
        experiment_config=experiment_config,
        env_factory=env_factory,
        sampling_config=sampling_config,
    )
    experiment = ppo.build()
    experiment.run("test")
    print(experiment)
