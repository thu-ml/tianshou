from test.highlevel.env_factory import ContinuousTestEnvFactory, DiscreteTestEnvFactory

import pytest

from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.experiment import (
    PPOExperimentBuilder,
    ExperimentConfig,
    EvaluationProtocolExperimentConfig,
    TrainSeedMechanism
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
        EvaluationProtocolExperimentConfig(
             persistence_enabled=False,
             train_seed_mechanism=TrainSeedMechanism.CONSECUTIVE,
             test_seeds=(2,3)),
         EvaluationProtocolExperimentConfig(persistence_enabled=False, train_seed_mechanism=TrainSeedMechanism.REPEAT,
                                             test_seeds=(2,3)),
        EvaluationProtocolExperimentConfig(persistence_enabled=False, train_seed_mechanism=TrainSeedMechanism.NONE,
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
    experiment.run("test",
                   record_seed_of_transition_to_buffer_test=experiment_config.record_seed_of_transition_to_buffer_test)
    print(experiment)
