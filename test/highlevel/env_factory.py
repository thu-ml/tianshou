from tianshou.highlevel.env import (
    EnvFactoryRegistered,
    VectorEnvType,
)


class DiscreteTestEnvFactory(EnvFactoryRegistered):
    def __init__(self):
        super().__init__(task="CartPole-v0", seed=42, venv_type=VectorEnvType.DUMMY)


class ContinuousTestEnvFactory(EnvFactoryRegistered):
    def __init__(self):
        super().__init__(task="Pendulum-v1", seed=42, venv_type=VectorEnvType.DUMMY)
