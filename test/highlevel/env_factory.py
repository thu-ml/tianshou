from tianshou.highlevel.env import (
    EnvFactoryGymnasium,
    VectorEnvType,
)


class DiscreteTestEnvFactory(EnvFactoryGymnasium):
    def __init__(self):
        super().__init__(task="CartPole-v0", seed=42, venv_type=VectorEnvType.DUMMY)


class ContinuousTestEnvFactory(EnvFactoryGymnasium):
    def __init__(self):
        super().__init__(task="Pendulum-v1", seed=42, venv_type=VectorEnvType.DUMMY)
