from tianshou.highlevel.env import (
    EnvFactoryRegistered,
    VectorEnvType,
)


class DiscreteTestEnvFactory(EnvFactoryRegistered):
    def __init__(self) -> None:
        super().__init__(
            task="CartPole-v1",
            venv_type=VectorEnvType.DUMMY,
        )


class ContinuousTestEnvFactory(EnvFactoryRegistered):
    def __init__(self) -> None:
        super().__init__(
            task="Pendulum-v1",
            venv_type=VectorEnvType.DUMMY,
        )
