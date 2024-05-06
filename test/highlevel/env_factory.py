from tianshou.highlevel.env import (
    EnvFactoryRegistered,
    VectorEnvType,
)


class DiscreteTestEnvFactory(EnvFactoryRegistered):
    def __init__(self) -> None:
        super().__init__(
            task="CartPole-v1",
            train_seed=42,
            test_seed=1337,
            venv_type=VectorEnvType.DUMMY,
        )


class ContinuousTestEnvFactory(EnvFactoryRegistered):
    def __init__(self) -> None:
        super().__init__(
            task="Pendulum-v1",
            train_seed=42,
            test_seed=1337,
            venv_type=VectorEnvType.DUMMY,
        )
