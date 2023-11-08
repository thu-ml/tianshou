from tianshou.highlevel.trainer import (
    TrainerEpochCallbackTest,
    TrainerEpochCallbackTrain,
    TrainingContext,
)
from tianshou.policy import DQNPolicy


class TestEpochCallbackDQNSetEps(TrainerEpochCallbackTest):
    def __init__(self, eps_test: float):
        self.eps_test = eps_test

    def callback(self, epoch: int, env_step: int, context: TrainingContext) -> None:
        policy: DQNPolicy = context.policy
        policy.set_eps(self.eps_test)


class TrainEpochCallbackNatureDQNEpsLinearDecay(TrainerEpochCallbackTrain):
    def __init__(self, eps_train: float, eps_train_final: float):
        self.eps_train = eps_train
        self.eps_train_final = eps_train_final

    def callback(self, epoch: int, env_step: int, context: TrainingContext) -> None:
        policy: DQNPolicy = context.policy
        logger = context.logger
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = self.eps_train - env_step / 1e6 * (self.eps_train - self.eps_train_final)
        else:
            eps = self.eps_train_final
        policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})
