from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Union, Optional

from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.trainer import test_episode


def offline_trainer(
    policy: BasePolicy,
    buffer: ReplayBuffer,
    test_collector: Collector,
    epochs: int,
    batch_size: int,
    episode_per_test: int,
    writer: Optional[SummaryWriter] = None,
    test_frequency: int = 1,
) -> Dict[str, Union[float, str]]:
    best_reward = -1
    best_policy = policy
    total_iter = 0
    iter_per_epoch = len(buffer) // batch_size

    for epoch in range(1, 1 + epochs):
        for iter in range(iter_per_epoch):
            total_iter += 1
            loss = policy.update(batch_size, buffer)
            if total_iter % test_frequency == 0:
                writer.add_scalar(
                    "train/loss",
                    loss["loss"],
                    global_step=total_iter,
                )

                test_result = test_episode(
                    policy,
                    test_collector,
                    None,
                    epoch,
                    episode_per_test,
                    writer,
                    total_iter,
                )

                if best_reward < test_result["rew"]:
                    best_reward = test_result["rew"]
                    best_policy.load_state_dict(policy.state_dict())

                print(f"------- epoch: {epoch}, iter: {total_iter} --------")
                print("loss:", loss["loss"])
                print("test_result:", test_result)
                print("best_reward:", best_reward)

    return {"best_reward": best_reward, "best_policy": best_policy}
