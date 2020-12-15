import time
import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Union, Callable, Optional

from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.utils import tqdm_config, MovAvg
from tianshou.trainer import test_episode, gather_info


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
                    "train/loss", loss['loss'], global_step=total_iter)

                test_result = test_episode(
                    policy, test_collector, None,
                    epoch, episode_per_test, writer, total_iter)
            
                if best_reward < test_result["rew"]:
                    best_reward = test_result["rew"]
                    best_policy = policy
                
                print(loss['loss'])
                print(test_result)
                print(best_reward)
                print('---------------')
    
    
    return {'best_reward': best_reward, 'best_policy': best_policy}

    # env_step, gradient_step = 0, 0
    # best_epoch, best_reward, best_reward_std = -1, -1.0, 0.0
    # stat: Dict[str, MovAvg] = {}
    # start_time = time.time()
    # train_collector.reset_stat()
    # test_collector.reset_stat()
    # test_in_train = test_in_train and train_collector.policy == policy
    # for epoch in range(1, 1 + max_epoch):
    #     # train
    #     policy.train()
    #     with tqdm.tqdm(
    #         total=step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config
    #     ) as t:
    #         while t.n < t.total:
    #             if train_fn:
    #                 train_fn(epoch, env_step)
    #             result = train_collector.collect(n_step=collect_per_step)
    #             env_step += int(result["n/st"])
    #             data = {
    #                 "env_step": str(env_step),
    #                 "rew": f"{result['rew']:.2f}",
    #                 "len": str(int(result["len"])),
    #                 "n/ep": str(int(result["n/ep"])),
    #                 "n/st": str(int(result["n/st"])),
    #                 "v/ep": f"{result['v/ep']:.2f}",
    #                 "v/st": f"{result['v/st']:.2f}",
    #             }
    #             if writer and env_step % log_interval == 0:
    #                 for k in result.keys():
    #                     writer.add_scalar(
    #                         "train/" + k, result[k], global_step=env_step)
    #             if test_in_train and stop_fn and stop_fn(result["rew"]):
    #                 test_result = test_episode(
    #                     policy, test_collector, test_fn,
    #                     epoch, episode_per_test, writer, env_step)
    #                 if stop_fn(test_result["rew"]):
    #                     if save_fn:
    #                         save_fn(policy)
    #                     for k in result.keys():
    #                         data[k] = f"{result[k]:.2f}"
    #                     t.set_postfix(**data)
    #                     return gather_info(
    #                         start_time, train_collector, test_collector,
    #                         test_result["rew"], test_result["rew_std"])
    #                 else:
    #                     policy.train()
    #             for i in range(update_per_step * min(
    #                     result["n/st"] // collect_per_step, t.total - t.n)):
    #                 gradient_step += 1
    #                 losses = policy.update(batch_size, train_collector.buffer)
    #                 for k in losses.keys():
    #                     if stat.get(k) is None:
    #                         stat[k] = MovAvg()
    #                     stat[k].add(losses[k])
    #                     data[k] = f"{stat[k].get():.6f}"
    #                     if writer and gradient_step % log_interval == 0:
    #                         writer.add_scalar(
    #                             k, stat[k].get(), global_step=gradient_step)
    #                 t.update(1)
    #                 t.set_postfix(**data)
    #         if t.n <= t.total:
    #             t.update()
    #     # test
    #     result = test_episode(policy, test_collector, test_fn, epoch,
    #                           episode_per_test, writer, env_step)
    #     if best_epoch == -1 or best_reward < result["rew"]:
    #         best_reward, best_reward_std = result["rew"], result["rew_std"]
    #         best_epoch = epoch
    #         if save_fn:
    #             save_fn(policy)
    #     if verbose:
    #         print(f"Epoch #{epoch}: test_reward: {result['rew']:.6f} ± "
    #               f"{result['rew_std']:.6f}, best_reward: {best_reward:.6f} ± "
    #               f"{best_reward_std:.6f} in #{best_epoch}")
    #     if stop_fn and stop_fn(best_reward):
    #         break
    # return gather_info(start_time, train_collector, test_collector,
    #                    best_reward, best_reward_std)
