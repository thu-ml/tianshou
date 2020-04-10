import time
import tqdm

from tianshou.utils import tqdm_config, MovAvg
from tianshou.trainer import test_episode, gather_info


def imitation_trainer(policy, learner, expert_collector, test_collector,
                      max_epoch, step_per_epoch, collect_per_step,
                      repeat_per_collect, episode_per_test, batch_size,
                      train_fn=None, test_fn=None, stop_fn=None,
                      writer=None, task='', peer=0, peer_decay_steps=0):
    global_step = 0
    best_epoch, best_reward = -1, -1
    stat = {}
    start_time = time.time()
    for epoch in range(1, 1 + max_epoch):
        # train
        policy.train()
        if train_fn:
            train_fn(epoch)
        with tqdm.tqdm(
                total=step_per_epoch, desc=f'Epoch #{epoch}',
                **tqdm_config) as t:
            while t.n < t.total:
                expert_collector.collect(n_episode=collect_per_step)
                result = test_collector.collect(n_episode=episode_per_test)

                data = {}
                if stop_fn and stop_fn(result['rew']):
                    for k in result.keys():
                        data[k] = f'{result[k]:.2f}'
                    t.set_postfix(**data)
                    return gather_info(
                        start_time, expert_collector, test_collector,
                        result['rew'])
                else:
                    policy.train()
                    if train_fn:
                        train_fn(epoch)

                decay = 1. if not peer_decay_steps else \
                    max(0., 1 - global_step / peer_decay_steps)
                losses = learner(policy, expert_collector.sample(0),
                                 batch_size, repeat_per_collect, peer * decay)
                expert_collector.reset_buffer()
                step = 1
                for k in losses.keys():
                    if isinstance(losses[k], list):
                        step = max(step, len(losses[k]))
                global_step += step
                for k in result.keys():
                    data[k] = f'{result[k]:.2f}'
                    if writer:
                        writer.add_scalar(
                            k + '_' + task if task else k,
                            result[k], global_step=global_step)
                for k in losses.keys():
                    if stat.get(k) is None:
                        stat[k] = MovAvg()
                    stat[k].add(losses[k])
                    data[k] = f'{stat[k].get():.6f}'
                    if writer and global_step:
                        writer.add_scalar(
                            k + '_' + task if task else k,
                            stat[k].get(), global_step=global_step)
                t.update(step)
                t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        # test
        result = test_episode(
            policy, test_collector, test_fn, epoch, episode_per_test)
        if best_epoch == -1 or best_reward < result['rew']:
            best_reward = result['rew']
            best_epoch = epoch
        print(f'Epoch #{epoch}: test_reward: {result["rew"]:.6f}, '
              f'best_reward: {best_reward:.6f} in #{best_epoch}')
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(
        start_time, expert_collector, test_collector, best_reward)
