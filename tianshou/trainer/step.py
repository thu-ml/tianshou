import time
import tqdm

from tianshou.utils import tqdm_config, MovAvg


def step_trainer(policy, train_collector, test_collector, max_epoch,
                 step_per_epoch, collect_per_step, episode_per_test,
                 batch_size, train_fn=None, test_fn=None, stop_fn=None,
                 writer=None, verbose=True):
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
                result = train_collector.collect(n_step=collect_per_step)
                for i in range(min(
                        result['n/st'] // collect_per_step,
                        t.total - t.n)):
                    global_step += 1
                    losses = policy.learn(train_collector.sample(batch_size))
                    data = {}
                    for k in result.keys():
                        data[k] = f'{result[k]:.2f}'
                        if writer:
                            writer.add_scalar(
                                k, result[k], global_step=global_step)
                    for k in losses.keys():
                        if stat.get(k) is None:
                            stat[k] = MovAvg()
                        stat[k].add(losses[k])
                        data[k] = f'{stat[k].get():.6f}'
                        if writer:
                            writer.add_scalar(
                                k, stat[k].get(), global_step=global_step)
                    t.update(1)
                    t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        # eval
        test_collector.reset_env()
        test_collector.reset_buffer()
        policy.eval()
        if test_fn:
            test_fn(epoch)
        result = test_collector.collect(n_episode=episode_per_test)
        if best_epoch == -1 or best_reward < result['rew']:
            best_reward = result['rew']
            best_epoch = epoch
        if verbose:
            print(f'Epoch #{epoch}: test_reward: {result["rew"]:.6f}, '
                  f'best_reward: {best_reward:.6f} in #{best_epoch}')
        if stop_fn(best_reward):
            break
    duration = time.time() - start_time
    return train_collector.collect_step, train_collector.collect_episode,\
        test_collector.collect_step, test_collector.collect_episode,\
        best_reward, duration
