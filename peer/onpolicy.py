import time
import tqdm

from tianshou.utils import tqdm_config, MovAvg
from tianshou.trainer import test_episode, gather_info


def onpolicy_trainer(A, B, max_epoch, step_per_epoch, collect_per_step,
                     repeat_per_collect, episode_per_test, batch_size,
                     copier=False, peer=0, verbose=True, test_fn=None,
                     task='', copier_batch_size=0):
    global_step = 0
    best_epoch, best_reward = -1, -1
    stat = {}
    start_time = time.time()
    for epoch in range(1, 1 + max_epoch):
        # train
        A.train()
        B.train()
        with tqdm.tqdm(
                total=step_per_epoch, desc=f'Epoch #{epoch}',
                **tqdm_config) as t:
            while t.n < t.total:
                for view, other_view in zip([A, B], [B, A]):
                    result = view.train_collector.collect(n_episode=collect_per_step)
                    data = {}
                    if view.stop_fn and view.stop_fn(result['rew']):
                        test_result = test_episode(
                            view.policy, view.test_collector, test_fn,
                            epoch, episode_per_test)
                        if view.stop_fn and view.stop_fn(test_result['rew']):
                            for k in result.keys():
                                data[k] = f'{result[k]:.2f}'
                            t.set_postfix(**data)
                            return gather_info(
                                start_time, view.train_collector, view.test_collector,
                                test_result['rew'])
                        else:
                            view.train()

                    batch = view.train_collector.sample(0)
                    for obs in batch.obs:
                        view.buffer.add(obs, {}, {}, {}, {}, {})  # only need obs
                    losses = view.policy.learn(batch, batch_size, repeat_per_collect)

                    # Learn from demonstration
                    if copier:
                        copier_batch, indice = view.buffer.sample(copier_batch_size)
                        # copier_batch = view.train_collector.process_fn(
                        #     copier_batch, view.buffer, indice)
                        demo = other_view.policy(copier_batch)
                        view.learn_from_demos(copier_batch, demo, peer=peer)

                    view.train_collector.reset_buffer()
                    step = 1
                    for k in losses.keys():
                        if isinstance(losses[k], list):
                            step = max(step, len(losses[k]))
                    global_step += step
                    for k in result.keys():
                        data[k] = f'{result[k]:.2f}'
                        view.writer.add_scalar(
                            k + '_' + task if task else k,
                            result[k], global_step=global_step)
                    for k in losses.keys():
                        if stat.get(k) is None:
                            stat[k] = MovAvg()
                        stat[k].add(losses[k])
                        data[k] = f'{stat[k].get():.4f}'
                        if global_step:
                            view.writer.add_scalar(
                                k + '_' + task if task else k,
                                stat[k].get(), global_step=global_step)
                    t.update(step)
                    t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        # test
        brk = False
        for view in A, B:
            result = test_episode(
                view.policy, view.test_collector, test_fn, epoch, episode_per_test)
            if best_epoch == -1 or best_reward < result['rew']:
                best_reward = result['rew']
                best_epoch = epoch
            if verbose:
                print(f'Epoch #{epoch}: test_reward: {result["rew"]:.4f}, '
                      f'best_reward: {best_reward:.4f} in #{best_epoch}')
            if view.stop_fn(best_reward):
                brk = True
        if brk:
            break
    return (
        gather_info(start_time, A.train_collector, A.test_collector, best_reward),
        gather_info(start_time, B.train_collector, B.test_collector, best_reward),
    )
