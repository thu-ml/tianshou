import time
import torch
import warnings
import numpy as np

from tianshou.utils import MovAvg
from tianshou.env import BaseVectorEnv
from tianshou.data import Batch, ReplayBuffer, ListReplayBuffer


class Collector(object):
    """The :class:`~tianshou.data.Collector` enables the policy to interact
    with different types of environments conveniently.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param env: an environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer`
        class, or a list of :class:`~tianshou.data.ReplayBuffer`. If set to
        ``None``, it will automatically assign a small-size
        :class:`~tianshou.data.ReplayBuffer`.
    :param int stat_size: for the moving average of recording speed, defaults
        to 100.

    Example:
    ::

        policy = PGPolicy(...)  # or other policies if you wish
        env = gym.make('CartPole-v0')
        replay_buffer = ReplayBuffer(size=10000)
        # here we set up a collector with a single environment
        collector = Collector(policy, env, buffer=replay_buffer)

        # the collector supports vectorized environments as well
        envs = VectorEnv([lambda: gym.make('CartPole-v0') for _ in range(3)])
        buffers = [ReplayBuffer(size=5000) for _ in range(3)]
        # you can also pass a list of replay buffer to collector, for multi-env
        # collector = Collector(policy, envs, buffer=buffers)
        collector = Collector(policy, envs, buffer=replay_buffer)

        # collect at least 3 episodes
        collector.collect(n_episode=3)
        # collect 1 episode for the first env, 3 for the third env
        collector.collect(n_episode=[1, 0, 3])
        # collect at least 2 steps
        collector.collect(n_step=2)
        # collect episodes with visual rendering (the render argument is the
        #   sleep time between rendering consecutive frames)
        collector.collect(n_episode=1, render=0.03)

        # sample data with a given number of batch-size:
        batch_data = collector.sample(batch_size=64)
        # policy.learn(batch_data)  # btw, vanilla policy gradient only
        #   supports on-policy training, so here we pick all data in the buffer
        batch_data = collector.sample(batch_size=0)
        policy.learn(batch_data)
        # on-policy algorithms use the collected data only once, so here we
        #   clear the buffer
        collector.reset_buffer()

    For the scenario of collecting data from multiple environments to a single
    buffer, the cache buffers will turn on automatically. It may return the
    data more than the given limitation.

    .. note::

        Please make sure the given environment has a time limitation.
    """

    def __init__(self, policy, env, buffer=None, stat_size=100, **kwargs):
        super().__init__()
        self.env = env
        self.env_num = 1
        self.collect_step = 0
        self.collect_episode = 0
        self.collect_time = 0
        self.buffer = buffer
        self.policy = policy
        self.process_fn = policy.process_fn
        self._multi_env = isinstance(env, BaseVectorEnv)
        self._multi_buf = False  # True if buf is a list
        # need multiple cache buffers only if storing in one buffer
        self._cached_buf = []
        if self._multi_env:
            self.env_num = len(env)
            if isinstance(self.buffer, list):
                assert len(self.buffer) == self.env_num, \
                    'The number of data buffer does not match the number of ' \
                    'input env.'
                self._multi_buf = True
            elif isinstance(self.buffer, ReplayBuffer) or self.buffer is None:
                self._cached_buf = [
                    ListReplayBuffer() for _ in range(self.env_num)]
            else:
                raise TypeError('The buffer in data collector is invalid!')
        self.stat_size = stat_size
        self.reset()

    def reset(self):
        """Reset all related variables in the collector."""
        self.reset_env()
        self.reset_buffer()
        # state over batch is either a list, an np.ndarray, or a torch.Tensor
        self.state = None
        self.step_speed = MovAvg(self.stat_size)
        self.episode_speed = MovAvg(self.stat_size)
        self.collect_step = 0
        self.collect_episode = 0
        self.collect_time = 0

    def reset_buffer(self):
        """Reset the main data buffer."""
        if self._multi_buf:
            for b in self.buffer:
                b.reset()
        else:
            if self.buffer is not None:
                self.buffer.reset()

    def get_env_num(self):
        """Return the number of environments the collector has."""
        return self.env_num

    def reset_env(self):
        """Reset all of the environment(s)' states and reset all of the cache
        buffers (if need).
        """
        self._obs = self.env.reset()
        self._act = self._rew = self._done = self._info = None
        if self._multi_env:
            self.reward = np.zeros(self.env_num)
            self.length = np.zeros(self.env_num)
        else:
            self.reward, self.length = 0, 0
        for b in self._cached_buf:
            b.reset()

    def seed(self, seed=None):
        """Reset all the seed(s) of the given environment(s)."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)

    def render(self, **kwargs):
        """Render all the environment(s)."""
        if hasattr(self.env, 'render'):
            return self.env.render(**kwargs)

    def close(self):
        """Close the environment(s)."""
        if hasattr(self.env, 'close'):
            self.env.close()

    def _make_batch(self, data):
        """Return [data]."""
        if isinstance(data, np.ndarray):
            return data[None]
        else:
            return np.array([data])

    def _reset_state(self, id):
        """Reset self.state[id]."""
        if self.state is None:
            return
        if isinstance(self.state, list):
            self.state[id] = None
        elif isinstance(self.state, dict):
            for k in self.state:
                if isinstance(self.state[k], list):
                    self.state[k][id] = None
                elif isinstance(self.state[k], torch.Tensor) or \
                        isinstance(self.state[k], np.ndarray):
                    self.state[k][id] = 0
        elif isinstance(self.state, torch.Tensor) or \
                isinstance(self.state, np.ndarray):
            self.state[id] = 0

    def collect(self, n_step=0, n_episode=0, render=None, log_fn=None):
        """Collect a specified number of step or episode.

        :param int n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect (in each
            environment).
        :type n_episode: int or list
        :param float render: the sleep time between rendering consecutive
            frames, defaults to ``None`` (no rendering).
        :param function log_fn: a function which receives env info, typically
            for tensorboard logging.

        .. note::

            One and only one collection number specification is permitted,
            either ``n_step`` or ``n_episode``.

        :return: A dict including the following keys

            * ``n/ep`` the collected number of episodes.
            * ``n/st`` the collected number of steps.
            * ``v/st`` the speed of steps per second.
            * ``v/ep`` the speed of episode per second.
            * ``rew`` the mean reward over collected episodes.
            * ``len`` the mean length over collected episodes.
        """
        warning_count = 0
        if not self._multi_env:
            n_episode = np.sum(n_episode)
        start_time = time.time()
        assert sum([(n_step != 0), (n_episode != 0)]) == 1, \
            "One and only one collection number specification is permitted!"
        cur_step = 0
        cur_episode = np.zeros(self.env_num) if self._multi_env else 0
        reward_sum = 0
        length_sum = 0
        while True:
            if warning_count >= 100000:
                warnings.warn(
                    'There are already many steps in an episode. '
                    'You should add a time limitation to your environment!',
                    Warning)
            if self._multi_env:
                batch_data = Batch(
                    obs=self._obs, act=self._act, rew=self._rew,
                    done=self._done, obs_next=None, info=self._info)
            else:
                batch_data = Batch(
                    obs=self._make_batch(self._obs),
                    act=self._make_batch(self._act),
                    rew=self._make_batch(self._rew),
                    done=self._make_batch(self._done),
                    obs_next=None,
                    info=self._make_batch(self._info))
            with torch.no_grad():
                result = self.policy(batch_data, self.state)
            self.state = result.state if hasattr(result, 'state') else None
            if isinstance(result.act, torch.Tensor):
                self._act = result.act.detach().cpu().numpy()
            elif not isinstance(self._act, np.ndarray):
                self._act = np.array(result.act)
            else:
                self._act = result.act
            obs_next, self._rew, self._done, self._info = self.env.step(
                self._act if self._multi_env else self._act[0])
            if log_fn is not None:
                log_fn(self._info)
            if render is not None:
                self.env.render()
                if render > 0:
                    time.sleep(render)
            self.length += 1
            self.reward += self._rew
            if self._multi_env:
                for i in range(self.env_num):
                    data = {
                        'obs': self._obs[i], 'act': self._act[i],
                        'rew': self._rew[i], 'done': self._done[i],
                        'obs_next': obs_next[i], 'info': self._info[i]}
                    if self._cached_buf:
                        warning_count += 1
                        self._cached_buf[i].add(**data)
                    elif self._multi_buf:
                        warning_count += 1
                        self.buffer[i].add(**data)
                        cur_step += 1
                    else:
                        warning_count += 1
                        if self.buffer is not None:
                            self.buffer.add(**data)
                        cur_step += 1
                    if self._done[i]:
                        if n_step != 0 or np.isscalar(n_episode) or \
                                cur_episode[i] < n_episode[i]:
                            cur_episode[i] += 1
                            reward_sum += self.reward[i]
                            length_sum += self.length[i]
                            if self._cached_buf:
                                cur_step += len(self._cached_buf[i])
                                if self.buffer is not None:
                                    self.buffer.update(self._cached_buf[i])
                        self.reward[i], self.length[i] = 0, 0
                        if self._cached_buf:
                            self._cached_buf[i].reset()
                        self._reset_state(i)
                if sum(self._done):
                    obs_next = self.env.reset(np.where(self._done)[0])
                if n_episode != 0:
                    if isinstance(n_episode, list) and \
                            (cur_episode >= np.array(n_episode)).all() or \
                            np.isscalar(n_episode) and \
                            cur_episode.sum() >= n_episode:
                        break
            else:
                if self.buffer is not None:
                    self.buffer.add(
                        self._obs, self._act[0], self._rew,
                        self._done, obs_next, self._info)
                cur_step += 1
                if self._done:
                    cur_episode += 1
                    reward_sum += self.reward
                    length_sum += self.length
                    self.reward, self.length = 0, 0
                    self.state = None
                    obs_next = self.env.reset()
                if n_episode != 0 and cur_episode >= n_episode:
                    break
            if n_step != 0 and cur_step >= n_step:
                break
            self._obs = obs_next
        self._obs = obs_next
        if self._multi_env:
            cur_episode = sum(cur_episode)
        duration = max(time.time() - start_time, 1e-9)
        self.step_speed.add(cur_step / duration)
        self.episode_speed.add(cur_episode / duration)
        self.collect_step += cur_step
        self.collect_episode += cur_episode
        self.collect_time += duration
        if isinstance(n_episode, list):
            n_episode = np.sum(n_episode)
        else:
            n_episode = max(cur_episode, 1)
        return {
            'n/ep': cur_episode,
            'n/st': cur_step,
            'v/st': self.step_speed.get(),
            'v/ep': self.episode_speed.get(),
            'rew': reward_sum / n_episode,
            'len': length_sum / n_episode,
        }

    def sample(self, batch_size):
        """Sample a data batch from the internal replay buffer. It will call
        :meth:`~tianshou.policy.BasePolicy.process_fn` before returning
        the final batch data.

        :param int batch_size: ``0`` means it will extract all the data from
            the buffer, otherwise it will extract the data with the given
            batch_size.
        """
        if self._multi_buf:
            if batch_size > 0:
                lens = [len(b) for b in self.buffer]
                total = sum(lens)
                batch_index = np.random.choice(
                    total, batch_size, p=np.array(lens) / total)
            else:
                batch_index = np.array([])
            batch_data = Batch()
            for i, b in enumerate(self.buffer):
                cur_batch = (batch_index == i).sum()
                if batch_size and cur_batch or batch_size <= 0:
                    batch, indice = b.sample(cur_batch)
                    batch = self.process_fn(batch, b, indice)
                    batch_data.append(batch)
        else:
            batch_data, indice = self.buffer.sample(batch_size)
            batch_data = self.process_fn(batch_data, self.buffer, indice)
        return batch_data
