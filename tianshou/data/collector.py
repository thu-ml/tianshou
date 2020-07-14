import gym
import time
import torch
import warnings
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable

from tianshou.utils import MovAvg
from tianshou.env import BaseVectorEnv
from tianshou.policy import BasePolicy
from tianshou.exploration import BaseNoise
from tianshou.data import Batch, ReplayBuffer, ListReplayBuffer, to_numpy


class Collector(object):
    """The :class:`~tianshou.data.Collector` enables the policy to interact
    with different types of environments conveniently.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer`
        class, or a list of :class:`~tianshou.data.ReplayBuffer`. If set to
        ``None``, it will automatically assign a small-size
        :class:`~tianshou.data.ReplayBuffer`.
    :param function preprocess_fn: a function called before the data has been
        added to the buffer, see issue #42 and :ref:`preprocess_fn`, defaults
        to ``None``.
    :param int stat_size: for the moving average of recording speed, defaults
        to 100.
    :param BaseNoise action_noise: add a noise to continuous action. Normally
        a policy already has a noise param for exploration in training phase,
        so this is recommended to use in test collector for some purpose.
    :param function reward_metric: to be used in multi-agent RL. The reward to
        report is of shape [agent_num], but we need to return a single scalar
        to monitor training. This function specifies what is the desired
        metric, e.g., the reward of agent 1 or the average reward over all
        agents. By default, the behavior is to select the reward of agent 1.

    The ``preprocess_fn`` is a function called before the data has been added
    to the buffer with batch format, which receives up to 7 keys as listed in
    :class:`~tianshou.data.Batch`. It will receive with only ``obs`` when the
    collector resets the environment. It returns either a dict or a
    :class:`~tianshou.data.Batch` with the modified keys and values. Examples
    are in "test/base/test_collector.py".

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

    def __init__(self,
                 policy: BasePolicy,
                 env: Union[gym.Env, BaseVectorEnv],
                 buffer: Optional[ReplayBuffer] = None,
                 preprocess_fn: Callable[[Any], Union[dict, Batch]] = None,
                 stat_size: Optional[int] = 100,
                 action_noise: Optional[BaseNoise] = None,
                 reward_metric: Optional[Callable[[np.ndarray], float]] = None,
                 ) -> None:
        super().__init__()
        self.env = env
        self.env_num = 1
        self.collect_time, self.collect_step, self.collect_episode = 0., 0, 0
        self.buffer = buffer
        self.policy = policy
        self.preprocess_fn = preprocess_fn
        self.process_fn = policy.process_fn
        self._multi_env = isinstance(env, BaseVectorEnv)
        # need multiple cache buffers only if storing in one buffer
        self._cached_buf = []
        if self._multi_env:
            self.env_num = len(env)
            self._cached_buf = [ListReplayBuffer()
                                for _ in range(self.env_num)]
        self.stat_size = stat_size
        self._action_noise = action_noise

        self._rew_metric = reward_metric or Collector._default_rew_metric
        self.reset()

    @staticmethod
    def _default_rew_metric(x):
        # this internal function is designed for single-agent RL
        # for multi-agent RL, a reward_metric must be provided
        assert np.asanyarray(x).size == 1, \
            'Please specify the reward_metric ' \
            'since the reward is not a scalar.'
        return x

    def reset(self) -> None:
        """Reset all related variables in the collector."""
        self.data = Batch(state={}, obs={}, act={}, rew={}, done={}, info={},
                          obs_next={}, policy={})
        self.reset_env()
        self.reset_buffer()
        self.step_speed = MovAvg(self.stat_size)
        self.episode_speed = MovAvg(self.stat_size)
        self.collect_time, self.collect_step, self.collect_episode = 0., 0, 0
        if self._action_noise is not None:
            self._action_noise.reset()

    def reset_buffer(self) -> None:
        """Reset the main data buffer."""
        if self.buffer is not None:
            self.buffer.reset()

    def get_env_num(self) -> int:
        """Return the number of environments the collector have."""
        return self.env_num

    def reset_env(self) -> None:
        """Reset all of the environment(s)' states and reset all of the cache
        buffers (if need).
        """
        obs = self.env.reset()
        if not self._multi_env:
            obs = self._make_batch(obs)
        if self.preprocess_fn:
            obs = self.preprocess_fn(obs=obs).get('obs', obs)
        self.data.obs = obs
        self.reward = 0.  # will be specified when the first data is ready
        self.length = np.zeros(self.env_num)
        for b in self._cached_buf:
            b.reset()

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> None:
        """Reset all the seed(s) of the given environment(s)."""
        return self.env.seed(seed)

    def render(self, **kwargs) -> None:
        """Render all the environment(s)."""
        return self.env.render(**kwargs)

    def close(self) -> None:
        """Close the environment(s)."""
        self.env.close()

    def _make_batch(self, data: Any) -> np.ndarray:
        """Return [data]."""
        if isinstance(data, np.ndarray):
            return data[None]
        else:
            return np.array([data])

    def _reset_state(self, id: Union[int, List[int]]) -> None:
        """Reset self.data.state[id]."""
        state = self.data.state  # it is a reference
        if isinstance(state, torch.Tensor):
            state[id].zero_()
        elif isinstance(state, np.ndarray):
            state[id] = None if state.dtype == np.object else 0
        elif isinstance(state, Batch):
            state.empty_(id)

    def collect(self,
                n_step: int = 0,
                n_episode: Union[int, List[int]] = 0,
                random: bool = False,
                render: Optional[float] = None,
                log_fn: Optional[Callable[[dict], None]] = None
                ) -> Dict[str, float]:
        """Collect a specified number of step or episode.

        :param int n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect (in each
            environment).
        :type n_episode: int or list
        :param bool random: whether to use random policy for collecting data,
            defaults to ``False``.
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
        if not self._multi_env:
            n_episode = np.sum(n_episode)
        start_time = time.time()
        assert sum([(n_step != 0), (n_episode != 0)]) == 1, \
            "One and only one collection number specification is permitted!"
        cur_step, cur_episode = 0, np.zeros(self.env_num)
        reward_sum, length_sum = 0., 0
        while True:
            if cur_step >= 100000 and cur_episode.sum() == 0:
                warnings.warn(
                    'There are already many steps in an episode. '
                    'You should add a time limitation to your environment!',
                    Warning)

            # restore the state and the input data
            last_state = self.data.state
            if last_state.is_empty():
                last_state = None
            self.data.update(state=Batch(), obs_next=Batch(), policy=Batch())

            # calculate the next action
            if random:
                action_space = self.env.action_space
                if isinstance(action_space, list):
                    result = Batch(act=[a.sample() for a in action_space])
                else:
                    result = Batch(act=self._make_batch(action_space.sample()))
            else:
                with torch.no_grad():
                    result = self.policy(self.data, last_state)

            # convert None to Batch(), since None is reserved for 0-init
            state = result.get('state', Batch())
            if state is None:
                state = Batch()
            self.data.state = state
            if hasattr(result, 'policy'):
                self.data.policy = to_numpy(result.policy)
            # save hidden state to policy._state, in order to save into buffer
            self.data.policy._state = self.data.state

            self.data.act = to_numpy(result.act)
            if self._action_noise is not None:
                self.data.act += self._action_noise(self.data.act.shape)

            # step in env
            obs_next, rew, done, info = self.env.step(
                self.data.act if self._multi_env else self.data.act[0])

            # move data to self.data
            if not self._multi_env:
                obs_next = self._make_batch(obs_next)
                rew = self._make_batch(rew)
                done = self._make_batch(done)
                info = self._make_batch(info)
            self.data.obs_next = obs_next
            self.data.rew = rew
            self.data.done = done
            self.data.info = info

            if log_fn:
                log_fn(info if self._multi_env else info[0])
            if render:
                self.render()
                if render > 0:
                    time.sleep(render)

            # add data into the buffer
            self.length += 1
            self.reward += self.data.rew
            if self.preprocess_fn:
                result = self.preprocess_fn(**self.data)
                self.data.update(result)
            if self._multi_env:  # cache_buffer branch
                for i in range(self.env_num):
                    self._cached_buf[i].add(**self.data[i])
                    if self.data.done[i]:
                        if n_step != 0 or np.isscalar(n_episode) or \
                                cur_episode[i] < n_episode[i]:
                            cur_episode[i] += 1
                            reward_sum += self.reward[i]
                            length_sum += self.length[i]
                            if self._cached_buf:
                                cur_step += len(self._cached_buf[i])
                                if self.buffer is not None:
                                    self.buffer.update(self._cached_buf[i])
                        self.reward[i], self.length[i] = 0., 0
                        if self._cached_buf:
                            self._cached_buf[i].reset()
                        self._reset_state(i)
                obs_next = self.data.obs_next
                if sum(self.data.done):
                    env_ind = np.where(self.data.done)[0]
                    obs_reset = self.env.reset(env_ind)
                    if self.preprocess_fn:
                        obs_next[env_ind] = self.preprocess_fn(
                            obs=obs_reset).get('obs', obs_reset)
                    else:
                        obs_next[env_ind] = obs_reset
                self.data.obs_next = obs_next
                if n_episode != 0:
                    if isinstance(n_episode, list) and \
                            (cur_episode >= np.array(n_episode)).all() or \
                            np.isscalar(n_episode) and \
                            cur_episode.sum() >= n_episode:
                        break
            else:  # single buffer, without cache_buffer
                if self.buffer is not None:
                    self.buffer.add(**self.data[0])
                cur_step += 1
                if self.data.done[0]:
                    cur_episode += 1
                    reward_sum += self.reward[0]
                    length_sum += self.length[0]
                    self.reward, self.length = 0., np.zeros(self.env_num)
                    self.data.state = Batch()
                    obs_next = self._make_batch(self.env.reset())
                    if self.preprocess_fn:
                        obs_next = self.preprocess_fn(obs=obs_next).get(
                            'obs', obs_next)
                    self.data.obs_next = obs_next
                if n_episode != 0 and cur_episode >= n_episode:
                    break
            if n_step != 0 and cur_step >= n_step:
                break
            self.data.obs = self.data.obs_next
        self.data.obs = self.data.obs_next

        # generate the statistics
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
        reward_sum /= n_episode
        if np.asanyarray(reward_sum).size > 1:  # non-scalar reward_sum
            reward_sum = self._rew_metric(reward_sum)
        return {
            'n/ep': cur_episode,
            'n/st': cur_step,
            'v/st': self.step_speed.get(),
            'v/ep': self.episode_speed.get(),
            'rew': reward_sum,
            'len': length_sum / n_episode,
        }

    def sample(self, batch_size: int) -> Batch:
        """Sample a data batch from the internal replay buffer. It will call
        :meth:`~tianshou.policy.BasePolicy.process_fn` before returning
        the final batch data.

        :param int batch_size: ``0`` means it will extract all the data from
            the buffer, otherwise it will extract the data with the given
            batch_size.
        """
        batch_data, indice = self.buffer.sample(batch_size)
        batch_data = self.process_fn(batch_data, self.buffer, indice)
        return batch_data
