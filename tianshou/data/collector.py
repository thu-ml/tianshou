import gym
import time
import torch
import warnings
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable

from tianshou.env import BaseVectorEnv, VectorEnv
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
        class. If set to ``None`` (testing phase), it will not store the data.
    :param function preprocess_fn: a function called before the data has been
        added to the buffer, see issue #42 and :ref:`preprocess_fn`, defaults
        to ``None``.
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
        collector = Collector(policy, envs, buffer=replay_buffer)

        # collect 3 episodes
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

    Collected data always consist of full episodes. So if only ``n_step``
    argument is give, the collector may return the data more than the
    ``n_step`` limitation.

    .. note::

        Please make sure the given environment has a time limitation.
    """

    def __init__(self,
                 policy: BasePolicy,
                 env: Union[gym.Env, BaseVectorEnv],
                 buffer: Optional[ReplayBuffer] = None,
                 preprocess_fn: Callable[[Any], Union[dict, Batch]] = None,
                 action_noise: Optional[BaseNoise] = None,
                 reward_metric: Optional[Callable[[np.ndarray], float]] = None,
                 ) -> None:
        super().__init__()
        if not isinstance(env, BaseVectorEnv):
            env = VectorEnv([lambda: env])
        self.env = env
        self.env_num = len(env)
        # need cache buffers before storing in the main buffer
        self._cached_buf = [ListReplayBuffer() for _ in range(self.env_num)]
        self.collect_time, self.collect_step, self.collect_episode = 0., 0, 0
        self.buffer = buffer
        self.policy = policy
        self.preprocess_fn = preprocess_fn
        self.process_fn = policy.process_fn
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
        if self.preprocess_fn:
            obs = self.preprocess_fn(obs=obs).get('obs', obs)
        self.data.obs = obs
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
                n_step: Optional[int] = None,
                n_episode: Optional[Union[int, List[int]]] = None,
                random: bool = False,
                render: Optional[float] = None,
                log_fn: Optional[Callable[[dict], None]] = None
                ) -> Dict[str, float]:
        """Collect a specified number of step or episode.

        :param int n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect. If it is
            an int, it means to collect totally ``n_episode`` episodes; if
            it is a list, it means to collect ``n_episode[i]`` episodes in
            the i-th environment
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
        assert (n_step and not n_episode) or (not n_step and n_episode), \
            "One and only one collection number specification is permitted!"
        start_time = time.time()
        step_count = 0
        # episode of each environment
        episode_count = np.zeros(self.env_num)
        reward_total = 0.0
        while True:
            if step_count >= 100000 and episode_count.sum() == 0:
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
                result = Batch(
                    act=[a.sample() for a in self.env.action_space])
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
            obs_next, rew, done, info = self.env.step(self.data.act)

            # move data to self.data
            self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)

            if log_fn:
                log_fn(self.data.info)
            if render:
                self.render()
                if render > 0:
                    time.sleep(render)

            # add data into the buffer
            if self.preprocess_fn:
                result = self.preprocess_fn(**self.data)
                self.data.update(result)
            for i in range(self.env_num):
                self._cached_buf[i].add(**self.data[i])
                if self.data.done[i]:
                    if n_step or np.isscalar(n_episode) or \
                            episode_count[i] < n_episode[i]:
                        episode_count[i] += 1
                        reward_total += np.asarray(
                            self._cached_buf[i].rew).sum(axis=0)
                        step_count += len(self._cached_buf[i])
                        if self.buffer is not None:
                            self.buffer.update(self._cached_buf[i])
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
            if n_episode:
                if isinstance(n_episode, list) and \
                        (episode_count >= np.array(n_episode)).all() or \
                        np.isscalar(n_episode) and \
                        episode_count.sum() >= n_episode:
                    break
            if n_step and step_count >= n_step:
                break
            self.data.obs = self.data.obs_next
        self.data.obs = self.data.obs_next

        # generate the statistics
        episode_count = sum(episode_count)
        duration = max(time.time() - start_time, 1e-9)
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += duration
        # average reward across the number of episodes
        reward_avg = reward_total / episode_count
        if np.asanyarray(reward_avg).size > 1:  # non-scalar reward_avg
            reward_avg = self._rew_metric(reward_avg)
        return {
            'n/ep': episode_count,
            'n/st': step_count,
            'v/st': step_count / duration,
            'v/ep': episode_count / duration,
            'rew': reward_avg,
            'len': step_count / episode_count,
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
