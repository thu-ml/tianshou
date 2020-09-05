import gym
import time
import torch
import warnings
import numpy as np
from copy import deepcopy
from typing import Any, Dict, List, Union, Optional, Callable

from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.policy import BasePolicy
from tianshou.exploration import BaseNoise
from tianshou.data import Batch, ReplayBuffer, ListReplayBuffer, to_numpy
from tianshou.data.batch import _create_value


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
        envs = DummyVectorEnv([lambda: gym.make('CartPole-v0')
                               for _ in range(3)])
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

    Collected data always consist of full episodes. So if only ``n_step``
    argument is give, the collector may return the data more than the
    ``n_step`` limitation. Same as ``n_episode`` for the multiple environment
    case.

    .. note::

        Please make sure the given environment has a time limitation.
    """

    def __init__(self,
                 policy: BasePolicy,
                 env: Union[gym.Env, BaseVectorEnv],
                 buffer: Optional[ReplayBuffer] = None,
                 preprocess_fn: Callable[[Any], Batch] = None,
                 action_noise: Optional[BaseNoise] = None,
                 reward_metric: Optional[Callable[[np.ndarray], float]] = None,
                 ) -> None:
        super().__init__()
        if not isinstance(env, BaseVectorEnv):
            env = DummyVectorEnv([lambda: env])
        self.env = env
        self.env_num = len(env)
        # environments that are available in step()
        # this means all environments in synchronous simulation
        # but only a subset of environments in asynchronous simulation
        self._ready_env_ids = np.arange(self.env_num)
        # self.async is a flag to indicate whether this collector works
        # with asynchronous simulation
        self.is_async = env.is_async
        # need cache buffers before storing in the main buffer
        self._cached_buf = [ListReplayBuffer() for _ in range(self.env_num)]
        self.buffer = buffer
        self.policy = policy
        self.preprocess_fn = preprocess_fn
        self.process_fn = policy.process_fn
        self._action_space = env.action_space
        self._action_noise = action_noise
        self._rew_metric = reward_metric or Collector._default_rew_metric
        # avoid creating attribute outside __init__
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
        # use empty Batch for ``state`` so that ``self.data`` supports slicing
        # convert empty Batch to None when passing data to policy
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
        self._ready_env_ids = np.arange(self.env_num)
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

    def _reset_state(self, id: Union[int, List[int]]) -> None:
        """Reset the hidden state: self.data.state[id]."""
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
                no_grad: bool = True,
                ) -> Dict[str, float]:
        """Collect a specified number of step or episode.

        :param int n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect. If it is an
            int, it means to collect at lease ``n_episode`` episodes; if it is
            a list, it means to collect exactly ``n_episode[i]`` episodes in
            the i-th environment
        :param bool random: whether to use random policy for collecting data,
            defaults to ``False``.
        :param float render: the sleep time between rendering consecutive
            frames, defaults to ``None`` (no rendering).
        :param bool no_grad: whether to retain gradient in policy.forward,
            defaults to ``True`` (no gradient retaining).

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
        assert (n_step is not None and n_episode is None and n_step > 0) or (
            n_step is None and n_episode is not None and np.sum(n_episode) > 0
        ), "Only one of n_step or n_episode is allowed in Collector.collect, "
        f"got n_step = {n_step}, n_episode = {n_episode}."
        start_time = time.time()
        step_count = 0
        # episode of each environment
        episode_count = np.zeros(self.env_num)
        # If n_episode is a list, and some envs have collected the required
        # number of episodes, these envs will be recorded in this list, and
        # they will not be stepped.
        finished_env_ids = []
        reward_total = 0.0
        whole_data = Batch()
        list_n_episode = False
        if n_episode is not None and not np.isscalar(n_episode):
            assert len(n_episode) == self.get_env_num()
            list_n_episode = True
            finished_env_ids = [
                i for i in self._ready_env_ids if n_episode[i] <= 0]
            self._ready_env_ids = np.array(
                [x for x in self._ready_env_ids if x not in finished_env_ids])
        while True:
            if step_count >= 100000 and episode_count.sum() == 0:
                warnings.warn(
                    'There are already many steps in an episode. '
                    'You should add a time limitation to your environment!',
                    Warning)

            is_async = self.is_async or len(finished_env_ids) > 0
            if is_async:
                # self.data are the data for all environments in async
                # simulation or some envs have finished,
                # **only a subset of data are disposed**,
                # so we store the whole data in ``whole_data``, let self.data
                # to be the data available in ready environments, and finally
                # set these back into all the data
                whole_data = self.data
                self.data = self.data[self._ready_env_ids]

            # restore the state and the input data
            last_state = self.data.state
            if isinstance(last_state, Batch) and last_state.is_empty():
                last_state = None
            self.data.update(state=Batch(), obs_next=Batch(), policy=Batch())

            # calculate the next action
            if random:
                spaces = self._action_space
                result = Batch(
                    act=[spaces[i].sample() for i in self._ready_env_ids])
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)

            state = result.get('state', Batch())
            # convert None to Batch(), since None is reserved for 0-init
            if state is None:
                state = Batch()
            self.data.update(state=state, policy=result.get('policy', Batch()))
            # save hidden state to policy._state, in order to save into buffer
            if not (isinstance(state, Batch) and state.is_empty()):
                self.data.policy._state = self.data.state

            self.data.act = to_numpy(result.act)
            if self._action_noise is not None:  # noqa
                self.data.act += self._action_noise(self.data.act.shape)

            # step in env
            if not is_async:
                obs_next, rew, done, info = self.env.step(self.data.act)
            else:
                # store computed actions, states, etc
                _batch_set_item(whole_data, self._ready_env_ids,
                                self.data, self.env_num)
                # fetch finished data
                obs_next, rew, done, info = self.env.step(
                    self.data.act, id=self._ready_env_ids)
                self._ready_env_ids = np.array([i['env_id'] for i in info])
                # get the stepped data
                self.data = whole_data[self._ready_env_ids]
            # move data to self.data
            self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)

            if render:
                self.render()
                time.sleep(render)

            # add data into the buffer
            if self.preprocess_fn:
                result = self.preprocess_fn(**self.data)
                self.data.update(result)

            for j, i in enumerate(self._ready_env_ids):
                # j is the index in current ready_env_ids
                # i is the index in all environments
                if self.buffer is None:
                    # users do not want to store data, so we store
                    # small fake data here to make the code clean
                    self._cached_buf[i].add(obs=0, act=0, rew=rew[j], done=0)
                else:
                    self._cached_buf[i].add(**self.data[j])

                if done[j]:
                    if not (list_n_episode and
                            episode_count[i] >= n_episode[i]):
                        episode_count[i] += 1
                        reward_total += np.sum(self._cached_buf[i].rew, axis=0)
                        step_count += len(self._cached_buf[i])
                        if self.buffer is not None:
                            self.buffer.update(self._cached_buf[i])
                        if list_n_episode and \
                                episode_count[i] >= n_episode[i]:
                            # env i has collected enough data, it has finished
                            finished_env_ids.append(i)
                    self._cached_buf[i].reset()
                    self._reset_state(j)
            obs_next = self.data.obs_next
            if sum(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = self._ready_env_ids[env_ind_local]
                obs_reset = self.env.reset(env_ind_global)
                if self.preprocess_fn:
                    obs_next[env_ind_local] = self.preprocess_fn(
                        obs=obs_reset).get('obs', obs_reset)
                else:
                    obs_next[env_ind_local] = obs_reset
            self.data.obs = obs_next
            if is_async:
                # set data back
                whole_data = deepcopy(whole_data)  # avoid reference in ListBuf
                _batch_set_item(whole_data, self._ready_env_ids,
                                self.data, self.env_num)
                # let self.data be the data in all environments again
                self.data = whole_data
            self._ready_env_ids = np.array(
                [x for x in self._ready_env_ids if x not in finished_env_ids])
            if n_step:
                if step_count >= n_step:
                    break
            else:
                if isinstance(n_episode, int) and \
                        episode_count.sum() >= n_episode:
                    break
                if isinstance(n_episode, list) and \
                        (episode_count >= n_episode).all():
                    break

        # finished envs are ready, and can be used for the next collection
        self._ready_env_ids = np.array(
            self._ready_env_ids.tolist() + finished_env_ids)

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
        :meth:`~tianshou.policy.BasePolicy.process_fn` before returning the
        final batch data.

        :param int batch_size: ``0`` means it will extract all the data from
            the buffer, otherwise it will extract the data with the given
            batch_size.
        """
        warnings.warn(
            'Collector.sample is deprecated and will cause error if you use '
            'prioritized experience replay! Collector.sample will be removed '
            'upon version 0.3. Use policy.update instead!', Warning)
        assert self.buffer is not None, "Cannot get sample from empty buffer!"
        batch_data, indice = self.buffer.sample(batch_size)
        batch_data = self.process_fn(batch_data, self.buffer, indice)
        return batch_data

    def close(self) -> None:
        warnings.warn(
            'Collector.close is deprecated and will be removed upon version '
            '0.3.', Warning)


def _batch_set_item(source: Batch, indices: np.ndarray,
                    target: Batch, size: int):
    # for any key chain k, there are four cases
    # 1. source[k] is non-reserved, but target[k] does not exist or is reserved
    # 2. source[k] does not exist or is reserved, but target[k] is non-reserved
    # 3. both source[k] and target[k] are non-reserved
    # 4. both source[k] and target[k] do not exist or are reserved, do nothing.
    # A special case in case 4, if target[k] is reserved but source[k] does
    # not exist, make source[k] reserved, too.
    for k, vt in target.items():
        if not isinstance(vt, Batch) or not vt.is_empty():
            # target[k] is non-reserved
            vs = source.get(k, Batch())
            if isinstance(vs, Batch):
                if vs.is_empty():
                    # case 2, use __dict__ to avoid many type checks
                    source.__dict__[k] = _create_value(vt[0], size)
                else:
                    assert isinstance(vt, Batch)
                    _batch_set_item(source.__dict__[k], indices, vt, size)
        else:
            # target[k] is reserved
            # case 1 or special case of case 4
            if k not in source.__dict__:
                source.__dict__[k] = Batch()
            continue
        source.__dict__[k][indices] = vt
