import time
import warnings
from typing import Any, Callable, Dict, Optional

import gym.spaces as space
import numpy as np
import torch

from tianshou.data import Batch, Collector, ReplayBuffer, to_numpy
from tianshou.env import BaseVectorEnv
from tianshou.policy import BasePolicy


class HERCollector(Collector):
    """Hindsight Experience Replay Collector.

    The collector will construct hindsight trajectory from achieved goals
    after one trajectory is fully collected.
    HER Collector provides two methods for relabel: `online` and `offline`.
    For details, please refer to https://arxiv.org/abs/1707.01495

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param dict_observation_space: a ``gym.spaces.Dict`` instance, which is
        used to get goal and achieved goal in the flattened observation
    :param function reward_fn: a function called to calculate reward.
        Often defined as `env.compute_reward()`
    :param str strategy: can be `online` or `offline`. `offline` strategy will add
        relabeled data directly back to the buffer, while `online` strategy will store
        the future achieved goal in  `batch.info.achieved_goal`,
        which can be used in `process_fn`to relabel data during the training process.
    :param int replay_k: proportion of data to be relabeled.
        For example, if `replay_k` is set to 4, then the collector will
        generate 4 new trajectory with relabeled data.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class.
        If set to None, it will not store the data. Default to None.
    :param function preprocess_fn: a function called before the data has been added to
        the buffer, see issue #42 and :ref:`preprocess_fn`. Default to None.
    :param bool exploration_noise: determine whether the action needs to be modified
        with corresponding policy's exploration noise. If so, "policy.
        exploration_noise(act, batch)" will be called automatically to add the
        exploration noise into action. Default to False.

    .. note::
        1. According to the result reported in the paper, only future replay
        is implemented in this collector.
        2. Make use your environment's `info` has `achieved_goal` attribution
        before use `online` replay strategy. it will be used for a Batch place holder.
        3. Observation normalization in the environment is not recommended,
        which bias the relabel.
        4. Success rate is also provided in the return to monitor the training
        progress.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: BaseVectorEnv,
        dict_observation_space: space.Dict,
        reward_fn: Callable[[np.ndarray, np.ndarray, Optional[dict]], np.ndarray],
        replay_k: int = 4,
        strategy: str = 'offline',
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
    ) -> None:
        # HER need dict observation space
        self.dict_observation_space = dict_observation_space
        self.reward_fn = reward_fn
        assert replay_k > 0, f'Replay k = {replay_k}, it must be a positive integer'
        self.replay_k = replay_k
        assert strategy == 'offline' or strategy == 'online', \
            f'Unsupported {strategy} replay strategy'
        self.strategy = strategy
        # Record the index of goal, achieved goal, and observation in obs,
        # which save the 80% of time to get goal compared to
        # use OpenAI gym's unflatten() function
        current_idx = 0
        self.obs_index_range = {}
        for (key, s) in dict_observation_space.spaces.items():
            self.obs_index_range[key] = np.arange(
                current_idx, current_idx + s.shape[0]
            )
            current_idx += s.shape[0]
        # assert type in base class
        self.data: Batch
        self.buffer: ReplayBuffer
        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
    ) -> Dict[str, Any]:
        if n_step is not None:
            assert n_episode is None, (
                f"Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if not n_step % self.env_num == 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer."
                )
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[:min(self.env_num, n_episode)]
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect()."
            )

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_success = []
        episode_lens = []
        episode_start_indices = []

        while True:
            assert len(self.data) == len(ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if random:
                self.data.update(
                    act=[self._action_space[i].sample() for i in ready_env_ids]
                )
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.act)
            # step in env
            result = self.env.step(action_remap, ready_env_ids)  # type: ignore
            obs_next, rew, done, info = result

            self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)
            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        policy=self.data.policy,
                        env_id=ready_env_ids,
                    )
                )

            if render:
                self.env.render(mode='rgb_array')
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                self.data, buffer_ids=ready_env_ids
            )

            # collect statistics
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_success.append(self.data[env_ind_local].info.is_success)
                episode_start_indices.append(ep_idx[env_ind_local])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                obs_reset = self.env.reset(env_ind_global)
                if self.preprocess_fn:
                    obs_reset = self.preprocess_fn(
                        obs=obs_reset, env_id=env_ind_global
                    ).get("obs", obs_reset)
                self.data.obs_next[env_ind_local] = obs_reset
                for i in env_ind_local:
                    self._reset_state(i)

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

                # use HER to create more trajectory
                for env_id in env_ind_global:  # enumerate env
                    # get recently collected data from buffer
                    env_buffer = self.buffer.buffers[env_id]
                    env_buffer_len = env_buffer.last_index[0] + 1
                    traj_len = ep_len[env_id]
                    obs_index_range = np.arange(
                        env_buffer_len - traj_len, env_buffer_len
                    ) % len(env_buffer)
                    original_trajectory = env_buffer[obs_index_range]
                    if self.strategy == 'offline':
                        new_trajactory_len = (
                            np.random.random(size=self.replay_k) * traj_len
                        ).astype(int) + 1
                        # relabel data and add back
                        for length in new_trajactory_len:
                            trajectory = Batch(original_trajectory[:length], copy=True)
                            new_goal = trajectory.obs_next[
                                length - 1, self.obs_index_range['achieved_goal']]
                            new_goals = np.repeat([new_goal], length, axis=0)
                            trajectory.obs[:, self.
                                           obs_index_range['desired_goal']] = new_goals
                            trajectory.obs_next[:, self.obs_index_range['desired_goal']
                                                ] = new_goals
                            trajectory.rew = self.reward_fn(
                                trajectory.obs_next[:, self.
                                                    obs_index_range['achieved_goal']],
                                new_goals, None
                            )
                            trajectory.done[-1] = True
                            for i in range(length):
                                env_buffer.add(trajectory[i])
                    elif self.strategy == 'online':
                        # record the achieved goal of future steps,
                        # to reduce the relabel time during the trainning
                        ag = original_trajectory.obs_next[:, self.obs_index_range[
                            'achieved_goal']]
                        for i, idx in enumerate(obs_index_range):
                            env_buffer.info.achieved_goal[idx] = ag[i:]

            self.data.obs = self.data.obs_next

            if (n_step and step_count >= n_step) or \
                    (n_episode and episode_count >= n_episode):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(
                obs={}, act={}, rew={}, done={}, obs_next={}, info={}, policy={}
            )
            self.reset_env()

        if episode_count > 0:
            rews, success, lens, idxs = list(
                map(
                    np.concatenate, [
                        episode_rews, episode_success, episode_lens,
                        episode_start_indices
                    ]
                )
            )
            rew_mean, rew_std = rews.mean(), rews.std()
            len_mean, len_std = lens.mean(), lens.std()
        else:
            rews, success, lens, idxs = np.array([]), np.array(
                []
            ), np.array([], int), np.array([], int)
            rew_mean = rew_std = len_mean = len_std = 0

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "success": success,
            "lens": lens,
            "idxs": idxs,
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
        }
