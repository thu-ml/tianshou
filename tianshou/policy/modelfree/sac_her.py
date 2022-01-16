from typing import Any, Callable, Optional, Tuple, Union

import gym.spaces as space
import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.exploration import BaseNoise
from tianshou.policy import BasePolicy, SACPolicy


class SACHERPolicy(SACPolicy):
    """Implementation of Hindsight Experience Replay Based on SAC. arXiv:1707.01495.

    The key difference is that we redesigned the process_fn to get relabel return,
    if the replay strategy is `offline`, then it will behave the same as `SACPolicy`.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
        regularization coefficient. Default to 0.2.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
        alpha is automatically tuned.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param BaseNoise exploration_noise: add a noise to action for exploration.
        Default to None. This is useful when solving hard-exploration problem.
    :param bool deterministic_eval: whether to use deterministic action (mean
        of Gaussian policy) instead of stochastic action sampled by the policy.
        Default to True.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.

    .. seealso::

        Please refer to :class:`~tianshou.policy.SACPolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        reward_fn: Callable[[np.ndarray, np.ndarray, Optional[dict]], np.ndarray],
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        exploration_noise: Optional[BaseNoise] = None,
        deterministic_eval: bool = True,
        dict_observation_space: space.Dict = None,
        future_k: float = 4,
        strategy: str = 'offline',
        **kwargs: Any,
    ) -> None:
        super().__init__(
            actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim, tau,
            gamma, alpha, reward_normalization, estimation_step, exploration_noise,
            deterministic_eval, **kwargs
        )
        self.future_k = future_k
        self.strategy = strategy
        self.future_p = 1 - (1. / (1 + future_k))
        self.reward_fn = reward_fn
        # get index information of observation
        self.dict_observation_space = dict_observation_space
        current_idx = 0
        self.index_range = {}
        for (key, s) in dict_observation_space.spaces.items():
            self.index_range[key] = np.arange(current_idx, current_idx + s.shape[0])
            current_idx += s.shape[0]

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        # Step1: get all index needed
        if self.strategy == 'offline':
            return super(SACHERPolicy, self).process_fn(batch, buffer, indices)
        assert not self._rew_norm, \
            "Reward normalization in computing n-step returns is unsupported now."
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()
                 ] = True  # consider unfinished case: remove it
        bsz = len(indices)  # get indice of sampled transitions
        indices = [indices]  # turn to list, prepare for expand next state e.g. [1,3]
        for _ in range(self._n_step - 1):
            indices.append(
                buffer.next(indices[-1])
            )  # append next state index e.g. [[1,3][2,4]]
        indices = np.stack(indices)
        terminal = indices[-1]  # next state

        # Step2: sample new goal
        batch = buffer[terminal]  # batch.obs: s_{t+n}
        new_goal = batch.obs_next[:, self.index_range['desired_goal']]
        for i in range(bsz):
            if np.random.random() < self.future_p:
                goals = batch.info.achieved_goal[i]
                if len(goals) != 0:
                    new_goal[i] = goals[int(np.random.random() * len(goals))]

        # Step3: relabel batch's obs, obs_next, reward, calculate Q
        batch.obs[:, self.index_range['desired_goal']] = new_goal
        batch.obs_next[:, self.index_range['desired_goal']] = new_goal
        batch.rew = self.reward_fn(
            batch.obs_next[:, self.index_range['achieved_goal']], new_goal, None
        )
        with torch.no_grad():
            obs_next_result = self(batch, input='obs_next')
            a_ = obs_next_result.act
            target_q_torch = torch.min(
                self.critic1_old(batch.obs_next, a_),
                self.critic2_old(batch.obs_next, a_),
            ) - self._alpha * obs_next_result.log_prob
        target_q = to_numpy(target_q_torch.reshape(bsz, -1))
        target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)

        # Step4: calculate N step return
        gamma_buffer = np.ones(self._n_step + 1)
        for i in range(1, self._n_step + 1):
            gamma_buffer[i] = gamma_buffer[i - 1] * self._gamma
        target_shape = target_q.shape
        bsz = target_shape[0]
        # change target_q to 2d array
        target_q = target_q.reshape(bsz, -1)
        returns = np.zeros(target_q.shape)  # n_step returrn
        gammas = np.full(indices[0].shape, self._n_step)
        for n in range(self._n_step - 1, -1, -1):
            now = indices[n]
            gammas[end_flag[now] > 0] = n + 1
            returns[end_flag[now] > 0] = 0.0
            new_rew = []
            old_obs_next = buffer.obs_next[now]
            new_rew.append(
                self.reward_fn(
                    old_obs_next[:, self.index_range['achieved_goal']], new_goal, None
                )
            )
            returns = np.array(new_rew).reshape(bsz, 1) + self._gamma * returns
        target_q = target_q * gamma_buffer[gammas].reshape(bsz, 1) + returns
        target_q = target_q.reshape(target_shape)
        # return values
        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch
