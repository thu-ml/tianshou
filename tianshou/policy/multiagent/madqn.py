import random
import torch
import numpy as np
from typing import Union, Optional

from tianshou.policy import DQNPolicy
from tianshou.data import Batch, to_numpy


class MultiAgentDQNPolicy(DQNPolicy):
    """DQN for multi-agent RL.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: greater than 1, the number of steps to look
        ahead.
    :param int target_update_freq: the target network update frequency (``0``
        if you do not use the target network).
    :param int agent_id: which agent is the policy playing

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation. Further usage can be found at :ref:`marl_example`.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optim: torch.optim.Optimizer,
                 discount_factor: float = 0.99,
                 estimation_step: int = 1,
                 target_update_freq: Optional[int] = 0,
                 agent_id: int = 0,
                 **kwargs) -> None:
        super().__init__(model=model,
                         optim=optim,
                         discount_factor=discount_factor,
                         estimation_step=estimation_step,
                         target_update_freq=target_update_freq,
                         **kwargs)
        self._agent_id = agent_id

    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                model: str = 'model',
                input: str = 'obs',
                eps: Optional[float] = None,
                **kwargs) -> Batch:
        """Compute action over the given batch data.

        :param float eps: in [0, 1], for epsilon-greedy exploration method.

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        ma_obs = getattr(batch, input)
        obs = getattr(ma_obs, "obs")
        q, h = model(obs, state=state, info=batch.info)
        if eps is None or not np.isclose(eps, 0):
            eps = self.eps
        else:
            eps = None
        actions = []
        for a_id, legal_actions, q_values in \
                zip(ma_obs.agent_id, ma_obs.legal_actions, q):
            legal_actions = list(legal_actions)
            if a_id != self._agent_id or (eps and np.random.rand() < eps):
                # the move of opponent or epsilon noisy move
                actions.append(random.choice(legal_actions))
            else:
                legal_actions = np.array(legal_actions)
                q_values = q_values[legal_actions]
                action = legal_actions[q_values.max(dim=0)[1]]
                actions.append(action)
        act = to_numpy(actions)
        return Batch(logits=q, act=act, state=h)
