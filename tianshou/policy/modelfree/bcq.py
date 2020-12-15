import torch
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Union, Optional

from tianshou.data import Batch, to_torch
from tianshou.policy import BasePolicy
import torch.nn as nn
from copy import deepcopy

from inspect import currentframe, getframeinfo

frameinfo = getframeinfo(currentframe())

# def p(x):
#     print(frameinfo.filename, frameinfo.lineno, x)


class BCQN(nn.Module):
    """BCQ NN for dialogue policy. It includes a net for imitation and a net for Q-value"""

    def __init__(
        self, input_size, n_actions, imitation_model_hidden_dim, policy_model_hidden_dim
    ):
        super(BCQN, self).__init__()
        self.q1 = nn.Linear(input_size, policy_model_hidden_dim)
        self.q2 = nn.Linear(policy_model_hidden_dim, policy_model_hidden_dim)
        self.q3 = nn.Linear(policy_model_hidden_dim, n_actions)

        self.i1 = nn.Linear(input_size, imitation_model_hidden_dim)
        self.i2 = nn.Linear(imitation_model_hidden_dim, imitation_model_hidden_dim)
        self.i3 = nn.Linear(imitation_model_hidden_dim, n_actions)

    def forward(self, state):
        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))

        i = F.relu(self.i1(state))
        i = F.relu(self.i2(i))
        i = F.relu(self.i3(i))
        return self.q3(q), F.log_softmax(i, dim=1), i


class BCQPolicy(BasePolicy):
    """Implementation discrete BCQ algorithm. Some code is from
    https://github.com/sfujim/BCQ/tree/master/discrete_BCQ

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> a)
    :param torch.optim.Optimizer optim: for optimizing the model.
    :param str mode: indicate the imitation type ("continuous" or "discrete"
        action space), defaults to "continuous".

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        tao: float,
        target_update_frequency: int,
        device: str,
        gamma: float,
        imitation_logits_penalty: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._policy_net = model
        self._optimizer = optim
        self._cnt = 0
        self._device = device
        # TODOzjl shanchu moren
        self._gamma = gamma
        self._tao = tao
        self._target_net = deepcopy(self._policy_net)
        self._target_net.eval()
        self._target_update_frequency = target_update_frequency
        self._imitation_logits_penalty = imitation_logits_penalty

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        batch.to_torch()
        state = batch.obs
        q, imt, _ = self._policy_net(state.float())
        imt = imt.exp()
        imt = (imt / imt.max(1, keepdim=True)[0] > self._tao).float()
        # Use large negative number to mask actions from argmax
        action = (imt * q + (1.0 - imt) * -1e8).argmax(1)
        return Batch(act=action, state=state, qvalue=q)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        batch.to_torch()
        if self._cnt % self._target_update_frequency == 0:
            self._target_net.load_state_dict(self._policy_net.state_dict())
            self._target_net.eval()

        non_final_mask = torch.tensor(
            tuple(map(lambda s: not s, batch.done)),
            device=self._device,
            dtype=torch.bool,
        )

        try:
            non_final_next_states = torch.cat(
                [s.obs.unsqueeze(0) for s in batch if not s.done], dim=0
            )
        except Exception:
            non_final_next_states = None

        # Compute the target Q value
        with torch.no_grad():
            expected_state_action_values = batch.rew.float()

            # Add target Q value for non-final next_state
            if non_final_next_states is not None:
                q, imt, _ = self._policy_net(non_final_next_states)
                imt = imt.exp()
                imt = (imt / imt.max(1, keepdim=True)[0] > self._tao).float()

                # Use large negative number to mask actions from argmax
                next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

                q, _, _ = self._target_net(non_final_next_states)
                q = q.gather(1, next_action).reshape(-1, 1)

                next_state_values = torch.zeros(len(batch), device=self._device).float()

                next_state_values[non_final_mask] = q.squeeze()
                expected_state_action_values += next_state_values * self._gamma

        # Get current Q estimate
        current_Q, imt, i = self._policy_net(batch.obs)

        current_Q = current_Q.gather(1, batch.act.unsqueeze(1)).squeeze()

        # Compute Q loss
        q_loss = F.smooth_l1_loss(current_Q, expected_state_action_values)
        i_loss = F.nll_loss(imt, batch.act.reshape(-1))

        Q_loss = q_loss + i_loss + self._imitation_logits_penalty * i.pow(2).mean()

        self._optimizer.zero_grad()
        Q_loss.backward()
        self._optimizer.step()

        return {"loss": Q_loss.item()}

        # if self._target and self._cnt % self._freq == 0:
        #     self.sync_weight()
        # self.optim.zero_grad()
        # weight = batch.pop("weight", 1.0)
        # q = self(batch).logits
        # q = q[np.arange(len(q)), batch.act]
        # r = to_torch_as(batch.returns.flatten(), q)
        # td = r - q
        # loss = (td.pow(2) * weight).mean()
        # batch.weight = td  # prio-buffer
        # loss.backward()
        # self.optim.step()
        # self._cnt += 1
        # return {"loss": loss.item()}

        # if self.mode == "continuous":  # regression
        #     a = self(batch).act
        #     a_ = to_torch(batch.act, dtype=torch.float32, device=a.device)
        #     loss = F.mse_loss(a, a_)  # type: ignore
        # elif self.mode == "discrete":  # classification
        #     a = self(batch).logits
        #     a_ = to_torch(batch.act, dtype=torch.long, device=a.device)
        #     loss = F.nll_loss(a, a_)  # type: ignore
        # loss.backward()
        # self.optim.step()
        # return {"loss": loss.item()}
