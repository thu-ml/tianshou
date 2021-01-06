import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Union, Optional

from tianshou.data import Batch
from tianshou.policy import BasePolicy
from copy import deepcopy


class BCQPolicy(BasePolicy):
    """Implementation discrete BCQ algorithm. Some code is from
    https://github.com/sfujim/BCQ/tree/master/discrete_BCQ

    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        tau: float,
        target_update_frequency: int,
        device: str,
        gamma: float,
        imitation_logits_penalty: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._policy_net = model
        self._policy_net.to(device)
        self._optimizer = optim
        self._cnt = 0
        self._device = device
        self._gamma = gamma
        self._tau = tau
        self._target_net = deepcopy(self._policy_net)
        self._target_net.eval()
        self._target_net.to(device)
        self._target_update_frequency = target_update_frequency
        self._imitation_logits_penalty = imitation_logits_penalty

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        batch.to_torch()

        state = batch.obs.to(self._device)
        q, imt, _ = self._policy_net(state.float())
        imt = imt.exp()
        imt = (imt / imt.max(1, keepdim=True)[0] > self._tau).float()

        # Use large negative number to mask actions from argmax
        action = (imt * q + (1.0 - imt) * -1e8).argmax(1)

        return Batch(act=action, state=state, qvalue=q)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        batch.to_torch()

        non_final_mask = torch.tensor(
            tuple(map(lambda s: not s, batch.done)),
            device=self._device,
            dtype=torch.bool,
        )

        non_final_next_states = None
        try:
            non_final_next_states = torch.cat(
                [s.obs.unsqueeze(0) for s in batch if not s.done], dim=0
            )
            non_final_next_states = non_final_next_states.to(self._device)
        except Exception:
            pass

        # Compute the target Q value
        with torch.no_grad():
            expected_state_action_values = batch.rew.float().to(self._device)

            # Add target Q value for non-final next_state
            if non_final_next_states is not None:
                q, imt, _ = self._policy_net(non_final_next_states)
                imt = imt.exp()
                imt = (imt / imt.max(1, keepdim=True)[0] > self._tau).float()

                # Use large negative number to mask actions from argmax
                next_action = (imt * q + (1 - imt) * -1e8).argmax(
                    1, keepdim=True
                )
                q, _, _ = self._target_net(non_final_next_states)
                q = q.gather(1, next_action).reshape(-1, 1)

                next_state_values = torch.zeros(
                    len(batch), device=self._device
                ).float()
                next_state_values[non_final_mask] = q.squeeze()

                expected_state_action_values += next_state_values * self._gamma

        # Get current Q estimate
        current_Q, imt, i = self._policy_net(batch.obs.to(self._device))
        current_Q = current_Q.gather(
            1, batch.act.unsqueeze(1).to(self._device)
        ).squeeze()

        # Compute Q loss
        q_loss = F.smooth_l1_loss(current_Q, expected_state_action_values)
        i_loss = F.nll_loss(imt, batch.act.reshape(-1).to(self._device))

        Q_loss = (
            q_loss + i_loss + self._imitation_logits_penalty * i.pow(2).mean()
        )

        self._optimizer.zero_grad()
        Q_loss.backward()
        self._optimizer.step()

        if self._cnt % self._target_update_frequency == 0:
            self._target_net.load_state_dict(self._policy_net.state_dict())
            self._target_net.eval()

        return {"loss": Q_loss.item()}
