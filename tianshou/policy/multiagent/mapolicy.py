import random
from abc import ABC

import numpy as np
from typing import Union, Optional, Dict, List

from tianshou.policy import BasePolicy
from tianshou.data import Batch, to_numpy, ReplayBuffer


class BaseMultiAgentPolicy(BasePolicy, ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.agent_id = 0

    def set_agent_id(self, agent_id: int) -> None:
        self.agent_id = agent_id


class RandomMultiAgentPolicy(BaseMultiAgentPolicy):
    def forward(self,
                batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                **kwargs) -> Batch:
        actions = []
        for legal_actions in batch.obs.legal_actions:
            legal_actions = list(legal_actions)
            actions.append(random.choice(legal_actions))
        act = to_numpy(actions)
        return Batch(act=act)

    def learn(self, batch: Batch, **kwargs)\
            -> Dict[str, Union[float, List[float]]]:
        return {}


class MultiAgentPolicyManager(BaseMultiAgentPolicy):
    def __init__(self, policies: List[BaseMultiAgentPolicy]):
        super().__init__()
        self.policies = policies
        for i, policy in enumerate(policies):
            # agent_id 0 is reserved for the environment proxy
            # (this MultiAgentPolicyManager)
            policy.set_agent_id(i + 1)

    def replace_policy(self, policy, agent_id):
        self.policies[agent_id - 1] = policy
        policy.set_agent_id(agent_id)

    def forward(self,
                batch: Batch,
                state: Optional[Union[dict, Batch]] = None,
                **kwargs) -> Batch:
        """
        :param state: if None, it means all agents have no state.
            If not None, it should contain keys of agent_1, agent_2, ...

        :return a Batch with the following contents
        {
        "act": actions corresponding to the input
        "state":{
            "agent_1": output state of agent_1's policy for the state
            "agent_2": xxx
            ...
            "agent_n": xxx}
        "out":{
            "agent_1": output of agent_1's policy for the input
            "agent_2": xxx
            ...
            "agent_n": xxx}
        }
        """
        results = []
        for policy in self.policies:
            # This part of code is difficult to understand.
            # Let's follow an example with two agents
            # batch.obs.agent_id is [1, 2, 1, 2, 1, 2]
            # each agent plays for three transitions
            # agent_index for agent 1 is [0, 2, 4]
            # agent_index for agent 2 is [1, 3, 5]
            # we separate the transition of each agent
            agent_index = np.nonzero(batch.obs.agent_id == policy.agent_id)[0]
            if len(agent_index) == 0:
                # (has_data, agent_index, out, act, state)
                results.append((False, None, Batch(), None, Batch()))
                continue
            tmp_batch = batch[agent_index]
            if isinstance(tmp_batch.rew, np.ndarray):
                # reward can be empty Batch (after initial reset) or nparray.
                tmp_batch.rew = tmp_batch.rew[:, policy.agent_id - 1]
            out = policy(
                batch=tmp_batch,
                state=state and state["agent_" + str(policy.agent_id)],
                **kwargs)
            act = out.act
            each_state = out.state \
                if (hasattr(out, 'state') and out.state is not None) \
                else Batch()
            results.append((True, agent_index, out, act, each_state))
        holder = Batch.cat([{'act': e[3]} for e in results if e[0]])
        state_dict = {}
        out_dict = {}
        for policy, (has_data, agent_index, out, act, state) in \
                zip(self.policies, results):
            if has_data:
                holder.act[agent_index] = act
            state_dict["agent_" + str(policy.agent_id)] = state
            out_dict["agent_" + str(policy.agent_id)] = out
        holder["out"] = out_dict
        holder["state"] = state_dict
        return holder

    def learn(self, batch: Batch, **kwargs)\
            -> Dict[str, Union[float, List[float]]]:
        results = {}
        for policy in self.policies:
            agent_index = np.nonzero(batch.obs.agent_id == policy.agent_id)[0]
            out = policy.learn(batch=batch[agent_index], **kwargs)
            for k, v in out.items():
                results["agent_" + str(policy.agent_id) + '/' + k] = v
        return results

    def process_fn(self, batch: Batch, buffer: ReplayBuffer,
                   indice: np.ndarray) -> Batch:
        results = []
        # save original multi-dimensional rew in save_rew,
        # set rew to the reward of each agent during their ``process_fn``,
        # and restore the original reward afterwards
        has_rew = isinstance(buffer.rew, np.ndarray)
        if has_rew:
            buffer.save_rew = buffer.rew
            buffer.rew = Batch()
        for policy in self.policies:
            agent_index = np.nonzero(batch.obs.agent_id == policy.agent_id)[0]
            if len(agent_index) == 0:
                # has_data, data, agent_index
                results.append([False, None, None])
                continue
            tmp_batch = batch[agent_index]
            tmp_indice = indice[agent_index]
            if isinstance(tmp_batch.rew, np.ndarray):
                # reward can be empty Batch (after initial reset) or nparray.
                tmp_batch.rew = tmp_batch.rew[:, policy.agent_id - 1]
            if has_rew:
                buffer.rew = buffer.save_rew[:, policy.agent_id - 1]
            output = policy.process_fn(tmp_batch, buffer, tmp_indice)
            if has_rew:
                buffer.rew = Batch()
            results.append([True, output, agent_index])
        if has_rew:
            buffer.rew = buffer.save_rew
        # incompatible keys will be padded with zeros
        # e.g. agent 1 batch has ``returns`` but agent 2 does not
        holder = Batch.cat(
            [data.condense() for (has_data, data, _) in results if has_data])
        for has_data, data, agent_index in results:
            if has_data:
                holder[agent_index] = data
        return holder
