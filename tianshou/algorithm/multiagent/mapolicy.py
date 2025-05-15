from collections.abc import Callable
from typing import Any, Generic, Literal, Protocol, Self, TypeVar, cast, overload

import numpy as np
from overrides import override
from sensai.util.helper import mark_used
from torch.nn import ModuleList

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.batch import BatchProtocol, IndexType
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.algorithm import Algorithm
from tianshou.algorithm.base import (
    OffPolicyAlgorithm,
    OnPolicyAlgorithm,
    Policy,
    TrainingStats,
)

try:
    from tianshou.env.pettingzoo_env import PettingZooEnv
except ImportError:
    PettingZooEnv = None  # type: ignore


mark_used(ActBatchProtocol)


class MapTrainingStats(TrainingStats):
    def __init__(
        self,
        agent_id_to_stats: dict[str | int, TrainingStats],
        train_time_aggregator: Literal["min", "max", "mean"] = "max",
    ) -> None:
        self._agent_id_to_stats = agent_id_to_stats
        train_times = [agent_stats.train_time for agent_stats in agent_id_to_stats.values()]
        match train_time_aggregator:
            case "max":
                aggr_function = max
            case "min":
                aggr_function = min
            case "mean":
                aggr_function = np.mean  # type: ignore
            case _:
                raise ValueError(
                    f"Unknown {train_time_aggregator=}",
                )
        self.train_time = aggr_function(train_times)
        self.smoothed_loss = {}

    @override
    def get_loss_stats_dict(self) -> dict[str, float]:
        """Collects loss_stats_dicts from all agents, prepends agent_id to all keys, and joins results."""
        result_dict = {}
        for agent_id, stats in self._agent_id_to_stats.items():
            agent_loss_stats_dict = stats.get_loss_stats_dict()
            for k, v in agent_loss_stats_dict.items():
                result_dict[f"{agent_id}/" + k] = v
        return result_dict


class MAPRolloutBatchProtocol(RolloutBatchProtocol, Protocol):
    # TODO: this might not be entirely correct.
    #  The whole MAP data processing pipeline needs more documentation and possibly some refactoring
    @overload
    def __getitem__(self, index: str) -> RolloutBatchProtocol:
        ...

    @overload
    def __getitem__(self, index: IndexType) -> Self:
        ...

    def __getitem__(self, index: str | IndexType) -> Any:
        ...


class MultiAgentPolicy(Policy):
    def __init__(self, policies: dict[str | int, Policy]):
        p0 = next(iter(policies.values()))
        super().__init__(
            action_space=p0.action_space,
            observation_space=p0.observation_space,
            action_scaling=False,
            action_bound_method=None,
        )
        self.policies = policies
        self._submodules = ModuleList(policies.values())

    _TArrOrActBatch = TypeVar("_TArrOrActBatch", bound="np.ndarray | ActBatchProtocol")

    def add_exploration_noise(
        self,
        act: _TArrOrActBatch,
        batch: ObsBatchProtocol,
    ) -> _TArrOrActBatch:
        """Add exploration noise from sub-policy onto act."""
        if not isinstance(batch.obs, Batch):
            raise TypeError(
                f"here only observations of type Batch are permitted, but got {type(batch.obs)}",
            )
        for agent_id, policy in self.policies.items():
            agent_index = np.nonzero(batch.obs.agent_id == agent_id)[0]
            if len(agent_index) == 0:
                continue
            act[agent_index] = policy.add_exploration_noise(act[agent_index], batch[agent_index])
        return act

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: dict | Batch | None = None,
        **kwargs: Any,
    ) -> Batch:
        """Dispatch batch data from obs.agent_id to every policy's forward.

        :param batch: TODO: document what is expected at input and make a BatchProtocol for it
        :param state: if None, it means all agents have no state. If not
            None, it should contain keys of "agent_1", "agent_2", ...

        :return: a Batch with the following contents:
            TODO: establish a BatcProtocol for this

        ::

            {
                "act": actions corresponding to the input
                "state": {
                    "agent_1": output state of agent_1's policy for the state
                    "agent_2": xxx
                    ...
                    "agent_n": xxx}
                "out": {
                    "agent_1": output of agent_1's policy for the input
                    "agent_2": xxx
                    ...
                    "agent_n": xxx}
            }
        """
        results: list[tuple[bool, np.ndarray, Batch, np.ndarray | Batch, Batch]] = []
        for agent_id, policy in self.policies.items():
            # This part of code is difficult to understand.
            # Let's follow an example with two agents
            # batch.obs.agent_id is [1, 2, 1, 2, 1, 2] (with batch_size == 6)
            # each agent plays for three transitions
            # agent_index for agent 1 is [0, 2, 4]
            # agent_index for agent 2 is [1, 3, 5]
            # we separate the transition of each agent according to agent_id
            agent_index = np.nonzero(batch.obs.agent_id == agent_id)[0]
            if len(agent_index) == 0:
                # (has_data, agent_index, out, act, state)
                results.append((False, np.array([-1]), Batch(), Batch(), Batch()))
                continue
            tmp_batch = batch[agent_index]
            if "rew" in tmp_batch.get_keys() and isinstance(tmp_batch.rew, np.ndarray):
                # reward can be empty Batch (after initial reset) or nparray.
                tmp_batch.rew = tmp_batch.rew[:, self.agent_idx[agent_id]]
            if not hasattr(tmp_batch.obs, "mask"):
                if hasattr(tmp_batch.obs, "obs"):
                    tmp_batch.obs = tmp_batch.obs.obs
                if hasattr(tmp_batch.obs_next, "obs"):
                    tmp_batch.obs_next = tmp_batch.obs_next.obs
            out = policy(
                batch=tmp_batch,
                state=None if state is None else state[agent_id],
                **kwargs,
            )
            act = out.act
            each_state = out.state if (hasattr(out, "state") and out.state is not None) else Batch()
            results.append((True, agent_index, out, act, each_state))
        holder: Batch = Batch.cat(
            [{"act": act} for (has_data, agent_index, out, act, each_state) in results if has_data],
        )
        state_dict, out_dict = {}, {}
        for (agent_id, _), (has_data, agent_index, out, act, state) in zip(
            self.policies.items(),
            results,
            strict=True,
        ):
            if has_data:
                holder.act[agent_index] = act
            state_dict[agent_id] = state
            out_dict[agent_id] = out
        holder["out"] = out_dict
        holder["state"] = state_dict
        return holder


TAlgorithm = TypeVar("TAlgorithm", bound=Algorithm)


class MARLDispatcher(Generic[TAlgorithm]):
    """
    Supports multi-agent learning by dispatching calls to the corresponding
    algorithm for each agent.
    """

    def __init__(self, algorithms: list[TAlgorithm], env: PettingZooEnv):
        agent_ids = env.agents
        assert len(algorithms) == len(agent_ids), "One policy must be assigned for each agent."
        self.algorithms: dict[str | int, TAlgorithm] = dict(zip(agent_ids, algorithms, strict=True))
        """maps agent_id to the corresponding algorithm."""
        self.agent_idx = env.agent_idx
        """maps agent_id to 0-based index."""

    def create_policy(self) -> MultiAgentPolicy:
        return MultiAgentPolicy({agent_id: a.policy for agent_id, a in self.algorithms.items()})

    def dispatch_process_fn(
        self,
        batch: MAPRolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> MAPRolloutBatchProtocol:
        """Dispatch batch data from `obs.agent_id` to every algorithm's processing function.

        Save original multi-dimensional rew in "save_rew", set rew to the
        reward of each agent during their "process_fn", and restore the
        original reward afterwards.
        """
        # TODO: maybe only str is actually allowed as agent_id? See MAPRolloutBatchProtocol
        results: dict[str | int, RolloutBatchProtocol] = {}
        assert isinstance(
            batch.obs,
            BatchProtocol,
        ), f"here only observations of type Batch are permitted, but got {type(batch.obs)}"
        # reward can be empty Batch (after initial reset) or nparray.
        has_rew = isinstance(buffer.rew, np.ndarray)
        if has_rew:  # save the original reward in save_rew
            # Since we do not override buffer.__setattr__, here we use _meta to
            # change buffer.rew, otherwise buffer.rew = Batch() has no effect.
            save_rew, buffer._meta.rew = buffer.rew, Batch()  # type: ignore
        for agent, algorithm in self.algorithms.items():
            agent_index = np.nonzero(batch.obs.agent_id == agent)[0]
            if len(agent_index) == 0:
                results[agent] = cast(RolloutBatchProtocol, Batch())
                continue
            tmp_batch, tmp_indice = batch[agent_index], indices[agent_index]
            if has_rew:
                tmp_batch.rew = tmp_batch.rew[:, self.agent_idx[agent]]
                buffer._meta.rew = save_rew[:, self.agent_idx[agent]]
            if not hasattr(tmp_batch.obs, "mask"):
                if hasattr(tmp_batch.obs, "obs"):
                    tmp_batch.obs = tmp_batch.obs.obs
                if hasattr(tmp_batch.obs_next, "obs"):
                    tmp_batch.obs_next = tmp_batch.obs_next.obs
            results[agent] = algorithm._preprocess_batch(tmp_batch, buffer, tmp_indice)
        if has_rew:  # restore from save_rew
            buffer._meta.rew = save_rew
        return cast(MAPRolloutBatchProtocol, Batch(results))

    def dispatch_update_with_batch(
        self,
        batch: MAPRolloutBatchProtocol,
        algorithm_update_with_batch_fn: Callable[[TAlgorithm, RolloutBatchProtocol], TrainingStats],
    ) -> MapTrainingStats:
        """Dispatch the respective subset of the batch data to each algorithm.

        :param batch: must map agent_ids to rollout batches
        :param algorithm_update_with_batch_fn: a function that performs the algorithm-specific
            update with the given agent-specific batch data
        """
        agent_id_to_stats = {}
        for agent_id, algorithm in self.algorithms.items():
            data = batch[agent_id]
            if len(data.get_keys()) != 0:
                train_stats = algorithm_update_with_batch_fn(algorithm, data)
                agent_id_to_stats[agent_id] = train_stats
        return MapTrainingStats(agent_id_to_stats)


class MultiAgentOffPolicyAlgorithm(OffPolicyAlgorithm[MultiAgentPolicy]):
    """Multi-agent reinforcement learning where each agent uses off-policy learning."""

    def __init__(
        self,
        *,
        algorithms: list[OffPolicyAlgorithm],
        env: PettingZooEnv,
    ) -> None:
        """
        :param algorithms: a list of off-policy algorithms.
        :param env: the multi-agent RL environment
        """
        self._dispatcher: MARLDispatcher[OffPolicyAlgorithm] = MARLDispatcher(algorithms, env)
        super().__init__(
            policy=self._dispatcher.create_policy(),
        )
        self._submodules = ModuleList(algorithms)

    def get_algorithm(self, agent_id: str | int) -> OffPolicyAlgorithm:
        return self._dispatcher.algorithms[agent_id]

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        batch = cast(MAPRolloutBatchProtocol, batch)
        return self._dispatcher.dispatch_process_fn(batch, buffer, indices)

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> MapTrainingStats:
        batch = cast(MAPRolloutBatchProtocol, batch)

        def update(algorithm: OffPolicyAlgorithm, data: RolloutBatchProtocol) -> TrainingStats:
            return algorithm._update_with_batch(data)

        return self._dispatcher.dispatch_update_with_batch(batch, update)


class MultiAgentOnPolicyAlgorithm(OnPolicyAlgorithm[MultiAgentPolicy]):
    """Multi-agent reinforcement learning where each agent uses on-policy learning."""

    def __init__(
        self,
        *,
        algorithms: list[OnPolicyAlgorithm],
        env: PettingZooEnv,
    ) -> None:
        """
        :param algorithms: a list of off-policy algorithms.
        :param env: the multi-agent RL environment
        """
        self._dispatcher: MARLDispatcher[OnPolicyAlgorithm] = MARLDispatcher(algorithms, env)
        super().__init__(
            policy=self._dispatcher.create_policy(),
        )
        self._submodules = ModuleList(algorithms)

    def get_algorithm(self, agent_id: str | int) -> OnPolicyAlgorithm:
        return self._dispatcher.algorithms[agent_id]

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        batch = cast(MAPRolloutBatchProtocol, batch)
        return self._dispatcher.dispatch_process_fn(batch, buffer, indices)

    def _update_with_batch(
        self, batch: RolloutBatchProtocol, batch_size: int | None, repeat: int
    ) -> MapTrainingStats:
        batch = cast(MAPRolloutBatchProtocol, batch)

        def update(algorithm: OnPolicyAlgorithm, data: RolloutBatchProtocol) -> TrainingStats:
            return algorithm._update_with_batch(data, batch_size, repeat)

        return self._dispatcher.dispatch_update_with_batch(batch, update)
