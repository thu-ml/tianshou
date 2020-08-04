import gym
import numpy as np
from multiprocessing import connection
from typing import List, Tuple, Union, Optional, Callable, Any

from tianshou.env import SubprocVectorEnv


class AsyncVectorEnv(SubprocVectorEnv):
    """Vectorized asynchronous environment wrapper based on subprocess.

    :param wait_num: used in asynchronous simulation if the time cost of
        ``env.step`` varies with time and synchronously waiting for all
        environments to finish a step is time-wasting. In that case, we can
        return when ``wait_num`` environments finish a step and keep on
        simulation in these environments. If ``None``, asynchronous simulation
        is disabled; else, ``1 <= wait_num <= env_num``.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]],
                 wait_num: Optional[int] = None) -> None:
        super().__init__(env_fns)
        self.wait_num = wait_num or len(env_fns)
        assert 1 <= self.wait_num <= len(env_fns), \
            f'wait_num should be in [1, {len(env_fns)}], but got {wait_num}'
        self.waiting_conn = []
        # environments in self.ready_id is actually ready
        # but environments in self.waiting_id are just waiting when checked,
        # and they may be ready now, but this is not known until we check it
        # in the step() function
        self.waiting_id = []
        # all environments are ready in the beginning
        self.ready_id = list(range(self.env_num))

    def _assert_and_transform_id(self,
                                 id: Optional[Union[int, List[int]]] = None
                                 ) -> List[int]:
        if id is None:
            id = list(range(self.env_num))
        elif np.isscalar(id):
            id = [id]
        for i in id:
            assert i not in self.waiting_id, \
                f'Cannot reset environment {i} which is stepping now!'
            assert i in self.ready_id, \
                f'Can only reset ready environments {self.ready_id}.'
        return id

    def reset(self, id: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        self._assert_is_closed()
        id = self._assert_and_transform_id(id)
        return super().reset(id)

    def render(self, **kwargs) -> List[Any]:
        self._assert_is_closed()
        if len(self.waiting_id) > 0:
            raise RuntimeError(
                f"Environments {self.waiting_id} are still "
                f"stepping, cannot render them now.")
        return super().render(**kwargs)

    def close(self) -> List[Any]:
        if self.closed:
            return []
        # finish remaining steps, and close
        self.step(None)
        return super().close()

    def step(self,
             action: Optional[np.ndarray],
             id: Optional[Union[int, List[int]]] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Provide the given action to the environments. The action sequence
        should correspond to the ``id`` argument, and the ``id`` argument
        should be a subset of the ``env_id`` in the last returned ``info``
        (initially they are env_ids of all the environments). If action is
        ``None``, fetch unfinished step() calls instead.
        """
        self._assert_is_closed()
        if action is not None:
            id = self._assert_and_transform_id(id)
            assert len(action) == len(id)
            for i, (act, env_id) in enumerate(zip(action, id)):
                self.parent_remote[env_id].send(['step', act])
                self.waiting_conn.append(self.parent_remote[env_id])
                self.waiting_id.append(env_id)
            self.ready_id = [x for x in self.ready_id if x not in id]
        result = []
        while len(self.waiting_conn) > 0 and len(result) < self.wait_num:
            ready_conns = connection.wait(self.waiting_conn)
            for conn in ready_conns:
                waiting_index = self.waiting_conn.index(conn)
                self.waiting_conn.pop(waiting_index)
                env_id = self.waiting_id.pop(waiting_index)
                ans = conn.recv()
                obs, rew, done, info = ans
                info["env_id"] = env_id
                result.append((obs, rew, done, info))
                self.ready_id.append(env_id)
        obs, rew, done, info = map(np.stack, zip(*result))
        return obs, rew, done, info
