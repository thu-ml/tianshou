from collections.abc import Callable, Sequence
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch

from tianshou.env.utils import ENV_TYPE, gym_new_venv_step_type
from tianshou.env.worker import (
    DummyEnvWorker,
    EnvWorker,
    RayEnvWorker,
    SubprocEnvWorker,
)

GYM_RESERVED_KEYS = [
    "metadata",
    "reward_range",
    "spec",
    "action_space",
    "observation_space",
]


class BaseVectorEnv:
    """Base class for vectorized environments.

    Usage:
    ::

        env_num = 8
        envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(env_num)])
        assert len(envs) == env_num

    It accepts a list of environment generators. In other words, an environment
    generator ``efn`` of a specific task means that ``efn()`` returns the
    environment of the given task, for example, ``gym.make(task)``.

    All of the VectorEnv must inherit :class:`~tianshou.env.BaseVectorEnv`.
    Here are some other usages:
    ::

        envs.seed(2)  # which is equal to the next line
        envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # set specific seed for each env
        obs = envs.reset()  # reset all environments
        obs = envs.reset([0, 5, 7])  # reset 3 specific environments
        obs, rew, done, info = envs.step([1] * 8)  # step synchronously
        envs.render()  # render all environments
        envs.close()  # close all environments

    .. warning::

        If you use your own environment, please make sure the ``seed`` method
        is set up properly, e.g.,
        ::

            def seed(self, seed):
                np.random.seed(seed)

        Otherwise, the outputs of these envs may be the same with each other.

    :param env_fns: a list of callable envs, ``env_fns[i]()`` generates the i-th env.
    :param worker_fn: a callable worker, ``worker_fn(env_fns[i])`` generates a
        worker which contains the i-th env.
    :param wait_num: use in asynchronous simulation if the time cost of
        ``env.step`` varies with time and synchronously waiting for all
        environments to finish a step is time-wasting. In that case, we can
        return when ``wait_num`` environments finish a step and keep on
        simulation in these environments. If ``None``, asynchronous simulation
        is disabled; else, ``1 <= wait_num <= env_num``.
    :param timeout: use in asynchronous simulation same as above, in each
        vectorized step it only deal with those environments spending time
        within ``timeout`` seconds.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], ENV_TYPE]],
        worker_fn: Callable[[Callable[[], ENV_TYPE]], EnvWorker],
        wait_num: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self._env_fns = env_fns
        # A VectorEnv contains a pool of EnvWorkers, which corresponds to
        # interact with the given envs (one worker <-> one env).
        self.workers = [worker_fn(fn) for fn in env_fns]
        self.worker_class = type(self.workers[0])
        assert issubclass(self.worker_class, EnvWorker)
        assert all(isinstance(w, self.worker_class) for w in self.workers)

        self.env_num = len(env_fns)
        self.wait_num = wait_num or len(env_fns)
        assert 1 <= self.wait_num <= len(env_fns), (
            f"wait_num should be in [1, {len(env_fns)}], but got {wait_num}"
        )
        self.timeout = timeout
        assert self.timeout is None or self.timeout > 0, (
            f"timeout is {timeout}, it should be positive if provided!"
        )
        self.is_async = self.wait_num != len(env_fns) or timeout is not None
        self.waiting_conn: list[EnvWorker] = []
        # environments in self.ready_id is actually ready
        # but environments in self.waiting_id are just waiting when checked,
        # and they may be ready now, but this is not known until we check it
        # in the step() function
        self.waiting_id: list[int] = []
        # all environments are ready in the beginning
        self.ready_id = list(range(self.env_num))
        self.is_closed = False

    def _assert_is_not_closed(self) -> None:
        assert not self.is_closed, (
            f"Methods of {self.__class__.__name__} cannot be called after close."
        )

    def __len__(self) -> int:
        """Return len(self), which is the number of environments."""
        return self.env_num

    def __getattribute__(self, key: str) -> Any:
        """Switch the attribute getter depending on the key.

        Any class who inherits ``gym.Env`` will inherit some attributes, like
        ``action_space``. However, we would like the attribute lookup to go straight
        into the worker (in fact, this vector env's action_space is always None).
        """
        if key in GYM_RESERVED_KEYS:  # reserved keys in gym.Env
            return self.get_env_attr(key)
        return super().__getattribute__(key)

    def get_env_attr(
        self,
        key: str,
        id: int | list[int] | np.ndarray | None = None,
    ) -> list[Any]:
        """Get an attribute from the underlying environments.

        If id is an int, retrieve the attribute denoted by key from the environment
        underlying the worker at index id. The result is returned as a list with one
        element. Otherwise, retrieve the attribute for all workers at indices id and
        return a list that is ordered correspondingly to id.

        :param str key: The key of the desired attribute.
        :param id: Indice(s) of the desired worker(s). Default to None for all env_id.

        :return list: The list of environment attributes.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        return [self.workers[j].get_env_attr(key) for j in id]

    def set_env_attr(
        self,
        key: str,
        value: Any,
        id: int | list[int] | np.ndarray | None = None,
    ) -> None:
        """Set an attribute in the underlying environments.

        If id is an int, set the attribute denoted by key from the environment
        underlying the worker at index id to value.
        Otherwise, set the attribute for all workers at indices id.

        :param str key: The key of the desired attribute.
        :param Any value: The new value of the attribute.
        :param id: Indice(s) of the desired worker(s). Default to None for all env_id.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        for j in id:
            self.workers[j].set_env_attr(key, value)

    def _wrap_id(
        self,
        id: int | list[int] | np.ndarray | None = None,
    ) -> list[int] | np.ndarray:
        if id is None:
            return list(range(self.env_num))
        return [id] if np.isscalar(id) else id  # type: ignore

    def _assert_id(self, id: list[int] | np.ndarray) -> None:
        for i in id:
            assert i not in self.waiting_id, (
                f"Cannot interact with environment {i} which is stepping now."
            )
            assert i in self.ready_id, f"Can only interact with ready environments {self.ready_id}."

    # TODO: for now, has to be kept in sync with reset in EnvPoolMixin
    #  In particular, can't rename env_id to env_ids
    def reset(
        self,
        env_id: int | list[int] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset the state of some envs and return initial observations.

        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        env_id = self._wrap_id(env_id)
        if self.is_async:
            self._assert_id(env_id)

        # send(None) == reset() in worker
        for id in env_id:
            self.workers[id].send(None, **kwargs)
        ret_list = [self.workers[id].recv() for id in env_id]

        assert (
            isinstance(ret_list[0], tuple | list)
            and len(ret_list[0]) == 2
            and isinstance(ret_list[0][1], dict)
        ), "The environment does not adhere to the Gymnasium's API."

        obs_list = [r[0] for r in ret_list]

        if isinstance(obs_list[0], tuple):  # type: ignore
            raise TypeError(
                "Tuple observation space is not supported. ",
                "Please change it to array or dict space",
            )
        try:
            obs = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs = np.array(obs_list, dtype=object)

        infos = np.array([r[1] for r in ret_list])
        return obs, infos

    def step(
        self,
        action: np.ndarray | torch.Tensor | None,
        id: int | list[int] | np.ndarray | None = None,
    ) -> gym_new_venv_step_type:
        """Run one timestep of some environments' dynamics.

        If id is None, run one timestep of all the environments` dynamics;
        otherwise run one timestep for some environments with given id,  either
        an int or a list. When the end of episode is reached, you are
        responsible for calling reset(id) to reset this environment`s state.

        Accept a batch of action and return a tuple (batch_obs, batch_rew,
        batch_done, batch_info) in numpy format.

        :param numpy.ndarray action: a batch of action provided by the agent.
            If the venv is async, the action can be None, which will result
            in all arrays in the returned tuple being empty.

        :return: A tuple consisting of either:

            * ``obs`` a numpy.ndarray, the agent's observation of current environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``terminated`` a numpy.ndarray, whether these episodes have been \
                terminated
            * ``truncated`` a numpy.ndarray, whether these episodes have been truncated
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)

        For the async simulation:

        Provide the given action to the environments. The action sequence
        should correspond to the ``id`` argument, and the ``id`` argument
        should be a subset of the ``env_id`` in the last returned ``info``
        (initially they are env_ids of all the environments). If action is
        None, fetch unfinished step() calls instead.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if not self.is_async:
            if action is None:
                raise ValueError("action must be not-None for non-async")
            assert len(action) == len(id)
            for i, j in enumerate(id):
                self.workers[j].send(action[i])
            result = []
            for j in id:
                env_return = self.workers[j].recv()
                env_return[-1]["env_id"] = j
                result.append(env_return)
        else:
            if action is not None:
                self._assert_id(id)
                assert len(action) == len(id)
                for act, env_id in zip(action, id, strict=True):
                    self.workers[env_id].send(act)
                    self.waiting_conn.append(self.workers[env_id])
                    self.waiting_id.append(env_id)
                self.ready_id = [x for x in self.ready_id if x not in id]
            ready_conns: list[EnvWorker] = []
            while not ready_conns:
                ready_conns = self.worker_class.wait(self.waiting_conn, self.wait_num, self.timeout)
            result = []
            for conn in ready_conns:
                waiting_index = self.waiting_conn.index(conn)
                self.waiting_conn.pop(waiting_index)
                env_id = self.waiting_id.pop(waiting_index)
                # env_return can be (obs, reward, done, info) or
                # (obs, reward, terminated, truncated, info)
                env_return = conn.recv()
                env_return[-1]["env_id"] = env_id  # Add `env_id` to info
                result.append(env_return)
                self.ready_id.append(env_id)
        obs_list, rew_list, term_list, trunc_list, info_list = tuple(zip(*result, strict=True))
        try:
            obs_stack = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs_stack = np.array(obs_list, dtype=object)
        return (
            obs_stack,
            np.stack(rew_list),
            np.stack(term_list),
            np.stack(trunc_list),
            np.stack(info_list),
        )

    def seed(self, seed: int | list[int] | None = None) -> list[list[int] | None]:
        """Set the seed for all environments.

        Accept ``None``, an int (which will extend ``i`` to
        ``[i, i + 1, i + 2, ...]``) or a list.

        :return: The list of seeds used in this env's random number generators.
            The first value in the list should be the "main" seed, or the value
            which a reproducer pass to "seed".
        """
        self._assert_is_not_closed()
        seed_list: list[None] | list[int]
        if seed is None:
            seed_list = [seed] * self.env_num
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.env_num)]
        else:
            seed_list = seed
        return [w.seed(s) for w, s in zip(self.workers, seed_list, strict=True)]

    def render(self, **kwargs: Any) -> list[Any]:
        """Render all of the environments."""
        self._assert_is_not_closed()
        if self.is_async and len(self.waiting_id) > 0:
            raise RuntimeError(
                f"Environments {self.waiting_id} are still stepping, cannot render them now.",
            )
        return [w.render(**kwargs) for w in self.workers]

    def close(self) -> None:
        """Close all of the environments.

        This function will be called only once (if not, it will be called during
        garbage collected). This way, ``close`` of all workers can be assured.
        """
        self._assert_is_not_closed()
        for w in self.workers:
            w.close()
        self.is_closed = True


class DummyVectorEnv(BaseVectorEnv):
    """Dummy vectorized environment wrapper, implemented in for-loop.

    This has the same interface as true vectorized environment, but the rollout does not happen in parallel.
    So, all workers just wait for each other and the environment is as efficient as using a single environment.
    This can be useful for testing or for demonstration purposes.

    A rare use-case would be using vector based interface, but parallelization is not desired
    (e.g. because of too much overhead). However, in such cases one should consider using a single environment.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], ENV_TYPE]],
        wait_num: int | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(env_fns, DummyEnvWorker, wait_num, timeout)


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.

        Additional arguments are:

        :param share_memory: whether to share memory between the main process and the worker process. Allows for
            shared buffers to exchange observations
        :param context: the context to use for multiprocessing. Usually it's fine to use the default context, but
            `spawn` as well as `fork` can have non-obvious side effects, see for example
            https://github.com/google-deepmind/mujoco/issues/742, or
            https://github.com/Farama-Foundation/Gymnasium/issues/222.
            Consider using 'fork' when using macOS and additional parallelization, for example via joblib.
            Defaults to None, which will use the default system context.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], ENV_TYPE]],
        wait_num: int | None = None,
        timeout: float | None = None,
        share_memory: bool = False,
        context: Literal["fork", "spawn"] | None = None,
    ) -> None:
        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            return SubprocEnvWorker(fn, share_memory=share_memory, context=context)

        super().__init__(
            env_fns,
            worker_fn,
            wait_num,
            timeout,
        )


class ShmemVectorEnv(BaseVectorEnv):
    """Optimized SubprocVectorEnv with shared buffers to exchange observations.

    ShmemVectorEnv has exactly the same API as SubprocVectorEnv.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], ENV_TYPE]],
        wait_num: int | None = None,
        timeout: float | None = None,
    ) -> None:
        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            return SubprocEnvWorker(fn, share_memory=True)

        super().__init__(env_fns, worker_fn, wait_num, timeout)


class RayVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on ray.

    This is a choice to run distributed environments in a cluster.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], ENV_TYPE]],
        wait_num: int | None = None,
        timeout: float | None = None,
    ) -> None:
        try:
            import ray
        except ImportError as exception:
            raise ImportError(
                "Please install ray to support RayVectorEnv: pip install ray",
            ) from exception
        if not ray.is_initialized():
            ray.init()
        super().__init__(env_fns, lambda env_fn: RayEnvWorker(env_fn), wait_num, timeout)
