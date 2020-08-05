import gym
import numpy as np
import warnings
from typing import List, Tuple, Union, Optional, Callable, Any
from tianshou.env.worker.base import EnvWorker
from tianshou.env.worker.subproc import SubProcEnvWorker
from tianshou.env.worker.dummy import SequentialEnvWorker


def run_once(f):
    """
    Run once decorator for a method in a class. Each instance can run
    the method at most once.
    """
    f.has_run_objects = set()

    def wrapper(self, *args, **kwargs):
        if self.unique_id in f.has_run_objects:
            raise RuntimeError(
                f'{f} can be called only once for object {self}')
        f.has_run_objects.add(self.unique_id)
        return f(self, *args, **kwargs)
    return wrapper


def generate_id():
    generate_id.i += 1
    return generate_id.i


generate_id.i = 0


class BaseVectorEnv(gym.Env):
    """Base class for vectorized environments wrapper. Usage:
    ::

        env_num = 8
        envs = VectorEnv([lambda: gym.make(task) for _ in range(env_num)])
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


    :param wait_num: used in asynchronous simulation if the time cost of
        ``env.step`` varies with time and synchronously waiting for all
        environments to finish a step is time-wasting. In that case, we can
        return when ``wait_num`` environments finish a step and keep on
        simulation in these environments. If ``None``, asynchronous simulation
        is disabled; else, ``1 <= wait_num <= env_num``.

    """

    def __init__(self,
                 env_fns: List[Callable[[], gym.Env]],
                 worker_fn: Callable[[Callable[[], gym.Env]], EnvWorker],
                 wait_num: Optional[int] = None,
                 ) -> None:
        self._env_fns = env_fns
        self.workers = [worker_fn(fn) for fn in env_fns]
        self.env_num = len(env_fns)
        self.worker_class = self.workers[0].__class__

        self.wait_num = wait_num or len(env_fns)
        assert 1 <= self.wait_num <= len(env_fns), \
            f'wait_num should be in [1, {len(env_fns)}], but got {wait_num}'
        self.is_async = wait_num is not None
        self.waiting_conn = []
        # environments in self.ready_id is actually ready
        # but environments in self.waiting_id are just waiting when checked,
        # and they may be ready now, but this is not known until we check it
        # in the step() function
        self.waiting_id = []
        # all environments are ready in the beginning
        self.ready_id = list(range(self.env_num))
        self.unique_id = generate_id()

    def __len__(self) -> int:
        """Return len(self), which is the number of environments."""
        return self.env_num

    def __getattribute__(self, key: str):
        """Switch between the default attribute getter or one
           looking at wrapped environment level depending on the key."""
        if key not in ('observation_space', 'action_space'):
            return super().__getattribute__(key)
        else:
            return self.__getattr__(key)

    def __getattr__(self, key: str):
        """Try to retrieve an attribute from each individual wrapped
           environment, if it does not belong to the wrapping vector
           environment class."""
        return [getattr(worker, key) for worker in self.workers]

    def _assert_and_transform_id(self,
                                 id: Optional[Union[int, List[int]]] = None
                                 ) -> List[int]:
        if id is None:
            id = list(range(self.env_num))
        elif np.isscalar(id):
            id = [id]
        for i in id:
            assert i not in self.waiting_id, \
                f'Cannot manipulate environment {i} which is stepping now!'
            assert i in self.ready_id, \
                f'Can only manipulate ready environments {self.ready_id}.'
        return id

    def reset(self, id: Optional[Union[int, List[int]]] = None):
        """Reset the state of all the environments and return initial
        observations if id is ``None``, otherwise reset the specific
        environments with given id, either an int or a list.
        """
        if id is None:
            id = range(self.env_num)
        elif np.isscalar(id):
            id = [id]
        if self.is_async:
            id = self._assert_and_transform_id(id)
        obs = np.stack([self.workers[i].reset() for i in id])
        return obs

    def step(self,
             action: Optional[np.ndarray],
             id: Optional[Union[int, List[int]]] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run one timestep of all the environments’ dynamics if id is
        ``None``, otherwise run one timestep for some environments
        with given id,  either an int or a list. When the end of
        episode is reached, you are responsible for calling reset(id)
        to reset this environment’s state.

        Accept a batch of action and return a tuple (obs, rew, done, info).

        :param numpy.ndarray action: a batch of action provided by the agent.

        :return: A tuple including four items:

            * ``obs`` a numpy.ndarray, the agent's observation of current \
                environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``done`` a numpy.ndarray, whether these episodes have ended, in \
                which case further step() calls will return undefined results
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)

        For the async simulation:

        Provide the given action to the environments. The action sequence
        should correspond to the ``id`` argument, and the ``id`` argument
        should be a subset of the ``env_id`` in the last returned ``info``
        (initially they are env_ids of all the environments). If action is
        ``None``, fetch unfinished step() calls instead.
        """
        if not self.is_async:
            if id is None:
                id = range(self.env_num)
            elif np.isscalar(id):
                id = [id]
            assert len(action) == len(id)
            result = [self.workers[j].step(action[i]) for
                      i, j in enumerate(id)]
            obs, rew, done, info = map(np.stack, zip(*result))
            return obs, rew, done, info
        else:
            if action is not None:
                id = self._assert_and_transform_id(id)
                assert len(action) == len(id)
                for i, (act, env_id) in enumerate(zip(action, id)):
                    self.workers[env_id].send_action(act)
                    self.waiting_conn.append(self.workers[env_id])
                    self.waiting_id.append(env_id)
                self.ready_id = [x for x in self.ready_id if x not in id]
            result = []
            while len(self.waiting_conn) > 0 and len(result) < self.wait_num:
                ready_conns = self.worker_class.wait(self.waiting_conn)
                for conn in ready_conns:
                    waiting_index = self.waiting_conn.index(conn)
                    self.waiting_conn.pop(waiting_index)
                    env_id = self.waiting_id.pop(waiting_index)
                    ans = conn.get_result()
                    obs, rew, done, info = ans
                    info["env_id"] = env_id
                    result.append((obs, rew, done, info))
                    self.ready_id.append(env_id)
            obs, rew, done, info = map(np.stack, zip(*result))
            return obs, rew, done, info

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> List[int]:
        """Set the seed for all environments.

        Accept ``None``, an int (which will extend ``i`` to
        ``[i, i + 1, i + 2, ...]``) or a list.

        :return: The list of seeds used in this env's random number \
        generators. The first value in the list should be the "main" seed, or \
        the value which a reproducer pass to "seed".
        """
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        result = [w.seed(s) for w, s in zip(self.workers, seed)]
        return result

    def render(self, **kwargs) -> List[Any]:
        """Render all of the environments."""
        if self.is_async and len(self.waiting_id) > 0:
            raise RuntimeError(
                f"Environments {self.waiting_id} are still "
                f"stepping, cannot render them now.")
        return [w.render(**kwargs) for w in self.workers]

    @run_once
    def close(self) -> List[Any]:
        """Close all of the environments. This function will be called
        only once (if not, it will be called during garbage collected).
        This way, ``close`` of all workers can be assured.
        """
        if self.is_async:
            # finish remaining steps, and close
            if len(self.waiting_conn) > 0:
                self.step(None)
        return [w.close() for w in self.workers]

    def __del__(self):
        """Close the environment before garbage collected"""
        try:
            self.close()
        except RuntimeError:
            # it has already been closed
            pass


class ForLoopVectorEnv(BaseVectorEnv):
    def __init__(self,
                 env_fns: List[Callable[[], gym.Env]],
                 wait_num: Optional[int] = None,
                 ) -> None:
        super(ForLoopVectorEnv, self).__init__(
            env_fns,
            lambda fn: SequentialEnvWorker(fn),
            wait_num=wait_num,
        )


class VectorEnv(BaseVectorEnv):
    def __init__(self,
                 env_fns: List[Callable[[], gym.Env]],
                 wait_num: Optional[int] = None,
                 ) -> None:
        warnings.warn(
            'VectorEnv is renamed to ForLoopVectorEnv, and will be removed'
            ' in 0.3. Use ForLoopVectorEnv instead!', DeprecationWarning)
        super(VectorEnv, self).__init__(
            env_fns,
            lambda fn: SequentialEnvWorker(fn),
            wait_num=wait_num,
        )


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self,
                 env_fns: List[Callable[[], gym.Env]],
                 wait_num: Optional[int] = None,
                 ) -> None:
        super(SubprocVectorEnv, self).__init__(
            env_fns,
            lambda fn: SubProcEnvWorker(fn),
            wait_num=wait_num,
        )


class ShmemVectorEnv(BaseVectorEnv):
    """Optimized version of SubprocVectorEnv that uses shared variables to
    communicate observations. SubprocVectorEnv has exactly the same API as
    SubprocVectorEnv.

    .. seealso::

        Please refer to :class:`~tianshou.env.SubprocVectorEnv` for more
        detailed explanation.
    """

    def __init__(self,
                 env_fns: List[Callable[[], gym.Env]],
                 wait_num: Optional[int] = None,
                 ) -> None:
        super(ShmemVectorEnv, self).__init__(
            env_fns,
            lambda fn: SubProcEnvWorker(fn, share_memory=True),
            wait_num=wait_num,
        )


class RayVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on
    `ray <https://github.com/ray-project/ray>`_. This is a choice to run
    distributed environments in a cluster.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self,
                 env_fns: List[Callable[[], gym.Env]],
                 wait_num: Optional[int] = None,
                 ) -> None:
        try:
            import ray
        except ImportError as e:
            raise ImportError(
                'Please install ray to support RayVectorEnv: pip install ray'
            ) from e

        if not ray.is_initialized():
            ray.init()
        from tianshou.env.worker.ray import RayEnvWorker
        super().__init__(
            env_fns,
            lambda fn: RayEnvWorker(fn),
            wait_num=wait_num,
        )
