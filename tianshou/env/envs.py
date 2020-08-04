import gym
import numpy as np
from typing import List, Tuple, Union, Optional, Callable, Any
from tianshou.env.worker.base import EnvWorker
from tianshou.env.worker.subproc import SubProcEnvWorker
from tianshou.env.worker.dummy import SequentialEnvWorker
from tianshou.env.worker.ray import RayEnvWorker


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
    """

    def __init__(self,
                 env_fns: List[Callable[[], gym.Env]],
                 worker_fn: Callable[[Callable[[], gym.Env]], EnvWorker]
                 ) -> None:
        self._env_fns = env_fns
        self.workers = [worker_fn(fn) for fn in env_fns]
        self.env_num = len(env_fns)

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

    def reset(self, id: Optional[Union[int, List[int]]] = None):
        """Reset the state of all the environments and return initial
        observations if id is ``None``, otherwise reset the specific
        environments with given id, either an int or a list.
        """
        if id is None:
            id = range(self.env_num)
        elif np.isscalar(id):
            id = [id]
        obs = np.stack([self.workers[i].reset() for i in id])
        return obs

    def step(self,
             action: np.ndarray,
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
        """
        if id is None:
            id = range(self.env_num)
        elif np.isscalar(id):
            id = [id]
        assert len(action) == len(id)
        result = [self.workers[j].step(action[i]) for i, j in enumerate(id)]
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
        return [w.render(**kwargs) for w in self.workers]

    def close(self) -> List[Any]:
        """Close all of the environments.

        Environments will automatically close() themselves when garbage
        collected or when the program exits.
        """
        return [w.close() for w in self.workers]


class VectorEnv(BaseVectorEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        super(VectorEnv, self).__init__(env_fns,
                                        lambda fn: SequentialEnvWorker(fn))


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        super(SubprocVectorEnv, self).__init__(
            env_fns,
            lambda fn: SubProcEnvWorker(fn)
        )


class ShmemVectorEnv(BaseVectorEnv):
    """Optimized version of SubprocVectorEnv that uses shared variables to
    communicate observations. SubprocVectorEnv has exactly the same API as
    SubprocVectorEnv.

    .. seealso::

        Please refer to :class:`~tianshou.env.SubprocVectorEnv` for more
        detailed explanation.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        super(ShmemVectorEnv, self).__init__(
            env_fns,
            lambda fn: SubProcEnvWorker(fn, share_memory=True)
        )


class RayVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on
    `ray <https://github.com/ray-project/ray>`_. This is a choice to run
    distributed environments in a cluster.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        try:
            import ray
        except ImportError as e:
            raise ImportError(
                'Please install ray to support RayVectorEnv: pip install ray'
            ) from e

        if not ray.is_initialized():
            ray.init()
        super().__init__(env_fns, lambda fn: RayEnvWorker(fn))
