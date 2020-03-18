import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe
try:
    import ray
except ImportError:
    pass

from tianshou.utils import CloudpickleWrapper


class EnvWrapper(object):
    def __init__(self, env):
        self.env = env

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def seed(self, seed=None):
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)

    def render(self, **kwargs):
        if hasattr(self.env, 'render'):
            self.env.render(**kwargs)

    def close(self):
        self.env.close()


class FrameStack(EnvWrapper):
    def __init__(self, env, stack_num):
        """Stack last k frames."""
        super().__init__(env)
        self.stack_num = stack_num
        self._frames = deque([], maxlen=stack_num)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.stack_num):
            self._frames.append(obs)
        return self._get_obs()

    def _get_obs(self):
        try:
            return np.concatenate(self._frames, axis=-1)
        except ValueError:
            return np.stack(self._frames, axis=-1)


class BaseVectorEnv(ABC):
    def __init__(self, env_fns, reset_after_done):
        self._env_fns = env_fns
        self.env_num = len(env_fns)
        self._reset_after_done = reset_after_done
        self._done = np.zeros(self.env_num)

    def is_reset_after_done(self):
        return self._reset_after_done

    def __len__(self):
        return self.env_num

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def seed(self, seed=None):
        pass

    @abstractmethod
    def render(self, **kwargs):
        pass

    @abstractmethod
    def close(self):
        pass


class VectorEnv(BaseVectorEnv):
    """docstring for VectorEnv"""

    def __init__(self, env_fns, reset_after_done=False):
        super().__init__(env_fns, reset_after_done)
        self.envs = [_() for _ in env_fns]

    def reset(self):
        self._done = np.zeros(self.env_num)
        self._obs = np.stack([e.reset() for e in self.envs])
        return self._obs

    def step(self, action):
        assert len(action) == self.env_num
        result = []
        for i, e in enumerate(self.envs):
            if not self.is_reset_after_done() and self._done[i]:
                result.append([
                    self._obs[i], self._rew[i], self._done[i], self._info[i]])
            else:
                result.append(e.step(action[i]))
        self._obs, self._rew, self._done, self._info = zip(*result)
        if self.is_reset_after_done() and sum(self._done):
            self._obs = np.stack(self._obs)
            for i in np.where(self._done)[0]:
                self._obs[i] = self.envs[i].reset()
        return np.stack(self._obs), np.stack(self._rew),\
            np.stack(self._done), np.stack(self._info)

    def seed(self, seed=None):
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        for e, s in zip(self.envs, seed):
            if hasattr(e, 'seed'):
                e.seed(s)

    def render(self, **kwargs):
        for e in self.envs:
            if hasattr(e, 'render'):
                e.render(**kwargs)

    def close(self):
        for e in self.envs:
            e.close()


def worker(parent, p, env_fn_wrapper, reset_after_done):
    parent.close()
    env = env_fn_wrapper.data()
    done = False
    try:
        while True:
            cmd, data = p.recv()
            if cmd == 'step':
                if reset_after_done or not done:
                    obs, rew, done, info = env.step(data)
                if reset_after_done and done:
                    # s_ is useless when episode finishes
                    obs = env.reset()
                p.send([obs, rew, done, info])
            elif cmd == 'reset':
                done = False
                p.send(env.reset())
            elif cmd == 'close':
                p.close()
                break
            elif cmd == 'render':
                p.send(env.render(**data) if hasattr(env, 'render') else None)
            elif cmd == 'seed':
                p.send(env.seed(data) if hasattr(env, 'seed') else None)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class SubprocVectorEnv(BaseVectorEnv):
    """docstring for SubProcVectorEnv"""

    def __init__(self, env_fns, reset_after_done=False):
        super().__init__(env_fns, reset_after_done)
        self.closed = False
        self.parent_remote, self.child_remote = \
            zip(*[Pipe() for _ in range(self.env_num)])
        self.processes = [
            Process(target=worker, args=(
                    parent, child,
                    CloudpickleWrapper(env_fn), reset_after_done), daemon=True)
            for (parent, child, env_fn) in zip(
                self.parent_remote, self.child_remote, env_fns)
        ]
        for p in self.processes:
            p.start()
        for c in self.child_remote:
            c.close()

    def step(self, action):
        assert len(action) == self.env_num
        for p, a in zip(self.parent_remote, action):
            p.send(['step', a])
        result = [p.recv() for p in self.parent_remote]
        obs, rew, done, info = zip(*result)
        return np.stack(obs), np.stack(rew), np.stack(done), np.stack(info)

    def reset(self):
        for p in self.parent_remote:
            p.send(['reset', None])
        return np.stack([p.recv() for p in self.parent_remote])

    def seed(self, seed=None):
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        for p, s in zip(self.parent_remote, seed):
            p.send(['seed', s])
        for p in self.parent_remote:
            p.recv()

    def render(self, **kwargs):
        for p in self.parent_remote:
            p.send(['render', kwargs])
        for p in self.parent_remote:
            p.recv()

    def close(self):
        if self.closed:
            return
        for p in self.parent_remote:
            p.send(['close', None])
        self.closed = True
        for p in self.processes:
            p.join()


class RayVectorEnv(BaseVectorEnv):
    """docstring for RayVectorEnv"""

    def __init__(self, env_fns, reset_after_done=False):
        super().__init__(env_fns, reset_after_done)
        try:
            if not ray.is_initialized():
                ray.init()
        except NameError:
            raise ImportError(
                'Please install ray to support RayVectorEnv: pip3 install ray')
        self.envs = [
            ray.remote(EnvWrapper).options(num_cpus=0).remote(e())
            for e in env_fns]

    def step(self, action):
        assert len(action) == self.env_num
        result_obj = []
        for i, e in enumerate(self.envs):
            if not self.is_reset_after_done() and self._done[i]:
                result_obj.append(None)
            else:
                result_obj.append(e.step.remote(action[i]))
        result = []
        for i, r in enumerate(result_obj):
            if r is None:
                result.append([
                    self._obs[i], self._rew[i], self._done[i], self._info[i]])
            else:
                result.append(ray.get(r))
        self._obs, self._rew, self._done, self._info = zip(*result)
        if self.is_reset_after_done() and sum(self._done):
            self._obs = np.stack(self._obs)
            index = np.where(self._done)[0]
            result_obj = []
            for i in range(len(index)):
                result_obj.append(self.envs[index[i]].reset.remote())
            for i in range(len(index)):
                self._obs[index[i]] = ray.get(result_obj[i])
        return np.stack(self._obs), np.stack(self._rew),\
            np.stack(self._done), np.stack(self._info)

    def reset(self):
        self._done = np.zeros(self.env_num)
        result_obj = [e.reset.remote() for e in self.envs]
        self._obs = np.stack([ray.get(r) for r in result_obj])
        return self._obs

    def seed(self, seed=None):
        if not hasattr(self.envs[0], 'seed'):
            return
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        result_obj = [e.seed.remote(s) for e, s in zip(self.envs, seed)]
        for r in result_obj:
            ray.get(r)

    def render(self, **kwargs):
        if not hasattr(self.envs[0], 'render'):
            return
        result_obj = [e.render.remote(**kwargs) for e in self.envs]
        for r in result_obj:
            ray.get(r)

    def close(self):
        result_obj = [e.close.remote() for e in self.envs]
        for r in result_obj:
            ray.get(r)
