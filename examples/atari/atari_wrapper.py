# Borrow a lot from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

import warnings
from collections import deque

import cv2
import gym
import numpy as np

from tianshou.env import ShmemVectorEnv

try:
    import envpool
except ImportError:
    envpool = None


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param gym.Env env: the environment to wrap.
    :param int noop_max: the maximum value of no-ops to run.
    """

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        self.env.reset()
        if hasattr(self.unwrapped.np_random, "integers"):
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame (frameskipping) using most recent raw
    observations (for max pooling across time steps)

    :param gym.Env env: the environment to wrap.
    :param int skip: number of `skip`-th frame.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Step the environment with the given action. Repeat action, sum
        reward, and max over last observations.
        """
        obs_list, total_reward, done = [], 0., False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            obs_list.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(obs_list[-2:], axis=0)
        return max_frame, total_reward, done, info


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over. It
    helps the value estimation.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal, then update lives to
        # handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few
            # frames, so its important to keep lives > 0, so that we only reset
            # once the environment is actually done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Calls the Gym environment reset, only when lives are exhausted. This
        way all states are still reachable even though lives are episodic, and
        the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs = self.env.step(0)[0]
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing.
    Related discussion: https://github.com/openai/baselines/issues/240

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        return self.env.step(1)[0]


class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 as done in the Nature paper and later work.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.size = 84
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=(self.size, self.size),
            dtype=env.observation_space.dtype
        )

    def observation(self, frame):
        """returns the current observation from a frame"""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize observations to 0~1.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        low = np.min(env.observation_space.low)
        high = np.max(env.observation_space.high)
        self.bias = low
        self.scale = high - low
        self.observation_space = gym.spaces.Box(
            low=0., high=1., shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        return (observation - self.bias) / self.scale


class ClipRewardEnv(gym.RewardWrapper):
    """clips the reward to {+1, 0, -1} by its sign.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.reward_range = (-1, 1)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign. Note: np.sign(0) == 0."""
        return np.sign(reward)


class FrameStack(gym.Wrapper):
    """Stack n_frames last frames.

    :param gym.Env env: the environment to wrap.
    :param int n_frames: the number of frames to stack.
    """

    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shape = (n_frames, ) + env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=shape,
            dtype=env.observation_space.dtype
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        # the original wrapper use `LazyFrames` but since we use np buffer,
        # it has no effect
        return np.stack(self.frames, axis=0)


def wrap_deepmind(
    env_id,
    episode_life=True,
    clip_rewards=True,
    frame_stack=4,
    scale=False,
    warp_frame=True
):
    """Configure environment for DeepMind-style Atari. The observation is
    channel-first: (c, h, w) instead of (h, w, c).

    :param str env_id: the atari environment id.
    :param bool episode_life: wrap the episode life wrapper.
    :param bool clip_rewards: wrap the reward clipping wrapper.
    :param int frame_stack: wrap the frame stacking wrapper.
    :param bool scale: wrap the scaling observation wrapper.
    :param bool warp_frame: wrap the grayscale + resize observation wrapper.
    :return: the wrapped atari environment.
    """
    assert 'NoFrameskip' in env_id
    env = gym.make(env_id)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if warp_frame:
        env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, frame_stack)
    return env


def make_atari_env(task, seed, training_num, test_num, **kwargs):
    """Wrapper function for Atari env.

    If EnvPool is installed, it will automatically switch to EnvPool's Atari env.

    :return: a tuple of (single env, training envs, test envs).
    """
    if envpool is not None:
        if kwargs.get("scale", 0):
            warnings.warn(
                "EnvPool does not include ScaledFloatFrame wrapper, "
                "please set `x = x / 255.0` inside CNN network's forward function."
            )
        # parameters convertion
        train_envs = env = envpool.make_gym(
            task.replace("NoFrameskip-v4", "-v5"),
            num_envs=training_num,
            seed=seed,
            episodic_life=True,
            reward_clip=True,
            stack_num=kwargs.get("frame_stack", 4),
        )
        test_envs = envpool.make_gym(
            task.replace("NoFrameskip-v4", "-v5"),
            num_envs=test_num,
            seed=seed,
            episodic_life=False,
            reward_clip=False,
            stack_num=kwargs.get("frame_stack", 4),
        )
    else:
        warnings.warn(
            "Recommend using envpool (pip install envpool) "
            "to run Atari games more efficiently."
        )
        env = wrap_deepmind(task, **kwargs)
        train_envs = ShmemVectorEnv(
            [
                lambda:
                wrap_deepmind(task, episode_life=True, clip_rewards=True, **kwargs)
                for _ in range(training_num)
            ]
        )
        test_envs = ShmemVectorEnv(
            [
                lambda:
                wrap_deepmind(task, episode_life=False, clip_rewards=False, **kwargs)
                for _ in range(test_num)
            ]
        )
        env.seed(seed)
        train_envs.seed(seed)
        test_envs.seed(seed)
    return env, train_envs, test_envs
