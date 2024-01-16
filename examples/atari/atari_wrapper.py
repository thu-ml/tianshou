# Borrow a lot from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
import logging
import warnings
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import Env

from tianshou.highlevel.env import (
    EnvFactoryRegistered,
    EnvMode,
    EnvPoolFactory,
    VectorEnvType,
)
from tianshou.highlevel.trainer import EpochStopCallback, TrainingContext

envpool_is_available = True
try:
    import envpool
except ImportError:
    envpool_is_available = False
    envpool = None
log = logging.getLogger(__name__)


def _parse_reset_result(reset_result):
    contains_info = (
        isinstance(reset_result, tuple)
        and len(reset_result) == 2
        and isinstance(reset_result[1], dict)
    )
    if contains_info:
        return reset_result[0], reset_result[1], contains_info
    return reset_result, {}, contains_info


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
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        _, info, return_info = _parse_reset_result(self.env.reset(**kwargs))
        if hasattr(self.unwrapped.np_random, "integers"):
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            step_result = self.env.step(self.noop_action)
            if len(step_result) == 4:
                obs, rew, done, info = step_result
            else:
                obs, rew, term, trunc, info = step_result
                done = term or trunc
            if done:
                obs, info, _ = _parse_reset_result(self.env.reset())
        if return_info:
            return obs, info
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame (frameskipping) using most recent raw observations (for max pooling across time steps).

    :param gym.Env env: the environment to wrap.
    :param int skip: number of `skip`-th frame.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Step the environment with the given action.

        Repeat action, sum reward, and max over last observations.
        """
        obs_list, total_reward = [], 0.0
        new_step_api = False
        for _ in range(self._skip):
            step_result = self.env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, term, trunc, info = step_result
                done = term or trunc
                new_step_api = True
            obs_list.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(obs_list[-2:], axis=0)
        if new_step_api:
            return max_frame, total_reward, term, trunc, info

        return max_frame, total_reward, done, info


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over.

    It helps the value estimation.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
        self._return_info = False

    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            new_step_api = False
        else:
            obs, reward, term, trunc, info = step_result
            done = term or trunc
            new_step_api = True

        self.was_real_done = done
        # check current lives, make loss of life terminal, then update lives to
        # handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few
            # frames, so its important to keep lives > 0, so that we only reset
            # once the environment is actually done.
            done = True
            term = True
        self.lives = lives
        if new_step_api:
            return obs, reward, term, trunc, info
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Calls the Gym environment reset, only when lives are exhausted.

        This way all states are still reachable even though lives are episodic, and
        the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info, self._return_info = _parse_reset_result(self.env.reset(**kwargs))
        else:
            # no-op step to advance from terminal/lost life state
            step_result = self.env.step(0)
            obs, info = step_result[0], step_result[-1]
        self.lives = self.env.unwrapped.ale.lives()
        if self._return_info:
            return obs, info
        return obs


class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing.

    Related discussion: https://github.com/openai/baselines/issues/240.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        _, _, return_info = _parse_reset_result(self.env.reset(**kwargs))
        obs = self.env.step(1)[0]
        return (obs, {}) if return_info else obs


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
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame):
        """Returns the current observation from a frame."""
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
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
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
        shape = (n_frames, *env.observation_space.shape)
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=shape,
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info, return_info = _parse_reset_result(self.env.reset(**kwargs))
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return (self._get_ob(), info) if return_info else self._get_ob()

    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            new_step_api = False
        else:
            obs, reward, term, trunc, info = step_result
            new_step_api = True
        self.frames.append(obs)
        if new_step_api:
            return self._get_ob(), reward, term, trunc, info
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        # the original wrapper use `LazyFrames` but since we use np buffer,
        # it has no effect
        return np.stack(self.frames, axis=0)


def wrap_deepmind(
    env: Env,
    episode_life=True,
    clip_rewards=True,
    frame_stack=4,
    scale=False,
    warp_frame=True,
):
    """Configure environment for DeepMind-style Atari.

    The observation is channel-first: (c, h, w) instead of (h, w, c).

    :param env: the Atari environment to wrap.
    :param bool episode_life: wrap the episode life wrapper.
    :param bool clip_rewards: wrap the reward clipping wrapper.
    :param int frame_stack: wrap the frame stacking wrapper.
    :param bool scale: wrap the scaling observation wrapper.
    :param bool warp_frame: wrap the grayscale + resize observation wrapper.
    :return: the wrapped atari environment.
    """
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
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


def make_atari_env(
    task,
    seed,
    training_num,
    test_num,
    scale: int | bool = False,
    frame_stack: int = 4,
):
    """Wrapper function for Atari env.

    If EnvPool is installed, it will automatically switch to EnvPool's Atari env.

    :return: a tuple of (single env, training envs, test envs).
    """
    env_factory = AtariEnvFactory(task, seed, frame_stack, scale=bool(scale))
    envs = env_factory.create_envs(training_num, test_num)
    return envs.env, envs.train_envs, envs.test_envs


class AtariEnvFactory(EnvFactoryRegistered):
    def __init__(
        self,
        task: str,
        seed: int,
        frame_stack: int,
        scale: bool = False,
        use_envpool_if_available: bool = True,
    ):
        assert "NoFrameskip" in task
        self.frame_stack = frame_stack
        self.scale = scale
        envpool_factory = None
        if use_envpool_if_available:
            if envpool_is_available:
                envpool_factory = self.EnvPoolFactory(self)
                log.info("Using envpool, because it available")
            else:
                log.info("Not using envpool, because it is not available")
        super().__init__(
            task=task,
            seed=seed,
            venv_type=VectorEnvType.SUBPROC_SHARED_MEM,
            envpool_factory=envpool_factory,
        )

    def create_env(self, mode: EnvMode) -> Env:
        env = super().create_env(mode)
        is_train = mode == EnvMode.TRAIN
        return wrap_deepmind(
            env,
            episode_life=is_train,
            clip_rewards=is_train,
            frame_stack=self.frame_stack,
            scale=self.scale,
        )

    class EnvPoolFactory(EnvPoolFactory):
        """Atari-specific envpool creation.
        Since envpool internally handles the functions that are implemented through the wrappers in `wrap_deepmind`,
        it sets the creation keyword arguments accordingly.
        """

        def __init__(self, parent: "AtariEnvFactory"):
            self.parent = parent
            if self.parent.scale:
                warnings.warn(
                    "EnvPool does not include ScaledFloatFrame wrapper, "
                    "please compensate by scaling inside your network's forward function (e.g. `x = x / 255.0` for Atari)",
                )

        def _transform_task(self, task: str) -> str:
            task = super()._transform_task(task)
            # TODO: Maybe warn user, explain why this is needed
            return task.replace("NoFrameskip-v4", "-v5")

        def _transform_kwargs(self, kwargs: dict, mode: EnvMode) -> dict:
            kwargs = super()._transform_kwargs(kwargs, mode)
            is_train = mode == EnvMode.TRAIN
            kwargs["reward_clip"] = is_train
            kwargs["episodic_life"] = is_train
            kwargs["stack_num"] = self.parent.frame_stack
            return kwargs


class AtariEpochStopCallback(EpochStopCallback):
    def __init__(self, task: str):
        self.task = task

    def should_stop(self, mean_rewards: float, context: TrainingContext) -> bool:
        env = context.envs.env
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        if "Pong" in self.task:
            return mean_rewards >= 20
        return False
