import cv2
import gym
import numpy as np
from gym.spaces.box import Box
from tianshou.data import Batch

SIZE = 84
FRAME = 4


def create_atari_environment(name=None, sticky_actions=True,
                             max_episode_steps=2000):
    game_version = 'v0' if sticky_actions else 'v4'
    name = '{}NoFrameskip-{}'.format(name, game_version)
    env = gym.make(name)
    env = env.env
    env = preprocessing(env, max_episode_steps=max_episode_steps)
    return env


def preprocess_fn(obs=None, act=None, rew=None, done=None,
                  obs_next=None, info=None, policy=None, **kwargs):
    if obs_next is not None:
        obs_next = np.reshape(obs_next, (-1, *obs_next.shape[2:]))
        obs_next = np.moveaxis(obs_next, 0, -1)
        obs_next = cv2.resize(obs_next, (SIZE, SIZE))
        obs_next = np.asanyarray(obs_next, dtype=np.uint8)
        obs_next = np.reshape(obs_next, (-1, FRAME, SIZE, SIZE))
        obs_next = np.moveaxis(obs_next, 1, -1)
    elif obs is not None:
        obs = np.reshape(obs, (-1, *obs.shape[2:]))
        obs = np.moveaxis(obs, 0, -1)
        obs = cv2.resize(obs, (SIZE, SIZE))
        obs = np.asanyarray(obs, dtype=np.uint8)
        obs = np.reshape(obs, (-1, FRAME, SIZE, SIZE))
        obs = np.moveaxis(obs, 1, -1)

    return Batch(obs=obs, act=act, rew=rew, done=done,
                 obs_next=obs_next, info=info)


class preprocessing(object):
    def __init__(self, env, frame_skip=4, terminal_on_life_loss=False,
                 size=84, max_episode_steps=2000):
        self.max_episode_steps = max_episode_steps
        self.env = env
        self.terminal_on_life_loss = terminal_on_life_loss
        self.frame_skip = frame_skip
        self.size = size
        self.count = 0
        obs_dims = self.env.observation_space

        self.screen_buffer = [
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
        ]

        self.game_over = False
        self.lives = 0

    @property
    def observation_space(self):
        return Box(low=0, high=255,
                   shape=(self.size, self.size, self.frame_skip),
                   dtype=np.uint8)

    def action_space(self):
        return self.env.action_space

    def reward_range(self):
        return self.env.reward_range

    def metadata(self):
        return self.env.metadata

    def close(self):
        return self.env.close()

    def reset(self):
        self.count = 0
        self.env.reset()
        self.lives = self.env.ale.lives()
        self._grayscale_obs(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)

        return np.array([self._pool_and_resize()
                         for _ in range(self.frame_skip)])

    def render(self, mode='human'):
        return self.env.render(mode)

    def step(self, action):
        total_reward = 0.
        observation = []
        for t in range(self.frame_skip):
            self.count += 1
            _, reward, terminal, info = self.env.step(action)
            total_reward += reward

            if self.terminal_on_life_loss:
                lives = self.env.ale.lives()
                is_terminal = terminal or lives < self.lives
                self.lives = lives
            else:
                is_terminal = terminal

            if is_terminal:
                break
            elif t >= self.frame_skip - 2:
                t_ = t - (self.frame_skip - 2)
                self._grayscale_obs(self.screen_buffer[t_])

            observation.append(self._pool_and_resize())
        if len(observation) == 0:
            observation = [self._pool_and_resize()
                           for _ in range(self.frame_skip)]
        while len(observation) > 0 and \
                len(observation) < self.frame_skip:
            observation.append(observation[-1])
        terminal = self.count >= self.max_episode_steps
        return np.array(observation), total_reward, \
            (terminal or is_terminal), info

    def _grayscale_obs(self, output):
        self.env.ale.getScreenGrayscale(output)
        return output

    def _pool_and_resize(self):
        if self.frame_skip > 1:
            np.maximum(self.screen_buffer[0], self.screen_buffer[1],
                       out=self.screen_buffer[0])

        return self.screen_buffer[0]
