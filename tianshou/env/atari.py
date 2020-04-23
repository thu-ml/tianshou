import cv2
import gym
import numpy as np
from gym.spaces.box import Box


def create_atari_environment(name=None, sticky_actions=True,
                             max_episode_steps=2000):
    game_version = 'v0' if sticky_actions else 'v4'
    name = '{}NoFrameskip-{}'.format(name, game_version)
    env = gym.make(name)
    env = env.env
    env = preprocessing(env, max_episode_steps=max_episode_steps)
    return env


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
        return Box(low=0, high=255, shape=(self.size, self.size, 4),
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

        return np.stack([
            self._pool_and_resize() for _ in range(self.frame_skip)], axis=-1)

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
        while len(observation) > 0 and len(observation) < self.frame_skip:
            observation.append(observation[-1])
        if len(observation) > 0:
            observation = np.stack(observation, axis=-1)
        else:
            observation = np.stack([
                self._pool_and_resize() for _ in range(self.frame_skip)],
                axis=-1)
        if self.count >= self.max_episode_steps:
            terminal = True
        else:
            terminal = False
        return observation, total_reward, (terminal or is_terminal), info

    def _grayscale_obs(self, output):
        self.env.ale.getScreenGrayscale(output)
        return output

    def _pool_and_resize(self):
        if self.frame_skip > 1:
            np.maximum(self.screen_buffer[0], self.screen_buffer[1],
                       out=self.screen_buffer[0])

        transformed_image = cv2.resize(self.screen_buffer[0],
                                       (self.size, self.size),
                                       interpolation=cv2.INTER_AREA)
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        # return np.expand_dims(int_image, axis=2)
        return int_image
