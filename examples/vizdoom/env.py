import os

import cv2
import gym
import numpy as np
import vizdoom as vzd


def normal_button_comb():
    actions = []
    m_forward = [[0.0], [1.0]]
    t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    for i in m_forward:
        for j in t_left_right:
            actions.append(i + j)
    return actions


def battle_button_comb():
    actions = []
    m_forward_backward = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    m_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    attack = [[0.0], [1.0]]
    speed = [[0.0], [1.0]]

    for m in attack:
        for n in speed:
            for j in m_left_right:
                for i in m_forward_backward:
                    for k in t_left_right:
                        actions.append(i + j + k + m + n)
    return actions


class Env(gym.Env):

    def __init__(self, cfg_path, frameskip=4, res=(4, 40, 60), save_lmp=False):
        super().__init__()
        self.save_lmp = save_lmp
        self.health_setting = "battle" in cfg_path
        if save_lmp:
            os.makedirs("lmps", exist_ok=True)
        self.res = res
        self.skip = frameskip
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=res, dtype=np.float32
        )
        self.game = vzd.DoomGame()
        self.game.load_config(cfg_path)
        self.game.init()
        if "battle" in cfg_path:
            self.available_actions = battle_button_comb()
        else:
            self.available_actions = normal_button_comb()
        self.action_num = len(self.available_actions)
        self.action_space = gym.spaces.Discrete(self.action_num)
        self.spec = gym.envs.registration.EnvSpec("vizdoom-v0")
        self.count = 0

    def get_obs(self):
        state = self.game.get_state()
        if state is None:
            return
        obs = state.screen_buffer
        self.obs_buffer[:-1] = self.obs_buffer[1:]
        self.obs_buffer[-1] = cv2.resize(obs, (self.res[-1], self.res[-2]))

    def reset(self):
        if self.save_lmp:
            self.game.new_episode(f"lmps/episode_{self.count}.lmp")
        else:
            self.game.new_episode()
        self.count += 1
        self.obs_buffer = np.zeros(self.res, dtype=np.uint8)
        self.get_obs()
        self.health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        self.killcount = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        self.ammo2 = self.game.get_game_variable(vzd.GameVariable.AMMO2)
        return self.obs_buffer

    def step(self, action):
        self.game.make_action(self.available_actions[action], self.skip)
        reward = 0.0
        self.get_obs()
        health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        if self.health_setting:
            reward += health - self.health
        elif health > self.health:  # positive health reward only for d1/d2
            reward += health - self.health
        self.health = health
        killcount = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        reward += 20 * (killcount - self.killcount)
        self.killcount = killcount
        ammo2 = self.game.get_game_variable(vzd.GameVariable.AMMO2)
        # if ammo2 > self.ammo2:
        reward += ammo2 - self.ammo2
        self.ammo2 = ammo2
        done = False
        info = {}
        if self.game.is_player_dead() or self.game.get_state() is None:
            done = True
        elif self.game.is_episode_finished():
            done = True
            info["TimeLimit.truncated"] = True
        return self.obs_buffer, reward, done, info

    def render(self):
        pass

    def close(self):
        self.game.close()


if __name__ == '__main__':
    # env = Env("maps/D1_basic.cfg", 4, (4, 84, 84))
    env = Env("maps/D3_battle.cfg", 4, (4, 84, 84))
    print(env.available_actions)
    action_num = env.action_space.n
    obs = env.reset()
    print(env.spec.reward_threshold)
    print(obs.shape, action_num)
    for _ in range(4000):
        obs, rew, done, info = env.step(0)
        if done:
            env.reset()
    print(obs.shape, rew, done)
    cv2.imwrite("test.png", obs.transpose(1, 2, 0)[..., :3])
