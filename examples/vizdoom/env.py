import os
from collections.abc import Sequence
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from numpy.typing import NDArray

from tianshou.env import ShmemVectorEnv

try:
    import envpool
except ImportError:
    envpool = None


def normal_button_comb() -> list:
    actions = []
    m_forward = [[0.0], [1.0]]
    t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    for i in m_forward:
        for j in t_left_right:
            actions.append(i + j)
    return actions


def battle_button_comb() -> list:
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
    def __init__(
        self,
        cfg_path: str,
        frameskip: int = 4,
        res: Sequence[int] = (4, 40, 60),
        save_lmp: bool = False,
    ) -> None:
        super().__init__()
        self.save_lmp = save_lmp
        self.health_setting = "battle" in cfg_path
        if save_lmp:
            os.makedirs("lmps", exist_ok=True)
        self.res = res
        self.skip = frameskip
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=res, dtype=np.float32)
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

    def get_obs(self) -> None:
        state = self.game.get_state()
        if state is None:
            return
        obs = state.screen_buffer
        self.obs_buffer[:-1] = self.obs_buffer[1:]
        self.obs_buffer[-1] = cv2.resize(obs, (self.res[-1], self.res[-2]))

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.uint8], dict[str, Any]]:
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
        return self.obs_buffer, {"TimeLimit.truncated": False}

    def step(self, action: int) -> tuple[NDArray[np.uint8], float, bool, bool, dict[str, Any]]:
        self.game.make_action(self.available_actions[action], self.skip)
        reward = 0.0
        self.get_obs()
        health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        if self.health_setting or health > self.health:  # positive health reward only for d1/d2
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
        return (
            self.obs_buffer,
            reward,
            done,
            info.get("TimeLimit.truncated", False),
            info,
        )

    def render(self) -> None:
        pass

    def close(self) -> None:
        self.game.close()


def make_vizdoom_env(
    task: str,
    frame_skip: int,
    res: tuple[int],
    save_lmp: bool = False,
    seed: int | None = None,
    num_training_envs: int = 10,
    num_test_envs: int = 10,
) -> tuple[Env, ShmemVectorEnv, ShmemVectorEnv]:
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        num_test_envs = min(cpu_count - 1, num_test_envs)
    if envpool is not None:
        task_id = "".join([i.capitalize() for i in task.split("_")]) + "-v1"
        lmp_save_dir = "lmps/" if save_lmp else ""
        reward_config = {
            "KILLCOUNT": [20.0, -20.0],
            "HEALTH": [1.0, 0.0],
            "AMMO2": [1.0, -1.0],
        }
        if "battle" in task:
            reward_config["HEALTH"] = [1.0, -1.0]
        env = training_envs = envpool.make_gymnasium(
            task_id,
            frame_skip=frame_skip,
            stack_num=res[0],
            seed=seed,
            num_envs=num_training_envs,
            reward_config=reward_config,
            use_combined_action=True,
            max_episode_steps=2625,
            use_inter_area_resize=False,
        )
        test_envs = envpool.make_gymnasium(
            task_id,
            frame_skip=frame_skip,
            stack_num=res[0],
            lmp_save_dir=lmp_save_dir,
            seed=seed,
            num_envs=num_test_envs,
            reward_config=reward_config,
            use_combined_action=True,
            max_episode_steps=2625,
            use_inter_area_resize=False,
        )
    else:
        cfg_path = f"maps/{task}.cfg"
        env = Env(cfg_path, frame_skip, res)
        training_envs = ShmemVectorEnv(
            [lambda: Env(cfg_path, frame_skip, res) for _ in range(num_training_envs)],
        )
        test_envs = ShmemVectorEnv(
            [lambda: Env(cfg_path, frame_skip, res, save_lmp) for _ in range(num_test_envs)],
        )
        training_envs.seed(seed)
        test_envs.seed(seed)
    return env, training_envs, test_envs


if __name__ == "__main__":
    # env = Env("maps/D1_basic.cfg", 4, (4, 84, 84))
    env = Env("maps/D3_battle.cfg", 4, (4, 84, 84))
    print(env.available_actions)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    action_num = env.action_space.n
    obs, _ = env.reset()
    if env.spec:
        print(env.spec.reward_threshold)
    print(obs.shape, action_num)
    for _ in range(4000):
        obs, rew, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            env.reset()
    print(obs.shape, rew, terminated, truncated)
    cv2.imwrite("test.png", obs.transpose(1, 2, 0)[..., :3])
