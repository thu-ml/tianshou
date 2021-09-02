#!/usr/bin/env python3

#####################################################################
# This script presents SPECTATOR mode. In SPECTATOR mode you play and
# your agent can learn from it.
# Configuration is loaded from "../../scenarios/<SCENARIO_NAME>.cfg" file.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from __future__ import print_function

from argparse import ArgumentParser
from time import sleep

import vizdoom as vzd

# import cv2

if __name__ == "__main__":
    parser = ArgumentParser("ViZDoom example showing how to use SPECTATOR mode.")
    parser.add_argument('-c', type=str, dest="config", default="D3_battle.cfg")
    parser.add_argument('-w', type=str, dest="wad_file", default="D3_battle.wad")
    args = parser.parse_args()
    game = vzd.DoomGame()

    # Choose scenario config file you wish to watch.
    # Don't load two configs cause the second will overrite the first one.
    # Multiple config files are ok but combining these ones doesn't make much sense.

    game.load_config(args.config)
    game.set_doom_scenario_path(args.wad_file)
    # Enables freelook in engine
    game.add_game_args("+freelook 1")

    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Enables spectator mode, so you can play.
    # Sounds strange but it is the agent who is supposed to watch not you.
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)

    game.init()

    episodes = 1

    for i in range(episodes):
        print("Episode #" + str(i + 1))

        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            print(state.screen_buffer.dtype, state.screen_buffer.shape)
            # cv2.imwrite(f'imgs/{state.number}.png', state.screen_buffer)

            # game.make_action([0, 0, 0])
            game.advance_action()
            last_action = game.get_last_action()
            reward = game.get_last_reward()

            print("State #" + str(state.number))
            print("Game variables: ", state.game_variables)
            print("Action:", last_action)
            print("Reward:", reward)
            print("=====================")

        print("Episode finished!")
        print("Total reward:", game.get_total_reward())
        print("************************")
        sleep(2.0)

    game.close()
