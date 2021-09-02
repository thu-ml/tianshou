# import cv2
import sys
import time

import tqdm
import vizdoom as vzd


def main(cfg_path="maps/D3_battle.cfg", lmp_path="test.lmp"):
    game = vzd.DoomGame()
    game.load_config(cfg_path)
    game.set_screen_format(vzd.ScreenFormat.CRCGCB)
    game.set_screen_resolution(vzd.ScreenResolution.RES_1024X576)
    game.set_window_visible(True)
    game.set_render_hud(True)
    game.init()
    game.replay_episode(lmp_path)

    killcount = 0
    with tqdm.trange(10500) as tq:
        while not game.is_episode_finished():
            game.advance_action()
            state = game.get_state()
            if state is None:
                break
            killcount = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            time.sleep(1 / 35)
            # cv2.imwrite(f"imgs/{tq.n}.png",
            #             state.screen_buffer.transpose(1, 2, 0)[..., ::-1])
            tq.update(1)
    game.close()
    print("killcount:", killcount)


if __name__ == '__main__':
    main(*sys.argv[-2:])
