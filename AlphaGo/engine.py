# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# $File: engine.py
# $Date: Tue Nov 28 00:0616 2017 +0800
# $Author: renyong15 © <mails.tsinghua.edu.cn>
#

from game import Game
import copy
import numpy as np
import utils


class GTPEngine():
    def __init__(self, **kwargs):
        try:
            self._game = kwargs['game_obj']
            self._game.clear()
        except:
            self._game = None

        try:
            self._name = kwargs['name']
        except:
            self._name = 'gtp engine'
        try:
            self._version = kwargs['version']
        except:
            self._version = 2

        self.disconnect = False

        self.known_commands = [
            field[4:] for field in dir(self) if field.startswith("cmd_")]

    def _vertex_point2string(self, vertex):
        if vertex == utils.PASS:
            return "pass"
        elif vertex == utils.RESIGN:
            return "resign"
        else:
            x, y = vertex
            return "{}{}".format("ABCDEFGHIJKLMNOPQRSTYVWYZ"[x - 1], y)

    def _vertex_string2point(self, s):
        if s is None:
            return False
        elif s.lower() == "pass":
            return utils.PASS
        elif len(s) > 1:
            x = "abcdefghijklmnopqrstuvwxyz".find(s[0].lower()) + 1
            if x == 0:
                return False
            if s[1:].isdigit():
                y = int(s[1:])
            else:
                return False
        else:
            return False
        return (x, y)

    def _parse_color(self, color):
        if color.lower() in ["b", "black"]:
            color = utils.BLACK
        elif color.lower() in ["w", "white"]:
            color = utils.WHITE
        else:
            color = None
        return color

    def _parse_move(self, move_string):
        color, move = move_string.split(" ", 1)
        color = self._parse_color(color)

        point = self._vertex_string2point(move)

        if point and color:
            return color, point
        else:
            return False

    def _parse_res(self, res, id_=None, success=True):
        if success:
            if id_:
                return '={} {}\n\n'.format(id_, res)
            else:
                return '= {}\n\n'.format(res)
        else:
            if id_:
                return '?{} {}\n\n'.format(id_, res)
            else:
                return '? {}\n\n'.format(res)

    def _parse_cmd(self, message):
        try:
            m = (message.strip().split(" ", 1) + [None])[:2]
            if m[0].isdigit():
                id_ = int(m[0])
                cmd, args = (m[1].split(" ", 1) + [None])[:2]
            else:
                id_ = None
                cmd, args = (m[0], m[1])
        except:
            return "invaild command"
        return id_, cmd, args

    def run_cmd(self, message):
        try:
            id_, cmd, args = self._parse_cmd(message)
        except:
            return self._parse_res("invaild message", id_, False)

        if cmd in self.known_commands:
            # dispatch
            # try:
            if True:
                res, flag = getattr(self, "cmd_" + cmd)(args)
                return self._parse_res(res, id_, flag)
                # except Exception as e:
                #    print(e)
                #    return self._parse_res("command excution failed", id_, False)
        else:
            return self._parse_res("unknown command", id_, False)

    def cmd_protocol_version(self, args, **kwargs):
        return 2, True

    def cmd_name(self, args, **kwargs):
        return self._name, True

    def cmd_version(self, args, **kwargs):
        return self._version, True

    def cmd_known_command(self, args, **kwargs):
        return 'true' if (args in self.known_commands) else 'false', True

    def cmd_list_commands(self, args, **kwargs):
        return self.known_commands, True

    def cmd_quit(self, args, **kwargs):
        self.disconnect = True
        return None, True

    def cmd_boardsize(self, board_size, **kwargs):
        if board_size.isdigit():
            self._game.set_size(int(board_size))
            return None, True
        else:
            return 'non digit size', False

    def cmd_clear_board(self, args, **kwargs):
        self._game.clear()
        return None, True

    def cmd_komi(self, komi, **kwargs):
        try:
            self._game.set_komi(float(komi))
            return None, True
        except ValueError:
            raise ValueError("syntax error")

    def cmd_play(self, args, **kwargs):
        move = self._parse_move(args)
        if move:
            color, vertex = move
            res = self._game.play_move(color, vertex)
            if res:
                return None, True
            else:
                return None, False
        return None, True

    def cmd_genmove(self, args, **kwargs):
        color = self._parse_color(args)
        if color:
            move = self._game.think_play_move(color)
            return self._vertex_point2string(move), True
        else:
            return 'unknown player', False

    def cmd_get_score(self, args, **kwargs):
        return self._game.game_engine.executor_get_score(self._game.board), True

    def cmd_show_board(self, args, **kwargs):
        board = copy.deepcopy(self._game.board)
        if isinstance(board, np.ndarray):
            board = board.flatten().tolist()
        return board, True

    def cmd_get_prob(self, args, **kwargs):
        return self._game.prob, True


if __name__ == "main":
    print ("test engine.py")
