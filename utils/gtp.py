import re


def pre_engine(s):
    s = re.sub("[^\t\n -~]", "", s)
    s = s.split("#")[0]
    s = s.replace("\t", " ")
    return s


def pre_controller(s):
    s = re.sub("[^\t\n -~]", "", s)
    s = s.replace("\t", " ")
    return s


def gtp_boolean(b):
    return "true" if b else "false"


def gtp_list(l):
    return "\n".join(l)


def gtp_color(color):
    # an arbitrary choice amongst a number of possibilities
    return {BLACK: "B", WHITE: "W"}[color]


def gtp_vertex(vertex):
    if vertex == PASS:
        return "pass"
    elif vertex == RESIGN:
        return "resign"
    else:
        x, y = vertex
        return "{}{}".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[x - 1], y)


def gtp_move(color, vertex):
    return " ".join([gtp_color(color), gtp_vertex(vertex)])


def parse_message(message):
    message = pre_engine(message).strip()
    first, rest = (message.split(" ", 1) + [None])[:2]
    if first.isdigit():
        message_id = int(first)
        if rest is not None:
            command, arguments = (rest.split(" ", 1) + [None])[:2]
        else:
            command, arguments = None, None
    else:
        message_id = None
        command, arguments = first, rest

    return message_id, command, arguments


WHITE = -1
BLACK = +1
EMPTY = 0

PASS = (0, 0)
RESIGN = "resign"


def parse_color(color):
    if color.lower() in ["b", "black"]:
        return BLACK
    elif color.lower() in ["w", "white"]:
        return WHITE
    else:
        return False


def parse_vertex(vertex_string):
    if vertex_string is None:
        return False
    elif vertex_string.lower() == "pass":
        return PASS
    elif len(vertex_string) > 1:
        x = "abcdefghjklmnopqrstuvwxyz".find(vertex_string[0].lower()) + 1
        if x == 0:
            return False
        if vertex_string[1:].isdigit():
            y = int(vertex_string[1:])
        else:
            return False
    else:
        return False
    return (x, y)


def parse_move(move_string):
    color_string, vertex_string = (move_string.split(" ") + [None])[:2]
    color = parse_color(color_string)
    if color is False:
        return False
    vertex = parse_vertex(vertex_string)
    if vertex is False:
        return False

    return color, vertex


MIN_BOARD_SIZE = 7
MAX_BOARD_SIZE = 19


def format_success(message_id, response=None):
    if response is None:
        response = ""
    else:
        response = " {}".format(response)
    if message_id:
        return "={}{}\n\n".format(message_id, response)
    else:
        return "={}\n\n".format(response)


def format_error(message_id, response):
    if response:
        response = " {}".format(response)
    if message_id:
        return "?{}{}\n\n".format(message_id, response)
    else:
        return "?{}\n\n".format(response)


class Engine(object):
    def __init__(self, game_obj, name="gtp (python library)", version="0.2"):

        self.size = 19
        self.komi = 6.5

        self._game = game_obj
        self._game.clear()

        self._name = name
        self._version = version

        self.disconnect = False

        self.known_commands = [
            field[4:] for field in dir(self) if field.startswith("cmd_")]

    def send(self, message):
        message_id, command, arguments = parse_message(message)
        if command in self.known_commands:
            try:
                return format_success(
                    message_id, getattr(self, "cmd_" + command)(arguments))
            except ValueError as exception:
                return format_error(message_id, exception.args[0])
        else:
            return format_error(message_id, "unknown command")

    def vertex_in_range(self, vertex):
        if vertex == PASS:
            return True
        if 1 <= vertex[0] <= self.size and 1 <= vertex[1] <= self.size:
            return True
        else:
            return False

    # commands

    def cmd_protocol_version(self, arguments):
        return 2

    def cmd_name(self, arguments):
        return self._name

    def cmd_version(self, arguments):
        return self._version

    def cmd_known_command(self, arguments):
        return gtp_boolean(arguments in self.known_commands)

    def cmd_list_commands(self, arguments):
        return gtp_list(self.known_commands)

    def cmd_quit(self, arguments):
        self.disconnect = True

    def cmd_boardsize(self, arguments):
        if arguments.isdigit():
            size = int(arguments)
            if MIN_BOARD_SIZE <= size <= MAX_BOARD_SIZE:
                self.size = size
                self._game.set_size(size)
            else:
                raise ValueError("unacceptable size")
        else:
            raise ValueError("non digit size")

    def cmd_clear_board(self, arguments):
        self._game.clear()

    def cmd_komi(self, arguments):
        try:
            komi = float(arguments)
            self.komi = komi
            self._game.set_komi(komi)
        except ValueError:
            raise ValueError("syntax error")

    def cmd_play(self, arguments):
        move = parse_move(arguments)
        if move:
            color, vertex = move
            if self.vertex_in_range(vertex):
                if self._game.make_move(color, vertex):
                    return
        raise ValueError("illegal move")

    def cmd_genmove(self, arguments):
        c = parse_color(arguments)
        if c:
            move = self._game.get_move(c)
            self._game.make_move(c, move)
            return gtp_vertex(move)
        else:
            raise ValueError("unknown player: {}".format(arguments))


class MinimalGame(object):
    def __init__(self, size=19, komi=6.5):
        self.size = size
        self.komi = 6.5
        self.board = [EMPTY] * (self.size * self.size)

    def _flatten(self, vertex):
        (x, y) = vertex
        return (x - 1) * self.size + (y - 1)

    def clear(self):
        self.board = [EMPTY] * (self.size * self.size)

    def make_move(self, color, vertex):
        # no legality check other than the space being empty..
        # no side-effects beyond placing the stone..
        if vertex == PASS:
            return True  # noop
        idx = self._flatten(vertex)
        if self.board[idx] == EMPTY:
            self.board[idx] = color
            return True
        else:
            return False

    def set_size(self, n):
        self.size = n
        self.clear()

    def set_komi(self, k):
        self.komi = k

    def get_move(self, color):
        # pass every time. At least it's legal
        return (0, 0)
