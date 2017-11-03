import gtp
import go
import utils



def translate_gtp_colors(gtp_color):
    if gtp_color == gtp.BLACK:
        return go.BLACK
    elif gtp_color == gtp.WHITE:
        return go.WHITE
    else:
        return go.EMPTY

class GtpInterface(object):
    def __init__(self):
        self.size = 9
        self.position = None
        self.komi = 6.5
        self.clear()

    def set_size(self, n):
        self.size = n
        go.set_board_size(n)
        self.clear()

    def set_komi(self, komi):
        self.komi = komi
        self.position.komi = komi

    def clear(self):
        self.position = go.Position(komi=self.komi)

    def accomodate_out_of_turn(self, color):
        if not translate_gtp_colors(color) == self.position.to_play:
            self.position.flip_playerturn(mutate=True)

    def make_move(self, color, vertex):
        coords = utils.parse_pygtp_coords(vertex)
        self.accomodate_out_of_turn(color)
        try:
            self.position = self.position.play_move(coords, color=translate_gtp_colors(color))
        except go.IllegalMove:
            return False
        return True

    def get_move(self, color):
        self.accomodate_out_of_turn(color)
        if self.should_resign(self.position):
            return gtp.RESIGN

        if self.should_pass(self.position):
            return gtp.PASS

        move = self.suggest_move(self.position)
        return utils.unparse_pygtp_coords(move)

    def should_resign(self, position):
        if position.caps[0] + 50 < position.caps[1]:
            return gtp.RESIGN

    def should_pass(self, position):
        # Pass if the opponent passes
        return position.n > 100 and position.recent and position.recent[-1].move == None

    def get_score(self):
        return self.position.result()

    def suggest_move(self, position):
        raise NotImplementedError

def make_gtp_instance(strategy_name, read_file):
    n = PolicyNetwork(use_cpu=True)
    n.initialize_variables(read_file)
    if strategy_name == 'random':
        instance = RandomPlayer()
    elif strategy_name == 'policy':
        instance = GreedyPolicyPlayer(n)
    elif strategy_name == 'randompolicy':
        instance = RandomPolicyPlayer(n)
    elif strategy_name == 'mcts':
        instance = MCTSPlayer(n)
    else:
        return None
    gtp_engine = gtp.Engine(instance)
    return gtp_engine