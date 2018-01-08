import argparse
import subprocess
import sys
import re
import Pyro4
import time
import os
import utils
from time import gmtime, strftime

python_version = sys.version_info

if python_version < (3, 0):
    import cPickle
else:
    import _pickle as cPickle

class Data(object):
    def __init__(self):
        self.boards = []
        self.probs = []
        self.winner = 0

    def reset(self):
        self.__init__()

if __name__ == '__main__':
    """
    Starting two different players which load network weights to evaluate the winning ratio.
    Note that, this function requires the installation of the Pyro4 library.
    """
    # TODO : we should set the network path in a more configurable way.
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--black_weight_path", type=str, default=None)
    parser.add_argument("--white_weight_path", type=str, default=None)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--game", type=str, default="go")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    # black_weight_path = "./checkpoints"
    # white_weight_path = "./checkpoints_origin"
    if args.black_weight_path is not None and (not os.path.exists(args.black_weight_path)):
        raise ValueError("Can't find the network weights for black player.")
    if args.white_weight_path is not None and (not os.path.exists(args.white_weight_path)):
        raise ValueError("Can't find the network weights for white player.")

    # kill the old server
    # kill_old_server = subprocess.Popen(['killall', 'pyro4-ns'])
    # print "kill the old pyro4 name server, the return code is : " + str(kill_old_server.wait())
    # time.sleep(1)

    # start a name server if no name server exists
    if len(os.popen('ps aux | grep pyro4-ns | grep -v grep').readlines()) == 0:
        start_new_server = subprocess.Popen(['pyro4-ns', '&'])
        print("Start Name Sever : " + str(start_new_server.pid))  # + str(start_new_server.wait())
        time.sleep(1)

    # start two different player with different network weights.
    server_list = subprocess.check_output(['pyro4-nsc', 'list'])
    current_time = strftime("%Y%m%d_%H%M%S", gmtime())

    black_role_name = 'black' + current_time
    white_role_name = 'white' + current_time

    black_player = subprocess.Popen(
        ['python', '-u', 'player.py', '--game=' + args.game, '--role=' + black_role_name,
         '--checkpoint_path=' + str(args.black_weight_path), '--debug=' + str(args.debug)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    bp_output = black_player.stdout.readline()
    bp_message = bp_output
    # '' means player.py failed to start, "Start requestLoop" means player.py start successfully
    while bp_output != '' and "Start requestLoop" not in bp_output:
        bp_output = black_player.stdout.readline()
        bp_message += bp_output
    print("============ " + black_role_name + " message ============" + "\n" + bp_message),

    white_player = subprocess.Popen(
        ['python', '-u', 'player.py', '--game=' + args.game, '--role=' + white_role_name,
         '--checkpoint_path=' + str(args.white_weight_path), '--debug=' + str(args.debug)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wp_output = white_player.stdout.readline()
    wp_message = wp_output
    while wp_output != '' and "Start requestLoop" not in wp_output:
        wp_output = white_player.stdout.readline()
        wp_message += wp_output
    print("============ " + white_role_name + " message ============" + "\n" + wp_message),

    server_list = ""
    while (black_role_name not in server_list) or (white_role_name not in server_list):
        if python_version < (3, 0):
            # TODO : @renyong what is the difference between those two options?
            server_list = subprocess.check_output(['pyro4-nsc', 'list'])
        else:
            server_list = subprocess.check_output(['pyro4-nsc', 'list'])
        print("Waiting for the server start...")
        time.sleep(1)
    print(server_list)
    print("Start black player at : " + str(black_player.pid))
    print("Start white player at : " + str(white_player.pid))

    data = Data()
    player = [None] * 2
    player[0] = Pyro4.Proxy("PYRONAME:" + black_role_name)
    player[1] = Pyro4.Proxy("PYRONAME:" + white_role_name)

    role = ["BLACK", "WHITE"]
    color = ['b', 'w']

    pattern = "[A-Z]{1}[0-9]{1}"
    space = re.compile("\s+")
    size = {"go":9, "reversi":8}
    show = ['.', 'X', 'O']

    evaluate_rounds = 100
    game_num = 0
    try:
        while True:
        # while game_num < evaluate_rounds:
            start_time = time.time()
            num = 0
            pass_flag = [False, False]
            print("Start game {}".format(game_num))
            # end the game if both palyer chose to pass, or play too much turns
            while not (pass_flag[0] and pass_flag[1]) and num < size[args.game] ** 2 * 2:
                turn = num % 2
                board = player[turn].run_cmd(str(num) + ' show_board')
                board = eval(board[board.index('['):board.index(']') + 1])
                for i in range(size[args.game]):
                    for j in range(size[args.game]):
                        print show[board[i * size[args.game] + j]] + " ",
                    print "\n",
                data.boards.append(board)
                start_time = time.time()
                move = player[turn].run_cmd(str(num) + ' genmove ' + color[turn])[:-1]
                print("\n" + role[turn] + " : " + str(move)),
                num += 1
                match = re.search(pattern, move)
                if match is not None:
                    # print "match : " + str(match.group())
                    play_or_pass = match.group()
                    pass_flag[turn] = False
                else:
                    # print "no match"
                    play_or_pass = ' PASS'
                    pass_flag[turn] = True
                result = player[1 - turn].run_cmd(str(num) + ' play ' + color[turn] + ' ' + play_or_pass + '\n')
                prob = player[turn].run_cmd(str(num) + ' get_prob')
                prob = space.sub(',', prob[prob.index('['):prob.index(']') + 1])
                prob = prob.replace('[,', '[')
                prob = prob.replace('],', ']')
                prob = eval(prob)
                data.probs.append(prob)
            score = player[0].run_cmd(str(num) + ' get_score')
            print("Finished : {}".format(score.split(" ")[1]))
            if eval(score.split(" ")[1]) > 0:
                data.winner = utils.BLACK
            if eval(score.split(" ")[1]) < 0:
                data.winner = utils.WHITE
            player[0].run_cmd(str(num) + ' clear_board')
            player[1].run_cmd(str(num) + ' clear_board')
            file_list = os.listdir(args.data_path)
            current_time = strftime("%Y%m%d_%H%M%S", gmtime())
            if os.path.exists(args.data_path + current_time + ".pkl"):
                time.sleep(1)
                current_time = strftime("%Y%m%d_%H%M%S", gmtime())
            with open(args.data_path + current_time + ".pkl", "wb") as file:
                picklestring = cPickle.dump(data, file)
            data.reset()
            game_num += 1
    except KeyboardInterrupt:
        pass

    subprocess.call(["kill", "-9", str(black_player.pid)])
    subprocess.call(["kill", "-9", str(white_player.pid)])
    print("Kill all player, finish all game.")
