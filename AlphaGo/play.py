import argparse
import subprocess
import sys
import re
import Pyro4
import time
import os

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
    parser.add_argument("--id", type=int, default=-1)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--game", type=str, default="go")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    # black_weight_path = "./checkpoints"
    # white_weight_path = "./checkpoints_origin"
    if args.black_weight_path is not None and (not os.path.exists(args.black_weight_path)):
        raise ValueError("Can't not find the network weights for black player.")
    if args.white_weight_path is not None and (not os.path.exists(args.white_weight_path)):
        raise ValueError("Can't not find the network weights for white player.")

    # kill the old server
    # kill_old_server = subprocess.Popen(['killall', 'pyro4-ns'])
    # print "kill the old pyro4 name server, the return code is : " + str(kill_old_server.wait())
    # time.sleep(1)

    # start a name server to find the remote object
    # start_new_server = subprocess.Popen(['pyro4-ns', '&'])
    # print "Start Name Sever : " + str(start_new_server.pid)  # + str(start_new_server.wait())
    # time.sleep(1)

    # start a name server if no name server exists
    if len(os.popen('ps aux | grep pyro4-ns | grep -v grep').readlines()) == 0:
        start_new_server = subprocess.Popen(['pyro4-ns', '&'])
        print("Start Name Sever : " + str(start_new_server.pid))  # + str(start_new_server.wait())
        time.sleep(1)

    # start two different player with different network weights.
    server_list = subprocess.check_output(['pyro4-nsc', 'list'])
    index = []
    if server_list is not None:
        server_list = server_list.split("\n")[3:-2]
        for s in server_list:
            id = s.split(" ")[0][5:]
            index.append(eval(id))
        index.sort()
    if args.id == -1:
        if index:
            args.id = index[-1] + 1
        else:
            args.id = 0
    else:
        if args.id in index:
            raise ValueError("Name exists in name server!")

    black_role_name = 'black' + str(args.id)
    white_role_name = 'white' + str(args.id)

    agent_v0 = subprocess.Popen(
        ['python', '-u', 'player.py', '--game=' + args.game, '--role=' + black_role_name,
         '--checkpoint_path=' + str(args.black_weight_path), '--debug=' + str(args.debug)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    agent_v1 = subprocess.Popen(
        ['python', '-u', 'player.py', '--game=' + game_name, '--role=' + white_role_name,
        '--checkpoint_path=' + str(args.white_weight_path), '--debug=' + str(args.debug)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    server_list = ""
    while (black_role_name not in server_list) or (white_role_name not in server_list):
        if python_version < (3, 0):
            server_list = subprocess.check_output(['pyro4-nsc', 'list'])
        else:
            server_list = subprocess.check_output(['pyro4-nsc', 'list'])
        print("Waiting for the server start...")
        time.sleep(1)
    print(server_list)
    print("Start black player at : " + str(agent_v0.pid))
    print("Start white player at : " + str(agent_v1.pid))

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

    evaluate_rounds = 1
    game_num = 0
    try:
        #while True:
        while game_num < evaluate_rounds:
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
                move = player[turn].run_cmd(str(num) + ' genmove ' + color[turn] + '\n')
                print(role[turn] + " : " + str(move)),
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
            score = player[turn].run_cmd(str(num) + ' get_score')
            print("Finished : {}".format(score.split(" ")[1]))
            # TODO: generalize the player
            if eval(score.split(" ")[1]) > 0:
                data.winner = 1
            if eval(score.split(" ")[1]) < 0:
                data.winner = -1
            player[0].run_cmd(str(num) + ' clear_board')
            player[1].run_cmd(str(num) + ' clear_board')
            file_list = os.listdir(args.data_path)
            if not file_list:
                data_num = 0
            else:
                file_list.sort(key=lambda file: os.path.getmtime(args.data_path + file) if not os.path.isdir(
                    args.data_path + file) else 0)
                data_num = eval(file_list[-1][:-4]) + 1
            with open("./data/" + str(data_num) + ".pkl", "wb") as file:
                picklestring = cPickle.dump(data, file)
            data.reset()
            game_num += 1
    except KeyboardInterrupt:
        pass

    ns = Pyro4.locateNS()
    ns.unregister(black_role_name)
    ns.unregister(white_role_name)
    subprocess.call(["kill", "-9", str(agent_v0.pid)])
    subprocess.call(["kill", "-9", str(agent_v1.pid)])
    print("Kill all player, finish all game.")
