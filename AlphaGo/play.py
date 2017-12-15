import subprocess
import sys
import re
import Pyro4
import time

#start a name server to find the remote object
kill_old_server = subprocess.Popen(['killall', 'pyro4-ns'])
print "kill old server, the return code is : " + str(kill_old_server.wait())
time.sleep(1)
start_new_server = subprocess.Popen(['pyro4-ns', '&'])
print "Start Name Sever : " + str(start_new_server.pid)# + str(start_new_server.wait())
time.sleep(1)
agent_v0 = subprocess.Popen(['python', '-u', 'player.py', '--role=black'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print "Start Player 0 at : " + str(agent_v0.pid)
agent_v1 = subprocess.Popen(['python', '-u', 'player.py', '--role=white', '--checkpoint_path=./checkpoints_origin/'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print "Start Player 1 at : " + str(agent_v1.pid)
time.sleep(5)

player = [None] * 2
player[0] = Pyro4.Proxy("PYRONAME:black")
player[1] = Pyro4.Proxy("PYRONAME:white")

role = ["BLACK", "WHITE"]
color = ['b', 'w']

pattern = "[A-Z]{1}[0-9]{1}"
size = 9
show = ['.', 'X', 'O']

game_num = 0
while game_num < 1:
    num = 0
    pass_flag = [False, False]
    print("Start game {}".format(game_num))
    # end the game if both palyer chose to pass, or play too much turns
    while not (pass_flag[0] and pass_flag[1]) and num < size ** 2 * 2:
        turn = num % 2
        move = player[turn].run_cmd(str(num) + ' genmove ' + color[turn] + '\n')
        print role[turn] + " : " + str(move),
        num += 1
        match = re.search(pattern, move)
        if match is not None:
            #print "match : " + str(match.group())
            play_or_pass = match.group()
            pass_flag[turn] = False
        else:
            #print "no match"
            play_or_pass = ' PASS'
            pass_flag[turn] = True
        result = player[1 - turn].run_cmd(str(num) + ' play ' + color[turn] + ' ' + play_or_pass + '\n')
        board = player[turn].run_cmd(str(num) + ' show_board')
        board = eval(board[board.index('['):board.index(']') + 1])
        for i in range(size):
            for j in range(size):
                print show[board[i * size + j]] + " ",
            print "\n",

    score = player[turn].run_cmd(str(num) + ' get_score')
    print "Finished : ", score.split(" ")[1]
    player[0].run_cmd(str(num) + ' clear_board')
    player[1].run_cmd(str(num) + ' clear_board')
    game_num += 1

subprocess.call(["kill", "-9", str(agent_v0.pid)])
subprocess.call(["kill", "-9", str(agent_v1.pid)])
print "Kill all player, finish all game."
