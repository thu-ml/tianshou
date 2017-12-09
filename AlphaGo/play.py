import subprocess
import sys
import re
import time
pattern = "[A-Z]{1}[0-9]{1}"
size = 9
agent_v1 = subprocess.Popen(['python', '-u', 'test.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
agent_v0 = subprocess.Popen(['python', '-u', 'test.py', '--checkpoint_path=./checkpoints_origin/'], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


num = 0
game_num = 0
black_pass = False
white_pass = False


while game_num < 10:
    print("Start game {}".format(game_num))
    while not (black_pass and white_pass) and num < size ** 2 * 2:
        print(num)
        if num % 2 == 0:
            print('BLACK TURN')
            agent_v1.stdin.write(str(num) + ' genmove b\n')
            agent_v1.stdin.flush()
            result = agent_v1.stdout.readline()
            sys.stdout.write(result)
            sys.stdout.flush()
            num += 1
            match = re.search(pattern, result)
            print("COPY BLACK")
            if match is not None:
                agent_v0.stdin.write(str(num) + ' play b ' + match.group() + '\n')
                agent_v0.stdin.flush()
                result = agent_v0.stdout.readline()
                sys.stdout.flush()
            else:
                agent_v0.stdin.write(str(num) + ' play b PASS\n')
                agent_v0.stdin.flush()
                result = agent_v0.stdout.readline()
                sys.stdout.flush()
            if re.search("pass", result) is not None:
                black_pass = True
            else:
                black_pass = False
        else:
            print('WHITE TURN')
            agent_v0.stdin.write(str(num) + ' genmove w\n')
            agent_v0.stdin.flush()
            result = agent_v0.stdout.readline()
            sys.stdout.write(result)
            sys.stdout.flush()
            num += 1
            match = re.search(pattern, result)
            print("COPY WHITE")
            if match is not None:
                agent_v1.stdin.write(str(num) + ' play w ' + match.group() + '\n')
                agent_v1.stdin.flush()
                result = agent_v1.stdout.readline()
                sys.stdout.flush()
            else:
                agent_v1.stdin.write(str(num) + ' play w PASS\n')
                agent_v1.stdin.flush()
                result = agent_v1.stdout.readline()
                sys.stdout.flush()
            if re.search("pass", result) is not None:
                black_pass = True
            else:
                black_pass = False

    print("Finished")
    print("\n")

    agent_v1.stdin.write('clear_board\n')
    agent_v1.stdin.flush()
    result = agent_v1.stdout.readline()
    sys.stdout.flush()

    agent_v0.stdin.write('clear_board\n')
    agent_v0.stdin.flush()
    result = agent_v0.stdout.readline()
    sys.stdout.flush()

    agent_v1.stdin.write('get_score\n')
    agent_v1.stdin.flush()
    result = agent_v1.stdout.readline()
    sys.stdout.write(result)
    sys.stdout.flush()
    game_num += 1
