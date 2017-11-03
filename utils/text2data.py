import numpy as np
import os

def hex2board(hex):
    scale = 16
    num_of_bits = 360
    binary = bin(int(hex[:-2], scale))[2:].zfill(num_of_bits) + hex[-2]
    board = np.zeros([361])
    for i in range(361):
        board[i] = int(binary[i])
    board = board.reshape(1,19,19,1)
    return board

def str2prob(str):
    p = str.split()
    prob = np.zeros([362])
    for i in range(362):
        prob[i] = float(p[i])
    prob = prob.reshape(1,362)
    return prob

dir = "/home/yama/tongzheng/leela-zero/autogtp/new_spr/"
name = os.listdir(dir)
text = []
for n in name:
    if n[-4:]==".txt":
        text.append(n)
print(text)
for t in text:
    num = 0
    boards = np.zeros([0, 19, 19, 17])
    board = np.zeros([1, 19, 19, 0])
    win = np.zeros([0, 1])
    p = np.zeros([0, 362])
    for line in open(dir + t):
        if num % 19 < 16:
            new_board = hex2board(line)
            board = np.concatenate([board, new_board], axis=3)
        if num % 19 == 16:
            if line == '0':
                new_board = np.ones([1, 19 ,19 ,1])
            if line == '1':
                new_board = np.zeros([1, 19, 19, 1])
            board = np.concatenate([board, new_board], axis=3)
            boards = np.concatenate([boards, board], axis=0)
            board = np.zeros([1, 19, 19, 0])
        if num % 19 == 17:
            p = np.concatenate([p,str2prob(line)], axis=0)
        if num % 19 == 18:
            win = np.concatenate([win, np.array(float(line)).reshape(1,1)], axis=0)
        num=num+1
    print "Finished " + t
    np.savez("data/"+t[:-4], boards=boards, win=win, p=p)
