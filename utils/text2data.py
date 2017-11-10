import numpy as np
import os
import time

def hex2board(hex):
    scale = 16
    num_of_bits = 360
    binary = bin(int(hex[:-2], scale))[2:].zfill(num_of_bits) + hex[-2]
    board = np.zeros([361], dtype='int8')
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
    if np.sum(np.isnan(prob))==0:
        return prob, True
    else:
        return 0, False

dir = "/home/yama/leela-zero/data/sgf-txt-files/"
name = os.listdir(dir)
text = []
for n in name:
    if n[-4:]==".txt":
        text.append(n)
print(text)
total_start = -time.time()
for t in text:
    start = -time.time()
    num = 0
    boards = []
    board = []
    win = []
    p = []
    flag = False
    for line in open(dir + t):
        if num % 19 == 0:
            flag = False
        if num % 19 < 16:
            new_board = hex2board(line)
            board.append(new_board)
        if num % 19 == 16:
            if int(line) == 0:
                new_board = np.ones([1, 19 ,19 ,1], dtype='int8')
            if int(line) == 1:
                new_board = np.zeros([1, 19, 19, 1], dtype='int8')
            board.append(new_board)
            board = np.concatenate(board, axis=3)
            boards.append(board)
            board = []
        if num % 19 == 17:
            if str2prob(line)[1] == True:
                p.append(str2prob(line)[0])
            else:
                flag = True
                boards = boards[:-1]
        if num % 19 == 18:
            if flag == False:
                win.append(np.array(int(line), dtype='int8').reshape(1,1))
        num=num+1
    boards = np.concatenate(boards, axis=0)
    win = np.concatenate(win, axis=0)
    p = np.concatenate(p, axis=0)
    print("Boards Shape: {}, Wins Shape: {}, Probs Shape : {}".format(boards.shape[0], win.shape[0], p.shape[0]))
    print("Finished {} Time {}".format(t, time.time()+start))
    np.savez("/home/tongzheng/meta-data/"+t[:-4], boards=boards, win=win, p=p)
    del boards, board, win, p
print("All finished! Time {}".format(time.time()+total_start))