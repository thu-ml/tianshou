import numpy as np
import os

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
for t in text:
    num = 0
    boards = np.zeros([0, 19, 19, 17], dtype='int8')
    board = np.zeros([1, 19, 19, 0], dtype='int8')
    win = np.zeros([0, 1], dtype='int8')
    p = np.zeros([0, 362])
    flag = False
    for line in open(dir + t):
	if num % 19 == 0:
	    flag = False
        if num % 19 < 16:
            new_board = hex2board(line)
            board = np.concatenate([board, new_board], axis=3)
        if num % 19 == 16:
            if int(line) == 0:
                new_board = np.ones([1, 19 ,19 ,1], dtype='int8')
            if int(line) == 1:
                new_board = np.zeros([1, 19, 19, 1], dtype='int8')
            board = np.concatenate([board, new_board], axis=3)
            boards = np.concatenate([boards, board], axis=0)
            board = np.zeros([1, 19, 19, 0], dtype='int8')
        if num % 19 == 17:
	    if str2prob(line)[1] == True:
            	p = np.concatenate([p,str2prob(line)[0]], axis=0)
	    else:
		flag = True
		boards = boards[:-1]
        if num % 19 == 18:
	    if flag == False:
            	win = np.concatenate([win, np.array(float(line), dtype='int8').reshape(1,1)], axis=0)
        num=num+1
    print("Boards Shape: {}, Wins Shape: {}, Probs Shape : {}".format(boards.shape[0], win.shape[0], p.shape[0]))
    print "Finished " + t
    np.savez("/home/tongzheng/meta-data/"+t[:-4], boards=boards, win=win, p=p)
