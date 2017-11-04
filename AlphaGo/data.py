import os

import numpy as np

path = "/raid/tongzheng/AG/self_play_204/"
name = os.listdir(path)
boards = np.zeros([0, 19, 19, 17])
wins = np.zeros([0, 1])
ps = np.zeros([0, 362])

for n in name:
	data = np.load(path + n)
	board = data["boards"]
	win = data["win"]
	p = data["p"]
	# board = np.zeros([0, 19, 19, 17])
	# win = np.zeros([0, 1])
	# p = np.zeros([0, 362])
	# for i in range(data["boards"].shape[3]):
	# 	board = np.concatenate([board, data["boards"][:,:,:,i].reshape(-1, 19, 19, 17)], axis=0)
	# 	win = np.concatenate([win, data["win"][:,i].reshape(-1, 1)], axis=0)
	# 	p = np.concatenate([p, data["p"][:,i].reshape(-1, 362)], axis=0)
	boards = np.concatenate([boards, board], axis=0)
	wins = np.concatenate([wins, win], axis=0)
	ps = np.concatenate([ps, p], axis=0)
	print("Finish " + n)

board_ori = boards
win_ori = wins
p_ori = ps
for i in range(1, 3):
	board = np.rot90(board_ori, i, (1, 2))
	p = np.concatenate(
		[np.rot90(p_ori[:, :-1].reshape(-1, 19, 19), i, (1, 2)).reshape(-1, 361), p_ori[:, -1].reshape(-1, 1)], axis=1)
	boards = np.concatenate([boards, board], axis=0)
	wins = np.concatenate([wins, win_ori], axis=0)
	ps = np.concatenate([ps, p], axis=0)

board = board_ori[:, ::-1]
p = np.concatenate([p_ori[:, :-1].reshape(-1, 19, 19)[:, ::-1].reshape(-1, 361), p_ori[:, -1].reshape(-1, 1)], axis=1)
boards = np.concatenate([boards, board], axis=0)
wins = np.concatenate([wins, win_ori], axis=0)
ps = np.concatenate([ps, p], axis=0)

board = board_ori[:, :, ::-1]
p = np.concatenate([p_ori[:, :-1].reshape(-1, 19, 19)[:, :, ::-1].reshape(-1, 361), p_ori[:, -1].reshape(-1, 1)],
				   axis=1)
boards = np.concatenate([boards, board], axis=0)
wins = np.concatenate([wins, win_ori], axis=0)
ps = np.concatenate([ps, p], axis=0)

board = board_ori[:, ::-1]
p = np.concatenate([np.rot90(p_ori[:, :-1].reshape(-1, 19, 19)[:, ::-1], 1, (1,2)).reshape(-1, 361), p_ori[:, -1].reshape(-1, 1)], axis=1)
boards = np.concatenate([boards, np.rot90(board, 1, (1,2))], axis=0)
wins = np.concatenate([wins, win_ori], axis=0)
ps = np.concatenate([ps, p], axis=0)

board = board_ori[:, :, ::-1]
p = np.concatenate([np.rot90(p_ori[:, :-1].reshape(-1, 19, 19)[:, :, ::-1], 1, (1,2)).reshape(-1, 361), p_ori[:, -1].reshape(-1, 1)],
				   axis=1)
boards = np.concatenate([boards, np.rot90(board, 1, (1,2))], axis=0)
wins = np.concatenate([wins, win_ori], axis=0)
ps = np.concatenate([ps, p], axis=0)

np.savez("data", boards=boards, wins=wins, ps=ps)