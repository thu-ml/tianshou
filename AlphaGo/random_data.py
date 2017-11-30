import os
import numpy as np
import time

path = "/home/tongzheng/meta-data/"
save_path = "/home/tongzheng/data/"
name = os.listdir(path)
print(len(name))
batch_size = 128
batch_num = 512

block_size = batch_size * batch_num
slots_num = 32


class block(object):
    def __init__(self, block_size, block_id):
        self.boards = []
        self.wins = []
        self.ps = []
        self.block_size = block_size
        self.block_id = block_id

    def concat(self, board, p, win):
        board = board.reshape(-1, 19, 19, 17)
        win = win.reshape(-1, 1)
        p = p.reshape(-1, 362)
        self.boards.append(board)
        self.wins.append(win)
        self.ps.append(p)

    def isfull(self):
        assert len(self.boards) == len(self.wins)
        assert len(self.boards) == len(self.ps)
        return len(self.boards) == self.block_size

    def save_and_reset(self, block_id):
        self.boards = np.concatenate(self.boards, axis=0)
        self.wins = np.concatenate(self.wins, axis=0)
        self.ps = np.concatenate(self.ps, axis=0)
        print ("Block {}, Boards shape {}, Wins Shape {}, Ps Shape {}".format(self.block_id, self.boards.shape[0],
                                                                              self.wins.shape[0], self.ps.shape[0]))
        np.savez(save_path + "block" + str(self.block_id), boards=self.boards, wins=self.wins, ps=self.ps)
        self.boards = []
        self.wins = []
        self.ps = []
        self.block_id = block_id

    def store_num(self):
        assert len(self.boards) == len(self.wins)
        assert len(self.boards) == len(self.ps)
        return len(self.boards)


def concat(block_list, board, win, p):
    global index
    seed = np.random.randint(slots_num)
    block_list[seed].concat(board, win, p)
    if block_list[seed].isfull():
        block_list[seed].save_and_reset(index)
        index = index + 1


block_list = []
for index in range(slots_num):
    block_list.append(block(block_size, index))
index = index + 1
for n in name:
    data = np.load(path + n)
    board = data["boards"]
    win = data["win"]
    p = data["p"]
    print("Start {}".format(n))
    print("Shape {}".format(board.shape[0]))
    start = -time.time()
    for i in range(board.shape[0]):
        board_ori = board[i].reshape(-1, 19, 19, 17)
        win_ori = win[i].reshape(-1, 1)
        p_ori = p[i].reshape(-1, 362)
        concat(block_list, board_ori, p_ori, win_ori)

        for t in range(1, 4):
            board_aug = np.rot90(board_ori, t, (1, 2))
            p_aug = np.concatenate(
                [np.rot90(p_ori[:, :-1].reshape(-1, 19, 19), t, (1, 2)).reshape(-1, 361), p_ori[:, -1].reshape(-1, 1)],
                axis=1)
            concat(block_list, board_aug, p_aug, win_ori)

        board_aug = board_ori[:, ::-1]
        p_aug = np.concatenate(
            [p_ori[:, :-1].reshape(-1, 19, 19)[:, ::-1].reshape(-1, 361), p_ori[:, -1].reshape(-1, 1)],
            axis=1)
        concat(block_list, board_aug, p_aug, win_ori)

        board_aug = board_ori[:, :, ::-1]
        p_aug = np.concatenate(
            [p_ori[:, :-1].reshape(-1, 19, 19)[:, :, ::-1].reshape(-1, 361), p_ori[:, -1].reshape(-1, 1)],
            axis=1)
        concat(block_list, board_aug, p_aug, win_ori)

        board_aug = np.rot90(board_ori[:, ::-1], 1, (1, 2))
        p_aug = np.concatenate(
            [np.rot90(p_ori[:, :-1].reshape(-1, 19, 19)[:, ::-1], 1, (1, 2)).reshape(-1, 361),
             p_ori[:, -1].reshape(-1, 1)],
            axis=1)
        concat(block_list, board_aug, p_aug, win_ori)

        board_aug = np.rot90(board_ori[:, :, ::-1], 1, (1, 2))
        p_aug = np.concatenate(
            [np.rot90(p_ori[:, :-1].reshape(-1, 19, 19)[:, :, ::-1], 1, (1, 2)).reshape(-1, 361),
             p_ori[:, -1].reshape(-1, 1)],
            axis=1)
        concat(block_list, board_aug, p_aug, win_ori)
    print ("Finished {} with time {}".format(n, time.time() + start))
    data_num = 0
    for i in range(slots_num):
        print("Block {} ".format(block_list[i].block_id) + "Size {}".format(block_list[i].store_num()))
        data_num = data_num + block_list[i].store_num()
    print ("Total data {}".format(data_num))

for i in range(slots_num):
    block_list[i].save_and_reset(block_list[i].block_id)
