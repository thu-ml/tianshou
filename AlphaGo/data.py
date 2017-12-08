import os
import threading
import numpy as np

size = 9
path = "/home/yama/leela-zero/data/npz-files/"
name = os.listdir(path)
print(len(name))
thread_num = 17
batch_num = len(name) // thread_num

def integrate(name, index):
    boards = np.zeros([0, size, size, 17])
    wins = np.zeros([0, 1])
    ps = np.zeros([0, size**2 + 1])
    for n in name:
        data = np.load(path + n)
        board = data["state"]
        win = data["winner"]
        p = data["prob"]
        # board = np.zeros([0, size, size, 17])
        # win = np.zeros([0, 1])
        # p = np.zeros([0, size**2 + 1])
        # for i in range(data["boards"].shape[3]):
        #       board = np.concatenate([board, data["boards"][:,:,:,i].reshape(-1, size, size, 17)], axis=0)
        #       win = np.concatenate([win, data["win"][:,i].reshape(-1, 1)], axis=0)
        # p = np.concatenate([p, data["p"][:,i].reshape(-1, size**2 + 1)], axis=0)
        boards = np.concatenate([boards, board], axis=0)
        wins = np.concatenate([wins, win], axis=0)
        ps = np.concatenate([ps, p], axis=0)
        # print("Finish " + n)
    print ("Integration {} Finished!".format(index))
    board_ori = boards
    win_ori = wins
    p_ori = ps
    for i in range(1, 3):
        board = np.rot90(board_ori, i, (1, 2))
        p = np.concatenate(
            [np.rot90(p_ori[:, :-1].reshape(-1, size, size), i, (1, 2)).reshape(-1, size**2), p_ori[:, -1].reshape(-1, 1)],
            axis=1)
        boards = np.concatenate([boards, board], axis=0)
        wins = np.concatenate([wins, win_ori], axis=0)
        ps = np.concatenate([ps, p], axis=0)

    board = board_ori[:, ::-1]
    p = np.concatenate([p_ori[:, :-1].reshape(-1, size, size)[:, ::-1].reshape(-1, size**2), p_ori[:, -1].reshape(-1, 1)],
                       axis=1)
    boards = np.concatenate([boards, board], axis=0)
    wins = np.concatenate([wins, win_ori], axis=0)
    ps = np.concatenate([ps, p], axis=0)

    board = board_ori[:, :, ::-1]
    p = np.concatenate([p_ori[:, :-1].reshape(-1, size, size)[:, :, ::-1].reshape(-1, size**2), p_ori[:, -1].reshape(-1, 1)],
                       axis=1)
    boards = np.concatenate([boards, board], axis=0)
    wins = np.concatenate([wins, win_ori], axis=0)
    ps = np.concatenate([ps, p], axis=0)

    board = board_ori[:, ::-1]
    p = np.concatenate(
        [np.rot90(p_ori[:, :-1].reshape(-1, size, size)[:, ::-1], 1, (1, 2)).reshape(-1, size**2), p_ori[:, -1].reshape(-1, 1)],
        axis=1)
    boards = np.concatenate([boards, np.rot90(board, 1, (1, 2))], axis=0)
    wins = np.concatenate([wins, win_ori], axis=0)
    ps = np.concatenate([ps, p], axis=0)

    board = board_ori[:, :, ::-1]
    p = np.concatenate(
        [np.rot90(p_ori[:, :-1].reshape(-1, size, size)[:, :, ::-1], 1, (1, 2)).reshape(-1, size**2),
         p_ori[:, -1].reshape(-1, 1)],
        axis=1)
    boards = np.concatenate([boards, np.rot90(board, 1, (1, 2))], axis=0)
    wins = np.concatenate([wins, win_ori], axis=0)
    ps = np.concatenate([ps, p], axis=0)

    np.savez("/home/tongzheng/data/data-" + str(index), state=boards, winner=wins, prob=ps)
    print ("Thread {} has finished.".format(index))
thread_list = list()
for i in range(thread_num):
    thread_list.append(threading.Thread(target=integrate, args=(name[batch_num * i:batch_num * (i + 1)], i,)))
for thread in thread_list:
    thread.start()
for thread in thread_list:
    thread.join()
