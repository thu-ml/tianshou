import os
import time
import sys

import numpy as np
import time
import tensorflow as tf
import tensorflow.contrib.layers as layers

import multi_gpu
import time
import copy

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def residual_block(input, is_training):
    normalizer_params = {'is_training': is_training,
                         'updates_collections': tf.GraphKeys.UPDATE_OPS}
    h = layers.conv2d(input, 256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                      normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params,
                      weights_regularizer=layers.l2_regularizer(1e-4))
    h = layers.conv2d(h, 256, kernel_size=3, stride=1, activation_fn=tf.identity,
                      normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params,
                      weights_regularizer=layers.l2_regularizer(1e-4))
    h = h + input
    return tf.nn.relu(h)


def policy_heads(input, is_training):
    normalizer_params = {'is_training': is_training,
                         'updates_collections': tf.GraphKeys.UPDATE_OPS}
    h = layers.conv2d(input, 2, kernel_size=1, stride=1, activation_fn=tf.nn.relu,
                      normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params,
                      weights_regularizer=layers.l2_regularizer(1e-4))
    h = layers.flatten(h)
    h = layers.fully_connected(h, 82, activation_fn=tf.identity, weights_regularizer=layers.l2_regularizer(1e-4))
    return h


def value_heads(input, is_training):
    normalizer_params = {'is_training': is_training,
                         'updates_collections': tf.GraphKeys.UPDATE_OPS}
    h = layers.conv2d(input, 2, kernel_size=1, stride=1, activation_fn=tf.nn.relu,
                      normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params,
                      weights_regularizer=layers.l2_regularizer(1e-4))
    h = layers.flatten(h)
    h = layers.fully_connected(h, 256, activation_fn=tf.nn.relu, weights_regularizer=layers.l2_regularizer(1e-4))
    h = layers.fully_connected(h, 1, activation_fn=tf.nn.tanh, weights_regularizer=layers.l2_regularizer(1e-4))
    return h


class Network(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 9, 9, 17])
        self.is_training = tf.placeholder(tf.bool, shape=[])
        self.z = tf.placeholder(tf.float32, shape=[None, 1])
        self.pi = tf.placeholder(tf.float32, shape=[None, 82])
        self.build_network()

    def build_network(self):
        h = layers.conv2d(self.x, 256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                          normalizer_fn=layers.batch_norm,
                          normalizer_params={'is_training': self.is_training,
                                             'updates_collections': tf.GraphKeys.UPDATE_OPS},
                          weights_regularizer=layers.l2_regularizer(1e-4))
        for i in range(4):
            h = residual_block(h, self.is_training)
        self.v = value_heads(h, self.is_training)
        self.p = policy_heads(h, self.is_training)
        # loss = tf.reduce_mean(tf.square(z-v)) - tf.multiply(pi, tf.log(tf.clip_by_value(tf.nn.softmax(p), 1e-8, tf.reduce_max(tf.nn.softmax(p)))))
        self.value_loss = tf.reduce_mean(tf.square(self.z - self.v))
        self.policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.pi, logits=self.p))

        self.reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.total_loss = self.value_loss + self.policy_loss + self.reg
        # train_op = tf.train.MomentumOptimizer(1e-4, momentum=0.9, use_nesterov=True).minimize(total_loss)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = tf.train.RMSPropOptimizer(1e-4).minimize(self.total_loss)
        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(max_to_keep=10, var_list=self.var_list)
        self.sess = multi_gpu.create_session()

    def train(self):
        data_path = "./training_data/"
        data_name = os.listdir(data_path)
        epochs = 100
        batch_size = 128

        result_path = "./checkpoints_origin/"
        with multi_gpu.create_session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt_file = tf.train.latest_checkpoint(result_path)
            if ckpt_file is not None:
                print('Restoring model from {}...'.format(ckpt_file))
                self.saver.restore(sess, ckpt_file)
            for epoch in range(epochs):
                for name in data_name:
                    data = np.load(data_path + name)
                    boards = data["boards"]
                    wins = data["wins"]
                    ps = data["ps"]
                    print (boards.shape)
                    print (wins.shape)
                    print (ps.shape)
                    batch_num = boards.shape[0] // batch_size
                    index = np.arange(boards.shape[0])
                    np.random.shuffle(index)
                    value_losses = []
                    policy_losses = []
                    regs = []
                    time_train = -time.time()
                    for iter in range(batch_num):
                        lv, lp, r, value, prob, _ = sess.run(
                            [self.value_loss, self.policy_loss, self.reg, self.v, tf.nn.softmax(self.p), self.train_op],
                            feed_dict={self.x: boards[
                                index[iter * batch_size:(iter + 1) * batch_size]],
                                       self.z: wins[index[
                                                    iter * batch_size:(iter + 1) * batch_size]],
                                       self.pi: ps[index[
                                                   iter * batch_size:(iter + 1) * batch_size]],
                                       self.is_training: True})
                        value_losses.append(lv)
                        policy_losses.append(lp)
                        regs.append(r)
                        if iter % 1 == 0:
                            print(
                                "Epoch: {}, Part {}, Iteration: {}, Time: {}, Value Loss: {}, Policy Loss: {}, Reg: {}".format(
                                    epoch, name, iter, time.time() + time_train, np.mean(np.array(value_losses)),
                                    np.mean(np.array(policy_losses)), np.mean(np.array(regs))))
                            time_train = -time.time()
                            value_losses = []
                            policy_losses = []
                            regs = []
                        if iter % 20 == 0:
                            save_path = "Epoch{}.Part{}.Iteration{}.ckpt".format(epoch, name, iter)
                            self.saver.save(sess, result_path + save_path)
                    del data, boards, wins, ps


                    # def forward(call_number):
                    #     # checkpoint_path = "/home/yama/rl/tianshou/AlphaGo/checkpoints"
                    #     checkpoint_path = "/home/jialian/stuGo/tianshou/stuGo/checkpoints/"
                    #     board_file = np.genfromtxt("/home/jialian/stuGo/tianshou/leela-zero/src/mcts_nn_files/board_" + call_number,
                    #                                dtype='str');
                    #     human_board = np.zeros((17, 19, 19))
                    #
                    #     # TODO : is it ok to ignore the last channel?
                    #     for i in range(17):
                    #         human_board[i] = np.array(list(board_file[i])).reshape(19, 19)
                    #     # print("============================")
                    #     # print("human board sum : " + str(np.sum(human_board[-1])))
                    #     # print("============================")
                    #     # print(human_board)
                    #     # print("============================")
                    #     # rint(human_board)
                    #     feed_board = human_board.transpose(1, 2, 0).reshape(1, 19, 19, 17)
                    #     # print(feed_board[:,:,:,-1])
                    #     # print(feed_board.shape)
                    #
                    #     # npz_board = np.load("/home/yama/rl/tianshou/AlphaGo/data/7f83928932f64a79bc1efdea268698ae.npz")
                    #     # print(npz_board["boards"].shape)
                    #     # feed_board = npz_board["boards"][10].reshape(-1, 19, 19, 17)
                    #     ##print(feed_board)
                    #     # show_board = feed_board[0].transpose(2, 0, 1)
                    #     # print("board shape : ", show_board.shape)
                    #     # print(show_board)
                    #
                    #     itflag = False
                    #     with multi_gpu.create_session() as sess:
                    #         sess.run(tf.global_variables_initializer())
                    #         ckpt_file = tf.train.latest_checkpoint(checkpoint_path)
                    #         if ckpt_file is not None:
                    #             # print('Restoring model from {}...'.format(ckpt_file))
                    #             saver.restore(sess, ckpt_file)
                    #         else:
                    #             raise ValueError("No model loaded")
                    #         res = sess.run([tf.nn.softmax(p), v], feed_dict={x: feed_board, is_training: itflag})
                    #         # res = sess.run([tf.nn.softmax(p),v], feed_dict={x:fix_board["boards"][300].reshape(-1, 19, 19, 17), is_training:False})
                    #         # res = sess.run([tf.nn.softmax(p),v], feed_dict={x:fix_board["boards"][50].reshape(-1, 19, 19, 17), is_training:True})
                    #         # print(np.argmax(res[0]))
                    #         np.savetxt(sys.stdout, res[0][0], fmt="%.6f", newline=" ")
                    #         np.savetxt(sys.stdout, res[1][0], fmt="%.6f", newline=" ")
                    #         pv_file = "/home/jialian/stuGotianshou/leela-zero/src/mcts_nn_files/policy_value"
                    #         np.savetxt(pv_file, np.concatenate((res[0][0], res[1][0])), fmt="%.6f", newline=" ")
                    #     # np.savetxt(pv_file, res[1][0], fmt="%.6f", newline=" ")
                    #     return res

    def forward(self, checkpoint_path):
        # checkpoint_path = "/home/tongzheng/tianshou/AlphaGo/checkpoints/"
        # sess = multi_gpu.create_session()
        # sess.run(tf.global_variables_initializer())
        if checkpoint_path is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            ckpt_file = tf.train.latest_checkpoint(checkpoint_path)
            if ckpt_file is not None:
            # print('Restoring model from {}...'.format(ckpt_file))
                self.saver.restore(self.sess, ckpt_file)
            # print('Successfully loaded')
            else:
                raise ValueError("No model loaded")
        # prior, value = sess.run([tf.nn.softmax(p), v], feed_dict={x: state, is_training: False})
        # return prior, value
        return self.sess


if __name__ == '__main__':
    # state = np.random.randint(0, 1, [256, 9, 9, 17])
    # net = Network()
    # net.train()
    # sess = net.forward()
    # start_time = time.time()
    # for i in range(100):
    #     sess.run([tf.nn.softmax(net.p), net.v], feed_dict={net.x: state, net.is_training: False})
    #     print("Step {}, use time {}".format(i, time.time() - start_time))
    #     start_time = time.time()
    net0 = Network()
    sess0 = net0.forward("./checkpoints/")
    print("Loaded")
    while True:
        pass

