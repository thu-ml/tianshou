import os
import time
import copy
import cPickle
from collections import deque

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import multi_gpu

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def residual_block(input, is_training):
    """
    one residual block

    :param input: a tensor, input of the residual block
    :param is_training: a placeholder, indicate whether the model is training or not
    :return: a tensor, output of the residual block
    """
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


def policy_head(input, is_training, action_num):
    """
    the head of policy branch

    :param input: a tensor, input of the policy head
    :param is_training: a placeholder, indicate whether the model is training or not
    :param action_num: action_num: an integer, number of unique actions at any state
    :return: a tensor: output of the policy head, shape [batch_size, action_num]
    """
    normalizer_params = {'is_training': is_training,
                         'updates_collections': tf.GraphKeys.UPDATE_OPS}
    h = layers.conv2d(input, 2, kernel_size=1, stride=1, activation_fn=tf.nn.relu,
                      normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params,
                      weights_regularizer=layers.l2_regularizer(1e-4))
    h = layers.flatten(h)
    h = layers.fully_connected(h, action_num, activation_fn=tf.identity,
                               weights_regularizer=layers.l2_regularizer(1e-4))
    return h


def value_head(input, is_training):
    """
    the head of value branch

    :param input: a tensor, input of the value head
    :param is_training: a placeholder, indicate whether the model is training or not
    :return: a tensor, output of the value head, shape [batch_size, 1]
    """
    normalizer_params = {'is_training': is_training,
                         'updates_collections': tf.GraphKeys.UPDATE_OPS}
    h = layers.conv2d(input, 2, kernel_size=1, stride=1, activation_fn=tf.nn.relu,
                      normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params,
                      weights_regularizer=layers.l2_regularizer(1e-4))
    h = layers.flatten(h)
    h = layers.fully_connected(h, 256, activation_fn=tf.nn.relu, weights_regularizer=layers.l2_regularizer(1e-4))
    h = layers.fully_connected(h, 1, activation_fn=tf.nn.tanh, weights_regularizer=layers.l2_regularizer(1e-4))
    return h


class Data(object):
    def __init__(self):
        self.boards = []
        self.probs = []
        self.winner = 0


class ResNet(object):
    def __init__(self, board_size, action_num, history_length=1, residual_block_num=10, black_checkpoint_path=None,
                 white_checkpoint_path=None):
        """
        the resnet model

        :param board_size: an integer, the board size
        :param action_num: an integer, number of unique actions at any state
        :param history_length: an integer, the history length to use, default is 1
        :param residual_block_num: an integer, the number of residual block, default is 20, at least 1
        :param black_checkpoint_path: a string, the path to the black checkpoint, default is None,
        :param white_checkpoint_path: a string, the path to the white checkpoint, default is None,
        """
        self.board_size = board_size
        self.action_num = action_num
        self.history_length = history_length
        self.black_checkpoint_path = black_checkpoint_path
        self.white_checkpoint_path = white_checkpoint_path
        self.x = tf.placeholder(tf.float32, shape=[None, self.board_size, self.board_size, 2 * self.history_length + 1])
        self.is_training = tf.placeholder(tf.bool, shape=[])
        self.z = tf.placeholder(tf.float32, shape=[None, 1])
        self.pi = tf.placeholder(tf.float32, shape=[None, self.action_num])
        self._build_network('black', residual_block_num)
        self._build_network('white', residual_block_num)
        self.sess = multi_gpu.create_session()
        self.sess.run(tf.global_variables_initializer())
        if black_checkpoint_path is not None:
            ckpt_file = tf.train.latest_checkpoint(black_checkpoint_path)
            if ckpt_file is not None:
                print('Restoring model from {}...'.format(ckpt_file))
                self.black_saver.restore(self.sess, ckpt_file)
                print('Successfully loaded')
            else:
                raise ValueError("No model in path {}".format(black_checkpoint_path))

        if white_checkpoint_path is not None:
            ckpt_file = tf.train.latest_checkpoint(white_checkpoint_path)
            if ckpt_file is not None:
                print('Restoring model from {}...'.format(ckpt_file))
                self.white_saver.restore(self.sess, ckpt_file)
                print('Successfully loaded')
            else:
                raise ValueError("No model in path {}".format(white_checkpoint_path))
        self.update = [tf.assign(black_params, white_params) for black_params, white_params in
                       zip(self.black_var_list, self.white_var_list)]

        # training hyper-parameters:
        self.window_length = 900
        self.save_freq = 5000
        self.training_data = {'states': deque(maxlen=self.window_length), 'probs': deque(maxlen=self.window_length),
                              'winner': deque(maxlen=self.window_length), 'length': deque(maxlen=self.window_length)}

    def _build_network(self, scope, residual_block_num):
        """
        build the network

        :param residual_block_num: an integer, the number of residual block
        :param checkpoint_path: a string, the path to the checkpoint, if None, use random initialization parameter
        :return: None
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            h = layers.conv2d(self.x, 256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                              normalizer_fn=layers.batch_norm,
                              normalizer_params={'is_training': self.is_training,
                                                 'updates_collections': tf.GraphKeys.UPDATE_OPS},
                              weights_regularizer=layers.l2_regularizer(1e-4))
            for i in range(residual_block_num - 1):
                h = residual_block(h, self.is_training)
            self.__setattr__(scope + '_v', value_head(h, self.is_training))
            self.__setattr__(scope + '_p', policy_head(h, self.is_training, self.action_num))
        self.__setattr__(scope + '_prob', tf.nn.softmax(self.__getattribute__(scope + '_p')))
        self.__setattr__(scope + '_value_loss', tf.reduce_mean(tf.square(self.z - self.__getattribute__(scope + '_v'))))
        self.__setattr__(scope + '_policy_loss',
                         tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.pi,
                                                                                logits=self.__getattribute__(
                                                                                    scope + '_p'))))

        self.__setattr__(scope + '_reg', tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope)))
        self.__setattr__(scope + '_total_loss', self.__getattribute__(scope + '_value_loss') + self.__getattribute__(
            scope + '_policy_loss') + self.__getattribute__(scope + '_reg'))
        self.__setattr__(scope + '_update_ops', tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope))
        self.__setattr__(scope + '_var_list', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        with tf.control_dependencies(self.__getattribute__(scope + '_update_ops')):
            self.__setattr__(scope + '_train_op',
                             tf.train.AdamOptimizer(1e-4).minimize(self.__getattribute__(scope + '_total_loss'),
                                                                   var_list=self.__getattribute__(scope + '_var_list')))
        self.__setattr__(scope + '_saver',
                         tf.train.Saver(max_to_keep=0, var_list=self.__getattribute__(scope + '_var_list')))

    def __call__(self, state):
        """

        :param history: a list, the history
        :param color: a string, indicate which one to play
        :return: a list of tensor, the predicted value and policy given the history and color
        """
        # Note : maybe we can use it for isolating test of MCTS
        # prob = [1.0 / self.action_num] * self.action_num
        # return [prob, np.random.uniform(-1, 1)]
        history, color = state
        if len(history) != self.history_length:
            raise ValueError(
                'The length of history cannot meet the need of the model, given {}, need {}'.format(len(history),
                                                                                                    self.history_length))
        eval_state = self._history2state(history, color)
        if color == +1:
            return self.sess.run([self.black_prob, self.black_v],
                                 feed_dict={self.x: eval_state, self.is_training: False})
        if color == -1:
            return self.sess.run([self.white_prob, self.white_v],
                                 feed_dict={self.x: eval_state, self.is_training: False})

    def _history2state(self, history, color):
        """
        convert the history to the state we need

        :param history: a list, the history
        :param color: a string, indicate which one to play
        :return: a ndarray, the state
        """
        state = np.zeros([1, self.board_size, self.board_size, 2 * self.history_length + 1])
        for i in range(self.history_length):
            state[0, :, :, i] = np.array(np.array(history[i]).flatten() == np.ones(self.board_size ** 2)).reshape(
                self.board_size,
                self.board_size)
            state[0, :, :, i + self.history_length] = np.array(
                np.array(history[i]).flatten() == -np.ones(self.board_size ** 2)).reshape(self.board_size,
                                                                                          self.board_size)
        # TODO: need a config to specify the BLACK and WHITE
        if color == +1:
            state[0, :, :, 2 * self.history_length] = np.ones([self.board_size, self.board_size])
        if color == -1:
            state[0, :, :, 2 * self.history_length] = np.zeros([self.board_size, self.board_size])
        return state

    # TODO: design the interface between the environment and training
    def train(self, mode='memory', *args, **kwargs):
        """
        The method to train the network

        :param target: a string, which to optimize, can only be "both", "black" and "white"
        :param mode: a string, how to optimize, can only be "memory" and "file"
        """
        if mode == 'memory':
            pass
        if mode == 'file':
            self._train_with_file(data_path=kwargs['data_path'], batch_size=kwargs['batch_size'],
                                  save_path=kwargs['save_path'])

    def _train_with_file(self, data_path, batch_size, save_path):
        # check if the path is valid
        if not os.path.exists(data_path):
            raise ValueError("{} doesn't exist".format(data_path))
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            os.mkdir(self.save_path + 'black')
            os.mkdir(self.save_path + 'white')

        new_file_list = []
        all_file_list = []
        training_data = {'states': [], 'probs': [], 'winner': []}

        iters = 0
        while True:
            new_file_list = list(set(os.listdir(data_path)).difference(all_file_list))
            while new_file_list:
                all_file_list = os.listdir(data_path)
                new_file_list.sort(
                    key=lambda file: os.path.getmtime(data_path + file) if not os.path.isdir(data_path + file) else 0)
                for file in new_file_list:
                    states, probs, winner = self._file_to_training_data(data_path + file)
                    assert states.shape[0] == probs.shape[0]
                    assert states.shape[0] == winner.shape[0]
                    self.training_data['states'].append(states)
                    self.training_data['probs'].append(probs)
                    self.training_data['winner'].append(winner)
                    self.training_data['length'].append(states.shape[0])
                new_file_list = list(set(os.listdir(data_path)).difference(all_file_list))

            if len(self.training_data['states']) != self.window_length:
                continue
            else:
                start_time = time.time()
                for i in range(batch_size):
                    priority = np.array(self.training_data['length']) / (
                            0.0 + np.sum(np.array(self.training_data['length'])))
                    game_num = np.random.choice(self.window_length, 1, p=priority)[0]
                    state_num = np.random.randint(self.training_data['length'][game_num])
                    rotate_times = np.random.randint(4)
                    reflect_times = np.random.randint(2)
                    reflect_orientation = np.random.randint(2)
                    training_data['states'].append(
                        self._preprocession(self.training_data['states'][game_num][state_num], reflect_times,
                                            reflect_orientation, rotate_times))
                    training_data['probs'].append(np.concatenate(
                        [self._preprocession(
                            self.training_data['probs'][game_num][state_num][:-1].reshape(self.board_size,
                                                                                          self.board_size, 1),
                            reflect_times,
                            reflect_orientation, rotate_times).reshape(1, self.board_size ** 2),
                         self.training_data['probs'][game_num][state_num][-1].reshape(1, 1)], axis=1))
                    training_data['winner'].append(self.training_data['winner'][game_num][state_num].reshape(1, 1))
                value_loss, policy_loss, reg, _ = self.sess.run(
                    [self.black_value_loss, self.black_policy_loss, self.black_reg, self.black_train_op],
                    feed_dict={self.x: np.concatenate(training_data['states'], axis=0),
                               self.z: np.concatenate(training_data['winner'], axis=0),
                               self.pi: np.concatenate(training_data['probs'], axis=0),
                               self.is_training: True})

                print("Iteration: {}, Time: {}, Value Loss: {}, Policy Loss: {}, Reg: {}".format(iters,
                                                                                                 time.time() - start_time,
                                                                                                 value_loss,
                                                                                                 policy_loss, reg))
                if iters % self.save_freq == 0:
                    ckpt_file = "Iteration{}.ckpt".format(iters)
                    self.black_saver.save(self.sess, self.save_path + 'black/' + ckpt_file)
                    self.sess.run(self.update)
                    self.white_saver.save(self.sess, self.save_path + 'white/' + ckpt_file)

                for key in training_data.keys():
                    training_data[key] = []
                iters += 1

    def _file_to_training_data(self, file_name):
        read = False
        with open(file_name, 'rb') as file:
            while not read:
                try:
                    file.seek(0)
                    data = cPickle.load(file)
                    read = True
                    print("{} Loaded!".format(file_name))
                except Exception as e:
                    print(e)
                    time.sleep(1)
        history = deque(maxlen=self.history_length)
        states = []
        probs = []
        winner = []
        for _ in range(self.history_length):
            # Note that 0 is specified, need a more general way like config
            history.append([0] * self.board_size ** 2)
        # Still, +1 is specified
        color = +1

        for [board, prob] in zip(data.boards, data.probs):
            history.append(board)
            states.append(self._history2state(history, color))
            probs.append(np.array(prob).reshape(1, self.board_size ** 2 + 1))
            winner.append(np.array(data.winner * color).reshape(1, 1))
            color *= -1
        states = np.concatenate(states, axis=0)
        probs = np.concatenate(probs, axis=0)
        winner = np.concatenate(winner, axis=0)
        return states, probs, winner

    def _preprocession(self, board, reflect_times=0, reflect_orientation=0, rotate_times=0):
        """
        preprocessing for augmentation

        :param board: a ndarray, board to process
        :param reflect_times: an integer, how many times to reflect
        :param reflect_orientation: an integer, which orientation to reflect
        :param rotate_times: an integer, how many times to rotate
        :return:
        """

        new_board = copy.deepcopy(board)
        if new_board.ndim == 3:
            new_board = np.expand_dims(new_board, axis=0)

        new_board = self._board_reflection(new_board, reflect_times, reflect_orientation)
        new_board = self._board_rotation(new_board, rotate_times)

        return new_board

    def _board_rotation(self, board, times):
        """
        rotate the board for augmentation
        note that board's shape should be [batch_size, board_size, board_size, channels]

        :param board: a ndarray, shape [batch_size, board_size, board_size, channels]
        :param times: an integer, how many times to rotate
        :return:
        """
        return np.rot90(board, times, (1, 2))

    def _board_reflection(self, board, times, orientation):
        """
        reflect the board for augmentation
        note that board's shape should be [batch_size, board_size, board_size, channels]

        :param board: a ndarray, shape [batch_size, board_size, board_size, channels]
        :param times: an integer, how many times to reflect
        :param orientation: an integer, which orientation to reflect
        :return:
        """
        new_board = copy.deepcopy(board)
        for _ in range(times):
            if orientation == 0:
                new_board = new_board[:, ::-1]
            if orientation == 1:
                new_board = new_board[:, :, ::-1]
        return new_board


if __name__ == "__main__":
    model = ResNet(board_size=8, action_num=65, history_length=1, black_checkpoint_path="./checkpoint/black", white_checkpoint_path="./checkpoint/white")
    model.train(mode="file", data_path="./data/", batch_size=128, save_path="./checkpoint/")
