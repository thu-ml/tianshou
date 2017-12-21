import os
import time
import sys
import cPickle

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


class ResNet(object):
    def __init__(self, board_size, action_num, history_length=1, residual_block_num=20, checkpoint_path=None):
        """
        the resnet model

        :param board_size: an integer, the board size
        :param action_num: an integer, number of unique actions at any state
        :param history_length: an integer, the history length to use, default is 1
        :param residual_block_num: an integer, the number of residual block, default is 20, at least 1
        :param checkpoint_path: a string, the path to the checkpoint, default is None,
        """
        self.board_size = board_size
        self.action_num = action_num
        self.history_length = history_length
        self.x = tf.placeholder(tf.float32, shape=[None, self.board_size, self.board_size, 2 * self.history_length + 1])
        self.is_training = tf.placeholder(tf.bool, shape=[])
        self.z = tf.placeholder(tf.float32, shape=[None, 1])
        self.pi = tf.placeholder(tf.float32, shape=[None, self.action_num])
        self._build_network(residual_block_num, checkpoint_path)

    def _build_network(self, residual_block_num, checkpoint_path):
        """
        build the network

        :param residual_block_num: an integer, the number of residual block
        :param checkpoint_path: a string, the path to the checkpoint, if None, use random initialization parameter
        :return: None
        """

        h = layers.conv2d(self.x, 256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                          normalizer_fn=layers.batch_norm,
                          normalizer_params={'is_training': self.is_training,
                                             'updates_collections': tf.GraphKeys.UPDATE_OPS},
                          weights_regularizer=layers.l2_regularizer(1e-4))
        for i in range(residual_block_num - 1):
            h = residual_block(h, self.is_training)
        self.v = value_head(h, self.is_training)
        self.p = policy_head(h, self.is_training, self.action_num)
        self.value_loss = tf.reduce_mean(tf.square(self.z - self.v))
        self.policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.pi, logits=self.p))

        self.reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.total_loss = self.value_loss + self.policy_loss + self.reg
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.total_loss)
        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(max_to_keep=10, var_list=self.var_list)
        self.sess = multi_gpu.create_session()
        self.sess.run(tf.global_variables_initializer())
        if checkpoint_path is not None:
            ckpt_file = tf.train.latest_checkpoint(checkpoint_path)
            if ckpt_file is not None:
                print('Restoring model from {}...'.format(ckpt_file))
                self.saver.restore(self.sess, ckpt_file)
                print('Successfully loaded')
            else:
                raise ValueError("No model in path {}".format(checkpoint_path))

    def __call__(self, state):
        """

        :param history: a list, the history
        :param color: a string, indicate which one to play
        :return: a list of tensor, the predicted value and policy given the history and color
        """
        history, color = state
        if len(history) != self.history_length:
            raise ValueError(
                'The length of history cannot meet the need of the model, given {}, need {}'.format(len(history),
                                                                                                    self.history_length))
        state = self._history2state(history, color)
        return self.sess.run([self.p, self.v], feed_dict={self.x: state, self.is_training: False})

    def _history2state(self, history, color):
        """
        convert the history to the state we need

        :param history: a list, the history
        :param color: a string, indicate which one to play
        :return: a ndarray, the state
        """
        state = np.zeros([1, self.board_size, self.board_size, 2 * self.history_length + 1])
        for i in range(self.history_length):
            state[0, :, :, i] = np.array(np.array(history[i]) == np.ones(self.board_size ** 2)).reshape(self.board_size,
                                                                                                        self.board_size)
            state[0, :, :, i + self.history_length] = np.array(
                np.array(history[i]) == -np.ones(self.board_size ** 2)).reshape(self.board_size, self.board_size)
        # TODO: need a config to specify the BLACK and WHITE
        if color == +1:
            state[0, :, :, 2 * self.history_length] = np.ones([self.board_size, self.board_size])
        if color == -1:
            state[0, :, :, 2 * self.history_length] = np.zeros([self.board_size, self.board_size])
        return state

    #TODO: design the interface between the environment and training
    def train(self, mode='memory', *args, **kwargs):
        if mode == 'memory':
            pass
        if mode == 'file':
            self.train_with_file(data_path=kwargs['data_path'], checkpoint_path=kwargs['checkpoint_path'])

    def train_with_file(self, data_path, checkpoint_path):
        if not os.path.exists(data_path):
            raise ValueError("{} doesn't exist".format(data_path))

        file_list = os.listdir(data_path)
        if file_list <= 50:
            time.sleep(1)
        else:
            file_list.sort(key=lambda file: os.path.getmtime(data_path + file) if not os.path.isdir(
                data_path + file) else 0)

