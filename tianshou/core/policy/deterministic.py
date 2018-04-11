import tensorflow as tf
import logging

from .base import PolicyBase
from ..random import OrnsteinUhlenbeckProcess
from ..utils import identify_dependent_variables


class Deterministic(PolicyBase):
    """
    deterministic policy as used in deterministic policy gradient (DDPG) methods
    """
    def __init__(self, network_callable, observation_placeholder, has_old_net=False, random_process=None):
        self.observation_placeholder = observation_placeholder
        self.managed_placeholders = {'observation': observation_placeholder}

        self.has_old_net = has_old_net

        network_scope = 'network'
        net_old_scope = 'net_old'

        # build network, action and value
        with tf.variable_scope(network_scope, reuse=tf.AUTO_REUSE):
            action = network_callable()[0]
            assert action is not None
            self.action = action

        weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.network_weights = identify_dependent_variables(self.action, weights)
        self._trainable_variables = [var for var in self.network_weights
                                    if var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

        # deal with target network
        if not has_old_net:
            self.sync_weights_ops = None
        else:  # then we need to build another tf graph as target network
            with tf.variable_scope('net_old', reuse=tf.AUTO_REUSE):
                self.action_old = network_callable()[0]

            old_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=net_old_scope)

            # re-filter to rule out some edge cases
            old_weights = [var for var in old_weights if var.name[:len(net_old_scope)] == net_old_scope]

            self.network_old_weights = identify_dependent_variables(self.action_old, old_weights)
            assert len(self.network_weights) == len(self.network_old_weights)

            self.sync_weights_ops = [tf.assign(variable_old, variable)
                                     for (variable_old, variable) in zip(self.network_old_weights, self.network_weights)]

        # random process for exploration for deterministic policies
        self.random_process = random_process or OrnsteinUhlenbeckProcess(
                                            theta=0.15, sigma=0.3, size=self.action.shape.as_list()[-1])

    @property
    def trainable_variables(self):
        return set(self._trainable_variables)

    def act(self, observation, my_feed_dict={}):
        sess = tf.get_default_session()

        # observation[None] adds one dimension at the beginning
        feed_dict = {self.observation_placeholder: observation[None]}
        feed_dict.update(my_feed_dict)
        sampled_action = sess.run(self.action, feed_dict=feed_dict)

        sampled_action = sampled_action[0] + self.random_process.sample()

        return sampled_action

    def reset(self):
        self.random_process.reset_states()

    def act_test(self, observation, my_feed_dict={}):
        sess = tf.get_default_session()

        # observation[None] adds one dimension at the beginning
        feed_dict = {self.observation_placeholder: observation[None]}
        feed_dict.update(my_feed_dict)
        sampled_action = sess.run(self.action, feed_dict=feed_dict)

        sampled_action = sampled_action[0]

        return sampled_action

    def sync_weights(self):
        """
        sync the weights of network_old. Direct copy the weights of network.
        :return:
        """
        if self.sync_weights_ops is not None:
            sess = tf.get_default_session()
            sess.run(self.sync_weights_ops)

    def eval_action(self, observation):
        """
        evaluate action in minibatch
        :param observation:
        :return: 2-D numpy array
        """
        sess = tf.get_default_session()

        feed_dict = {self.observation_placeholder: observation}
        action = sess.run(self.action, feed_dict=feed_dict)

        return action

    def eval_action_old(self, observation):
        """
        evaluate action in minibatch
        :param observation:
        :return: 2-D numpy array
        """
        sess = tf.get_default_session()

        feed_dict = {self.observation_placeholder: observation}
        action = sess.run(self.action_old, feed_dict=feed_dict)

        return action