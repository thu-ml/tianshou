from __future__ import absolute_import
import logging
import tensorflow as tf

from .base import ValueFunctionBase
from ..utils import identify_dependent_variables


class ActionValue(ValueFunctionBase):
    """
    class of action values Q(s, a).
    """
    def __init__(self, network_callable, observation_placeholder, action_placeholder, has_old_net=False):
        self.observation_placeholder = observation_placeholder
        self.action_placeholder = action_placeholder
        self.managed_placeholders = {'observation': observation_placeholder, 'action': action_placeholder}
        self.has_old_net = has_old_net

        network_scope = 'network'
        net_old_scope = 'net_old'

        with tf.variable_scope(network_scope, reuse=tf.AUTO_REUSE):
            value_tensor = network_callable()[1]
            assert value_tensor is not None

        super(ActionValue, self).__init__(value_tensor, observation_placeholder=observation_placeholder)

        weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.network_weights = identify_dependent_variables(self.value_tensor, weights)
        self._trainable_variables = [var for var in self.network_weights
                                    if var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

        # deal with target network
        if has_old_net:
            with tf.variable_scope(net_old_scope, reuse=tf.AUTO_REUSE):
                self.value_tensor_old = tf.squeeze(network_callable()[1])

            old_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=net_old_scope)

            # re-filter to rule out some edge cases
            old_weights = [var for var in old_weights if var.name[:len(net_old_scope)] == net_old_scope]

            self.network_old_weights = identify_dependent_variables(self.value_tensor_old, old_weights)
            assert len(self.network_weights) == len(self.network_old_weights)

            self.sync_weights_ops = [tf.assign(variable_old, variable)
                                     for (variable_old, variable) in
                                     zip(self.network_old_weights, self.network_weights)]
        else:
            self.sync_weights_ops = None

    @property
    def trainable_variables(self):
        return set(self._trainable_variables)

    def eval_value(self, observation, action):
        """
        :param observation: numpy array of observations, of shape (batchsize, observation_dim).
        :param action: numpy array of actions, of shape (batchsize, action_dim)
        # TODO: Atari discrete action should have dim 1. Super Mario may should have, say, dim 5, where each can be 0/1
        :return: numpy array of state values, of shape (batchsize, )
        # TODO: dealing with the last dim of 1 in V(s) and Q(s, a)
        """
        sess = tf.get_default_session()
        return sess.run(self.value_tensor, feed_dict=
        {self.observation_placeholder: observation, self.action_placeholder: action})

    def eval_value_old(self, observation, action):
        """
        eval value using target network
        :param observation: numpy array of obs
        :param action: numpy array of action
        :return: numpy array of action value
        """
        sess = tf.get_default_session()
        feed_dict = {self.observation_placeholder: observation, self.action_placeholder: action}
        return sess.run(self.value_tensor_old, feed_dict=feed_dict)

    def sync_weights(self):
        """
        sync the weights of network_old. Direct copy the weights of network.
        :return:
        """
        if self.sync_weights_ops is not None:
            sess = tf.get_default_session()
            sess.run(self.sync_weights_ops)


class DQN(ValueFunctionBase):
    """
    class of the very DQN architecture. Instead of feeding s and a to the network to get a value, DQN feed s to the
    network and the last layer is Q(s, *) for all actions
    """
    def __init__(self, network_callable, observation_placeholder, has_old_net=False):
        self.observation_placeholder = observation_placeholder
        self.action_placeholder = action_placeholder = tf.placeholder(tf.int32, shape=(None,), name='action_value.DQN/action_placeholder')
        self.managed_placeholders = {'observation': observation_placeholder, 'action': action_placeholder}
        self.has_old_net = has_old_net

        network_scope = 'network'
        net_old_scope = 'net_old'

        with tf.variable_scope(network_scope, reuse=tf.AUTO_REUSE):
            value_tensor = network_callable()[1]
            assert value_tensor is not None

            self._value_tensor_all_actions = value_tensor

            self.num_actions = value_tensor.shape.as_list()[-1]

            batch_size = tf.shape(value_tensor)[0]
            batch_dim_index = tf.range(batch_size)
            indices = tf.stack([batch_dim_index, action_placeholder], axis=1)
            canonical_value_tensor = tf.gather_nd(value_tensor, indices)

        super(DQN, self).__init__(canonical_value_tensor, observation_placeholder=observation_placeholder)

        weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.network_weights = identify_dependent_variables(self.value_tensor, weights)
        self._trainable_variables = [var for var in self.network_weights
                                    if var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

        # deal with target network
        if has_old_net:
            with tf.variable_scope(net_old_scope, reuse=tf.AUTO_REUSE):
                value_tensor = network_callable()[1]
                self.value_tensor_all_actions_old = value_tensor

                batch_size = tf.shape(value_tensor)[0]
                batch_dim_index = tf.range(batch_size)
                indices = tf.stack([batch_dim_index, action_placeholder], axis=1)
                canonical_value_tensor = tf.gather_nd(value_tensor, indices)

                self.value_tensor_old = canonical_value_tensor

            old_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=net_old_scope)

            # re-filter to rule out some edge cases
            old_weights = [var for var in old_weights if var.name[:len(net_old_scope)] == net_old_scope]

            self.network_old_weights = identify_dependent_variables(self.value_tensor_old, old_weights)
            assert len(self.network_weights) == len(self.network_old_weights)

            self.sync_weights_ops = [tf.assign(variable_old, variable)
                                     for (variable_old, variable) in
                                     zip(self.network_old_weights, self.network_weights)]
        else:
            self.sync_weights_ops = None

    @property
    def trainable_variables(self):
        return set(self._trainable_variables)

    def eval_value_all_actions(self, observation):
        """
        :param observation:
        :return: numpy array of Q(s, *) given s, of shape (batchsize, num_actions)
        """
        sess = tf.get_default_session()
        return sess.run(self._value_tensor_all_actions, feed_dict={self.observation_placeholder: observation})

    @property
    def value_tensor_all_actions(self):
        return self._value_tensor_all_actions

    def eval_value_old(self, observation, action):
        """
        eval value using target network
        :param observation: numpy array of obs
        :param action: numpy array of action
        :return: numpy array of action value
        """
        sess = tf.get_default_session()
        feed_dict = {self.observation_placeholder: observation, self.action_placeholder: action}
        return sess.run(self.value_tensor_old, feed_dict=feed_dict)

    def eval_value_all_actions_old(self, observation):
        """
        :param observation:
        :return: numpy array of Q(s, *) given s, of shape (batchsize, num_actions)
        """
        sess = tf.get_default_session()
        return sess.run(self.value_tensor_all_actions_old, feed_dict={self.observation_placeholder: observation})

    def sync_weights(self):
        """
        sync the weights of network_old. Direct copy the weights of network.
        :return:
        """
        if self.sync_weights_ops is not None:
            sess = tf.get_default_session()
            sess.run(self.sync_weights_ops)
