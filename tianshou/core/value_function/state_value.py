from __future__ import absolute_import

import tensorflow as tf
import logging

from .base import ValueFunctionBase
from ..utils import identify_dependent_variables


class StateValue(ValueFunctionBase):
    """
    class of state values V(s).
    """
    def __init__(self, network_callable, observation_placeholder, has_old_net=False):
        self.observation_placeholder = observation_placeholder
        self.managed_placeholders = {'observation': observation_placeholder}

        self.has_old_net = has_old_net

        network_scope = 'network'
        net_old_scope = 'net_old'

        # build network, action and value
        with tf.variable_scope(network_scope, reuse=tf.AUTO_REUSE):
            value_tensor = network_callable()[1]
            assert value_tensor is not None

        super(StateValue, self).__init__(observation_placeholder=observation_placeholder, value_tensor=value_tensor)

        weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.network_weights = identify_dependent_variables(self.value_tensor, weights)
        self._trainable_variables = [var for var in self.network_weights
                                    if var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

        # deal with target network
        if has_old_net:
            with tf.variable_scope(net_old_scope, reuse=tf.AUTO_REUSE):
                self.value_tensor_old = network_callable()[1]

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

    def eval_value(self, observation):
        """

        :param observation: numpy array of observations, of shape (batchsize, observation_dim).
        :return: numpy array of state values, of shape (batchsize, )
        # TODO: dealing with the last dim of 1 in V(s) and Q(s, a), this should rely on the action shape returned by env
        """
        sess = tf.get_default_session()
        return sess.run(self.value_tensor, feed_dict={self.observation_placeholder: observation})

    def sync_weights(self):
        if self.sync_weights_ops is not None:
            sess = tf.get_default_session()
            sess.run(self.sync_weights_ops)
