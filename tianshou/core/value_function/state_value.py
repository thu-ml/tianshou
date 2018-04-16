from __future__ import absolute_import

import tensorflow as tf
import logging

from .base import ValueFunctionBase
from ..utils import identify_dependent_variables


class StateValue(ValueFunctionBase):
    """
    Class for state value functions V(s). The input of the value network is states and the output
    of the value network is directly the V-value of the input state.

    :param network_callable: A Python callable returning (action head, value head). When called it builds
        the tf graph and returns a Tensor of the value on the value head.
    :param observation_placeholder: A :class:`tf.placeholder`. The observation placeholder for s in V(s)
        in the network graph.
    :param has_old_net: A bool defaulting to ``False``. If true this class will create another graph with another
        set of :class:`tf.Variable` s to be the "old net". The "old net" could be the target networks as in DQN
        and DDPG, or just an old net to help optimization as in PPO.
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
        """
        The trainable variables of the value network in a Python **set**. It contains only the :class:`tf.Variable` s
        that affect the value.
        """
        return set(self._trainable_variables)

    def eval_value(self, observation, my_feed_dict={}):
        """
        Evaluate value in minibatch using the current network.

        :param observation: An array-like, of shape (batch_size,) + observation_shape.
        :param my_feed_dict: Optional. A dict defaulting to empty.
            Specifies placeholders such as dropout and batch_norm except observation.

        :return: A numpy array of shape (batch_size,). The corresponding state value for each observation.
        """
        sess = tf.get_default_session()
        feed_dict = {self.observation_placeholder: observation}
        feed_dict.update(my_feed_dict)
        return sess.run(self.value_tensor, feed_dict=feed_dict)

    def eval_value_old(self, observation, my_feed_dict={}):
        """
        Evaluate value in minibatch using the old net.

        :param observation: An array-like, of shape (batch_size,) + observation_shape.
        :param my_feed_dict: Optional. A dict defaulting to empty.
            Specifies placeholders such as dropout and batch_norm except observation.

        :return: A numpy array of shape (batch_size,). The corresponding state value for each observation.
        """
        sess = tf.get_default_session()
        feed_dict = {self.observation_placeholder: observation}
        feed_dict.update(my_feed_dict)
        return sess.run(self.value_tensor_old, feed_dict=feed_dict)

    def sync_weights(self):
        """
        Sync the variables of the "old net" to be the same as the current network.
        """
        if self.sync_weights_ops is not None:
            sess = tf.get_default_session()
            sess.run(self.sync_weights_ops)
