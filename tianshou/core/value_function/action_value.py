from __future__ import absolute_import
import logging
import tensorflow as tf

from .base import ValueFunctionBase
from ..utils import identify_dependent_variables


class ActionValue(ValueFunctionBase):
    """
    Class for action values Q(s, a). The input of the value network is states and actions and the output
    of the value network is directly the Q-value of the input (state, action) pairs.

    :param network_callable: A Python callable returning (action head, value head). When called it builds
        the tf graph and returns a Tensor of the value on the value head.
    :param observation_placeholder: A :class:`tf.placeholder`. The observation placeholder for s in Q(s, a)
        in the network graph.
    :param action_placeholder: A :class:`tf.placeholder`. The action placeholder for a in Q(s, a)
        in the network graph.
    :param has_old_net: A bool defaulting to ``False``. If true this class will create another graph with another
        set of :class:`tf.Variable` s to be the "old net". The "old net" could be the target networks as in DQN
        and DDPG, or just an old net to help optimization as in PPO.
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
        """
        The trainable variables of the value network in a Python **set**. It contains only the :class:`tf.Variable` s
        that affect the value.
        """
        return set(self._trainable_variables)

    def eval_value(self, observation, action, my_feed_dict={}):
        """
        Evaluate value in minibatch using the current network.

        :param observation: An array-like, of shape (batch_size,) + observation_shape.
        :param action: An array-like, of shape (batch_size,) + action_shape.
        :param my_feed_dict: Optional. A dict defaulting to empty.
            Specifies placeholders such as dropout and batch_norm except observation and action.

        :return: A numpy array of shape (batch_size,). The corresponding action value for each observation.
        """
        sess = tf.get_default_session()
        feed_dict = {self.observation_placeholder: observation, self.action_placeholder: action}
        feed_dict.update(my_feed_dict)
        return sess.run(self.value_tensor, feed_dict=feed_dict)

    def eval_value_old(self, observation, action, my_feed_dict={}):
        """
        Evaluate value in minibatch using the old net.

        :param observation: An array-like, of shape (batch_size,) + observation_shape.
        :param action: An array-like, of shape (batch_size,) + action_shape.
        :param my_feed_dict: Optional. A dict defaulting to empty.
            Specifies placeholders such as dropout and batch_norm except observation and action.

        :return: A numpy array of shape (batch_size,). The corresponding action value for each observation.
        """
        sess = tf.get_default_session()
        feed_dict = {self.observation_placeholder: observation, self.action_placeholder: action}
        feed_dict.update(my_feed_dict)
        return sess.run(self.value_tensor_old, feed_dict=feed_dict)

    def sync_weights(self):
        """
        Sync the variables of the "old net" to be the same as the current network.
        """
        if self.sync_weights_ops is not None:
            sess = tf.get_default_session()
            sess.run(self.sync_weights_ops)


class DQN(ValueFunctionBase):
    """
    Class for the special action value function DQN. Instead of feeding s and a to the network to get a value,
    DQN feeds s to the network and gets at the last layer Q(s, \*) for all actions under this state. Still, as
    :class:`ActionValue`, this class still builds the Q(s, a) value Tensor. It can only be used with discrete
    (and finite) action spaces.

    :param network_callable: A Python callable returning (action head, value head). When called it builds
        the tf graph and returns a Tensor of Q(s, \*) on the value head.
    :param observation_placeholder: A :class:`tf.placeholder`. The observation placeholder for s in Q(s, \*)
        in the network graph.
    :param has_old_net: A bool defaulting to ``False``. If true this class will create another graph with another
        set of :class:`tf.Variable` s to be the "old net". The "old net" could be the target networks as in DQN
        and DDPG, or just an old net to help optimization as in PPO.
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
        """
        The trainable variables of the value network in a Python **set**. It contains only the :class:`tf.Variable` s
        that affect the value.
        """
        return set(self._trainable_variables)

    def eval_value(self, observation, action, my_feed_dict={}):
        """
        Evaluate value Q(s, a) in minibatch using the current network.

        :param observation: An array-like, of shape (batch_size,) + observation_shape.
        :param action: An array-like, of shape (batch_size,) + action_shape.
        :param my_feed_dict: Optional. A dict defaulting to empty.
            Specifies placeholders such as dropout and batch_norm except observation and action.

        :return: A numpy array of shape (batch_size,). The corresponding action value for each observation.
        """
        sess = tf.get_default_session()
        feed_dict = {self.observation_placeholder: observation, self.action_placeholder: action}
        feed_dict.update(my_feed_dict)
        return sess.run(self.value_tensor, feed_dict=feed_dict)

    def eval_value_old(self, observation, action, my_feed_dict={}):
        """
        Evaluate value Q(s, a) in minibatch using the old net.

        :param observation: An array-like, of shape (batch_size,) + observation_shape.
        :param action: An array-like, of shape (batch_size,) + action_shape.
        :param my_feed_dict: Optional. A dict defaulting to empty.
            Specifies placeholders such as dropout and batch_norm except observation and action.

        :return: A numpy array of shape (batch_size,). The corresponding action value for each observation.
        """
        sess = tf.get_default_session()
        feed_dict = {self.observation_placeholder: observation, self.action_placeholder: action}
        feed_dict.update(my_feed_dict)
        return sess.run(self.value_tensor_old, feed_dict=feed_dict)

    @property
    def value_tensor_all_actions(self):
        """The Tensor for Q(s, \*)"""
        return self._value_tensor_all_actions

    def eval_value_all_actions(self, observation, my_feed_dict={}):
        """
        Evaluate values Q(s, \*) in minibatch using the current network.

        :param observation: An array-like, of shape (batch_size,) + observation_shape.
        :param my_feed_dict: Optional. A dict defaulting to empty.
            Specifies placeholders such as dropout and batch_norm except observation and action.

        :return: A numpy array of shape (batch_size, num_actions). The corresponding action values for each observation.
        """
        sess = tf.get_default_session()
        feed_dict = {self.observation_placeholder: observation}
        feed_dict.update(my_feed_dict)
        return sess.run(self._value_tensor_all_actions, feed_dict=feed_dict)

    def eval_value_all_actions_old(self, observation, my_feed_dict={}):
        """
        Evaluate values Q(s, \*) in minibatch using the old net.

        :param observation: An array-like, of shape (batch_size,) + observation_shape.
        :param my_feed_dict: Optional. A dict defaulting to empty.
            Specifies placeholders such as dropout and batch_norm except observation and action.

        :return: A numpy array of shape (batch_size, num_actions). The corresponding action values for each observation.
        """
        sess = tf.get_default_session()
        feed_dict = {self.observation_placeholder: observation}
        feed_dict.update(my_feed_dict)
        return sess.run(self.value_tensor_all_actions_old, feed_dict=feed_dict)

    def sync_weights(self):
        """
        Sync the variables of the "old net" to be the same as the current network.
        """
        if self.sync_weights_ops is not None:
            sess = tf.get_default_session()
            sess.run(self.sync_weights_ops)
