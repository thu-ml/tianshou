from __future__ import absolute_import

from .base import ValueFunctionBase
import tensorflow as tf


class ActionValue(ValueFunctionBase):
    """
    class of action values Q(s, a).
    """
    def __init__(self, network_callable, observation_placeholder, action_placeholder, weight_update=1):
        self._observation_placeholder = observation_placeholder
        self._action_placeholder = action_placeholder
        self.managed_placeholders = {'observation': observation_placeholder, 'action': action_placeholder}
        self.weight_update = weight_update
        self.interaction_count = -1  # defaults to -1. only useful if weight_update > 1.

        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            value_tensor = network_callable()[-1]

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        super(ActionValue, self).__init__(value_tensor, observation_placeholder=observation_placeholder)

        # deal with target network
        if self.weight_update == 1:
            self.weight_update_ops = None
            self.sync_weights_ops = None
        else:  # then we need to build another tf graph as target network
            with tf.variable_scope('net_old', reuse=tf.AUTO_REUSE):
                value_tensor = network_callable()[-1]
                self.value_tensor_old = tf.squeeze(value_tensor)

        network_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
        network_old_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net_old')

        assert len(network_weights) == len(network_old_weights)
        self.sync_weights_ops = [tf.assign(variable_old, variable)
                                 for (variable_old, variable) in zip(network_old_weights, network_weights)]

        if weight_update == 0:
            self.weight_update_ops = self.sync_weights_ops
        elif 0 < weight_update < 1:  # useful in DDPG
            self.weight_update_ops = [tf.assign(variable_old,
                                                weight_update * variable + (1 - weight_update) * variable_old)
                                      for (variable_old, variable) in zip(network_old_weights, network_weights)]
        else:
            self.interaction_count = 0
            import math
            self.weight_update = math.ceil(weight_update)
            self.weight_update_ops = self.sync_weights_ops

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
        {self._observation_placeholder: observation, self._action_placeholder: action})

    def eval_value_old(self, observation, action):
        """
        eval value using target network
        :param observation: numpy array of obs
        :param action: numpy array of action
        :return: numpy array of action value
        """
        sess = tf.get_default_session()
        feed_dict = {self._observation_placeholder: observation, self._action_placeholder: action}
        return sess.run(self.value_tensor_old, feed_dict=feed_dict)

    def sync_weights(self):
        """
        sync the weights of network_old. Direct copy the weights of network.
        :return:
        """
        if self.sync_weights_ops is not None:
            sess = tf.get_default_session()
            sess.run(self.sync_weights_ops)

    def update_weights(self):
        """
        updates the weights of policy_old.
        :return:
        """
        if self.weight_update_ops is not None:
            sess = tf.get_default_session()
            sess.run(self.weight_update_ops)


class DQN(ValueFunctionBase):
    """
    class of the very DQN architecture. Instead of feeding s and a to the network to get a value, DQN feed s to the
    network and the last layer is Q(s, *) for all actions
    """
    def __init__(self, network_callable, observation_placeholder, weight_update=1):
        """
        :param value_tensor: of shape (batchsize, num_actions)
        :param observation_placeholder: of shape (batchsize, observation_dim)
        :param action_placeholder: of shape (batchsize, )
        """
        self._observation_placeholder = observation_placeholder
        self.action_placeholder = action_placeholder = tf.placeholder(tf.int32, shape=(None,), name='action_value.DQN/action_placeholder')
        self.managed_placeholders = {'observation': observation_placeholder, 'action': action_placeholder}
        self.weight_update = weight_update
        self.interaction_count = -1  # defaults to -1. only useful if weight_update > 1.

        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            value_tensor = network_callable()[-1]

            self._value_tensor_all_actions = value_tensor

            self.num_actions = value_tensor.shape.as_list()[-1]

            batch_size = tf.shape(value_tensor)[0]
            batch_dim_index = tf.range(batch_size)
            indices = tf.stack([batch_dim_index, action_placeholder], axis=1)
            canonical_value_tensor = tf.gather_nd(value_tensor, indices)

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        super(DQN, self).__init__(canonical_value_tensor, observation_placeholder=observation_placeholder)

        # deal with target network
        if self.weight_update == 1:
            self.weight_update_ops = None
            self.sync_weights_ops = None
        else:  # then we need to build another tf graph as target network
            with tf.variable_scope('net_old', reuse=tf.AUTO_REUSE):
                value_tensor = network_callable()[-1]
                self.value_tensor_all_actions_old = value_tensor

                batch_size = tf.shape(value_tensor)[0]
                batch_dim_index = tf.range(batch_size)
                indices = tf.stack([batch_dim_index, action_placeholder], axis=1)
                canonical_value_tensor = tf.gather_nd(value_tensor, indices)

                self.value_tensor_old = canonical_value_tensor

        network_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
        network_old_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net_old')

        assert len(network_weights) == len(network_old_weights)
        self.sync_weights_ops = [tf.assign(variable_old, variable)
                                 for (variable_old, variable) in zip(network_old_weights, network_weights)]

        if weight_update == 0:
            self.weight_update_ops = self.sync_weights_ops
        elif 0 < weight_update < 1:  # useful in DDPG
            pass
        else:
            self.interaction_count = 0
            import math
            self.weight_update = math.ceil(weight_update)
            self.weight_update_ops = self.sync_weights_ops


    def eval_value_all_actions(self, observation):
        """
        :param observation:
        :return: numpy array of Q(s, *) given s, of shape (batchsize, num_actions)
        """
        sess = tf.get_default_session()
        return sess.run(self._value_tensor_all_actions, feed_dict={self._observation_placeholder: observation})

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
        feed_dict = {self._observation_placeholder: observation, self.action_placeholder: action}
        return sess.run(self.value_tensor_old, feed_dict=feed_dict)

    def eval_value_all_actions_old(self, observation):
        """
        :param observation:
        :return: numpy array of Q(s, *) given s, of shape (batchsize, num_actions)
        """
        sess = tf.get_default_session()
        return sess.run(self.value_tensor_all_actions_old, feed_dict={self._observation_placeholder: observation})

    def sync_weights(self):
        """
        sync the weights of network_old. Direct copy the weights of network.
        :return:
        """
        if self.sync_weights_ops is not None:
            sess = tf.get_default_session()
            sess.run(self.sync_weights_ops)

    def update_weights(self):
        """
        updates the weights of policy_old.
        :return:
        """
        if self.weight_update_ops is not None:
            sess = tf.get_default_session()
            sess.run(self.weight_update_ops)