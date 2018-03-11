import tensorflow as tf
from .base import PolicyBase
from ..random import OrnsteinUhlenbeckProcess

class Deterministic(PolicyBase):
    """
    deterministic policy as used in deterministic policy gradient methods
    """
    def __init__(self, policy_callable, observation_placeholder, weight_update=1, random_process=None):
        self._observation_placeholder = observation_placeholder
        self.managed_placeholders = {'observation': observation_placeholder}
        self.weight_update = weight_update
        self.interaction_count = -1  # defaults to -1. only useful if weight_update > 1.

        # build network, action and value
        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            action, _ = policy_callable()
            self.action = action
            # TODO: self._action should be exactly the action tensor to run that directly gives action_dim

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        # deal with target network
        if self.weight_update == 1:
            self.weight_update_ops = None
            self.sync_weights_ops = None
        else:  # then we need to build another tf graph as target network
            with tf.variable_scope('net_old', reuse=tf.AUTO_REUSE):
                action, _ = policy_callable()
                self.action_old = action

            network_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
            network_old_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net_old')
            # TODO: use a scope that the user will almost surely not use. so get_collection will return
            # the correct weights and old_weights, since it filters by regular expression
            # or we write a util to parse the variable names and use only the topmost scope

            assert len(network_weights) == len(network_old_weights)
            self.sync_weights_ops = [tf.assign(variable_old, variable)
                                     for (variable_old, variable) in zip(network_old_weights, network_weights)]

            if weight_update == 0:
                self.weight_update_ops = self.sync_weights_ops
            elif 0 < weight_update < 1:  # as in DDPG
                self.weight_update_ops = [tf.assign(variable_old,
                                                    weight_update * variable + (1 - weight_update) * variable_old)
                                          for (variable_old, variable) in zip(network_old_weights, network_weights)]
            else:
                self.interaction_count = 0  # as in DQN
                import math
                self.weight_update = math.ceil(weight_update)

        self.random_process = random_process or OrnsteinUhlenbeckProcess(
                                            theta=0.15, sigma=0.2, size=self.action.shape.as_list()[-1])

    @property
    def action_shape(self):
        return self.action.shape.as_list()[1:]

    def act(self, observation, my_feed_dict={}):
        # TODO: this may be ugly. also maybe huge problem when parallel
        sess = tf.get_default_session()
        # observation[None] adds one dimension at the beginning

        feed_dict = {self._observation_placeholder: observation[None]}
        feed_dict.update(my_feed_dict)
        sampled_action = sess.run(self.action, feed_dict=feed_dict)

        sampled_action = sampled_action[0] + self.random_process.sample()

        return sampled_action

    def reset(self):
        self.random_process.reset_states()

    def act_test(self, observation, my_feed_dict={}):
        sess = tf.get_default_session()
        # observation[None] adds one dimension at the beginning

        feed_dict = {self._observation_placeholder: observation[None]}
        feed_dict.update(my_feed_dict)
        sampled_action = sess.run(self.action, feed_dict=feed_dict)

        sampled_action = sampled_action[0]

        return sampled_action

    def update_weights(self):
        """
        updates the weights of policy_old.
        :return:
        """
        if self.weight_update_ops is not None:
            sess = tf.get_default_session()
            sess.run(self.weight_update_ops)

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

        feed_dict = {self._observation_placeholder: observation}
        action = sess.run(self.action, feed_dict=feed_dict)

        return action

    def eval_action_old(self, observation):
        """
        evaluate action in minibatch
        :param observation:
        :return: 2-D numpy array
        """
        sess = tf.get_default_session()

        feed_dict = {self._observation_placeholder: observation}
        action = sess.run(self.action_old, feed_dict=feed_dict)

        return action