from __future__ import absolute_import

from .base import PolicyBase
import tensorflow as tf
import numpy as np


class DQN(PolicyBase):
    """
    use DQN from value_function as a member
    """
    def __init__(self, dqn, epsilon_train=0.1, epsilon_test=0.05):
        self.action_value = dqn
        self._argmax_action = tf.argmax(dqn.value_tensor_all_actions, axis=1)
        self.weight_update = dqn.weight_update
        if self.weight_update > 1:
            self.interaction_count = 0
        else:
            self.interaction_count = -1

        self.epsilon_train = epsilon_train
        self.epsilon_test = epsilon_test

    def act(self, observation, my_feed_dict={}):
        sess = tf.get_default_session()
        if self.weight_update > 1:
            if self.interaction_count % self.weight_update == 0:
                self.update_weights()

        feed_dict = {self.action_value._observation_placeholder: observation[None]}
        feed_dict.update(my_feed_dict)
        action = sess.run(self._argmax_action, feed_dict=feed_dict)

        # epsilon_greedy
        if np.random.rand() < self.epsilon_train:
            action = np.random.randint(self.action_value.num_actions)

        if self.weight_update > 0:
            self.interaction_count += 1

        return np.squeeze(action)

    def act_test(self, observation, my_feed_dict={}):
        sess = tf.get_default_session()
        if self.weight_update > 1:
            if self.interaction_count % self.weight_update == 0:
                self.update_weights()

        feed_dict = {self.action_value._observation_placeholder: observation[None]}
        feed_dict.update(my_feed_dict)
        action = sess.run(self._argmax_action, feed_dict=feed_dict)

        # epsilon_greedy
        if np.random.rand() < self.epsilon_test:
            action = np.random.randint(self.action_value.num_actions)

        if self.weight_update > 0:
            self.interaction_count += 1

        return np.squeeze(action)

    @property
    def q_net(self):
        return self.action_value

    def sync_weights(self):
        """
        sync the weights of network_old. Direct copy the weights of network.
        :return:
        """
        if self.action_value.sync_weights_ops is not None:
            self.action_value.sync_weights()

    def update_weights(self):
        """
        updates the weights of policy_old.
        :return:
        """
        if self.action_value.weight_update_ops is not None:
            self.action_value.update_weights()

    def set_epsilon_train(self, epsilon):
        self.epsilon_train = epsilon

    def set_epsilon_test(self, epsilon):
        self.epsilon_test = epsilon
