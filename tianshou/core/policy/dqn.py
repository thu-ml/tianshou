from __future__ import absolute_import

from .base import PolicyBase
import tensorflow as tf
import numpy as np


class DQN(PolicyBase):
    """
    use DQN from value_function as a member

    Policy derived from a Deep-Q Network (DQN). It should be constructed from a :class:`tianshou.core.value_function.DQN`.
    Action is the argmax of the Q-values (usually with further :math:`\epsilon`-greedy).
    It can only be applied to discrete action spaces.

    :param dqn: A :class:`tianshou.core.value_function.DQN`. The Q-value network to derive this policy.
    :param epsilon_train: A float in range :math:`[0, 1]`. The :math:`\epsilon` used in :math:`\epsilon`-greedy
        during training while interacting with the environment.
    :param epsilon_test: A float in range :math:`[0, 1]`. The :math:`\epsilon` used in :math:`\epsilon`-greedy
        during test while interacting with the environment.
    """
    def __init__(self, dqn, epsilon_train=0.1, epsilon_test=0.05):
        self.action_value = dqn
        self.action = tf.argmax(dqn.value_tensor_all_actions, axis=1)

        self.epsilon_train = epsilon_train
        self.epsilon_test = epsilon_test

    def act(self, observation, my_feed_dict={}):
        """
        Return action given observation, with :math:`\epsilon`-greedy using ``self.epsilon_train``.

        :param observation: An array-like with rank the same as a single observation of the environment.
            Its "batch_size" is 1, but should not be explicitly set. This method will add the dimension
            of "batch_size" to the first dimension.
        :param my_feed_dict: Optional. A dict defaulting to empty.
            Specifies placeholders such as dropout and batch_norm except observation.

        :return: A numpy array.
            Action given the single observation. Its "batch_size" is 1, but should not be explicitly set.
        """
        sess = tf.get_default_session()

        feed_dict = {self.action_value.observation_placeholder: observation[None]}
        feed_dict.update(my_feed_dict)
        action = sess.run(self.action, feed_dict=feed_dict)

        # epsilon_greedy
        if np.random.rand() < self.epsilon_train:
            action = np.random.randint(self.action_value.num_actions)

        return np.squeeze(action)

    def act_test(self, observation, my_feed_dict={}):
        """
        Return action given observation, with :math:`\epsilon`-greedy using ``self.epsilon_test``.

        :param observation: An array-like with rank the same as a single observation of the environment.
            Its "batch_size" is 1, but should not be explicitly set. This method will add the dimension
            of "batch_size" to the first dimension.
        :param my_feed_dict: Optional. A dict defaulting to empty.
            Specifies placeholders such as dropout and batch_norm except observation.

        :return: A numpy array.
            Action given the single observation. Its "batch_size" is 1, but should not be explicitly set.
        """
        sess = tf.get_default_session()

        feed_dict = {self.action_value.observation_placeholder: observation[None]}
        feed_dict.update(my_feed_dict)
        action = sess.run(self.action, feed_dict=feed_dict)

        # epsilon_greedy
        if np.random.rand() < self.epsilon_test:
            action = np.random.randint(self.action_value.num_actions)

        return np.squeeze(action)

    @property
    def q_net(self):
        """The DQN (:class:`tianshou.core.value_function.DQN`) this policy based on."""
        return self.action_value

    def sync_weights(self):
        """
        Sync the variables of the "old net" to be the same as the current network.
        """
        if self.action_value.sync_weights_ops is not None:
            self.action_value.sync_weights()

    def set_epsilon_train(self, epsilon):
        """
        Set the :math:`\epsilon` in :math:`\epsilon`-greedy during training.
        :param epsilon: A float in range :math:`[0, 1]`.
        """
        self.epsilon_train = epsilon

    def set_epsilon_test(self, epsilon):
        """
        Set the :math:`\epsilon` in :math:`\epsilon`-greedy during training.
        :param epsilon: A float in range :math:`[0, 1]`.
        """
        self.epsilon_test = epsilon
