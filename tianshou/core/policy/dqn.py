from tianshou.core.policy.base import QValuePolicy
import tensorflow as tf


class DQN(QValuePolicy):
    """
    The policy as in DQN
    """

    def __init__(self, logits, observation_placeholder, dtype=None, **kwargs):
        self._logits = tf.convert_to_tensor(logits)
        if dtype is None:
            dtype = tf.int32
        self._n_categories = self._logits.get_shape()[-1].value

        super(DQN, self).__init__(observation_placeholder)

        net = tf.layers.conv2d(self._observation_placeholder, 16, 8, 4, 'valid', activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 32, 4, 2, 'valid', activation=tf.nn.relu)
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, 256, activation=tf.nn.relu, use_bias=True)
        self._value = tf.layers.dense(net, self._n_categories)

    def _act(self, observation, exploration=None):  # first implement no exploration
        """
        return the action (int) to be executed.
        no exploration when exploration=None.
        """
        sess = tf.get_default_session()
        sampled_action = sess.run(tf.multinomial(self.logits, num_samples=1),
                                  feed_dict={self._observation_placeholder: observation[None]})
        return sampled_action

    @property
    def logits(self):
        return self._logits

    @property
    def n_categories(self):
        return self._n_categories

    def values(self, observation):
        """
        returns the Q(s, a) values (float) for all actions a at observation s
        """
        sess = tf.get_default_session()
        value = sess.run(self._value, feed_dict={self._observation_placeholder: observation[None]})
        return value

    def values_tensor(self):
        """
        returns the tensor of the values for all actions a at observation s
        """
        return self._value
