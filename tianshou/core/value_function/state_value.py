from __future__ import absolute_import

from .base import ValueFunctionBase
import tensorflow as tf


class StateValue(ValueFunctionBase):
    """
    class of state values V(s).
    """
    def __init__(self, value_tensor, observation_placeholder):
        super(StateValue, self).__init__(
            value_tensor=value_tensor,
            observation_placeholder=observation_placeholder
        )

    def get_value(self, observation):
        """

        :param observation: numpy array of observations, of shape (batchsize, observation_dim).
        :return: numpy array of state values, of shape (batchsize, )
        # TODO: dealing with the last dim of 1 in V(s) and Q(s, a), this should rely on the action shape returned by env
        """
        sess = tf.get_default_session()
        return sess.run(self.get_value_tensor(), feed_dict={self._observation_placeholder: observation})