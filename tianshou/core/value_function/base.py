from __future__ import absolute_import

import tensorflow as tf

# TODO: linear feature baseline also in tf?
class ValueFunctionBase(object):
    """
    base class of value functions. Children include state values V(s) and action values Q(s, a)
    """
    def __init__(self, value_tensor, observation_placeholder):
        self._observation_placeholder = observation_placeholder
        self._value_tensor = tf.squeeze(value_tensor)  # canonical values has shape (batchsize, )

    def get_value(self, **kwargs):
        """

        :return: batch of corresponding values in numpy array
        """
        raise NotImplementedError()

    def get_value_tensor(self):
        """

        :return: tensor of the corresponding values
        """
        return self._value_tensor
