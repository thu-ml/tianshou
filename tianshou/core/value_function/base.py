from __future__ import absolute_import

import tensorflow as tf

__all__ = []


class ValueFunctionBase(object):
    """
    Base class for value functions, including S-values and Q-values. The only
    mandatory method for a value function class is:

    :func:`eval_value`, which runs the graph and evaluates the corresponding value.

    :param value_tensor: a Tensor. The tensor of V(s) or Q(s, a).
    :param observation_placeholder: a :class:`tf.placeholder`. The observation placeholder of the network graph.
    """
    def __init__(self, value_tensor, observation_placeholder):
        self.observation_placeholder = observation_placeholder
        self._value_tensor = tf.squeeze(value_tensor)  # canonical value has shape (batchsize, )

    def eval_value(self, **kwargs):
        """
        Runs the graph and evaluates the corresponding value.
        """
        raise NotImplementedError()

    @property
    def value_tensor(self):
        """Tensor of the corresponding value"""
        return self._value_tensor
