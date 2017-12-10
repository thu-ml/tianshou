#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
import warnings

import tensorflow as tf

# from zhusuan.utils import add_name_scope


__all__ = [
    'StochasticPolicy',
]


class StochasticPolicy(object):
    """
    The :class:`Distribution` class is the base class for various probabilistic
    distributions which support batch inputs, generating batches of samples and
    evaluate probabilities at batches of given values.

    The typical input shape for a :class:`Distribution` is like
    ``batch_shape + input_shape``. where ``input_shape`` represents the shape
    of non-batch input parameter, :attr:`batch_shape` represents how many
    independent inputs are fed into the distribution.

    Samples generated are of shape
    ``([n_samples]+ )batch_shape + value_shape``. The first additional axis
    is omitted only when passed `n_samples` is None (by default), in which
    case one sample is generated. :attr:`value_shape` is the non-batch value
    shape of the distribution. For a univariate distribution, its
    :attr:`value_shape` is [].

    There are cases where a batch of random variables are grouped into a
    single event so that their probabilities should be computed together. This
    is achieved by setting `group_ndims` argument, which defaults to 0.
    The last `group_ndims` number of axes in :attr:`batch_shape` are
    grouped into a single event. For example,
    ``Normal(..., group_ndims=1)`` will set the last axis of its
    :attr:`batch_shape` to a single event, i.e., a multivariate Normal with
    identity covariance matrix.

    When evaluating probabilities at given values, the given Tensor should be
    broadcastable to shape ``(... + )batch_shape + value_shape``. The returned
    Tensor has shape ``(... + )batch_shape[:-group_ndims]``.

    .. seealso::

        :doc:`/concepts`

    For both, the parameter `dtype` represents type of samples. For discrete,
    can be set by user. For continuous, automatically determined from parameter
    types.

    The value type of `prob` and `log_prob` will be `param_dtype` which is
    deduced from the parameter(s) when initializating. And `dtype` must be
    among `int16`, `int32`, `int64`, `float16`, `float32` and `float64`.

    When two or more parameters are tensors and they have different type,
    `TypeError` will be raised.

    :param dtype: The value type of samples from the distribution.
    :param param_dtype: The parameter(s) type of the distribution.
    :param is_continuous: Whether the distribution is continuous.
    :param is_reparameterized: A bool. Whether the gradients of samples can
        and are allowed to propagate back into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param use_path_derivative: A bool. Whether when taking the gradients
        of the log-probability to propagate them through the parameters
        of the distribution (False meaning you do propagate them). This
        is based on the paper "Sticking the Landing: Simple,
        Lower-Variance Gradient Estimators for Variational Inference"
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in :attr:`batch_shape` (counted from the end) that are
        grouped into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See above for more detailed explanation.
    """

    def __init__(self,
                 act_dtype,
                 param_dtype,
                 is_continuous,
                 observation_placeholder,
                 group_ndims=0, # maybe useful for repeat_action
                 **kwargs):

        self._act_dtype = act_dtype
        self._param_dtype = param_dtype
        self._is_continuous = is_continuous
        self._observation_placeholder = observation_placeholder
        if isinstance(group_ndims, int):
            if group_ndims < 0:
                raise ValueError("group_ndims must be non-negative.")
            self._group_ndims = group_ndims
        else:
            group_ndims = tf.convert_to_tensor(group_ndims, tf.int32)
            _assert_rank_op = tf.assert_rank(
                group_ndims, 0,
                message="group_ndims should be a scalar (0-D Tensor).")
            _assert_nonnegative_op = tf.assert_greater_equal(
                group_ndims, 0,
                message="group_ndims must be non-negative.")
            with tf.control_dependencies([_assert_rank_op,
                                          _assert_nonnegative_op]):
                self._group_ndims = tf.identity(group_ndims)

    @property
    def act_dtype(self):
        """The sample data type of the policy."""
        return self._act_dtype

    @property
    def param_dtype(self):
        """The parameter(s) type of the distribution."""
        return self._param_dtype

    @property
    def is_continuous(self):
        """Whether the distribution is continuous."""
        return self._is_continuous

    @property
    def group_ndims(self):
        """
        The number of dimensions in :attr:`batch_shape` (counted from the end)
        that are grouped into a single event, so that their probabilities are
        calculated together. See `Distribution` for more detailed explanation.
        """
        return self._group_ndims

    # @add_name_scope
    def act(self, observation):
        """
        sample(n_samples=None)

        Return samples from the distribution. When `n_samples` is None (by
        default), one sample of shape ``batch_shape + value_shape`` is
        generated. For a scalar `n_samples`, the returned Tensor has a new
        sample dimension with size `n_samples` inserted at ``axis=0``, i.e.,
        the shape of samples is ``[n_samples] + batch_shape + value_shape``.

        :param n_samples: A 0-D `int32` Tensor or None. How many independent
            samples to draw from the distribution.
        :return: A Tensor of samples.
        """
        return self._act(observation)

        if n_samples is None:
            samples = self._sample(n_samples=1)
            return tf.squeeze(samples, axis=0)
        elif isinstance(n_samples, int):
            return self._sample(n_samples)
        else:
            n_samples = tf.convert_to_tensor(n_samples, dtype=tf.int32)
            _assert_rank_op = tf.assert_rank(
                n_samples, 0,
                message="n_samples should be a scalar (0-D Tensor).")
            with tf.control_dependencies([_assert_rank_op]):
                samples = self._sample(n_samples)
            return samples

    def _act(self, observation):
        """
        Private method for subclasses to rewrite the :meth:`sample` method.
        """
        raise NotImplementedError()

    # @add_name_scope
    def log_prob(self, sampled_action):
        """
        log_prob(sampled_action)

        Compute log probability density (mass) function at `given` value.

        :param given: A Tensor. The value at which to evaluate log probability
            density (mass) function. Must be able to broadcast to have a shape
            of ``(... + )batch_shape + value_shape``.
        :return: A Tensor of shape ``(... + )batch_shape[:-group_ndims]``.
        """
        log_p = self._log_prob(sampled_action)
        return tf.reduce_sum(log_p, tf.range(-self._group_ndims, 0))

    # @add_name_scope
    def prob(self, sampled_action):
        """
        prob(given)

        Compute probability density (mass) function at `given` value.

        :param given: A Tensor. The value at which to evaluate probability
            density (mass) function. Must be able to broadcast to have a shape
            of ``(... + )batch_shape + value_shape``.
        :return: A Tensor of shape ``(... + )batch_shape[:-group_ndims]``.
        """
        p = self._prob(sampled_action)
        return tf.reduce_prod(p, tf.range(-self._group_ndims, 0))

    def _log_prob(self, sampled_action):
        """
        Private method for subclasses to rewrite the :meth:`log_prob` method.
        """
        raise NotImplementedError()

    def _prob(self, sampled_action):
        """
        Private method for subclasses to rewrite the :meth:`prob` method.
        """
        raise NotImplementedError()