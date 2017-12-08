#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

from .base import StochasticPolicy


__all__ = [
    'OnehotCategorical',
    'OnehotDiscrete',
]


class OnehotCategorical(StochasticPolicy):
    """
    The class of one-hot Categorical distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param logits: A N-D (N >= 1) `float` Tensor of shape (...,
        n_categories). Each slice `[i, j, ..., k, :]` represents the
        un-normalized log probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log p

    :param dtype: The value type of samples from the distribution.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.

    A single sample is a N-D Tensor with the same shape as logits. Each slice
    `[i, j, ..., k, :]` is a one-hot vector of the selected category.
    """

    def __init__(self, logits, obs_placeholder, dtype=None, group_ndims=0, **kwargs):
        self._logits = tf.convert_to_tensor(logits)

        if dtype is None:
            dtype = tf.int32
        # assert_same_float_and_int_dtype([], dtype)

        tf.assert_rank(self._logits, rank=2) # TODO: flexible policy output rank?
        self._n_categories = self._logits.get_shape()[-1].value

        super(OnehotCategorical, self).__init__(
            act_dtype=dtype,
            param_dtype=self._logits.dtype,
            is_continuous=False,
            obs_placeholder=obs_placeholder,
            group_ndims=group_ndims,
            **kwargs)

    @property
    def logits(self):
        """The un-normalized log probabilities."""
        return self._logits

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._n_categories

    def _act(self, observation):
        sess = tf.get_default_session() # TODO: this may be ugly. also maybe huge problem when parallel
        sampled_action = sess.run(tf.multinomial(self.logits, num_samples=1), feed_dict={self._obs_placeholder: observation[None]})

        sampled_action = sampled_action[0, 0]

        return sampled_action

    def _log_prob(self, sampled_action):
        sampled_action_onehot = tf.one_hot(sampled_action, self.n_categories, dtype=self.act_dtype)
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sampled_action, logits=self.logits)

        # given = tf.cast(given, self.param_dtype)
        # given, logits = maybe_explicit_broadcast(
        #     given, self.logits, 'given', 'logits')
        # if (given.get_shape().ndims == 2) or (logits.get_shape().ndims == 2):
        #     given_flat = given
        #     logits_flat = logits
        # else:
        #     given_flat = tf.reshape(given, [-1, self.n_categories])
        #     logits_flat = tf.reshape(logits, [-1, self.n_categories])
        # log_p_flat = -tf.nn.softmax_cross_entropy_with_logits(
        #     labels=given_flat, logits=logits_flat)
        # if (given.get_shape().ndims == 2) or (logits.get_shape().ndims == 2):
        #     log_p = log_p_flat
        # else:
        #     log_p = tf.reshape(log_p_flat, tf.shape(logits)[:-1])
        #     if given.get_shape() and logits.get_shape():
        #         log_p.set_shape(tf.broadcast_static_shape(
        #             given.get_shape(), logits.get_shape())[:-1])
        # return log_p

    def _prob(self, sampled_action):
        return tf.exp(self._log_prob(sampled_action))


OnehotDiscrete = OnehotCategorical