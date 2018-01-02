#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

from .base import StochasticPolicy


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

    def __init__(self, logits, observation_placeholder, dtype=None, group_ndims=0, **kwargs):
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
            observation_placeholder=observation_placeholder,
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
        # TODO: this may be ugly. also maybe huge problem when parallel
        sess = tf.get_default_session()
        # observation[None] adds one dimension at the beginning
        sampled_action = sess.run(tf.multinomial(self.logits, num_samples=1),
                                           feed_dict={self._observation_placeholder: observation[None]})

        sampled_action = sampled_action[0, 0]

        return sampled_action

    def _log_prob(self, sampled_action):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sampled_action, logits=self.logits)


    def _prob(self, sampled_action):
        return tf.exp(self._log_prob(sampled_action))


OnehotDiscrete = OnehotCategorical


class Normal(StochasticPolicy):
    """
        The :class:`Normal' class is the Normal policy

        :param mean:
        :param std:
        :param group_ndims
        :param observation_placeholder
    """
    def __init__(self,
                 mean = 0.,
                 logstd = 1.,
                 group_ndims = 1,
                 observation_placeholder = None,
                 **kwargs):

        self._mean = tf.convert_to_tensor(mean, dtype = tf.float32)
        self._logstd = tf.convert_to_tensor(logstd, dtype = tf.float32)
        self._std = tf.exp(self._logstd)

        super(Normal, self).__init__(
            act_dtype = tf.float32,
            param_dtype = tf.float32,
            is_continuous = True,
            observation_placeholder = observation_placeholder,
            group_ndims = group_ndims,
            **kwargs)

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def logstd(self):
        return self._logstd

    def _act(self, observation):
        # TODO: getting session like this maybe ugly. also maybe huge problem when parallel
        sess = tf.get_default_session()
        mean, std = self._mean, self._std
        shape = tf.broadcast_dynamic_shape(tf.shape(self._mean),\
                                           tf.shape(self._std))


        # observation[None] adds one dimension at the beginning
        sampled_action = sess.run(tf.random_normal(tf.concat([[1], shape], 0),
                                  dtype = tf.float32) * std + mean,
                                  feed_dict={self._observation_placeholder: observation[None]})
        sampled_action = sampled_action[0, 0]
        return sampled_action


    def _log_prob(self, sampled_action):
        mean, logstd = self._mean, self._logstd
        c = -0.5 * np.log(2 * np.pi)
        precision = tf.exp(-2 * logstd)
        return c - logstd - 0.5 * precision * tf.square(sampled_action - mean)

    def _prob(self, sampled_action):
        return tf.exp(self._log_prob(sampled_action))