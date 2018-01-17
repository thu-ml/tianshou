#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

from .base import StochasticPolicy

# TODO: the following, especially the target network construction should be refactored to be more neat
# even if policy_callable don't return a distribution class
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

    def __init__(self,
                 policy_callable,
                 observation_placeholder,
                 weight_update=1,
                 group_ndims=0,
                 **kwargs):
        self.managed_placeholders = {'observation': observation_placeholder}
        self.weight_update = weight_update
        self.interaction_count = -1  # defaults to -1. only useful if weight_update > 1.

        # build network, action and value
        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            logits, value_head = policy_callable()
            self._logits = tf.convert_to_tensor(logits, dtype=tf.float32)
            self._action = tf.multinomial(self._logits, num_samples=1)
            # TODO: self._action should be exactly the action tensor to run that directly gives action_dim

            if value_head is not None:
                pass

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        # deal with target network
        if self.weight_update == 1:
            self.weight_update_ops = None
            self.sync_weights_ops = None
        else:  # then we need to build another tf graph as target network
            with tf.variable_scope('net_old', reuse=tf.AUTO_REUSE):
                logits, value_head = policy_callable()
                self._logits_old = tf.convert_to_tensor(logits, dtype=tf.float32)

                if value_head is not None:  # useful in DDPG
                    pass

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
                pass
            else:
                self.interaction_count = 0  # as in DQN
                import math
                self.weight_update = math.ceil(weight_update)

        tf.assert_rank(self._logits, rank=2) # TODO: flexible policy output rank, e.g. RNN
        self._n_categories = self._logits.get_shape()[-1].value

        super(OnehotCategorical, self).__init__(
            act_dtype=tf.int32,
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

    @property
    def action_shape(self):
        return ()

    def _act(self, observation, my_feed_dict):
        # TODO: this may be ugly. also maybe huge problem when parallel
        sess = tf.get_default_session()
        # observation[None] adds one dimension at the beginning

        feed_dict = {self._observation_placeholder: observation[None]}
        feed_dict.update(my_feed_dict)
        sampled_action = sess.run(self._action, feed_dict=feed_dict)

        sampled_action = sampled_action[0, 0]

        return sampled_action

    def _log_prob(self, sampled_action):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sampled_action, logits=self.logits)

    def _prob(self, sampled_action):
        return tf.exp(self._log_prob(sampled_action))

    def _log_prob_old(self, sampled_action):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sampled_action, logits=self._logits_old)

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
                 policy_callable,
                 observation_placeholder,
                 weight_update=1,
                 group_ndims=1,
                 **kwargs):
        self.managed_placeholders = {'observation': observation_placeholder}
        self.weight_update = weight_update
        self.interaction_count = -1  # defaults to -1. only useful if weight_update > 1.

        # build network, action and value
        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            mean, logstd, value_head = policy_callable()
            self._mean = tf.convert_to_tensor(mean, dtype = tf.float32)
            self._logstd = tf.convert_to_tensor(logstd, dtype = tf.float32)
            self._std = tf.exp(self._logstd)

            shape = tf.broadcast_dynamic_shape(tf.shape(self._mean), tf.shape(self._std))
            self._action = tf.random_normal(tf.concat([[1], shape], 0), dtype = tf.float32) * self._std + self._mean
            # TODO: self._action should be exactly the action tensor to run that directly gives action_dim

            if value_head is not None:
                pass

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        # deal with target network
        if self.weight_update == 1:
            self.weight_update_ops = None
            self.sync_weights_ops = None
        else:  # then we need to build another tf graph as target network
            with tf.variable_scope('net_old', reuse=tf.AUTO_REUSE):
                mean, logstd, value_head = policy_callable()
                self._mean_old = tf.convert_to_tensor(mean, dtype=tf.float32)
                self._logstd_old = tf.convert_to_tensor(logstd, dtype=tf.float32)

                if value_head is not None:  # useful in DDPG
                    pass

            network_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
            network_old_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net_old')
            # TODO: use a scope that the user will almost surely not use. so get_collection will return
            # the correct weights and old_weights, since it filters by regular expression

            assert len(network_weights) == len(network_old_weights)
            self.sync_weights_ops = [tf.assign(variable_old, variable)
                                     for (variable_old, variable) in zip(network_old_weights, network_weights)]

            if weight_update == 0:
                self.weight_update_ops = self.sync_weights_ops
            elif 0 < weight_update < 1:
                pass
            else:
                self.interaction_count = 0
                import math
                self.weight_update = math.ceil(weight_update)



        super(Normal, self).__init__(
            act_dtype=tf.float32,
            param_dtype=tf.float32,
            is_continuous=True,
            observation_placeholder=observation_placeholder,
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

    @property
    def action_shape(self):
        return tuple(self._mean.shape.as_list()[1:])

    def _act(self, observation, my_feed_dict):
        # TODO: getting session like this maybe ugly. also maybe huge problem when parallel
        sess = tf.get_default_session()

        # observation[None] adds one dimension at the beginning
        feed_dict = {self._observation_placeholder: observation[None]}
        feed_dict.update(my_feed_dict)
        sampled_action = sess.run(self._action, feed_dict=feed_dict)
        sampled_action = sampled_action[0, 0]
        return sampled_action


    def _log_prob(self, sampled_action):
        mean, logstd = self._mean, self._logstd
        c = -0.5 * np.log(2 * np.pi)
        precision = tf.exp(-2 * logstd)
        return c - logstd - 0.5 * precision * tf.square(sampled_action - mean)

    def _prob(self, sampled_action):
        return tf.exp(self._log_prob(sampled_action))

    def _log_prob_old(self, sampled_action):
        """
        return the log_prob of the old policy when constructing tf graphs. Raises error when there's no old policy.
        :param sampled_action: the placeholder for sampled actions during interaction with the environment.
        :return: tensor of the log_prob of the old policy
        """
        if self.weight_update == 1:
            raise AttributeError('Policy has no policy_old since it\'s initialized with weight_update=1!')

        mean, logstd = self._mean_old, self._logstd_old
        c = -0.5 * np.log(2 * np.pi)
        precision = tf.exp(-2 * logstd)
        return c - logstd - 0.5 * precision * tf.square(sampled_action - mean)

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